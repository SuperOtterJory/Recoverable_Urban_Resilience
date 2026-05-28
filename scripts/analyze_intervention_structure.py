"""Analyze where optimized intervention primitives are allocated."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from recoverable_resilience.calibration import calibrate_city, load_yaml
from recoverable_resilience.paths import find_repo_root


ATTRIBUTES = [
    "origin_exposure_p",
    "initial_deficit_b0",
    "destination_importance",
    "local_disturbance_h",
    "out_degree",
    "in_degree",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--optimization-dir", default=None)
    parser.add_argument("--output-dir", default="results/city_structure")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    optimization_dir = root / (args.optimization_dir or config["project"]["output_dir"])
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    interventions = pd.read_csv(optimization_dir / "tables" / "optimization_interventions.csv")
    unit_frames = []
    allocation_frames = []
    for city in config["calibration"]["cities"]:
        print(f"Analyzing intervention structure for {city}")
        params = calibrate_city(city, config, root=root)
        unit_attr = unit_attributes(params)
        unit_frames.append(unit_attr)
        city_alloc = interventions[interventions["city"] == city].copy()
        allocation_frames.append(allocation_by_unit(city_alloc, unit_attr))

    units = pd.concat(unit_frames, ignore_index=True)
    allocation = pd.concat(allocation_frames, ignore_index=True)
    primitive_summary = summarize_primitive_allocation(allocation)
    top_units = summarize_top_units(allocation)
    nontrivial = summarize_nontrivial_patterns(allocation)

    units.to_csv(table_dir / "unit_structural_attributes.csv", index=False)
    allocation.to_csv(table_dir / "intervention_unit_allocation.csv", index=False)
    primitive_summary.to_csv(table_dir / "intervention_primitive_structure.csv", index=False)
    top_units.to_csv(table_dir / "intervention_top_units.csv", index=False)
    nontrivial.to_csv(table_dir / "intervention_nontrivial_patterns.csv", index=False)
    write_report(report_dir / "intervention_structure_report_zh.md", primitive_summary, top_units, nontrivial)
    print(f"Wrote intervention-structure analysis to {output_dir}")


def unit_attributes(params: Any) -> pd.DataFrame:
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    p = np.asarray(params.p, dtype=float)
    b0 = np.asarray(params.b0, dtype=float)
    destination_importance = np.asarray(q.T @ p).ravel()
    local_disturbance = np.asarray(params.h.sum(axis=1), dtype=float)
    out_degree = np.diff(q.indptr)
    in_degree = np.diff(q.tocsc().indptr)
    df = pd.DataFrame(
        {
            "city": params.city,
            "unit": params.units,
            "origin_exposure_p": p,
            "initial_deficit_b0": b0,
            "destination_importance": destination_importance,
            "local_disturbance_h": local_disturbance,
            "out_degree": out_degree,
            "in_degree": in_degree,
        }
    )
    for attr in ATTRIBUTES:
        df[f"{attr}_rank_pct"] = rank_pct(df[attr])
        df[f"{attr}_top10"] = df[f"{attr}_rank_pct"] >= 0.90
        df[f"{attr}_top25"] = df[f"{attr}_rank_pct"] >= 0.75
    return df


def allocation_by_unit(interventions: pd.DataFrame, unit_attr: pd.DataFrame) -> pd.DataFrame:
    interventions["unit"] = interventions["unit"].astype(str)
    unit_attr = unit_attr.copy()
    unit_attr["unit"] = unit_attr["unit"].astype(str)
    interventions["effective_cost"] = pd.to_numeric(interventions["effective_cost"], errors="coerce").fillna(0.0)
    interventions["u"] = pd.to_numeric(interventions["u"], errors="coerce").fillna(0.0)
    interventions["e"] = pd.to_numeric(interventions["e"], errors="coerce").fillna(0.0)
    grouped = interventions.groupby(["city", "unit", "intervention"], as_index=False).agg(
        total_u=("u", "sum"),
        total_e=("e", "sum"),
        total_cost=("effective_cost", "sum"),
    )
    grouped = grouped.merge(unit_attr, on=["city", "unit"], how="left")
    city_total = grouped.groupby(["city", "intervention"])["total_cost"].transform("sum")
    grouped["primitive_cost_share"] = grouped["total_cost"] / city_total.replace(0, np.nan)
    return grouped


def summarize_primitive_allocation(allocation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, primitive), group in allocation.groupby(["city", "intervention"]):
        cost = group["total_cost"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "city": city,
            "intervention": primitive,
            "total_cost": float(cost.sum()),
            "active_unit_count": int((cost > 1e-10).sum()),
            "active_unit_share": float((cost > 1e-10).mean()),
            "cost_gini_across_units": gini(cost),
            "top_10pct_cost_share": top_share(cost, 0.10),
        }
        for attr in ATTRIBUTES:
            row[f"cost_weighted_mean_{attr}_rank_pct"] = weighted_mean(
                group[f"{attr}_rank_pct"].to_numpy(dtype=float), cost
            )
            row[f"cost_share_in_top10_{attr}"] = cost_share(group[f"{attr}_top10"].to_numpy(dtype=bool), cost)
            row[f"cost_share_in_top25_{attr}"] = cost_share(group[f"{attr}_top25"].to_numpy(dtype=bool), cost)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["city", "intervention"])


def summarize_top_units(allocation: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    for (city, primitive), group in allocation.groupby(["city", "intervention"]):
        top = group.sort_values("total_cost", ascending=False).head(top_n).copy()
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            rows.append(
                {
                    "city": city,
                    "intervention": primitive,
                    "rank": rank,
                    "unit": row.unit,
                    "total_cost": row.total_cost,
                    "primitive_cost_share": row.primitive_cost_share,
                    "origin_exposure_rank_pct": row.origin_exposure_p_rank_pct,
                    "initial_deficit_rank_pct": row.initial_deficit_b0_rank_pct,
                    "destination_importance_rank_pct": row.destination_importance_rank_pct,
                    "out_degree_rank_pct": row.out_degree_rank_pct,
                    "in_degree_rank_pct": row.in_degree_rank_pct,
                }
            )
    return pd.DataFrame(rows)


def summarize_nontrivial_patterns(allocation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, primitive), group in allocation.groupby(["city", "intervention"]):
        cost = group["total_cost"].to_numpy(dtype=float)
        active = cost > 1e-10
        high_cost = np.zeros(len(group), dtype=bool)
        if active.any():
            active_cost = cost[active]
            cutoff = np.sort(active_cost)[-max(1, int(np.ceil(active_cost.size * 0.10)))]
            high_cost = active & (cost >= cutoff)
        rows.append(
            {
                "city": city,
                "intervention": primitive,
                "high_cost_units_not_top10_origin_exposure_share": float(
                    (~group.loc[high_cost, "origin_exposure_p_top10"]).mean()
                )
                if high_cost.any()
                else np.nan,
                "active_units_not_top25_origin_exposure_share": float(
                    (~group.loc[active, "origin_exposure_p_top25"]).mean()
                )
                if active.any()
                else np.nan,
                "cost_share_outside_top10_origin_exposure": 1.0
                - cost_share(group["origin_exposure_p_top10"].to_numpy(dtype=bool), cost),
                "cost_share_outside_top10_initial_deficit": 1.0
                - cost_share(group["initial_deficit_b0_top10"].to_numpy(dtype=bool), cost),
                "cost_share_outside_top10_destination_importance": 1.0
                - cost_share(group["destination_importance_top10"].to_numpy(dtype=bool), cost),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    primitive_summary: pd.DataFrame,
    top_units: pd.DataFrame,
    nontrivial: pd.DataFrame,
) -> None:
    key_cols = [
        "city",
        "intervention",
        "active_unit_share",
        "cost_gini_across_units",
        "cost_weighted_mean_origin_exposure_p_rank_pct",
        "cost_weighted_mean_initial_deficit_b0_rank_pct",
        "cost_weighted_mean_destination_importance_rank_pct",
        "cost_share_in_top10_origin_exposure_p",
        "cost_share_in_top10_initial_deficit_b0",
        "cost_share_in_top10_destination_importance",
    ]
    city_mean = primitive_summary.groupby("intervention", as_index=False).mean(numeric_only=True)
    nontrivial_mean = nontrivial.groupby("intervention", as_index=False).mean(numeric_only=True)
    lines = [
        "# Intervention Structure Analysis",
        "",
        "## 解释",
        "",
        "本分析使用 full-zone base LP 的优化投放明细，比较 R/C/S 三类资源是否倾向于投放到高 origin exposure、高 initial deficit、高 destination importance 或高网络度数的区域。",
        "",
        "## 跨城市平均",
        "",
        dataframe_to_markdown(city_mean[[
            "intervention",
            "active_unit_share",
            "cost_gini_across_units",
            "cost_weighted_mean_origin_exposure_p_rank_pct",
            "cost_weighted_mean_initial_deficit_b0_rank_pct",
            "cost_weighted_mean_destination_importance_rank_pct",
            "cost_share_in_top10_origin_exposure_p",
            "cost_share_in_top10_initial_deficit_b0",
            "cost_share_in_top10_destination_importance",
        ]]),
        "",
        "## 非直觉模式",
        "",
        dataframe_to_markdown(nontrivial_mean),
        "",
        "## 城市-资源明细",
        "",
        dataframe_to_markdown(primitive_summary[key_cols], max_rows=30),
        "",
        "## Top allocated units",
        "",
        dataframe_to_markdown(top_units.head(40), max_rows=40),
        "",
        "## 初步结论",
        "",
        "1. R/C/S 都不是简单投向最活跃 origin 区域；相当一部分成本落在 top-10% origin exposure 之外。",
        "2. S 更接近 origin-side intervention，因为它直接减少 origin-level experienced loss `ell_i`；R 和 C 更接近 destination/local deficit intervention，因为它们作用在 `b_i` 或 `d_i`。",
        "3. 高成本 unit 同时看重 exposure、deficit 和 dependence position，而不是单一活跃度指标。这是优化相对 heuristic 的主要来源之一。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_pct(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return values.rank(method="average", pct=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def cost_share(mask: np.ndarray, cost: np.ndarray) -> float:
    total = float(np.nansum(cost))
    if total <= 0:
        return float("nan")
    return float(np.nansum(cost[mask]) / total)


def top_share(values: np.ndarray, pct: float) -> float:
    values = values[np.isfinite(values)]
    total = float(values.sum())
    if total <= 0 or len(values) == 0:
        return float("nan")
    n = max(1, int(np.ceil(len(values) * pct)))
    return float(np.sort(values)[-n:].sum() / total)


def gini(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    values = np.sort(np.maximum(values, 0.0))
    total = values.sum()
    if total <= 0:
        return float("nan")
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * total) - (n + 1) / n)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


if __name__ == "__main__":
    main()

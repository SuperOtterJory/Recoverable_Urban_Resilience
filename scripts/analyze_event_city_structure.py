"""Analyze observed-event LP outputs as city-structure evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


STRUCTURE_FEATURES = [
    "n_units",
    "q_nnz",
    "q_density",
    "od_rows",
    "origin_zone_count",
    "destination_zone_count",
    "od_density_observed",
    "within_zone_volume_share",
    "top10_destination_volume_share",
    "destination_volume_hhi",
    "destination_volume_gini",
    "congested_volume_share_speed_ratio_lt_0_8",
    "doc_over_1_volume_share",
    "volume_weighted_speed_kmph",
    "top10_link_volume_share",
    "link_volume_hhi",
    "tmc_deficit_gini",
    "top_10pct_tmc_deficit_share",
]

RAIN_FEATURES = [
    "event_total_precip",
    "event_peak_precip",
    "event_peak_positive_abnormal_deficit",
    "weighted_b0",
    "weighted_h_total",
]

UNIT_ATTRIBUTES = [
    "origin_exposure_p",
    "destination_importance",
    "initial_deficit_b0",
    "local_disturbance_h",
    "out_degree",
    "in_degree",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--event-optimization-dir", default="results/event_optimization")
    parser.add_argument("--output-dir", default="results/event_city_structure")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root, root / args.event_optimization_dir)
    event_dataset = build_event_dataset(data)
    city_dataset = build_city_dataset(event_dataset, data)
    city_corr = build_city_correlations(city_dataset)
    partial_corr = build_partial_event_correlations(event_dataset)
    unit_attr = build_unit_attributes(root, config, event_dataset, data)
    allocation = build_intervention_allocation(data["interventions"], unit_attr)
    primitive_summary = summarize_primitive_allocation(allocation, unit_attr)
    top_units = summarize_top_units(allocation)
    nontrivial = summarize_nontrivial_patterns(primitive_summary)

    event_dataset.to_csv(table_dir / "event_structure_dataset.csv", index=False)
    city_dataset.to_csv(table_dir / "event_city_structure_dataset.csv", index=False)
    city_corr.to_csv(table_dir / "event_city_structure_correlations.csv", index=False)
    partial_corr.to_csv(table_dir / "event_structure_partial_correlations.csv", index=False)
    unit_attr.to_csv(table_dir / "event_unit_structural_attributes.csv", index=False)
    allocation.to_csv(table_dir / "event_intervention_unit_allocation.csv", index=False)
    primitive_summary.to_csv(table_dir / "event_intervention_primitive_structure.csv", index=False)
    top_units.to_csv(table_dir / "event_intervention_top_units.csv", index=False)
    nontrivial.to_csv(table_dir / "event_intervention_nontrivial_patterns.csv", index=False)
    make_figures(event_dataset, city_dataset, partial_corr, primitive_summary, figure_dir)
    write_report(
        report_dir / "event_city_structure_report_zh.md",
        city_dataset,
        city_corr,
        partial_corr,
        primitive_summary,
        nontrivial,
        top_units,
    )
    print(f"Wrote event city-structure analysis to {output_dir}")


def load_inputs(root: Path, event_optimization_dir: Path) -> dict[str, pd.DataFrame]:
    opt = event_optimization_dir / "tables"
    mining = root / "results" / "data_mining" / "tables"
    calibration = root / "results" / "event_calibration" / "tables"
    return {
        "summary": read_csv(opt / "event_optimization_summary.csv"),
        "calibration": read_csv(opt / "event_calibration_summary.csv"),
        "interventions": read_csv(opt / "event_optimization_interventions.csv"),
        "demand": read_csv(mining / "demand_network_summary.csv"),
        "speed": read_csv(mining / "speed_deficit_summary.csv"),
        "tmc": read_csv(mining / "speed_tmc_deficit_concentration.csv"),
        "rain_events": read_csv(mining / "rainfall_event_impact_summary.csv"),
        "dynamics": read_csv(calibration / "event_dynamic_calibration_summary.csv"),
        "event_details": read_csv(mining / "rainfall_event_impact_details.csv"),
        "abnormal": read_csv(mining / "speed_hourly_abnormal_deficit.csv"),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in df.columns:
        if column not in {"city", "scenario", "status", "event_start", "event_end", "unit", "intervention", "h_signal_source"}:
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().sum() > 0:
                df[column] = converted
    return df


def build_event_dataset(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = data["summary"].copy()
    if summary.empty:
        raise FileNotFoundError("event_optimization_summary.csv is empty or missing.")
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    calibration = data["calibration"].copy()
    if not calibration.empty:
        keys = ["city", "event_id", "scenario"]
        summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype("Int64")
        calibration["event_id"] = pd.to_numeric(calibration["event_id"], errors="coerce").astype("Int64")
        keep_cols = keys + [
            col
            for col in ["q_nnz", "q_density", "mean_a_retention", "total_disturbance", "pwl_enabled"]
            if col in calibration.columns
        ]
        summary = summary.merge(calibration[keep_cols], on=keys, how="left", suffixes=("", "_cal"))
    for key in ["demand", "speed", "tmc", "rain_events", "dynamics"]:
        table = data[key].copy()
        if not table.empty:
            summary = summary.merge(table, on="city", how="left", suffixes=("", f"_{key}"))
    summary["od_sparsity"] = 1.0 - summary["q_density"]
    summary["log_n_units"] = np.log1p(summary["n_units"])
    summary["log_od_rows"] = np.log1p(summary["od_rows"])
    summary["rain_intensity"] = np.log1p(summary["event_total_precip"]) + np.log1p(summary["event_peak_precip"])
    summary["recoverable_percent"] = 100 * summary["recoverable_fraction"]
    return summary


def build_city_dataset(event_dataset: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    agg = event_dataset.groupby("city", as_index=False).agg(
        event_count=("event_id", "nunique"),
        mean_recoverable_fraction=("recoverable_fraction", "mean"),
        median_recoverable_fraction=("recoverable_fraction", "median"),
        p25_recoverable_fraction=("recoverable_fraction", lambda s: s.quantile(0.25)),
        p75_recoverable_fraction=("recoverable_fraction", lambda s: s.quantile(0.75)),
        mean_baseline_objective=("baseline_objective", "mean"),
        mean_optimized_objective=("optimized_objective", "mean"),
        mean_weighted_b0=("weighted_b0", "mean"),
        mean_weighted_h_total=("weighted_h_total", "mean"),
        mean_event_total_precip=("event_total_precip", "mean"),
        mean_event_peak_precip=("event_peak_precip", "mean"),
        mean_event_peak_positive_abnormal_deficit=("event_peak_positive_abnormal_deficit", "mean"),
        n_units=("n_units", "first"),
        q_nnz=("q_nnz", "first"),
        q_density=("q_density", "first"),
    )
    cost_cols = [col for col in ["total_cost_R", "total_cost_C", "total_cost_S", "total_intervention_cost"] if col in event_dataset.columns]
    if cost_cols:
        costs = event_dataset.groupby("city", as_index=False)[cost_cols].sum()
        for primitive in ["R", "C", "S"]:
            col = f"total_cost_{primitive}"
            if col in costs.columns and "total_intervention_cost" in costs.columns:
                costs[f"cost_share_{primitive}"] = costs[col] / costs["total_intervention_cost"].replace(0, np.nan)
        agg = agg.merge(costs, on="city", how="left")
    for key in ["demand", "speed", "tmc", "rain_events", "dynamics"]:
        table = data[key].copy()
        if not table.empty:
            agg = agg.merge(table, on="city", how="left", suffixes=("", f"_{key}"))
    agg["od_sparsity"] = 1.0 - agg["q_density"]
    agg["log_n_units"] = np.log1p(agg["n_units"])
    agg["log_od_rows"] = np.log1p(agg["od_rows"])
    return agg.sort_values("mean_recoverable_fraction", ascending=False)


def build_city_correlations(city_dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = ["mean_recoverable_fraction", "mean_baseline_objective", "cost_share_R", "cost_share_C", "cost_share_S"]
    for feature in STRUCTURE_FEATURES + ["od_sparsity", "log_n_units", "log_od_rows", "a_retention", "rain_kernel_sum"]:
        for target in targets:
            add_corr(rows, city_dataset, feature, target, level="city")
    return pd.DataFrame(rows).sort_values(["target", "abs_spearman"], ascending=[True, False])


def build_partial_event_correlations(event_dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controls = ["event_total_precip", "event_peak_precip"]
    stronger_controls = controls + ["weighted_b0", "weighted_h_total"]
    for feature in STRUCTURE_FEATURES + ["od_sparsity", "log_n_units", "log_od_rows", "a_retention", "rain_kernel_sum"]:
        add_partial_corr(rows, event_dataset, feature, "recoverable_fraction", controls, "control_rain_intensity")
        add_partial_corr(rows, event_dataset, feature, "recoverable_fraction", stronger_controls, "control_rain_and_observed_shock")
    return pd.DataFrame(rows).sort_values(["control_set", "abs_partial_pearson"], ascending=[True, False])


def add_corr(rows: list[dict[str, Any]], df: pd.DataFrame, feature: str, target: str, *, level: str) -> None:
    if feature not in df.columns or target not in df.columns:
        return
    pair = df[["city", feature, target]].dropna()
    if len(pair) < 4 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
        return
    spearman = pair[feature].corr(pair[target], method="spearman")
    rows.append(
        {
            "level": level,
            "feature": feature,
            "target": target,
            "n": len(pair),
            "pearson": pair[feature].corr(pair[target], method="pearson"),
            "spearman": spearman,
            "abs_spearman": abs(spearman),
        }
    )


def add_partial_corr(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    feature: str,
    target: str,
    controls: list[str],
    control_set: str,
) -> None:
    columns = ["city", feature, target, *controls]
    if any(column not in df.columns for column in columns):
        return
    pair = df[columns].dropna()
    if len(pair) < 20 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
        return
    x_resid = residualize(pair[feature].to_numpy(dtype=float), pair[controls].to_numpy(dtype=float))
    y_resid = residualize(pair[target].to_numpy(dtype=float), pair[controls].to_numpy(dtype=float))
    if np.std(x_resid) <= 1e-12 or np.std(y_resid) <= 1e-12:
        return
    corr = float(np.corrcoef(x_resid, y_resid)[0, 1])
    rows.append(
        {
            "control_set": control_set,
            "feature": feature,
            "target": target,
            "controls": ", ".join(controls),
            "n_events": len(pair),
            "n_cities": int(pair["city"].nunique()),
            "partial_pearson": corr,
            "abs_partial_pearson": abs(corr),
        }
    )


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(controls)), controls])
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ beta


def build_unit_attributes(
    root: Path,
    config: dict[str, Any],
    event_dataset: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    abnormal = data["abnormal"].copy()
    if "hour" in abnormal.columns:
        abnormal["hour"] = pd.to_datetime(abnormal["hour"])
    dynamics = data["dynamics"]
    dynamic_by_city = {row["city"]: row for _, row in dynamics.iterrows()}
    unit_frames = []
    for city, group in event_dataset.groupby("city"):
        event = group.sort_values("event_peak_positive_abnormal_deficit", ascending=False).iloc[0]
        params = calibrate_observed_event_city(
            city,
            config,
            event,
            dynamic_by_city[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        unit_frames.append(unit_attributes(params))
    return pd.concat(unit_frames, ignore_index=True)


def unit_attributes(params: Any) -> pd.DataFrame:
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    p = np.asarray(params.p, dtype=float)
    destination_importance = np.asarray(q.T @ p).ravel()
    out_degree = np.diff(q.indptr)
    in_degree = np.diff(q.tocsc().indptr)
    df = pd.DataFrame(
        {
            "city": params.city,
            "unit": params.units,
            "origin_exposure_p": p,
            "destination_importance": destination_importance,
            "initial_deficit_b0": params.b0,
            "local_disturbance_h": params.h.sum(axis=1),
            "out_degree": out_degree,
            "in_degree": in_degree,
        }
    )
    for attr in UNIT_ATTRIBUTES:
        df[f"{attr}_rank_pct"] = rank_pct(df[attr])
        df[f"{attr}_top10"] = df[f"{attr}_rank_pct"] >= 0.90
        df[f"{attr}_top25"] = df[f"{attr}_rank_pct"] >= 0.75
    return df


def build_intervention_allocation(interventions: pd.DataFrame, unit_attr: pd.DataFrame) -> pd.DataFrame:
    if interventions.empty:
        return pd.DataFrame()
    interventions = interventions.copy()
    interventions["unit"] = interventions["unit"].astype(str)
    unit_attr = unit_attr.copy()
    unit_attr["unit"] = unit_attr["unit"].astype(str)
    grouped = interventions.groupby(["city", "unit", "intervention"], as_index=False).agg(
        total_u=("u", "sum"),
        total_e=("e", "sum"),
        total_cost=("effective_cost", "sum"),
        event_count=("event_id", "nunique"),
    )
    return grouped.merge(unit_attr, on=["city", "unit"], how="left")


def summarize_primitive_allocation(allocation: pd.DataFrame, unit_attr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if allocation.empty:
        return pd.DataFrame()
    unit_counts = unit_attr.groupby("city")["unit"].nunique().to_dict()
    for (city, primitive), group in allocation.groupby(["city", "intervention"]):
        cost = group["total_cost"].to_numpy(dtype=float)
        total_cost = float(np.nansum(cost))
        row: dict[str, Any] = {
            "city": city,
            "intervention": primitive,
            "total_cost": total_cost,
            "active_unit_count": int((cost > 1e-10).sum()),
            "active_unit_share": int((cost > 1e-10).sum()) / max(int(unit_counts.get(city, len(group))), 1),
            "cost_gini_across_active_units": gini(cost),
            "top_10pct_active_cost_share": top_share(cost, 0.10),
        }
        for attr in UNIT_ATTRIBUTES:
            values = group[f"{attr}_rank_pct"].to_numpy(dtype=float)
            row[f"cost_weighted_mean_{attr}_rank_pct"] = weighted_mean(values, cost)
            row[f"cost_share_in_top10_{attr}"] = cost_share(group[f"{attr}_top10"].to_numpy(dtype=bool), cost)
            row[f"cost_share_in_top25_{attr}"] = cost_share(group[f"{attr}_top25"].to_numpy(dtype=bool), cost)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["city", "intervention"])


def summarize_top_units(allocation: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    if allocation.empty:
        return pd.DataFrame()
    for (city, primitive), group in allocation.groupby(["city", "intervention"]):
        total = group["total_cost"].sum()
        top = group.sort_values("total_cost", ascending=False).head(top_n)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            rows.append(
                {
                    "city": city,
                    "intervention": primitive,
                    "rank": rank,
                    "unit": row.unit,
                    "total_cost": row.total_cost,
                    "primitive_cost_share": row.total_cost / total if total > 0 else np.nan,
                    "origin_exposure_rank_pct": row.origin_exposure_p_rank_pct,
                    "destination_importance_rank_pct": row.destination_importance_rank_pct,
                    "initial_deficit_rank_pct": row.initial_deficit_b0_rank_pct,
                    "out_degree_rank_pct": row.out_degree_rank_pct,
                    "in_degree_rank_pct": row.in_degree_rank_pct,
                }
            )
    return pd.DataFrame(rows)


def summarize_nontrivial_patterns(primitive_summary: pd.DataFrame) -> pd.DataFrame:
    if primitive_summary.empty:
        return pd.DataFrame()
    rows = []
    for primitive, group in primitive_summary.groupby("intervention"):
        rows.append(
            {
                "intervention": primitive,
                "mean_cost_share_outside_top10_origin_exposure": float(
                    np.nanmean(1.0 - group["cost_share_in_top10_origin_exposure_p"])
                ),
                "mean_cost_share_outside_top10_destination_importance": float(
                    np.nanmean(1.0 - group["cost_share_in_top10_destination_importance"])
                ),
                "mean_cost_share_outside_top10_initial_deficit": float(
                    np.nanmean(1.0 - group["cost_share_in_top10_initial_deficit_b0"])
                ),
                "mean_cost_weighted_origin_rank": float(np.nanmean(group["cost_weighted_mean_origin_exposure_p_rank_pct"])),
                "mean_cost_weighted_destination_rank": float(
                    np.nanmean(group["cost_weighted_mean_destination_importance_rank_pct"])
                ),
                "mean_cost_weighted_deficit_rank": float(np.nanmean(group["cost_weighted_mean_initial_deficit_b0_rank_pct"])),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    event_dataset: pd.DataFrame,
    city_dataset: pd.DataFrame,
    partial_corr: pd.DataFrame,
    primitive_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_city_boxplot(event_dataset, figure_dir / "event_recoverability_by_city.png")
    make_partial_corr_figure(partial_corr, figure_dir / "structure_partial_correlations.png")
    make_resource_rank_figure(primitive_summary, figure_dir / "resource_rank_profiles.png")
    make_scatter(city_dataset, "q_density", "mean_recoverable_fraction", figure_dir / "event_recoverability_vs_q_density.png")
    make_scatter(city_dataset, "od_density_observed", "mean_recoverable_fraction", figure_dir / "event_recoverability_vs_od_density.png")


def make_city_boxplot(event_dataset: pd.DataFrame, path: Path) -> None:
    if event_dataset.empty:
        return
    cities = (
        event_dataset.groupby("city")["recoverable_fraction"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    values = [event_dataset.loc[event_dataset["city"] == city, "recoverable_fraction"].dropna().to_numpy() for city in cities]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.boxplot(values, labels=cities, showmeans=True)
    ax.set_ylabel("Recoverable fraction")
    ax.set_title("Observed rainfall-event LP recoverability")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_partial_corr_figure(partial_corr: pd.DataFrame, path: Path) -> None:
    if partial_corr.empty:
        return
    top = partial_corr[partial_corr["control_set"] == "control_rain_intensity"].head(12).copy()
    top = top.sort_values("partial_pearson")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in top["partial_pearson"]]
    ax.barh(top["feature"], top["partial_pearson"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Partial Pearson with recoverable fraction")
    ax.set_title("Structure after controlling rainfall intensity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_resource_rank_figure(primitive_summary: pd.DataFrame, path: Path) -> None:
    if primitive_summary.empty:
        return
    cols = [
        "cost_weighted_mean_origin_exposure_p_rank_pct",
        "cost_weighted_mean_destination_importance_rank_pct",
        "cost_weighted_mean_initial_deficit_b0_rank_pct",
    ]
    means = primitive_summary.groupby("intervention")[cols].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(means.index))
    width = 0.24
    labels = ["origin exposure", "destination importance", "event deficit"]
    for idx, col in enumerate(cols):
        ax.bar(x + (idx - 1) * width, means[col], width=width, label=labels[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cost-weighted unit rank percentile")
    ax.set_title("Where R/C/S resources concentrate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_scatter(dataset: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    if x_col not in dataset.columns or y_col not in dataset.columns:
        return
    df = dataset[["city", x_col, y_col]].dropna()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(df[x_col], df[y_col], s=70, color="#0f766e")
    for row in df.itertuples():
        ax.annotate(row.city, (getattr(row, x_col), getattr(row, y_col)), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    city_dataset: pd.DataFrame,
    city_corr: pd.DataFrame,
    partial_corr: pd.DataFrame,
    primitive_summary: pd.DataFrame,
    nontrivial: pd.DataFrame,
    top_units: pd.DataFrame,
) -> None:
    city_columns = [
        "city",
        "event_count",
        "mean_recoverable_fraction",
        "median_recoverable_fraction",
        "mean_event_peak_positive_abnormal_deficit",
        "n_units",
        "q_density",
        "od_density_observed",
        "cost_share_R",
        "cost_share_C",
        "cost_share_S",
    ]
    corr_cols = ["feature", "target", "n", "spearman", "pearson"]
    partial_cols = ["control_set", "feature", "n_events", "n_cities", "partial_pearson"]
    primitive_cols = [
        "city",
        "intervention",
        "active_unit_share",
        "top_10pct_active_cost_share",
        "cost_weighted_mean_origin_exposure_p_rank_pct",
        "cost_weighted_mean_destination_importance_rank_pct",
        "cost_weighted_mean_initial_deficit_b0_rank_pct",
        "cost_share_in_top10_origin_exposure_p",
        "cost_share_in_top10_destination_importance",
        "cost_share_in_top10_initial_deficit_b0",
    ]
    text = "\n".join(
        [
            "# Observed-Event City-Structure Analysis",
            "",
            "## 这版模型到底优化什么",
            "",
            "这一版不再是一个城市一个集计代表场景，而是一个 speed-overlap 且出现正异常速度损失的 rainfall event 对应一个 12 小时 LP。状态 0 是该事件开始小时；状态 1-12 是事件开始后的 12 个小时。",
            "",
            "`b0` 来自事件开始小时相对 matched temporal baseline 的正异常速度损失；`A` 来自动态标定的自然滞留率；`h[t]` 是观测异常序列在扣除 `A * y[t-1]` 后仍然增加的正创新量。因此这里的 `h` 不再是所有降雨平均出来的固定 profile。",
            "",
            "## 城市层结果",
            "",
            dataframe_to_markdown(city_dataset[[col for col in city_columns if col in city_dataset.columns]], max_rows=20),
            "",
            "## 城市结构相关性",
            "",
            "样本城市数只有 7，所以这些相关性用于发现结构假设，不作为最终 law 的显著性证明。",
            "",
            dataframe_to_markdown(city_corr[city_corr["target"] == "mean_recoverable_fraction"][corr_cols].head(12)),
            "",
            "## 抛开降雨强度后的结构信号",
            "",
            "下表在 event level 上先控制 `event_total_precip` 与 `event_peak_precip`，再看城市结构变量与 recoverable fraction 的残差相关。由于同一城市有多个事件，独立样本数不能简单按事件数理解，但它能帮助判断结果是否只是降雨强度驱动。",
            "",
            dataframe_to_markdown(partial_corr[partial_corr["control_set"] == "control_rain_intensity"][partial_cols].head(12)),
            "",
            "## R/C/S 投放倾向",
            "",
            dataframe_to_markdown(nontrivial, max_rows=10),
            "",
            dataframe_to_markdown(primitive_summary[primitive_cols].head(30), max_rows=30),
            "",
            "## Top allocated units",
            "",
            dataframe_to_markdown(top_units.head(30), max_rows=30),
            "",
            "## 初步解释",
            "",
            "1. 这版结论更接近城市结构问题：所有城市使用相同 budget rule、delay、cost、eta 与 diminishing returns；变化来自 OD 结构、速度异常形态和真实事件时间序列。",
            "2. 如果结构变量在控制降雨强度后仍与 recoverability 相关，它比“预算越大越好”更值得写成 paper 的 law，因为它不是模型旋钮的直接结果。",
            "3. R/C/S 的投放不会完全等同于“最高活跃度区域优先”。若大量成本落在 top-10% origin exposure 之外，说明优化在寻找网络依赖位置、destination importance 或损失传播杠杆，而不是简单追逐 OD 活跃度。",
            "4. 当前数据仍是 observational calibration。真正能闭合因果链的下一步，是把这些 event-level LP 输出作为结构假设，再进入模型部分做更严格的 counterfactual 验证。",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


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

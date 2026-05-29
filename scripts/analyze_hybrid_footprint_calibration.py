"""Test how TMC-derived event footprints would change action-value laws.

V33 showed that raw TMC speed abnormalities contain event-zone spatial
footprints that differ sharply from the current OD-vulnerability template.
This script is the next diagnostic step: it keeps the observed city-level
event signal fixed, replaces only the spatial allocation of b0/h with a
hybrid OD-template + TMC-footprint field, and compares the resulting
first-order action-value law with the current calibration.

The script intentionally does not solve new LPs.  It asks whether the
optimizer-derived learning target would materially change before paying the
cost of a full hybrid-calibration LP rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learn_recovery_laws import build_event_action_frame, event_concentration
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import RecoveryLPParameters


EPS = 1e-12
DEFAULT_FOOTPRINT_BLENDS = (0.50,)
DEFAULT_MAIN_BLEND = 0.50
warnings.filterwarnings("ignore", category=FutureWarning, module="learn_recovery_laws")
EMPTY_INTERVENTIONS = pd.DataFrame(
    {
        "city": pd.Series(dtype="object"),
        "event_id": pd.Series(dtype="int64"),
        "scenario": pd.Series(dtype="object"),
        "unit": pd.Series(dtype="object"),
        "t": pd.Series(dtype="int64"),
        "intervention": pd.Series(dtype="object"),
        "optimized_u": pd.Series(dtype="float64"),
        "optimized_e": pd.Series(dtype="float64"),
        "optimized_cost": pd.Series(dtype="float64"),
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/hybrid_footprint_calibration")
    parser.add_argument("--footprint-blends", nargs="*", type=float, default=list(DEFAULT_FOOTPRINT_BLENDS))
    parser.add_argument("--main-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root)
    event_metrics, concentration, city_summary, metrics = run_analysis(
        root,
        config,
        data,
        footprint_blends=tuple(args.footprint_blends),
        main_blend=float(args.main_blend),
        footprint_floor=float(args.footprint_floor),
        max_relative=float(args.max_relative),
    )

    write_table(event_metrics, table_dir / "hybrid_footprint_event_metrics.csv")
    write_table(concentration, table_dir / "hybrid_footprint_concentration.csv")
    write_table(city_summary, table_dir / "hybrid_footprint_city_summary.csv")
    (table_dir / "hybrid_footprint_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(event_metrics, city_summary, figure_dir)
    write_report(report_dir / "hybrid_footprint_calibration_report_zh.md", metrics, city_summary)
    print(f"Wrote hybrid footprint calibration sensitivity to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(root / "results" / "event_optimization" / "tables" / "event_optimization_summary.csv"),
        "events": pd.read_csv(
            root / "results" / "data_mining" / "tables" / "rainfall_event_impact_details.csv",
            parse_dates=["event_start", "event_end"],
        ),
        "dynamics": pd.read_csv(root / "results" / "event_calibration" / "tables" / "event_dynamic_calibration_summary.csv"),
        "abnormal": pd.read_csv(
            root / "results" / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv",
            parse_dates=["hour"],
        ),
        "footprint_zone": pd.read_csv(
            root / "results" / "event_spatial_footprint_proxy" / "tables" / "event_zone_speed_footprint.csv.gz"
        ),
        "footprint_summary": pd.read_csv(
            root / "results" / "event_spatial_footprint_proxy" / "tables" / "event_spatial_footprint_summary.csv"
        ),
    }


def run_analysis(
    root: Path,
    config: dict[str, Any],
    data: dict[str, pd.DataFrame],
    *,
    footprint_blends: tuple[float, ...],
    main_blend: float,
    footprint_floor: float,
    max_relative: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    abnormal = data["abnormal"].copy()
    footprint = data["footprint_zone"].copy()
    footprint["event_id"] = pd.to_numeric(footprint["event_id"], errors="coerce").astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }

    event_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    total_events = len(summary)
    for idx, row in enumerate(summary.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{total_events}] Hybrid footprint sensitivity for {city} event {event_id}", flush=True)
        event_row = event_lookup.get((city, event_id))
        if event_row is None:
            continue
        footprint_group = footprint_groups.get((city, event_id), pd.DataFrame())
        if footprint_group.empty:
            continue
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        base_summary = series_from_row(row)
        base_full = build_event_action_frame(base_params, base_summary, event_row, EMPTY_INTERVENTIONS)
        base_conc = event_concentration(base_summary, base_full)
        base_finite_conc = value_tail_stats(base_full, "finite_deficit_area_value")
        base_unit_value = unit_value_table(base_full, "marginal_resource_value")
        base_finite_unit_value = unit_value_table(base_full, "finite_deficit_area_value")
        concentration_rows.append(
            {
                **base_conc,
                "calibration": "od_template",
                "footprint_blend": 0.0,
                "baseline_objective_recomputed": no_intervention_objective(base_params),
                **{f"finite_{key}": value for key, value in base_finite_conc.items()},
            }
        )

        for blend in footprint_blends:
            hybrid_params, diagnostics = build_hybrid_params(
                base_params,
                footprint_group,
                footprint_blend=float(blend),
                footprint_floor=footprint_floor,
                max_relative=max_relative,
            )
            hybrid_summary = base_summary.copy()
            hybrid_summary["baseline_objective"] = no_intervention_objective(hybrid_params)
            hybrid_summary["optimized_objective"] = np.nan
            hybrid_summary["recoverable_fraction"] = np.nan
            hybrid_summary["total_budget"] = base_summary["total_budget"]
            hybrid_full = build_event_action_frame(hybrid_params, hybrid_summary, event_row, EMPTY_INTERVENTIONS)
            hybrid_conc = event_concentration(hybrid_summary, hybrid_full)
            hybrid_finite_conc = value_tail_stats(hybrid_full, "finite_deficit_area_value")
            hybrid_unit_value = unit_value_table(hybrid_full, "marginal_resource_value")
            hybrid_finite_unit_value = unit_value_table(hybrid_full, "finite_deficit_area_value")
            comparison = compare_action_fields(
                city,
                event_id,
                base_full,
                hybrid_full,
                base_unit_value,
                hybrid_unit_value,
                base_finite_unit_value,
                hybrid_finite_unit_value,
                footprint_group,
            )
            concentration_rows.append(
                {
                    **hybrid_conc,
                    "calibration": "hybrid_footprint",
                    "footprint_blend": float(blend),
                    "baseline_objective_recomputed": float(hybrid_summary["baseline_objective"]),
                    **{f"finite_{key}": value for key, value in hybrid_finite_conc.items()},
                    **diagnostics,
                }
            )
            event_rows.append(
                {
                    "city": city,
                    "event_id": event_id,
                    "event_start": str(row.event_start),
                    "footprint_blend": float(blend),
                    "base_top_5pct_value_share": base_conc["top_5pct_value_share"],
                    "hybrid_top_5pct_value_share": hybrid_conc["top_5pct_value_share"],
                    "delta_top_5pct_value_share": hybrid_conc["top_5pct_value_share"] - base_conc["top_5pct_value_share"],
                    "base_marginal_value_gini": base_conc["marginal_value_gini"],
                    "hybrid_marginal_value_gini": hybrid_conc["marginal_value_gini"],
                    "delta_marginal_value_gini": hybrid_conc["marginal_value_gini"] - base_conc["marginal_value_gini"],
                    "base_total_marginal_value_proxy": base_conc["total_marginal_value_proxy"],
                    "hybrid_total_marginal_value_proxy": hybrid_conc["total_marginal_value_proxy"],
                    "hybrid_to_base_total_value_ratio": hybrid_conc["total_marginal_value_proxy"]
                    / max(base_conc["total_marginal_value_proxy"], EPS),
                    "base_finite_top_5pct_value_share": base_finite_conc["top_5pct_value_share"],
                    "hybrid_finite_top_5pct_value_share": hybrid_finite_conc["top_5pct_value_share"],
                    "delta_finite_top_5pct_value_share": hybrid_finite_conc["top_5pct_value_share"]
                    - base_finite_conc["top_5pct_value_share"],
                    "base_finite_value_gini": base_finite_conc["marginal_value_gini"],
                    "hybrid_finite_value_gini": hybrid_finite_conc["marginal_value_gini"],
                    "delta_finite_value_gini": hybrid_finite_conc["marginal_value_gini"] - base_finite_conc["marginal_value_gini"],
                    "base_total_finite_value_proxy": base_finite_conc["total_marginal_value_proxy"],
                    "hybrid_total_finite_value_proxy": hybrid_finite_conc["total_marginal_value_proxy"],
                    "hybrid_to_base_total_finite_value_ratio": hybrid_finite_conc["total_marginal_value_proxy"]
                    / max(base_finite_conc["total_marginal_value_proxy"], EPS),
                    "base_baseline_objective": base_conc["baseline_objective"],
                    "hybrid_baseline_objective": hybrid_conc["baseline_objective"],
                    "hybrid_to_base_baseline_objective_ratio": hybrid_conc["baseline_objective"]
                    / max(base_conc["baseline_objective"], EPS),
                    "event_peak_positive_abnormal_deficit": float(row.event_peak_positive_abnormal_deficit),
                    "event_total_precip": float(row.event_total_precip),
                    **diagnostics,
                    **comparison,
                }
            )
    event_metrics = pd.DataFrame(event_rows).sort_values(["footprint_blend", "city", "event_start", "event_id"])
    concentration = pd.DataFrame(concentration_rows).sort_values(["calibration", "footprint_blend", "city", "event_start", "event_id"])
    city_summary = build_city_summary(event_metrics)
    metrics = build_metrics(event_metrics, city_summary, main_blend=main_blend)
    return event_metrics, concentration, city_summary, metrics


def series_from_row(row: Any) -> pd.Series:
    return pd.Series(row._asdict() if hasattr(row, "_asdict") else dict(row))


def build_hybrid_params(
    base_params: RecoveryLPParameters,
    footprint_group: pd.DataFrame,
    *,
    footprint_blend: float,
    footprint_floor: float,
    max_relative: float,
) -> tuple[RecoveryLPParameters, dict[str, float]]:
    p = np.asarray(base_params.p, dtype=float)
    od_relative = recover_spatial_relative(base_params)
    footprint_relative = footprint_relative_vector(
        base_params.units,
        p,
        footprint_group,
        floor=footprint_floor,
        max_relative=max_relative,
    )
    hybrid_relative = (1.0 - footprint_blend) * od_relative + footprint_blend * footprint_relative
    hybrid_relative = normalize_relative(hybrid_relative, p)
    hybrid_relative = clip_relative(hybrid_relative, p, max_relative=max_relative)

    city_b0_signal = float(np.sum(p * base_params.b0))
    city_h_signal = np.asarray(base_params.h.T @ p, dtype=float)
    hybrid_b0 = np.clip(city_b0_signal * hybrid_relative, 0.0, 0.90)
    hybrid_h = np.zeros_like(base_params.h)
    for t in range(1, base_params.horizon + 1):
        hybrid_h[:, t] = np.clip(city_h_signal[t] * hybrid_relative, 0.0, 0.30)
    metadata = dict(base_params.metadata or {})
    metadata.update(
        {
            "spatial_calibration": "hybrid_od_template_tmc_footprint",
            "footprint_blend": float(footprint_blend),
            "footprint_floor": float(footprint_floor),
            "footprint_max_relative": float(max_relative),
        }
    )
    hybrid = RecoveryLPParameters(
        city=base_params.city,
        units=list(base_params.units),
        p=base_params.p.copy(),
        q=base_params.q.copy(),
        b0=hybrid_b0,
        a=base_params.a.copy(),
        h=hybrid_h,
        eta={k: v.copy() for k, v in base_params.eta.items()},
        cost={k: v.copy() for k, v in base_params.cost.items()},
        u_cap={k: v.copy() for k, v in (base_params.u_cap or {}).items()},
        u_segment_cap={k: v.copy() for k, v in (base_params.u_segment_cap or {}).items()} or None,
        segment_effectiveness={k: v.copy() for k, v in (base_params.segment_effectiveness or {}).items()} or None,
        period_budget=base_params.period_budget.copy(),
        total_budget=float(base_params.total_budget),
        delays=dict(base_params.delays),
        delta_c=float(base_params.delta_c),
        delta_s=float(base_params.delta_s),
        delta_t=float(base_params.delta_t),
        metadata=metadata,
    )
    diagnostics = {
        "footprint_relative_max": float(np.max(footprint_relative)),
        "footprint_relative_gini": gini(footprint_relative),
        "hybrid_relative_max": float(np.max(hybrid_relative)),
        "hybrid_relative_gini": gini(hybrid_relative),
        "od_footprint_relative_cosine": cosine_similarity(od_relative, footprint_relative),
        "od_hybrid_relative_cosine": cosine_similarity(od_relative, hybrid_relative),
    }
    return hybrid, diagnostics


def recover_spatial_relative(params: RecoveryLPParameters) -> np.ndarray:
    p = np.asarray(params.p, dtype=float)
    city_b0 = float(np.sum(p * params.b0))
    if city_b0 > EPS:
        return normalize_relative(params.b0 / city_b0, p)
    h_total = params.h.sum(axis=1)
    city_h = float(np.sum(p * h_total))
    if city_h > EPS:
        return normalize_relative(h_total / city_h, p)
    return np.ones(params.n_units, dtype=float)


def footprint_relative_vector(
    units: list[str],
    p: np.ndarray,
    footprint_group: pd.DataFrame,
    *,
    floor: float,
    max_relative: float,
) -> np.ndarray:
    weight_by_zone = footprint_group.groupby("zone_id")["zone_weight"].sum()
    weights = pd.Series(0.0, index=pd.Index(units, dtype=str))
    weights.loc[weights.index.intersection(weight_by_zone.index.astype(str))] = weight_by_zone.reindex(
        weights.index.intersection(weight_by_zone.index.astype(str)),
        fill_value=0.0,
    ).to_numpy(dtype=float)
    values = weights.to_numpy(dtype=float)
    values = np.clip(values, 0.0, None)
    if values.sum() <= EPS:
        values = np.full(len(units), 1.0 / max(len(units), 1), dtype=float)
    else:
        values = values / values.sum()
    n = len(values)
    floor = float(np.clip(floor, 0.0, 0.95))
    values = (1.0 - floor) * values + floor * (1.0 / max(n, 1))
    values = values / max(values.sum(), EPS)
    relative = values / max(float(np.sum(p * values)), EPS)
    return clip_relative(relative, p, max_relative=max_relative)


def normalize_relative(relative: np.ndarray, p: np.ndarray) -> np.ndarray:
    relative = np.asarray(relative, dtype=float)
    relative = np.clip(relative, 0.0, None)
    denom = max(float(np.sum(p * relative)), EPS)
    return relative / denom


def clip_relative(relative: np.ndarray, p: np.ndarray, *, max_relative: float) -> np.ndarray:
    clipped = np.clip(relative, 0.0, float(max_relative))
    return normalize_relative(clipped, p)


def no_intervention_objective(params: RecoveryLPParameters) -> float:
    b = params.b0.copy()
    objective = 0.0
    for t in range(params.horizon + 1):
        ell = np.clip(params.q @ b, 0.0, 1.0)
        objective += float(params.delta_t * np.sum(params.p * ell))
        if t == params.horizon:
            break
        b = np.clip(params.a * b + params.h[:, t + 1], 0.0, 1.0)
    return objective


def value_tail_stats(full: pd.DataFrame, value_col: str) -> dict[str, float]:
    values = pd.to_numeric(full[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(values.sum())
    top_n = max(1, int(math.ceil(0.05 * len(values))))
    top_sum = float(values.sort_values(ascending=False).head(top_n).sum())
    return {
        "top_5pct_value_share": top_sum / max(total, EPS),
        "marginal_value_gini": gini(values.to_numpy(dtype=float)),
        "total_marginal_value_proxy": total,
    }


def unit_value_table(full: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        full.groupby("unit", as_index=False)
        .agg(unit_value=(value_col, "sum"))
        .sort_values("unit_value", ascending=False)
    )


def compare_action_fields(
    city: str,
    event_id: int,
    base_full: pd.DataFrame,
    hybrid_full: pd.DataFrame,
    base_unit_value: pd.DataFrame,
    hybrid_unit_value: pd.DataFrame,
    base_finite_unit_value: pd.DataFrame,
    hybrid_finite_unit_value: pd.DataFrame,
    footprint_group: pd.DataFrame,
) -> dict[str, float]:
    keys = ["unit", "t", "intervention"]
    base = base_full[keys + ["marginal_resource_value"]].rename(columns={"marginal_resource_value": "base_value"})
    hybrid = hybrid_full[keys + ["marginal_resource_value"]].rename(columns={"marginal_resource_value": "hybrid_value"})
    pair = base.merge(hybrid, on=keys, how="inner")
    action_spearman = safe_corr(pair["base_value"], pair["hybrid_value"], method="spearman")
    action_pearson = safe_corr(pair["base_value"], pair["hybrid_value"], method="pearson")
    top5_action_base = top_action_set(pair, "base_value", 0.05)
    top5_action_hybrid = top_action_set(pair, "hybrid_value", 0.05)
    top1_action_base = top_action_set(pair, "base_value", 0.01)
    top1_action_hybrid = top_action_set(pair, "hybrid_value", 0.01)
    base_top20_units = top_unit_set(base_unit_value, 20)
    hybrid_top20_units = top_unit_set(hybrid_unit_value, 20)
    n_units = max(base_unit_value["unit"].nunique(), hybrid_unit_value["unit"].nunique())
    top5pct_n = max(1, int(math.ceil(0.05 * n_units)))
    base_top5pct_units = top_unit_set(base_unit_value, top5pct_n)
    hybrid_top5pct_units = top_unit_set(hybrid_unit_value, top5pct_n)
    footprint_top20_units = set(footprint_group.sort_values("zone_weight", ascending=False).head(20)["zone_id"].astype(str))
    footprint_top5pct_units = set(footprint_group.sort_values("zone_weight", ascending=False).head(top5pct_n)["zone_id"].astype(str))
    footprint_mass = footprint_group.groupby("zone_id")["zone_weight"].sum()
    base_top20_mass = set_mass(footprint_mass, base_top20_units)
    hybrid_top20_mass = set_mass(footprint_mass, hybrid_top20_units)
    base_top5pct_mass = set_mass(footprint_mass, base_top5pct_units)
    hybrid_top5pct_mass = set_mass(footprint_mass, hybrid_top5pct_units)

    finite_base = base_full[keys + ["finite_deficit_area_value"]].rename(columns={"finite_deficit_area_value": "base_value"})
    finite_hybrid = hybrid_full[keys + ["finite_deficit_area_value"]].rename(columns={"finite_deficit_area_value": "hybrid_value"})
    finite_pair = finite_base.merge(finite_hybrid, on=keys, how="inner")
    finite_top5_action_base = top_action_set(finite_pair, "base_value", 0.05)
    finite_top5_action_hybrid = top_action_set(finite_pair, "hybrid_value", 0.05)
    finite_top1_action_base = top_action_set(finite_pair, "base_value", 0.01)
    finite_top1_action_hybrid = top_action_set(finite_pair, "hybrid_value", 0.01)
    finite_base_top20_units = top_unit_set(base_finite_unit_value, 20)
    finite_hybrid_top20_units = top_unit_set(hybrid_finite_unit_value, 20)
    finite_base_top5pct_units = top_unit_set(base_finite_unit_value, top5pct_n)
    finite_hybrid_top5pct_units = top_unit_set(hybrid_finite_unit_value, top5pct_n)
    finite_base_top20_mass = set_mass(footprint_mass, finite_base_top20_units)
    finite_hybrid_top20_mass = set_mass(footprint_mass, finite_hybrid_top20_units)
    finite_base_top5pct_mass = set_mass(footprint_mass, finite_base_top5pct_units)
    finite_hybrid_top5pct_mass = set_mass(footprint_mass, finite_hybrid_top5pct_units)

    return {
        "action_value_spearman": action_spearman,
        "action_value_pearson": action_pearson,
        "top5pct_action_jaccard": jaccard(top5_action_base, top5_action_hybrid),
        "top1pct_action_jaccard": jaccard(top1_action_base, top1_action_hybrid),
        "top20_unit_jaccard_base_hybrid": jaccard(base_top20_units, hybrid_top20_units),
        "top5pct_unit_jaccard_base_hybrid": jaccard(base_top5pct_units, hybrid_top5pct_units),
        "base_top20_unit_footprint_jaccard": jaccard(base_top20_units, footprint_top20_units),
        "hybrid_top20_unit_footprint_jaccard": jaccard(hybrid_top20_units, footprint_top20_units),
        "delta_top20_unit_footprint_jaccard": jaccard(hybrid_top20_units, footprint_top20_units)
        - jaccard(base_top20_units, footprint_top20_units),
        "base_top5pct_unit_footprint_jaccard": jaccard(base_top5pct_units, footprint_top5pct_units),
        "hybrid_top5pct_unit_footprint_jaccard": jaccard(hybrid_top5pct_units, footprint_top5pct_units),
        "delta_top5pct_unit_footprint_jaccard": jaccard(hybrid_top5pct_units, footprint_top5pct_units)
        - jaccard(base_top5pct_units, footprint_top5pct_units),
        "base_top20_units_footprint_mass": base_top20_mass,
        "hybrid_top20_units_footprint_mass": hybrid_top20_mass,
        "delta_top20_units_footprint_mass": hybrid_top20_mass - base_top20_mass,
        "base_top5pct_units_footprint_mass": base_top5pct_mass,
        "hybrid_top5pct_units_footprint_mass": hybrid_top5pct_mass,
        "delta_top5pct_units_footprint_mass": hybrid_top5pct_mass - base_top5pct_mass,
        "finite_action_value_spearman": safe_corr(finite_pair["base_value"], finite_pair["hybrid_value"], method="spearman"),
        "finite_action_value_pearson": safe_corr(finite_pair["base_value"], finite_pair["hybrid_value"], method="pearson"),
        "finite_top5pct_action_jaccard": jaccard(finite_top5_action_base, finite_top5_action_hybrid),
        "finite_top1pct_action_jaccard": jaccard(finite_top1_action_base, finite_top1_action_hybrid),
        "finite_top20_unit_jaccard_base_hybrid": jaccard(finite_base_top20_units, finite_hybrid_top20_units),
        "finite_top5pct_unit_jaccard_base_hybrid": jaccard(finite_base_top5pct_units, finite_hybrid_top5pct_units),
        "base_finite_top20_unit_footprint_jaccard": jaccard(finite_base_top20_units, footprint_top20_units),
        "hybrid_finite_top20_unit_footprint_jaccard": jaccard(finite_hybrid_top20_units, footprint_top20_units),
        "delta_finite_top20_unit_footprint_jaccard": jaccard(finite_hybrid_top20_units, footprint_top20_units)
        - jaccard(finite_base_top20_units, footprint_top20_units),
        "base_finite_top5pct_unit_footprint_jaccard": jaccard(finite_base_top5pct_units, footprint_top5pct_units),
        "hybrid_finite_top5pct_unit_footprint_jaccard": jaccard(finite_hybrid_top5pct_units, footprint_top5pct_units),
        "delta_finite_top5pct_unit_footprint_jaccard": jaccard(finite_hybrid_top5pct_units, footprint_top5pct_units)
        - jaccard(finite_base_top5pct_units, footprint_top5pct_units),
        "base_finite_top20_units_footprint_mass": finite_base_top20_mass,
        "hybrid_finite_top20_units_footprint_mass": finite_hybrid_top20_mass,
        "delta_finite_top20_units_footprint_mass": finite_hybrid_top20_mass - finite_base_top20_mass,
        "base_finite_top5pct_units_footprint_mass": finite_base_top5pct_mass,
        "hybrid_finite_top5pct_units_footprint_mass": finite_hybrid_top5pct_mass,
        "delta_finite_top5pct_units_footprint_mass": finite_hybrid_top5pct_mass - finite_base_top5pct_mass,
        "footprint_zone_count": int(footprint_group["zone_id"].nunique()),
    }


def top_action_set(pair: pd.DataFrame, column: str, pct: float) -> set[tuple[str, int, str]]:
    n = max(1, int(math.ceil(len(pair) * pct)))
    rows = pair.sort_values(column, ascending=False).head(n)
    return set((str(row.unit), int(row.t), str(row.intervention)) for row in rows.itertuples(index=False))


def top_unit_set(unit_value: pd.DataFrame, n: int) -> set[str]:
    return set(unit_value.sort_values("unit_value", ascending=False).head(max(1, int(n)))["unit"].astype(str))


def set_mass(weights: pd.Series, units: set[str]) -> float:
    if not units:
        return 0.0
    index = pd.Index([str(unit) for unit in units])
    return float(weights.reindex(index, fill_value=0.0).sum())


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (blend, city), group in event_metrics.groupby(["footprint_blend", "city"], sort=True):
        rows.append(
            {
                "footprint_blend": float(blend),
                "city": city,
                "n_events": int(group["event_id"].nunique()),
                "mean_action_value_spearman": float(group["action_value_spearman"].mean()),
                "mean_top5pct_action_jaccard": float(group["top5pct_action_jaccard"].mean()),
                "mean_top20_unit_jaccard_base_hybrid": float(group["top20_unit_jaccard_base_hybrid"].mean()),
                "mean_finite_action_value_spearman": float(group["finite_action_value_spearman"].mean()),
                "mean_finite_top5pct_action_jaccard": float(group["finite_top5pct_action_jaccard"].mean()),
                "mean_finite_top20_unit_jaccard_base_hybrid": float(group["finite_top20_unit_jaccard_base_hybrid"].mean()),
                "mean_base_top20_units_footprint_mass": float(group["base_top20_units_footprint_mass"].mean()),
                "mean_hybrid_top20_units_footprint_mass": float(group["hybrid_top20_units_footprint_mass"].mean()),
                "mean_delta_top20_units_footprint_mass": float(group["delta_top20_units_footprint_mass"].mean()),
                "mean_base_top5pct_units_footprint_mass": float(group["base_top5pct_units_footprint_mass"].mean()),
                "mean_hybrid_top5pct_units_footprint_mass": float(group["hybrid_top5pct_units_footprint_mass"].mean()),
                "mean_delta_top5pct_units_footprint_mass": float(group["delta_top5pct_units_footprint_mass"].mean()),
                "mean_base_finite_top20_units_footprint_mass": float(group["base_finite_top20_units_footprint_mass"].mean()),
                "mean_hybrid_finite_top20_units_footprint_mass": float(group["hybrid_finite_top20_units_footprint_mass"].mean()),
                "mean_delta_finite_top20_units_footprint_mass": float(group["delta_finite_top20_units_footprint_mass"].mean()),
                "mean_base_finite_top5pct_units_footprint_mass": float(group["base_finite_top5pct_units_footprint_mass"].mean()),
                "mean_hybrid_finite_top5pct_units_footprint_mass": float(group["hybrid_finite_top5pct_units_footprint_mass"].mean()),
                "mean_delta_finite_top5pct_units_footprint_mass": float(group["delta_finite_top5pct_units_footprint_mass"].mean()),
                "base_top5_value_share_range": float(group["base_top_5pct_value_share"].max() - group["base_top_5pct_value_share"].min()),
                "hybrid_top5_value_share_range": float(group["hybrid_top_5pct_value_share"].max() - group["hybrid_top_5pct_value_share"].min()),
                "mean_delta_top5_value_share": float(group["delta_top_5pct_value_share"].mean()),
                "mean_abs_delta_top5_value_share": float(group["delta_top_5pct_value_share"].abs().mean()),
                "base_finite_top5_value_share_range": float(
                    group["base_finite_top_5pct_value_share"].max() - group["base_finite_top_5pct_value_share"].min()
                ),
                "hybrid_finite_top5_value_share_range": float(
                    group["hybrid_finite_top_5pct_value_share"].max() - group["hybrid_finite_top_5pct_value_share"].min()
                ),
                "mean_delta_finite_top5_value_share": float(group["delta_finite_top_5pct_value_share"].mean()),
                "mean_abs_delta_finite_top5_value_share": float(group["delta_finite_top_5pct_value_share"].abs().mean()),
                "mean_hybrid_to_base_baseline_objective_ratio": float(group["hybrid_to_base_baseline_objective_ratio"].mean()),
                "mean_od_hybrid_relative_cosine": float(group["od_hybrid_relative_cosine"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["footprint_blend", "mean_action_value_spearman"])


def build_metrics(event_metrics: pd.DataFrame, city_summary: pd.DataFrame, *, main_blend: float) -> dict[str, Any]:
    main = event_metrics[np.isclose(event_metrics["footprint_blend"], main_blend)].copy()
    main_city = city_summary[np.isclose(city_summary["footprint_blend"], main_blend)].copy()
    return {
        "main_blend": float(main_blend),
        "n_events": int(main["event_id"].nunique()) if not main.empty else 0,
        "n_cities": int(main["city"].nunique()) if not main.empty else 0,
        "mean_action_value_spearman": float(main["action_value_spearman"].mean()) if not main.empty else np.nan,
        "median_action_value_spearman": float(main["action_value_spearman"].median()) if not main.empty else np.nan,
        "mean_top5pct_action_jaccard": float(main["top5pct_action_jaccard"].mean()) if not main.empty else np.nan,
        "mean_top20_unit_jaccard_base_hybrid": float(main["top20_unit_jaccard_base_hybrid"].mean()) if not main.empty else np.nan,
        "mean_finite_action_value_spearman": float(main["finite_action_value_spearman"].mean()) if not main.empty else np.nan,
        "median_finite_action_value_spearman": float(main["finite_action_value_spearman"].median()) if not main.empty else np.nan,
        "mean_finite_top5pct_action_jaccard": float(main["finite_top5pct_action_jaccard"].mean()) if not main.empty else np.nan,
        "mean_finite_top20_unit_jaccard_base_hybrid": float(main["finite_top20_unit_jaccard_base_hybrid"].mean()) if not main.empty else np.nan,
        "mean_base_top20_units_footprint_mass": float(main["base_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_hybrid_top20_units_footprint_mass": float(main["hybrid_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_delta_top20_units_footprint_mass": float(main["delta_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_base_top5pct_units_footprint_mass": float(main["base_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_hybrid_top5pct_units_footprint_mass": float(main["hybrid_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_delta_top5pct_units_footprint_mass": float(main["delta_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_base_finite_top20_units_footprint_mass": float(main["base_finite_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_hybrid_finite_top20_units_footprint_mass": float(main["hybrid_finite_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_delta_finite_top20_units_footprint_mass": float(main["delta_finite_top20_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_base_finite_top5pct_units_footprint_mass": float(main["base_finite_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_hybrid_finite_top5pct_units_footprint_mass": float(main["hybrid_finite_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_delta_finite_top5pct_units_footprint_mass": float(main["delta_finite_top5pct_units_footprint_mass"].mean()) if not main.empty else np.nan,
        "mean_delta_top5_value_share": float(main["delta_top_5pct_value_share"].mean()) if not main.empty else np.nan,
        "mean_abs_delta_top5_value_share": float(main["delta_top_5pct_value_share"].abs().mean()) if not main.empty else np.nan,
        "mean_delta_finite_top5_value_share": float(main["delta_finite_top_5pct_value_share"].mean()) if not main.empty else np.nan,
        "mean_abs_delta_finite_top5_value_share": float(main["delta_finite_top_5pct_value_share"].abs().mean()) if not main.empty else np.nan,
        "base_top5_zero_variance_city_count": int((main_city["base_top5_value_share_range"] <= 1e-10).sum()) if not main_city.empty else 0,
        "hybrid_top5_zero_variance_city_count": int((main_city["hybrid_top5_value_share_range"] <= 1e-10).sum()) if not main_city.empty else 0,
        "base_finite_top5_zero_variance_city_count": int((main_city["base_finite_top5_value_share_range"] <= 1e-10).sum())
        if not main_city.empty
        else 0,
        "hybrid_finite_top5_zero_variance_city_count": int((main_city["hybrid_finite_top5_value_share_range"] <= 1e-10).sum())
        if not main_city.empty
        else 0,
        "mean_base_top5_value_share_range": float(main_city["base_top5_value_share_range"].mean()) if not main_city.empty else np.nan,
        "mean_hybrid_top5_value_share_range": float(main_city["hybrid_top5_value_share_range"].mean()) if not main_city.empty else np.nan,
        "mean_base_finite_top5_value_share_range": float(main_city["base_finite_top5_value_share_range"].mean()) if not main_city.empty else np.nan,
        "mean_hybrid_finite_top5_value_share_range": float(main_city["hybrid_finite_top5_value_share_range"].mean()) if not main_city.empty else np.nan,
        "mean_hybrid_to_base_baseline_objective_ratio": float(main["hybrid_to_base_baseline_objective_ratio"].mean()) if not main.empty else np.nan,
        "lowest_action_spearman_city": str(main_city.iloc[0]["city"]) if not main_city.empty else "",
        "highest_footprint_capture_gain_city": str(
            main_city.sort_values("mean_delta_finite_top5pct_units_footprint_mass", ascending=False).iloc[0]["city"]
        )
        if not main_city.empty
        else "",
    }


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_figures(event_metrics: pd.DataFrame, city_summary: pd.DataFrame, figure_dir: Path) -> None:
    if event_metrics.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    main_blend = DEFAULT_MAIN_BLEND if DEFAULT_MAIN_BLEND in set(event_metrics["footprint_blend"]) else event_metrics["footprint_blend"].median()
    main = event_metrics[np.isclose(event_metrics["footprint_blend"], main_blend)].copy()
    main_city = city_summary[np.isclose(city_summary["footprint_blend"], main_blend)].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for blend, group in event_metrics.groupby("footprint_blend"):
        ax.scatter(
            group["finite_action_value_spearman"],
            group["delta_finite_top5pct_units_footprint_mass"],
            s=34,
            alpha=0.65,
            label=f"blend={blend:.2f}",
        )
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.45)
    ax.set_xlabel("Base vs hybrid finite-value Spearman")
    ax.set_ylabel("Hybrid - base footprint mass captured by top 5% finite-value units")
    ax.set_title("Magnitude-aware value field shifts under hybrid footprint calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "hybrid_action_shift_vs_footprint_gain.png", dpi=180)
    plt.close(fig)

    if not main_city.empty:
        ordered = main_city.sort_values("mean_delta_finite_top5pct_units_footprint_mass")
        y = np.arange(len(ordered))
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.barh(y - 0.18, ordered["mean_base_finite_top5pct_units_footprint_mass"], height=0.36, label="OD-template", color="#94a3b8")
        ax.barh(y + 0.18, ordered["mean_hybrid_finite_top5pct_units_footprint_mass"], height=0.36, label="Hybrid", color="#2563eb")
        ax.set_yticks(y, ordered["city"])
        ax.set_xlabel("Observed footprint mass captured by top 5% finite-value units")
        ax.set_title(f"Observed-footprint alignment at hybrid blend {main_blend:.2f}")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "hybrid_footprint_capture_by_city.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.bar(main_city["city"], main_city["hybrid_finite_top5_value_share_range"], color="#2ca58d", label="Hybrid finite")
        ax.scatter(main_city["city"], main_city["base_finite_top5_value_share_range"], color="#111827", label="OD-template finite")
        ax.scatter(main_city["city"], main_city["hybrid_top5_value_share_range"], color="#b45309", marker="x", label="Hybrid small-signal")
        ax.set_ylabel("Within-city range of top-5% value share")
        ax.set_title("Finite-value top-tail variation changes more than small-signal variation")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "hybrid_top_tail_within_city_variation.png", dpi=180)
        plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], city_summary: pd.DataFrame) -> None:
    lines = [
        "# Hybrid Footprint Calibration Sensitivity V34",
        "",
        "本版把 V33 的 TMC-derived event-zone footprint 放入 calibration 敏感性测试：保持每个事件的城市级 `b0` 和 `h[t]` 总信号不变，只改变空间分配，从纯 OD vulnerability template 改为 `OD-template + observed TMC footprint` 的混合场。",
        "",
        "这里同时报告两种 action-value field：",
        "",
        "1. `small-signal`：当前 learning 主标签，只判断未来是否仍有正损失，因此对损失幅度不敏感。",
        "2. `finite/magnitude-aware`：使用 `finite_deficit_area_value`，把未来损失幅度也放入 action value。",
        "",
        "## 关键结论",
        "",
        f"- 主分析 blend = {metrics['main_blend']:.2f}，覆盖 {metrics['n_events']} 个事件、{metrics['n_cities']} 个城市。",
        f"- small-signal field 几乎完全不变：base vs hybrid Spearman = {fmt(metrics['mean_action_value_spearman'])}，top-5% action Jaccard = {fmt(metrics['mean_top5pct_action_jaccard'])}。",
        f"- magnitude-aware finite field 明显改变：base vs hybrid Spearman = {fmt(metrics['mean_finite_action_value_spearman'])}，top-5% action Jaccard = {fmt(metrics['mean_finite_top5pct_action_jaccard'])}。",
        f"- finite top-5% units 捕获的 observed footprint mass 从 {fmt(metrics['mean_base_finite_top5pct_units_footprint_mass'])} 变为 {fmt(metrics['mean_hybrid_finite_top5pct_units_footprint_mass'])}，平均变化 {fmt(metrics['mean_delta_finite_top5pct_units_footprint_mass'])}。",
        f"- finite top-20 units 捕获的 observed footprint mass 从 {fmt(metrics['mean_base_finite_top20_units_footprint_mass'])} 变为 {fmt(metrics['mean_hybrid_finite_top20_units_footprint_mass'])}。",
        f"- small-signal top-tail 零城市内变化的城市数为 {metrics['base_top5_zero_variance_city_count']} -> {metrics['hybrid_top5_zero_variance_city_count']}；finite field 为 {metrics['base_finite_top5_zero_variance_city_count']} -> {metrics['hybrid_finite_top5_zero_variance_city_count']}。",
        f"- hybrid/base no-intervention objective ratio 平均为 {fmt(metrics['mean_hybrid_to_base_baseline_objective_ratio'])}，说明这一步主要改变空间分布，而不是事件总强度。",
        "",
        "## 城市摘要",
        "",
        "| city | blend | small Spearman | finite Spearman | finite top5 Jaccard | base finite footprint mass | hybrid finite footprint mass | delta | finite base top5 range | finite hybrid top5 range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not city_summary.empty:
        main = city_summary[np.isclose(city_summary["footprint_blend"], metrics["main_blend"])].copy()
        for row in main.sort_values("mean_delta_finite_top5pct_units_footprint_mass", ascending=False).itertuples(index=False):
            lines.append(
                "| {city} | {blend:.2f} | {small_spearman} | {finite_spearman} | {finite_jaccard} | {base_mass} | {hybrid_mass} | {delta} | {base_range} | {hybrid_range} |".format(
                    city=row.city,
                    blend=float(row.footprint_blend),
                    small_spearman=fmt(row.mean_action_value_spearman),
                    finite_spearman=fmt(row.mean_finite_action_value_spearman),
                    finite_jaccard=fmt(row.mean_finite_top5pct_action_jaccard),
                    base_mass=fmt(row.mean_base_finite_top5pct_units_footprint_mass),
                    hybrid_mass=fmt(row.mean_hybrid_finite_top5pct_units_footprint_mass),
                    delta=fmt(row.mean_delta_finite_top5pct_units_footprint_mass),
                    base_range=fmt(row.base_finite_top5_value_share_range),
                    hybrid_range=fmt(row.hybrid_finite_top5_value_share_range),
                )
            )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "这个结果不是简单的“footprint 无效”。更准确地说：当前 small-signal law 的标签定义把所有仍有正损失的 zone 视为 active，因此只要 OD-template 与 hybrid 都让大多数 zone 保持正 deficit，事件 footprint 就不会改变该标签的排序。",
            "",
            "但 magnitude-aware finite field 会随 hybrid footprint 改变，说明 V33 发现的空间信号确实能进入 recoverability learning target。下一步需要在 hybrid calibration 下重新求解 full LP，并检查 finite/residual law 的变化是否会转化为最终优化选择，而不只是 first-order proxy 的变化。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return np.nan
    return float(pair["a"].corr(pair["b"], method=method))


def jaccard(a: set[Any], b: set[Any]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > EPS else np.nan


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= EPS:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - (n + 1.0) / n)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


if __name__ == "__main__":
    main()

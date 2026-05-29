"""Validate hybrid footprint calibration with representative full LP solves.

V34 showed that observed TMC-derived event footprints change the
magnitude-aware finite-value field, while leaving the current small-signal
target unchanged.  This script takes the next step: select representative
footprint-sensitive events, solve the full LP under hybrid spatial
calibration, and compare optimized action support against the base
OD-template LP support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_hybrid_footprint_calibration import (
    DEFAULT_MAIN_BLEND,
    build_hybrid_params,
    load_inputs,
    no_intervention_objective,
)
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_recovery_lp


EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--v34-dir", default="results/hybrid_footprint_calibration")
    parser.add_argument("--output-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--events-per-city", type=int, default=1)
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))
    time_limit = float(args.time_limit_seconds if args.time_limit_seconds is not None else solver.get("time_limit_seconds", 300))

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_events": table_dir / "hybrid_lp_selected_events.csv",
        "event_metrics": table_dir / "hybrid_lp_event_metrics.csv",
        "selected_actions": table_dir / "hybrid_lp_selected_actions.csv",
        "city_summary": table_dir / "hybrid_lp_city_summary.csv",
        "metrics": table_dir / "hybrid_lp_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    data = load_inputs(root)
    v34_events = pd.read_csv(root / args.v34_dir / "tables" / "hybrid_footprint_event_metrics.csv", parse_dates=["event_start"])
    selected_events = select_representative_events(
        v34_events,
        events_per_city=int(args.events_per_city),
        footprint_blend=float(args.footprint_blend),
        cities=args.cities,
    )
    write_table(selected_events, paths["selected_events"])

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamics = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    abnormal = data["abnormal"].copy()
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"] == "OPTIMAL") & (base_summary["scenario"] == "base")].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype(int)
    base_lookup = {(row.city, int(row.event_id)): row for row in base_summary.itertuples(index=False)}
    base_interventions = pd.read_csv(root / "results" / "event_optimization" / "tables" / "event_optimization_interventions.csv")
    base_interventions["event_id"] = pd.to_numeric(base_interventions["event_id"], errors="coerce").astype(int)
    footprint = data["footprint_zone"].copy()
    footprint["event_id"] = pd.to_numeric(footprint["event_id"], errors="coerce").astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }

    completed = completed_keys(paths["event_metrics"]) if args.resume else set()
    event_rows: list[dict[str, Any]] = []
    total_jobs = len(selected_events)
    for job_idx, selected in enumerate(selected_events.itertuples(index=False), start=1):
        city = str(selected.city)
        event_id = int(selected.event_id)
        if (city, event_id) in completed:
            print(f"[{job_idx}/{total_jobs}] Skipping completed hybrid LP {city} event {event_id}", flush=True)
            continue
        print(f"[{job_idx}/{total_jobs}] Solving hybrid LP {city} event {event_id}", flush=True)
        event_row = event_lookup[(city, event_id)]
        base_row = base_lookup[(city, event_id)]
        footprint_group = footprint_groups[(city, event_id)]
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamics[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        hybrid_params, diagnostics = build_hybrid_params(
            base_params,
            footprint_group,
            footprint_blend=float(args.footprint_blend),
            footprint_floor=float(args.footprint_floor),
            max_relative=float(args.max_relative),
        )
        hybrid_baseline = no_intervention_objective(hybrid_params)
        base_event_actions = positive_actions(
            base_interventions[
                (base_interventions["city"].astype(str) == city)
                & (base_interventions["event_id"] == event_id)
                & (base_interventions["scenario"].astype(str) == "base")
            ]
        )
        try:
            optimized = solve_recovery_lp(
                hybrid_params,
                output_flag=bool(solver.get("output_flag", False)),
                method=method,
                time_limit_seconds=time_limit,
            )
            hybrid_actions = positive_actions(optimized.interventions)
            row = build_event_metrics(
                selected,
                base_row,
                hybrid_params,
                hybrid_baseline,
                optimized_objective=float(optimized.objective),
                optimized_status=str(optimized.status),
                runtime_seconds=float(optimized.runtime_seconds),
                base_actions=base_event_actions,
                hybrid_actions=hybrid_actions,
                footprint_group=footprint_group,
                diagnostics=diagnostics,
                error="",
            )
            event_rows.append(row)
            append_csv(annotate_selected_actions(hybrid_actions, row), paths["selected_actions"])
        except Exception as exc:  # pragma: no cover - long LP diagnostics
            row = build_error_row(selected, base_row, hybrid_params, hybrid_baseline, diagnostics, str(exc))
            event_rows.append(row)

        append_csv(pd.DataFrame([event_rows[-1]]), paths["event_metrics"])

    all_events = pd.read_csv(paths["event_metrics"]) if paths["event_metrics"].exists() else pd.DataFrame(event_rows)
    if not all_events.empty and {"city", "event_id"}.issubset(all_events.columns):
        all_events = all_events.drop_duplicates(["city", "event_id"], keep="last").reset_index(drop=True)
        write_table(all_events, paths["event_metrics"])
    all_actions = pd.read_csv(paths["selected_actions"]) if paths["selected_actions"].exists() else pd.DataFrame()
    city_summary = build_city_summary(all_events)
    metrics = build_metrics(all_events, city_summary)
    write_table(city_summary, paths["city_summary"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(all_events, city_summary, figure_dir)
    write_report(report_dir / "hybrid_footprint_lp_validation_report_zh.md", selected_events, all_events, city_summary, metrics)
    print(f"Wrote hybrid footprint LP validation to {output_dir}")


def select_representative_events(
    v34_events: pd.DataFrame,
    *,
    events_per_city: int,
    footprint_blend: float,
    cities: list[str] | None,
) -> pd.DataFrame:
    frame = v34_events[np.isclose(v34_events["footprint_blend"], footprint_blend)].copy()
    if cities:
        frame = frame[frame["city"].isin(cities)].copy()
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype(int)
    frame = frame.sort_values(
        ["city", "delta_finite_top5pct_units_footprint_mass", "finite_top5pct_action_jaccard"],
        ascending=[True, False, True],
    )
    selected = frame.groupby("city", as_index=False).head(max(1, int(events_per_city))).copy()
    keep = [
        "city",
        "event_id",
        "event_start",
        "footprint_blend",
        "finite_action_value_spearman",
        "finite_top5pct_action_jaccard",
        "delta_finite_top5pct_units_footprint_mass",
        "base_finite_top5pct_units_footprint_mass",
        "hybrid_finite_top5pct_units_footprint_mass",
        "hybrid_to_base_baseline_objective_ratio",
    ]
    return selected[keep].sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)


def positive_actions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for col in ["u", "e", "effective_cost", "cost"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    mask = pd.Series(False, index=out.index)
    for col in ["u", "e", "effective_cost"]:
        if col in out:
            mask |= out[col].abs() > 1e-10
    out = out[mask].copy()
    if "unit" in out:
        out["unit"] = out["unit"].astype(str)
    if "t" in out:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").fillna(-1).astype(int)
    if "intervention" in out:
        out["intervention"] = out["intervention"].astype(str)
    return out


def build_event_metrics(
    selected: Any,
    base_row: Any,
    hybrid_params: Any,
    hybrid_baseline: float,
    *,
    optimized_objective: float,
    optimized_status: str,
    runtime_seconds: float,
    base_actions: pd.DataFrame,
    hybrid_actions: pd.DataFrame,
    footprint_group: pd.DataFrame,
    diagnostics: dict[str, float],
    error: str,
) -> dict[str, Any]:
    weights = footprint_weights(footprint_group)
    base_status = str(getattr(base_row, "status", ""))
    base_baseline = float(base_row.baseline_objective)
    base_objective = float(base_row.optimized_objective)
    base_recoverable = fraction_recovered(base_baseline, base_objective)
    hybrid_recoverable = fraction_recovered(hybrid_baseline, optimized_objective)
    base_unit_mass = selected_unit_mass(base_actions, weights)
    hybrid_unit_mass = selected_unit_mass(hybrid_actions, weights)
    base_cost_score = selected_cost_weighted_footprint_score(base_actions, weights)
    hybrid_cost_score = selected_cost_weighted_footprint_score(hybrid_actions, weights)
    return {
        "city": str(selected.city),
        "event_id": int(selected.event_id),
        "event_start": str(selected.event_start),
        "n_units": int(hybrid_params.n_units),
        "base_status": base_status,
        "hybrid_status": optimized_status,
        "runtime_seconds": runtime_seconds,
        "base_baseline_objective": base_baseline,
        "hybrid_baseline_objective": float(hybrid_baseline),
        "hybrid_to_base_baseline_objective_ratio": float(hybrid_baseline / max(base_baseline, EPS)),
        "base_optimized_objective": base_objective,
        "hybrid_optimized_objective": optimized_objective,
        "base_recoverable_fraction": base_recoverable,
        "hybrid_recoverable_fraction": hybrid_recoverable,
        "delta_recoverable_fraction": hybrid_recoverable - base_recoverable,
        "base_total_intervention_cost": float(base_actions["effective_cost"].sum()) if "effective_cost" in base_actions else np.nan,
        "hybrid_total_intervention_cost": float(hybrid_actions["effective_cost"].sum()) if "effective_cost" in hybrid_actions else np.nan,
        "base_selected_action_count": int(len(base_actions)),
        "hybrid_selected_action_count": int(len(hybrid_actions)),
        "selected_action_jaccard": action_jaccard(base_actions, hybrid_actions),
        "selected_unit_jaccard": unit_jaccard(base_actions, hybrid_actions),
        "base_selected_unit_footprint_mass": base_unit_mass,
        "hybrid_selected_unit_footprint_mass": hybrid_unit_mass,
        "delta_selected_unit_footprint_mass": hybrid_unit_mass - base_unit_mass,
        "base_selected_cost_weighted_footprint_score": base_cost_score,
        "hybrid_selected_cost_weighted_footprint_score": hybrid_cost_score,
        "delta_selected_cost_weighted_footprint_score": hybrid_cost_score - base_cost_score,
        "v34_finite_action_value_spearman": float(selected.finite_action_value_spearman),
        "v34_finite_top5pct_action_jaccard": float(selected.finite_top5pct_action_jaccard),
        "v34_delta_finite_top5pct_units_footprint_mass": float(selected.delta_finite_top5pct_units_footprint_mass),
        "v34_base_finite_top5pct_units_footprint_mass": float(selected.base_finite_top5pct_units_footprint_mass),
        "v34_hybrid_finite_top5pct_units_footprint_mass": float(selected.hybrid_finite_top5pct_units_footprint_mass),
        "footprint_zone_count": int(footprint_group["zone_id"].nunique()),
        "error": error,
        **diagnostics,
    }


def build_error_row(
    selected: Any,
    base_row: Any,
    hybrid_params: Any,
    hybrid_baseline: float,
    diagnostics: dict[str, float],
    error: str,
) -> dict[str, Any]:
    return {
        "city": str(selected.city),
        "event_id": int(selected.event_id),
        "event_start": str(selected.event_start),
        "n_units": int(hybrid_params.n_units),
        "base_status": str(getattr(base_row, "status", "")),
        "hybrid_status": "ERROR",
        "runtime_seconds": np.nan,
        "base_baseline_objective": float(base_row.baseline_objective),
        "hybrid_baseline_objective": float(hybrid_baseline),
        "hybrid_to_base_baseline_objective_ratio": float(hybrid_baseline / max(float(base_row.baseline_objective), EPS)),
        "base_optimized_objective": float(base_row.optimized_objective),
        "hybrid_optimized_objective": np.nan,
        "base_recoverable_fraction": fraction_recovered(float(base_row.baseline_objective), float(base_row.optimized_objective)),
        "hybrid_recoverable_fraction": np.nan,
        "delta_recoverable_fraction": np.nan,
        "base_total_intervention_cost": np.nan,
        "hybrid_total_intervention_cost": np.nan,
        "base_selected_action_count": np.nan,
        "hybrid_selected_action_count": np.nan,
        "selected_action_jaccard": np.nan,
        "selected_unit_jaccard": np.nan,
        "base_selected_unit_footprint_mass": np.nan,
        "hybrid_selected_unit_footprint_mass": np.nan,
        "delta_selected_unit_footprint_mass": np.nan,
        "base_selected_cost_weighted_footprint_score": np.nan,
        "hybrid_selected_cost_weighted_footprint_score": np.nan,
        "delta_selected_cost_weighted_footprint_score": np.nan,
        "v34_finite_action_value_spearman": float(selected.finite_action_value_spearman),
        "v34_finite_top5pct_action_jaccard": float(selected.finite_top5pct_action_jaccard),
        "v34_delta_finite_top5pct_units_footprint_mass": float(selected.delta_finite_top5pct_units_footprint_mass),
        "v34_base_finite_top5pct_units_footprint_mass": float(selected.base_finite_top5pct_units_footprint_mass),
        "v34_hybrid_finite_top5pct_units_footprint_mass": float(selected.hybrid_finite_top5pct_units_footprint_mass),
        "footprint_zone_count": np.nan,
        "error": error,
        **diagnostics,
    }


def annotate_selected_actions(actions: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    out = actions.copy()
    out["calibration"] = "hybrid_footprint"
    for key in ["city", "event_id", "event_start", "hybrid_status"]:
        out[key] = row[key]
    return out


def footprint_weights(footprint_group: pd.DataFrame) -> pd.Series:
    weights = footprint_group.groupby("zone_id")["zone_weight"].sum()
    weights.index = weights.index.astype(str)
    total = float(weights.sum())
    return weights / max(total, EPS)


def selected_unit_mass(actions: pd.DataFrame, weights: pd.Series) -> float:
    if actions.empty or "unit" not in actions:
        return 0.0
    units = pd.Index(actions["unit"].astype(str).unique())
    return float(weights.reindex(units, fill_value=0.0).sum())


def selected_cost_weighted_footprint_score(actions: pd.DataFrame, weights: pd.Series) -> float:
    if actions.empty or "unit" not in actions or "effective_cost" not in actions:
        return 0.0
    costs = pd.to_numeric(actions["effective_cost"], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(costs.sum())
    if total <= EPS:
        return 0.0
    unit_weights = actions["unit"].astype(str).map(weights).fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(costs.to_numpy(dtype=float) * unit_weights) / total)


def action_jaccard(a: pd.DataFrame, b: pd.DataFrame) -> float:
    set_a = action_set(a)
    set_b = action_set(b)
    union = set_a | set_b
    return float(len(set_a & set_b) / len(union)) if union else np.nan


def unit_jaccard(a: pd.DataFrame, b: pd.DataFrame) -> float:
    set_a = set(a["unit"].astype(str)) if not a.empty and "unit" in a else set()
    set_b = set(b["unit"].astype(str)) if not b.empty and "unit" in b else set()
    union = set_a | set_b
    return float(len(set_a & set_b) / len(union)) if union else np.nan


def action_set(actions: pd.DataFrame) -> set[tuple[str, int, str]]:
    if actions.empty:
        return set()
    return {
        (str(row.unit), int(row.t), str(row.intervention))
        for row in actions[["unit", "t", "intervention"]].itertuples(index=False)
    }


def fraction_recovered(baseline_objective: float, objective: float) -> float:
    return float(1.0 - objective / baseline_objective) if baseline_objective > EPS else np.nan


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    ok = event_metrics[event_metrics["hybrid_status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby("city", as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_base_selected_unit_footprint_mass=("base_selected_unit_footprint_mass", "mean"),
            mean_hybrid_selected_unit_footprint_mass=("hybrid_selected_unit_footprint_mass", "mean"),
            mean_delta_selected_unit_footprint_mass=("delta_selected_unit_footprint_mass", "mean"),
            mean_base_selected_cost_weighted_footprint_score=("base_selected_cost_weighted_footprint_score", "mean"),
            mean_hybrid_selected_cost_weighted_footprint_score=("hybrid_selected_cost_weighted_footprint_score", "mean"),
            mean_delta_selected_cost_weighted_footprint_score=("delta_selected_cost_weighted_footprint_score", "mean"),
            mean_selected_action_jaccard=("selected_action_jaccard", "mean"),
            mean_selected_unit_jaccard=("selected_unit_jaccard", "mean"),
            mean_delta_recoverable_fraction=("delta_recoverable_fraction", "mean"),
            mean_v34_delta_finite_top5pct_units_footprint_mass=("v34_delta_finite_top5pct_units_footprint_mass", "mean"),
        )
        .sort_values("mean_delta_selected_unit_footprint_mass", ascending=False)
    )


def build_metrics(event_metrics: pd.DataFrame, city_summary: pd.DataFrame) -> dict[str, Any]:
    ok = event_metrics[event_metrics["hybrid_status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy() if not event_metrics.empty else pd.DataFrame()
    optimal = event_metrics[event_metrics["hybrid_status"].astype(str) == "OPTIMAL"].copy() if not event_metrics.empty else pd.DataFrame()
    return {
        "n_selected_events": int(event_metrics["event_id"].nunique()) if not event_metrics.empty else 0,
        "n_successful_lp_events": int(ok["event_id"].nunique()) if not ok.empty else 0,
        "n_optimal_lp_events": int(optimal["event_id"].nunique()) if not optimal.empty else 0,
        "n_cities": int(ok["city"].nunique()) if not ok.empty else 0,
        "mean_delta_selected_unit_footprint_mass": safe_mean(ok, "delta_selected_unit_footprint_mass"),
        "mean_base_selected_unit_footprint_mass": safe_mean(ok, "base_selected_unit_footprint_mass"),
        "mean_hybrid_selected_unit_footprint_mass": safe_mean(ok, "hybrid_selected_unit_footprint_mass"),
        "mean_delta_selected_cost_weighted_footprint_score": safe_mean(ok, "delta_selected_cost_weighted_footprint_score"),
        "mean_base_selected_cost_weighted_footprint_score": safe_mean(ok, "base_selected_cost_weighted_footprint_score"),
        "mean_hybrid_selected_cost_weighted_footprint_score": safe_mean(ok, "hybrid_selected_cost_weighted_footprint_score"),
        "mean_selected_action_jaccard": safe_mean(ok, "selected_action_jaccard"),
        "mean_selected_unit_jaccard": safe_mean(ok, "selected_unit_jaccard"),
        "mean_delta_recoverable_fraction": safe_mean(ok, "delta_recoverable_fraction"),
        "mean_hybrid_to_base_baseline_objective_ratio": safe_mean(ok, "hybrid_to_base_baseline_objective_ratio"),
        "mean_v34_delta_finite_top5pct_units_footprint_mass": safe_mean(ok, "v34_delta_finite_top5pct_units_footprint_mass"),
        "selected_delta_vs_v34_delta_corr": safe_corr(
            ok.get("delta_selected_unit_footprint_mass", pd.Series(dtype=float)),
            ok.get("v34_delta_finite_top5pct_units_footprint_mass", pd.Series(dtype=float)),
        ),
        "largest_selected_footprint_gain_city": str(city_summary.iloc[0]["city"]) if not city_summary.empty else "",
    }


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return np.nan
    return float(pair["a"].corr(pair["b"], method="spearman"))


def make_figures(event_metrics: pd.DataFrame, city_summary: pd.DataFrame, figure_dir: Path) -> None:
    if event_metrics.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    ok = event_metrics[event_metrics["hybrid_status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    if ok.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(ok))
    labels = [f"{row.city}\n{int(row.event_id)}" for row in ok.itertuples(index=False)]
    ax.bar(x - 0.18, ok["base_selected_unit_footprint_mass"], width=0.36, label="Base LP", color="#94a3b8")
    ax.bar(x + 0.18, ok["hybrid_selected_unit_footprint_mass"], width=0.36, label="Hybrid LP", color="#2563eb")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Observed footprint mass covered by selected units")
    ax.set_title("Full LP selected support under hybrid footprint calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "hybrid_lp_selected_footprint_mass.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(ok["v34_delta_finite_top5pct_units_footprint_mass"], ok["delta_selected_unit_footprint_mass"], s=70, color="#2563eb")
    for row in ok.itertuples(index=False):
        ax.annotate(str(row.city), (row.v34_delta_finite_top5pct_units_footprint_mass, row.delta_selected_unit_footprint_mass), fontsize=8)
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.5)
    ax.set_xlabel("V34 finite top-5% footprint-mass gain")
    ax.set_ylabel("Hybrid LP selected-unit footprint-mass gain")
    ax.set_title("Do first-order finite shifts appear in full LP support?")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "hybrid_lp_shift_vs_first_order.png", dpi=180)
    plt.close(fig)

    if not city_summary.empty:
        ordered = city_summary.sort_values("mean_delta_selected_cost_weighted_footprint_score")
        y = np.arange(len(ordered))
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.barh(y - 0.18, ordered["mean_base_selected_cost_weighted_footprint_score"], height=0.36, label="Base LP", color="#94a3b8")
        ax.barh(y + 0.18, ordered["mean_hybrid_selected_cost_weighted_footprint_score"], height=0.36, label="Hybrid LP", color="#2ca58d")
        ax.set_yticks(y, ordered["city"])
        ax.set_xlabel("Cost-weighted selected footprint score")
        ax.set_title("Resource spending shifts toward observed footprint")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "hybrid_lp_cost_weighted_footprint_score.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    selected_events: pd.DataFrame,
    event_metrics: pd.DataFrame,
    city_summary: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Hybrid Footprint Full-LP Validation V35",
        "",
        "本版从 V34 的 footprint-sensitive events 中每个城市选择 finite footprint gain 最大的代表事件，并在 hybrid OD-template + TMC-footprint calibration 下重新求解完整 LP。",
        "",
        "## 关键结论",
        "",
        f"- 代表性事件数：{metrics['n_selected_events']}；成功返回可行 LP 解：{metrics['n_successful_lp_events']}；其中 OPTIMAL：{metrics['n_optimal_lp_events']}。",
        f"- base LP selected units 捕获的 observed footprint mass 平均为 {fmt(metrics['mean_base_selected_unit_footprint_mass'])}，hybrid LP 为 {fmt(metrics['mean_hybrid_selected_unit_footprint_mass'])}，变化 {fmt(metrics['mean_delta_selected_unit_footprint_mass'])}。",
        f"- cost-weighted selected footprint score 从 {fmt(metrics['mean_base_selected_cost_weighted_footprint_score'])} 到 {fmt(metrics['mean_hybrid_selected_cost_weighted_footprint_score'])}，变化 {fmt(metrics['mean_delta_selected_cost_weighted_footprint_score'])}。",
        f"- base vs hybrid selected-action Jaccard 平均为 {fmt(metrics['mean_selected_action_jaccard'])}，selected-unit Jaccard 平均为 {fmt(metrics['mean_selected_unit_jaccard'])}。",
        f"- hybrid/base no-intervention objective ratio 平均为 {fmt(metrics['mean_hybrid_to_base_baseline_objective_ratio'])}；recoverable fraction 平均变化为 {fmt(metrics['mean_delta_recoverable_fraction'])}。",
        f"- 对照 V34，代表事件的 finite top-5% footprint-mass gain 平均为 {fmt(metrics['mean_v34_delta_finite_top5pct_units_footprint_mass'])}，但 full LP selected-unit gain 只有 {fmt(metrics['mean_delta_selected_unit_footprint_mass'])}。",
        "",
        "## Selected Events",
        "",
        table_to_markdown(selected_events),
        "",
        "## Event Metrics",
        "",
        table_to_markdown(event_metrics),
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary),
        "",
        "## 解释",
        "",
        "这版结果显示：V34 的 magnitude-aware first-order footprint shift 真实存在，但只有很小一部分转化成完整 LP 的 selected support。full LP 在代表事件中仍然高度接近 base OD-template support，说明预算约束、deployment caps、response delay、三类资源的替代关系和 diminishing returns 会重新吸收大部分 footprint signal。",
        "",
        "因此目前可以写成一个重要边界：event-specific footprint 会改变 magnitude-aware recoverability field，但在当前管理参数和小段资源设定下，还不能直接推出最终优化投放会大幅转向 observed footprint。后续若要把 footprint-specific recovery law 作为主结论，需要进一步做全量 hybrid LP、残差 law closure，或让 b0/h 的 region-level footprint 与资源上限、成本、响应机制一起重新标定。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    out = df.head(max_rows).copy()
    cols = out.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(format_cell(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float) or isinstance(value, np.floating):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.4f}"
    text = str(value)
    return text.replace("|", "/")


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "hybrid_status"}.issubset(existing.columns):
        return set()
    finished = existing[existing["hybrid_status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    return {(str(row.city), int(row.event_id)) for row in finished[["city", "event_id"]].itertuples(index=False)}


if __name__ == "__main__":
    main()

"""Validate observed-footprint preferences as explicit LP objectives.

V40 scanned footprint weights in an adaptive residual replay policy.  This
script asks the stricter V41 question: what changes when observed event
footprint coverage is put directly into the LP objective as a secondary
linear deployment reward?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_hybrid_absorption_mechanisms import footprint_weights
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters, solve_recovery_lp


EPS = 1e-12
DEFAULT_LAMBDAS = (0.0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.50)
LOSS_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
SUCCESS_STATUSES = {"OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/multiobjective_footprint_lp_validation")
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
    parser.add_argument("--lambda-grid", nargs="*", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))
    time_limit = float(args.time_limit_seconds if args.time_limit_seconds is not None else solver.get("time_limit_seconds", 300))

    lambdas = sorted({float(value) for value in args.lambda_grid if np.isfinite(value) and value >= 0})
    if 0.0 not in lambdas:
        lambdas.insert(0, 0.0)

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "event_metrics": table_dir / "multiobjective_lp_event_metrics.csv",
        "selected_actions": table_dir / "multiobjective_lp_selected_actions.csv.gz",
        "summary": table_dir / "multiobjective_lp_summary.csv",
        "frontier": table_dir / "multiobjective_lp_frontier.csv",
        "event_best": table_dir / "multiobjective_lp_event_best.csv",
        "metrics": table_dir / "multiobjective_lp_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    event_metrics, selected_actions = run_validation(
        root,
        config,
        hybrid_lp_dir=root / args.hybrid_lp_dir,
        lambdas=lambdas,
        footprint_blend=float(args.footprint_blend),
        footprint_floor=float(args.footprint_floor),
        max_relative=float(args.max_relative),
        method=method,
        time_limit=time_limit,
        event_metric_path=paths["event_metrics"],
        selected_action_path=paths["selected_actions"],
        resume=bool(args.resume),
    )
    event_metrics = add_lambda0_deltas(event_metrics)
    write_table(event_metrics, paths["event_metrics"])
    summary = build_summary(event_metrics, lambdas)
    frontier = build_pareto_frontier(summary)
    event_best = build_event_best(event_metrics)
    metrics = build_metrics(summary, frontier, event_best)

    write_table(summary, paths["summary"])
    write_table(frontier, paths["frontier"])
    write_table(event_best, paths["event_best"])
    if not selected_actions.empty:
        write_table(selected_actions, paths["selected_actions"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(summary, frontier, event_best, event_metrics, figure_dir)
    write_report(report_dir / "multiobjective_footprint_lp_validation_report_zh.md", metrics, summary, frontier, event_best)
    print(f"Wrote explicit multi-objective footprint LP validation to {output_dir}")


def run_validation(
    root: Path,
    config: dict[str, Any],
    *,
    hybrid_lp_dir: Path,
    lambdas: list[float],
    footprint_blend: float,
    footprint_floor: float,
    max_relative: float,
    method: int,
    time_limit: float,
    event_metric_path: Path,
    selected_action_path: Path,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_inputs(root)
    lp_dir = hybrid_lp_dir / "tables"
    hybrid_events = pd.read_csv(lp_dir / "hybrid_lp_event_metrics.csv", parse_dates=["event_start"])
    selected_events = pd.read_csv(lp_dir / "hybrid_lp_selected_events.csv", parse_dates=["event_start"])
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"].eq("OPTIMAL")) & (base_summary["scenario"].eq("base"))].copy()

    for frame in [hybrid_events, selected_events, base_summary, data["events"], data["footprint_zone"]]:
        if "event_id" in frame:
            frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")

    event_lookup = {
        (row.city, int(row.event_id)): row
        for row in data["events"].dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    selected_lookup = {
        (row.city, int(row.event_id)): row
        for row in selected_events.dropna(subset=["event_id"]).itertuples(index=False)
    }
    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = footprint["event_id"].astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }

    ok_events = hybrid_events[
        hybrid_events["hybrid_status"].astype(str).eq("OPTIMAL")
        & hybrid_events["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)
    completed = completed_keys(event_metric_path) if resume else set()
    event_rows: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    total_jobs = len(ok_events) * len(lambdas)

    job_idx = 0
    for event_idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        event_key = (city, event_id)
        print(f"[event {event_idx}/{len(ok_events)}] Preparing multi-objective LP for {city} event {event_id}", flush=True)
        event_row = event_lookup[event_key]
        footprint_group = footprint_groups[event_key]
        selected_row = selected_lookup[event_key]
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=data["abnormal"],
            root=root,
        )
        hybrid_params, diagnostics = build_hybrid_params(
            base_params,
            footprint_group,
            footprint_blend=footprint_blend,
            footprint_floor=footprint_floor,
            max_relative=max_relative,
        )
        baseline_objective = no_intervention_objective(hybrid_params)
        reference_objective = float(row.hybrid_optimized_objective)
        reference_gain = baseline_objective - reference_objective
        weights = footprint_weights(footprint_group, hybrid_params.units)
        reward_score = footprint_reward_score(weights)

        for lambda_footprint in lambdas:
            job_idx += 1
            if (city, event_id, float(lambda_footprint)) in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed lambda={lambda_footprint:g} for {city} event {event_id}", flush=True)
                continue
            print(f"[{job_idx}/{total_jobs}] Solving lambda={lambda_footprint:g} for {city} event {event_id}", flush=True)
            coeff = build_reward_coefficients(
                hybrid_params,
                reward_score,
                baseline_objective=baseline_objective,
                lambda_footprint=float(lambda_footprint),
            )
            try:
                solved = solve_recovery_lp(
                    hybrid_params,
                    output_flag=False,
                    method=method,
                    time_limit_seconds=time_limit,
                    linear_u_reward=coeff,
                )
                true_objective = true_loss_objective(solved.trajectory, hybrid_params)
                actions = positive_actions(solved.interventions)
                row_out = build_event_row(
                    city,
                    event_id,
                    row,
                    selected_row,
                    hybrid_params,
                    lambda_footprint=float(lambda_footprint),
                    baseline_objective=baseline_objective,
                    reference_objective=reference_objective,
                    reference_gain=reference_gain,
                    true_objective=true_objective,
                    modified_objective=float(solved.objective),
                    status=str(solved.status),
                    runtime_seconds=float(solved.runtime_seconds),
                    actions=actions,
                    weights=weights,
                    reward_score=reward_score,
                    diagnostics=diagnostics,
                    error="",
                )
                event_rows.append(row_out)
                append_csv(pd.DataFrame([row_out]), event_metric_path)
                annotated = annotate_actions(actions, row_out, weights, reward_score)
                if not annotated.empty:
                    action_frames.append(annotated)
                    append_csv(annotated, selected_action_path)
            except Exception as exc:  # pragma: no cover - long LP diagnostics
                row_out = build_error_row(
                    city,
                    event_id,
                    row,
                    selected_row,
                    hybrid_params,
                    lambda_footprint=float(lambda_footprint),
                    baseline_objective=baseline_objective,
                    reference_objective=reference_objective,
                    reference_gain=reference_gain,
                    diagnostics=diagnostics,
                    error=str(exc),
                )
                event_rows.append(row_out)
                append_csv(pd.DataFrame([row_out]), event_metric_path)

    all_events = pd.read_csv(event_metric_path, parse_dates=["event_start"]) if event_metric_path.exists() else pd.DataFrame(event_rows)
    if not all_events.empty:
        all_events["event_id"] = pd.to_numeric(all_events["event_id"], errors="coerce").astype(int)
        all_events["lambda_footprint"] = pd.to_numeric(all_events["lambda_footprint"], errors="coerce")
        all_events = all_events.drop_duplicates(["city", "event_id", "lambda_footprint"], keep="last").reset_index(drop=True)
    all_actions = pd.read_csv(selected_action_path) if selected_action_path.exists() else pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    return all_events, all_actions


def build_reward_coefficients(
    params: RecoveryLPParameters,
    reward_score: pd.Series,
    *,
    baseline_objective: float,
    lambda_footprint: float,
) -> dict[str, np.ndarray]:
    score = reward_score.reindex(pd.Index(params.units, dtype=str), fill_value=0.0).to_numpy(dtype=float)
    total_budget = max(float(params.total_budget), EPS)
    scale = float(lambda_footprint) * float(baseline_objective) / total_budget
    return {
        key: scale * score[:, None] * np.asarray(params.cost[key], dtype=float)
        for key in INTERVENTIONS
    }


def footprint_reward_score(weights: pd.Series) -> pd.Series:
    values = weights.astype(float).clip(lower=0.0)
    max_value = float(values.max())
    if max_value <= EPS:
        return pd.Series(0.0, index=values.index)
    return values / max_value


def true_loss_objective(trajectory: pd.DataFrame, params: RecoveryLPParameters) -> float:
    return float(params.delta_t * pd.to_numeric(trajectory["weighted_loss"], errors="coerce").fillna(0.0).sum())


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


def build_event_row(
    city: str,
    event_id: int,
    v35_row: Any,
    selected_row: Any,
    params: RecoveryLPParameters,
    *,
    lambda_footprint: float,
    baseline_objective: float,
    reference_objective: float,
    reference_gain: float,
    true_objective: float,
    modified_objective: float,
    status: str,
    runtime_seconds: float,
    actions: pd.DataFrame,
    weights: pd.Series,
    reward_score: pd.Series,
    diagnostics: dict[str, float],
    error: str,
) -> dict[str, Any]:
    gain = baseline_objective - true_objective
    recoverable = gain / max(baseline_objective, EPS)
    fraction_of_reference = gain / max(reference_gain, EPS)
    footprint = footprint_metrics(actions, weights, reward_score, params.n_units, params.total_budget)
    return {
        "city": city,
        "event_id": int(event_id),
        "event_start": str(getattr(v35_row, "event_start")),
        "lambda_footprint": float(lambda_footprint),
        "n_units": int(params.n_units),
        "status": status,
        "runtime_seconds": runtime_seconds,
        "baseline_objective": float(baseline_objective),
        "reference_hybrid_objective": float(reference_objective),
        "reference_hybrid_gain": float(reference_gain),
        "true_objective": float(true_objective),
        "modified_objective": float(modified_objective),
        "recoverable_fraction": float(recoverable),
        "fraction_of_reference_lp_gain": float(fraction_of_reference),
        "objective_delta_vs_reference": float(true_objective - reference_objective),
        "total_intervention_cost": float(actions["effective_cost"].sum()) if "effective_cost" in actions else 0.0,
        "selected_action_count": int(len(actions)),
        "selected_unit_count": int(actions["unit"].nunique()) if "unit" in actions else 0,
        "v34_delta_finite_top5pct_units_footprint_mass": safe_float(
            getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)
        ),
        "v35_delta_selected_unit_footprint_mass": safe_float(getattr(v35_row, "delta_selected_unit_footprint_mass", np.nan)),
        "error": error,
        **footprint,
        **diagnostics,
    }


def build_error_row(
    city: str,
    event_id: int,
    v35_row: Any,
    selected_row: Any,
    params: RecoveryLPParameters,
    *,
    lambda_footprint: float,
    baseline_objective: float,
    reference_objective: float,
    reference_gain: float,
    diagnostics: dict[str, float],
    error: str,
) -> dict[str, Any]:
    return {
        "city": city,
        "event_id": int(event_id),
        "event_start": str(getattr(v35_row, "event_start")),
        "lambda_footprint": float(lambda_footprint),
        "n_units": int(params.n_units),
        "status": "ERROR",
        "runtime_seconds": np.nan,
        "baseline_objective": float(baseline_objective),
        "reference_hybrid_objective": float(reference_objective),
        "reference_hybrid_gain": float(reference_gain),
        "true_objective": np.nan,
        "modified_objective": np.nan,
        "recoverable_fraction": np.nan,
        "fraction_of_reference_lp_gain": np.nan,
        "objective_delta_vs_reference": np.nan,
        "total_intervention_cost": np.nan,
        "selected_action_count": np.nan,
        "selected_unit_count": np.nan,
        "v34_delta_finite_top5pct_units_footprint_mass": safe_float(
            getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)
        ),
        "v35_delta_selected_unit_footprint_mass": safe_float(getattr(v35_row, "delta_selected_unit_footprint_mass", np.nan)),
        "error": error,
        **diagnostics,
    }


def footprint_metrics(
    actions: pd.DataFrame,
    weights: pd.Series,
    reward_score: pd.Series,
    n_units: int,
    total_budget: float,
) -> dict[str, float]:
    if actions.empty or "unit" not in actions:
        return {
            "selected_unit_footprint_mass": 0.0,
            "cost_weighted_footprint_mass": 0.0,
            "cost_weighted_footprint_reward_score": 0.0,
            "top5_allocated_unit_footprint_mass": 0.0,
            "top10_allocated_unit_footprint_mass": 0.0,
            "footprint_reward_budget_share": 0.0,
        }
    costs = pd.to_numeric(actions["effective_cost"], errors="coerce").fillna(0.0).clip(lower=0.0)
    total_cost = float(costs.sum())
    unit = actions["unit"].astype(str)
    unit_weights = unit.map(weights).fillna(0.0).to_numpy(dtype=float)
    unit_reward = unit.map(reward_score).fillna(0.0).to_numpy(dtype=float)
    cost_by_unit = actions.assign(_cost=costs).groupby("unit")["_cost"].sum().sort_values(ascending=False)
    top5_n = max(1, int(math.ceil(0.05 * n_units)))
    top10_n = max(1, int(math.ceil(0.10 * n_units)))
    selected_units = pd.Index(unit.unique())
    return {
        "selected_unit_footprint_mass": float(weights.reindex(selected_units, fill_value=0.0).sum()),
        "cost_weighted_footprint_mass": float(np.sum(costs.to_numpy(dtype=float) * unit_weights) / max(total_cost, EPS)),
        "cost_weighted_footprint_reward_score": float(np.sum(costs.to_numpy(dtype=float) * unit_reward) / max(total_cost, EPS)),
        "top5_allocated_unit_footprint_mass": float(weights.reindex(cost_by_unit.head(top5_n).index.astype(str), fill_value=0.0).sum()),
        "top10_allocated_unit_footprint_mass": float(weights.reindex(cost_by_unit.head(top10_n).index.astype(str), fill_value=0.0).sum()),
        "footprint_reward_budget_share": float(np.sum(costs.to_numpy(dtype=float) * unit_reward) / max(float(total_budget), EPS)),
    }


def annotate_actions(
    actions: pd.DataFrame,
    row: dict[str, Any],
    weights: pd.Series,
    reward_score: pd.Series,
) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    out = actions.copy()
    for key in ["city", "event_id", "event_start", "lambda_footprint", "status"]:
        out[key] = row[key]
    out["unit_footprint_weight"] = out["unit"].astype(str).map(weights).fillna(0.0)
    out["unit_footprint_reward_score"] = out["unit"].astype(str).map(reward_score).fillna(0.0)
    return out


def add_lambda0_deltas(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return event_metrics
    out = event_metrics.copy()
    derived_cols = [
        col
        for col in out.columns
        if col.startswith("lambda0_")
        or col
        in {
            "delta_fraction_vs_lambda0",
            "delta_top5_allocated_mass_vs_lambda0",
            "delta_selected_mass_vs_lambda0",
            "delta_cost_weighted_mass_vs_lambda0",
        }
    ]
    if derived_cols:
        out = out.drop(columns=derived_cols)
    ok = out[out["status"].astype(str).isin(SUCCESS_STATUSES)].copy()
    lambda0 = ok[np.isclose(ok["lambda_footprint"], 0.0)].copy()
    keep = [
        "city",
        "event_id",
        "fraction_of_reference_lp_gain",
        "top5_allocated_unit_footprint_mass",
        "selected_unit_footprint_mass",
        "cost_weighted_footprint_mass",
        "cost_weighted_footprint_reward_score",
    ]
    lambda0 = lambda0[keep].rename(
        columns={
            "fraction_of_reference_lp_gain": "lambda0_fraction_of_reference_lp_gain",
            "top5_allocated_unit_footprint_mass": "lambda0_top5_allocated_unit_footprint_mass",
            "selected_unit_footprint_mass": "lambda0_selected_unit_footprint_mass",
            "cost_weighted_footprint_mass": "lambda0_cost_weighted_footprint_mass",
            "cost_weighted_footprint_reward_score": "lambda0_cost_weighted_footprint_reward_score",
        }
    )
    out = out.merge(lambda0, on=["city", "event_id"], how="left")
    out["delta_fraction_vs_lambda0"] = out["fraction_of_reference_lp_gain"] - out["lambda0_fraction_of_reference_lp_gain"]
    out["delta_top5_allocated_mass_vs_lambda0"] = (
        out["top5_allocated_unit_footprint_mass"] - out["lambda0_top5_allocated_unit_footprint_mass"]
    )
    out["delta_selected_mass_vs_lambda0"] = out["selected_unit_footprint_mass"] - out["lambda0_selected_unit_footprint_mass"]
    out["delta_cost_weighted_mass_vs_lambda0"] = out["cost_weighted_footprint_mass"] - out["lambda0_cost_weighted_footprint_mass"]
    return out


def build_summary(event_metrics: pd.DataFrame, lambdas: list[float]) -> pd.DataFrame:
    ok = event_metrics[event_metrics["status"].astype(str).isin(SUCCESS_STATUSES)].copy() if not event_metrics.empty else pd.DataFrame()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        group = ok[np.isclose(ok["lambda_footprint"], lambda_value)].copy()
        rows.append(
            {
                "lambda_footprint": float(lambda_value),
                "n_events": count_city_events(group),
                "mean_fraction_of_reference_lp_gain": safe_mean(group, "fraction_of_reference_lp_gain"),
                "mean_delta_fraction_vs_lambda0": safe_mean(group, "delta_fraction_vs_lambda0"),
                "mean_recoverable_fraction": safe_mean(group, "recoverable_fraction"),
                "mean_top5_allocated_unit_footprint_mass": safe_mean(group, "top5_allocated_unit_footprint_mass"),
                "mean_delta_top5_allocated_mass_vs_lambda0": safe_mean(group, "delta_top5_allocated_mass_vs_lambda0"),
                "mean_selected_unit_footprint_mass": safe_mean(group, "selected_unit_footprint_mass"),
                "mean_cost_weighted_footprint_mass": safe_mean(group, "cost_weighted_footprint_mass"),
                "mean_cost_weighted_footprint_reward_score": safe_mean(group, "cost_weighted_footprint_reward_score"),
                "mean_footprint_reward_budget_share": safe_mean(group, "footprint_reward_budget_share"),
                "mean_runtime_seconds": safe_mean(group, "runtime_seconds"),
                "n_optimal": int(group["status"].astype(str).eq("OPTIMAL").sum()) if not group.empty else 0,
                "n_time_limit": int(group["status"].astype(str).eq("TIME_LIMIT").sum()) if not group.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def count_city_events(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if {"city", "event_id"}.issubset(frame.columns):
        return int(frame[["city", "event_id"]].drop_duplicates().shape[0])
    return int(frame["event_id"].nunique()) if "event_id" in frame.columns else int(len(frame))


def build_pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    frame = summary.dropna(subset=["mean_fraction_of_reference_lp_gain", "mean_top5_allocated_unit_footprint_mass"]).copy()
    rows = []
    for idx, row in frame.iterrows():
        dominated = frame[
            (frame["mean_fraction_of_reference_lp_gain"] >= row["mean_fraction_of_reference_lp_gain"] - 1e-12)
            & (frame["mean_top5_allocated_unit_footprint_mass"] >= row["mean_top5_allocated_unit_footprint_mass"] - 1e-12)
            & (
                (frame["mean_fraction_of_reference_lp_gain"] > row["mean_fraction_of_reference_lp_gain"] + 1e-12)
                | (frame["mean_top5_allocated_unit_footprint_mass"] > row["mean_top5_allocated_unit_footprint_mass"] + 1e-12)
            )
        ]
        if dominated.empty:
            rows.append(idx)
    return frame.loc[rows].sort_values(["mean_top5_allocated_unit_footprint_mass", "mean_fraction_of_reference_lp_gain"]).reset_index(drop=True)


def build_event_best(event_metrics: pd.DataFrame) -> pd.DataFrame:
    ok = event_metrics[event_metrics["status"].astype(str).isin(SUCCESS_STATUSES)].copy() if not event_metrics.empty else pd.DataFrame()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in ok.groupby(["city", "event_id"], sort=True):
        lambda0 = group[np.isclose(group["lambda_footprint"], 0.0)]
        base_fraction = safe_float(lambda0["fraction_of_reference_lp_gain"].iloc[0]) if not lambda0.empty else np.nan
        base_top5 = safe_float(lambda0["top5_allocated_unit_footprint_mass"].iloc[0]) if not lambda0.empty else np.nan
        for threshold in LOSS_THRESHOLDS:
            candidates = group[group["delta_fraction_vs_lambda0"] >= -threshold].copy()
            if candidates.empty:
                continue
            best = candidates.sort_values(["top5_allocated_unit_footprint_mass", "fraction_of_reference_lp_gain"], ascending=[False, False]).iloc[0]
            rows.append(
                {
                    "city": city,
                    "event_id": int(event_id),
                    "loss_threshold": float(threshold),
                    "lambda_footprint": float(best["lambda_footprint"]),
                    "fraction_of_reference_lp_gain": safe_float(best["fraction_of_reference_lp_gain"]),
                    "delta_fraction_vs_lambda0": safe_float(best["delta_fraction_vs_lambda0"]),
                    "top5_allocated_unit_footprint_mass": safe_float(best["top5_allocated_unit_footprint_mass"]),
                    "delta_top5_allocated_mass_vs_lambda0": safe_float(best["top5_allocated_unit_footprint_mass"] - base_top5),
                    "lambda0_fraction_of_reference_lp_gain": base_fraction,
                    "lambda0_top5_allocated_unit_footprint_mass": base_top5,
                }
            )
    return pd.DataFrame(rows)


def build_metrics(summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_lambdas": int(summary["lambda_footprint"].nunique()) if not summary.empty else 0,
        "n_events": int(summary["n_events"].max()) if not summary.empty else 0,
        "pareto_frontier_points": int(len(frontier)) if not frontier.empty else 0,
    }
    if summary.empty:
        return metrics
    lambda0 = summary[np.isclose(summary["lambda_footprint"], 0.0)]
    if not lambda0.empty:
        base = lambda0.iloc[0]
        metrics.update(
            {
                "lambda0_fraction": safe_float(base["mean_fraction_of_reference_lp_gain"]),
                "lambda0_top5_footprint_mass": safe_float(base["mean_top5_allocated_unit_footprint_mass"]),
                "lambda0_cost_weighted_footprint_mass": safe_float(base["mean_cost_weighted_footprint_mass"]),
            }
        )
    gain_best = summary.sort_values("mean_fraction_of_reference_lp_gain", ascending=False).iloc[0]
    footprint_best = summary.sort_values("mean_top5_allocated_unit_footprint_mass", ascending=False).iloc[0]
    metrics.update(
        {
            "max_gain_lambda": safe_float(gain_best["lambda_footprint"]),
            "max_gain_fraction": safe_float(gain_best["mean_fraction_of_reference_lp_gain"]),
            "max_gain_top5_footprint_mass": safe_float(gain_best["mean_top5_allocated_unit_footprint_mass"]),
            "max_footprint_lambda": safe_float(footprint_best["lambda_footprint"]),
            "max_footprint_fraction": safe_float(footprint_best["mean_fraction_of_reference_lp_gain"]),
            "max_footprint_top5_mass": safe_float(footprint_best["mean_top5_allocated_unit_footprint_mass"]),
        }
    )
    lambda0_fraction = metrics.get("lambda0_fraction", np.nan)
    lambda0_top5 = metrics.get("lambda0_top5_footprint_mass", np.nan)
    for threshold in LOSS_THRESHOLDS:
        candidates = summary[summary["mean_delta_fraction_vs_lambda0"] >= -threshold].copy()
        if candidates.empty:
            continue
        best = candidates.sort_values(["mean_top5_allocated_unit_footprint_mass", "mean_fraction_of_reference_lp_gain"], ascending=[False, False]).iloc[0]
        suffix = threshold_suffix(threshold)
        best_events = event_best[np.isclose(event_best["loss_threshold"], threshold)].copy() if not event_best.empty else pd.DataFrame()
        metrics.update(
            {
                f"best_lambda_loss_le_{suffix}": safe_float(best["lambda_footprint"]),
                f"best_fraction_loss_le_{suffix}": safe_float(best["mean_fraction_of_reference_lp_gain"]),
                f"best_delta_fraction_loss_le_{suffix}": safe_float(best["mean_fraction_of_reference_lp_gain"] - lambda0_fraction),
                f"best_top5_footprint_loss_le_{suffix}": safe_float(best["mean_top5_allocated_unit_footprint_mass"]),
                f"best_delta_top5_footprint_loss_le_{suffix}": safe_float(best["mean_top5_allocated_unit_footprint_mass"] - lambda0_top5),
                f"event_mean_delta_top5_loss_le_{suffix}": safe_mean(best_events, "delta_top5_allocated_mass_vs_lambda0"),
                f"event_positive_delta_share_loss_le_{suffix}": safe_mean(
                    best_events.assign(positive_delta=best_events["delta_top5_allocated_mass_vs_lambda0"] > 0)
                    if not best_events.empty
                    else best_events,
                    "positive_delta",
                ),
            }
        )
    return metrics


def make_figures(
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    event_best: pd.DataFrame,
    event_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(
        summary["mean_top5_allocated_unit_footprint_mass"],
        summary["mean_fraction_of_reference_lp_gain"],
        color="#2563eb",
        marker="o",
        linewidth=1.7,
        label="lambda path",
    )
    if not frontier.empty:
        ax.scatter(
            frontier["mean_top5_allocated_unit_footprint_mass"],
            frontier["mean_fraction_of_reference_lp_gain"],
            s=90,
            color="#dc2626",
            label="Pareto frontier",
            zorder=3,
        )
    for row in summary.itertuples(index=False):
        ax.annotate(f"{row.lambda_footprint:g}", (row.mean_top5_allocated_unit_footprint_mass, row.mean_fraction_of_reference_lp_gain), fontsize=8)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("Top 5% allocated-unit observed-footprint mass")
    ax.set_ylabel("Mean fraction of hybrid LP gain")
    ax.set_title("Explicit LP recovery-footprint frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "multiobjective_lp_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(summary))
    labels = [f"{value:g}" for value in summary["lambda_footprint"]]
    ax1.plot(x, -summary["mean_delta_fraction_vs_lambda0"], color="#dc2626", marker="o", label="gain loss vs lambda=0")
    ax1.set_ylabel("Mean LP-gain fraction loss")
    ax1.set_xticks(x, labels, rotation=25)
    ax1.set_xlabel("LP footprint objective weight")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["mean_delta_top5_allocated_mass_vs_lambda0"], color="#2563eb", marker="s", label="footprint gain")
    ax2.set_ylabel("Top 5% footprint-mass gain")
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, frameon=False, loc="best")
    ax1.set_title("LP objective trade-off path")
    fig.tight_layout()
    fig.savefig(figure_dir / "multiobjective_lp_lambda_path.png", dpi=180)
    plt.close(fig)

    ok = event_metrics[event_metrics["status"].astype(str).isin(SUCCESS_STATUSES)].copy()
    if not ok.empty:
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        for (city, event_id), group in ok.groupby(["city", "event_id"], sort=True):
            group = group.sort_values("lambda_footprint")
            ax.plot(
                group["top5_allocated_unit_footprint_mass"],
                group["fraction_of_reference_lp_gain"],
                marker="o",
                linewidth=1.1,
                alpha=0.78,
                label=f"{city} {int(event_id)}",
            )
        ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.45)
        ax.set_xlabel("Top 5% allocated-unit observed-footprint mass")
        ax.set_ylabel("Fraction of hybrid LP gain")
        ax.set_title("Event-level explicit LP trade-off curves")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncols=2)
        fig.tight_layout()
        fig.savefig(figure_dir / "multiobjective_lp_event_curves.png", dpi=180)
        plt.close(fig)

    if not event_best.empty:
        near = event_best[np.isclose(event_best["loss_threshold"], 0.005)].copy()
        if not near.empty:
            labels = [f"{row.city}\n{int(row.event_id)}" for row in near.itertuples(index=False)]
            x = np.arange(len(near))
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            ax.bar(x, near["delta_top5_allocated_mass_vs_lambda0"], color="#2563eb", alpha=0.86)
            ax.axhline(0, color="#111827", linewidth=1, alpha=0.5)
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.set_ylabel("Top 5% footprint-mass gain vs lambda=0")
            ax.set_title("Best explicit LP footprint gain within 0.005 gain-fraction loss")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figure_dir / "multiobjective_lp_near_loss_event_gain.png", dpi=180)
            plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame) -> None:
    lines = [
        "# Explicit Multi-Objective Footprint LP Validation V41",
        "",
        "本版本把 observed event footprint 直接放入 LP 目标函数，而不再只在 residual replay policy 里改变排序。每个 lambda 解一个完整 LP：",
        "",
        "`min true_loss - lambda * baseline_loss * sum(cost_ikt * footprint_score_i * u_ikt) / total_budget`",
        "",
        "其中 `true_loss` 仍然是论文主目标；第二项表示研究者愿意用多少主目标损失，换取资源投向 observed footprint 更强的区域。报告中的 gain fraction 始终用原始 true loss 重新计算，不使用带 reward 的修改目标。",
        "",
        "## 关键结论",
        "",
        f"- 代表性 hybrid LP 事件数：{metrics.get('n_events', 0)}；lambda 数：{metrics.get('n_lambdas', 0)}；Pareto frontier 点数：{metrics.get('pareto_frontier_points', 0)}。",
        f"- lambda=0 的平均 hybrid-LP gain fraction：{fmt(metrics.get('lambda0_fraction'))}；top-5% allocated-unit footprint mass：{fmt(metrics.get('lambda0_top5_footprint_mass'))}。",
        f"- 最高 recovery gain 出现在 lambda={fmt(metrics.get('max_gain_lambda'))}，gain fraction={fmt(metrics.get('max_gain_fraction'))}，top-5% footprint mass={fmt(metrics.get('max_gain_top5_footprint_mass'))}。",
        f"- 最高 footprint coverage 出现在 lambda={fmt(metrics.get('max_footprint_lambda'))}，gain fraction={fmt(metrics.get('max_footprint_fraction'))}，top-5% footprint mass={fmt(metrics.get('max_footprint_top5_mass'))}。",
    ]
    for threshold in LOSS_THRESHOLDS:
        suffix = threshold_suffix(threshold)
        if f"best_lambda_loss_le_{suffix}" not in metrics:
            continue
        lines.append(
            "- 在平均 LP-gain fraction 损失不超过 {thr:g} 时，最佳 lambda={lam}，top-5% footprint mass 从 lambda=0 增加 {delta_mass}，gain fraction 变化 {delta_gain}。".format(
                thr=threshold,
                lam=fmt(metrics.get(f"best_lambda_loss_le_{suffix}")),
                delta_mass=fmt(metrics.get(f"best_delta_top5_footprint_loss_le_{suffix}")),
                delta_gain=fmt(metrics.get(f"best_delta_fraction_loss_le_{suffix}")),
            )
        )
    lines.extend(
        [
            "",
            "## Lambda Summary",
            "",
            table_to_markdown(summary),
            "",
            "## Pareto Frontier",
            "",
            table_to_markdown(frontier),
            "",
            "## Event Best Within Loss Thresholds",
            "",
            table_to_markdown(event_best),
            "",
            "## 解释",
            "",
            "这个检验比 V40 更接近论文里可以写成模型扩展的结论：如果 footprint 是一个社会偏好或公平/可见影响目标，它应该作为显式 secondary objective 进入 LP，而不是被包装成纯 recovery law。",
            "",
            "如果小 lambda 就能提高 footprint 且几乎不损失 true recovery gain，说明 footprint 可以作为 tie-breaker；如果更大的 footprint improvement 必须牺牲 recovery gain，则论文应把它写成 recovery-vs-footprint frontier，而不是声称 observed footprint 自然就是最高恢复价值区域。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys(path: Path) -> set[tuple[str, int, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "lambda_footprint", "status"}.issubset(existing.columns):
        return set()
    finished = existing[existing["status"].astype(str).isin(SUCCESS_STATUSES | {"ERROR"})].copy()
    return {
        (str(row.city), int(row.event_id), float(row.lambda_footprint))
        for row in finished[["city", "event_id", "lambda_footprint"]].itertuples(index=False)
    }


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return safe_float(pd.to_numeric(frame[column], errors="coerce").mean())


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def threshold_suffix(value: float) -> str:
    return str(value).replace(".", "p")


def fmt(value: Any) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    out = out.fillna("").astype(str)
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.to_numpy():
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

"""Broaden the explicit footprint multi-objective LP validation.

V41 solved a detailed lambda frontier on six representative hybrid-footprint
LP events.  This V42 script tests whether the same recovery-footprint boundary
appears on additional footprint-sensitive city-events.  It keeps the lambda
grid compact so the broader closure remains computationally tractable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from analyze_hybrid_absorption_mechanisms import footprint_weights
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_recovery_lp
from validate_multiobjective_footprint_lp import (
    LOSS_THRESHOLDS,
    SUCCESS_STATUSES,
    add_lambda0_deltas,
    annotate_actions,
    append_csv,
    build_event_best,
    build_metrics,
    build_pareto_frontier,
    build_reward_coefficients,
    build_summary,
    fmt,
    footprint_metrics,
    footprint_reward_score,
    make_figures,
    positive_actions,
    safe_float,
    table_to_markdown,
    true_loss_objective,
    write_table,
)


EPS = 1e-12
DEFAULT_LAMBDAS = (0.0, 0.02, 0.05)
DEFAULT_EXCLUDED_CITIES = ("New York",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--v34-dir", default="results/hybrid_footprint_calibration")
    parser.add_argument("--v41-dir", default="results/multiobjective_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/broader_multiobjective_footprint_lp_validation")
    parser.add_argument("--events-per-city", type=int, default=2)
    parser.add_argument("--exclude-cities", nargs="*", default=list(DEFAULT_EXCLUDED_CITIES))
    parser.add_argument("--include-v41-events", action="store_true")
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
        "selected_events": table_dir / "broader_multiobjective_lp_selected_events.csv",
        "event_metrics": table_dir / "broader_multiobjective_lp_event_metrics.csv",
        "selected_actions": table_dir / "broader_multiobjective_lp_selected_actions.csv.gz",
        "summary": table_dir / "broader_multiobjective_lp_summary.csv",
        "frontier": table_dir / "broader_multiobjective_lp_frontier.csv",
        "event_best": table_dir / "broader_multiobjective_lp_event_best.csv",
        "metrics": table_dir / "broader_multiobjective_lp_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    v34 = pd.read_csv(root / args.v34_dir / "tables" / "hybrid_footprint_event_metrics.csv", parse_dates=["event_start"])
    v41_events = read_v41_events(root, args.v41_dir) if not args.include_v41_events else set()
    selected_events = select_broader_events(
        v34,
        events_per_city=int(args.events_per_city),
        excluded_cities=set(args.exclude_cities or []),
        excluded_events=v41_events,
    )
    write_table(selected_events, paths["selected_events"])

    event_metrics, selected_actions = run_validation(
        root,
        config,
        selected_events,
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
    metrics.update(
        {
            "n_selected_events": int(len(selected_events)),
            "n_excluded_v41_events": int(len(v41_events)),
            "excluded_cities": ", ".join(args.exclude_cities or []),
        }
    )

    write_table(summary, paths["summary"])
    write_table(frontier, paths["frontier"])
    write_table(event_best, paths["event_best"])
    if not selected_actions.empty:
        write_table(selected_actions, paths["selected_actions"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(summary, frontier, event_best, event_metrics, figure_dir)
    write_report(report_dir / "broader_multiobjective_footprint_lp_validation_report_zh.md", metrics, selected_events, summary, frontier, event_best)
    print(f"Wrote broader explicit multi-objective footprint LP validation to {output_dir}")


def read_v41_events(root: Path, v41_dir: str) -> set[tuple[str, int]]:
    path = root / v41_dir / "tables" / "multiobjective_lp_event_metrics.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or not {"city", "event_id"}.issubset(df.columns):
        return set()
    return {(str(row.city), int(row.event_id)) for row in df[["city", "event_id"]].drop_duplicates().itertuples(index=False)}


def select_broader_events(
    v34: pd.DataFrame,
    *,
    events_per_city: int,
    excluded_cities: set[str],
    excluded_events: set[tuple[str, int]],
) -> pd.DataFrame:
    frame = v34.copy()
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype(int)
    frame = frame[~frame["city"].astype(str).isin(excluded_cities)].copy()
    if excluded_events:
        frame = frame[
            ~frame.apply(lambda row: (str(row["city"]), int(row["event_id"])) in excluded_events, axis=1)
        ].copy()
    frame["selection_score"] = (
        pd.to_numeric(frame["delta_finite_top5pct_units_footprint_mass"], errors="coerce").fillna(0.0)
        - 0.05 * pd.to_numeric(frame["finite_top5pct_action_jaccard"], errors="coerce").fillna(0.0)
    )
    selected = (
        frame.sort_values(["city", "selection_score", "event_start"], ascending=[True, False, True])
        .groupby("city", as_index=False)
        .head(max(1, int(events_per_city)))
        .sort_values(["city", "event_start", "event_id"])
        .reset_index(drop=True)
    )
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
        "selection_score",
    ]
    return selected[[col for col in keep if col in selected.columns]]


def run_validation(
    root: Path,
    config: dict[str, Any],
    selected_events: pd.DataFrame,
    *,
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
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    footprint = data["footprint_zone"].copy()
    footprint["event_id"] = pd.to_numeric(footprint["event_id"], errors="coerce").astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }
    selected_lookup = {
        (row.city, int(row.event_id)): row
        for row in selected_events.itertuples(index=False)
    }
    completed = completed_keys(event_metric_path) if resume else set()
    event_rows: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    total_jobs = len(selected_events) * len(lambdas)
    job_idx = 0
    for event_idx, selected in enumerate(selected_events.itertuples(index=False), start=1):
        city = str(selected.city)
        event_id = int(selected.event_id)
        event_key = (city, event_id)
        print(f"[event {event_idx}/{len(selected_events)}] Preparing broader LP for {city} event {event_id}", flush=True)
        event_row = event_lookup[event_key]
        footprint_group = footprint_groups[event_key]
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
        weights = footprint_weights(footprint_group, hybrid_params.units)
        reward_score = footprint_reward_score(weights)
        reference_objective = np.nan
        reference_gain = np.nan
        for lambda_footprint in lambdas:
            job_idx += 1
            key = (city, event_id, float(lambda_footprint))
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed lambda={lambda_footprint:g} for {city} event {event_id}", flush=True)
                continue
            if lambda_footprint > 0 and not np.isfinite(reference_gain):
                row = broader_error_row(
                    city,
                    event_id,
                    selected_lookup[event_key],
                    hybrid_params,
                    lambda_footprint=float(lambda_footprint),
                    baseline_objective=baseline_objective,
                    reference_objective=reference_objective,
                    reference_gain=reference_gain,
                    diagnostics=diagnostics,
                    error="lambda0_reference_missing",
                )
                event_rows.append(row)
                append_csv(pd.DataFrame([row]), event_metric_path)
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
                if np.isclose(lambda_footprint, 0.0):
                    reference_objective = true_objective
                    reference_gain = baseline_objective - true_objective
                actions = positive_actions(solved.interventions)
                row = broader_event_row(
                    city,
                    event_id,
                    selected_lookup[event_key],
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
                event_rows.append(row)
                append_csv(pd.DataFrame([row]), event_metric_path)
                annotated = annotate_actions(actions, row, weights, reward_score)
                if not annotated.empty:
                    action_frames.append(annotated)
                    append_csv(annotated, selected_action_path)
            except Exception as exc:  # pragma: no cover - long LP diagnostics
                row = broader_error_row(
                    city,
                    event_id,
                    selected_lookup[event_key],
                    hybrid_params,
                    lambda_footprint=float(lambda_footprint),
                    baseline_objective=baseline_objective,
                    reference_objective=reference_objective,
                    reference_gain=reference_gain,
                    diagnostics=diagnostics,
                    error=str(exc),
                )
                event_rows.append(row)
                append_csv(pd.DataFrame([row]), event_metric_path)

    all_events = pd.read_csv(event_metric_path, parse_dates=["event_start"]) if event_metric_path.exists() else pd.DataFrame(event_rows)
    if not all_events.empty:
        all_events["event_id"] = pd.to_numeric(all_events["event_id"], errors="coerce").astype(int)
        all_events["lambda_footprint"] = pd.to_numeric(all_events["lambda_footprint"], errors="coerce")
        all_events = all_events.drop_duplicates(["city", "event_id", "lambda_footprint"], keep="last").reset_index(drop=True)
    all_actions = pd.read_csv(selected_action_path) if selected_action_path.exists() else pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    return all_events, all_actions


def broader_event_row(
    city: str,
    event_id: int,
    selected: Any,
    params: Any,
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
    footprint = footprint_metrics(actions, weights, reward_score, params.n_units, params.total_budget)
    return {
        "city": city,
        "event_id": int(event_id),
        "event_start": str(getattr(selected, "event_start")),
        "lambda_footprint": float(lambda_footprint),
        "n_units": int(params.n_units),
        "status": status,
        "runtime_seconds": runtime_seconds,
        "baseline_objective": float(baseline_objective),
        "reference_hybrid_objective": float(reference_objective),
        "reference_hybrid_gain": float(reference_gain),
        "true_objective": float(true_objective),
        "modified_objective": float(modified_objective),
        "recoverable_fraction": float(gain / max(baseline_objective, EPS)),
        "fraction_of_reference_lp_gain": float(gain / max(reference_gain, EPS)),
        "objective_delta_vs_reference": float(true_objective - reference_objective),
        "total_intervention_cost": float(actions["effective_cost"].sum()) if "effective_cost" in actions else 0.0,
        "selected_action_count": int(len(actions)),
        "selected_unit_count": int(actions["unit"].nunique()) if "unit" in actions else 0,
        "v34_delta_finite_top5pct_units_footprint_mass": safe_float(
            getattr(selected, "delta_finite_top5pct_units_footprint_mass", np.nan)
        ),
        "v35_delta_selected_unit_footprint_mass": np.nan,
        "error": error,
        **footprint,
        **diagnostics,
    }


def broader_error_row(
    city: str,
    event_id: int,
    selected: Any,
    params: Any,
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
        "event_start": str(getattr(selected, "event_start")),
        "lambda_footprint": float(lambda_footprint),
        "n_units": int(params.n_units),
        "status": "ERROR",
        "runtime_seconds": np.nan,
        "baseline_objective": float(baseline_objective),
        "reference_hybrid_objective": safe_float(reference_objective),
        "reference_hybrid_gain": safe_float(reference_gain),
        "true_objective": np.nan,
        "modified_objective": np.nan,
        "recoverable_fraction": np.nan,
        "fraction_of_reference_lp_gain": np.nan,
        "objective_delta_vs_reference": np.nan,
        "total_intervention_cost": np.nan,
        "selected_action_count": np.nan,
        "selected_unit_count": np.nan,
        "v34_delta_finite_top5pct_units_footprint_mass": safe_float(
            getattr(selected, "delta_finite_top5pct_units_footprint_mass", np.nan)
        ),
        "v35_delta_selected_unit_footprint_mass": np.nan,
        "error": error,
        **diagnostics,
    }


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


def write_report(
    path: Path,
    metrics: dict[str, Any],
    selected_events: pd.DataFrame,
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    event_best: pd.DataFrame,
) -> None:
    lines = [
        "# Broader Explicit Multi-Objective Footprint LP Validation V42",
        "",
        "本版本在 V41 的 6 个代表性事件之外，选择更多 footprint-sensitive city-events，直接求解显式 recovery--footprint 二目标 LP。默认排除 New York，因为 1940-zone hybrid LP 在前序版本中已经形成计算边界。",
        "",
        "## 关键结论",
        "",
        f"- selected events: {metrics.get('n_selected_events', 0)}; successful events per lambda: {metrics.get('n_events', 0)}; lambdas: {metrics.get('n_lambdas', 0)}; excluded cities: {metrics.get('excluded_cities', '')}.",
        f"- lambda=0 gain/LP = {fmt(metrics.get('lambda0_fraction'))}; top-5% footprint mass = {fmt(metrics.get('lambda0_top5_footprint_mass'))}.",
        f"- <=0.005 gain-loss best lambda = {fmt(metrics.get('best_lambda_loss_le_0p005'))}; footprint delta = {fmt(metrics.get('best_delta_top5_footprint_loss_le_0p005'))}; gain delta = {fmt(metrics.get('best_delta_fraction_loss_le_0p005'))}.",
        f"- max-footprint lambda = {fmt(metrics.get('max_footprint_lambda'))}; footprint mass = {fmt(metrics.get('max_footprint_top5_mass'))}; gain/LP = {fmt(metrics.get('max_footprint_fraction'))}.",
    ]
    for threshold in LOSS_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        key = f"best_lambda_loss_le_{suffix}"
        if key not in metrics:
            continue
        lines.append(
            "- loss <= {thr:g}: lambda={lam}, footprint delta={mass}, gain delta={gain}".format(
                thr=threshold,
                lam=fmt(metrics.get(key)),
                mass=fmt(metrics.get(f"best_delta_top5_footprint_loss_le_{suffix}")),
                gain=fmt(metrics.get(f"best_delta_fraction_loss_le_{suffix}")),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Events",
            "",
            table_to_markdown(selected_events),
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
            "如果 V42 中 lambda=0.02 仍然以很小 recovery loss 换来 footprint gain，就说明 V41 的 near-free tie-breaker 不是只来自 6 个代表性事件。若 lambda=0.05 或更高出现明显 gain loss，则继续支持论文把 footprint 写成 secondary objective frontier，而不是 recovery-only law。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return
    lines = [
        "# Broader Explicit Multi-Objective Footprint LP Validation V42",
        "",
        "本版本在 V41 的 6 个代表性事件之外，选择更多 footprint-sensitive city-events，直接求解显式 recovery--footprint 二目标 LP。默认排除 New York，因为 1940-zone hybrid LP 在前序版本中已经形成计算边界。",
        "",
        "## 关键结论",
        "",
        f"- selected events: {metrics.get('n_selected_events', 0)}; successful events per lambda: {metrics.get('n_events', 0)}; lambdas: {metrics.get('n_lambdas', 0)}; excluded cities: {metrics.get('excluded_cities', '')}.",
        f"- lambda=0 gain/LP = {fmt(metrics.get('lambda0_fraction'))}; top-5% footprint mass = {fmt(metrics.get('lambda0_top5_footprint_mass'))}.",
        f"- <=0.005 gain-loss best lambda = {fmt(metrics.get('best_lambda_loss_le_0p005'))}; footprint delta = {fmt(metrics.get('best_delta_top5_footprint_loss_le_0p005'))}; gain delta = {fmt(metrics.get('best_delta_fraction_loss_le_0p005'))}.",
        f"- max-footprint lambda = {fmt(metrics.get('max_footprint_lambda'))}; footprint mass = {fmt(metrics.get('max_footprint_top5_mass'))}; gain/LP = {fmt(metrics.get('max_footprint_fraction'))}.",
    ]
    for threshold in LOSS_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        key = f"best_lambda_loss_le_{suffix}"
        if key not in metrics:
            continue
        lines.append(
            "- loss <= {thr:g}: lambda={lam}, footprint delta={mass}, gain delta={gain}".format(
                thr=threshold,
                lam=fmt(metrics.get(key)),
                mass=fmt(metrics.get(f"best_delta_top5_footprint_loss_le_{suffix}")),
                gain=fmt(metrics.get(f"best_delta_fraction_loss_le_{suffix}")),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Events",
            "",
            table_to_markdown(selected_events),
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
            "如果 V42 中 lambda=0.02 仍然以很小 recovery loss 换来 footprint gain，就说明 V41 的 near-free tie-breaker 不是只来自 6 个代表性事件。若 lambda=0.05 或更高出现明显 gain loss，则继续支持论文把 footprint 写成 secondary objective frontier，而不是 recovery-only law。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

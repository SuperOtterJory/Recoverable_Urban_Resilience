"""Validate finite-budget recovery laws against parameter-ensemble LP optima."""

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

from learn_recovery_laws import (
    build_budget_segments,
    build_event_action_frame,
    load_inputs,
    prepare_interventions,
    allocate_greedy_policy,
    replay_policy_allocations,
)
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters, solve_recovery_lp
from run_residual_greedy_policy import allocate_residual_greedy


EPS = 1e-12
DEFAULT_SCENARIOS = ("cheap_all", "slow_response_4h", "R_favored", "C_favored", "S_favored")

PARAMETER_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "parameter_scenario": "base",
        "description": "original calibrated eta/cost/delay",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "low_efficiency_all",
        "description": "all intervention effectiveness reduced by 25%",
        "eta_scale": {"R": 0.75, "C": 0.75, "S": 0.75},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "high_efficiency_all",
        "description": "all intervention effectiveness increased by 25%",
        "eta_scale": {"R": 1.25, "C": 1.25, "S": 1.25},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "expensive_all",
        "description": "all intervention costs increased by 25%",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.25, "C": 1.25, "S": 1.25},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "cheap_all",
        "description": "all intervention costs reduced by 25%",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 0.75, "C": 0.75, "S": 0.75},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "slow_response_4h",
        "description": "each intervention is delayed by four additional hours",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 4, "C": 4, "S": 4},
    },
    {
        "parameter_scenario": "R_favored",
        "description": "durable restoration is relatively more efficient and cheaper",
        "eta_scale": {"R": 1.35, "C": 0.90, "S": 0.90},
        "cost_scale": {"R": 0.85, "C": 1.10, "S": 1.10},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "C_favored",
        "description": "temporary capacity is relatively more efficient and cheaper",
        "eta_scale": {"R": 0.90, "C": 1.35, "S": 0.90},
        "cost_scale": {"R": 1.10, "C": 0.85, "S": 1.10},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "S_favored",
        "description": "substitution/control is relatively more efficient and cheaper",
        "eta_scale": {"R": 0.90, "C": 0.90, "S": 1.35},
        "cost_scale": {"R": 1.10, "C": 1.10, "S": 0.85},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/parameter_ensemble_optimum_validation")
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--events-per-city", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--max-reference-runtime-seconds", type=float, default=45.0)
    parser.add_argument("--time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--replan-budget-share", type=float, default=0.05)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_events": table_dir / "selected_events.csv",
        "scenarios": table_dir / "parameter_lp_scenarios.csv",
        "optima": table_dir / "parameter_lp_optima.csv",
        "policy": table_dir / "parameter_policy_validation.csv",
        "summary": table_dir / "parameter_policy_summary.csv",
        "city_summary": table_dir / "parameter_city_policy_summary.csv",
        "metrics": table_dir / "parameter_ensemble_optimum_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    scenarios = resolve_scenarios(args.scenarios)
    write_table(pd.DataFrame([scenario_row(s) for s in scenarios]), paths["scenarios"])

    data = load_inputs(root)
    residual_metrics = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv")
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"] == "OPTIMAL") & (base_summary["scenario"] == "base")].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype(int)
    selected_events = select_representative_events(
        residual_metrics,
        base_summary,
        events_per_city=int(args.events_per_city),
        max_events=int(args.max_events),
        max_reference_runtime_seconds=float(args.max_reference_runtime_seconds),
    )
    write_table(selected_events, paths["selected_events"])

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()
    completed = completed_keys(paths["optima"]) if args.resume else set()
    rng = np.random.default_rng(20260529)

    total_jobs = len(selected_events) * len(scenarios)
    job_idx = 0
    for _, base_row in selected_events.iterrows():
        city = str(base_row["city"])
        event_id = int(base_row["event_id"])
        event_row = event_lookup.get((city, event_id))
        if event_row is None or city not in dynamic_lookup:
            continue
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        event_interventions = interventions[
            (interventions["city"] == city)
            & (interventions["event_id"] == event_id)
            & (interventions["scenario"] == "base")
        ]

        for scenario in scenarios:
            job_idx += 1
            scenario_name = str(scenario["parameter_scenario"])
            key = (city, event_id, scenario_name)
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed {city} event {event_id} / {scenario_name}", flush=True)
                continue
            print(f"[{job_idx}/{total_jobs}] Solving parameter LP {city} event {event_id} / {scenario_name}", flush=True)
            scenario_params = apply_parameter_scenario(base_params, scenario)
            try:
                optimized = solve_recovery_lp(
                    scenario_params,
                    output_flag=bool(solver.get("output_flag", False)),
                    method=method,
                    time_limit_seconds=float(args.time_limit_seconds),
                )
                baseline_objective = float(base_row["baseline_objective"])
                scenario_lp_gain = max(baseline_objective - float(optimized.objective), EPS)
                opt_row = {
                    **event_metadata(base_row, scenario, scenario_params),
                    "status": optimized.status,
                    "runtime_seconds": float(optimized.runtime_seconds),
                    "scenario_optimized_objective": float(optimized.objective),
                    "scenario_lp_recoverable_fraction": fraction_recovered(baseline_objective, float(optimized.objective)),
                    "scenario_lp_gain": float(scenario_lp_gain),
                    "error": "",
                }
                append_csv(pd.DataFrame([opt_row]), paths["optima"])

                full = build_event_action_frame(scenario_params, base_row, event_row, event_interventions)
                full["scenario"] = scenario_name
                full["total_budget"] = float(scenario_params.total_budget)
                full["budget_fraction_of_baseline"] = float(scenario_params.total_budget) / max(baseline_objective, EPS)
                segments = build_budget_segments(full, config, rng)
                policy_rows = build_policy_rows(
                    full,
                    segments,
                    scenario_params,
                    base_row,
                    scenario,
                    optimized_objective=float(optimized.objective),
                    lp_status=str(optimized.status),
                    replan_budget_share=float(args.replan_budget_share),
                    max_replans=int(args.max_replans),
                )
                append_csv(pd.DataFrame(policy_rows), paths["policy"])
            except Exception as exc:  # pragma: no cover - long batch diagnostics
                print(f"ERROR {city} event {event_id} / {scenario_name}: {exc}", flush=True)
                error_row = {
                    **event_metadata(base_row, scenario, scenario_params),
                    "status": "ERROR",
                    "runtime_seconds": np.nan,
                    "scenario_optimized_objective": np.nan,
                    "scenario_lp_recoverable_fraction": np.nan,
                    "scenario_lp_gain": np.nan,
                    "error": str(exc),
                }
                append_csv(pd.DataFrame([error_row]), paths["optima"])

    optima = pd.read_csv(paths["optima"]) if paths["optima"].exists() else pd.DataFrame()
    policy = pd.read_csv(paths["policy"]) if paths["policy"].exists() else pd.DataFrame()
    summary = summarize_policy(policy)
    city_summary = summarize_city_policy(policy)
    metrics = build_metrics(optima, policy, summary)
    write_table(summary, paths["summary"])
    write_table(city_summary, paths["city_summary"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(policy, summary, figure_dir)
    write_report(
        report_dir / "parameter_ensemble_optimum_validation_report_zh.md",
        selected_events,
        optima,
        policy,
        summary,
        city_summary,
        metrics,
        scenarios,
        args,
    )
    print(f"Wrote parameter-ensemble optimum validation to {output_dir}")


def resolve_scenarios(names: list[str]) -> list[dict[str, Any]]:
    available = {str(row["parameter_scenario"]): dict(row) for row in PARAMETER_SCENARIOS}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown parameter scenarios: {missing}. Available: {sorted(available)}")
    return [available[name] for name in names]


def scenario_row(scenario: dict[str, Any]) -> dict[str, Any]:
    row = {
        "parameter_scenario": scenario["parameter_scenario"],
        "description": scenario["description"],
    }
    for key in INTERVENTIONS:
        row[f"eta_scale_{key}"] = float(scenario["eta_scale"][key])
        row[f"cost_scale_{key}"] = float(scenario["cost_scale"][key])
        row[f"delay_add_{key}"] = int(scenario["delay_add"][key])
    return row


def select_representative_events(
    residual_metrics: pd.DataFrame,
    base_summary: pd.DataFrame,
    *,
    events_per_city: int,
    max_events: int,
    max_reference_runtime_seconds: float,
) -> pd.DataFrame:
    residual = residual_metrics.copy()
    residual["event_id"] = pd.to_numeric(residual["event_id"], errors="coerce").astype(int)
    base_cols = [
        "city",
        "event_id",
        "event_start",
        "n_units",
        "runtime_seconds",
        "baseline_objective",
        "optimized_objective",
        "recoverable_fraction",
        "total_budget",
        "weighted_b0",
        "weighted_h_total",
        "event_total_precip",
        "event_peak_precip",
        "event_peak_positive_abnormal_deficit",
    ]
    merged = residual.merge(base_summary[base_cols], on=["city", "event_id", "event_start"], how="left", suffixes=("", "_base"))
    merged = merged[merged["runtime_seconds"].fillna(np.inf) <= max_reference_runtime_seconds].copy()
    merged = merged.sort_values(["city", "residual_gain_improvement_over_static"], ascending=[True, False])
    per_city = merged.groupby("city", as_index=False).head(max(events_per_city, 1)).copy()
    per_city = per_city.sort_values("residual_gain_improvement_over_static", ascending=False).head(max_events).copy()
    per_city["selection_note"] = "highest_residual_improvement_under_runtime_guard"
    keep = [
        "city",
        "event_id",
        "event_start",
        "selection_note",
        "n_units",
        "runtime_seconds",
        "baseline_objective",
        "optimized_objective",
        "recoverable_fraction",
        "static_fraction_of_lp_gain",
        "residual_fraction_of_lp_gain",
        "residual_gain_improvement_over_static",
        "residual_gap_to_lp",
        "total_budget",
        "weighted_b0",
        "weighted_h_total",
        "event_total_precip",
        "event_peak_precip",
        "event_peak_positive_abnormal_deficit",
    ]
    return per_city[[column for column in keep if column in per_city.columns]].reset_index(drop=True)


def apply_parameter_scenario(params: RecoveryLPParameters, scenario: dict[str, Any]) -> RecoveryLPParameters:
    eta = {
        key: params.eta[key].copy() * float(scenario["eta_scale"][key])
        for key in INTERVENTIONS
    }
    cost = {
        key: params.cost[key].copy() * float(scenario["cost_scale"][key])
        for key in INTERVENTIONS
    }
    delays = {
        key: max(0, int(params.delays.get(key, 0)) + int(scenario["delay_add"][key]))
        for key in INTERVENTIONS
    }
    return RecoveryLPParameters(
        city=params.city,
        units=list(params.units),
        p=params.p.copy(),
        q=params.q.copy(),
        b0=params.b0.copy(),
        a=params.a.copy(),
        h=params.h.copy(),
        eta=eta,
        cost=cost,
        u_cap={key: value.copy() for key, value in (params.u_cap or {}).items()},
        u_segment_cap={key: value.copy() for key, value in (params.u_segment_cap or {}).items()} or None,
        segment_effectiveness={key: value.copy() for key, value in (params.segment_effectiveness or {}).items()} or None,
        period_budget=params.period_budget.copy(),
        total_budget=float(params.total_budget),
        delays=delays,
        delta_c=float(params.delta_c),
        delta_s=float(params.delta_s),
        delta_t=float(params.delta_t),
        metadata={
            **dict(params.metadata or {}),
            "parameter_scenario": str(scenario["parameter_scenario"]),
            "parameter_scenario_description": str(scenario["description"]),
        },
    )


def event_metadata(base_row: pd.Series, scenario: dict[str, Any], scenario_params: RecoveryLPParameters) -> dict[str, Any]:
    row = {
        "city": str(base_row["city"]),
        "event_id": int(base_row["event_id"]),
        "event_start": str(base_row["event_start"]),
        "parameter_scenario": str(scenario["parameter_scenario"]),
        "description": str(scenario["description"]),
        "n_units": int(base_row["n_units"]),
        "baseline_objective": float(base_row["baseline_objective"]),
        "base_optimized_objective": float(base_row["optimized_objective"]),
        "base_lp_recoverable_fraction": float(base_row["recoverable_fraction"]),
        "base_runtime_seconds": float(base_row["runtime_seconds"]),
        "base_static_fraction_of_lp_gain": float(base_row["static_fraction_of_lp_gain"]),
        "base_residual_fraction_of_lp_gain": float(base_row["residual_fraction_of_lp_gain"]),
        "base_residual_improvement": float(base_row["residual_gain_improvement_over_static"]),
        "scenario_total_budget": float(scenario_params.total_budget),
        "mean_period_budget": float(np.mean(scenario_params.period_budget)),
        "delay_R": int(scenario_params.delays.get("R", 0)),
        "delay_C": int(scenario_params.delays.get("C", 0)),
        "delay_S": int(scenario_params.delays.get("S", 0)),
        "event_peak_positive_abnormal_deficit": float(base_row["event_peak_positive_abnormal_deficit"]),
        "event_total_precip": float(base_row["event_total_precip"]),
    }
    for key in INTERVENTIONS:
        row[f"eta_scale_{key}"] = float(scenario["eta_scale"][key])
        row[f"cost_scale_{key}"] = float(scenario["cost_scale"][key])
        row[f"delay_add_{key}"] = int(scenario["delay_add"][key])
    return row


def build_policy_rows(
    full: pd.DataFrame,
    segments: pd.DataFrame,
    scenario_params: RecoveryLPParameters,
    base_row: pd.Series,
    scenario: dict[str, Any],
    *,
    optimized_objective: float,
    lp_status: str,
    replan_budget_share: float,
    max_replans: int,
) -> list[dict[str, Any]]:
    scenario_name = str(scenario["parameter_scenario"])
    baseline_objective = float(base_row["baseline_objective"])
    scenario_lp_gain = max(baseline_objective - optimized_objective, EPS)
    rows: list[dict[str, Any]] = []

    feasible = np.zeros(len(segments), dtype=bool)
    for intervention in INTERVENTIONS:
        delay = int(scenario_params.delays.get(intervention, 0))
        mask = segments["intervention"].eq(intervention).to_numpy()
        feasible[mask] = segments.loc[mask, "t"].to_numpy(dtype=int) >= delay
    scenario_segments = segments.loc[feasible & (segments["oracle_value_per_cost"] > 0.0)].copy()

    static = allocate_greedy_policy(
        scenario_segments,
        "oracle_value_per_cost",
        period_budget=scenario_params.period_budget,
        total_budget=float(scenario_params.total_budget),
    )
    static_replay = replay_policy_allocations(static["allocations"], scenario_params)
    rows.append(
        policy_result_row(
            base_row,
            scenario,
            "static_small_signal_greedy",
            baseline_objective,
            optimized_objective,
            scenario_lp_gain,
            replay_objective=float(static_replay["objective"]),
            allocated_cost=float(static["allocated_cost"]),
            selected_action_count=int(static["selected_action_count"]),
            lp_status=lp_status,
        )
    )

    residual = allocate_residual_greedy(
        segments,
        scenario_params,
        baseline_objective=baseline_objective,
        replan_budget_share=replan_budget_share,
        max_replans=max_replans,
    )
    residual_replay = replay_policy_allocations(residual["allocations"], scenario_params)
    rows.append(
        policy_result_row(
            base_row,
            scenario,
            "residual_finite_greedy",
            baseline_objective,
            optimized_objective,
            scenario_lp_gain,
            replay_objective=float(residual_replay["objective"]),
            allocated_cost=float(residual["allocated_cost"]),
            selected_action_count=int(residual["selected_action_count"]),
            lp_status=lp_status,
        )
    )
    return rows


def policy_result_row(
    base_row: pd.Series,
    scenario: dict[str, Any],
    policy: str,
    baseline_objective: float,
    optimized_objective: float,
    scenario_lp_gain: float,
    *,
    replay_objective: float,
    allocated_cost: float,
    selected_action_count: int,
    lp_status: str,
) -> dict[str, Any]:
    replay_gain = baseline_objective - replay_objective
    return {
        "city": str(base_row["city"]),
        "event_id": int(base_row["event_id"]),
        "event_start": str(base_row["event_start"]),
        "parameter_scenario": str(scenario["parameter_scenario"]),
        "description": str(scenario["description"]),
        "policy": policy,
        "lp_status": lp_status,
        "baseline_objective": float(baseline_objective),
        "scenario_optimized_objective": float(optimized_objective),
        "scenario_lp_gain": float(scenario_lp_gain),
        "replay_objective": float(replay_objective),
        "replay_gain": float(replay_gain),
        "replay_recoverable_fraction": fraction_recovered(baseline_objective, replay_objective),
        "fraction_of_scenario_lp_gain": float(replay_gain / scenario_lp_gain),
        "gap_to_scenario_lp_gain": float(1.0 - replay_gain / scenario_lp_gain),
        "allocated_cost": float(allocated_cost),
        "selected_action_count": int(selected_action_count),
        "base_static_fraction_of_lp_gain": float(base_row["static_fraction_of_lp_gain"]),
        "base_residual_fraction_of_lp_gain": float(base_row["residual_fraction_of_lp_gain"]),
        "base_residual_improvement": float(base_row["residual_gain_improvement_over_static"]),
        "event_peak_positive_abnormal_deficit": float(base_row["event_peak_positive_abnormal_deficit"]),
        "event_total_precip": float(base_row["event_total_precip"]),
    }


def summarize_policy(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    return (
        policy.groupby(["parameter_scenario", "policy"], as_index=False)
        .agg(
            n_event_scenarios=("event_id", "count"),
            mean_fraction_of_scenario_lp_gain=("fraction_of_scenario_lp_gain", "mean"),
            median_fraction_of_scenario_lp_gain=("fraction_of_scenario_lp_gain", "median"),
            mean_gap_to_scenario_lp_gain=("gap_to_scenario_lp_gain", "mean"),
            median_gap_to_scenario_lp_gain=("gap_to_scenario_lp_gain", "median"),
            mean_recoverable_fraction=("replay_recoverable_fraction", "mean"),
            mean_allocated_cost=("allocated_cost", "mean"),
            mean_selected_action_count=("selected_action_count", "mean"),
        )
        .sort_values(["parameter_scenario", "policy"])
    )


def summarize_city_policy(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    return (
        policy.groupby(["city", "policy"], as_index=False)
        .agg(
            n_event_scenarios=("event_id", "count"),
            mean_fraction_of_scenario_lp_gain=("fraction_of_scenario_lp_gain", "mean"),
            median_fraction_of_scenario_lp_gain=("fraction_of_scenario_lp_gain", "median"),
            mean_gap_to_scenario_lp_gain=("gap_to_scenario_lp_gain", "mean"),
        )
        .sort_values(["policy", "mean_fraction_of_scenario_lp_gain"], ascending=[True, False])
    )


def build_metrics(optima: pd.DataFrame, policy: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    ok = optima[optima["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy() if not optima.empty else pd.DataFrame()
    pivot = policy.pivot_table(
        index=["city", "event_id", "parameter_scenario"],
        columns="policy",
        values="fraction_of_scenario_lp_gain",
        aggfunc="first",
    ).reset_index() if not policy.empty else pd.DataFrame()
    if {"static_small_signal_greedy", "residual_finite_greedy"}.issubset(pivot.columns):
        pivot["residual_minus_static"] = pivot["residual_finite_greedy"] - pivot["static_small_signal_greedy"]
    residual_policy = policy[policy["policy"].eq("residual_finite_greedy")].copy() if "policy" in policy else pd.DataFrame()
    static_policy = policy[policy["policy"].eq("static_small_signal_greedy")].copy() if "policy" in policy else pd.DataFrame()
    worst_residual = (
        summary[summary["policy"].eq("residual_finite_greedy")]
        .sort_values("mean_fraction_of_scenario_lp_gain")
        .head(1)
        if "policy" in summary
        else pd.DataFrame()
    )
    worst_row = worst_residual.iloc[0] if not worst_residual.empty else pd.Series(dtype=float)
    return {
        "n_selected_events": int(ok[["city", "event_id"]].drop_duplicates().shape[0]) if not ok.empty else 0,
        "n_parameter_scenarios": int(ok["parameter_scenario"].nunique()) if "parameter_scenario" in ok else 0,
        "n_successful_lp_scenarios": int(len(ok)),
        "n_policy_rows": int(len(policy)),
        "mean_lp_runtime_seconds": safe_float(ok["runtime_seconds"].mean()) if "runtime_seconds" in ok else np.nan,
        "max_lp_runtime_seconds": safe_float(ok["runtime_seconds"].max()) if "runtime_seconds" in ok else np.nan,
        "mean_residual_fraction_of_scenario_lp_gain": safe_float(residual_policy["fraction_of_scenario_lp_gain"].mean()) if "fraction_of_scenario_lp_gain" in residual_policy else np.nan,
        "median_residual_fraction_of_scenario_lp_gain": safe_float(residual_policy["fraction_of_scenario_lp_gain"].median()) if "fraction_of_scenario_lp_gain" in residual_policy else np.nan,
        "mean_static_fraction_of_scenario_lp_gain": safe_float(static_policy["fraction_of_scenario_lp_gain"].mean()) if "fraction_of_scenario_lp_gain" in static_policy else np.nan,
        "mean_residual_minus_static": safe_float(pivot["residual_minus_static"].mean()) if "residual_minus_static" in pivot else np.nan,
        "positive_residual_improvement_share": safe_float((pivot["residual_minus_static"] > 1e-6).mean()) if "residual_minus_static" in pivot else np.nan,
        "worst_residual_parameter_scenario": str(worst_row.get("parameter_scenario", "")),
        "worst_residual_mean_fraction_of_scenario_lp_gain": safe_float(worst_row.get("mean_fraction_of_scenario_lp_gain")),
    }


def make_figures(policy: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    if policy.empty or summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [scenario["parameter_scenario"] for scenario in PARAMETER_SCENARIOS if scenario["parameter_scenario"] in set(summary["parameter_scenario"])]
    policies = ["static_small_signal_greedy", "residual_finite_greedy"]
    colors = {"static_small_signal_greedy": "#94a3b8", "residual_finite_greedy": "#2563eb"}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(order))
    width = 0.36
    for idx, policy_name in enumerate(policies):
        values = []
        for scenario in order:
            match = summary[(summary["parameter_scenario"] == scenario) & (summary["policy"] == policy_name)]
            values.append(float(match["mean_fraction_of_scenario_lp_gain"].iloc[0]) if not match.empty else np.nan)
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=policy_name.replace("_", " "), color=colors[policy_name])
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xticks(x, order, rotation=24, ha="right")
    ax.set_ylabel("Policy gain / parameter-scenario LP gain")
    ax.set_title("Full LP closure under parameter ensembles")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "parameter_policy_fraction_of_lp.png", dpi=180)
    plt.close(fig)

    pivot = policy.pivot_table(
        index=["city", "event_id", "parameter_scenario"],
        columns="policy",
        values="fraction_of_scenario_lp_gain",
        aggfunc="first",
    ).reset_index()
    if set(policies).issubset(pivot.columns):
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        for scenario in order:
            subset = pivot[pivot["parameter_scenario"] == scenario]
            ax.scatter(
                subset["static_small_signal_greedy"],
                subset["residual_finite_greedy"],
                s=58,
                alpha=0.78,
                label=scenario,
            )
        values = pivot[policies].to_numpy(dtype=float)
        low = max(0.0, float(np.nanmin(values)) - 0.04)
        high = min(1.15, float(np.nanmax(values)) + 0.04)
        ax.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1, alpha=0.45)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Static small-signal / scenario LP")
        ax.set_ylabel("Residual finite greedy / scenario LP")
        ax.set_title("Residual law under parameter shifts")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_vs_static_parameter_lp.png", dpi=180)
        plt.close(fig)

    residual = policy[policy["policy"] == "residual_finite_greedy"].copy()
    residual = residual.sort_values("gap_to_scenario_lp_gain", ascending=False).head(18)
    if not residual.empty:
        labels = residual.apply(lambda row: f"{row['city']} {int(row['event_id'])} {row['parameter_scenario']}", axis=1)
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        y = np.arange(len(residual))
        ax.barh(y, residual["gap_to_scenario_lp_gain"], color="#ef4444", alpha=0.78)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Residual policy gap to parameter-scenario LP gain")
        ax.set_title("Largest remaining parameter-closure gaps")
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_parameter_gap_to_lp.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    selected_events: pd.DataFrame,
    optima: pd.DataFrame,
    policy: pd.DataFrame,
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    metrics: dict[str, Any],
    scenarios: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    status_counts = optima["status"].astype(str).value_counts().to_dict() if not optima.empty and "status" in optima else {}
    lines = [
        "# Parameter-Ensemble LP Optimum Validation V24",
        "",
        "## 本版回答的问题",
        "",
        "V23 只在 action-token field 上重算 first-order small-signal target，检验的是局部 action-value law 的参数稳定性。V24 进一步对代表性 city-event 重新求解参数扰动后的完整 Gurobi LP optimum，并把 static small-signal greedy 与 residual finite greedy 放回同一参数场景中 replay。这样分母不再是 base LP gain，而是每个 eta/cost/delay/channel-favored 场景自己的 LP gain。",
        "",
        "这仍然不是全 105 event、全 11 parameter scenarios 的最终闭合；它是一个代表性 full-LP closure，用来检验 V23 的 first-order 发现能否延伸到 finite-budget residual interaction。",
        "",
        "## 求解覆盖",
        "",
        f"- selected events: {len(selected_events)}",
        f"- parameter scenarios: {', '.join(str(row['parameter_scenario']) for row in scenarios)}",
        f"- successful LP scenario rows: {metrics['n_successful_lp_scenarios']}",
        f"- LP status counts: {status_counts}",
        f"- mean/max LP runtime seconds: {metrics['mean_lp_runtime_seconds']:.2f} / {metrics['max_lp_runtime_seconds']:.2f}",
        f"- replan budget share: {args.replan_budget_share:.2%}",
        "",
        "## 关键结果",
        "",
        f"- static small-signal mean policy/LP gain = {metrics['mean_static_fraction_of_scenario_lp_gain']:.4f}",
        f"- residual finite greedy mean policy/LP gain = {metrics['mean_residual_fraction_of_scenario_lp_gain']:.4f}",
        f"- residual-minus-static mean = {metrics['mean_residual_minus_static']:.4f}",
        f"- positive residual improvement share = {metrics['positive_residual_improvement_share']:.4f}",
        f"- weakest residual scenario = {metrics['worst_residual_parameter_scenario']} at {metrics['worst_residual_mean_fraction_of_scenario_lp_gain']:.4f}",
        "",
        "解释：如果 residual finite greedy 仍接近 1，说明 action-level law 经过 residual state 更新后不仅在 base regime 成立，也能在 eta/cost/delay/channel-favored 参数扰动下接近对应场景的真实 LP 上界。若某些场景 gap 变大，说明参数改变后 LP 的全局同时优化、period budget shadow price 或 R/C/S 替代关系变得更重要。",
        "",
        "## Representative Events",
        "",
        table_to_markdown(selected_events),
        "",
        "## Scenario Summary",
        "",
        table_to_markdown(summary),
    ]
    if not city_summary.empty:
        lines.extend(["", "## City Summary", "", table_to_markdown(city_summary)])
    if not policy.empty:
        pivot = policy.pivot_table(
            index=["city", "event_id", "parameter_scenario"],
            columns="policy",
            values="fraction_of_scenario_lp_gain",
            aggfunc="first",
        ).reset_index()
        if {"static_small_signal_greedy", "residual_finite_greedy"}.issubset(pivot.columns):
            pivot["residual_minus_static"] = pivot["residual_finite_greedy"] - pivot["static_small_signal_greedy"]
            hardest = pivot.sort_values("residual_finite_greedy").head(12)
            lines.extend(["", "## Hardest Residual Cases", "", table_to_markdown(hardest)])
    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            "V24 的参数场景改变 eta、cost 和 delay，但没有改变 OD demand、事件空间 footprint、自然恢复或预算制度本身。eta scaling 也沿用了既有 perturbation 逻辑：deployment cap 保持 base calibration，效率改变会同时改变单位资源效果和可达到的最大有效效果。因此它是管理参数敏感性检验，而不是真实干预因果识别。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "parameter_scenario", "status"}.issubset(existing.columns):
        return set()
    valid = existing[existing["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    return {
        (str(row.city), int(row.event_id), str(row.parameter_scenario))
        for row in valid[["city", "event_id", "parameter_scenario"]].itertuples(index=False)
    }


def one_row(df: pd.DataFrame, **filters: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df:
            return pd.Series(dtype=float)
        mask &= df[key].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=float)
    return df.loc[mask].iloc[0]


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def fraction_recovered(baseline_objective: float, objective: float) -> float:
    return float(1.0 - objective / baseline_objective) if baseline_objective > EPS else np.nan


def table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    compact = df.copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.10g")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

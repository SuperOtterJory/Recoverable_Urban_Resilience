"""Run full-zone LPs for observed rainfall-event scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recoverable_resilience.calibration import calibration_summary, load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_recovery_lp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--events", default="results/data_mining/tables/rainfall_event_impact_details.csv")
    parser.add_argument("--dynamics", default="results/event_calibration/tables/event_dynamic_calibration_summary.csv")
    parser.add_argument("--output-dir", default="results/event_optimization")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--max-events-per-city", type=int, default=None)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / "event_optimization_summary.csv"
    calibration_path = table_dir / "event_calibration_summary.csv"
    intervention_path = table_dir / "event_optimization_interventions.csv"
    if not args.resume:
        for path in [summary_path, calibration_path, intervention_path]:
            if path.exists():
                path.unlink()

    events = load_events(root / args.events)
    dynamics = pd.read_csv(root / args.dynamics)
    dynamic_by_city = {row["city"]: row for _, row in dynamics.iterrows()}
    abnormal = pd.read_csv(root / config["project"]["data_mining_tables_dir"] / "speed_hourly_abnormal_deficit.csv", parse_dates=["hour"])

    cities = args.cities or config["calibration"]["cities"]
    events = events[events["city"].isin(cities)].copy()
    if args.max_events_per_city is not None:
        events = (
            events.sort_values(["city", "peak_positive_abnormal_deficit"], ascending=[True, False])
            .groupby("city", as_index=False)
            .head(args.max_events_per_city)
            .sort_values(["city", "event_start"])
        )

    scenarios = config["scenarios"]
    if args.scenarios:
        scenarios = [scenario for scenario in scenarios if scenario["name"] in set(args.scenarios)]

    solver = config.get("solver", {})
    completed = completed_keys(summary_path) if args.resume else set()
    total_jobs = len(events) * len(scenarios)
    job_idx = 0
    for _, event in events.iterrows():
        city = event["city"]
        if city not in dynamic_by_city:
            append_csv(pd.DataFrame([error_row(event, "missing_dynamic_calibration")]), summary_path)
            continue
        for scenario in scenarios:
            job_idx += 1
            scenario_name = scenario["name"]
            key = (city, int(event["event_id"]), scenario_name)
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed {city} event {int(event['event_id'])} / {scenario_name}", flush=True)
                continue
            print(
                f"[{job_idx}/{total_jobs}] Solving {city} event {int(event['event_id'])} "
                f"{event['event_start']} / {scenario_name}",
                flush=True,
            )
            try:
                params = calibrate_observed_event_city(
                    city,
                    config,
                    event,
                    dynamic_by_city[city],
                    scenario_override=scenario,
                    abnormal_hourly=abnormal,
                    root=root,
                )
                calibration_row = calibration_summary(params)
                calibration_row.update(event_metadata(params.metadata, scenario_name))
                append_csv(pd.DataFrame([calibration_row]), calibration_path)

                baseline_objective = no_intervention_objective(params)
                time_limit = args.time_limit_seconds
                if time_limit is None:
                    time_limit = float(solver.get("time_limit_seconds", 300))
                optimized = solve_recovery_lp(
                    params,
                    output_flag=bool(solver.get("output_flag", False)),
                    method=int(args.method if args.method is not None else solver.get("method", -1)),
                    time_limit_seconds=float(time_limit),
                )
                optimized.baseline_objective = baseline_objective
                optimized.recoverable_fraction = (
                    1.0 - optimized.objective / baseline_objective if baseline_objective > 1e-10 else np.nan
                )
                optimized.summary["baseline_objective"] = baseline_objective
                optimized.summary["recoverable_fraction"] = optimized.recoverable_fraction
                row = dict(optimized.summary)
                row.update(event_metadata(params.metadata, scenario_name))
                row["baseline_objective"] = baseline_objective
                row["optimized_objective"] = optimized.objective
                row["recoverable_fraction"] = optimized.recoverable_fraction
                row["weighted_b0"] = float(np.sum(params.p * params.b0))
                row["weighted_h_total"] = float(np.sum(params.p[:, None] * params.h))
                row["weighted_h_peak"] = float(np.max(params.h.T @ params.p))
                row["budget_intensity"] = scenario.get("budget_intensity", config["interventions"]["budget_intensity"])
                row["error"] = ""
                append_csv(pd.DataFrame([row]), summary_path)

                positive_interventions = optimized.interventions[
                    (optimized.interventions["u"] > 1e-10)
                    | (optimized.interventions["e"] > 1e-10)
                    | (optimized.interventions["effective_cost"] > 1e-10)
                ].copy()
                if not positive_interventions.empty:
                    for key, value in event_metadata(params.metadata, scenario_name).items():
                        positive_interventions[key] = value
                    append_csv(positive_interventions, intervention_path)
            except Exception as exc:  # pragma: no cover - keeps batch jobs inspectable
                print(f"ERROR {city} event {event['event_id']}: {exc}", flush=True)
                append_csv(pd.DataFrame([error_row(event, str(exc), scenario_name)]), summary_path)
    print(f"Wrote event optimization outputs to {output_dir}")


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_start", "event_end"])
    impact_available = events["impact_available"].astype(str).str.lower().isin({"true", "1", "yes"})
    events["peak_positive_abnormal_deficit"] = pd.to_numeric(
        events["peak_positive_abnormal_deficit"],
        errors="coerce",
    ).fillna(0.0)
    events = events[impact_available & (events["peak_positive_abnormal_deficit"] > 0)].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").fillna(-1).astype(int)
    events = events.sort_values(["city", "event_start"]).reset_index(drop=True)
    return events


def event_metadata(metadata: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    keys = [
        "scenario_type",
        "event_id",
        "event_start",
        "event_end",
        "event_total_precip",
        "event_peak_precip",
        "event_peak_positive_abnormal_deficit",
        "city_signal_b0",
        "city_signal_h_total",
        "city_signal_h_peak",
        "dynamic_a_retention",
        "dynamic_rain_kernel_sum",
        "h_observed_innovation_steps",
        "h_rain_kernel_fallback_steps",
        "h_signal_source",
    ]
    row = {key: metadata.get(key) for key in keys}
    row["scenario"] = scenario_name
    return row


def completed_keys(summary_path: Path) -> set[tuple[str, int, str]]:
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(summary_path)
    if existing.empty or not {"city", "event_id", "scenario", "status"}.issubset(existing.columns):
        return set()
    existing = existing[existing["status"] == "OPTIMAL"].copy()
    return {
        (str(row.city), int(row.event_id), str(row.scenario))
        for row in existing[["city", "event_id", "scenario"]].itertuples(index=False)
    }


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def no_intervention_objective(params: Any) -> float:
    b = params.b0.copy()
    objective = 0.0
    for t in range(params.horizon + 1):
        ell = np.clip(params.q @ b, 0.0, 1.0)
        objective += float(params.delta_t * np.sum(params.p * ell))
        if t == params.horizon:
            break
        b = np.clip(params.a * b + params.h[:, t + 1], 0.0, 1.0)
    return objective


def error_row(event: pd.Series, error: str, scenario_name: str = "base") -> dict[str, Any]:
    return {
        "city": event.get("city"),
        "status": "ERROR",
        "scenario": scenario_name,
        "event_id": event.get("event_id"),
        "event_start": event.get("event_start"),
        "event_end": event.get("event_end"),
        "event_total_precip": event.get("total_precip"),
        "event_peak_precip": event.get("peak_precip"),
        "event_peak_positive_abnormal_deficit": event.get("peak_positive_abnormal_deficit"),
        "error": error,
    }


if __name__ == "__main__":
    main()

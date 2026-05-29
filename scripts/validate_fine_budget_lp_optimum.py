"""Validate fine-budget leverage laws against representative LP optima."""

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

from learn_recovery_laws import load_inputs
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_recovery_lp


DEFAULT_BUDGET_SCALES = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
POLICY_ORDER = [
    "activated_bottleneck_law",
    "greedy_oracle",
    "random_positive",
    "exposure_only",
    "deficit_only",
    "structure_only",
]
EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/fine_budget_lp_validation")
    parser.add_argument("--budget-scales", nargs="*", type=float, default=DEFAULT_BUDGET_SCALES)
    parser.add_argument("--events-per-city", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=7)
    parser.add_argument("--max-reference-runtime-seconds", type=float, default=30.0)
    parser.add_argument("--time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))
    budget_scales = sorted(set(float(scale) for scale in args.budget_scales))

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_events": table_dir / "selected_events.csv",
        "optima": table_dir / "fine_budget_lp_optima.csv",
        "policy": table_dir / "fine_budget_policy_vs_lp.csv",
        "summary": table_dir / "fine_budget_lp_summary.csv",
        "phase": table_dir / "fine_budget_lp_phase_tests.csv",
        "city_summary": table_dir / "fine_budget_lp_city_summary.csv",
        "metrics": table_dir / "fine_budget_lp_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    data = load_inputs(root)
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"] == "OPTIMAL") & (base_summary["scenario"] == "base")].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype(int)
    residual_metrics = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv")
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
    abnormal = data["abnormal"].copy()
    completed = completed_keys(paths["optima"]) if args.resume else set()

    if not args.postprocess_only:
        total_jobs = len(selected_events) * len(budget_scales)
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
            for budget_scale in budget_scales:
                job_idx += 1
                key = (city, event_id, budget_key(budget_scale))
                if key in completed:
                    print(f"[{job_idx}/{total_jobs}] Skipping completed {city} event {event_id} / budget {budget_scale:g}", flush=True)
                    continue
                print(f"[{job_idx}/{total_jobs}] Solving fine-budget LP {city} event {event_id} / budget {budget_scale:g}", flush=True)
                scenario_params = base_params.copy_with_budget(float(budget_scale))
                try:
                    optimized = solve_recovery_lp(
                        scenario_params,
                        output_flag=bool(solver.get("output_flag", False)),
                        method=method,
                        time_limit_seconds=float(args.time_limit_seconds),
                    )
                    append_csv(
                        pd.DataFrame(
                            [
                                optimum_row(
                                    base_row,
                                    budget_scale,
                                    scenario_params,
                                    status=str(optimized.status),
                                    runtime_seconds=float(optimized.runtime_seconds),
                                    optimized_objective=float(optimized.objective),
                                    error="",
                                )
                            ]
                        ),
                        paths["optima"],
                    )
                except Exception as exc:  # pragma: no cover - long batch diagnostics
                    print(f"ERROR {city} event {event_id} / budget {budget_scale:g}: {exc}", flush=True)
                    append_csv(
                        pd.DataFrame(
                            [
                                optimum_row(
                                    base_row,
                                    budget_scale,
                                    scenario_params,
                                    status="ERROR",
                                    runtime_seconds=np.nan,
                                    optimized_objective=np.nan,
                                    error=str(exc),
                                )
                            ]
                        ),
                        paths["optima"],
                    )

    optima = pd.read_csv(paths["optima"]) if paths["optima"].exists() else pd.DataFrame()
    optima = deduplicate_optima(optima)
    write_table(optima, paths["optima"])
    replay = pd.read_csv(root / "results" / "budget_fine_sweep" / "tables" / "fine_budget_policy_replay.csv")
    policy = build_policy_vs_lp(optima, replay)
    summary = build_summary(policy, optima)
    phase = build_phase_tests(summary)
    city_summary = build_city_summary(policy)
    metrics = build_metrics(optima, policy, summary, phase)

    write_table(policy, paths["policy"])
    write_table(summary, paths["summary"])
    write_table(phase, paths["phase"])
    write_table(city_summary, paths["city_summary"])
    paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(summary, city_summary, policy, figure_dir)
    write_report(
        report_dir / "fine_budget_lp_validation_report_zh.md",
        selected_events,
        optima,
        policy,
        summary,
        phase,
        city_summary,
        metrics,
        args,
    )
    print(f"Wrote fine-budget LP validation to {output_dir}")


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
    merged = merged.sort_values(["city", "residual_gain_improvement_over_static"], ascending=[True, False])
    selected: list[pd.DataFrame] = []
    for city, group in merged.groupby("city", sort=True):
        eligible = group[group["runtime_seconds"].fillna(np.inf) <= max_reference_runtime_seconds].copy()
        if eligible.empty:
            eligible = group.sort_values("runtime_seconds").head(max(events_per_city, 1)).copy()
            eligible["selection_note"] = "fastest_available_no_event_under_runtime_guard"
        else:
            eligible = eligible.head(max(events_per_city, 1)).copy()
            eligible["selection_note"] = "highest_residual_improvement_under_runtime_guard"
        selected.append(eligible)
    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    out = out.sort_values(["city", "residual_gain_improvement_over_static"], ascending=[True, False]).reset_index(drop=True)
    if max_events:
        out = out.head(max_events).copy()
    out["selection_rank_in_city"] = out.groupby("city").cumcount() + 1
    keep = [
        "city",
        "event_id",
        "event_start",
        "selection_note",
        "selection_rank_in_city",
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
    return out[[col for col in keep if col in out.columns]]


def optimum_row(
    base_row: pd.Series,
    budget_scale: float,
    scenario_params: Any,
    *,
    status: str,
    runtime_seconds: float,
    optimized_objective: float,
    error: str,
) -> dict[str, Any]:
    baseline_objective = safe_float(base_row.get("baseline_objective"))
    scenario_lp_gain = baseline_objective - optimized_objective if np.isfinite(optimized_objective) else np.nan
    return {
        "city": str(base_row["city"]),
        "event_id": int(base_row["event_id"]),
        "event_start": str(base_row["event_start"]),
        "budget_scale": float(budget_scale),
        "budget_key": budget_key(budget_scale),
        "n_units": int(float(base_row["n_units"])),
        "status": status,
        "runtime_seconds": runtime_seconds,
        "baseline_objective": baseline_objective,
        "scenario_optimized_objective": optimized_objective,
        "scenario_lp_gain": scenario_lp_gain,
        "scenario_lp_recoverable_fraction": scenario_lp_gain / baseline_objective if baseline_objective > EPS and np.isfinite(scenario_lp_gain) else np.nan,
        "scenario_total_budget": float(scenario_params.total_budget),
        "mean_period_budget": float(np.mean(scenario_params.period_budget)),
        "base_total_budget": safe_float(base_row.get("total_budget")),
        "base_recoverable_fraction": safe_float(base_row.get("recoverable_fraction")),
        "base_runtime_seconds": safe_float(base_row.get("runtime_seconds")),
        "base_residual_improvement": safe_float(base_row.get("residual_gain_improvement_over_static")),
        "event_peak_positive_abnormal_deficit": safe_float(base_row.get("event_peak_positive_abnormal_deficit")),
        "event_total_precip": safe_float(base_row.get("event_total_precip")),
        "error": error,
    }


def build_policy_vs_lp(optima: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    if optima.empty or replay.empty:
        return pd.DataFrame()
    opt = optima.copy()
    opt["event_id"] = pd.to_numeric(opt["event_id"], errors="coerce").astype(int)
    opt["budget_scale"] = pd.to_numeric(opt["budget_scale"], errors="coerce")
    rep = replay.copy()
    rep["event_id"] = pd.to_numeric(rep["event_id"], errors="coerce").astype(int)
    rep["budget_scale"] = pd.to_numeric(rep["budget_scale"], errors="coerce")
    rep = rep[rep["policy_score"].isin(POLICY_ORDER)].copy()
    keep_replay = [
        "city",
        "event_id",
        "budget_scale",
        "policy_score",
        "allocated_cost",
        "value_proxy",
        "selected_action_count",
        "replay_objective",
        "replay_gain",
        "replay_recoverable_fraction",
    ]
    keep_opt = [
        "city",
        "event_id",
        "event_start",
        "budget_scale",
        "n_units",
        "status",
        "runtime_seconds",
        "baseline_objective",
        "scenario_optimized_objective",
        "scenario_lp_gain",
        "scenario_lp_recoverable_fraction",
        "scenario_total_budget",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    out = opt[keep_opt].merge(rep[keep_replay], on=["city", "event_id", "budget_scale"], how="left")
    out = out.rename(columns={"status": "lp_status", "runtime_seconds": "lp_runtime_seconds"})
    out["fraction_of_scenario_lp_gain"] = out["replay_gain"] / out["scenario_lp_gain"].replace(0.0, np.nan)
    out["gap_to_scenario_lp_gain"] = 1.0 - out["fraction_of_scenario_lp_gain"]
    law = out[out["policy_score"].eq("activated_bottleneck_law")][["city", "event_id", "budget_scale", "replay_gain", "replay_recoverable_fraction"]].rename(
        columns={"replay_gain": "law_replay_gain", "replay_recoverable_fraction": "law_replay_recoverable_fraction"}
    )
    out = out.merge(law, on=["city", "event_id", "budget_scale"], how="left")
    out["law_minus_policy_replay_gain"] = out["law_replay_gain"] - out["replay_gain"]
    out["law_minus_policy_recoverable_fraction"] = out["law_replay_recoverable_fraction"] - out["replay_recoverable_fraction"]
    out["law_minus_policy_fraction_of_lp_gain"] = out["law_minus_policy_replay_gain"] / out["scenario_lp_gain"].replace(0.0, np.nan)
    return out.sort_values(["budget_scale", "city", "event_id", "policy_score"])


def build_summary(policy: pd.DataFrame, optima: pd.DataFrame) -> pd.DataFrame:
    if optima.empty:
        return pd.DataFrame()
    valid_opt = optima[optima["status"].astype(str).eq("OPTIMAL")].copy()
    if valid_opt.empty:
        valid_opt = optima[optima["scenario_lp_gain"].notna()].copy()
    rows: list[dict[str, Any]] = []
    valid_policy = policy[policy["lp_status"].astype(str).eq("OPTIMAL")].copy() if not policy.empty else pd.DataFrame()
    if valid_policy.empty and not policy.empty:
        valid_policy = policy.copy()
    for budget_scale, group in valid_opt.groupby("budget_scale", sort=True):
        row: dict[str, Any] = {
            "budget_scale": float(budget_scale),
            "n_lp_events": int(len(group)),
            "mean_lp_gain": safe_mean(group["scenario_lp_gain"]),
            "mean_lp_recoverable_fraction": safe_mean(group["scenario_lp_recoverable_fraction"]),
            "mean_lp_runtime_seconds": safe_mean(group["runtime_seconds"]),
        }
        pgroup = valid_policy[np.isclose(valid_policy["budget_scale"].astype(float), float(budget_scale))] if not valid_policy.empty else pd.DataFrame()
        for policy_score in POLICY_ORDER:
            pg = pgroup[pgroup["policy_score"].eq(policy_score)]
            row[f"mean_{policy_score}_fraction_of_lp_gain"] = safe_mean(pg["fraction_of_scenario_lp_gain"]) if not pg.empty else np.nan
            row[f"mean_{policy_score}_recoverable_fraction"] = safe_mean(pg["replay_recoverable_fraction"]) if not pg.empty else np.nan
        law = pgroup[pgroup["policy_score"].eq("activated_bottleneck_law")]
        random = pgroup[pgroup["policy_score"].eq("random_positive")]
        if not law.empty and not random.empty:
            merged = law[["city", "event_id", "replay_gain", "replay_recoverable_fraction", "scenario_lp_gain"]].merge(
                random[["city", "event_id", "replay_gain", "replay_recoverable_fraction"]],
                on=["city", "event_id"],
                suffixes=("_law", "_random"),
            )
            merged["law_minus_random_replay_gain"] = merged["replay_gain_law"] - merged["replay_gain_random"]
            merged["law_minus_random_recoverable_fraction"] = (
                merged["replay_recoverable_fraction_law"] - merged["replay_recoverable_fraction_random"]
            )
            merged["law_minus_random_fraction_of_lp_gain"] = merged["law_minus_random_replay_gain"] / merged["scenario_lp_gain"].replace(0.0, np.nan)
            row["mean_law_minus_random_replay_gain"] = safe_mean(merged["law_minus_random_replay_gain"])
            row["mean_law_minus_random_recoverable_fraction"] = safe_mean(merged["law_minus_random_recoverable_fraction"])
            row["mean_law_minus_random_fraction_of_lp_gain"] = safe_mean(merged["law_minus_random_fraction_of_lp_gain"])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("budget_scale").reset_index(drop=True)
    if not out.empty:
        out["mean_lp_gain_per_budget_scale"] = out["mean_lp_gain"] / out["budget_scale"].replace(0.0, np.nan)
        out["mean_law_minus_random_gain_per_budget_scale"] = out["mean_law_minus_random_replay_gain"] / out["budget_scale"].replace(0.0, np.nan)
        out = add_incremental_slopes(out)
    return out


def add_incremental_slopes(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    scales = out["budget_scale"].to_numpy(dtype=float)
    for col in [
        "mean_lp_gain",
        "mean_lp_recoverable_fraction",
        "mean_law_minus_random_replay_gain",
        "mean_law_minus_random_fraction_of_lp_gain",
    ]:
        if col not in out:
            continue
        values = out[col].to_numpy(dtype=float)
        slope = np.full(len(out), np.nan)
        if len(out) > 1:
            slope[1:] = np.diff(values) / np.maximum(np.diff(scales), EPS)
        out[f"incremental_{col}"] = slope
    return out


def build_phase_tests(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metrics = [
        "mean_lp_gain",
        "mean_lp_gain_per_budget_scale",
        "mean_lp_recoverable_fraction",
        "mean_law_minus_random_replay_gain",
        "mean_law_minus_random_gain_per_budget_scale",
        "mean_law_minus_random_fraction_of_lp_gain",
        "mean_activated_bottleneck_law_fraction_of_lp_gain",
        "mean_random_positive_fraction_of_lp_gain",
        "incremental_mean_lp_gain",
        "incremental_mean_law_minus_random_replay_gain",
    ]
    return pd.DataFrame([phase_row(summary, metric) for metric in metrics if metric in summary])


def phase_row(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    work = frame[["budget_scale", metric]].dropna().sort_values("budget_scale")
    if work.empty:
        return {"metric": metric}
    values = work[metric].to_numpy(dtype=float)
    scales = work["budget_scale"].to_numpy(dtype=float)
    peak_pos = int(np.nanargmax(values))
    diffs = np.diff(values)
    return {
        "metric": metric,
        "peak_budget_scale": float(scales[peak_pos]),
        "peak_value": float(values[peak_pos]),
        "first_budget_value": float(values[0]),
        "last_budget_value": float(values[-1]),
        "interior_peak_supported": bool(0 < peak_pos < len(values) - 1),
        "monotone_increasing": bool(np.all(diffs >= -1e-10)) if len(diffs) else False,
        "monotone_decreasing": bool(np.all(diffs <= 1e-10)) if len(diffs) else False,
        "first_to_peak_gain": float(values[peak_pos] - values[0]),
        "peak_to_last_change": float(values[-1] - values[peak_pos]),
    }


def build_city_summary(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    valid = policy[policy["lp_status"].astype(str).eq("OPTIMAL")].copy()
    if valid.empty:
        valid = policy.copy()
    rows: list[dict[str, Any]] = []
    for (city, budget_scale), group in valid.groupby(["city", "budget_scale"], sort=True):
        law = group[group["policy_score"].eq("activated_bottleneck_law")]
        random = group[group["policy_score"].eq("random_positive")]
        row = {
            "city": city,
            "budget_scale": float(budget_scale),
            "n_event_policy_rows": int(len(group)),
            "mean_lp_recoverable_fraction": safe_mean(group["scenario_lp_recoverable_fraction"]),
            "mean_law_fraction_of_lp_gain": safe_mean(law["fraction_of_scenario_lp_gain"]) if not law.empty else np.nan,
            "mean_random_fraction_of_lp_gain": safe_mean(random["fraction_of_scenario_lp_gain"]) if not random.empty else np.nan,
            "mean_law_recoverable_fraction": safe_mean(law["replay_recoverable_fraction"]) if not law.empty else np.nan,
            "mean_random_recoverable_fraction": safe_mean(random["replay_recoverable_fraction"]) if not random.empty else np.nan,
        }
        if not law.empty and not random.empty:
            law_row = law.iloc[0]
            random_row = random.iloc[0]
            row["law_minus_random_replay_gain"] = safe_float(law_row["replay_gain"]) - safe_float(random_row["replay_gain"])
            row["law_minus_random_fraction_of_lp_gain"] = row["mean_law_fraction_of_lp_gain"] - row["mean_random_fraction_of_lp_gain"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["city", "budget_scale"])


def build_metrics(optima: pd.DataFrame, policy: pd.DataFrame, summary: pd.DataFrame, phase: pd.DataFrame) -> dict[str, Any]:
    def get_phase(metric: str) -> pd.Series:
        match = phase[phase["metric"].eq(metric)] if not phase.empty and "metric" in phase else pd.DataFrame()
        return match.iloc[0] if not match.empty else pd.Series(dtype=object)

    optimal = optima[optima["status"].astype(str).eq("OPTIMAL")] if not optima.empty and "status" in optima else pd.DataFrame()
    law_phase = get_phase("mean_law_minus_random_replay_gain")
    law_fraction_phase = get_phase("mean_law_minus_random_fraction_of_lp_gain")
    lp_gain_phase = get_phase("mean_lp_gain")
    lp_per_budget_phase = get_phase("mean_lp_gain_per_budget_scale")
    base = nearest_budget_row(summary, 1.0)
    return {
        "n_selected_events": int(optima[["city", "event_id"]].drop_duplicates().shape[0]) if not optima.empty else 0,
        "n_budget_scales": int(optima["budget_scale"].nunique()) if not optima.empty else 0,
        "n_lp_jobs": int(len(optima)),
        "n_optimal_lp_jobs": int(len(optimal)),
        "lp_status_counts": optima["status"].astype(str).value_counts().to_dict() if not optima.empty and "status" in optima else {},
        "mean_lp_runtime_seconds": safe_mean(optima["runtime_seconds"]) if not optima.empty and "runtime_seconds" in optima else np.nan,
        "max_lp_runtime_seconds": safe_float(optima["runtime_seconds"].max()) if not optima.empty and "runtime_seconds" in optima else np.nan,
        "lp_gain_peak_budget": safe_float(lp_gain_phase.get("peak_budget_scale")),
        "lp_gain_interior_peak_supported": parse_bool(lp_gain_phase.get("interior_peak_supported")),
        "lp_gain_monotone_increasing": parse_bool(lp_gain_phase.get("monotone_increasing")),
        "lp_gain_per_budget_peak_budget": safe_float(lp_per_budget_phase.get("peak_budget_scale")),
        "lp_gain_per_budget_monotone_decreasing": parse_bool(lp_per_budget_phase.get("monotone_decreasing")),
        "law_random_abs_peak_budget": safe_float(law_phase.get("peak_budget_scale")),
        "law_random_abs_interior_peak_supported": parse_bool(law_phase.get("interior_peak_supported")),
        "law_random_abs_monotone_increasing": parse_bool(law_phase.get("monotone_increasing")),
        "law_random_fraction_peak_budget": safe_float(law_fraction_phase.get("peak_budget_scale")),
        "law_random_fraction_monotone_decreasing": parse_bool(law_fraction_phase.get("monotone_decreasing")),
        "base_budget_lp_recoverable_fraction": safe_float(base.get("mean_lp_recoverable_fraction")),
        "base_budget_law_fraction_of_lp_gain": safe_float(base.get("mean_activated_bottleneck_law_fraction_of_lp_gain")),
        "base_budget_random_fraction_of_lp_gain": safe_float(base.get("mean_random_positive_fraction_of_lp_gain")),
        "base_budget_law_minus_random_fraction_of_lp_gain": safe_float(base.get("mean_law_minus_random_fraction_of_lp_gain")),
        "base_budget_law_minus_random_replay_gain": safe_float(base.get("mean_law_minus_random_replay_gain")),
        "mean_law_fraction_of_lp_gain_all_budgets": safe_mean(
            policy.loc[policy["policy_score"].eq("activated_bottleneck_law"), "fraction_of_scenario_lp_gain"]
        )
        if not policy.empty
        else np.nan,
    }


def nearest_budget_row(summary: pd.DataFrame, budget_scale: float) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    idx = (summary["budget_scale"].astype(float) - budget_scale).abs().idxmin()
    return summary.loc[idx]


def make_figures(summary: pd.DataFrame, city_summary: pd.DataFrame, policy: pd.DataFrame, figure_dir: Path) -> None:
    if summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = summary["budget_scale"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(x, summary["mean_lp_recoverable_fraction"], marker="o", label="LP optimum")
    ax.plot(x, summary["mean_activated_bottleneck_law_recoverable_fraction"], marker="o", label="activated law replay")
    ax.plot(x, summary["mean_random_positive_recoverable_fraction"], marker="o", label="random-positive replay")
    ax.set_xlabel("Budget scale")
    ax.set_ylabel("Recoverable fraction")
    ax.set_title("Fine-budget representative LP closure")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fine_budget_lp_recoverability_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(x, summary["mean_activated_bottleneck_law_fraction_of_lp_gain"], marker="o", label="law / LP")
    ax.plot(x, summary["mean_random_positive_fraction_of_lp_gain"], marker="o", label="random / LP")
    ax.plot(x, summary["mean_law_minus_random_fraction_of_lp_gain"], marker="s", linestyle="--", label="law-random / LP")
    ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Budget scale")
    ax.set_ylabel("Fraction of scenario LP gain")
    ax.set_title("Policy closure against budget-specific LP optima")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fine_budget_policy_fraction_of_lp.png", dpi=180)
    plt.close(fig)

    if not city_summary.empty:
        pivot = city_summary.pivot_table(index="city", columns="budget_scale", values="mean_law_fraction_of_lp_gain")
        fig, ax = plt.subplots(figsize=(9.4, 5.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(1.0, np.nanmax(pivot.to_numpy(dtype=float))))
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{col:g}" for col in pivot.columns], rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_xlabel("Budget scale")
        ax.set_title("Activated law fraction of LP gain by city")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Law replay gain / LP gain")
        fig.tight_layout()
        fig.savefig(figure_dir / "fine_budget_lp_city_heatmap.png", dpi=180)
        plt.close(fig)

    if not policy.empty:
        law = policy[policy["policy_score"].eq("activated_bottleneck_law")].copy()
        random = policy[policy["policy_score"].eq("random_positive")].copy()
        pair = law[["city", "event_id", "budget_scale", "fraction_of_scenario_lp_gain"]].merge(
            random[["city", "event_id", "budget_scale", "fraction_of_scenario_lp_gain"]],
            on=["city", "event_id", "budget_scale"],
            suffixes=("_law", "_random"),
        )
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(6.8, 6.2))
            scatter = ax.scatter(
                pair["fraction_of_scenario_lp_gain_random"],
                pair["fraction_of_scenario_lp_gain_law"],
                c=pair["budget_scale"],
                cmap="viridis",
                s=62,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
            )
            low = max(0.0, float(np.nanmin(pair[["fraction_of_scenario_lp_gain_random", "fraction_of_scenario_lp_gain_law"]].to_numpy())) - 0.04)
            high = min(1.25, float(np.nanmax(pair[["fraction_of_scenario_lp_gain_random", "fraction_of_scenario_lp_gain_law"]].to_numpy())) + 0.04)
            ax.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1, alpha=0.45)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_xlabel("Random-positive / LP gain")
            ax.set_ylabel("Activated law / LP gain")
            ax.set_title("Law closure dominates random under budget-specific LP")
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Budget scale")
            fig.tight_layout()
            fig.savefig(figure_dir / "fine_budget_law_vs_random_lp_fraction.png", dpi=180)
            plt.close(fig)


def write_report(
    path: Path,
    selected_events: pd.DataFrame,
    optima: pd.DataFrame,
    policy: pd.DataFrame,
    summary: pd.DataFrame,
    phase: pd.DataFrame,
    city_summary: pd.DataFrame,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Fine-Budget LP Optimum Validation V28",
        "",
        "## 这一版做了什么",
        "",
        "V28 在 V27 的 fine budget proxy/replay sweep 之后，选择一组代表性 city-event，并在同一组预算尺度上重新求解 budget-specific full LP optimum。这样可以检查两件事：第一，V27 的 law/random replay 结论是否仍然接近完整优化上界；第二，预算曲线本身是否呈现绝对量增加但单位预算收益递减的形态。",
        "",
        "## 代表事件选择",
        "",
        f"- events per city: {args.events_per_city}",
        f"- max events: {args.max_events}",
        f"- base runtime guard: {args.max_reference_runtime_seconds:.0f} seconds",
        "",
        table_to_markdown(selected_events),
        "",
        "## 主要指标",
        "",
        f"- selected events: {metrics['n_selected_events']}",
        f"- budget scales: {metrics['n_budget_scales']}",
        f"- LP jobs: {metrics['n_lp_jobs']}; optimal LP jobs: {metrics['n_optimal_lp_jobs']}",
        f"- LP status counts: {metrics['lp_status_counts']}",
        f"- mean/max LP runtime seconds: {metrics['mean_lp_runtime_seconds']:.2f} / {metrics['max_lp_runtime_seconds']:.2f}",
        f"- LP gain peak budget: {metrics['lp_gain_peak_budget']:.2f}; interior peak supported = {metrics['lp_gain_interior_peak_supported']}; monotone increasing = {metrics['lp_gain_monotone_increasing']}",
        f"- LP gain per budget peak: {metrics['lp_gain_per_budget_peak_budget']:.2f}; monotone decreasing = {metrics['lp_gain_per_budget_monotone_decreasing']}",
        f"- law-random absolute replay gain peak budget: {metrics['law_random_abs_peak_budget']:.2f}; interior peak supported = {metrics['law_random_abs_interior_peak_supported']}",
        f"- law-random fraction-of-LP peak budget: {metrics['law_random_fraction_peak_budget']:.2f}; monotone decreasing = {metrics['law_random_fraction_monotone_decreasing']}",
        f"- base-budget LP recoverable fraction: {metrics['base_budget_lp_recoverable_fraction']:.4f}",
        f"- base-budget law / LP gain: {metrics['base_budget_law_fraction_of_lp_gain']:.4f}; random / LP gain: {metrics['base_budget_random_fraction_of_lp_gain']:.4f}",
        f"- base-budget law-random / LP gain: {metrics['base_budget_law_minus_random_fraction_of_lp_gain']:.4f}",
        f"- mean law / LP gain across all budget rows: {metrics['mean_law_fraction_of_lp_gain_all_budgets']:.4f}",
        "",
        "## 解释",
        "",
        "如果所有或几乎所有 LP jobs 都达到 OPTIMAL，那么这一版可以作为 V27 的更强预算闭合检验。这里的分母不再是 base LP gain 或 proxy value，而是每个代表事件、每个预算尺度重新求得的 scenario-specific LP gain。若 activated law 在这个分母下仍保持高 fraction-of-LP，同时 law-random 的相对优势在低预算更强，就说明 V27 的 diminishing leverage 不是单纯由 proxy normalization 造成的。",
        "",
        "## Budget Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Phase Tests",
        "",
        table_to_markdown(phase),
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    required = {"city", "event_id", "budget_scale", "status"}
    if existing.empty or not required.issubset(existing.columns):
        return set()
    valid = existing[existing["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    return {
        (str(row.city), int(row.event_id), budget_key(row.budget_scale))
        for row in valid[["city", "event_id", "budget_scale"]].itertuples(index=False)
    }


def deduplicate_optima(optima: pd.DataFrame) -> pd.DataFrame:
    if optima.empty:
        return optima
    out = optima.copy()
    out["_row_order"] = np.arange(len(out))
    out["_status_rank"] = out["status"].astype(str).map({"OPTIMAL": 3, "SUBOPTIMAL": 2, "TIME_LIMIT": 1}).fillna(0)
    out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype(int)
    out["budget_scale"] = pd.to_numeric(out["budget_scale"], errors="coerce")
    out["budget_key"] = out["budget_scale"].map(budget_key)
    out = (
        out.sort_values(["city", "event_id", "budget_scale", "_status_rank", "_row_order"])
        .groupby(["city", "event_id", "budget_key"], as_index=False)
        .tail(1)
        .sort_values(["city", "event_id", "budget_scale"])
        .drop(columns=["_row_order", "_status_rank"])
        .reset_index(drop=True)
    )
    return out


def budget_key(value: float) -> str:
    return f"{float(value):.6g}"


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if len(values.dropna()) else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return bool(value)
    except Exception:
        return False


def table_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_"
    compact = df.head(max_rows).copy()
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

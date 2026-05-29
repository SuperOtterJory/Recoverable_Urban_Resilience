"""Validate residual recovery laws against scenario-specific LP optima."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learn_recovery_laws import (
    POLICY_SCENARIOS,
    build_budget_segments,
    build_event_action_frame,
    load_inputs,
    prepare_interventions,
    replay_policy_allocations,
)
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_recovery_lp
from run_residual_greedy_policy import allocate_residual_greedy


EPS = 1e-12
DEFAULT_SCENARIOS = ("low_budget", "high_budget", "delay_4h", "scarce_and_late")
SCENARIO_ORDER = [str(row["policy_scenario"]) for row in POLICY_SCENARIOS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/scenario_optimum_validation")
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--events-per-city", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-reference-runtime-seconds", type=float, default=180.0)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--replan-budget-share", type=float, default=0.05)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    time_limit = args.time_limit_seconds
    if time_limit is None:
        time_limit = float(solver.get("time_limit_seconds", 300))
    method = int(args.method if args.method is not None else solver.get("method", -1))

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_events": table_dir / "selected_events.csv",
        "optima": table_dir / "scenario_lp_optima.csv",
        "policy": table_dir / "scenario_policy_validation.csv",
        "summary": table_dir / "scenario_policy_summary.csv",
        "city_summary": table_dir / "scenario_city_policy_summary.csv",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    data = load_inputs(root)
    reference_replay = pd.read_csv(root / "results" / "law_learning" / "tables" / "fixed_policy_replay.csv")
    residual_metrics = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv")
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"] == "OPTIMAL") & (base_summary["scenario"] == "base")].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype(int)

    selected_events = select_representative_events(
        residual_metrics,
        base_summary,
        events_per_city=int(args.events_per_city),
        max_reference_runtime_seconds=float(args.max_reference_runtime_seconds),
        max_events=args.max_events,
    )
    write_table(selected_events, paths["selected_events"])

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()
    scenarios = resolve_scenarios(args.scenarios)
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
        params = calibrate_observed_event_city(
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
        full = build_event_action_frame(params, base_row, event_row, event_interventions)
        segments = build_budget_segments(full, config, rng)

        for scenario in scenarios:
            scenario_name = str(scenario["policy_scenario"])
            budget_scale = float(scenario["budget_scale"])
            delay_add = int(scenario["delay_add_hours"])
            job_idx += 1
            key = (city, event_id, scenario_name)
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed {city} event {event_id} / {scenario_name}", flush=True)
                continue
            print(f"[{job_idx}/{total_jobs}] Solving scenario optimum {city} event {event_id} / {scenario_name}", flush=True)
            scenario_delays = {name: int(value) + delay_add for name, value in params.delays.items()}
            scenario_params = params.copy_with_budget(budget_scale, delays=scenario_delays)
            try:
                optimized = solve_recovery_lp(
                    scenario_params,
                    output_flag=bool(solver.get("output_flag", False)),
                    method=method,
                    time_limit_seconds=float(time_limit),
                )
                baseline_objective = float(base_row["baseline_objective"])
                scenario_lp_gain = max(baseline_objective - float(optimized.objective), EPS)
                opt_row = {
                    **job_metadata(base_row, scenario_name, budget_scale, delay_add, scenario_params),
                    "status": optimized.status,
                    "runtime_seconds": float(optimized.runtime_seconds),
                    "scenario_optimized_objective": float(optimized.objective),
                    "scenario_lp_recoverable_fraction": fraction_recovered(baseline_objective, float(optimized.objective)),
                    "scenario_lp_gain": float(scenario_lp_gain),
                    "base_lp_recoverable_fraction": float(base_row["recoverable_fraction"]),
                    "base_optimized_objective": float(base_row["optimized_objective"]),
                    "error": "",
                }
                append_csv(pd.DataFrame([opt_row]), paths["optima"])

                policy_rows = build_policy_rows(
                    full,
                    segments,
                    scenario_params,
                    reference_replay,
                    base_row,
                    scenario_name,
                    budget_scale,
                    delay_add,
                    optimized_objective=float(optimized.objective),
                    lp_status=str(optimized.status),
                    replan_budget_share=float(args.replan_budget_share),
                    max_replans=int(args.max_replans),
                )
                append_csv(pd.DataFrame(policy_rows), paths["policy"])
            except Exception as exc:  # pragma: no cover - batch diagnostics
                print(f"ERROR {city} event {event_id} / {scenario_name}: {exc}", flush=True)
                error = {
                    **job_metadata(base_row, scenario_name, budget_scale, delay_add, scenario_params),
                    "status": "ERROR",
                    "runtime_seconds": np.nan,
                    "scenario_optimized_objective": np.nan,
                    "scenario_lp_recoverable_fraction": np.nan,
                    "scenario_lp_gain": np.nan,
                    "base_lp_recoverable_fraction": float(base_row["recoverable_fraction"]),
                    "base_optimized_objective": float(base_row["optimized_objective"]),
                    "error": str(exc),
                }
                append_csv(pd.DataFrame([error]), paths["optima"])

    optima = pd.read_csv(paths["optima"]) if paths["optima"].exists() else pd.DataFrame()
    policy = pd.read_csv(paths["policy"]) if paths["policy"].exists() else pd.DataFrame()
    summary = summarize_policy(policy)
    city_summary = summarize_city_policy(policy)
    write_table(summary, paths["summary"])
    write_table(city_summary, paths["city_summary"])
    make_figures(policy, summary, figure_dir)
    write_report(
        report_dir / "scenario_optimum_validation_report_zh.md",
        selected_events,
        optima,
        policy,
        summary,
        city_summary,
        scenarios,
        float(args.max_reference_runtime_seconds),
    )
    print(f"Wrote scenario optimum validation to {output_dir}")


def resolve_scenarios(names: list[str]) -> list[dict[str, Any]]:
    available = {str(row["policy_scenario"]): dict(row) for row in POLICY_SCENARIOS}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown policy scenarios: {missing}. Available: {sorted(available)}")
    return [available[name] for name in names]


def select_representative_events(
    residual_metrics: pd.DataFrame,
    base_summary: pd.DataFrame,
    *,
    events_per_city: int,
    max_reference_runtime_seconds: float,
    max_events: int | None,
) -> pd.DataFrame:
    metrics = residual_metrics.copy()
    metrics["event_id"] = pd.to_numeric(metrics["event_id"], errors="coerce").astype(int)
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
    merged = metrics.merge(base_summary[base_cols], on=["city", "event_id", "event_start"], how="left", suffixes=("", "_base"))
    merged = merged.sort_values(["city", "residual_gain_improvement_over_static"], ascending=[True, False])
    merged["interaction_rank_in_city"] = merged.groupby("city")["residual_gain_improvement_over_static"].rank(
        method="first",
        ascending=False,
    )
    selected: list[pd.DataFrame] = []
    for city, city_rows in merged.groupby("city", sort=True):
        eligible = city_rows[city_rows["runtime_seconds"].fillna(np.inf) <= max_reference_runtime_seconds].copy()
        if eligible.empty:
            eligible = city_rows.head(max(events_per_city, 1)).copy()
            eligible["selection_note"] = "no_event_under_runtime_guard"
        else:
            eligible = eligible.head(max(events_per_city, 1)).copy()
            eligible["selection_note"] = "highest_interaction_gain_under_runtime_guard"
        selected.append(eligible)
    out = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    out = out.sort_values(["city", "interaction_rank_in_city", "event_start"]).reset_index(drop=True)
    if max_events is not None:
        out = out.head(max_events).copy()
    keep = [
        "city",
        "event_id",
        "event_start",
        "selection_note",
        "interaction_rank_in_city",
        "runtime_seconds",
        "n_units",
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


def job_metadata(
    base_row: pd.Series,
    scenario_name: str,
    budget_scale: float,
    delay_add: int,
    scenario_params: Any,
) -> dict[str, Any]:
    return {
        "city": str(base_row["city"]),
        "event_id": int(base_row["event_id"]),
        "event_start": str(base_row["event_start"]),
        "policy_scenario": scenario_name,
        "budget_scale": float(budget_scale),
        "delay_add_hours": int(delay_add),
        "n_units": int(base_row["n_units"]),
        "baseline_objective": float(base_row["baseline_objective"]),
        "scenario_total_budget": float(scenario_params.total_budget),
        "mean_period_budget": float(np.mean(scenario_params.period_budget)),
        "delay_R": int(scenario_params.delays.get("R", 0)),
        "delay_C": int(scenario_params.delays.get("C", 0)),
        "delay_S": int(scenario_params.delays.get("S", 0)),
        "event_peak_positive_abnormal_deficit": float(base_row["event_peak_positive_abnormal_deficit"]),
        "event_total_precip": float(base_row["event_total_precip"]),
        "base_runtime_seconds": float(base_row["runtime_seconds"]),
        "base_static_fraction_of_lp_gain": float(base_row["static_fraction_of_lp_gain"]),
        "base_residual_fraction_of_lp_gain": float(base_row["residual_fraction_of_lp_gain"]),
        "base_residual_improvement": float(base_row["residual_gain_improvement_over_static"]),
    }


def build_policy_rows(
    full: pd.DataFrame,
    segments: pd.DataFrame,
    scenario_params: Any,
    reference_replay: pd.DataFrame,
    base_row: pd.Series,
    scenario_name: str,
    budget_scale: float,
    delay_add: int,
    *,
    optimized_objective: float,
    lp_status: str,
    replan_budget_share: float,
    max_replans: int,
) -> list[dict[str, Any]]:
    city = str(base_row["city"])
    event_id = int(base_row["event_id"])
    baseline_objective = float(base_row["baseline_objective"])
    scenario_lp_gain = max(baseline_objective - optimized_objective, EPS)
    rows: list[dict[str, Any]] = []

    static = reference_replay[
        reference_replay["city"].astype(str).eq(city)
        & (pd.to_numeric(reference_replay["event_id"], errors="coerce").astype(int) == event_id)
        & reference_replay["policy_scenario"].astype(str).eq(scenario_name)
        & reference_replay["policy_score"].astype(str).eq("greedy_oracle")
    ]
    if not static.empty:
        static_row = static.iloc[0]
        rows.append(
            policy_result_row(
                base_row,
                scenario_name,
                budget_scale,
                delay_add,
                "static_small_signal_greedy",
                baseline_objective,
                optimized_objective,
                scenario_lp_gain,
                replay_objective=float(static_row["replay_objective"]),
                allocated_cost=float(static_row["allocated_cost"]),
                selected_action_count=int(static_row["selected_action_count"]),
                lp_status=lp_status,
            )
        )

    result = allocate_residual_greedy(
        segments,
        scenario_params,
        baseline_objective=baseline_objective,
        replan_budget_share=replan_budget_share,
        max_replans=max_replans,
    )
    replay = replay_policy_allocations(result["allocations"], scenario_params)
    rows.append(
        policy_result_row(
            base_row,
            scenario_name,
            budget_scale,
            delay_add,
            "residual_finite_greedy",
            baseline_objective,
            optimized_objective,
            scenario_lp_gain,
            replay_objective=float(replay["objective"]),
            allocated_cost=float(result["allocated_cost"]),
            selected_action_count=int(result["selected_action_count"]),
            lp_status=lp_status,
        )
    )
    return rows


def policy_result_row(
    base_row: pd.Series,
    scenario_name: str,
    budget_scale: float,
    delay_add: int,
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
        "policy_scenario": scenario_name,
        "budget_scale": float(budget_scale),
        "delay_add_hours": int(delay_add),
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


def fraction_recovered(baseline_objective: float, objective: float) -> float:
    return float(1.0 - objective / baseline_objective) if baseline_objective > EPS else np.nan


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "policy_scenario", "status"}.issubset(existing.columns):
        return set()
    valid = existing[existing["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    return {
        (str(row.city), int(row.event_id), str(row.policy_scenario))
        for row in valid[["city", "event_id", "policy_scenario"]].itertuples(index=False)
    }


def summarize_policy(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    return (
        policy.groupby(["policy_scenario", "budget_scale", "delay_add_hours", "policy"], as_index=False)
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
        .sort_values(["policy_scenario", "policy"])
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


def make_figures(policy: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    if policy.empty or summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [name for name in SCENARIO_ORDER if name in set(summary["policy_scenario"])]
    policies = ["static_small_signal_greedy", "residual_finite_greedy"]
    colors = {"static_small_signal_greedy": "#94a3b8", "residual_finite_greedy": "#2563eb"}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(order))
    width = 0.36
    for idx, policy_name in enumerate(policies):
        values = []
        for scenario in order:
            match = summary[(summary["policy_scenario"] == scenario) & (summary["policy"] == policy_name)]
            values.append(float(match["mean_fraction_of_scenario_lp_gain"].iloc[0]) if not match.empty else np.nan)
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=policy_name.replace("_", " "),
            color=colors[policy_name],
        )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("Policy gain / scenario LP gain")
    ax.set_title("Scenario-specific LP closure")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "scenario_policy_fraction_of_lp.png", dpi=180)
    plt.close(fig)

    pivot = policy.pivot_table(
        index=["city", "event_id", "policy_scenario"],
        columns="policy",
        values="fraction_of_scenario_lp_gain",
        aggfunc="first",
    ).reset_index()
    if set(policies).issubset(pivot.columns):
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        for scenario in order:
            subset = pivot[pivot["policy_scenario"] == scenario]
            ax.scatter(
                subset["static_small_signal_greedy"],
                subset["residual_finite_greedy"],
                s=58,
                alpha=0.78,
                label=scenario,
            )
        low = max(0.0, float(np.nanmin(pivot[policies].to_numpy(dtype=float))) - 0.04)
        high = min(1.15, float(np.nanmax(pivot[policies].to_numpy(dtype=float))) + 0.04)
        ax.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1, alpha=0.45)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Static small-signal / scenario LP")
        ax.set_ylabel("Residual finite greedy / scenario LP")
        ax.set_title("Residual replanning closes finite-budget gap")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_vs_static_scenario_lp.png", dpi=180)
        plt.close(fig)

    residual = policy[policy["policy"] == "residual_finite_greedy"].copy()
    residual = residual.sort_values("gap_to_scenario_lp_gain", ascending=False).head(18)
    if not residual.empty:
        labels = residual.apply(lambda row: f"{row['city']} {int(row['event_id'])} {row['policy_scenario']}", axis=1)
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        y = np.arange(len(residual))
        ax.barh(y, residual["gap_to_scenario_lp_gain"], color="#ef4444", alpha=0.78)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Residual policy gap to scenario LP gain")
        ax.set_title("Largest remaining closure gaps")
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_gap_to_scenario_lp.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    selected_events: pd.DataFrame,
    optima: pd.DataFrame,
    policy: pd.DataFrame,
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    max_reference_runtime_seconds: float,
) -> None:
    lines = [
        "# Scenario-Specific LP Optimum Validation V9",
        "",
        "## 这一版回答什么问题",
        "",
        "V8 已经证明 residual finite greedy 在预算和延迟扰动下稳定优于 static small-signal greedy，但 V8 的非 base 场景没有重新求解 Gurobi optimum。V9 补上这个闭合检验：对一组代表性 city-event，在 low/high budget、delay 和 scarce-and-late 场景下重新求 scenario-specific LP optimum，然后比较 static 与 residual law policy 能获得各自 optimum 的多少。",
        "",
        "为了控制求解成本，本版不是全量 105 事件闭合，而是每城优先选择 base residual-over-static improvement 最高、且 base LP runtime 不超过 "
        f"{max_reference_runtime_seconds:.0f} 秒的代表事件。这个设计的目的不是替代全量 robustness，而是先验证 V7/V8 的 finite-budget law 在真正非 base LP optimum 下是否仍然成立。",
        "",
        "## 代表事件",
        "",
        table_to_markdown(
            selected_events[
                [
                    "city",
                    "event_id",
                    "interaction_rank_in_city",
                    "runtime_seconds",
                    "static_fraction_of_lp_gain",
                    "residual_fraction_of_lp_gain",
                    "residual_gain_improvement_over_static",
                    "selection_note",
                ]
            ]
        ),
        "",
        "## 求解覆盖",
        "",
        f"- selected events: {len(selected_events)}",
        f"- scenarios: {', '.join(str(row['policy_scenario']) for row in scenarios)}",
        f"- LP jobs with returned rows: {len(optima)}",
    ]
    if not optima.empty:
        status_counts = optima["status"].astype(str).value_counts().to_dict()
        lines.append(f"- LP status counts: {status_counts}")
        if "runtime_seconds" in optima:
            lines.append(f"- mean LP runtime seconds: {optima['runtime_seconds'].mean():.2f}")
            lines.append(f"- max LP runtime seconds: {optima['runtime_seconds'].max():.2f}")
    if not summary.empty:
        lines.extend(
            [
                "",
                "## Policy vs Scenario LP Optimum",
                "",
                table_to_markdown(summary),
            ]
        )
    if not policy.empty:
        pivot = policy.pivot_table(
            index=["city", "event_id", "policy_scenario"],
            columns="policy",
            values="fraction_of_scenario_lp_gain",
            aggfunc="first",
        ).reset_index()
        if {"static_small_signal_greedy", "residual_finite_greedy"}.issubset(pivot.columns):
            pivot["residual_minus_static"] = pivot["residual_finite_greedy"] - pivot["static_small_signal_greedy"]
            lines.extend(
                [
                    "",
                    "## 关键闭合结论",
                    "",
                    f"- mean static / scenario LP gain: {pivot['static_small_signal_greedy'].mean():.4f}",
                    f"- mean residual / scenario LP gain: {pivot['residual_finite_greedy'].mean():.4f}",
                    f"- mean residual-minus-static: {pivot['residual_minus_static'].mean():.4f}",
                    f"- positive residual improvement share: {(pivot['residual_minus_static'] > 1e-6).mean():.4f}",
                    "",
                    "解释：这里的分母已经不再是 base LP gain，而是每个 budget/delay 场景重新求解得到的 scenario-specific LP gain。因此它比 V8 更直接地回答 residual finite-budget law 是否接近对应场景的真实优化上界。",
                ]
            )
    if not city_summary.empty:
        lines.extend(["", "## City Summary", "", table_to_markdown(city_summary)])
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "如果本版结果显示 residual law 在代表性非 base 场景中仍接近 scenario optimum，下一步就可以扩大到更多事件，或者转向提取更明确的 event-level decision-criticality law。若某些场景 residual gap 明显，则需要分析 gap 是否来自 period budget shadow price、R/C/S 互补关系，还是 LP 全局同时优化带来的剩余优势。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

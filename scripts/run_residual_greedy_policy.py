"""Validate an interaction-aware residual greedy policy for finite-budget recovery laws."""

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

from learn_recovery_laws import (
    INTERVENTIONS,
    add_relative_replay_metrics,
    build_budget_segments,
    build_event_action_frame,
    load_inputs,
    prepare_interventions,
    replay_policy_allocations,
    replay_row,
)
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/residual_greedy_policy")
    parser.add_argument("--replan-budget-share", type=float, default=0.05)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--max-events", type=int, default=None)
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
    replay_reference = pd.read_csv(root / "results" / "law_learning" / "tables" / "fixed_policy_replay.csv")
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    if args.max_events is not None:
        summary = summary.sort_values(["city", "event_start", "event_id"]).head(args.max_events).copy()

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()

    allocation_frames: list[pd.DataFrame] = []
    pass_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    total_events = len(summary)
    rng = np.random.default_rng(20260529)
    for idx, row in enumerate(summary.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{total_events}] Residual greedy for {city} event {event_id}", flush=True)
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
        full = build_event_action_frame(params, row, event_row, event_interventions)
        segments = build_budget_segments(full, config, rng)
        result = allocate_residual_greedy(
            segments,
            params,
            baseline_objective=float(row.baseline_objective),
            replan_budget_share=float(args.replan_budget_share),
            max_replans=int(args.max_replans),
        )
        allocations = result["allocations"]
        if not allocations.empty:
            allocation_frames.append(allocations)
        pass_rows.extend(result["pass_rows"])
        replay = replay_policy_allocations(allocations, params)
        replay_rows.append(
            replay_row(
                full,
                policy_scenario="base",
                budget_scale=1.0,
                delay_add_hours=0,
                policy_score="residual_finite_greedy",
                allocated_cost=float(result["allocated_cost"]),
                value_proxy=float(result["value_proxy"]),
                selected_action_count=int(result["selected_action_count"]),
                replay_objective=replay["objective"],
                baseline_objective=float(row.baseline_objective),
                optimized_objective=float(row.optimized_objective),
                lp_recoverable_fraction=float(row.recoverable_fraction),
            )
        )

    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    residual_replay = pd.DataFrame(replay_rows)
    combined_replay = combine_replay_reference(replay_reference, residual_replay)
    event_metrics = build_event_metrics(combined_replay)
    city_summary = build_city_summary(event_metrics)
    pass_summary = summarize_passes(pd.DataFrame(pass_rows))

    write_table(allocations, table_dir / "residual_greedy_allocations.csv.gz")
    write_table(pd.DataFrame(pass_rows), table_dir / "residual_greedy_passes.csv")
    write_table(residual_replay, table_dir / "residual_greedy_replay.csv")
    write_table(event_metrics, table_dir / "residual_greedy_event_metrics.csv")
    write_table(city_summary, table_dir / "residual_greedy_city_summary.csv")
    write_table(pass_summary, table_dir / "residual_greedy_pass_summary.csv")

    make_figures(event_metrics, city_summary, pass_summary, figure_dir)
    write_report(report_dir / "residual_greedy_policy_report_zh.md", event_metrics, city_summary, pass_summary)
    print(f"Wrote residual greedy validation to {output_dir}")


def allocate_residual_greedy(
    segments: pd.DataFrame,
    params: Any,
    *,
    baseline_objective: float,
    replan_budget_share: float,
    max_replans: int,
) -> dict[str, Any]:
    if segments.empty or params.total_budget <= EPS:
        return empty_result()
    work = segments.copy().reset_index(drop=True)
    unit_to_idx = {unit: idx for idx, unit in enumerate(params.units)}
    work["unit_idx"] = work["unit"].astype(str).map(unit_to_idx)
    work = work.dropna(subset=["unit_idx"]).copy().reset_index(drop=True)
    if work.empty:
        return empty_result()
    work["unit_idx"] = work["unit_idx"].astype(int)
    unit_idx = work["unit_idx"].to_numpy(dtype=int)
    t_arr = work["t"].to_numpy(dtype=int)
    intervention_arr = work["intervention"].astype(str).to_numpy()
    cost = work["cost"].to_numpy(dtype=float)
    multipliers = work["segment_effectiveness_multiplier"].to_numpy(dtype=float)
    remaining_segment_cost = work["segment_cost_cap"].to_numpy(dtype=float).copy()
    remaining_segment_cost = np.maximum(remaining_segment_cost, 0.0)
    delay_feasible = np.zeros(len(work), dtype=bool)
    for intervention in INTERVENTIONS:
        delay = int(params.delays.get(intervention, 0))
        delay_feasible |= (intervention_arr == intervention) & (t_arr >= delay)

    remaining_total = float(params.total_budget)
    remaining_period = np.asarray(params.period_budget, dtype=float).copy()
    batch_budget = max(float(params.total_budget) * max(replan_budget_share, 1e-4), 1e-9)
    effects = {key: np.zeros((params.n_units, params.horizon), dtype=float) for key in INTERVENTIONS}
    allocations: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    value_proxy = 0.0
    allocated_cost = 0.0
    selected_actions: set[tuple[str, int, str]] = set()

    for pass_id in range(max_replans):
        if remaining_total <= EPS or np.all(remaining_period <= EPS):
            break
        states = simulate_states(params, effects)
        scores = score_segments_from_residual_state(
            params,
            states,
            work,
            unit_idx,
            t_arr,
            intervention_arr,
            cost,
            multipliers,
            remaining_segment_cost,
            remaining_period,
            remaining_total,
            baseline_objective,
        )
        valid = (
            delay_feasible
            & (remaining_segment_cost > EPS)
            & np.isfinite(scores)
            & (scores > EPS)
            & (remaining_period[np.clip(t_arr, 0, params.horizon - 1)] > EPS)
        )
        if not valid.any():
            break
        order = np.flatnonzero(valid)[np.argsort(scores[valid])[::-1]]
        pass_budget = min(batch_budget, remaining_total)
        pass_allocated = 0.0
        pass_value = 0.0
        pass_actions = 0
        for pos in order:
            t = int(t_arr[pos])
            if t < 0 or t >= params.horizon:
                continue
            available = min(float(remaining_segment_cost[pos]), remaining_total, float(remaining_period[t]), pass_budget - pass_allocated)
            if available <= EPS:
                continue
            intervention = str(intervention_arr[pos])
            i = int(unit_idx[pos])
            allocated_u = available / max(float(cost[pos]), EPS)
            effect = float(params.eta[intervention][i, t]) * float(multipliers[pos]) * allocated_u
            effects[intervention][i, t] += effect
            remaining_segment_cost[pos] -= available
            remaining_total -= available
            remaining_period[t] -= available
            value = available * float(scores[pos])
            row = work.iloc[int(pos)]
            allocations.append(
                {
                    "city": row["city"],
                    "event_id": int(row["event_id"]),
                    "event_start": row["event_start"],
                    "scenario": row["scenario"],
                    "unit": str(row["unit"]),
                    "t": t,
                    "intervention": intervention,
                    "segment": int(row["segment"]),
                    "allocated_cost": float(available),
                    "allocated_u": float(allocated_u),
                    "value_proxy": float(value),
                    "oracle_value_per_cost": float(scores[pos]),
                    "law_value_score": float(row.get("law_value_score", np.nan)),
                    "segment_effectiveness_multiplier": float(multipliers[pos]),
                    "residual_pass": int(pass_id),
                    "remaining_total_after": float(remaining_total),
                }
            )
            selected_actions.add((str(row["unit"]), t, intervention))
            allocated_cost += available
            value_proxy += value
            pass_allocated += available
            pass_value += value
            pass_actions += 1
            if remaining_total <= EPS or pass_allocated >= pass_budget - EPS:
                break
        pass_rows.append(
            {
                "city": str(work["city"].iloc[0]),
                "event_id": int(work["event_id"].iloc[0]),
                "pass_id": int(pass_id),
                "allocated_cost": float(pass_allocated),
                "value_proxy": float(pass_value),
                "selected_segment_count": int(pass_actions),
                "remaining_total": float(remaining_total),
                "objective_before_pass": float(states["objective"]),
            }
        )
        if pass_allocated <= EPS:
            break
    return {
        "allocated_cost": allocated_cost,
        "value_proxy": value_proxy,
        "selected_action_count": len(selected_actions),
        "allocations": pd.DataFrame(allocations),
        "pass_rows": pass_rows,
    }


def simulate_states(params: Any, effects: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    n = params.n_units
    horizon = params.horizon
    b = np.zeros((n, horizon + 1), dtype=float)
    r_c = np.zeros((n, horizon + 1), dtype=float)
    r_s = np.zeros((n, horizon + 1), dtype=float)
    d = np.zeros((n, horizon + 1), dtype=float)
    ell = np.zeros((n, horizon + 1), dtype=float)
    b[:, 0] = params.b0
    objective = 0.0
    for t in range(horizon + 1):
        d[:, t] = np.clip(b[:, t] - r_c[:, t], 0.0, 1.0)
        ell[:, t] = np.clip(params.q @ d[:, t] - r_s[:, t], 0.0, 1.0)
        objective += float(params.delta_t * np.sum(params.p * ell[:, t]))
        if t == horizon:
            break
        b[:, t + 1] = np.clip(params.a * b[:, t] + params.h[:, t + 1] - effects["R"][:, t], 0.0, 1.0)
        r_c[:, t + 1] = np.clip((1.0 - params.delta_c) * r_c[:, t] + effects["C"][:, t], 0.0, 1.0)
        r_s[:, t + 1] = np.clip((1.0 - params.delta_s) * r_s[:, t] + effects["S"][:, t], 0.0, 1.0)
    return {"b": b, "r_c": r_c, "r_s": r_s, "d": d, "ell": ell, "objective": objective}


def score_segments_from_residual_state(
    params: Any,
    states: dict[str, np.ndarray | float],
    work: pd.DataFrame,
    unit_idx: np.ndarray,
    t_arr: np.ndarray,
    intervention_arr: np.ndarray,
    cost: np.ndarray,
    multipliers: np.ndarray,
    remaining_segment_cost: np.ndarray,
    remaining_period: np.ndarray,
    remaining_total: float,
    baseline_objective: float,
) -> np.ndarray:
    del work
    n = params.n_units
    horizon = params.horizon
    d = np.asarray(states["d"], dtype=float)
    ell = np.asarray(states["ell"], dtype=float)
    active_access = ell > 1e-10
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    active_destination_importance = np.zeros((n, horizon + 1), dtype=float)
    for future_t in range(horizon + 1):
        active_origin_weight = params.p * active_access[:, future_t].astype(float)
        active_destination_importance[:, future_t] = np.asarray(q.T @ active_origin_weight).ravel()

    scores = np.zeros(len(unit_idx), dtype=float)
    period_available = remaining_period[np.clip(t_arr, 0, horizon - 1)]
    available_cost_all = np.minimum.reduce(
        [
            np.maximum(remaining_segment_cost, 0.0),
            np.maximum(period_available, 0.0),
            np.full(len(unit_idx), max(float(remaining_total), 0.0), dtype=float),
        ]
    )
    for intervention in INTERVENTIONS:
        intervention_mask = intervention_arr == intervention
        for t in range(horizon):
            positions = np.flatnonzero(intervention_mask & (t_arr == t) & (available_cost_all > EPS))
            if len(positions) == 0:
                continue
            i = unit_idx[positions]
            available_cost = available_cost_all[positions]
            effect = (
                params.eta[intervention][i, t]
                * multipliers[positions]
                * available_cost
                / np.maximum(cost[positions], EPS)
            )
            gain = np.zeros(len(positions), dtype=float)
            for offset, future_t in enumerate(range(t + 1, horizon + 1)):
                if intervention == "R":
                    decay = params.a[i] ** offset
                    reducible = d[i, future_t]
                    weight = active_destination_importance[i, future_t]
                elif intervention == "C":
                    decay = (1.0 - params.delta_c) ** offset
                    reducible = d[i, future_t]
                    weight = active_destination_importance[i, future_t]
                else:
                    decay = (1.0 - params.delta_s) ** offset
                    reducible = ell[i, future_t]
                    weight = params.p[i]
                gain += weight * np.minimum(effect * decay, reducible)
            scores[positions] = gain / np.maximum(baseline_objective * available_cost, EPS)
    return scores


def combine_replay_reference(reference: pd.DataFrame, residual_replay: pd.DataFrame) -> pd.DataFrame:
    base = reference[
        reference["policy_scenario"].eq("base")
        & reference["policy_score"].isin(["lp_optimizer_replay", "greedy_oracle", "activated_bottleneck_law", "exposure_only", "deficit_only"])
    ].copy()
    derived_cols = [
        "greedy_replay_gain",
        "greedy_replay_recoverable_fraction",
        "relative_to_greedy_replay_gain",
        "replay_recoverable_gap_to_lp",
    ]
    base = base.drop(columns=[col for col in derived_cols if col in base.columns])
    residual_replay = residual_replay.drop(columns=[col for col in derived_cols if col in residual_replay.columns])
    combined = pd.concat([base, residual_replay], ignore_index=True, sort=False)
    return add_relative_replay_metrics(combined)


def build_event_metrics(combined_replay: pd.DataFrame) -> pd.DataFrame:
    base = combined_replay[combined_replay["policy_scenario"].eq("base")].copy()
    rows = []
    for key, group in base.groupby(["city", "event_id", "event_start"]):
        lp = group[group["policy_score"].eq("lp_optimizer_replay")]
        static = group[group["policy_score"].eq("greedy_oracle")]
        residual = group[group["policy_score"].eq("residual_finite_greedy")]
        if lp.empty or static.empty or residual.empty:
            continue
        lp_row = lp.iloc[0]
        static_row = static.iloc[0]
        residual_row = residual.iloc[0]
        base_lp_gain = max(float(lp_row["replay_gain"]), EPS)
        static_gain = float(static_row["replay_gain"])
        residual_gain = float(residual_row["replay_gain"])
        rows.append(
            {
                "city": key[0],
                "event_id": int(key[1]),
                "event_start": key[2],
                "baseline_objective": float(lp_row["baseline_objective"]),
                "lp_recoverable_fraction": float(lp_row["replay_recoverable_fraction"]),
                "static_greedy_recoverable_fraction": float(static_row["replay_recoverable_fraction"]),
                "residual_greedy_recoverable_fraction": float(residual_row["replay_recoverable_fraction"]),
                "static_fraction_of_lp_gain": float(static_gain / base_lp_gain),
                "residual_fraction_of_lp_gain": float(residual_gain / base_lp_gain),
                "residual_gain_improvement_over_static": float((residual_gain - static_gain) / base_lp_gain),
                "residual_gap_to_lp": float(1.0 - residual_gain / base_lp_gain),
                "static_gap_to_lp": float(1.0 - static_gain / base_lp_gain),
                "static_selected_action_count": int(static_row["selected_action_count"]),
                "residual_selected_action_count": int(residual_row["selected_action_count"]),
                "residual_allocated_cost": float(residual_row["allocated_cost"]),
                "event_peak_positive_abnormal_deficit": float(lp_row["event_peak_positive_abnormal_deficit"]),
                "event_total_precip": float(lp_row["event_total_precip"]),
            }
        )
    return pd.DataFrame(rows).sort_values("residual_gain_improvement_over_static", ascending=False)


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        event_metrics.groupby("city", as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_static_fraction_of_lp_gain=("static_fraction_of_lp_gain", "mean"),
            mean_residual_fraction_of_lp_gain=("residual_fraction_of_lp_gain", "mean"),
            mean_residual_gain_improvement=("residual_gain_improvement_over_static", "mean"),
            median_residual_gain_improvement=("residual_gain_improvement_over_static", "median"),
            mean_residual_gap_to_lp=("residual_gap_to_lp", "mean"),
            mean_static_selected_actions=("static_selected_action_count", "mean"),
            mean_residual_selected_actions=("residual_selected_action_count", "mean"),
        )
        .sort_values("mean_residual_gain_improvement", ascending=False)
    )


def summarize_passes(passes: pd.DataFrame) -> pd.DataFrame:
    if passes.empty:
        return pd.DataFrame()
    return (
        passes.groupby(["city", "event_id"], as_index=False)
        .agg(
            n_replans=("pass_id", "count"),
            mean_pass_cost=("allocated_cost", "mean"),
            max_pass_id=("pass_id", "max"),
            final_remaining_total=("remaining_total", "last"),
        )
        .sort_values(["city", "event_id"])
    )


def make_figures(event_metrics: pd.DataFrame, city_summary: pd.DataFrame, pass_summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(
        event_metrics["static_fraction_of_lp_gain"],
        event_metrics["residual_fraction_of_lp_gain"],
        c=event_metrics["residual_gain_improvement_over_static"],
        cmap="viridis",
        s=58,
        alpha=0.82,
    )
    low = min(event_metrics["static_fraction_of_lp_gain"].min(), event_metrics["residual_fraction_of_lp_gain"].min(), 0.0)
    ax.plot([low, 1.02], [low, 1.02], color="#111827", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("Static small-signal greedy gain / LP gain")
    ax.set_ylabel("Residual finite greedy gain / LP gain")
    ax.set_title("Residual replanning versus static one-pass ranking")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_vs_static_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ordered = city_summary.sort_values("mean_residual_fraction_of_lp_gain")
    y = np.arange(len(ordered))
    ax.barh(y - 0.18, ordered["mean_static_fraction_of_lp_gain"], height=0.36, label="static", color="#94a3b8")
    ax.barh(y + 0.18, ordered["mean_residual_fraction_of_lp_gain"], height=0.36, label="residual", color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["city"])
    ax.axvline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("Mean replay gain / LP gain")
    ax.set_title("Residual greedy improvement by city")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_improvement_by_city.png", dpi=180)
    plt.close(fig)

    top = event_metrics.sort_values("residual_gain_improvement_over_static", ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels = top["city"] + " #" + top["event_id"].astype(str)
    ax.bar(labels, top["residual_gain_improvement_over_static"], color="#0f766e")
    ax.set_ylabel("Residual improvement over static greedy\n(fraction of LP gain)")
    ax.set_title("Events where residual replanning helps most")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "top_residual_improvements.png", dpi=180)
    plt.close(fig)

    if not pass_summary.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.hist(pass_summary["n_replans"], bins=20, color="#7c3aed", alpha=0.85)
        ax.set_xlabel("Number of residual replans")
        ax.set_ylabel("Event count")
        ax.set_title("Residual greedy replan count")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_replan_count.png", dpi=180)
        plt.close(fig)


def write_report(path: Path, event_metrics: pd.DataFrame, city_summary: pd.DataFrame, pass_summary: pd.DataFrame) -> None:
    mean_static = float(event_metrics["static_fraction_of_lp_gain"].mean())
    mean_residual = float(event_metrics["residual_fraction_of_lp_gain"].mean())
    median_residual = float(event_metrics["residual_fraction_of_lp_gain"].median())
    mean_improvement = float(event_metrics["residual_gain_improvement_over_static"].mean())
    positive_share = float((event_metrics["residual_gain_improvement_over_static"] > 1e-6).mean())
    mean_gap = float(event_metrics["residual_gap_to_lp"].mean())
    mean_replans = float(pass_summary["n_replans"].mean()) if not pass_summary.empty else np.nan
    top_events = event_metrics.head(15)[
        [
            "city",
            "event_id",
            "static_fraction_of_lp_gain",
            "residual_fraction_of_lp_gain",
            "residual_gain_improvement_over_static",
            "residual_gap_to_lp",
            "static_selected_action_count",
            "residual_selected_action_count",
        ]
    ]
    lines = [
        "# Residual Greedy Policy V7",
        "",
        "## 这一版做了什么",
        "",
        "V6 说明 finite-budget gap 主要来自一阶 small-signal 排序在完整预算下的局部饱和和动作交互。V7 因此实现了 residual finite greedy：不是一次性按 passive trajectory 的一阶值排序，而是把预算分成若干 replan pass；每一轮先 replay 当前已分配资源，得到 residual `b/rC/rS/ell`，再用 `min(candidate_effect, remaining_loss)` 估计下一段资源的有限段平均边际值。",
        "",
        "这个 policy 仍然是解释性 law，不是重新求解 LP。它的核心思想是：",
        "",
        "```text",
        "finite_budget_value(segment | current_state)",
        "  ~= sum_future exposure_active",
        "      * min(segment_effect_decay, residual_loss)",
        "      / segment_cost",
        "```",
        "",
        "## 主要结果",
        "",
        f"- static small-signal greedy 平均获得 LP gain 的 {mean_static:.4f}",
        f"- residual finite greedy 平均获得 LP gain 的 {mean_residual:.4f}，中位数 {median_residual:.4f}",
        f"- residual 相比 static 的平均提升为 LP gain 的 {mean_improvement:.4f}",
        f"- residual 有正提升的事件比例为 {positive_share:.4f}",
        f"- residual 剩余 gap 为 LP gain 的 {mean_gap:.4f}",
        f"- 平均 replan 次数为 {mean_replans:.2f}",
        "",
        "## 城市层面",
        "",
        dataframe_to_markdown(city_summary),
        "",
        "## 提升最大的事件",
        "",
        dataframe_to_markdown(top_events),
        "",
        "## 解释",
        "",
        "如果 residual greedy 明显高于 static greedy，说明 V6 的判断成立：完整预算 law 需要显式考虑 residual state 和有限段截断，而不能只依赖 first-order action score。如果提升有限，则说明 LP 的剩余优势更多来自全局同时优化、period budget shadow price 或 R/C/S 联合互补，而不是简单 residual re-ranking。",
        "",
        "下一步可以把 residual score 中的 shadow-price 信息进一步显式化：估计每个小时 period budget 的机会成本，或用 LP dual/shadow price 直接学习 finite-budget allocation law。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def empty_result() -> dict[str, Any]:
    return {
        "allocated_cost": 0.0,
        "value_proxy": 0.0,
        "selected_action_count": 0,
        "allocations": pd.DataFrame(),
        "pass_rows": [],
    }


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    df.to_csv(path, index=False, float_format="%.10g", compression=compression)


if __name__ == "__main__":
    main()

"""Validate footprint-aware tie-breakers inside residual finite-budget policies.

V38 tested footprint-aware scores as one-pass finite-budget replay policies.
This script asks the stricter V39 question: when the strongest current
finite-budget law re-scores actions against the residual state after each
deployment pass, can observed event footprints still improve spatial coverage
without sacrificing recovery gain?
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

from analyze_footprint_aware_law_frontier import add_score_columns
from analyze_hybrid_absorption_mechanisms import footprint_weights, hybrid_summary_row, prepare_interventions_with_caps
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from learn_recovery_laws import build_event_action_frame, replay_optimizer_solution, replay_policy_allocations
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS
from run_residual_greedy_policy import score_segments_from_residual_state, simulate_states
from validate_footprint_aware_policy import build_policy_segments, policy_row


EPS = 1e-12
POLICY_SPECS: list[dict[str, Any]] = [
    {"policy_score": "residual_finite_greedy", "footprint_lambda": 0.0, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_plus_0p1_footprint", "footprint_lambda": 0.1, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_plus_0p25_footprint", "footprint_lambda": 0.25, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_plus_0p5_footprint", "footprint_lambda": 0.5, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_plus_1p0_footprint", "footprint_lambda": 1.0, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_plus_2p0_footprint", "footprint_lambda": 2.0, "finite_lambda": 0.0, "mode": "additive"},
    {"policy_score": "residual_x_footprint_rank", "footprint_lambda": 1.0, "finite_lambda": 0.0, "mode": "multiply_footprint"},
    {"policy_score": "residual_plus_0p25_finite", "footprint_lambda": 0.0, "finite_lambda": 0.25, "mode": "additive"},
    {"policy_score": "residual_plus_0p5_finite", "footprint_lambda": 0.0, "finite_lambda": 0.5, "mode": "additive"},
    {"policy_score": "residual_plus_1p0_finite", "footprint_lambda": 0.0, "finite_lambda": 1.0, "mode": "additive"},
    {"policy_score": "residual_x_finite_rank", "footprint_lambda": 0.0, "finite_lambda": 1.0, "mode": "multiply_finite"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/residual_footprint_policy_validation")
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
    parser.add_argument("--replan-budget-share", type=float, default=0.05)
    parser.add_argument("--max-replans", type=int, default=80)
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
    lp_dir = root / args.hybrid_lp_dir / "tables"
    event_metrics = pd.read_csv(lp_dir / "hybrid_lp_event_metrics.csv", parse_dates=["event_start"])
    selected_events = pd.read_csv(lp_dir / "hybrid_lp_selected_events.csv", parse_dates=["event_start"])
    hybrid_selected_actions = pd.read_csv(lp_dir / "hybrid_lp_selected_actions.csv")
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"].eq("OPTIMAL")) & (base_summary["scenario"].eq("base"))].copy()

    for frame in [event_metrics, selected_events, hybrid_selected_actions, base_summary, data["events"], data["footprint_zone"]]:
        if "event_id" in frame:
            frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")

    event_lookup = {
        (row.city, int(row.event_id)): row
        for row in data["events"].dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    base_summary_lookup = {
        (row.city, int(row.event_id)): row
        for row in base_summary.dropna(subset=["event_id"]).itertuples(index=False)
    }
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
    hybrid_prepared = prepare_interventions_with_caps(hybrid_selected_actions, scenario="base")

    policy_rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    pass_frames: list[pd.DataFrame] = []
    ok_events = event_metrics[
        event_metrics["hybrid_status"].astype(str).eq("OPTIMAL")
        & event_metrics["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)

    for idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{len(ok_events)}] Residual footprint replay for {city} event {event_id}", flush=True)
        event_row = event_lookup[(city, event_id)]
        base_row = base_summary_lookup[(city, event_id)]
        selected_row = selected_lookup[(city, event_id)]
        footprint_group = footprint_groups[(city, event_id)]

        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=data["abnormal"],
            root=root,
        )
        hybrid_params, _ = build_hybrid_params(
            base_params,
            footprint_group,
            footprint_blend=float(args.footprint_blend),
            footprint_floor=float(args.footprint_floor),
            max_relative=float(args.max_relative),
        )
        hybrid_summary = hybrid_summary_row(row, base_row, hybrid_params, event_row)
        hybrid_actions = hybrid_prepared[
            hybrid_prepared["city"].astype(str).eq(city)
            & hybrid_prepared["event_id"].eq(event_id)
            & hybrid_prepared["scenario"].astype(str).eq("base")
        ].copy()
        full = build_event_action_frame(hybrid_params, hybrid_summary, event_row, hybrid_actions)
        weights = footprint_weights(footprint_group, hybrid_params.units)
        full = add_score_columns(full, weights)
        segments = build_policy_segments(full, config, score_ids=[])
        baseline_objective = no_intervention_objective(hybrid_params)
        optimized_objective = float(getattr(row, "hybrid_optimized_objective"))
        hybrid_lp_gain = baseline_objective - optimized_objective
        optimizer_replay = replay_optimizer_solution(full, hybrid_params)
        policy_rows.append(
            policy_row(
                city,
                event_id,
                row,
                "hybrid_lp_replay",
                allocations=pd.DataFrame(),
                replay_objective=float(optimizer_replay["objective"]),
                baseline_objective=baseline_objective,
                optimized_objective=optimized_objective,
                hybrid_lp_gain=hybrid_lp_gain,
                weights=weights,
                full=full,
                v34_delta=safe_float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)),
                v35_delta=safe_float(getattr(row, "delta_selected_unit_footprint_mass", np.nan)),
            )
        )

        for spec in POLICY_SPECS:
            result = allocate_adaptive_residual(
                segments,
                hybrid_params,
                baseline_objective=baseline_objective,
                replan_budget_share=float(args.replan_budget_share),
                max_replans=int(args.max_replans),
                footprint_lambda=float(spec["footprint_lambda"]),
                finite_lambda=float(spec["finite_lambda"]),
                mode=str(spec["mode"]),
            )
            allocations = result["allocations"]
            if not allocations.empty:
                allocations = allocations.copy()
                allocations["policy_score"] = str(spec["policy_score"])
                allocation_frames.append(allocations)
            passes = pd.DataFrame(result["pass_rows"])
            if not passes.empty:
                passes["policy_score"] = str(spec["policy_score"])
                pass_frames.append(passes)
            replay = replay_policy_allocations(allocations, hybrid_params)
            policy_rows.append(
                policy_row(
                    city,
                    event_id,
                    row,
                    str(spec["policy_score"]),
                    allocations=allocations,
                    replay_objective=float(replay["objective"]),
                    baseline_objective=baseline_objective,
                    optimized_objective=optimized_objective,
                    hybrid_lp_gain=hybrid_lp_gain,
                    weights=weights,
                    full=full,
                    v34_delta=safe_float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)),
                    v35_delta=safe_float(getattr(row, "delta_selected_unit_footprint_mass", np.nan)),
                )
            )

    policy = pd.DataFrame(policy_rows)
    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    passes = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
    summary = build_policy_summary(policy)
    event_delta = build_event_delta(policy)
    metrics = build_metrics(policy, summary)

    write_table(policy, table_dir / "residual_footprint_policy_event_metrics.csv")
    write_table(summary, table_dir / "residual_footprint_policy_summary.csv")
    write_table(event_delta, table_dir / "residual_footprint_policy_event_delta.csv")
    write_table(passes, table_dir / "residual_footprint_policy_passes.csv")
    write_table(allocations, table_dir / "residual_footprint_policy_allocations.csv.gz")
    (table_dir / "residual_footprint_policy_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(summary, event_delta, policy, figure_dir)
    write_report(report_dir / "residual_footprint_policy_validation_report_zh.md", metrics, summary, event_delta)
    print(f"Wrote residual footprint policy validation to {output_dir}")


def allocate_adaptive_residual(
    segments: pd.DataFrame,
    params: Any,
    *,
    baseline_objective: float,
    replan_budget_share: float,
    max_replans: int,
    footprint_lambda: float,
    finite_lambda: float,
    mode: str,
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
    footprint_rank = pd.to_numeric(work["footprint_unit_rank"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    finite_rank = pd.to_numeric(work["finite_unit_rank"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    remaining_segment_cost = np.maximum(work["segment_cost_cap"].to_numpy(dtype=float), 0.0)
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
    policy_value_proxy = 0.0
    allocated_cost = 0.0
    selected_actions: set[tuple[str, int, str]] = set()

    for pass_id in range(max_replans):
        if remaining_total <= EPS or np.all(remaining_period <= EPS):
            break
        states = simulate_states(params, effects)
        residual_scores = score_segments_from_residual_state(
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
        policy_scores = combine_residual_scores(
            residual_scores,
            footprint_rank,
            finite_rank,
            footprint_lambda=footprint_lambda,
            finite_lambda=finite_lambda,
            mode=mode,
        )
        valid = (
            delay_feasible
            & (remaining_segment_cost > EPS)
            & np.isfinite(policy_scores)
            & (policy_scores > EPS)
            & (residual_scores > EPS)
            & (remaining_period[np.clip(t_arr, 0, params.horizon - 1)] > EPS)
        )
        if not valid.any():
            break
        order = np.flatnonzero(valid)[np.argsort(policy_scores[valid])[::-1]]
        pass_budget = min(batch_budget, remaining_total)
        pass_allocated = 0.0
        pass_value = 0.0
        pass_policy_value = 0.0
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
            residual_value = available * float(residual_scores[pos])
            policy_value = available * float(policy_scores[pos])
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
                    "value_proxy": float(residual_value),
                    "policy_value_proxy": float(policy_value),
                    "oracle_value_per_cost": float(residual_scores[pos]),
                    "policy_score_value": float(policy_scores[pos]),
                    "segment_effectiveness_multiplier": float(multipliers[pos]),
                    "footprint_weight": float(row.get("footprint_weight", 0.0)),
                    "footprint_unit_rank": float(footprint_rank[pos]),
                    "finite_unit_rank": float(finite_rank[pos]),
                    "residual_pass": int(pass_id),
                    "remaining_total_after": float(remaining_total),
                }
            )
            selected_actions.add((str(row["unit"]), t, intervention))
            allocated_cost += available
            value_proxy += residual_value
            policy_value_proxy += policy_value
            pass_allocated += available
            pass_value += residual_value
            pass_policy_value += policy_value
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
                "policy_value_proxy": float(pass_policy_value),
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
        "policy_value_proxy": policy_value_proxy,
        "selected_action_count": len(selected_actions),
        "allocations": pd.DataFrame(allocations),
        "pass_rows": pass_rows,
    }


def combine_residual_scores(
    residual_scores: np.ndarray,
    footprint_rank: np.ndarray,
    finite_rank: np.ndarray,
    *,
    footprint_lambda: float,
    finite_lambda: float,
    mode: str,
) -> np.ndarray:
    residual = np.asarray(residual_scores, dtype=float)
    if mode == "multiply_footprint":
        return residual * np.maximum(footprint_rank, 0.0)
    if mode == "multiply_finite":
        return residual * np.maximum(finite_rank, 0.0)
    modifier = 1.0 + float(footprint_lambda) * np.maximum(footprint_rank, 0.0)
    modifier += float(finite_lambda) * np.maximum(finite_rank, 0.0)
    return residual * modifier


def build_policy_summary(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    grouped = (
        policy.groupby("policy_score", as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_fraction_of_hybrid_lp_gain=("fraction_of_hybrid_lp_gain", "mean"),
            median_fraction_of_hybrid_lp_gain=("fraction_of_hybrid_lp_gain", "median"),
            mean_replay_recoverable_fraction=("replay_recoverable_fraction", "mean"),
            mean_allocated_unit_footprint_mass=("allocated_unit_footprint_mass", "mean"),
            mean_allocated_top5pct_unit_footprint_mass=("allocated_top5pct_unit_footprint_mass", "mean"),
            mean_cost_weighted_footprint_score=("allocated_cost_weighted_footprint_score", "mean"),
            mean_unit_jaccard_with_hybrid_lp=("selected_unit_jaccard_with_hybrid_lp", "mean"),
            mean_action_cost_jaccard_with_hybrid_lp=("action_cost_jaccard_with_hybrid_lp", "mean"),
            mean_action_cost_overlap_share_lp=("action_cost_overlap_share_lp", "mean"),
            mean_selected_action_count=("selected_action_count", "mean"),
            mean_selected_unit_count=("selected_unit_count", "mean"),
        )
        .copy()
    )
    residual = grouped[grouped["policy_score"].eq("residual_finite_greedy")]
    if not residual.empty:
        row = residual.iloc[0]
        grouped["delta_fraction_vs_residual"] = (
            grouped["mean_fraction_of_hybrid_lp_gain"] - float(row["mean_fraction_of_hybrid_lp_gain"])
        )
        grouped["delta_top5_footprint_vs_residual"] = (
            grouped["mean_allocated_top5pct_unit_footprint_mass"] - float(row["mean_allocated_top5pct_unit_footprint_mass"])
        )
        grouped["delta_cost_weighted_footprint_vs_residual"] = (
            grouped["mean_cost_weighted_footprint_score"] - float(row["mean_cost_weighted_footprint_score"])
        )
    order = ["hybrid_lp_replay", *[str(spec["policy_score"]) for spec in POLICY_SPECS]]
    grouped["policy_order"] = grouped["policy_score"].map({name: idx for idx, name in enumerate(order)}).fillna(999)
    return grouped.sort_values(["policy_order", "mean_fraction_of_hybrid_lp_gain"], ascending=[True, False]).drop(columns=["policy_order"])


def build_event_delta(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    residual = policy[policy["policy_score"].eq("residual_finite_greedy")][
        [
            "city",
            "event_id",
            "fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass",
            "allocated_cost_weighted_footprint_score",
        ]
    ].rename(
        columns={
            "fraction_of_hybrid_lp_gain": "residual_fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass": "residual_top5_footprint_mass",
            "allocated_cost_weighted_footprint_score": "residual_cost_weighted_footprint_score",
        }
    )
    out = policy.merge(residual, on=["city", "event_id"], how="left")
    out["delta_fraction_vs_residual"] = out["fraction_of_hybrid_lp_gain"] - out["residual_fraction_of_hybrid_lp_gain"]
    out["delta_top5_footprint_vs_residual"] = out["allocated_top5pct_unit_footprint_mass"] - out["residual_top5_footprint_mass"]
    out["delta_cost_weighted_footprint_vs_residual"] = (
        out["allocated_cost_weighted_footprint_score"] - out["residual_cost_weighted_footprint_score"]
    )
    return out.sort_values(["city", "event_id", "policy_score"])


def build_metrics(policy: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    residual = one_row(summary, policy_score="residual_finite_greedy")
    weak = one_row(summary, policy_score="residual_plus_0p25_footprint")
    stronger = one_row(summary, policy_score="residual_plus_1p0_footprint")
    gated = one_row(summary, policy_score="residual_x_footprint_rank")
    finite = one_row(summary, policy_score="residual_plus_0p5_finite")
    candidates = summary[~summary["policy_score"].isin(["hybrid_lp_replay", "residual_finite_greedy"])].copy()
    residual_fraction = safe_float(residual.get("mean_fraction_of_hybrid_lp_gain"))
    nonnegative = candidates[candidates["delta_fraction_vs_residual"] >= -1e-9] if "delta_fraction_vs_residual" in candidates else pd.DataFrame()
    near = candidates[candidates["mean_fraction_of_hybrid_lp_gain"] >= residual_fraction - 0.01] if np.isfinite(residual_fraction) else pd.DataFrame()
    best_nonnegative = best_by_footprint(nonnegative)
    best_near = best_by_footprint(near)
    best_gain = candidates.sort_values("mean_fraction_of_hybrid_lp_gain", ascending=False).head(1) if not candidates.empty else pd.DataFrame()
    return {
        "n_events": safe_int(policy[policy["policy_score"].eq("residual_finite_greedy")]["event_id"].nunique()),
        "n_policies": safe_int(summary["policy_score"].nunique()),
        "residual_fraction_of_hybrid_lp_gain": safe_float(residual.get("mean_fraction_of_hybrid_lp_gain")),
        "residual_top5_footprint_mass": safe_float(residual.get("mean_allocated_top5pct_unit_footprint_mass")),
        "residual_cost_weighted_footprint": safe_float(residual.get("mean_cost_weighted_footprint_score")),
        "weak_footprint_policy": "residual_plus_0p25_footprint",
        "weak_footprint_fraction": safe_float(weak.get("mean_fraction_of_hybrid_lp_gain")),
        "weak_footprint_top5_mass": safe_float(weak.get("mean_allocated_top5pct_unit_footprint_mass")),
        "weak_footprint_delta_fraction": safe_float(weak.get("delta_fraction_vs_residual")),
        "weak_footprint_delta_top5": safe_float(weak.get("delta_top5_footprint_vs_residual")),
        "strong_footprint_policy": "residual_plus_1p0_footprint",
        "strong_footprint_fraction": safe_float(stronger.get("mean_fraction_of_hybrid_lp_gain")),
        "strong_footprint_top5_mass": safe_float(stronger.get("mean_allocated_top5pct_unit_footprint_mass")),
        "strong_footprint_delta_fraction": safe_float(stronger.get("delta_fraction_vs_residual")),
        "strong_footprint_delta_top5": safe_float(stronger.get("delta_top5_footprint_vs_residual")),
        "gated_footprint_policy": "residual_x_footprint_rank",
        "gated_footprint_fraction": safe_float(gated.get("mean_fraction_of_hybrid_lp_gain")),
        "gated_footprint_top5_mass": safe_float(gated.get("mean_allocated_top5pct_unit_footprint_mass")),
        "gated_footprint_delta_fraction": safe_float(gated.get("delta_fraction_vs_residual")),
        "gated_footprint_delta_top5": safe_float(gated.get("delta_top5_footprint_vs_residual")),
        "finite_policy": "residual_plus_0p5_finite",
        "finite_fraction": safe_float(finite.get("mean_fraction_of_hybrid_lp_gain")),
        "finite_top5_mass": safe_float(finite.get("mean_allocated_top5pct_unit_footprint_mass")),
        "finite_delta_fraction": safe_float(finite.get("delta_fraction_vs_residual")),
        "finite_delta_top5": safe_float(finite.get("delta_top5_footprint_vs_residual")),
        "best_nonnegative_policy": first_string(best_nonnegative, "policy_score"),
        "best_nonnegative_fraction": safe_first(best_nonnegative, "mean_fraction_of_hybrid_lp_gain"),
        "best_nonnegative_top5_mass": safe_first(best_nonnegative, "mean_allocated_top5pct_unit_footprint_mass"),
        "best_nonnegative_delta_fraction": safe_first(best_nonnegative, "delta_fraction_vs_residual"),
        "best_nonnegative_delta_top5": safe_first(best_nonnegative, "delta_top5_footprint_vs_residual"),
        "best_near_no_loss_policy": first_string(best_near, "policy_score"),
        "best_near_no_loss_fraction": safe_first(best_near, "mean_fraction_of_hybrid_lp_gain"),
        "best_near_no_loss_top5_mass": safe_first(best_near, "mean_allocated_top5pct_unit_footprint_mass"),
        "best_near_no_loss_delta_fraction": safe_first(best_near, "delta_fraction_vs_residual"),
        "best_near_no_loss_delta_top5": safe_first(best_near, "delta_top5_footprint_vs_residual"),
        "best_gain_policy": first_string(best_gain, "policy_score"),
        "best_gain_fraction": safe_first(best_gain, "mean_fraction_of_hybrid_lp_gain"),
    }


def best_by_footprint(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.sort_values(
        ["mean_allocated_top5pct_unit_footprint_mass", "mean_fraction_of_hybrid_lp_gain"],
        ascending=[False, False],
    ).head(1)


def make_figures(summary: pd.DataFrame, event_delta: pd.DataFrame, policy: pd.DataFrame, figure_dir: Path) -> None:
    if summary.empty:
        return
    plot = summary[~summary["policy_score"].eq("hybrid_lp_replay")].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        plot["mean_allocated_top5pct_unit_footprint_mass"],
        plot["mean_fraction_of_hybrid_lp_gain"],
        s=82,
        color="#2563eb",
        alpha=0.82,
    )
    for _, row in plot.iterrows():
        ax.annotate(str(row["policy_score"]), (row["mean_allocated_top5pct_unit_footprint_mass"], row["mean_fraction_of_hybrid_lp_gain"]), fontsize=7)
    residual = plot[plot["policy_score"].eq("residual_finite_greedy")]
    if not residual.empty:
        ax.axhline(float(residual.iloc[0]["mean_fraction_of_hybrid_lp_gain"]), color="#111827", linestyle="--", linewidth=1, alpha=0.45)
        ax.axvline(float(residual.iloc[0]["mean_allocated_top5pct_unit_footprint_mass"]), color="#111827", linestyle="--", linewidth=1, alpha=0.45)
    ax.set_xlabel("Allocated top-5% unit footprint mass")
    ax.set_ylabel("Replay gain / hybrid LP gain")
    ax.set_title("Adaptive residual footprint policy trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_policy_gain_vs_footprint.png", dpi=180)
    plt.close(fig)

    ordered = plot.sort_values("mean_fraction_of_hybrid_lp_gain", ascending=False)
    x = np.arange(len(ordered))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar(x - width / 2, ordered["mean_fraction_of_hybrid_lp_gain"], width=width, color="#0f766e", label="gain / LP")
    ax.bar(x + width / 2, ordered["mean_allocated_top5pct_unit_footprint_mass"], width=width, color="#f59e0b", label="footprint mass")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["policy_score"], rotation=35, ha="right")
    ax.set_ylabel("Mean metric")
    ax.set_title("Residual policies: gain and footprint")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_policy_summary_bars.png", dpi=180)
    plt.close(fig)

    focus = event_delta[event_delta["policy_score"].isin(["residual_plus_0p25_footprint", "residual_plus_1p0_footprint", "residual_x_footprint_rank"])].copy()
    if not focus.empty:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for score, group in focus.groupby("policy_score"):
            ax.scatter(
                group["delta_top5_footprint_vs_residual"],
                group["delta_fraction_vs_residual"],
                label=score,
                s=58,
                alpha=0.82,
            )
        ax.axhline(0.0, color="#111827", linewidth=1, linestyle="--", alpha=0.45)
        ax.axvline(0.0, color="#111827", linewidth=1, linestyle="--", alpha=0.45)
        ax.set_xlabel("Delta top-5% footprint mass vs residual")
        ax.set_ylabel("Delta gain / hybrid LP vs residual")
        ax.set_title("Event-level residual footprint deltas")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "residual_event_policy_deltas.png", dpi=180)
        plt.close(fig)

    if not policy.empty:
        heat = policy[~policy["policy_score"].eq("hybrid_lp_replay")].pivot_table(
            index="city",
            columns="policy_score",
            values="fraction_of_hybrid_lp_gain",
            aggfunc="mean",
        )
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(9.5, 4.6))
            values = heat.to_numpy(dtype=float)
            im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=max(0.0, np.nanmin(values)), vmax=max(1.0, np.nanmax(values)))
            ax.set_xticks(np.arange(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(heat.index)))
            ax.set_yticklabels(heat.index)
            ax.set_title("Residual replay gain fraction by city and policy")
            fig.colorbar(im, ax=ax, label="gain / hybrid LP")
            fig.tight_layout()
            fig.savefig(figure_dir / "residual_policy_gain_heatmap.png", dpi=180)
            plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], summary: pd.DataFrame, event_delta: pd.DataFrame) -> None:
    lines = [
        "# Residual Footprint-Aware Policy Validation V39",
        "",
        "V39 把 V38 的静态 footprint-aware replay 推进到 adaptive residual replay。每个 policy 都先 replay 已经分配的资源，得到当前 residual `b/rC/rS/ell`，再用 residual finite score 重新排序候选部署段；footprint 或 finite rank 只作为 residual score 的弱乘法修正。",
        "",
        "## 主要结论",
        "",
        f"- 覆盖事件数：{metrics['n_events']}；比较 policy 数：{metrics['n_policies']}。",
        f"- `residual_finite_greedy` replay gain / hybrid LP gain = {fmt(metrics['residual_fraction_of_hybrid_lp_gain'])}，top-5% allocated-unit footprint mass = {fmt(metrics['residual_top5_footprint_mass'])}。",
        f"- 弱 footprint residual policy `{metrics['weak_footprint_policy']}` gain / LP = {fmt(metrics['weak_footprint_fraction'])}，footprint mass = {fmt(metrics['weak_footprint_top5_mass'])}，gain delta = {fmt(metrics['weak_footprint_delta_fraction'])}，footprint delta = {fmt(metrics['weak_footprint_delta_top5'])}。",
        f"- 强 footprint residual policy `{metrics['strong_footprint_policy']}` gain / LP = {fmt(metrics['strong_footprint_fraction'])}，footprint mass = {fmt(metrics['strong_footprint_top5_mass'])}，gain delta = {fmt(metrics['strong_footprint_delta_fraction'])}，footprint delta = {fmt(metrics['strong_footprint_delta_top5'])}。",
        f"- footprint-gated residual policy `{metrics['gated_footprint_policy']}` gain / LP = {fmt(metrics['gated_footprint_fraction'])}，footprint mass = {fmt(metrics['gated_footprint_top5_mass'])}，gain delta = {fmt(metrics['gated_footprint_delta_fraction'])}，footprint delta = {fmt(metrics['gated_footprint_delta_top5'])}。",
        f"- 不低于 residual gain 的 footprint 候选中，footprint mass 最高的是 `{metrics['best_nonnegative_policy']}`：gain / LP = {fmt(metrics['best_nonnegative_fraction'])}，footprint mass = {fmt(metrics['best_nonnegative_top5_mass'])}，gain delta = {fmt(metrics['best_nonnegative_delta_fraction'])}，footprint delta = {fmt(metrics['best_nonnegative_delta_top5'])}。",
        f"- gain 损失不超过 0.01 的候选中，footprint mass 最高的是 `{metrics['best_near_no_loss_policy']}`：gain / LP = {fmt(metrics['best_near_no_loss_fraction'])}，footprint mass = {fmt(metrics['best_near_no_loss_top5_mass'])}，gain delta = {fmt(metrics['best_near_no_loss_delta_fraction'])}，footprint delta = {fmt(metrics['best_near_no_loss_delta_top5'])}。",
        "",
        "## 解释",
        "",
        "如果 residual footprint 修正几乎不能增加 footprint mass，说明 residual re-scoring 已经把可修复损失、OD exposure、时序可行性和 diminishing returns 吸收进了主排序，footprint 只能作为非常弱的解释性标签。如果 footprint 修正能在不损失 gain 的情况下提高 footprint mass，则可以把 observed footprint 写成 residual law 的二级 tie-breaker，而不是独立目标。",
        "",
        "## Policy Summary",
        "",
        table_to_markdown(summary, max_rows=30),
        "",
        "## Event-Level Deltas",
        "",
        table_to_markdown(
            event_delta[
                event_delta["policy_score"].isin(["residual_plus_0p25_footprint", "residual_plus_1p0_footprint", "residual_x_footprint_rank"])
            ],
            max_rows=30,
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def empty_result() -> dict[str, Any]:
    return {
        "allocated_cost": 0.0,
        "value_proxy": 0.0,
        "policy_value_proxy": 0.0,
        "selected_action_count": 0,
        "allocations": pd.DataFrame(),
        "pass_rows": [],
    }


def one_row(df: pd.DataFrame, **filters: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df:
            return pd.Series(dtype=float)
        mask &= df[column].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=float)
    return df.loc[mask].iloc[0]


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return np.nan
    return number if np.isfinite(number) else np.nan


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def safe_first(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return safe_float(frame.iloc[0][column])


def first_string(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    return str(frame.iloc[0][column])


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if not np.isfinite(number) else f"{number:.4f}"


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
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

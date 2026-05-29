"""Validate footprint-aware scores as finite-budget replay policies.

V37 showed that observed event footprints can enter the action-value law as a
gated refinement without losing much alignment with representative hybrid LP
selected support.  This script tests the stricter policy question: if those
scores are used to allocate a finite budget with period budgets, deployment
caps, delays, and piecewise-linear diminishing returns, do they preserve replay
gain while moving resources toward observed footprints?
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

from analyze_footprint_aware_law_frontier import add_score_columns, build_scores, set_mass
from analyze_hybrid_absorption_mechanisms import footprint_weights, hybrid_summary_row, prepare_interventions_with_caps
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from learn_recovery_laws import build_event_action_frame, replay_optimizer_solution, replay_policy_allocations
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS


EPS = 1e-12
POLICY_SCORE_IDS = [
    "small_signal",
    "small_plus_0p1_footprint",
    "small_plus_0p25_footprint",
    "small_plus_0p5_footprint",
    "small_plus_1p0_footprint",
    "small_plus_2p0_footprint",
    "small_plus_4p0_footprint",
    "activation_plus_0p1_footprint",
    "activation_plus_0p25_footprint",
    "activation_plus_0p5_footprint",
    "activation_plus_1p0_footprint",
    "activation_plus_2p0_footprint",
    "activation_plus_4p0_footprint",
    "small_plus_0p1_finite",
    "small_plus_0p25_finite",
    "small_plus_0p5_finite",
    "small_plus_1p0_finite",
    "small_plus_2p0_finite",
    "small_plus_4p0_finite",
    "small_x_finite_rank",
    "finite_x_small_rank",
    "finite_value",
    "small_x_footprint_rank",
    "activation_no_eta",
    "activation_x_footprint",
    "footprint_only",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/footprint_aware_policy_validation")
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
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
    ok_events = event_metrics[
        event_metrics["hybrid_status"].astype(str).eq("OPTIMAL")
        & event_metrics["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)

    for idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{len(ok_events)}] Footprint-aware finite-budget replay for {city} event {event_id}", flush=True)
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
        score_defs = build_scores(full)
        for score_id, values in score_defs.items():
            full[f"score__{score_id}"] = np.asarray(values, dtype=float)
        segments = build_policy_segments(full, config, score_ids=POLICY_SCORE_IDS)
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
        for score_id in POLICY_SCORE_IDS:
            score_col = f"score__{score_id}"
            if score_col not in segments:
                continue
            result = allocate_by_score(segments, score_col, hybrid_params)
            allocations = result["allocations"]
            if not allocations.empty:
                allocations = allocations.copy()
                allocations["policy_score"] = score_id
                allocation_frames.append(allocations)
            replay = replay_policy_allocations(allocations, hybrid_params)
            policy_rows.append(
                policy_row(
                    city,
                    event_id,
                    row,
                    score_id,
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
    summary = build_policy_summary(policy)
    event_delta = build_event_delta(policy)
    metrics = build_metrics(policy, summary)

    write_table(policy, table_dir / "footprint_aware_policy_event_metrics.csv")
    write_table(summary, table_dir / "footprint_aware_policy_summary.csv")
    write_table(event_delta, table_dir / "footprint_aware_policy_event_delta.csv")
    write_table(allocations, table_dir / "footprint_aware_policy_allocations.csv.gz")
    (table_dir / "footprint_aware_policy_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(summary, event_delta, policy, figure_dir)
    write_report(report_dir / "footprint_aware_policy_validation_report_zh.md", metrics, summary, event_delta)
    print(f"Wrote footprint-aware policy validation to {output_dir}")


def build_policy_segments(full: pd.DataFrame, config: dict[str, Any], *, score_ids: list[str]) -> pd.DataFrame:
    pwl = config["interventions"].get("pwl_diminishing_returns", {})
    if bool(pwl.get("enabled", False)):
        segment_shares = np.asarray(pwl["segment_cap_shares"], dtype=float)
        segment_shares = segment_shares / segment_shares.sum()
        multipliers_by_k = {
            key: np.asarray(pwl["effectiveness_multipliers"][key], dtype=float)
            for key in INTERVENTIONS
        }
    else:
        segment_shares = np.array([1.0], dtype=float)
        multipliers_by_k = {key: np.array([1.0], dtype=float) for key in INTERVENTIONS}

    base_cols = [
        "city",
        "event_id",
        "event_start",
        "scenario",
        "unit",
        "t",
        "intervention",
        "cost",
        "u_cap",
        "marginal_resource_value",
        "finite_deficit_area_value",
        "footprint_weight",
        "footprint_unit_rank",
        "small_signal_unit_rank",
        "finite_unit_rank",
    ]
    score_cols = [f"score__{score_id}" for score_id in score_ids if f"score__{score_id}" in full]
    frames: list[pd.DataFrame] = []
    for intervention in INTERVENTIONS:
        frame = full.loc[full["intervention"].astype(str).eq(intervention), base_cols + score_cols].copy()
        multipliers = multipliers_by_k[intervention]
        for segment_id, (share, multiplier) in enumerate(zip(segment_shares, multipliers, strict=True)):
            seg = frame.copy()
            seg["segment"] = int(segment_id)
            seg["segment_share"] = float(share)
            seg["segment_effectiveness_multiplier"] = float(multiplier)
            seg["segment_cost_cap"] = seg["cost"] * seg["u_cap"] * float(share)
            seg["oracle_value_per_cost"] = seg["marginal_resource_value"] * float(multiplier)
            for score_col in score_cols:
                seg[score_col] = pd.to_numeric(seg[score_col], errors="coerce").fillna(0.0) * float(multiplier)
            frames.append(seg)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out[out["segment_cost_cap"] > EPS].reset_index(drop=True)


def allocate_by_score(segments: pd.DataFrame, score_col: str, params: Any) -> dict[str, Any]:
    if segments.empty or params.total_budget <= EPS:
        return {"allocations": pd.DataFrame(), "allocated_cost": 0.0, "value_proxy": 0.0}
    scores = pd.to_numeric(segments[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    valid = scores > EPS
    if not valid.any():
        return {"allocations": pd.DataFrame(), "allocated_cost": 0.0, "value_proxy": 0.0}
    order = np.flatnonzero(valid)[np.argsort(scores[valid])[::-1]]
    remaining_total = float(params.total_budget)
    remaining_period = np.asarray(params.period_budget, dtype=float).copy()
    rows: list[dict[str, Any]] = []
    allocated_cost = 0.0
    value_proxy = 0.0
    for pos in order:
        if remaining_total <= EPS or np.all(remaining_period <= EPS):
            break
        row = segments.iloc[int(pos)]
        t = int(row["t"])
        intervention = str(row["intervention"])
        if t < 0 or t >= params.horizon or t < int(params.delays.get(intervention, 0)):
            continue
        available = min(float(row["segment_cost_cap"]), remaining_total, float(remaining_period[t]))
        if available <= EPS:
            continue
        allocated_u = available / max(float(row["cost"]), EPS)
        proxy = available * float(row["oracle_value_per_cost"])
        rows.append(
            {
                "city": str(row["city"]),
                "event_id": int(row["event_id"]),
                "event_start": str(row["event_start"]),
                "scenario": str(row["scenario"]),
                "unit": str(row["unit"]),
                "t": t,
                "intervention": intervention,
                "segment": int(row["segment"]),
                "allocated_cost": float(available),
                "allocated_u": float(allocated_u),
                "value_proxy": float(proxy),
                "oracle_value_per_cost": float(row["oracle_value_per_cost"]),
                "policy_score_value": float(row[score_col]),
                "segment_effectiveness_multiplier": float(row["segment_effectiveness_multiplier"]),
                "footprint_weight": float(row["footprint_weight"]),
            }
        )
        remaining_total -= available
        remaining_period[t] -= available
        allocated_cost += available
        value_proxy += proxy
    return {"allocations": pd.DataFrame(rows), "allocated_cost": allocated_cost, "value_proxy": value_proxy}


def policy_row(
    city: str,
    event_id: int,
    v35_row: Any,
    policy_score: str,
    *,
    allocations: pd.DataFrame,
    replay_objective: float,
    baseline_objective: float,
    optimized_objective: float,
    hybrid_lp_gain: float,
    weights: pd.Series,
    full: pd.DataFrame,
    v34_delta: float,
    v35_delta: float,
) -> dict[str, Any]:
    replay_gain = baseline_objective - replay_objective
    allocated_units = set(allocations["unit"].astype(str)) if not allocations.empty else set(full.loc[full["optimized_cost"] > EPS, "unit"].astype(str))
    top_units = top_allocated_units(allocations, full)
    selected_units = set(full.loc[full["optimized_cost"] > EPS, "unit"].astype(str))
    action_overlap = action_cost_overlap(full, allocations)
    total_cost = float(allocations["allocated_cost"].sum()) if not allocations.empty else float(full["optimized_cost"].sum())
    cost_weighted_footprint = cost_weighted_footprint_score(allocations, full, weights)
    return {
        "city": city,
        "event_id": int(event_id),
        "event_start": str(getattr(v35_row, "event_start", "")),
        "policy_score": policy_score,
        "n_units": int(getattr(v35_row, "n_units", full["unit"].nunique())),
        "baseline_objective": float(baseline_objective),
        "hybrid_optimized_objective": float(optimized_objective),
        "replay_objective": float(replay_objective),
        "hybrid_lp_gain": float(hybrid_lp_gain),
        "replay_gain": float(replay_gain),
        "fraction_of_hybrid_lp_gain": safe_div(replay_gain, hybrid_lp_gain),
        "replay_recoverable_fraction": safe_div(replay_gain, baseline_objective),
        "gap_to_hybrid_lp_gain": 1.0 - safe_div(replay_gain, hybrid_lp_gain),
        "allocated_cost": total_cost,
        "total_budget": float(getattr(v35_row, "hybrid_total_intervention_cost", np.nan)) if hasattr(v35_row, "hybrid_total_intervention_cost") else total_cost,
        "selected_action_count": int(len(action_set(allocations))) if not allocations.empty else int((full["optimized_cost"] > EPS).sum()),
        "selected_unit_count": int(len(allocated_units)),
        "allocated_unit_footprint_mass": set_mass(weights, allocated_units),
        "allocated_top5pct_unit_footprint_mass": set_mass(weights, top_units),
        "allocated_cost_weighted_footprint_score": cost_weighted_footprint,
        "selected_unit_jaccard_with_hybrid_lp": jaccard(allocated_units, selected_units),
        "action_cost_jaccard_with_hybrid_lp": action_overlap["action_cost_jaccard"],
        "action_cost_overlap_share_policy": action_overlap["policy_overlap_share"],
        "action_cost_overlap_share_lp": action_overlap["lp_overlap_share"],
        "v34_delta_finite_top5pct_units_footprint_mass": v34_delta,
        "v35_delta_selected_unit_footprint_mass": v35_delta,
        "event_peak_positive_abnormal_deficit": safe_float(getattr(v35_row, "event_peak_positive_abnormal_deficit", np.nan)),
        "event_total_precip": safe_float(getattr(v35_row, "event_total_precip", np.nan)),
    }


def top_allocated_units(allocations: pd.DataFrame, full: pd.DataFrame) -> set[str]:
    n_units = int(full["unit"].nunique())
    top_n = max(1, int(math.ceil(0.05 * n_units)))
    if allocations.empty:
        source = full.groupby("unit", as_index=False).agg(cost=("optimized_cost", "sum"))
    else:
        source = allocations.groupby("unit", as_index=False).agg(cost=("allocated_cost", "sum"))
    return set(source.sort_values("cost", ascending=False).head(top_n)["unit"].astype(str))


def cost_weighted_footprint_score(allocations: pd.DataFrame, full: pd.DataFrame, weights: pd.Series) -> float:
    if allocations.empty:
        costs = full.groupby("unit")["optimized_cost"].sum()
    else:
        costs = allocations.groupby("unit")["allocated_cost"].sum()
    total = float(costs.sum())
    if total <= EPS:
        return np.nan
    return float((costs * weights.reindex(costs.index.astype(str), fill_value=0.0)).sum() / total)


def action_cost_overlap(full: pd.DataFrame, allocations: pd.DataFrame) -> dict[str, float]:
    keys = ["unit", "t", "intervention"]
    lp = full.groupby(keys, as_index=False).agg(lp_cost=("optimized_cost", "sum"))
    if allocations.empty:
        policy = lp.rename(columns={"lp_cost": "policy_cost"})
    else:
        policy = allocations.groupby(keys, as_index=False).agg(policy_cost=("allocated_cost", "sum"))
    merged = lp.merge(policy, on=keys, how="outer").fillna({"lp_cost": 0.0, "policy_cost": 0.0})
    overlap = np.minimum(merged["lp_cost"], merged["policy_cost"])
    lp_total = float(merged["lp_cost"].sum())
    policy_total = float(merged["policy_cost"].sum())
    overlap_total = float(overlap.sum())
    union = lp_total + policy_total - overlap_total
    return {
        "action_cost_jaccard": safe_div(overlap_total, union),
        "policy_overlap_share": safe_div(overlap_total, policy_total),
        "lp_overlap_share": safe_div(overlap_total, lp_total),
    }


def action_set(allocations: pd.DataFrame) -> set[tuple[str, int, str]]:
    if allocations.empty:
        return set()
    return {
        (str(row.unit), int(row.t), str(row.intervention))
        for row in allocations[["unit", "t", "intervention"]].drop_duplicates().itertuples(index=False)
    }


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
    small = grouped[grouped["policy_score"].eq("small_signal")]
    if not small.empty:
        row = small.iloc[0]
        grouped["delta_fraction_vs_small_signal"] = (
            grouped["mean_fraction_of_hybrid_lp_gain"] - float(row["mean_fraction_of_hybrid_lp_gain"])
        )
        grouped["delta_top5_footprint_vs_small_signal"] = (
            grouped["mean_allocated_top5pct_unit_footprint_mass"] - float(row["mean_allocated_top5pct_unit_footprint_mass"])
        )
        grouped["delta_cost_weighted_footprint_vs_small_signal"] = (
            grouped["mean_cost_weighted_footprint_score"] - float(row["mean_cost_weighted_footprint_score"])
        )
    grouped["policy_frontier_utility"] = (
        grouped["mean_fraction_of_hybrid_lp_gain"]
        + grouped["mean_allocated_top5pct_unit_footprint_mass"]
        - 0.5 * np.maximum(-grouped.get("delta_fraction_vs_small_signal", 0.0), 0.0)
    )
    order = ["hybrid_lp_replay", *POLICY_SCORE_IDS]
    grouped["policy_order"] = grouped["policy_score"].map({name: idx for idx, name in enumerate(order)}).fillna(999)
    return grouped.sort_values(["policy_order", "mean_fraction_of_hybrid_lp_gain"], ascending=[True, False]).drop(columns=["policy_order"])


def build_event_delta(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    small = policy[policy["policy_score"].eq("small_signal")][
        [
            "city",
            "event_id",
            "fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass",
            "allocated_cost_weighted_footprint_score",
        ]
    ].rename(
        columns={
            "fraction_of_hybrid_lp_gain": "small_fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass": "small_top5_footprint_mass",
            "allocated_cost_weighted_footprint_score": "small_cost_weighted_footprint_score",
        }
    )
    out = policy.merge(small, on=["city", "event_id"], how="left")
    out["delta_fraction_vs_small_signal"] = out["fraction_of_hybrid_lp_gain"] - out["small_fraction_of_hybrid_lp_gain"]
    out["delta_top5_footprint_vs_small_signal"] = out["allocated_top5pct_unit_footprint_mass"] - out["small_top5_footprint_mass"]
    out["delta_cost_weighted_footprint_vs_small_signal"] = (
        out["allocated_cost_weighted_footprint_score"] - out["small_cost_weighted_footprint_score"]
    )
    return out.sort_values(["city", "event_id", "policy_score"])


def build_metrics(policy: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    small = one_row(summary, policy_score="small_signal")
    gated = one_row(summary, policy_score="finite_x_small_rank")
    light = one_row(summary, policy_score="activation_plus_0p5_footprint")
    footprint = one_row(summary, policy_score="footprint_only")
    finite = one_row(summary, policy_score="finite_value")
    candidates = summary[~summary["policy_score"].isin(["hybrid_lp_replay", "small_signal"])].copy()
    small_fraction = safe_float(small.get("mean_fraction_of_hybrid_lp_gain"))
    near = candidates[candidates["mean_fraction_of_hybrid_lp_gain"] >= small_fraction - 0.02] if np.isfinite(small_fraction) else pd.DataFrame()
    best_near = near.sort_values(
        ["mean_allocated_top5pct_unit_footprint_mass", "mean_fraction_of_hybrid_lp_gain"],
        ascending=[False, False],
    ).head(1)
    nonnegative = candidates[candidates["delta_fraction_vs_small_signal"] >= -1e-9] if "delta_fraction_vs_small_signal" in candidates else pd.DataFrame()
    best_nonnegative = nonnegative.sort_values(
        ["mean_allocated_top5pct_unit_footprint_mass", "mean_fraction_of_hybrid_lp_gain"],
        ascending=[False, False],
    ).head(1)
    best_gain = summary[~summary["policy_score"].eq("hybrid_lp_replay")].sort_values(
        "mean_fraction_of_hybrid_lp_gain",
        ascending=False,
    ).head(1)
    return {
        "n_events": safe_int(policy[policy["policy_score"].eq("small_signal")]["event_id"].nunique()),
        "n_policies": safe_int(summary["policy_score"].nunique()),
        "small_signal_fraction_of_hybrid_lp_gain": safe_float(small.get("mean_fraction_of_hybrid_lp_gain")),
        "small_signal_top5_footprint_mass": safe_float(small.get("mean_allocated_top5pct_unit_footprint_mass")),
        "small_signal_cost_weighted_footprint": safe_float(small.get("mean_cost_weighted_footprint_score")),
        "finite_x_small_fraction_of_hybrid_lp_gain": safe_float(gated.get("mean_fraction_of_hybrid_lp_gain")),
        "finite_x_small_top5_footprint_mass": safe_float(gated.get("mean_allocated_top5pct_unit_footprint_mass")),
        "finite_x_small_delta_fraction": safe_float(gated.get("delta_fraction_vs_small_signal")),
        "finite_x_small_delta_top5_footprint": safe_float(gated.get("delta_top5_footprint_vs_small_signal")),
        "activation_plus_0p5_footprint_fraction": safe_float(light.get("mean_fraction_of_hybrid_lp_gain")),
        "activation_plus_0p5_footprint_top5_mass": safe_float(light.get("mean_allocated_top5pct_unit_footprint_mass")),
        "activation_plus_0p5_footprint_delta_fraction": safe_float(light.get("delta_fraction_vs_small_signal")),
        "activation_plus_0p5_footprint_delta_top5": safe_float(light.get("delta_top5_footprint_vs_small_signal")),
        "finite_value_fraction_of_hybrid_lp_gain": safe_float(finite.get("mean_fraction_of_hybrid_lp_gain")),
        "finite_value_top5_footprint_mass": safe_float(finite.get("mean_allocated_top5pct_unit_footprint_mass")),
        "footprint_only_fraction_of_hybrid_lp_gain": safe_float(footprint.get("mean_fraction_of_hybrid_lp_gain")),
        "footprint_only_top5_footprint_mass": safe_float(footprint.get("mean_allocated_top5pct_unit_footprint_mass")),
        "footprint_only_delta_fraction": safe_float(footprint.get("delta_fraction_vs_small_signal")),
        "best_near_no_loss_policy": str(best_near.iloc[0]["policy_score"]) if not best_near.empty else "",
        "best_near_no_loss_fraction": safe_first(best_near, "mean_fraction_of_hybrid_lp_gain"),
        "best_near_no_loss_top5_footprint_mass": safe_first(best_near, "mean_allocated_top5pct_unit_footprint_mass"),
        "best_near_no_loss_delta_fraction": safe_first(best_near, "delta_fraction_vs_small_signal"),
        "best_near_no_loss_delta_top5_footprint": safe_first(best_near, "delta_top5_footprint_vs_small_signal"),
        "best_nonnegative_gain_policy": str(best_nonnegative.iloc[0]["policy_score"]) if not best_nonnegative.empty else "",
        "best_nonnegative_gain_fraction": safe_first(best_nonnegative, "mean_fraction_of_hybrid_lp_gain"),
        "best_nonnegative_gain_top5_footprint_mass": safe_first(best_nonnegative, "mean_allocated_top5pct_unit_footprint_mass"),
        "best_nonnegative_gain_delta_fraction": safe_first(best_nonnegative, "delta_fraction_vs_small_signal"),
        "best_nonnegative_gain_delta_top5_footprint": safe_first(best_nonnegative, "delta_top5_footprint_vs_small_signal"),
        "best_gain_policy": str(best_gain.iloc[0]["policy_score"]) if not best_gain.empty else "",
        "best_gain_fraction": safe_first(best_gain, "mean_fraction_of_hybrid_lp_gain"),
    }


def make_figures(summary: pd.DataFrame, event_delta: pd.DataFrame, policy: pd.DataFrame, figure_dir: Path) -> None:
    if summary.empty:
        return
    plot = summary[~summary["policy_score"].eq("hybrid_lp_replay")].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        plot["mean_allocated_top5pct_unit_footprint_mass"],
        plot["mean_fraction_of_hybrid_lp_gain"],
        s=80,
        color="#2563eb",
        alpha=0.82,
    )
    for _, row in plot.iterrows():
        ax.annotate(str(row["policy_score"]), (row["mean_allocated_top5pct_unit_footprint_mass"], row["mean_fraction_of_hybrid_lp_gain"]), fontsize=7)
    ax.set_xlabel("Allocated top-5% unit footprint mass")
    ax.set_ylabel("Replay gain / hybrid LP gain")
    ax.set_title("Footprint-aware finite-budget policy trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_gain_vs_footprint.png", dpi=180)
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
    ax.set_title("Finite-budget policies: gain and footprint")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_summary_bars.png", dpi=180)
    plt.close(fig)

    if not event_delta.empty:
        focus = event_delta[event_delta["policy_score"].isin(["finite_x_small_rank", "activation_plus_0p5_footprint", "footprint_only"])].copy()
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for score, group in focus.groupby("policy_score"):
            ax.scatter(
                group["delta_top5_footprint_vs_small_signal"],
                group["delta_fraction_vs_small_signal"],
                label=score,
                s=56,
                alpha=0.82,
            )
        ax.axhline(0.0, color="#111827", linewidth=1, linestyle="--", alpha=0.45)
        ax.axvline(0.0, color="#111827", linewidth=1, linestyle="--", alpha=0.45)
        ax.set_xlabel("Delta top-5% footprint mass vs small-signal")
        ax.set_ylabel("Delta gain / hybrid LP vs small-signal")
        ax.set_title("Event-level footprint policy deltas")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "event_policy_deltas.png", dpi=180)
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
            im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1.0, np.nanmax(heat.to_numpy(dtype=float))))
            ax.set_xticks(np.arange(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(heat.index)))
            ax.set_yticklabels(heat.index)
            ax.set_title("Replay gain fraction by city and policy")
            fig.colorbar(im, ax=ax, label="gain / hybrid LP")
            fig.tight_layout()
            fig.savefig(figure_dir / "policy_gain_heatmap.png", dpi=180)
            plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], summary: pd.DataFrame, event_delta: pd.DataFrame) -> None:
    lines = [
        "# Footprint-Aware Finite-Budget Policy Validation V38",
        "",
        "本版把 V37 的 footprint-aware score 从 support frontier 推进一步，放到完整有限预算 replay 中检验。每个 policy 都在同一个 hybrid calibration、同一个 total/period budget、同一组 delay/cap/diminishing 约束下按 score 贪心分配连续资源，然后 replay 恢复动力学，并与代表性 hybrid LP optimum 比较。",
        "",
        "## 主要结论",
        "",
        f"- 覆盖事件数：{metrics['n_events']}；比较 policy 数：{metrics['n_policies']}。",
        f"- `small_signal` replay gain / hybrid LP gain = {fmt(metrics['small_signal_fraction_of_hybrid_lp_gain'])}，top-5% allocated-unit footprint mass = {fmt(metrics['small_signal_top5_footprint_mass'])}。",
        f"- V37 的 near-no-loss support score `finite_x_small_rank` 在 replay 中 gain / LP = {fmt(metrics['finite_x_small_fraction_of_hybrid_lp_gain'])}，footprint mass = {fmt(metrics['finite_x_small_top5_footprint_mass'])}；相对 small-signal 的 gain delta = {fmt(metrics['finite_x_small_delta_fraction'])}，footprint delta = {fmt(metrics['finite_x_small_delta_top5_footprint'])}。",
        f"- 轻量 footprint 加权 `activation_plus_0p5_footprint` replay gain / LP = {fmt(metrics['activation_plus_0p5_footprint_fraction'])}，footprint mass = {fmt(metrics['activation_plus_0p5_footprint_top5_mass'])}；gain delta = {fmt(metrics['activation_plus_0p5_footprint_delta_fraction'])}，footprint delta = {fmt(metrics['activation_plus_0p5_footprint_delta_top5'])}。",
        f"- `footprint_only` replay gain / LP = {fmt(metrics['footprint_only_fraction_of_hybrid_lp_gain'])}，footprint mass = {fmt(metrics['footprint_only_top5_footprint_mass'])}，gain delta = {fmt(metrics['footprint_only_delta_fraction'])}。",
        f"- 在不低于 small-signal replay gain 的候选中，footprint 覆盖最高的是 `{metrics['best_nonnegative_gain_policy']}`，gain / LP = {fmt(metrics['best_nonnegative_gain_fraction'])}，footprint mass = {fmt(metrics['best_nonnegative_gain_top5_footprint_mass'])}，gain delta = {fmt(metrics['best_nonnegative_gain_delta_fraction'])}，footprint delta = {fmt(metrics['best_nonnegative_gain_delta_top5_footprint'])}。",
        f"- 在 replay gain 损失不超过 0.02 的候选中，footprint 覆盖最高的是 `{metrics['best_near_no_loss_policy']}`，gain / LP = {fmt(metrics['best_near_no_loss_fraction'])}，footprint mass = {fmt(metrics['best_near_no_loss_top5_footprint_mass'])}，footprint delta = {fmt(metrics['best_near_no_loss_delta_top5_footprint'])}。",
        "",
        "## 解释",
        "",
        "如果 gated footprint score 在 support frontier 上好、但 replay gain 明显下降，说明它目前更适合作为空间解释 refinement，而不是直接 policy score。反之，如果它几乎保持 replay gain 且提高 footprint 投放，就可以把 footprint-aware law 写得更强：observed footprint 可以作为有限预算恢复策略中的二级排序项。",
        "",
        "## Policy Summary",
        "",
        table_to_markdown(summary, max_rows=30),
        "",
        "## Event-Level Deltas",
        "",
        table_to_markdown(
            event_delta[
                event_delta["policy_score"].isin(["finite_x_small_rank", "activation_plus_0p5_footprint", "footprint_only"])
            ],
            max_rows=30,
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


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


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return np.nan
    return len(a & b) / max(len(a | b), 1)


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
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()

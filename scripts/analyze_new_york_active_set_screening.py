"""Design active-set screens for New York-scale footprint-aware LP closure.

The V43 New York audit found strong event-footprint signal but did not close
the direct hybrid-footprint LP.  This diagnostic does not solve a new LP.  It
asks whether a principled active action set can preserve the value/footprint
signals while reducing the action-variable side of the New York formulation.

The intended use is decomposition design: keep all New York state and OD access
constraints, but allow intervention deployment only on active units selected
from structural and event-specific value fields.
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
from scipy import sparse

from analyze_hybrid_absorption_mechanisms import footprint_weights
from analyze_hybrid_footprint_calibration import (
    DEFAULT_MAIN_BLEND,
    EMPTY_INTERVENTIONS,
    build_hybrid_params,
    load_inputs,
    no_intervention_objective,
    series_from_row,
)
from analyze_new_york_footprint_lp_boundary import lp_size_estimate
from learn_recovery_laws import build_event_action_frame
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters


EPS = 1e-12
SCREEN_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
STRATEGY_LABELS = {
    "small_signal_only": "small-signal value only",
    "finite_value_only": "finite value only",
    "footprint_only": "observed footprint only",
    "od_structure_only": "OD exposure only",
    "value_union": "small + finite value union",
    "value_footprint_union": "value + footprint union",
    "value_footprint_structure_union": "value + footprint + OD structure union",
}
RECOMMENDED_STRATEGY = "value_footprint_structure_union"
RECOMMENDED_SCREEN_MODE = "token_columns"
SCREEN_MODE_LABELS = {
    "unit_block": "active units with all actions",
    "token_columns": "active action columns",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--boundary-dir", default="results/new_york_footprint_lp_boundary")
    parser.add_argument("--output-dir", default="results/new_york_active_set_screening")
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
    selected = pd.read_csv(
        root / args.boundary_dir / "tables" / "new_york_boundary_selected_events.csv",
        parse_dates=["event_start"],
    )

    event_metrics, unit_scores = run_screening(
        root,
        config,
        data,
        selected,
        footprint_blend=float(args.footprint_blend),
        footprint_floor=float(args.footprint_floor),
        max_relative=float(args.max_relative),
    )
    threshold_summary = build_threshold_summary(event_metrics)
    metrics = build_metrics(event_metrics, threshold_summary)

    write_table(event_metrics, table_dir / "new_york_active_set_event_metrics.csv")
    write_table(threshold_summary, table_dir / "new_york_active_set_threshold_summary.csv")
    write_table(unit_scores, table_dir / "new_york_active_set_unit_scores.csv")
    (table_dir / "new_york_active_set_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(event_metrics, threshold_summary, figure_dir)
    write_report(
        report_dir / "new_york_active_set_screening_report_zh.md",
        event_metrics,
        threshold_summary,
        metrics,
    )
    print(f"Wrote New York active-set screening analysis to {output_dir}")


def run_screening(
    root: Path,
    config: dict[str, Any],
    data: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    *,
    footprint_blend: float,
    footprint_floor: float,
    max_relative: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    selected["event_id"] = pd.to_numeric(selected["event_id"], errors="coerce").astype(int)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    summary = data["summary"].copy()
    summary = summary[(summary["status"].astype(str).eq("OPTIMAL")) & (summary["scenario"].astype(str).eq("base"))].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    summary_lookup = {(row.city, int(row.event_id)): row for row in summary.itertuples(index=False)}

    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = pd.to_numeric(footprint["event_id"], errors="coerce").astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }

    rows: list[dict[str, Any]] = []
    unit_rows: list[pd.DataFrame] = []
    for idx, selected_row in enumerate(selected.sort_values(["event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(selected_row.city)
        event_id = int(selected_row.event_id)
        print(f"[{idx}/{len(selected)}] Screening active sets for {city} event {event_id}", flush=True)
        event_key = (city, event_id)
        event_row = event_lookup[event_key]
        base_summary = series_from_row(summary_lookup[event_key])
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
        hybrid_summary = base_summary.copy()
        hybrid_summary["baseline_objective"] = no_intervention_objective(hybrid_params)
        hybrid_summary["optimized_objective"] = np.nan
        hybrid_summary["recoverable_fraction"] = np.nan
        hybrid_full = build_event_action_frame(hybrid_params, hybrid_summary, event_row, EMPTY_INTERVENTIONS)
        weights = footprint_weights(footprint_group, hybrid_params.units)
        event_unit_scores = build_unit_scores(hybrid_params, hybrid_full, weights)
        event_token_scores = build_token_scores(hybrid_full, weights)
        event_unit_scores.insert(0, "city", city)
        event_unit_scores.insert(1, "event_id", event_id)
        unit_rows.append(event_unit_scores)

        full_size = lp_size_estimate(hybrid_params)
        q_nnz = int(hybrid_params.q.nnz) if sparse.issparse(hybrid_params.q) else int(np.count_nonzero(hybrid_params.q))
        for requested_fraction in SCREEN_FRACTIONS:
            active_sets = build_active_sets(event_unit_scores, requested_fraction)
            for strategy, active_units in active_sets.items():
                rows.append(
                    {
                        "city": city,
                        "event_id": event_id,
                        "event_start": str(selected_row.event_start),
                        "screen_mode": "unit_block",
                        "screen_mode_label": SCREEN_MODE_LABELS["unit_block"],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "requested_screen_fraction": float(requested_fraction),
                        **screen_metrics(
                            hybrid_params,
                            event_unit_scores,
                            active_units,
                            full_size,
                            q_nnz,
                        ),
                        "footprint_blend": float(footprint_blend),
                        **diagnostics,
                    }
                )
            active_tokens = build_token_active_sets(event_token_scores, requested_fraction)
            for strategy, token_ids in active_tokens.items():
                rows.append(
                    {
                        "city": city,
                        "event_id": event_id,
                        "event_start": str(selected_row.event_start),
                        "screen_mode": "token_columns",
                        "screen_mode_label": SCREEN_MODE_LABELS["token_columns"],
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "requested_screen_fraction": float(requested_fraction),
                        **token_screen_metrics(
                            hybrid_params,
                            event_token_scores,
                            token_ids,
                            full_size,
                            q_nnz,
                            unit_scores=event_unit_scores,
                        ),
                        "footprint_blend": float(footprint_blend),
                        **diagnostics,
                    }
                )
    event_metrics = pd.DataFrame(rows).sort_values(["strategy", "requested_screen_fraction", "event_start", "event_id"])
    unit_scores = pd.concat(unit_rows, ignore_index=True) if unit_rows else pd.DataFrame()
    return event_metrics, unit_scores


def build_unit_scores(params: RecoveryLPParameters, full: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    static_cols = [
        "unit",
        "origin_exposure",
        "destination_importance",
        "b0",
        "h_total",
        "h_peak",
        "a_retention",
        "local_need",
        "out_degree",
        "in_degree",
    ]
    static = full[static_cols].drop_duplicates("unit").copy()
    value = (
        full.groupby("unit", as_index=False)
        .agg(
            small_signal_value=("marginal_resource_value", "sum"),
            finite_value=("finite_deficit_area_value", "sum"),
            max_small_signal_value=("marginal_resource_value", "max"),
            max_finite_value=("finite_deficit_area_value", "max"),
            positive_action_tokens=("marginal_resource_value", lambda values: int((pd.to_numeric(values, errors="coerce") > EPS).sum())),
        )
    )
    out = static.merge(value, on="unit", how="left")
    out["footprint_weight"] = out["unit"].astype(str).map(weights).fillna(0.0).astype(float)
    out["od_structure_score"] = 0.5 * normalize_score(out["origin_exposure"]) + 0.5 * normalize_score(out["destination_importance"])
    out["need_structure_score"] = normalize_score(out["local_need"]) * (0.25 + normalize_score(out["destination_importance"]))
    out["small_signal_rank"] = rank_pct(out["small_signal_value"])
    out["finite_value_rank"] = rank_pct(out["finite_value"])
    out["footprint_rank"] = rank_pct(out["footprint_weight"])
    out["od_structure_rank"] = rank_pct(out["od_structure_score"])
    out["need_structure_rank"] = rank_pct(out["need_structure_score"])
    out["hybrid_screen_score"] = (
        0.35 * out["small_signal_rank"]
        + 0.25 * out["finite_value_rank"]
        + 0.25 * out["footprint_rank"]
        + 0.15 * out["od_structure_rank"]
    )
    out["n_units"] = params.n_units
    return out.sort_values(["small_signal_value", "finite_value", "footprint_weight"], ascending=False).reset_index(drop=True)


def build_token_scores(full: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    token = full[
        [
            "unit",
            "t",
            "intervention",
            "delay_feasible",
            "origin_exposure",
            "destination_importance",
            "local_need",
            "marginal_resource_value",
            "finite_deficit_area_value",
            "law_exposure_term",
            "active_weighted_horizon",
        ]
    ].copy()
    token["unit"] = token["unit"].astype(str)
    token["token_id"] = (
        token["unit"].astype(str)
        + "|"
        + token["t"].astype(int).astype(str)
        + "|"
        + token["intervention"].astype(str)
    )
    token["small_signal_value"] = pd.to_numeric(token["marginal_resource_value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    token["finite_value"] = pd.to_numeric(token["finite_deficit_area_value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    token["footprint_weight"] = token["unit"].map(weights).fillna(0.0).astype(float)
    token["od_structure_score"] = np.where(
        token["intervention"].astype(str).eq("S"),
        token["origin_exposure"],
        token["destination_importance"],
    )
    token["need_structure_score"] = (
        pd.to_numeric(token["local_need"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(token["law_exposure_term"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * pd.to_numeric(token["active_weighted_horizon"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    token["feasible_action"] = pd.to_numeric(token["delay_feasible"], errors="coerce").fillna(0.0) > 0.5
    feasible = token["feasible_action"]
    for column in ["small_signal_value", "finite_value", "footprint_weight", "od_structure_score", "need_structure_score"]:
        token.loc[~feasible, column] = 0.0
    token["small_signal_rank"] = rank_pct(token["small_signal_value"])
    token["finite_value_rank"] = rank_pct(token["finite_value"])
    token["footprint_rank"] = rank_pct(token["footprint_weight"])
    token["od_structure_rank"] = rank_pct(token["od_structure_score"])
    token["need_structure_rank"] = rank_pct(token["need_structure_score"])
    token["hybrid_screen_score"] = (
        0.35 * token["small_signal_rank"]
        + 0.25 * token["finite_value_rank"]
        + 0.25 * token["footprint_rank"]
        + 0.15 * token["od_structure_rank"]
    )
    return token.sort_values(["small_signal_value", "finite_value", "footprint_weight"], ascending=False).reset_index(drop=True)


def build_active_sets(unit_scores: pd.DataFrame, requested_fraction: float) -> dict[str, set[str]]:
    return {
        "small_signal_only": top_units(unit_scores, "small_signal_value", requested_fraction),
        "finite_value_only": top_units(unit_scores, "finite_value", requested_fraction),
        "footprint_only": top_units(unit_scores, "footprint_weight", requested_fraction),
        "od_structure_only": top_units(unit_scores, "od_structure_score", requested_fraction),
        "value_union": union_sets(
            top_units(unit_scores, "small_signal_value", requested_fraction),
            top_units(unit_scores, "finite_value", requested_fraction),
        ),
        "value_footprint_union": union_sets(
            top_units(unit_scores, "small_signal_value", requested_fraction),
            top_units(unit_scores, "finite_value", requested_fraction),
            top_units(unit_scores, "footprint_weight", requested_fraction),
        ),
        "value_footprint_structure_union": union_sets(
            top_units(unit_scores, "small_signal_value", requested_fraction),
            top_units(unit_scores, "finite_value", requested_fraction),
            top_units(unit_scores, "footprint_weight", requested_fraction),
            top_units(unit_scores, "od_structure_score", requested_fraction),
            top_units(unit_scores, "need_structure_score", requested_fraction),
        ),
    }


def build_token_active_sets(token_scores: pd.DataFrame, requested_fraction: float) -> dict[str, set[str]]:
    return {
        "small_signal_only": top_tokens(token_scores, "small_signal_value", requested_fraction),
        "finite_value_only": top_tokens(token_scores, "finite_value", requested_fraction),
        "footprint_only": top_tokens(token_scores, "footprint_weight", requested_fraction),
        "od_structure_only": top_tokens(token_scores, "od_structure_score", requested_fraction),
        "value_union": union_sets(
            top_tokens(token_scores, "small_signal_value", requested_fraction),
            top_tokens(token_scores, "finite_value", requested_fraction),
        ),
        "value_footprint_union": union_sets(
            top_tokens(token_scores, "small_signal_value", requested_fraction),
            top_tokens(token_scores, "finite_value", requested_fraction),
            top_tokens(token_scores, "footprint_weight", requested_fraction),
        ),
        "value_footprint_structure_union": union_sets(
            top_tokens(token_scores, "small_signal_value", requested_fraction),
            top_tokens(token_scores, "finite_value", requested_fraction),
            top_tokens(token_scores, "footprint_weight", requested_fraction),
            top_tokens(token_scores, "od_structure_score", requested_fraction),
            top_tokens(token_scores, "need_structure_score", requested_fraction),
        ),
    }


def screen_metrics(
    params: RecoveryLPParameters,
    unit_scores: pd.DataFrame,
    active_units: set[str],
    full_size: dict[str, float],
    q_nnz: int,
) -> dict[str, Any]:
    active = unit_scores["unit"].astype(str).isin(active_units)
    active_count = int(active.sum())
    n_units = int(params.n_units)
    restricted = restricted_lp_size_estimate(
        n_units=n_units,
        active_units=active_count,
        q_nnz=q_nnz,
        horizon=int(params.horizon),
        use_pwl=params.u_segment_cap is not None and params.segment_effectiveness is not None,
        segment_count=next_segment_count(params),
        delays=params.delays,
    )
    return {
        "n_units": n_units,
        "active_unit_count": active_count,
        "active_unit_fraction": active_count / max(n_units, 1),
        "full_action_tokens": int(full_size["estimated_action_tokens"]),
        "active_action_tokens": int(active_count * params.horizon * len(INTERVENTIONS)),
        "active_action_token_fraction": active_count / max(n_units, 1),
        "full_estimated_total_variables": int(full_size["estimated_total_variables"]),
        "restricted_estimated_total_variables": int(restricted["estimated_total_variables"]),
        "restricted_total_variable_fraction": restricted["estimated_total_variables"] / max(full_size["estimated_total_variables"], EPS),
        "restricted_variable_reduction": 1.0 - restricted["estimated_total_variables"] / max(full_size["estimated_total_variables"], EPS),
        "full_estimated_total_constraints": int(full_size["estimated_total_constraints"]),
        "restricted_estimated_total_constraints": int(restricted["estimated_total_constraints"]),
        "restricted_constraint_fraction": restricted["estimated_total_constraints"] / max(full_size["estimated_total_constraints"], EPS),
        "small_signal_value_capture": mass_capture(unit_scores, active, "small_signal_value"),
        "finite_value_capture": mass_capture(unit_scores, active, "finite_value"),
        "footprint_mass_capture": mass_capture(unit_scores, active, "footprint_weight"),
        "origin_exposure_capture": mass_capture(unit_scores, active, "origin_exposure"),
        "destination_importance_capture": mass_capture(unit_scores, active, "destination_importance"),
        "local_need_capture": mass_capture(unit_scores, active, "local_need"),
        "positive_footprint_unit_share": positive_unit_share(unit_scores, active, "footprint_weight"),
        "positive_value_unit_share": positive_unit_share(unit_scores, active, "small_signal_value"),
        "top5_small_units_share": top_share(unit_scores, active, "small_signal_value", 0.05),
        "top5_finite_units_share": top_share(unit_scores, active, "finite_value", 0.05),
        "top5_footprint_units_share": top_share(unit_scores, active, "footprint_weight", 0.05),
    }


def token_screen_metrics(
    params: RecoveryLPParameters,
    token_scores: pd.DataFrame,
    active_token_ids: set[str],
    full_size: dict[str, float],
    q_nnz: int,
    *,
    unit_scores: pd.DataFrame,
) -> dict[str, Any]:
    active = token_scores["token_id"].astype(str).isin(active_token_ids)
    active_token_count = int(active.sum())
    active_units = set(token_scores.loc[active, "unit"].astype(str))
    active_unit_mask = unit_scores["unit"].astype(str).isin(active_units)
    restricted = restricted_token_lp_size_estimate(
        n_units=int(params.n_units),
        active_tokens=active_token_count,
        q_nnz=q_nnz,
        horizon=int(params.horizon),
        use_pwl=params.u_segment_cap is not None and params.segment_effectiveness is not None,
        segment_count=next_segment_count(params),
    )
    return {
        "n_units": int(params.n_units),
        "active_unit_count": int(active_unit_mask.sum()),
        "active_unit_fraction": float(active_unit_mask.sum() / max(params.n_units, 1)),
        "full_action_tokens": int(full_size["estimated_action_tokens"]),
        "active_action_tokens": active_token_count,
        "active_action_token_fraction": active_token_count / max(int(full_size["estimated_action_tokens"]), 1),
        "full_estimated_total_variables": int(full_size["estimated_total_variables"]),
        "restricted_estimated_total_variables": int(restricted["estimated_total_variables"]),
        "restricted_total_variable_fraction": restricted["estimated_total_variables"] / max(full_size["estimated_total_variables"], EPS),
        "restricted_variable_reduction": 1.0 - restricted["estimated_total_variables"] / max(full_size["estimated_total_variables"], EPS),
        "full_estimated_total_constraints": int(full_size["estimated_total_constraints"]),
        "restricted_estimated_total_constraints": int(restricted["estimated_total_constraints"]),
        "restricted_constraint_fraction": restricted["estimated_total_constraints"] / max(full_size["estimated_total_constraints"], EPS),
        "small_signal_value_capture": mass_capture(token_scores, active, "small_signal_value"),
        "finite_value_capture": mass_capture(token_scores, active, "finite_value"),
        "footprint_mass_capture": mass_capture(unit_scores, active_unit_mask, "footprint_weight"),
        "origin_exposure_capture": mass_capture(unit_scores, active_unit_mask, "origin_exposure"),
        "destination_importance_capture": mass_capture(unit_scores, active_unit_mask, "destination_importance"),
        "local_need_capture": mass_capture(unit_scores, active_unit_mask, "local_need"),
        "positive_footprint_unit_share": positive_unit_share(unit_scores, active_unit_mask, "footprint_weight"),
        "positive_value_unit_share": positive_unit_share(unit_scores, active_unit_mask, "small_signal_value"),
        "top5_small_units_share": top_share(unit_scores, active_unit_mask, "small_signal_value", 0.05),
        "top5_finite_units_share": top_share(unit_scores, active_unit_mask, "finite_value", 0.05),
        "top5_footprint_units_share": top_share(unit_scores, active_unit_mask, "footprint_weight", 0.05),
    }


def restricted_lp_size_estimate(
    *,
    n_units: int,
    active_units: int,
    q_nnz: int,
    horizon: int,
    use_pwl: bool,
    segment_count: int,
    delays: dict[str, int],
) -> dict[str, int | float]:
    n = int(n_units)
    m = int(active_units)
    t = int(horizon)
    k_count = len(INTERVENTIONS)
    state_vars = 5 * n * (t + 1)
    action_u_vars = k_count * m * t
    effect_vars = k_count * m * t
    segment_vars = k_count * m * t * int(segment_count) if use_pwl else 0
    total_vars = state_vars + action_u_vars + effect_vars + segment_vars

    initial_constraints = 3 * n
    transition_constraints = 3 * n * t
    local_access_constraints = 2 * n * (t + 1)
    segment_sum_constraints = k_count * m * t if use_pwl else 0
    segment_cap_constraints = k_count * m * t * int(segment_count) if use_pwl else 0
    effectiveness_constraints = k_count * m * t
    deployment_cap_constraints = k_count * m * t
    delay_constraints = sum(max(0, min(t, int(delays.get(key, 0)))) * m for key in INTERVENTIONS)
    budget_constraints = t + 1
    total_constraints = (
        initial_constraints
        + transition_constraints
        + local_access_constraints
        + segment_sum_constraints
        + segment_cap_constraints
        + effectiveness_constraints
        + deployment_cap_constraints
        + delay_constraints
        + budget_constraints
    )
    return {
        "estimated_total_variables": int(total_vars),
        "estimated_total_constraints": int(total_constraints),
        "estimated_access_nonzero_terms": int(q_nnz * (t + 1)),
    }


def restricted_token_lp_size_estimate(
    *,
    n_units: int,
    active_tokens: int,
    q_nnz: int,
    horizon: int,
    use_pwl: bool,
    segment_count: int,
) -> dict[str, int | float]:
    n = int(n_units)
    a = int(active_tokens)
    t = int(horizon)
    state_vars = 5 * n * (t + 1)
    action_u_vars = a
    effect_vars = a
    segment_vars = a * int(segment_count) if use_pwl else 0
    total_vars = state_vars + action_u_vars + effect_vars + segment_vars

    initial_constraints = 3 * n
    transition_constraints = 3 * n * t
    local_access_constraints = 2 * n * (t + 1)
    segment_sum_constraints = a if use_pwl else 0
    segment_cap_constraints = a * int(segment_count) if use_pwl else 0
    effectiveness_constraints = a
    deployment_cap_constraints = a
    budget_constraints = t + 1
    total_constraints = (
        initial_constraints
        + transition_constraints
        + local_access_constraints
        + segment_sum_constraints
        + segment_cap_constraints
        + effectiveness_constraints
        + deployment_cap_constraints
        + budget_constraints
    )
    return {
        "estimated_total_variables": int(total_vars),
        "estimated_total_constraints": int(total_constraints),
        "estimated_access_nonzero_terms": int(q_nnz * (t + 1)),
    }


def build_threshold_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    agg_cols = [
        "active_unit_count",
        "active_unit_fraction",
        "active_action_token_fraction",
        "restricted_total_variable_fraction",
        "restricted_variable_reduction",
        "restricted_constraint_fraction",
        "small_signal_value_capture",
        "finite_value_capture",
        "footprint_mass_capture",
        "origin_exposure_capture",
        "destination_importance_capture",
        "local_need_capture",
        "positive_footprint_unit_share",
        "positive_value_unit_share",
        "top5_small_units_share",
        "top5_finite_units_share",
        "top5_footprint_units_share",
    ]
    grouped = (
        event_metrics.groupby(
            ["screen_mode", "screen_mode_label", "strategy", "strategy_label", "requested_screen_fraction"],
            as_index=False,
        )[agg_cols]
        .agg(["mean", "min"])
    )
    grouped.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    grouped = grouped.rename(
        columns={
            "screen_mode_": "screen_mode",
            "screen_mode_label_": "screen_mode_label",
            "strategy_": "strategy",
            "strategy_label_": "strategy_label",
            "requested_screen_fraction_": "requested_screen_fraction",
        }
    )
    return grouped.sort_values(["screen_mode", "strategy", "requested_screen_fraction"]).reset_index(drop=True)


def build_metrics(event_metrics: pd.DataFrame, threshold_summary: pd.DataFrame) -> dict[str, Any]:
    recommended = threshold_summary[
        threshold_summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & threshold_summary["strategy"].eq(RECOMMENDED_STRATEGY)
    ].copy()
    unit_recommended = threshold_summary[
        threshold_summary["screen_mode"].eq("unit_block")
        & threshold_summary["strategy"].eq(RECOMMENDED_STRATEGY)
    ].copy()
    base = row_at_fraction(recommended, 0.10)
    unit_base = row_at_fraction(unit_recommended, 0.10)
    feasible = recommended[
        (recommended["small_signal_value_capture_min"] >= 0.95)
        & (recommended["finite_value_capture_min"] >= 0.95)
        & (recommended["footprint_mass_capture_min"] >= 0.80)
    ].sort_values("active_action_token_fraction_mean")
    smallest = feasible.iloc[0] if not feasible.empty else None
    best_footprint_low_fraction = recommended[recommended["active_action_token_fraction_mean"] <= 0.30].sort_values(
        ["footprint_mass_capture_mean", "finite_value_capture_mean"],
        ascending=[False, False],
    )
    best = best_footprint_low_fraction.iloc[0] if not best_footprint_low_fraction.empty else None
    metrics = {
        "n_events": int(event_metrics[["city", "event_id"]].drop_duplicates().shape[0]) if not event_metrics.empty else 0,
        "n_strategies": int(event_metrics["strategy"].nunique()) if not event_metrics.empty else 0,
        "recommended_strategy": RECOMMENDED_STRATEGY,
        "recommended_screen_mode": RECOMMENDED_SCREEN_MODE,
        "recommended_fraction": 0.10,
        "recommended_mean_active_units": safe_float(base, "active_unit_count_mean"),
        "recommended_mean_active_unit_fraction": safe_float(base, "active_unit_fraction_mean"),
        "recommended_mean_active_action_token_fraction": safe_float(base, "active_action_token_fraction_mean"),
        "recommended_mean_variable_reduction": safe_float(base, "restricted_variable_reduction_mean"),
        "recommended_mean_constraint_fraction": safe_float(base, "restricted_constraint_fraction_mean"),
        "recommended_mean_small_signal_capture": safe_float(base, "small_signal_value_capture_mean"),
        "recommended_min_small_signal_capture": safe_float(base, "small_signal_value_capture_min"),
        "recommended_mean_finite_capture": safe_float(base, "finite_value_capture_mean"),
        "recommended_min_finite_capture": safe_float(base, "finite_value_capture_min"),
        "recommended_mean_footprint_capture": safe_float(base, "footprint_mass_capture_mean"),
        "recommended_min_footprint_capture": safe_float(base, "footprint_mass_capture_min"),
        "recommended_mean_destination_importance_capture": safe_float(base, "destination_importance_capture_mean"),
        "unit_block_10pct_active_unit_fraction": safe_float(unit_base, "active_unit_fraction_mean"),
        "unit_block_10pct_small_signal_capture": safe_float(unit_base, "small_signal_value_capture_mean"),
        "unit_block_10pct_finite_capture": safe_float(unit_base, "finite_value_capture_mean"),
        "unit_block_10pct_footprint_capture": safe_float(unit_base, "footprint_mass_capture_mean"),
        "smallest_all_threshold_fraction": safe_float(smallest, "requested_screen_fraction"),
        "smallest_all_threshold_active_fraction": safe_float(smallest, "active_unit_fraction_mean"),
        "smallest_all_threshold_active_action_token_fraction": safe_float(smallest, "active_action_token_fraction_mean"),
        "smallest_all_threshold_variable_reduction": safe_float(smallest, "restricted_variable_reduction_mean"),
        "smallest_all_threshold_small_capture_min": safe_float(smallest, "small_signal_value_capture_min"),
        "smallest_all_threshold_finite_capture_min": safe_float(smallest, "finite_value_capture_min"),
        "smallest_all_threshold_footprint_capture_min": safe_float(smallest, "footprint_mass_capture_min"),
        "best_under_30pct_fraction": safe_float(best, "requested_screen_fraction"),
        "best_under_30pct_active_fraction": safe_float(best, "active_unit_fraction_mean"),
        "best_under_30pct_active_action_token_fraction": safe_float(best, "active_action_token_fraction_mean"),
        "best_under_30pct_footprint_capture": safe_float(best, "footprint_mass_capture_mean"),
        "best_under_30pct_finite_capture": safe_float(best, "finite_value_capture_mean"),
        "interpretation": (
            "Active-set screening is a decomposition design result: it preserves all New York zones and OD access "
            "constraints, but restricts intervention action variables to a signal-rich candidate set."
        ),
    }
    return metrics


def make_figures(event_metrics: pd.DataFrame, threshold_summary: pd.DataFrame, figure_dir: Path) -> None:
    make_frontier_figure(threshold_summary, figure_dir / "new_york_active_set_capture_frontier.png")
    make_size_figure(threshold_summary, figure_dir / "new_york_active_set_size_reduction.png")
    make_strategy_tradeoff_figure(threshold_summary, figure_dir / "new_york_active_set_strategy_tradeoff.png")


def make_frontier_figure(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[
        summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & summary["strategy"].eq(RECOMMENDED_STRATEGY)
    ].sort_values("active_action_token_fraction_mean")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    series = [
        ("small_signal_value_capture_mean", "small-signal value", "#4C78A8"),
        ("finite_value_capture_mean", "finite value", "#F58518"),
        ("footprint_mass_capture_mean", "observed footprint", "#E45756"),
        ("destination_importance_capture_mean", "OD destination importance", "#54A24B"),
    ]
    for column, label, color in series:
        ax.plot(frame["active_action_token_fraction_mean"], frame[column], marker="o", linewidth=2.0, label=label, color=color)
    ax.set_xlabel("Active action-token fraction")
    ax.set_ylabel("Captured mass")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("New York active-set screening frontier")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_size_figure(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[
        summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & summary["strategy"].eq(RECOMMENDED_STRATEGY)
        & summary["requested_screen_fraction"].isin([0.05, 0.10, 0.20, 0.30])
    ].sort_values("requested_screen_fraction")
    if frame.empty:
        return
    labels = [f"{100 * value:.0f}%" for value in frame["requested_screen_fraction"]]
    x = np.arange(len(frame))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, frame["active_action_token_fraction_mean"], width, label="action-token fraction", color="#4C78A8")
    ax.bar(x + width / 2, frame["restricted_total_variable_fraction_mean"], width, label="total-variable fraction", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of full New York formulation")
    ax.set_xlabel("Requested per-signal top share")
    ax.set_title("Active actions reduce variables while full state/access stays global")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_strategy_tradeoff_figure(summary: pd.DataFrame, path: Path) -> None:
    frame = summary[
        summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & summary["requested_screen_fraction"].eq(0.10)
    ].copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.scatter(
        frame["finite_value_capture_mean"],
        frame["footprint_mass_capture_mean"],
        s=90 * np.maximum(frame["active_unit_fraction_mean"], 0.03),
        color="#4C78A8",
        alpha=0.8,
    )
    for row in frame.itertuples(index=False):
        ax.annotate(str(row.strategy).replace("_", "\n"), (row.finite_value_capture_mean, row.footprint_mass_capture_mean), fontsize=7)
    ax.set_xlabel("Finite-value capture")
    ax.set_ylabel("Observed-footprint capture")
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Screening signals are complementary at the 10% setting")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(path: Path, event_metrics: pd.DataFrame, summary: pd.DataFrame, metrics: dict[str, Any]) -> None:
    recommended = summary[
        summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & summary["strategy"].eq(RECOMMENDED_STRATEGY)
    ].copy()
    unit_recommended = summary[
        summary["screen_mode"].eq("unit_block")
        & summary["strategy"].eq(RECOMMENDED_STRATEGY)
        & summary["requested_screen_fraction"].eq(0.10)
    ].copy()
    display_cols = [
        "requested_screen_fraction",
        "active_unit_count_mean",
        "active_unit_fraction_mean",
        "active_action_token_fraction_mean",
        "restricted_total_variable_fraction_mean",
        "small_signal_value_capture_mean",
        "finite_value_capture_mean",
        "footprint_mass_capture_mean",
        "destination_importance_capture_mean",
        "small_signal_value_capture_min",
        "finite_value_capture_min",
        "footprint_mass_capture_min",
    ]
    strategy_10 = summary[
        summary["screen_mode"].eq(RECOMMENDED_SCREEN_MODE)
        & summary["requested_screen_fraction"].eq(0.10)
    ].copy()
    strategy_cols = [
        "strategy",
        "active_action_token_fraction_mean",
        "active_unit_fraction_mean",
        "small_signal_value_capture_mean",
        "finite_value_capture_mean",
        "footprint_mass_capture_mean",
        "destination_importance_capture_mean",
    ]
    lines = [
        "# New York Active-Set Screening V44",
        "",
        "## 结论",
        "",
        (
            "这版不是求解 New York footprint-aware LP 最优解，而是为后续 New York-scale decomposition 设计 active action set。"
            "做法是保留 1,940 个 New York OD zones、完整状态演化和完整 OD access-loss 约束，只限制哪些 unit-time-intervention action columns 可以产生 R/C/S deployment variables。"
        ),
        "",
        (
            f"推荐的 `{RECOMMENDED_SCREEN_MODE}/{RECOMMENDED_STRATEGY}` 在每类 token 信号取 top 10% 后取并集，平均 active units 为 "
            f"{metrics['recommended_mean_active_units']:.1f}，约占全城 {metrics['recommended_mean_active_unit_fraction']:.1%}，"
            f"实际 active action tokens 约占全量 {metrics['recommended_mean_active_action_token_fraction']:.1%}。"
            f"在仍保留全城状态和 access 约束的情况下，估计总变量数降低 {metrics['recommended_mean_variable_reduction']:.1%}。"
        ),
        "",
        (
            f"这个 active set 平均捕获 small-signal value {metrics['recommended_mean_small_signal_capture']:.1%}，"
            f"finite value {metrics['recommended_mean_finite_capture']:.1%}，observed footprint mass "
            f"{metrics['recommended_mean_footprint_capture']:.1%}，OD destination importance "
            f"{metrics['recommended_mean_destination_importance_capture']:.1%}。"
            f"最弱事件中的 small/finite/footprint 捕获分别为 "
            f"{metrics['recommended_min_small_signal_capture']:.1%}/"
            f"{metrics['recommended_min_finite_capture']:.1%}/"
            f"{metrics['recommended_min_footprint_capture']:.1%}。"
        ),
        "",
        (
            f"如果要求最弱事件也同时达到 small-signal >=95%、finite >=95%、footprint >=80%，"
            f"当前联合筛选需要 requested fraction {metrics['smallest_all_threshold_fraction']:.0%}，"
            f"实际 active action tokens {metrics['smallest_all_threshold_active_action_token_fraction']:.1%}，"
            f"active units {metrics['smallest_all_threshold_active_fraction']:.1%}，"
            f"总变量仍只能降低 {metrics['smallest_all_threshold_variable_reduction']:.1%}。"
            "这说明 New York 的恢复价值高度分散，不能把它压成少数区域问题。"
        ),
        "",
        (
            "解释：这说明 New York 未闭合的问题不一定需要回退到小 OD。更自然的下一步是大规模全城 LP 的 decomposition/warm-start："
            "全城结构仍在模型里，但候选投放动作先由 value、footprint 和 OD structure 共同筛选。unit-block 初版过粗，top 10% 联合筛选只能捕获 "
            f"{metrics['unit_block_10pct_small_signal_capture']:.1%} small-signal value 和 "
            f"{metrics['unit_block_10pct_finite_capture']:.1%} finite value；token-column 筛选更贴近 LP 的变量结构。"
        ),
        "",
        "## Recommended Frontier",
        "",
        table_to_markdown(recommended[display_cols]),
        "",
        "## Unit-Block Baseline at 10%",
        "",
        table_to_markdown(unit_recommended[display_cols]),
        "",
        "## Strategy Comparison at 10%",
        "",
        table_to_markdown(strategy_10[strategy_cols].sort_values("footprint_mass_capture_mean", ascending=False)),
        "",
        "## 写作含义",
        "",
        "1. 这不是“逃避 New York 大规模”，而是把大规模问题拆成全城状态/约束 + 结构化 active action columns。",
        "2. unit-block 筛选告诉我们 New York 的 recovery value 不是只集中在少数 zones；这反而支持 column generation，而不是简单删城市区域。",
        "3. footprint-only 能覆盖 footprint，但会弱化 recovery value；small/finite value 只看恢复信号则可能漏掉 event footprint。联合筛选说明 city structure law 和 event footprint 是互补信号。",
        "4. 后续真正需要模型部分完成的是：在这些 active columns 上重解 restricted LP，再做 column generation 或 active-set expansion，检验是否逼近 full New York optimum。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_units(unit_scores: pd.DataFrame, column: str, fraction: float) -> set[str]:
    n = len(unit_scores)
    count = max(1, int(math.ceil(float(fraction) * n)))
    return set(unit_scores.sort_values(column, ascending=False).head(count)["unit"].astype(str))


def top_tokens(token_scores: pd.DataFrame, column: str, fraction: float) -> set[str]:
    feasible = token_scores[token_scores["feasible_action"]].copy()
    if feasible.empty:
        return set()
    count = max(1, int(math.ceil(float(fraction) * len(feasible))))
    return set(feasible.sort_values(column, ascending=False).head(count)["token_id"].astype(str))


def union_sets(*sets: set[str]) -> set[str]:
    out: set[str] = set()
    for values in sets:
        out |= values
    return out


def mass_capture(unit_scores: pd.DataFrame, active: pd.Series, column: str) -> float:
    values = pd.to_numeric(unit_scores[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    return float(values[active].sum() / total) if total > EPS else np.nan


def positive_unit_share(unit_scores: pd.DataFrame, active: pd.Series, column: str) -> float:
    values = pd.to_numeric(unit_scores[column], errors="coerce").fillna(0.0)
    positive = values > EPS
    return float((active & positive).sum() / max(int(positive.sum()), 1))


def top_share(unit_scores: pd.DataFrame, active: pd.Series, column: str, fraction: float) -> float:
    count = max(1, int(math.ceil(float(fraction) * len(unit_scores))))
    top_index = unit_scores.sort_values(column, ascending=False).head(count).index
    return float(active.loc[top_index].sum() / max(count, 1))


def normalize_score(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    return values / total if total > EPS else pd.Series(0.0, index=values.index)


def rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(numeric) <= 1:
        return pd.Series(1.0, index=numeric.index)
    return numeric.rank(method="average", pct=True)


def next_segment_count(params: RecoveryLPParameters) -> int:
    if params.u_segment_cap is None:
        return 0
    return int(next(iter(params.u_segment_cap.values())).shape[2])


def row_at_fraction(frame: pd.DataFrame, fraction: float) -> pd.Series | None:
    if frame.empty:
        return None
    diff = (frame["requested_screen_fraction"] - float(fraction)).abs()
    return frame.loc[diff.idxmin()]


def safe_float(row: pd.Series | None, column: str) -> float:
    if row is None:
        return float("nan")
    try:
        value = float(row[column])
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def table_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    return frame.to_markdown(index=False)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()

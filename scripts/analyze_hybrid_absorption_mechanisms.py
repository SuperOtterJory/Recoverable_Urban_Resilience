"""Explain why hybrid-footprint first-order shifts are absorbed by the full LP.

This diagnostic uses the V35 representative hybrid LP solutions.  It does
not solve new LPs.  Instead, it rebuilds the base and hybrid calibrated
action-token fields, joins them to the optimized support, and measures which
finite-budget mechanisms limit the conversion from footprint-sensitive
finite-value shifts into final selected actions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from learn_recovery_laws import build_event_action_frame, prepare_interventions
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


EPS = 1e-12
INTERVENTIONS = ("R", "C", "S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/hybrid_absorption_mechanisms")
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
    base_selected_actions = pd.read_csv(root / "results" / "event_optimization" / "tables" / "event_optimization_interventions.csv")
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"].eq("OPTIMAL")) & (base_summary["scenario"].eq("base"))].copy()

    for frame in [event_metrics, selected_events, hybrid_selected_actions, base_selected_actions, base_summary, data["events"], data["footprint_zone"]]:
        if "event_id" in frame:
            frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")

    event_lookup = {
        (row.city, int(row.event_id)): row for row in data["events"].dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    base_summary_lookup = {
        (row.city, int(row.event_id)): row for row in base_summary.dropna(subset=["event_id"]).itertuples(index=False)
    }
    v35_lookup = {
        (row.city, int(row.event_id)): row for row in event_metrics.dropna(subset=["event_id"]).itertuples(index=False)
    }
    selected_lookup = {
        (row.city, int(row.event_id)): row for row in selected_events.dropna(subset=["event_id"]).itertuples(index=False)
    }
    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = footprint["event_id"].astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }

    base_prepared = prepare_interventions_with_caps(base_selected_actions)
    hybrid_prepared = prepare_interventions_with_caps(hybrid_selected_actions, scenario="base")

    rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    ok_events = event_metrics[
        event_metrics["hybrid_status"].astype(str).eq("OPTIMAL")
        & event_metrics["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)
    for idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{len(ok_events)}] Explaining absorption for {city} event {event_id}", flush=True)
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

        base_event_actions = base_prepared[
            base_prepared["city"].astype(str).eq(city)
            & base_prepared["event_id"].eq(event_id)
            & base_prepared["scenario"].astype(str).eq("base")
        ].copy()
        hybrid_event_actions = hybrid_prepared[
            hybrid_prepared["city"].astype(str).eq(city)
            & hybrid_prepared["event_id"].eq(event_id)
            & hybrid_prepared["scenario"].astype(str).eq("base")
        ].copy()
        base_full = build_event_action_frame(base_params, base_row, event_row, base_event_actions)
        hybrid_full = build_event_action_frame(hybrid_params, hybrid_summary, event_row, hybrid_event_actions)

        weights = footprint_weights(footprint_group, hybrid_params.units)
        event_row_metrics = event_absorption_metrics(
            row,
            selected_row,
            base_params,
            hybrid_params,
            base_full,
            hybrid_full,
            base_event_actions,
            hybrid_event_actions,
            weights,
            config,
        )
        rows.append(event_row_metrics)
        channel_rows.extend(channel_mix_rows(city, event_id, base_event_actions, hybrid_event_actions))
        unit_rows.extend(unit_diagnostic_rows(city, event_id, hybrid_full, hybrid_event_actions, weights))
        support_rows.extend(support_shift_rows(city, event_id, base_event_actions, hybrid_event_actions, weights))

    event_absorption = pd.DataFrame(rows).sort_values(["city", "event_id"])
    channel_mix = pd.DataFrame(channel_rows)
    unit_diagnostics = pd.DataFrame(unit_rows)
    support_shift = pd.DataFrame(support_rows)
    metrics = build_metrics(event_absorption, channel_mix, unit_diagnostics, support_shift)
    city_summary = build_city_summary(event_absorption)

    write_table(event_absorption, table_dir / "hybrid_absorption_event_metrics.csv")
    write_table(city_summary, table_dir / "hybrid_absorption_city_summary.csv")
    write_table(channel_mix, table_dir / "hybrid_absorption_channel_mix.csv")
    write_table(unit_diagnostics, table_dir / "hybrid_absorption_unit_diagnostics.csv")
    write_table(support_shift, table_dir / "hybrid_absorption_support_shift.csv")
    (table_dir / "hybrid_absorption_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(event_absorption, channel_mix, unit_diagnostics, figure_dir)
    write_report(report_dir / "hybrid_absorption_mechanisms_report_zh.md", metrics, event_absorption, city_summary, channel_mix, unit_diagnostics, support_shift)
    print(f"Wrote hybrid absorption mechanism analysis to {output_dir}")


def hybrid_summary_row(v35_row: Any, base_row: Any, params: Any, event_row: Any) -> SimpleNamespace:
    p = np.asarray(params.p, dtype=float)
    return SimpleNamespace(
        city=str(v35_row.city),
        event_id=int(v35_row.event_id),
        event_start=str(v35_row.event_start),
        event_total_precip=float(getattr(base_row, "event_total_precip", getattr(event_row, "total_precip", 0.0))),
        event_peak_precip=float(getattr(base_row, "event_peak_precip", getattr(event_row, "peak_precip", 0.0))),
        event_peak_positive_abnormal_deficit=float(getattr(base_row, "event_peak_positive_abnormal_deficit", 0.0)),
        weighted_b0=float(np.sum(p * params.b0)),
        weighted_h_total=float(np.sum(p[:, None] * params.h)),
        baseline_objective=float(v35_row.hybrid_baseline_objective),
        optimized_objective=float(v35_row.hybrid_optimized_objective),
        recoverable_fraction=float(v35_row.hybrid_recoverable_fraction),
        total_budget=float(params.total_budget),
    )


def prepare_interventions_with_caps(raw: pd.DataFrame, *, scenario: str | None = None) -> pd.DataFrame:
    df = raw.copy()
    if scenario is not None:
        df["scenario"] = scenario
    elif "scenario" not in df:
        df["scenario"] = "base"
    prepared = prepare_interventions(df)
    for col in ["event_id", "t"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    df["unit"] = df["unit"].astype(str)
    df["intervention"] = df["intervention"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    if "available_u_cap" in df:
        cap = (
            df.groupby(["city", "event_id", "scenario", "unit", "t", "intervention"], as_index=False)
            .agg(available_u_cap=("available_u_cap", "max"))
        )
        prepared = prepared.merge(cap, on=["city", "event_id", "scenario", "unit", "t", "intervention"], how="left")
    return prepared


def footprint_weights(footprint_group: pd.DataFrame, units: list[str]) -> pd.Series:
    raw = footprint_group.groupby("zone_id")["zone_weight"].sum()
    weights = pd.Series(0.0, index=pd.Index([str(unit) for unit in units], dtype=str))
    weights.loc[weights.index.intersection(raw.index.astype(str))] = raw.reindex(
        weights.index.intersection(raw.index.astype(str)),
        fill_value=0.0,
    ).to_numpy(dtype=float)
    total = float(weights.sum())
    return weights / total if total > EPS else weights


def event_absorption_metrics(
    v35_row: Any,
    selected_row: Any,
    base_params: Any,
    hybrid_params: Any,
    base_full: pd.DataFrame,
    hybrid_full: pd.DataFrame,
    base_actions: pd.DataFrame,
    hybrid_actions: pd.DataFrame,
    weights: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    city = str(v35_row.city)
    event_id = int(v35_row.event_id)
    n_units = hybrid_params.n_units
    top_n = max(1, int(math.ceil(0.05 * n_units)))

    base_support = support_sets(base_actions)
    hybrid_support = support_sets(hybrid_actions)
    footprint_top_units = set(weights.sort_values(ascending=False).head(top_n).index.astype(str))
    hybrid_selected_units = hybrid_support["units"]
    base_selected_units = base_support["units"]
    hybrid_finite_top_units = top_units_by_value(hybrid_full, "finite_deficit_area_value", top_n)
    hybrid_small_top_units = top_units_by_value(hybrid_full, "marginal_resource_value", top_n)

    base_budget = budget_usage(base_actions, base_params.period_budget, base_params.total_budget)
    hybrid_budget = budget_usage(hybrid_actions, hybrid_params.period_budget, hybrid_params.total_budget)
    base_cap = cap_and_diminishing_usage(base_actions, config)
    hybrid_cap = cap_and_diminishing_usage(hybrid_actions, config)

    base_only_units = base_selected_units - hybrid_selected_units
    hybrid_only_units = hybrid_selected_units - base_selected_units
    common_units = base_selected_units & hybrid_selected_units

    v34_delta = float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan))
    selected_delta = float(getattr(v35_row, "delta_selected_unit_footprint_mass", np.nan))
    return {
        "city": city,
        "event_id": event_id,
        "event_start": str(getattr(v35_row, "event_start", "")),
        "n_units": int(n_units),
        "v34_delta_finite_top5pct_units_footprint_mass": v34_delta,
        "v35_delta_selected_unit_footprint_mass": selected_delta,
        "first_order_to_lp_transfer_ratio": safe_div(selected_delta, v34_delta),
        "selected_unit_jaccard": float(getattr(v35_row, "selected_unit_jaccard", np.nan)),
        "selected_action_jaccard": float(getattr(v35_row, "selected_action_jaccard", np.nan)),
        "base_selected_unit_footprint_mass": float(getattr(v35_row, "base_selected_unit_footprint_mass", np.nan)),
        "hybrid_selected_unit_footprint_mass": float(getattr(v35_row, "hybrid_selected_unit_footprint_mass", np.nan)),
        "hybrid_finite_top5pct_units_footprint_mass": set_mass(weights, hybrid_finite_top_units),
        "hybrid_small_top5pct_units_footprint_mass": set_mass(weights, hybrid_small_top_units),
        "footprint_top5pct_selected_unit_share": safe_div(len(footprint_top_units & hybrid_selected_units), len(footprint_top_units)),
        "footprint_top5pct_finite_unit_share": safe_div(len(footprint_top_units & hybrid_finite_top_units), len(footprint_top_units)),
        "finite_top5pct_selected_unit_share": safe_div(len(hybrid_finite_top_units & hybrid_selected_units), len(hybrid_finite_top_units)),
        "small_top5pct_selected_unit_share": safe_div(len(hybrid_small_top_units & hybrid_selected_units), len(hybrid_small_top_units)),
        "hybrid_only_unit_count": len(hybrid_only_units),
        "base_only_unit_count": len(base_only_units),
        "common_unit_count": len(common_units),
        "hybrid_only_unit_footprint_mass": set_mass(weights, hybrid_only_units),
        "base_only_unit_footprint_mass": set_mass(weights, base_only_units),
        "hybrid_only_minus_base_only_footprint_mass": set_mass(weights, hybrid_only_units) - set_mass(weights, base_only_units),
        **prefix_dict(base_budget, "base_"),
        **prefix_dict(hybrid_budget, "hybrid_"),
        **prefix_dict(base_cap, "base_"),
        **prefix_dict(hybrid_cap, "hybrid_"),
    }


def budget_usage(actions: pd.DataFrame, period_budget: np.ndarray, total_budget: float) -> dict[str, float]:
    if actions.empty:
        return {
            "total_budget_usage": 0.0,
            "max_period_budget_usage": 0.0,
            "mean_period_budget_usage": 0.0,
            "binding_period_share_95pct": 0.0,
            "binding_period_share_99pct": 0.0,
            "early_t0_cost_share": 0.0,
            "early_t0_t1_cost_share": 0.0,
        }
    frame = actions.copy()
    frame["t"] = pd.to_numeric(frame["t"], errors="coerce").fillna(-1).astype(int)
    frame["optimized_cost"] = pd.to_numeric(frame["optimized_cost"], errors="coerce").fillna(0.0)
    by_t = frame.groupby("t")["optimized_cost"].sum()
    usage = []
    for t, budget in enumerate(np.asarray(period_budget, dtype=float)):
        usage.append(float(by_t.get(t, 0.0)) / max(float(budget), EPS))
    usage_arr = np.asarray(usage, dtype=float)
    total_cost = float(frame["optimized_cost"].sum())
    return {
        "total_budget_usage": total_cost / max(float(total_budget), EPS),
        "max_period_budget_usage": float(np.max(usage_arr)) if len(usage_arr) else 0.0,
        "mean_period_budget_usage": float(np.mean(usage_arr)) if len(usage_arr) else 0.0,
        "binding_period_share_95pct": float(np.mean(usage_arr >= 0.95)) if len(usage_arr) else 0.0,
        "binding_period_share_99pct": float(np.mean(usage_arr >= 0.99)) if len(usage_arr) else 0.0,
        "early_t0_cost_share": float(by_t.get(0, 0.0)) / max(total_cost, EPS),
        "early_t0_t1_cost_share": float(by_t.get(0, 0.0) + by_t.get(1, 0.0)) / max(total_cost, EPS),
    }


def cap_and_diminishing_usage(actions: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    if actions.empty:
        return {
            "mean_selected_u_cap_ratio": np.nan,
            "cost_weighted_selected_u_cap_ratio": np.nan,
            "selected_cost_share_cap_saturated_95pct": 0.0,
            "selected_cost_share_beyond_first_segment": 0.0,
            "selected_cost_share_beyond_second_segment": 0.0,
        }
    frame = actions.copy()
    for col in ["optimized_u", "optimized_cost", "available_u_cap"]:
        if col not in frame:
            if col == "optimized_u" and "u" in frame:
                frame[col] = frame["u"]
            elif col == "optimized_cost" and "effective_cost" in frame:
                frame[col] = frame["effective_cost"]
            else:
                frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["optimized_u", "optimized_cost", "available_u_cap"])
    frame = frame[frame["available_u_cap"] > EPS].copy()
    if frame.empty:
        return {
            "mean_selected_u_cap_ratio": np.nan,
            "cost_weighted_selected_u_cap_ratio": np.nan,
            "selected_cost_share_cap_saturated_95pct": np.nan,
            "selected_cost_share_beyond_first_segment": np.nan,
            "selected_cost_share_beyond_second_segment": np.nan,
        }
    frame["u_cap_ratio"] = (frame["optimized_u"] / frame["available_u_cap"]).clip(lower=0.0)
    total_cost = float(frame["optimized_cost"].sum())
    shares = np.asarray(config["interventions"].get("pwl_diminishing_returns", {}).get("segment_cap_shares", [1.0]), dtype=float)
    shares = shares / max(float(shares.sum()), EPS)
    first_break = float(shares[0]) if len(shares) else 1.0
    second_break = float(shares[:2].sum()) if len(shares) > 1 else 1.0
    beyond_first = np.maximum(frame["u_cap_ratio"].to_numpy(dtype=float) - first_break, 0.0)
    beyond_second = np.maximum(frame["u_cap_ratio"].to_numpy(dtype=float) - second_break, 0.0)
    cost = frame["optimized_cost"].to_numpy(dtype=float)
    return {
        "mean_selected_u_cap_ratio": float(frame["u_cap_ratio"].mean()),
        "cost_weighted_selected_u_cap_ratio": weighted_mean(frame["u_cap_ratio"], frame["optimized_cost"]),
        "selected_cost_share_cap_saturated_95pct": float(frame.loc[frame["u_cap_ratio"] >= 0.95, "optimized_cost"].sum()) / max(total_cost, EPS),
        "selected_cost_share_beyond_first_segment": float(np.sum(cost * np.minimum(beyond_first, 1.0 - first_break) / np.maximum(frame["u_cap_ratio"].to_numpy(dtype=float), EPS))) / max(total_cost, EPS),
        "selected_cost_share_beyond_second_segment": float(np.sum(cost * np.minimum(beyond_second, 1.0 - second_break) / np.maximum(frame["u_cap_ratio"].to_numpy(dtype=float), EPS))) / max(total_cost, EPS),
    }


def support_sets(actions: pd.DataFrame) -> dict[str, set[Any]]:
    if actions.empty:
        return {"actions": set(), "units": set()}
    frame = actions.copy()
    return {
        "actions": set((str(row.unit), int(row.t), str(row.intervention)) for row in frame.itertuples(index=False)),
        "units": set(frame["unit"].astype(str)),
    }


def top_units_by_value(full: pd.DataFrame, value_col: str, n: int) -> set[str]:
    table = full.groupby("unit", as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    return set(table.head(max(1, int(n)))["unit"].astype(str))


def channel_mix_rows(city: str, event_id: int, base_actions: pd.DataFrame, hybrid_actions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for calibration, actions in [("base", base_actions), ("hybrid_footprint", hybrid_actions)]:
        total = float(pd.to_numeric(actions.get("optimized_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        for intervention in INTERVENTIONS:
            mask = actions["intervention"].astype(str).eq(intervention) if not actions.empty else pd.Series(dtype=bool)
            cost = float(pd.to_numeric(actions.loc[mask, "optimized_cost"], errors="coerce").fillna(0.0).sum()) if not actions.empty else 0.0
            rows.append(
                {
                    "city": city,
                    "event_id": event_id,
                    "calibration": calibration,
                    "intervention": intervention,
                    "selected_cost": cost,
                    "selected_cost_share": cost / max(total, EPS),
                }
            )
    return rows


def unit_diagnostic_rows(
    city: str,
    event_id: int,
    hybrid_full: pd.DataFrame,
    hybrid_actions: pd.DataFrame,
    weights: pd.Series,
) -> list[dict[str, Any]]:
    top_n = max(1, int(math.ceil(0.05 * weights.size)))
    footprint_top = set(weights.sort_values(ascending=False).head(top_n).index.astype(str))
    selected_units = set(hybrid_actions["unit"].astype(str)) if not hybrid_actions.empty else set()
    finite_top = top_units_by_value(hybrid_full, "finite_deficit_area_value", top_n)
    small_top = top_units_by_value(hybrid_full, "marginal_resource_value", top_n)
    unit = (
        hybrid_full.groupby("unit", as_index=False)
        .agg(
            footprint_weight=("unit", lambda s: 0.0),
            destination_importance_rank=("destination_importance_rank", "first"),
            origin_exposure_rank=("origin_exposure_rank", "first"),
            local_need_rank=("local_need_rank", "first"),
            local_remaining_rank_mean=("local_remaining_rank", "mean"),
            active_weighted_horizon_mean=("active_weighted_horizon", "mean"),
            eta_per_cost_rank_mean=("eta_per_cost_rank", "mean"),
            small_signal_unit_value=("marginal_resource_value", "sum"),
            finite_unit_value=("finite_deficit_area_value", "sum"),
        )
        .copy()
    )
    unit["unit"] = unit["unit"].astype(str)
    unit["footprint_weight"] = weights.reindex(unit["unit"], fill_value=0.0).to_numpy(dtype=float)
    unit["small_signal_unit_rank"] = rank_pct(unit["small_signal_unit_value"].to_numpy(dtype=float))
    unit["finite_unit_rank"] = rank_pct(unit["finite_unit_value"].to_numpy(dtype=float))
    groups = {
        "footprint_top5pct_units": footprint_top,
        "hybrid_finite_top5pct_units": finite_top,
        "hybrid_small_signal_top5pct_units": small_top,
        "hybrid_selected_units": selected_units,
        "footprint_top5pct_not_selected": footprint_top - selected_units,
    }
    rows: list[dict[str, Any]] = []
    for group_name, units in groups.items():
        sub = unit[unit["unit"].isin(units)]
        rows.append(
            {
                "city": city,
                "event_id": event_id,
                "unit_group": group_name,
                "unit_count": int(len(sub)),
                "footprint_mass": float(sub["footprint_weight"].sum()),
                "mean_destination_importance_rank": safe_mean(sub, "destination_importance_rank"),
                "mean_origin_exposure_rank": safe_mean(sub, "origin_exposure_rank"),
                "mean_local_need_rank": safe_mean(sub, "local_need_rank"),
                "mean_local_remaining_rank": safe_mean(sub, "local_remaining_rank_mean"),
                "mean_active_weighted_horizon": safe_mean(sub, "active_weighted_horizon_mean"),
                "mean_eta_per_cost_rank": safe_mean(sub, "eta_per_cost_rank_mean"),
                "mean_small_signal_unit_rank": safe_mean(sub, "small_signal_unit_rank"),
                "mean_finite_unit_rank": safe_mean(sub, "finite_unit_rank"),
            }
        )
    return rows


def support_shift_rows(city: str, event_id: int, base_actions: pd.DataFrame, hybrid_actions: pd.DataFrame, weights: pd.Series) -> list[dict[str, Any]]:
    base_units = set(base_actions["unit"].astype(str)) if not base_actions.empty else set()
    hybrid_units = set(hybrid_actions["unit"].astype(str)) if not hybrid_actions.empty else set()
    groups = {
        "common_units": base_units & hybrid_units,
        "base_only_units": base_units - hybrid_units,
        "hybrid_only_units": hybrid_units - base_units,
    }
    rows = []
    for name, units in groups.items():
        rows.append(
            {
                "city": city,
                "event_id": event_id,
                "support_group": name,
                "unit_count": len(units),
                "footprint_mass": set_mass(weights, units),
            }
        )
    return rows


def build_city_summary(event_absorption: pd.DataFrame) -> pd.DataFrame:
    if event_absorption.empty:
        return pd.DataFrame()
    return (
        event_absorption.groupby("city", as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_first_order_to_lp_transfer_ratio=("first_order_to_lp_transfer_ratio", "mean"),
            mean_selected_unit_jaccard=("selected_unit_jaccard", "mean"),
            mean_hybrid_total_budget_usage=("hybrid_total_budget_usage", "mean"),
            mean_hybrid_binding_period_share_95pct=("hybrid_binding_period_share_95pct", "mean"),
            mean_hybrid_cost_weighted_u_cap_ratio=("hybrid_cost_weighted_selected_u_cap_ratio", "mean"),
            mean_hybrid_cost_share_beyond_first_segment=("hybrid_selected_cost_share_beyond_first_segment", "mean"),
            mean_footprint_top5pct_selected_unit_share=("footprint_top5pct_selected_unit_share", "mean"),
            mean_finite_top5pct_selected_unit_share=("finite_top5pct_selected_unit_share", "mean"),
            mean_hybrid_only_minus_base_only_footprint_mass=("hybrid_only_minus_base_only_footprint_mass", "mean"),
        )
        .sort_values("city")
    )


def build_metrics(
    event_absorption: pd.DataFrame,
    channel_mix: pd.DataFrame,
    unit_diagnostics: pd.DataFrame,
    support_shift: pd.DataFrame,
) -> dict[str, Any]:
    hybrid_channels = channel_mix[channel_mix["calibration"].eq("hybrid_footprint")]
    base_channels = channel_mix[channel_mix["calibration"].eq("base")]
    footprint_not_selected = unit_diagnostics[unit_diagnostics["unit_group"].eq("footprint_top5pct_not_selected")]
    footprint_top = unit_diagnostics[unit_diagnostics["unit_group"].eq("footprint_top5pct_units")]
    hybrid_selected = unit_diagnostics[unit_diagnostics["unit_group"].eq("hybrid_selected_units")]
    hybrid_only = support_shift[support_shift["support_group"].eq("hybrid_only_units")]
    base_only = support_shift[support_shift["support_group"].eq("base_only_units")]
    return {
        "n_events": int(len(event_absorption)),
        "n_cities": int(event_absorption["city"].nunique()) if not event_absorption.empty else 0,
        "mean_v34_delta_finite_top5pct_units_footprint_mass": safe_mean(event_absorption, "v34_delta_finite_top5pct_units_footprint_mass"),
        "mean_v35_delta_selected_unit_footprint_mass": safe_mean(event_absorption, "v35_delta_selected_unit_footprint_mass"),
        "mean_first_order_to_lp_transfer_ratio": safe_mean(event_absorption, "first_order_to_lp_transfer_ratio"),
        "median_first_order_to_lp_transfer_ratio": safe_median(event_absorption, "first_order_to_lp_transfer_ratio"),
        "mean_selected_unit_jaccard": safe_mean(event_absorption, "selected_unit_jaccard"),
        "mean_selected_action_jaccard": safe_mean(event_absorption, "selected_action_jaccard"),
        "mean_hybrid_total_budget_usage": safe_mean(event_absorption, "hybrid_total_budget_usage"),
        "mean_hybrid_max_period_budget_usage": safe_mean(event_absorption, "hybrid_max_period_budget_usage"),
        "mean_hybrid_binding_period_share_95pct": safe_mean(event_absorption, "hybrid_binding_period_share_95pct"),
        "mean_hybrid_binding_period_share_99pct": safe_mean(event_absorption, "hybrid_binding_period_share_99pct"),
        "mean_hybrid_cost_weighted_selected_u_cap_ratio": safe_mean(event_absorption, "hybrid_cost_weighted_selected_u_cap_ratio"),
        "mean_hybrid_selected_cost_share_cap_saturated_95pct": safe_mean(event_absorption, "hybrid_selected_cost_share_cap_saturated_95pct"),
        "mean_hybrid_selected_cost_share_beyond_first_segment": safe_mean(event_absorption, "hybrid_selected_cost_share_beyond_first_segment"),
        "mean_hybrid_selected_cost_share_beyond_second_segment": safe_mean(event_absorption, "hybrid_selected_cost_share_beyond_second_segment"),
        "mean_hybrid_early_t0_t1_cost_share": safe_mean(event_absorption, "hybrid_early_t0_t1_cost_share"),
        "mean_footprint_top5pct_selected_unit_share": safe_mean(event_absorption, "footprint_top5pct_selected_unit_share"),
        "mean_footprint_top5pct_finite_unit_share": safe_mean(event_absorption, "footprint_top5pct_finite_unit_share"),
        "mean_finite_top5pct_selected_unit_share": safe_mean(event_absorption, "finite_top5pct_selected_unit_share"),
        "mean_small_top5pct_selected_unit_share": safe_mean(event_absorption, "small_top5pct_selected_unit_share"),
        "mean_hybrid_only_minus_base_only_footprint_mass": safe_mean(event_absorption, "hybrid_only_minus_base_only_footprint_mass"),
        "mean_hybrid_only_footprint_mass": safe_mean(hybrid_only, "footprint_mass"),
        "mean_base_only_footprint_mass": safe_mean(base_only, "footprint_mass"),
        "hybrid_channel_cost_share_R": channel_share(hybrid_channels, "R"),
        "hybrid_channel_cost_share_C": channel_share(hybrid_channels, "C"),
        "hybrid_channel_cost_share_S": channel_share(hybrid_channels, "S"),
        "base_channel_cost_share_R": channel_share(base_channels, "R"),
        "base_channel_cost_share_C": channel_share(base_channels, "C"),
        "base_channel_cost_share_S": channel_share(base_channels, "S"),
        "footprint_top5_not_selected_mean_destination_rank": safe_mean(footprint_not_selected, "mean_destination_importance_rank"),
        "footprint_top5_not_selected_mean_small_signal_rank": safe_mean(footprint_not_selected, "mean_small_signal_unit_rank"),
        "footprint_top5_not_selected_mean_finite_rank": safe_mean(footprint_not_selected, "mean_finite_unit_rank"),
        "footprint_top5_mean_destination_rank": safe_mean(footprint_top, "mean_destination_importance_rank"),
        "footprint_top5_mean_small_signal_rank": safe_mean(footprint_top, "mean_small_signal_unit_rank"),
        "footprint_top5_mean_finite_rank": safe_mean(footprint_top, "mean_finite_unit_rank"),
        "hybrid_selected_mean_destination_rank": safe_mean(hybrid_selected, "mean_destination_importance_rank"),
        "hybrid_selected_mean_small_signal_rank": safe_mean(hybrid_selected, "mean_small_signal_unit_rank"),
        "hybrid_selected_mean_finite_rank": safe_mean(hybrid_selected, "mean_finite_unit_rank"),
    }


def make_figures(event_absorption: pd.DataFrame, channel_mix: pd.DataFrame, unit_diagnostics: pd.DataFrame, figure_dir: Path) -> None:
    if event_absorption.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        event_absorption["v34_delta_finite_top5pct_units_footprint_mass"],
        event_absorption["v35_delta_selected_unit_footprint_mass"],
        s=70,
        color="#2563eb",
        alpha=0.85,
    )
    for row in event_absorption.itertuples(index=False):
        ax.annotate(str(row.city), (row.v34_delta_finite_top5pct_units_footprint_mass, row.v35_delta_selected_unit_footprint_mass), fontsize=8)
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.45)
    ax.set_xlabel("V34 finite top-5% footprint-mass gain")
    ax.set_ylabel("V35 selected-unit footprint-mass gain")
    ax.set_title("First-order footprint shift mostly disappears in full LP support")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "first_order_to_lp_transfer.png", dpi=180)
    plt.close(fig)

    summary = {
        "total budget used": safe_mean(event_absorption, "hybrid_total_budget_usage"),
        "periods >95% budget": safe_mean(event_absorption, "hybrid_binding_period_share_95pct"),
        "cost-weighted cap ratio": safe_mean(event_absorption, "hybrid_cost_weighted_selected_u_cap_ratio"),
        "cost beyond first segment": safe_mean(event_absorption, "hybrid_selected_cost_share_beyond_first_segment"),
        "footprint top-5% selected": safe_mean(event_absorption, "footprint_top5pct_selected_unit_share"),
        "finite top-5% selected": safe_mean(event_absorption, "finite_top5pct_selected_unit_share"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(summary.keys())
    values = [summary[label] for label in labels]
    ax.barh(labels, values, color="#0f766e")
    ax.set_xlim(0, max(1.0, np.nanmax(values) * 1.1))
    ax.set_xlabel("Mean share / ratio")
    ax.set_title("Hybrid LP absorption diagnostics")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "absorption_mechanism_summary.png", dpi=180)
    plt.close(fig)

    if not channel_mix.empty:
        pivot = (
            channel_mix.groupby(["calibration", "intervention"])["selected_cost_share"]
            .mean()
            .unstack("intervention")
            .reindex(["base", "hybrid_footprint"])
            .fillna(0.0)
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bottom = np.zeros(len(pivot))
        colors = {"R": "#2563eb", "C": "#f59e0b", "S": "#10b981"}
        for intervention in INTERVENTIONS:
            values = pivot.get(intervention, pd.Series(0.0, index=pivot.index)).to_numpy(dtype=float)
            ax.bar(pivot.index, values, bottom=bottom, color=colors[intervention], label=intervention)
            bottom += values
        ax.set_ylabel("Mean selected cost share")
        ax.set_title("Selected channel mix changes little under hybrid footprint")
        ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "channel_mix_base_vs_hybrid.png", dpi=180)
        plt.close(fig)

    if not unit_diagnostics.empty:
        plot = unit_diagnostics[unit_diagnostics["unit_group"].isin(["footprint_top5pct_units", "hybrid_selected_units"])].copy()
        grouped = plot.groupby("unit_group", as_index=False).agg(
            mean_destination_importance_rank=("mean_destination_importance_rank", "mean"),
            mean_small_signal_unit_rank=("mean_small_signal_unit_rank", "mean"),
            mean_finite_unit_rank=("mean_finite_unit_rank", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(grouped))
        width = 0.25
        cols = [
            ("mean_destination_importance_rank", "OD rank"),
            ("mean_small_signal_unit_rank", "small-signal rank"),
            ("mean_finite_unit_rank", "finite rank"),
        ]
        for offset, (col, label) in enumerate(cols):
            ax.bar(x + (offset - 1) * width, grouped[col], width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["unit_group"], rotation=10)
        ax.set_ylabel("Mean percentile rank")
        ax.set_title("Footprint units improve finite rank but not selected-support alignment")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "footprint_vs_selected_unit_ranks.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    event_absorption: pd.DataFrame,
    city_summary: pd.DataFrame,
    channel_mix: pd.DataFrame,
    unit_diagnostics: pd.DataFrame,
    support_shift: pd.DataFrame,
) -> None:
    lines = [
        "# Hybrid Footprint Absorption Mechanisms V36",
        "",
        "本版不重新求解 LP，而是复用 V35 的 representative hybrid-footprint LP 解，重建 base/hybrid action-value 场，并解释 V34 的 first-order footprint shift 为什么只有很小一部分进入完整 LP selected support。",
        "",
        "## 主要结论",
        "",
        f"- 代表事件数为 {metrics['n_events']}，覆盖 {metrics['n_cities']} 个城市。",
        f"- V34 finite top-5% footprint-mass gain 平均为 {fmt(metrics['mean_v34_delta_finite_top5pct_units_footprint_mass'])}，V35 selected-unit footprint-mass gain 平均为 {fmt(metrics['mean_v35_delta_selected_unit_footprint_mass'])}，转化率均值为 {fmt(metrics['mean_first_order_to_lp_transfer_ratio'])}。",
        f"- hybrid/base selected support 仍高度重叠：selected-unit Jaccard {fmt(metrics['mean_selected_unit_jaccard'])}，selected-action Jaccard {fmt(metrics['mean_selected_action_jaccard'])}。",
        f"- hybrid LP 平均使用总预算 {fmt(metrics['mean_hybrid_total_budget_usage'])}；平均 {fmt(metrics['mean_hybrid_binding_period_share_95pct'])} 的时段单期预算达到 95% 以上。",
        f"- selected action 的 cost-weighted deployment-cap ratio 平均为 {fmt(metrics['mean_hybrid_cost_weighted_selected_u_cap_ratio'])}，cap saturation share 为 {fmt(metrics['mean_hybrid_selected_cost_share_cap_saturated_95pct'])}；只有 {fmt(metrics['mean_hybrid_selected_cost_share_beyond_first_segment'])} 的 selected cost 进入第一段以后。",
        f"- footprint top-5% units 只有 {fmt(metrics['mean_footprint_top5pct_selected_unit_share'])} 被 selected support 覆盖；hybrid finite top-5% units 也只有 {fmt(metrics['mean_finite_top5pct_selected_unit_share'])} 进入 selected support。",
        f"- footprint top-5% units 的 finite rank 平均为 {fmt(metrics['footprint_top5_mean_finite_rank'])}，但 small-signal rank 只有 {fmt(metrics['footprint_top5_mean_small_signal_rank'])}；hybrid selected units 的 small-signal rank 为 {fmt(metrics['hybrid_selected_mean_small_signal_rank'])}。",
        f"- hybrid-only units 相对 base-only units 的 footprint mass 增量平均为 {fmt(metrics['mean_hybrid_only_minus_base_only_footprint_mass'])}，说明确实有轻微转向，但规模很小。",
        "",
        "## 机制解释",
        "",
        "1. **first-order shift 与 full LP support 是不同对象**：V34 的 finite-value top tail 会向 observed footprint 移动，但完整 LP 同时面对总预算、单期预算、deployment caps、三段 diminishing returns 和 R/C/S 替代。first-order top tail 改变不保证最终支持集大幅重排。",
        "",
        "2. **总预算绑定，但单期预算不是主要解释**：hybrid LP 基本用满总预算，说明资源仍然稀缺；但达到 95% 单期预算的时段比例很低。因此 V36 不支持“主要是 period budget 卡住 footprint 转移”的解释。",
        "",
        "3. **caps 和 diminishing returns 是边界条件，但不是当前代表事件的主吸收机制**：selected action 的 cap ratio 只有中等水平，几乎没有 action 达到 95% cap，进入后续 diminishing segment 的成本份额也较小。这说明 footprint 没有大幅转向，并不是因为 footprint 区域被 cap 完全堵住。",
        "",
        "4. **channel mix 仍由响应机制主导**：C/S/R 的成本份额在 base 和 hybrid 之间变化很小，说明当前资源通道、delay 和效果衰减仍然控制了完整 LP 的主要形状。",
        "",
        "5. **真正的吸收发生在 value definition 层**：footprint top units 的 finite rank 很高，但 small-signal/OD-exposure rank 明显低于最终 selected units。完整 LP 的 selected support 更接近 small-signal top tail，而不是 observed footprint top tail；因此 footprint 是 magnitude signal，不是单独的 recovery-action law。",
        "",
        "## Event Metrics",
        "",
        table_to_markdown(event_absorption, max_rows=20),
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary, max_rows=20),
        "",
        "## Channel Mix",
        "",
        table_to_markdown(channel_mix, max_rows=20),
        "",
        "## Unit Diagnostics",
        "",
        table_to_markdown(unit_diagnostics, max_rows=30),
        "",
        "## Support Shift",
        "",
        table_to_markdown(support_shift, max_rows=30),
        "",
        "## 写作含义",
        "",
        "V36 可以把 V35 的边界写成更强的机制结论：event footprint 是真实数据中的 spatial signal，且能改变 magnitude-aware finite-value field；但是在当前管理 regime 下，完整 LP 的预算、caps、delay、channel substitution 和 diminishing returns 会把大部分 first-order footprint shift 吸收掉。因此论文不应声称 observed footprint 会直接决定最终投放，而应声称 recoverability 需要 footprint、OD exposure、future-loss horizon 与 finite-budget feasibility 的联合激活。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def prefix_dict(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def set_mass(weights: pd.Series, units: set[str]) -> float:
    if not units:
        return 0.0
    return float(weights.reindex(pd.Index([str(unit) for unit in units]), fill_value=0.0).sum())


def rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(pct=True, method="average").to_numpy(dtype=float)


def safe_div(num: Any, den: Any) -> float:
    num = float(num) if np.isfinite(float(num)) else np.nan
    den = float(den) if np.isfinite(float(den)) else np.nan
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def safe_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight_arr = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr >= 0)
    if not valid.any() or float(weight_arr[valid].sum()) <= EPS:
        return np.nan
    return float(np.sum(value_arr[valid] * weight_arr[valid]) / np.sum(weight_arr[valid]))


def channel_share(channels: pd.DataFrame, intervention: str) -> float:
    if channels.empty:
        return np.nan
    sub = channels[channels["intervention"].astype(str).eq(intervention)]
    return safe_mean(sub, "selected_cost_share")


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


if __name__ == "__main__":
    main()

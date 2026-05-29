"""Build action-token learning data and first-pass recoverability laws."""

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

from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


INTERVENTIONS = ("R", "C", "S")
RNG_SEED = 20260529
POLICY_SCENARIOS = (
    {"policy_scenario": "base", "budget_scale": 1.0, "delay_add_hours": 0},
    {"policy_scenario": "low_budget", "budget_scale": 0.5, "delay_add_hours": 0},
    {"policy_scenario": "high_budget", "budget_scale": 2.0, "delay_add_hours": 0},
    {"policy_scenario": "delay_2h", "budget_scale": 1.0, "delay_add_hours": 2},
    {"policy_scenario": "delay_4h", "budget_scale": 1.0, "delay_add_hours": 4},
    {"policy_scenario": "scarce_and_late", "budget_scale": 0.5, "delay_add_hours": 2},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/law_learning")
    parser.add_argument("--top-actions-per-event", type=int, default=700)
    parser.add_argument("--random-actions-per-event", type=int, default=500)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
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
    action_tokens, concentration, greedy_actions, policy_simulation, policy_replay = build_action_token_dataset(
        root,
        config,
        data,
        top_actions_per_event=args.top_actions_per_event,
        random_actions_per_event=args.random_actions_per_event,
    )
    action_tokens = add_learning_features(action_tokens)
    loco_metrics, predictions = run_leave_city_out_models(action_tokens, ridge_alpha=args.ridge_alpha)
    regime_metrics = run_leave_regime_out_models(action_tokens, ridge_alpha=args.ridge_alpha)
    policy_capture = evaluate_policy_scores(action_tokens)
    event_law = build_event_level_law_table(data["summary"], concentration)
    policy_summary = summarize_policy_simulation(policy_simulation)
    replay_summary = summarize_policy_replay(policy_replay)

    legacy_action_tokens = table_dir / "action_value_tokens.csv"
    if legacy_action_tokens.exists():
        legacy_action_tokens.unlink()
    write_table(action_tokens, table_dir / "action_value_tokens.csv.gz")
    write_table(concentration, table_dir / "event_value_concentration.csv")
    write_table(loco_metrics, table_dir / "leave_city_out_metrics.csv")
    write_table(regime_metrics, table_dir / "leave_regime_out_metrics.csv")
    write_table(predictions, table_dir / "action_value_predictions.csv")
    write_table(greedy_actions, table_dir / "greedy_oracle_actions.csv.gz")
    write_table(policy_simulation, table_dir / "budget_policy_simulation.csv")
    write_table(policy_summary, table_dir / "budget_policy_summary.csv")
    write_table(policy_replay, table_dir / "fixed_policy_replay.csv")
    write_table(replay_summary, table_dir / "fixed_policy_replay_summary.csv")
    write_table(policy_capture, table_dir / "policy_score_value_capture.csv")
    write_table(event_law, table_dir / "event_level_top_tail_law.csv")
    make_figures(action_tokens, loco_metrics, policy_capture, event_law, policy_summary, replay_summary, figure_dir)
    write_report(
        report_dir / "law_learning_report_zh.md",
        action_tokens,
        concentration,
        loco_metrics,
        regime_metrics,
        policy_capture,
        event_law,
        policy_simulation,
        policy_summary,
        policy_replay,
        replay_summary,
    )
    print(f"Wrote law-learning outputs to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "results"
    return {
        "summary": pd.read_csv(tables / "event_optimization" / "tables" / "event_optimization_summary.csv"),
        "interventions": pd.read_csv(tables / "event_optimization" / "tables" / "event_optimization_interventions.csv"),
        "events": pd.read_csv(tables / "data_mining" / "tables" / "rainfall_event_impact_details.csv", parse_dates=["event_start", "event_end"]),
        "dynamics": pd.read_csv(tables / "event_calibration" / "tables" / "event_dynamic_calibration_summary.csv"),
        "abnormal": pd.read_csv(tables / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv", parse_dates=["hour"]),
    }


def build_action_token_dataset(
    root: Path,
    config: dict[str, Any],
    data: dict[str, pd.DataFrame],
    *,
    top_actions_per_event: int,
    random_actions_per_event: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {
        (row.city, int(row.event_id)): row for row in events.itertuples(index=False)
    }
    dynamics = data["dynamics"]
    dynamic_lookup = {row["city"]: row for _, row in dynamics.iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()

    token_frames: list[pd.DataFrame] = []
    concentration_rows: list[dict[str, Any]] = []
    greedy_action_frames: list[pd.DataFrame] = []
    policy_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    total_events = len(summary)
    for idx, row in enumerate(summary.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = row.city
        event_id = int(row.event_id)
        print(f"[{idx}/{total_events}] Building action tokens for {city} event {event_id}", flush=True)
        event_row = event_lookup.get((city, event_id))
        if event_row is None:
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
        event_greedy_actions, event_policy_rows, event_replay_rows = simulate_policy_suite(full, params, config, rng)
        greedy_action_frames.append(event_greedy_actions)
        policy_rows.extend(event_policy_rows)
        replay_rows.extend(event_replay_rows)
        full = merge_greedy_oracle_labels(full, event_greedy_actions)
        concentration_rows.append(event_concentration(row, full))
        keep = choose_token_sample(full, top_actions_per_event, random_actions_per_event, rng)
        token_frames.append(full.loc[keep].copy())

    tokens = pd.concat(token_frames, ignore_index=True)
    concentration = pd.DataFrame(concentration_rows).sort_values(["city", "event_start", "event_id"])
    greedy_actions = pd.concat(greedy_action_frames, ignore_index=True) if greedy_action_frames else pd.DataFrame()
    policy_simulation = pd.DataFrame(policy_rows)
    if not policy_simulation.empty:
        policy_simulation = add_relative_policy_metrics(policy_simulation)
    policy_replay = pd.DataFrame(replay_rows)
    if not policy_replay.empty:
        policy_replay = add_relative_replay_metrics(policy_replay)
    return tokens, concentration, greedy_actions, policy_simulation, policy_replay


def prepare_interventions(interventions: pd.DataFrame) -> pd.DataFrame:
    df = interventions.copy()
    df["unit"] = df["unit"].astype(str)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(int)
    grouped = df.groupby(["city", "event_id", "scenario", "unit", "t", "intervention"], as_index=False).agg(
        optimized_u=("u", "sum"),
        optimized_e=("e", "sum"),
        optimized_cost=("effective_cost", "sum"),
    )
    grouped["t"] = pd.to_numeric(grouped["t"], errors="coerce").astype(int)
    return grouped


def build_event_action_frame(
    params: Any,
    summary_row: Any,
    event_row: Any,
    event_interventions: pd.DataFrame,
) -> pd.DataFrame:
    n = params.n_units
    horizon = params.horizon
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    p = np.asarray(params.p, dtype=float)
    destination_importance = np.asarray(q.T @ p).ravel()
    out_degree = np.diff(q.indptr).astype(float)
    in_degree = np.diff(q.tocsc().indptr).astype(float)
    passive_b, passive_ell = passive_no_intervention(params)
    local_remaining = remaining_area(passive_b)
    access_remaining = remaining_area(passive_ell)
    h_total = params.h.sum(axis=1)
    h_peak = params.h.max(axis=1)
    local_need = np.clip(params.b0 + h_total, 0.02, 1.0)
    value_eff = effective_output_values(params, passive_b, passive_ell, destination_importance)

    base = pd.DataFrame(
        {
            "unit": np.asarray(params.units, dtype=str),
            "origin_exposure": p,
            "destination_importance": destination_importance,
            "b0": params.b0,
            "h_total": h_total,
            "h_peak": h_peak,
            "a_retention": params.a,
            "local_need": local_need,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "origin_exposure_rank": rank_pct(p),
            "destination_importance_rank": rank_pct(destination_importance),
            "b0_rank": rank_pct(params.b0),
            "h_total_rank": rank_pct(h_total),
            "out_degree_rank": rank_pct(out_degree),
            "in_degree_rank": rank_pct(in_degree),
            "local_need_rank": rank_pct(local_need),
        }
    )
    rows: list[pd.DataFrame] = []
    for t in range(horizon):
        time_remaining_frac = (horizon - t) / horizon
        t_base = base.copy()
        t_base["t"] = t
        t_base["t_frac"] = t / max(horizon - 1, 1)
        t_base["time_remaining_frac"] = time_remaining_frac
        t_base["passive_b_t"] = passive_b[:, t]
        t_base["passive_ell_t"] = passive_ell[:, t]
        t_base["local_remaining_area"] = local_remaining[:, t]
        t_base["access_remaining_area"] = access_remaining[:, t]
        t_base["local_remaining_rank"] = rank_pct(local_remaining[:, t])
        t_base["access_remaining_rank"] = rank_pct(access_remaining[:, t])
        t_base["passive_b_rank"] = rank_pct(passive_b[:, t])
        t_base["passive_ell_rank"] = rank_pct(passive_ell[:, t])
        for intervention in INTERVENTIONS:
            k_base = t_base.copy()
            k_base["intervention"] = intervention
            k_base["intervention_R"] = 1.0 if intervention == "R" else 0.0
            k_base["intervention_C"] = 1.0 if intervention == "C" else 0.0
            k_base["intervention_S"] = 1.0 if intervention == "S" else 0.0
            feasible = t >= int(params.delays.get(intervention, 0))
            k_base["delay_feasible"] = float(feasible)
            k_base["delay_fraction"] = int(params.delays.get(intervention, 0)) / horizon
            eta = params.eta[intervention][:, t]
            cost = params.cost[intervention][:, t]
            k_base["eta"] = eta
            k_base["cost"] = cost
            k_base["eta_per_cost"] = eta / np.maximum(cost, 1e-12)
            k_base["eta_per_cost_rank"] = rank_pct(k_base["eta_per_cost"].to_numpy(dtype=float))
            k_base["u_cap"] = params.u_cap[intervention][:, t]
            eff_value = value_eff[intervention][:, t]
            k_base["marginal_effect_value"] = eff_value / max(float(summary_row.baseline_objective), 1e-12)
            k_base["marginal_resource_value"] = np.where(
                feasible,
                eff_value * eta / np.maximum(cost, 1e-12) / max(float(summary_row.baseline_objective), 1e-12),
                0.0,
            )
            k_base["law_future_deficit_area"] = np.where(
                intervention == "S",
                k_base["access_remaining_area"],
                k_base["local_remaining_area"],
            )
            k_base["law_exposure_term"] = np.where(
                intervention == "S",
                k_base["origin_exposure"],
                k_base["destination_importance"],
            )
            k_base["activated_bottleneck_score"] = activated_bottleneck_score(k_base, intervention)
            k_base["deficit_only_score"] = np.where(
                intervention == "S",
                k_base["access_remaining_rank"],
                k_base["local_remaining_rank"],
            )
            k_base["exposure_only_score"] = np.where(
                intervention == "S",
                k_base["origin_exposure_rank"],
                k_base["destination_importance_rank"],
            )
            k_base["structure_only_score"] = np.where(
                intervention == "S",
                k_base["origin_exposure_rank"] * (1.0 - k_base["out_degree_rank"]),
                k_base["destination_importance_rank"] * (1.0 - k_base["out_degree_rank"]),
            )
            rows.append(k_base)
    full = pd.concat(rows, ignore_index=True)
    full.insert(0, "city", params.city)
    full.insert(1, "event_id", int(summary_row.event_id))
    full.insert(2, "event_start", str(summary_row.event_start))
    full["scenario"] = "base"
    full["event_total_precip"] = float(summary_row.event_total_precip)
    full["event_peak_precip"] = float(summary_row.event_peak_precip)
    full["event_peak_positive_abnormal_deficit"] = float(summary_row.event_peak_positive_abnormal_deficit)
    full["weighted_b0"] = float(summary_row.weighted_b0)
    full["weighted_h_total"] = float(summary_row.weighted_h_total)
    full["baseline_objective"] = float(summary_row.baseline_objective)
    full["optimized_objective"] = float(summary_row.optimized_objective)
    full["recoverable_fraction"] = float(summary_row.recoverable_fraction)
    full["total_budget"] = float(summary_row.total_budget)
    full["budget_fraction_of_baseline"] = float(summary_row.total_budget) / max(float(summary_row.baseline_objective), 1e-12)
    full = full.merge(event_interventions, on=["city", "event_id", "scenario", "unit", "t", "intervention"], how="left")
    for col in ["optimized_u", "optimized_e", "optimized_cost"]:
        full[col] = full[col].fillna(0.0)
    full["selected_by_optimizer"] = full["optimized_u"] > 1e-12
    full["optimized_value_proxy"] = full["optimized_e"] * full["marginal_effect_value"]
    full["rain_regime"] = event_regime(float(summary_row.event_peak_positive_abnormal_deficit))
    return full


def simulate_policy_suite(
    full: pd.DataFrame,
    params: Any,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    segments = build_budget_segments(full, config, rng)
    policy_scores = {
        "greedy_oracle": "oracle_value_per_cost",
        "activated_bottleneck_law": "law_value_score",
        "exposure_only": "exposure_policy_score",
        "deficit_only": "deficit_policy_score",
        "structure_only": "structure_policy_score",
        "random_positive": "random_policy_score",
    }
    rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    base_greedy_actions: pd.DataFrame | None = None
    baseline_objective = float(full["baseline_objective"].iloc[0])
    optimized_objective = float(full["optimized_objective"].iloc[0])
    lp_recoverable_fraction = float(full["recoverable_fraction"].iloc[0])
    optimizer_replay = replay_optimizer_solution(full, params)
    replay_rows.append(
        replay_row(
            full,
            policy_scenario="base",
            budget_scale=1.0,
            delay_add_hours=0,
            policy_score="lp_optimizer_replay",
            allocated_cost=float(full["optimized_cost"].sum()),
            value_proxy=np.nan,
            selected_action_count=int((full["optimized_cost"] > 1e-12).sum()),
            replay_objective=optimizer_replay["objective"],
            baseline_objective=baseline_objective,
            optimized_objective=optimized_objective,
            lp_recoverable_fraction=lp_recoverable_fraction,
        )
    )
    for scenario in POLICY_SCENARIOS:
        period_budget = params.period_budget * float(scenario["budget_scale"])
        total_budget = float(params.total_budget) * float(scenario["budget_scale"])
        delay_add = int(scenario["delay_add_hours"])
        feasible = np.zeros(len(segments), dtype=bool)
        for intervention in INTERVENTIONS:
            delay = int(params.delays.get(intervention, 0)) + delay_add
            mask = segments["intervention"].eq(intervention).to_numpy()
            feasible[mask] = segments.loc[mask, "t"].to_numpy(dtype=int) >= delay
        scenario_segments = segments.loc[feasible & (segments["oracle_value_per_cost"] > 0.0)].copy()
        for policy_name, score_col in policy_scores.items():
            result = allocate_greedy_policy(
                scenario_segments,
                score_col,
                period_budget=period_budget,
                total_budget=total_budget,
            )
            rows.append(
                {
                    "city": str(full["city"].iloc[0]),
                    "event_id": int(full["event_id"].iloc[0]),
                    "event_start": str(full["event_start"].iloc[0]),
                    "policy_scenario": scenario["policy_scenario"],
                    "budget_scale": float(scenario["budget_scale"]),
                    "delay_add_hours": delay_add,
                    "policy_score": policy_name,
                    "allocated_cost": float(result["allocated_cost"]),
                    "value_proxy": float(result["value_proxy"]),
                    "selected_segment_count": int(result["selected_segment_count"]),
                    "selected_action_count": int(result["selected_action_count"]),
                    "baseline_objective": float(full["baseline_objective"].iloc[0]),
                    "recoverable_fraction": float(full["recoverable_fraction"].iloc[0]),
                    "event_peak_positive_abnormal_deficit": float(full["event_peak_positive_abnormal_deficit"].iloc[0]),
                    "event_total_precip": float(full["event_total_precip"].iloc[0]),
                }
            )
            replay = replay_policy_allocations(result["allocations"], params)
            replay_rows.append(
                replay_row(
                    full,
                    policy_scenario=str(scenario["policy_scenario"]),
                    budget_scale=float(scenario["budget_scale"]),
                    delay_add_hours=delay_add,
                    policy_score=policy_name,
                    allocated_cost=float(result["allocated_cost"]),
                    value_proxy=float(result["value_proxy"]),
                    selected_action_count=int(result["selected_action_count"]),
                    replay_objective=replay["objective"],
                    baseline_objective=baseline_objective,
                    optimized_objective=optimized_objective,
                    lp_recoverable_fraction=lp_recoverable_fraction,
                )
            )
            if scenario["policy_scenario"] == "base" and policy_name == "greedy_oracle":
                base_greedy_actions = result["allocations"].copy()
    if base_greedy_actions is None:
        base_greedy_actions = pd.DataFrame()
    return base_greedy_actions, rows, replay_rows


def build_budget_segments(full: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
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
    segment_frames: list[pd.DataFrame] = []
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
        "activated_bottleneck_score",
        "deficit_only_score",
        "exposure_only_score",
        "structure_only_score",
        "baseline_objective",
        "recoverable_fraction",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    for intervention in INTERVENTIONS:
        frame = full.loc[full["intervention"] == intervention, base_cols].copy()
        multipliers = multipliers_by_k[intervention]
        for segment_id, (share, multiplier) in enumerate(zip(segment_shares, multipliers, strict=True)):
            seg = frame.copy()
            seg["segment"] = segment_id
            seg["segment_share"] = float(share)
            seg["segment_effectiveness_multiplier"] = float(multiplier)
            seg["segment_cost_cap"] = seg["cost"] * seg["u_cap"] * float(share)
            seg["oracle_value_per_cost"] = seg["marginal_resource_value"] * float(multiplier)
            seg["law_value_score"] = seg["activated_bottleneck_score"] * float(multiplier)
            seg["deficit_policy_score"] = seg["deficit_only_score"] * float(multiplier)
            seg["exposure_policy_score"] = seg["exposure_only_score"] * float(multiplier)
            seg["structure_policy_score"] = seg["structure_only_score"] * float(multiplier)
            seg["random_policy_score"] = rng.random(len(seg))
            segment_frames.append(seg)
    segments = pd.concat(segment_frames, ignore_index=True)
    return segments[segments["segment_cost_cap"] > 1e-12].reset_index(drop=True)


def allocate_greedy_policy(
    segments: pd.DataFrame,
    score_col: str,
    *,
    period_budget: np.ndarray,
    total_budget: float,
) -> dict[str, Any]:
    if segments.empty or total_budget <= 1e-12:
        return empty_policy_result()
    scores = segments[score_col].to_numpy(dtype=float)
    valid = np.isfinite(scores) & (scores > 0)
    if not valid.any():
        return empty_policy_result()
    order = np.argsort(scores[valid])[::-1]
    valid_positions = np.flatnonzero(valid)
    ordered_positions = valid_positions[order]
    remaining_total = float(total_budget)
    remaining_period = np.asarray(period_budget, dtype=float).copy()
    allocations: list[dict[str, Any]] = []
    selected_actions: set[tuple[str, int, str]] = set()
    value_proxy = 0.0
    allocated_cost = 0.0
    for pos in ordered_positions:
        row = segments.iloc[int(pos)]
        t = int(row["t"])
        if t < 0 or t >= len(remaining_period):
            continue
        if remaining_total <= 1e-12 or np.all(remaining_period <= 1e-12):
            break
        available = min(float(row["segment_cost_cap"]), remaining_total, float(remaining_period[t]))
        if available <= 1e-12:
            continue
        value = available * float(row["oracle_value_per_cost"])
        allocations.append(
            {
                "city": row["city"],
                "event_id": int(row["event_id"]),
                "event_start": row["event_start"],
                "scenario": row["scenario"],
                "unit": str(row["unit"]),
                "t": t,
                "intervention": row["intervention"],
                "segment": int(row["segment"]),
                "allocated_cost": available,
                "allocated_u": available / max(float(row["cost"]), 1e-12),
                "value_proxy": value,
                "oracle_value_per_cost": float(row["oracle_value_per_cost"]),
                "law_value_score": float(row["law_value_score"]),
                "segment_effectiveness_multiplier": float(row["segment_effectiveness_multiplier"]),
            }
        )
        selected_actions.add((str(row["unit"]), t, str(row["intervention"])))
        remaining_total -= available
        remaining_period[t] -= available
        allocated_cost += available
        value_proxy += value
    allocation_frame = pd.DataFrame(allocations)
    return {
        "allocated_cost": allocated_cost,
        "value_proxy": value_proxy,
        "selected_segment_count": len(allocations),
        "selected_action_count": len(selected_actions),
        "allocations": allocation_frame,
    }


def empty_policy_result() -> dict[str, Any]:
    return {
        "allocated_cost": 0.0,
        "value_proxy": 0.0,
        "selected_segment_count": 0,
        "selected_action_count": 0,
        "allocations": pd.DataFrame(),
    }


def replay_row(
    full: pd.DataFrame,
    *,
    policy_scenario: str,
    budget_scale: float,
    delay_add_hours: int,
    policy_score: str,
    allocated_cost: float,
    value_proxy: float,
    selected_action_count: int,
    replay_objective: float,
    baseline_objective: float,
    optimized_objective: float,
    lp_recoverable_fraction: float,
) -> dict[str, Any]:
    base_lp_gain = max(baseline_objective - optimized_objective, 1e-12)
    replay_gain = baseline_objective - replay_objective
    return {
        "city": str(full["city"].iloc[0]),
        "event_id": int(full["event_id"].iloc[0]),
        "event_start": str(full["event_start"].iloc[0]),
        "policy_scenario": policy_scenario,
        "budget_scale": float(budget_scale),
        "delay_add_hours": int(delay_add_hours),
        "policy_score": policy_score,
        "allocated_cost": float(allocated_cost),
        "value_proxy": float(value_proxy) if np.isfinite(value_proxy) else np.nan,
        "selected_action_count": int(selected_action_count),
        "baseline_objective": baseline_objective,
        "optimized_objective": optimized_objective,
        "lp_recoverable_fraction": lp_recoverable_fraction,
        "replay_objective": float(replay_objective),
        "replay_gain": float(replay_gain),
        "replay_recoverable_fraction": float(1.0 - replay_objective / baseline_objective) if baseline_objective > 1e-12 else np.nan,
        "replay_fraction_of_base_lp_gain": float(replay_gain / base_lp_gain),
        "objective_gap_to_base_lp": float(replay_objective - optimized_objective),
        "event_peak_positive_abnormal_deficit": float(full["event_peak_positive_abnormal_deficit"].iloc[0]),
        "event_total_precip": float(full["event_total_precip"].iloc[0]),
    }


def replay_optimizer_solution(full: pd.DataFrame, params: Any) -> dict[str, float]:
    effects = empty_effects(params)
    unit_to_idx = {unit: idx for idx, unit in enumerate(params.units)}
    used = full[full["optimized_e"] > 1e-12]
    for row in used.itertuples(index=False):
        unit = str(row.unit)
        if unit not in unit_to_idx:
            continue
        effects[str(row.intervention)][unit_to_idx[unit], int(row.t)] += float(row.optimized_e)
    return replay_effects(params, effects)


def replay_policy_allocations(allocations: pd.DataFrame, params: Any) -> dict[str, float]:
    effects = empty_effects(params)
    if allocations.empty:
        return replay_effects(params, effects)
    unit_to_idx = {unit: idx for idx, unit in enumerate(params.units)}
    for row in allocations.itertuples(index=False):
        unit = str(row.unit)
        intervention = str(row.intervention)
        if unit not in unit_to_idx or intervention not in effects:
            continue
        i = unit_to_idx[unit]
        t = int(row.t)
        if t < 0 or t >= params.horizon:
            continue
        effects[intervention][i, t] += (
            float(params.eta[intervention][i, t])
            * float(row.segment_effectiveness_multiplier)
            * float(row.allocated_u)
        )
    return replay_effects(params, effects)


def empty_effects(params: Any) -> dict[str, np.ndarray]:
    return {
        key: np.zeros((params.n_units, params.horizon), dtype=float)
        for key in INTERVENTIONS
    }


def replay_effects(params: Any, effects: dict[str, np.ndarray]) -> dict[str, float]:
    b = params.b0.copy()
    r_c = np.zeros(params.n_units, dtype=float)
    r_s = np.zeros(params.n_units, dtype=float)
    objective = 0.0
    final_weighted_b = np.nan
    final_weighted_ell = np.nan
    for t in range(params.horizon + 1):
        d = np.clip(b - r_c, 0.0, 1.0)
        ell = np.clip(params.q @ d - r_s, 0.0, 1.0)
        objective += float(params.delta_t * np.sum(params.p * ell))
        if t == params.horizon:
            final_weighted_b = float(np.sum(params.p * b))
            final_weighted_ell = float(np.sum(params.p * ell))
            break
        b = np.clip(params.a * b + params.h[:, t + 1] - effects["R"][:, t], 0.0, 1.0)
        r_c = np.clip((1.0 - params.delta_c) * r_c + effects["C"][:, t], 0.0, 1.0)
        r_s = np.clip((1.0 - params.delta_s) * r_s + effects["S"][:, t], 0.0, 1.0)
    return {
        "objective": float(objective),
        "final_weighted_b": final_weighted_b,
        "final_weighted_ell": final_weighted_ell,
    }


def merge_greedy_oracle_labels(full: pd.DataFrame, greedy_actions: pd.DataFrame) -> pd.DataFrame:
    out = full.copy()
    if greedy_actions.empty:
        out["greedy_oracle_cost"] = 0.0
        out["greedy_oracle_u"] = 0.0
        out["greedy_oracle_value_proxy"] = 0.0
        out["greedy_selected_by_oracle"] = False
        return out
    grouped = greedy_actions.groupby(["city", "event_id", "scenario", "unit", "t", "intervention"], as_index=False).agg(
        greedy_oracle_cost=("allocated_cost", "sum"),
        greedy_oracle_u=("allocated_u", "sum"),
        greedy_oracle_value_proxy=("value_proxy", "sum"),
    )
    out = out.merge(grouped, on=["city", "event_id", "scenario", "unit", "t", "intervention"], how="left")
    for col in ["greedy_oracle_cost", "greedy_oracle_u", "greedy_oracle_value_proxy"]:
        out[col] = out[col].fillna(0.0)
    out["greedy_selected_by_oracle"] = out["greedy_oracle_cost"] > 1e-12
    return out


def add_relative_policy_metrics(policy_simulation: pd.DataFrame) -> pd.DataFrame:
    out = policy_simulation.copy()
    keys = ["city", "event_id", "policy_scenario"]
    oracle = out[out["policy_score"] == "greedy_oracle"][keys + ["value_proxy"]].rename(
        columns={"value_proxy": "oracle_value_proxy"}
    )
    out = out.merge(oracle, on=keys, how="left")
    out["relative_to_greedy_oracle"] = out["value_proxy"] / out["oracle_value_proxy"].replace(0.0, np.nan)
    out["value_proxy_vs_lp_recoverable_fraction"] = out["value_proxy"] / out["recoverable_fraction"].replace(0.0, np.nan)
    return out


def add_relative_replay_metrics(policy_replay: pd.DataFrame) -> pd.DataFrame:
    out = policy_replay.copy()
    keys = ["city", "event_id", "policy_scenario"]
    greedy = out[out["policy_score"] == "greedy_oracle"][keys + ["replay_gain", "replay_recoverable_fraction"]].rename(
        columns={
            "replay_gain": "greedy_replay_gain",
            "replay_recoverable_fraction": "greedy_replay_recoverable_fraction",
        }
    )
    out = out.merge(greedy, on=keys, how="left")
    out["relative_to_greedy_replay_gain"] = out["replay_gain"] / out["greedy_replay_gain"].replace(0.0, np.nan)
    out["replay_recoverable_gap_to_lp"] = out["lp_recoverable_fraction"] - out["replay_recoverable_fraction"]
    return out


def passive_no_intervention(params: Any) -> tuple[np.ndarray, np.ndarray]:
    n = params.n_units
    horizon = params.horizon
    b = np.zeros((n, horizon + 1), dtype=float)
    ell = np.zeros((n, horizon + 1), dtype=float)
    b[:, 0] = params.b0
    for t in range(horizon + 1):
        ell[:, t] = np.clip(params.q @ b[:, t], 0.0, 1.0)
        if t == horizon:
            break
        b[:, t + 1] = np.clip(params.a * b[:, t] + params.h[:, t + 1], 0.0, 1.0)
    return b, ell


def remaining_area(values: np.ndarray) -> np.ndarray:
    n, total_steps = values.shape
    horizon = total_steps - 1
    out = np.zeros((n, horizon), dtype=float)
    for t in range(horizon):
        out[:, t] = values[:, t + 1 :].sum(axis=1)
    return out


def effective_output_values(
    params: Any,
    passive_b: np.ndarray,
    passive_ell: np.ndarray,
    destination_importance: np.ndarray,
) -> dict[str, np.ndarray]:
    n = params.n_units
    horizon = params.horizon
    values = {key: np.zeros((n, horizon), dtype=float) for key in INTERVENTIONS}
    for t in range(horizon):
        steps = horizon - t
        k = np.arange(steps, dtype=float)
        r_decay = params.a[:, None] ** k[None, :]
        c_decay = (1.0 - params.delta_c) ** k
        s_decay = (1.0 - params.delta_s) ** k
        future_b = passive_b[:, t + 1 :]
        future_ell = passive_ell[:, t + 1 :]
        values["R"][:, t] = destination_importance * np.minimum(r_decay, future_b).sum(axis=1)
        values["C"][:, t] = destination_importance * np.minimum(c_decay[None, :], future_b).sum(axis=1)
        values["S"][:, t] = params.p * np.minimum(s_decay[None, :], future_ell).sum(axis=1)
    return values


def activated_bottleneck_score(frame: pd.DataFrame, intervention: str) -> np.ndarray:
    future_deficit_area = frame["law_future_deficit_area"].to_numpy(dtype=float)
    exposure = frame["law_exposure_term"].to_numpy(dtype=float)
    eta_per_cost = frame["eta_per_cost"].to_numpy(dtype=float)
    feasible = frame["delay_feasible"].to_numpy(dtype=float)
    return feasible * future_deficit_area * exposure * eta_per_cost


def choose_token_sample(
    full: pd.DataFrame,
    top_actions_per_event: int,
    random_actions_per_event: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = full.index[full["selected_by_optimizer"].to_numpy(dtype=bool)].to_numpy()
    top = full["marginal_resource_value"].nlargest(min(top_actions_per_event, len(full))).index.to_numpy()
    remaining = np.setdiff1d(full.index.to_numpy(), np.union1d(selected, top), assume_unique=False)
    random_n = min(random_actions_per_event, len(remaining))
    random = rng.choice(remaining, size=random_n, replace=False) if random_n > 0 else np.array([], dtype=int)
    return np.unique(np.concatenate([selected, top, random]))


def event_concentration(summary_row: Any, full: pd.DataFrame) -> dict[str, Any]:
    values = full["marginal_resource_value"].to_numpy(dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = float(values.sum())
    row = {
        "city": summary_row.city,
        "event_id": int(summary_row.event_id),
        "event_start": str(summary_row.event_start),
        "candidate_action_count": int(len(values)),
        "positive_action_count": int((values > 0).sum()),
        "total_marginal_value_proxy": total,
        "marginal_value_gini": gini(values),
        "top_1pct_value_share": top_share(values, 0.01),
        "top_5pct_value_share": top_share(values, 0.05),
        "top_10pct_value_share": top_share(values, 0.10),
        "recoverable_fraction": float(summary_row.recoverable_fraction),
        "baseline_objective": float(summary_row.baseline_objective),
        "event_peak_positive_abnormal_deficit": float(summary_row.event_peak_positive_abnormal_deficit),
        "event_total_precip": float(summary_row.event_total_precip),
    }
    for intervention in INTERVENTIONS:
        mask = full["intervention"].to_numpy() == intervention
        row[f"top_5pct_value_share_{intervention}"] = top_share(values[mask], 0.05)
        row[f"total_value_share_{intervention}"] = float(values[mask].sum() / total) if total > 0 else np.nan
    selected = full["selected_by_optimizer"].to_numpy(dtype=bool)
    row["optimizer_selected_value_share"] = float(values[selected].sum() / total) if total > 0 else np.nan
    row["optimizer_selected_action_share"] = float(selected.mean())
    return row


def add_learning_features(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    df["log_event_total_precip"] = np.log1p(df["event_total_precip"])
    df["log_event_peak_precip"] = np.log1p(df["event_peak_precip"])
    df["od_scarcity"] = np.clip(1.0 - df["out_degree_rank"], 0.0, 1.0)
    df["deficit_x_exposure"] = df["local_remaining_rank"] * df["destination_importance_rank"]
    df["access_x_origin"] = df["access_remaining_rank"] * df["origin_exposure_rank"]
    df["deficit_x_exposure_x_scarcity"] = df["deficit_x_exposure"] * (0.1 + df["od_scarcity"])
    df["law_score_log"] = np.log1p(1000.0 * df["activated_bottleneck_score"])
    df["target_log"] = np.log1p(1000.0 * df["marginal_resource_value"])
    df["greedy_oracle_target_log"] = np.log1p(1000.0 * df["greedy_oracle_value_proxy"])
    return df


MODEL_FEATURES = [
    "t_frac",
    "time_remaining_frac",
    "delay_fraction",
    "delay_feasible",
    "intervention_R",
    "intervention_C",
    "intervention_S",
    "origin_exposure_rank",
    "destination_importance_rank",
    "b0_rank",
    "h_total_rank",
    "local_remaining_rank",
    "access_remaining_rank",
    "passive_b_rank",
    "passive_ell_rank",
    "out_degree_rank",
    "in_degree_rank",
    "od_scarcity",
    "local_need_rank",
    "eta_per_cost_rank",
    "a_retention",
    "budget_fraction_of_baseline",
    "log_event_total_precip",
    "log_event_peak_precip",
    "event_peak_positive_abnormal_deficit",
    "weighted_b0",
    "weighted_h_total",
    "deficit_x_exposure",
    "access_x_origin",
    "deficit_x_exposure_x_scarcity",
    "activated_bottleneck_score",
    "law_score_log",
]


def run_leave_city_out_models(tokens: pd.DataFrame, *, ridge_alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for city in sorted(tokens["city"].unique()):
        train = tokens[tokens["city"] != city].copy()
        test = tokens[tokens["city"] == city].copy()
        model = fit_ridge(train[MODEL_FEATURES], train["target_log"], alpha=ridge_alpha)
        pred_log = predict_ridge(model, test[MODEL_FEATURES])
        test["predicted_value_surrogate"] = np.expm1(pred_log) / 1000.0
        prediction_frames.append(
            test[
                [
                    "city",
                    "event_id",
                    "unit",
                    "t",
                    "intervention",
                    "marginal_resource_value",
                    "activated_bottleneck_score",
                    "predicted_value_surrogate",
                    "selected_by_optimizer",
                    "greedy_selected_by_oracle",
                    "greedy_oracle_value_proxy",
                ]
            ].copy()
        )
        metric_rows.append({"split": "leave_city_out", "heldout": city, **prediction_metrics(test, "predicted_value_surrogate")})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return pd.DataFrame(metric_rows), predictions


def run_leave_regime_out_models(tokens: pd.DataFrame, *, ridge_alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in ["low", "medium", "high"]:
        train = tokens[tokens["rain_regime"] != regime].copy()
        test = tokens[tokens["rain_regime"] == regime].copy()
        if train.empty or test.empty:
            continue
        model = fit_ridge(train[MODEL_FEATURES], train["target_log"], alpha=ridge_alpha)
        pred_log = predict_ridge(model, test[MODEL_FEATURES])
        test["predicted_value_surrogate"] = np.expm1(pred_log) / 1000.0
        rows.append({"split": "leave_regime_out", "heldout": regime, **prediction_metrics(test, "predicted_value_surrogate")})
    return pd.DataFrame(rows)


def fit_ridge(x: pd.DataFrame, y: pd.Series, *, alpha: float) -> dict[str, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mean = np.nanmean(x_arr, axis=0)
    std = np.nanstd(x_arr, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    x_std = np.nan_to_num((x_arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
    return {"coef": coef, "mean": mean, "std": std}


def predict_ridge(model: dict[str, np.ndarray], x: pd.DataFrame) -> np.ndarray:
    x_arr = x.to_numpy(dtype=float)
    x_std = np.nan_to_num((x_arr - model["mean"]) / model["std"], nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return np.maximum(design @ model["coef"], 0.0)


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["marginal_resource_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[["city", "event_id"]].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": frame["marginal_resource_value"].corr(frame[score_col], method="spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in [0.01, 0.05, 0.10]:
        out[f"top_{int(frac*100)}pct_value_capture"] = mean_event_top_capture(frame, score_col, frac)
    return out


def evaluate_policy_scores(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_cols = {
        "surrogate_placeholder": None,
        "activated_bottleneck_law": "activated_bottleneck_score",
        "deficit_only": "deficit_only_score",
        "exposure_only": "exposure_only_score",
        "structure_only": "structure_only_score",
        "optimizer_selected": "selected_by_optimizer",
    }
    for name, score_col in score_cols.items():
        if score_col is None:
            continue
        metrics = {"policy_score": name, "n_tokens": int(len(tokens))}
        for frac in [0.01, 0.05, 0.10]:
            metrics[f"top_{int(frac*100)}pct_value_capture"] = mean_event_top_capture(tokens, score_col, frac)
        metrics["mean_spearman_by_event"] = mean_event_spearman(tokens, score_col)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("top_5pct_value_capture", ascending=False)


def summarize_policy_simulation(policy_simulation: pd.DataFrame) -> pd.DataFrame:
    if policy_simulation.empty:
        return pd.DataFrame()
    summary = (
        policy_simulation.groupby(["policy_scenario", "budget_scale", "delay_add_hours", "policy_score"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_value_proxy=("value_proxy", "mean"),
            median_value_proxy=("value_proxy", "median"),
            mean_relative_to_greedy_oracle=("relative_to_greedy_oracle", "mean"),
            median_relative_to_greedy_oracle=("relative_to_greedy_oracle", "median"),
            mean_allocated_cost=("allocated_cost", "mean"),
            mean_selected_action_count=("selected_action_count", "mean"),
            mean_value_vs_lp_recoverable_fraction=("value_proxy_vs_lp_recoverable_fraction", "mean"),
        )
        .sort_values(["policy_scenario", "mean_relative_to_greedy_oracle"], ascending=[True, False])
    )
    return summary


def summarize_policy_replay(policy_replay: pd.DataFrame) -> pd.DataFrame:
    if policy_replay.empty:
        return pd.DataFrame()
    summary = (
        policy_replay.groupby(["policy_scenario", "budget_scale", "delay_add_hours", "policy_score"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_replay_recoverable_fraction=("replay_recoverable_fraction", "mean"),
            median_replay_recoverable_fraction=("replay_recoverable_fraction", "median"),
            mean_fraction_of_base_lp_gain=("replay_fraction_of_base_lp_gain", "mean"),
            median_fraction_of_base_lp_gain=("replay_fraction_of_base_lp_gain", "median"),
            mean_relative_to_greedy_replay_gain=("relative_to_greedy_replay_gain", "mean"),
            median_relative_to_greedy_replay_gain=("relative_to_greedy_replay_gain", "median"),
            mean_gap_to_base_lp_recoverable=("replay_recoverable_gap_to_lp", "mean"),
            mean_allocated_cost=("allocated_cost", "mean"),
            mean_selected_action_count=("selected_action_count", "mean"),
        )
        .sort_values(["policy_scenario", "mean_fraction_of_base_lp_gain"], ascending=[True, False])
    )
    return summary


def build_event_level_law_table(summary: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    opt = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    opt["event_id"] = pd.to_numeric(opt["event_id"], errors="coerce").astype(int)
    merged = opt.merge(concentration, on=["city", "event_id", "event_start"], how="left", suffixes=("", "_conc"))
    merged["loss_magnitude_rank"] = rank_pct(merged["baseline_objective"].to_numpy(dtype=float))
    merged["recoverable_rank"] = rank_pct(merged["recoverable_fraction"].to_numpy(dtype=float))
    merged["top_tail_rank"] = rank_pct(merged["top_5pct_value_share"].to_numpy(dtype=float))
    merged["decision_criticality_score"] = (
        merged["recoverable_fraction"] * merged["top_5pct_value_share"] * merged["marginal_value_gini"]
    )
    merged["decision_criticality_rank"] = rank_pct(merged["decision_criticality_score"].to_numpy(dtype=float))
    cols = [
        "city",
        "event_id",
        "event_start",
        "baseline_objective",
        "recoverable_fraction",
        "top_1pct_value_share",
        "top_5pct_value_share",
        "top_10pct_value_share",
        "marginal_value_gini",
        "optimizer_selected_value_share",
        "loss_magnitude_rank",
        "recoverable_rank",
        "top_tail_rank",
        "decision_criticality_score",
        "decision_criticality_rank",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    return merged[cols].sort_values("decision_criticality_score", ascending=False)


def mean_event_top_capture(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    captures = []
    for _, group in frame.groupby(["city", "event_id"]):
        values = group["marginal_resource_value"].to_numpy(dtype=float)
        scores = group[score_col].astype(float).to_numpy()
        if values.sum() <= 0 or len(group) == 0:
            continue
        k = max(1, int(np.ceil(len(group) * frac)))
        best = np.sort(values)[-k:].sum()
        chosen = values[np.argsort(scores)[-k:]].sum()
        captures.append(float(chosen / best)) if best > 0 else None
    return float(np.mean(captures)) if captures else np.nan


def mean_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    values = []
    for _, group in frame.groupby(["city", "event_id"]):
        if group["marginal_resource_value"].nunique() < 2 or group[score_col].nunique() < 2:
            continue
        values.append(group["marginal_resource_value"].corr(group[score_col], method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def make_figures(
    tokens: pd.DataFrame,
    loco_metrics: pd.DataFrame,
    policy_capture: pd.DataFrame,
    event_law: pd.DataFrame,
    policy_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ordered = policy_capture.sort_values("top_5pct_value_capture", ascending=True)
    ax.barh(ordered["policy_score"], ordered["top_5pct_value_capture"], color="#2563eb")
    ax.set_xlabel("Mean top-5% value capture")
    ax.set_title("Law score versus simple action-ranking baselines")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_score_value_capture.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(tokens["activated_bottleneck_score"], tokens["marginal_resource_value"], s=7, alpha=0.25, color="#0f766e")
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("Activated-bottleneck law score")
    ax.set_ylabel("Marginal resource value proxy")
    ax.set_title("Local law score and action value field")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "activated_law_vs_action_value.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(event_law["baseline_objective"], event_law["top_5pct_value_share"], c=event_law["recoverable_fraction"], cmap="viridis", s=70)
    ax.set_xlabel("Observed/passive loss proxy")
    ax.set_ylabel("Top-5% recovery-value concentration")
    ax.set_title("Event-level top-tail decision criticality")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "event_top_tail_law.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(loco_metrics["heldout"], loco_metrics["top_5pct_value_capture"], color="#7c3aed")
    ax.set_ylabel("Top-5% value capture")
    ax.set_title("Leave-one-city-out surrogate performance")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "leave_city_out_capture.png", dpi=180)
    plt.close(fig)

    if not policy_summary.empty:
        plot_policy = policy_summary[
            policy_summary["policy_score"].isin(
                ["activated_bottleneck_law", "exposure_only", "deficit_only", "structure_only", "random_positive"]
            )
        ].copy()
        scenario_order = ["low_budget", "base", "high_budget", "delay_2h", "delay_4h", "scarce_and_late"]
        plot_policy["policy_scenario"] = pd.Categorical(
            plot_policy["policy_scenario"],
            categories=scenario_order,
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for policy_name, group in plot_policy.sort_values("policy_scenario").groupby("policy_score"):
            ax.plot(
                group["policy_scenario"].astype(str),
                group["mean_relative_to_greedy_oracle"],
                marker="o",
                linewidth=2,
                label=policy_name,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Mean value relative to greedy oracle")
        ax.set_title("Budget-aware law-guided policy stress test")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "budget_delay_policy_stress_test.png", dpi=180)
        plt.close(fig)

    if not replay_summary.empty:
        replay_plot = replay_summary[
            replay_summary["policy_score"].isin(
                ["lp_optimizer_replay", "activated_bottleneck_law", "exposure_only", "deficit_only", "random_positive"]
            )
        ].copy()
        scenario_order = ["base", "low_budget", "high_budget", "delay_2h", "delay_4h", "scarce_and_late"]
        replay_plot["policy_scenario"] = pd.Categorical(
            replay_plot["policy_scenario"],
            categories=scenario_order,
            ordered=True,
        )
        fig, ax = plt.subplots(figsize=(9.4, 5.0))
        for policy_name, group in replay_plot.sort_values("policy_scenario").groupby("policy_score"):
            ax.plot(
                group["policy_scenario"].astype(str),
                group["mean_fraction_of_base_lp_gain"],
                marker="o",
                linewidth=2,
                label=policy_name,
            )
        ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", alpha=0.45)
        ax.set_ylabel("Mean replay gain / base LP optimized gain")
        ax.set_title("Fixed-policy replay through recovery dynamics")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "fixed_policy_replay_vs_lp.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    tokens: pd.DataFrame,
    concentration: pd.DataFrame,
    loco_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    policy_capture: pd.DataFrame,
    event_law: pd.DataFrame,
    policy_simulation: pd.DataFrame,
    policy_summary: pd.DataFrame,
    policy_replay: pd.DataFrame,
    replay_summary: pd.DataFrame,
) -> None:
    loco_top5_mean = float(loco_metrics["top_5pct_value_capture"].mean()) if not loco_metrics.empty else np.nan
    loco_spearman_mean = float(loco_metrics["spearman"].mean()) if not loco_metrics.empty else np.nan
    law_row = policy_capture[policy_capture["policy_score"] == "activated_bottleneck_law"]
    law_top5 = float(law_row["top_5pct_value_capture"].iloc[0]) if not law_row.empty else np.nan
    law_spearman = float(law_row["mean_spearman_by_event"].iloc[0]) if not law_row.empty else np.nan
    event_top5_mean = float(concentration["top_5pct_value_share"].mean()) if not concentration.empty else np.nan
    event_gini_mean = float(concentration["marginal_value_gini"].mean()) if not concentration.empty else np.nan
    base_policy = (
        policy_summary[policy_summary["policy_scenario"].eq("base")]
        if not policy_summary.empty and "policy_scenario" in policy_summary
        else pd.DataFrame()
    )
    law_base = base_policy[base_policy["policy_score"].eq("activated_bottleneck_law")]
    law_base_relative = (
        float(law_base["mean_relative_to_greedy_oracle"].iloc[0])
        if not law_base.empty
        else np.nan
    )
    simple_base = base_policy[base_policy["policy_score"].isin(["exposure_only", "deficit_only", "structure_only", "random_positive"])]
    best_simple_base = (
        float(simple_base["mean_relative_to_greedy_oracle"].max())
        if not simple_base.empty
        else np.nan
    )
    greedy_selected_share = float(tokens["greedy_selected_by_oracle"].mean()) if "greedy_selected_by_oracle" in tokens else np.nan
    base_replay = (
        replay_summary[replay_summary["policy_scenario"].eq("base")]
        if not replay_summary.empty and "policy_scenario" in replay_summary
        else pd.DataFrame()
    )
    law_replay_base = base_replay[base_replay["policy_score"].eq("activated_bottleneck_law")]
    law_replay_fraction = (
        float(law_replay_base["mean_fraction_of_base_lp_gain"].iloc[0])
        if not law_replay_base.empty
        else np.nan
    )
    law_replay_gap = (
        float(law_replay_base["mean_gap_to_base_lp_recoverable"].iloc[0])
        if not law_replay_base.empty
        else np.nan
    )
    best_simple_replay = base_replay[
        base_replay["policy_score"].isin(["exposure_only", "deficit_only", "structure_only", "random_positive"])
    ]
    best_simple_replay_fraction = (
        float(best_simple_replay["mean_fraction_of_base_lp_gain"].max())
        if not best_simple_replay.empty
        else np.nan
    )
    optimizer_replay_base = base_replay[base_replay["policy_score"].eq("lp_optimizer_replay")]
    optimizer_replay_gap = (
        float(optimizer_replay_base["mean_gap_to_base_lp_recoverable"].iloc[0])
        if not optimizer_replay_base.empty
        else np.nan
    )
    lines = [
        "# Learning and Law Discovery V3",
        "",
        "## 本版本做了什么",
        "",
        "这一版把 event-level optimization outputs 转换成 action-token 学习问题。每个 token 表示 `city-event-unit-time-intervention`。目标不是直接学习 optimizer 是否选择该 token，而是构造一个可解释的 marginal recovery-value proxy：单位资源投到该 token 后，沿着无干预的被动恢复轨迹，估计它能减少多少未来加权功能损失。",
        "",
        "V2 在 V1 的静态 action-value field 之上，新增了 budget-aware greedy oracle。V3 进一步把 greedy/law/simple baseline 生成的固定 allocation 放回复原始恢复动力学中 replay，直接计算 fixed policy 的 12 小时 objective 和 recoverable fraction。因此现在不仅能问“一个 law-guided policy 在 proxy 上接近 greedy oracle 吗”，也能问“它在实际 `b, rC, rS, ell` 演化里能拿到 LP optimum 的多少恢复收益”。",
        "",
        "## 数据规模",
        "",
        f"- sampled action tokens: {len(tokens):,}",
        f"- city-event scenarios: {tokens[['city', 'event_id']].drop_duplicates().shape[0]}",
        f"- full candidate-action concentration rows: {len(concentration)}",
        f"- policy stress-test rows: {len(policy_simulation):,}",
        f"- fixed-policy replay rows: {len(policy_replay):,}",
        f"- sampled-token greedy oracle selected share: {greedy_selected_share:.4f}",
        f"- mean event top-5% value share: {event_top5_mean:.4f}",
        f"- mean event marginal-value gini: {event_gini_mean:.4f}",
        "",
        "## Action Value Label",
        "",
        "每个 action 的 label 不是 optimizer 的 0/1 选择，而是 marginal resource value。直观上，如果一个 action 作用于未来仍会持续存在的损失、又覆盖高 OD 暴露区域、且单位成本效率高，那么它的恢复价值就高。",
        "",
        "对 `R` 和 `C` 类 action，价值主要来自某个 region 本地 deficit 被降低后，通过 OD dependence `Q` 减少其他 origins 的 accessibility loss；因此 exposure 使用 destination importance。对 `S` 类 action，价值直接作用在 origin 的 experienced loss 上；因此 exposure 使用 origin exposure。",
        "",
        "```text",
        "marginal_resource_value(i,t,k)",
        "  = future_effect_value(i,t,k)",
        "    * eta(i,t,k) / cost(i,t,k)",
        "    / passive_event_loss",
        "```",
        "",
        "## Interpretable Law Score",
        "",
        "这一版的可解释 law score 保留 action label 的核心结构，但不直接使用 optimizer 选择结果：",
        "",
        "```text",
        "activated_bottleneck_score(i,t,k)",
        "  ≈ delay_feasible(i,t,k)",
        "    × future_deficit_area(i,t,k)",
        "    × OD_exposure_or_destination_importance(i,k)",
        "    × intervention_efficiency_per_cost(i,t,k)",
        "```",
        "",
        "这里的 `future_deficit_area` 已经包含剩余时间窗口，因此不再额外乘一个简单的 time rank。早期草稿里我把 `1 - out_degree_rank` 当作 substitutability scarcity 强行乘进去，结果明显拉低排序表现；这一版先移除这个不稳定项，把 substitutability 留到后续用更可靠的替代路径或网络冗余指标刻画。",
        "",
        "## Budget-Aware Greedy Oracle",
        "",
        "V2 增加的 greedy oracle 不是重新求解 LP，而是一个可解释的 budget-aware policy simulator。它做三件事：",
        "",
        "1. 把每个 `unit-time-intervention` 的部署上限按 PWL diminishing returns 拆成多个 segment。",
        "2. 每个 segment 的真实评价仍用 oracle marginal value per cost，但乘上该 segment 的 diminishing multiplier。",
        "3. 在 total budget、period budget 和 delay feasibility 下，按某个 policy score 贪心选择 segment。",
        "",
        "因此 `greedy_oracle` 是这个解析标签体系下的上界 policy；`activated_bottleneck_law`、`exposure_only`、`deficit_only`、`structure_only`、`random_positive` 都在同一预算约束下与它比较。",
        "",
        "## Fixed-Policy Replay",
        "",
        "V3 的 replay validation 不再只看 action-value proxy。对每个 policy 生成的固定 allocation，脚本把分段资源量转换成 `R/C/S` 的实际 effect，然后按原始恢复动力学逐小时前推：",
        "",
        "```text",
        "b[t+1]  = clip(a * b[t] + h[t+1] - e_R[t], 0, 1)",
        "rC[t+1] = clip((1 - delta_C) * rC[t] + e_C[t], 0, 1)",
        "rS[t+1] = clip((1 - delta_S) * rS[t] + e_S[t], 0, 1)",
        "ell[t]  = clip(Q * max(b[t] - rC[t], 0) - rS[t], 0, 1)",
        "```",
        "",
        "这里 `lp_optimizer_replay` 是一个 sanity check：把 Gurobi 输出的 optimized effects 放回同一个 replay engine，应该接近原 LP objective。这个检查用于确认 replay engine 和 LP 目标是一致的。",
        "",
        "## 关键结果概览",
        "",
        f"- Leave-one-city-out mean Spearman: {loco_spearman_mean:.4f}",
        f"- Leave-one-city-out mean top-5% value capture: {loco_top5_mean:.4f}",
        f"- Activated-bottleneck law top-5% value capture: {law_top5:.4f}",
        f"- Activated-bottleneck law mean event Spearman: {law_spearman:.4f}",
        f"- Base scenario law policy / greedy oracle: {law_base_relative:.4f}",
        f"- Base scenario best simple baseline / greedy oracle: {best_simple_base:.4f}",
        f"- Base scenario law replay gain / LP optimized gain: {law_replay_fraction:.4f}",
        f"- Base scenario best simple replay gain / LP optimized gain: {best_simple_replay_fraction:.4f}",
        f"- LP optimizer replay mean recoverable-fraction gap: {optimizer_replay_gap:.4g}",
        f"- Base scenario law replay mean recoverable-fraction gap to LP: {law_replay_gap:.4f}",
        "",
        "## Leave-One-City-Out Surrogate",
        "",
        dataframe_to_markdown(loco_metrics),
        "",
        "## Leave-Regime-Out Surrogate",
        "",
        dataframe_to_markdown(regime_metrics),
        "",
        "## Law Score 与 Baselines",
        "",
        dataframe_to_markdown(policy_capture),
        "",
        "解释：`optimizer_selected` 在 action-value 排序里不一定最高，因为 optimizer 选择受到总预算、单期预算、部署上限、分段边际收益和替代 action 的共同约束；而 law score 评价的是“单个 action 的边际价值排序”。因此这里更应该看 law score 是否能捕捉 value field 的 top tail，而不是是否复刻 optimizer 的最终稀疏解。",
        "",
        "## Budget/Delay Policy Validation",
        "",
        dataframe_to_markdown(policy_summary, max_rows=40),
        "",
        "解释：这个表不是新的 Gurobi LP 解，而是用同一 action-value field 做的预算约束 stress test。它用于检验 law 在不同预算和延迟条件下是否仍能作为资源分配 policy 接近 greedy oracle。真正的最终版还需要对关键 budget/delay scenario 重新求解 LP 来闭合验证。",
        "",
        "## Fixed-Policy Replay Validation",
        "",
        dataframe_to_markdown(replay_summary, max_rows=40),
        "",
        "解释：这个表的核心列是 `mean_fraction_of_base_lp_gain`，即 fixed policy replay 得到的恢复收益占 base LP optimized gain 的比例。对于 base scenario，这就是 law/simple baseline 与 LP optimum 的直接差距；对于 low/high budget 或 delay scenario，它仍以 base LP gain 作参照，因此用于观察趋势，而不是声明这些新场景下的最优性。",
        "",
        "## Event-Level Top-Tail Law",
        "",
        dataframe_to_markdown(event_law.head(15)),
        "",
        "## 当前可读出的初步 law",
        "",
        "Local activated-bottleneck law 的第一版可写成：",
        "",
        "```text",
        "recovery_value(action)",
        "  ≈ persistent_future_deficit_area",
        "    × exposed_OD_importance_or_origin_exposure",
        "    × intervention_efficiency_per_cost",
        "    × response_feasibility",
        "```",
        "",
        "Event top-tail law 的第一版可写成：",
        "",
        "```text",
        "decision_criticality(event)",
        "  ≈ recoverable_fraction",
        "    × top_tail_concentration_of_action_values",
        "    × inequality_of_recovery_value_field",
        "```",
        "",
        "## 需要继续改进的地方",
        "",
        "1. 下一版应挑选代表性 city-event 做 single-action marginal LP 或 perturbed optimum stability，用真实 LP 边际值验证当前解析 action label。",
        "2. 当前 budget/delay augmentation 已有 fixed-policy replay，但还不是重新求解 Gurobi 的多情景 optimum；后续应挑选代表性 scenario 重新优化验证。",
        "3. 当前 surrogate 是 ridge baseline，不是最终神经模型；后续可升级为 factorized action-value scorer 或 graph surrogate。",
        "4. 当前 substitutability 没有被可靠刻画。简单 out-degree scarcity 在本数据中会损害排序，后续应加入替代路径、网络冗余或 OD rerouting proxy。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), 0.5, dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def top_share(values: np.ndarray, pct: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0 or values.sum() <= 0:
        return np.nan
    n = max(1, int(np.ceil(len(values) * pct)))
    return float(np.sort(values)[-n:].sum() / values.sum())


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    values = np.sort(np.maximum(values, 0.0))
    total = values.sum()
    if total <= 0:
        return np.nan
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * total) - (n + 1) / n)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def event_regime(peak_positive_abnormal_deficit: float) -> str:
    if peak_positive_abnormal_deficit < 0.015:
        return "low"
    if peak_positive_abnormal_deficit < 0.035:
        return "medium"
    return "high"


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

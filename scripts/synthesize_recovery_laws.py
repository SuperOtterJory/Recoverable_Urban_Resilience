"""Synthesize learning-to-law evidence into paper-ready summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "law_synthesis"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_tables(root)
    metrics = build_metrics(data)
    evidence = build_evidence_ladder(metrics)
    closure = build_policy_closure_table(data)
    city_closure = build_city_closure_table(data)
    decision_examples = build_decision_examples(data)
    top_tail_correlations = build_top_tail_correlations(data)
    limitations = build_limitations(data)

    write_table(evidence, table_dir / "law_evidence_ladder.csv")
    write_table(closure, table_dir / "law_policy_closure_summary.csv")
    write_table(city_closure, table_dir / "law_city_closure_summary.csv")
    write_table(decision_examples, table_dir / "law_event_decision_examples.csv")
    write_table(top_tail_correlations, table_dir / "law_top_tail_correlations.csv")
    write_table(limitations, table_dir / "law_limitations_and_next_steps.csv")
    (table_dir / "law_synthesis_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(data, closure, city_closure, figure_dir)
    write_report(
        report_dir / "law_synthesis_report_zh.md",
        metrics,
        evidence,
        closure,
        city_closure,
        decision_examples,
        top_tail_correlations,
        limitations,
    )
    print(f"Wrote law synthesis to {output_dir}")


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    results = root / "results"
    return {
        "lp_validation": read_csv(results / "law_learning_lp_validation" / "tables" / "single_action_lp_marginal_summary.csv"),
        "policy_capture": read_csv(results / "law_learning" / "tables" / "policy_score_value_capture.csv"),
        "fixed_replay_summary": read_csv(results / "law_learning" / "tables" / "fixed_policy_replay_summary.csv"),
        "finite_gap_city": read_csv(results / "finite_budget_gap" / "tables" / "finite_budget_gap_city_summary.csv"),
        "finite_gap_event": read_csv(results / "finite_budget_gap" / "tables" / "finite_budget_gap_event_metrics.csv"),
        "residual_event": read_csv(results / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv"),
        "residual_city": read_csv(results / "residual_greedy_policy" / "tables" / "residual_greedy_city_summary.csv"),
        "stress_scenario": read_csv(results / "residual_greedy_stress" / "tables" / "residual_stress_scenario_summary.csv"),
        "scenario_policy": read_csv(results / "scenario_optimum_validation" / "tables" / "scenario_policy_validation.csv"),
        "scenario_optima": read_csv(results / "scenario_optimum_validation" / "tables" / "scenario_lp_optima.csv"),
        "scenario_summary": read_csv(results / "scenario_optimum_validation" / "tables" / "scenario_policy_summary.csv"),
        "perturbed_solves": read_csv(results / "perturbed_optimum_stability" / "tables" / "perturbed_solve_summary.csv"),
        "perturbed_overlap": read_csv(results / "perturbed_optimum_stability" / "tables" / "perturbed_policy_overlap.csv"),
        "event_law": read_csv(results / "law_learning" / "tables" / "event_level_top_tail_law.csv"),
        "leave_city": read_csv(results / "law_learning" / "tables" / "leave_city_out_metrics.csv"),
        "leave_regime": read_csv(results / "law_learning" / "tables" / "leave_regime_out_metrics.csv"),
        "symbolic_formula": read_csv(results / "symbolic_law_extraction" / "tables" / "symbolic_formula_metrics.csv"),
        "symbolic_ablation": read_csv(results / "symbolic_law_extraction" / "tables" / "feature_group_ablation.csv"),
        "symbolic_leave_city": read_csv(results / "symbolic_law_extraction" / "tables" / "formula_leave_city_metrics.csv"),
        "budget_phase_summary": read_csv(results / "budget_leverage_phase" / "tables" / "budget_leverage_summary.csv"),
        "budget_phase_tests": read_csv(results / "budget_leverage_phase" / "tables" / "budget_phase_tests.csv"),
        "fine_budget_summary": read_csv(results / "budget_fine_sweep" / "tables" / "fine_budget_summary.csv"),
        "fine_budget_phase": read_csv(results / "budget_fine_sweep" / "tables" / "fine_budget_phase_tests.csv"),
        "fine_budget_metrics": read_json_table(results / "budget_fine_sweep" / "tables" / "fine_budget_metrics.json"),
        "early_metrics": read_csv(results / "early_predictability" / "tables" / "early_predictability_metrics.csv"),
        "early_best": read_csv(results / "early_predictability" / "tables" / "early_best_metrics_by_target.csv"),
        "nonobvious_summary": read_csv(results / "nonobvious_action_laws" / "tables" / "heuristic_failure_summary.csv"),
        "nonobvious_hidden": read_csv(results / "nonobvious_action_laws" / "tables" / "hidden_gem_summary.csv"),
        "nonobvious_reasons": read_csv(results / "nonobvious_action_laws" / "tables" / "heuristic_failure_reason_summary.csv"),
        "nonobvious_persistence": read_csv(results / "nonobvious_action_laws" / "tables" / "persistence_vs_peak_summary.csv"),
        "factorized_summary": read_csv(results / "factorized_action_surrogate" / "tables" / "factorized_model_summary.csv"),
        "factorized_increments": read_csv(results / "factorized_action_surrogate" / "tables" / "factorized_incremental_gains.csv"),
        "graph_summary": read_csv(results / "graph_structure_ablation" / "tables" / "graph_structure_model_summary.csv"),
        "graph_gaps": read_csv(results / "graph_structure_ablation" / "tables" / "graph_structure_shuffle_gaps.csv"),
        "regime_model_summary": read_csv(results / "event_regime_generalization" / "tables" / "regime_model_summary.csv"),
        "regime_gap_summary": read_csv(results / "event_regime_generalization" / "tables" / "regime_gap_summary.csv"),
        "temporal_model_summary": read_csv(results / "temporal_generalization" / "tables" / "temporal_model_summary.csv"),
        "temporal_gap_summary": read_csv(results / "temporal_generalization" / "tables" / "temporal_gap_summary.csv"),
        "temporal_metrics": read_json_table(results / "temporal_generalization" / "tables" / "temporal_generalization_metrics.json"),
        "neural_summary": read_csv(results / "neural_surrogate_leakage" / "tables" / "neural_surrogate_summary.csv"),
        "neural_metrics": read_json_table(results / "neural_surrogate_leakage" / "tables" / "neural_surrogate_leakage_metrics.json"),
        "objective_summary": read_csv(results / "training_objective_ablation" / "tables" / "objective_model_summary.csv"),
        "objective_improvements": read_csv(results / "training_objective_ablation" / "tables" / "objective_improvement_summary.csv"),
        "parameter_summary": read_csv(results / "parameter_deconfounded_law" / "tables" / "parameter_deconfounded_model_summary.csv"),
        "parameter_increments": read_csv(results / "parameter_deconfounded_law" / "tables" / "parameter_deconfounded_increments.csv"),
        "parameter_channel": read_csv(results / "parameter_deconfounded_law" / "tables" / "channel_neutral_score_summary.csv"),
        "event_decision_metrics": read_json_table(results / "event_decision_criticality" / "tables" / "event_decision_criticality_metrics.json"),
        "event_decision_variance": read_csv(results / "event_decision_criticality" / "tables" / "event_variance_decomposition.csv"),
        "event_decision_phase": read_csv(results / "event_decision_criticality" / "tables" / "event_phase_summary.csv"),
        "od_message_summary": read_csv(results / "od_message_passing_surrogate" / "tables" / "od_message_model_summary.csv"),
        "od_message_increments": read_csv(results / "od_message_passing_surrogate" / "tables" / "od_message_incremental_gains.csv"),
        "od_message_metrics": read_json_table(results / "od_message_passing_surrogate" / "tables" / "od_message_passing_metrics.json"),
        "parameter_ensemble_metrics": read_json_table(results / "parameter_ensemble_stability" / "tables" / "parameter_ensemble_stability_metrics.json"),
        "parameter_ensemble_model": read_csv(results / "parameter_ensemble_stability" / "tables" / "parameter_ensemble_model_summary.csv"),
        "parameter_ensemble_score": read_csv(results / "parameter_ensemble_stability" / "tables" / "parameter_ensemble_score_summary.csv"),
        "parameter_ensemble_token": read_csv(results / "parameter_ensemble_stability" / "tables" / "parameter_ensemble_token_summary.csv"),
        "parameter_lp_summary": read_csv(results / "parameter_ensemble_optimum_validation" / "tables" / "parameter_policy_summary.csv"),
        "parameter_lp_optima": read_csv(results / "parameter_ensemble_optimum_validation" / "tables" / "parameter_lp_optima.csv"),
        "parameter_lp_metrics": read_json_table(results / "parameter_ensemble_optimum_validation" / "tables" / "parameter_ensemble_optimum_metrics.json"),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([data])


def build_metrics(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    lp_validation = data["lp_validation"]
    small_signal = one_row(lp_validation, label="small_signal_derivative_label", group="all")
    finite_label = one_row(lp_validation, label="finite_deficit_area_label", group="all")

    policy_capture = data["policy_capture"]
    law_capture = one_row(policy_capture, policy_score="activated_bottleneck_law")
    exposure_capture = one_row(policy_capture, policy_score="exposure_only")
    deficit_capture = one_row(policy_capture, policy_score="deficit_only")
    structure_capture = one_row(policy_capture, policy_score="structure_only")

    fixed = data["fixed_replay_summary"]
    fixed_base_law = one_row(fixed, policy_scenario="base", policy_score="activated_bottleneck_law")
    fixed_base_lp = one_row(fixed, policy_scenario="base", policy_score="lp_optimizer_replay")

    finite_event = data["finite_gap_event"]
    residual_event = data["residual_event"]
    stress = data["stress_scenario"]
    stress_base = one_row(stress, policy_scenario="base")
    stress_low = one_row(stress, policy_scenario="low_budget")
    stress_delay4 = one_row(stress, policy_scenario="delay_4h")
    stress_scarce = one_row(stress, policy_scenario="scarce_and_late")

    scenario_policy = data["scenario_policy"]
    scenario_pivot = scenario_policy.pivot_table(
        index=["city", "event_id", "policy_scenario"],
        columns="policy",
        values="fraction_of_scenario_lp_gain",
        aggfunc="first",
    )
    if {"static_small_signal_greedy", "residual_finite_greedy"}.issubset(scenario_pivot.columns):
        scenario_pivot = scenario_pivot.reset_index()
        scenario_pivot["residual_minus_static"] = (
            scenario_pivot["residual_finite_greedy"] - scenario_pivot["static_small_signal_greedy"]
        )
    else:
        scenario_pivot = pd.DataFrame()

    optima = data["scenario_optima"]
    perturbed_solves = data["perturbed_solves"]
    perturbed_overlap = data["perturbed_overlap"]
    event_law = data["event_law"]
    leave_city = data["leave_city"]
    leave_regime = data["leave_regime"]
    symbolic = data["symbolic_formula"]
    symbolic_ablation = data["symbolic_ablation"]
    symbolic_leave_city = data["symbolic_leave_city"]
    budget_phase = data["budget_phase_summary"]
    budget_tests = data["budget_phase_tests"]
    fine_budget_metrics = one_row(data["fine_budget_metrics"])
    early_metrics = data["early_metrics"]
    early_best = data["early_best"]
    activated_symbolic = one_row(symbolic, formula_id="F7_activated_recovery_law")
    minimal_log_symbolic = one_row(symbolic, formula_id="R7_minimal_log_law")
    exposure_symbolic = one_row(symbolic, formula_id="F2_exposure")
    full_ablation = one_row(symbolic_ablation, ablation_id="full_interpretable")
    without_od = one_row(symbolic_ablation, ablation_id="without_od_exposure_structure")
    leave_group_rows = (
        symbolic_ablation[symbolic_ablation["mode"].astype(str).eq("leave_one_group_out")].copy()
        if "mode" in symbolic_ablation
        else pd.DataFrame()
    )
    if not leave_group_rows.empty and "top5_capture_drop_vs_full" in leave_group_rows:
        largest_drop = leave_group_rows.sort_values("top5_capture_drop_vs_full", ascending=False).iloc[0]
    else:
        largest_drop = pd.Series(dtype=float)
    budget_abs_random = one_row(budget_tests, metric="mean_proxy_leverage_vs_random_positive")
    budget_ratio_random = one_row(budget_tests, metric="mean_proxy_ratio_vs_random_positive")
    budget_per_cost = one_row(budget_tests, metric="mean_residual_minus_static_per_cost")
    budget_low = one_row(budget_phase, budget_scale=0.5)
    budget_base = one_row(budget_phase, budget_scale=1.0)
    budget_high = one_row(budget_phase, budget_scale=2.0)
    early_decision_best = one_row(early_best, target="decision_criticality_score")
    early_recoverable_best = one_row(early_best, target="recoverable_fraction")
    early_decision_2h = one_row(early_metrics, target="decision_criticality_score", feature_group="all_early", window_hours=2)
    early_recoverable_2h = one_row(early_metrics, target="recoverable_fraction", feature_group="all_early", window_hours=2)
    early_decision_static_1h = one_row(early_metrics, target="decision_criticality_score", feature_group="static_city", window_hours=1)
    early_decision_speed_1h = one_row(early_metrics, target="decision_criticality_score", feature_group="early_speed", window_hours=1)
    early_decision_rain_1h = one_row(early_metrics, target="decision_criticality_score", feature_group="early_rain", window_hours=1)
    nonobvious_summary = data["nonobvious_summary"]
    nonobvious_hidden = data["nonobvious_hidden"]
    nonobvious_reasons = data["nonobvious_reasons"]
    nonobvious_persistence = data["nonobvious_persistence"]
    nonobvious_deficit = one_row(nonobvious_summary, heuristic="deficit_only")
    nonobvious_exposure = one_row(nonobvious_summary, heuristic="exposure_only")
    nonobvious_structure = one_row(nonobvious_summary, heuristic="structure_only")
    nonobvious_hidden_all = one_row(nonobvious_hidden, scope="all_events")
    nonobvious_deficit_reasons = one_row(nonobvious_reasons, heuristic="deficit_only")
    nonobvious_structure_reasons = one_row(nonobvious_reasons, heuristic="structure_only")
    nonobvious_peak = one_row(nonobvious_persistence, feature="peak_event_disturbance")
    nonobvious_remaining = one_row(nonobvious_persistence, feature="remaining_local_area")
    factorized_summary = data["factorized_summary"]
    factorized_increments = data["factorized_increments"]
    factorized_m1 = one_row(factorized_summary, model_id="M1_deficit_only")
    factorized_m5 = one_row(factorized_summary, model_id="M5_full_additive")
    factorized_m6 = one_row(factorized_summary, model_id="M6_full_interaction")
    factorized_m7 = one_row(factorized_summary, model_id="M7_factorized_low_dim")
    factorized_m8 = one_row(factorized_summary, model_id="M8_factorized_interaction")
    factorized_add_od = one_row(factorized_increments, comparison="add_od_exposure")
    factorized_add_time = one_row(factorized_increments, comparison="add_time_feasibility")
    factorized_add_highdim_interaction = one_row(factorized_increments, comparison="add_explicit_interactions")
    factorized_add_lowdim_interaction = one_row(factorized_increments, comparison="factorized_add_interactions")
    graph_summary = data["graph_summary"]
    graph_gaps = data["graph_gaps"]
    graph_no_graph = one_row(graph_summary, model_id="G1_local_dynamic_no_graph")
    graph_observed_full = one_row(graph_summary, model_id="G6_local_plus_observed_od_graph")
    graph_shuffled_full = one_row(graph_summary, model_id="G7_local_plus_shuffled_od_graph")
    graph_factorized_observed = one_row(graph_summary, model_id="G8_factorized_observed_od")
    graph_factorized_shuffled = one_row(graph_summary, model_id="G9_factorized_shuffled_od")
    graph_full_alignment = one_row(graph_gaps, comparison="full_od_graph_alignment")
    graph_factorized_alignment = one_row(graph_gaps, comparison="factorized_od_alignment")
    graph_observed_over_no_graph = one_row(graph_gaps, comparison="observed_graph_over_no_graph")
    regime_summary = data["regime_model_summary"]
    regime_gaps = data["regime_gap_summary"]
    regime_factorized = one_row(regime_summary, model_id="R1_factorized_low_dim")
    regime_full = one_row(regime_summary, model_id="R2_full_additive")
    regime_interaction = one_row(regime_summary, model_id="R3_full_interaction")
    regime_factorized_vs_full = regime_gaps[regime_gaps["comparison"].eq("factorized_vs_full_additive")]
    regime_interaction_vs_full = regime_gaps[regime_gaps["comparison"].eq("full_interaction_vs_full_additive")]
    temporal_summary = data["temporal_model_summary"]
    temporal_gaps = data["temporal_gap_summary"]
    temporal_metrics = one_row(data["temporal_metrics"])
    temporal_factorized_main = one_row(temporal_summary, scope="main", model_id="R1_factorized_low_dim")
    temporal_full_main = one_row(temporal_summary, scope="main", model_id="R2_full_additive")
    temporal_interaction_main = one_row(temporal_summary, scope="main", model_id="R3_full_interaction")
    temporal_activated_main = one_row(temporal_summary, scope="main", model_id="H4_activated_law")
    temporal_deficit_main = one_row(temporal_summary, scope="main", model_id="H1_deficit_only")
    temporal_factorized_forward = one_row(temporal_summary, scope="forward", model_id="R1_factorized_low_dim")
    temporal_full_forward = one_row(temporal_summary, scope="forward", model_id="R2_full_additive")
    temporal_factorized_year = one_row(temporal_summary, scope="year_confounded", model_id="R1_factorized_low_dim")
    temporal_full_year = one_row(temporal_summary, scope="year_confounded", model_id="R2_full_additive")
    temporal_factorized_vs_full_main = temporal_gaps[
        temporal_gaps["comparison"].eq("factorized_vs_full_additive")
        & temporal_gaps["split_role"].eq("main_within_city_chronological")
    ] if not temporal_gaps.empty and "split_role" in temporal_gaps else pd.DataFrame()
    temporal_activated_vs_deficit_main = temporal_gaps[
        temporal_gaps["comparison"].eq("activated_law_vs_deficit_only")
        & temporal_gaps["split_role"].eq("main_within_city_chronological")
    ] if not temporal_gaps.empty and "split_role" in temporal_gaps else pd.DataFrame()
    neural_metrics = one_row(data["neural_metrics"])
    objective_summary = data["objective_summary"]
    objective_improvements = data["objective_improvements"]
    objective_factorized = one_row(objective_improvements, feature_set="factorized_low_dim")
    objective_full = one_row(objective_improvements, feature_set="full_additive")
    objective_interaction = one_row(objective_improvements, feature_set="full_interaction")
    objective_factorized_raw = one_row(objective_summary, feature_set="factorized_low_dim", objective_id="O1_log_value")
    objective_full_best = one_row(
        objective_summary,
        feature_set="full_additive",
        objective_id=str(objective_full.get("best_objective_id", "")),
    )
    parameter_summary = data["parameter_summary"]
    parameter_increments = data["parameter_increments"]
    parameter_channel = data["parameter_channel"]
    parameter_clock = one_row(parameter_summary, model_id="P0_policy_clock_only")
    parameter_efficiency = one_row(parameter_summary, model_id="P1_clock_plus_efficiency")
    parameter_exposure = one_row(parameter_summary, model_id="P3_add_od_exposure")
    parameter_structure = one_row(parameter_summary, model_id="P4_add_structure_scarcity")
    parameter_light_factorized = one_row(parameter_summary, model_id="P5_parameter_light_factorized")
    parameter_full_factorized = one_row(parameter_summary, model_id="P6_full_factorized")
    parameter_full_additive = one_row(parameter_summary, model_id="P7_full_additive")
    parameter_add_efficiency = one_row(parameter_increments, comparison="add_calibrated_efficiency")
    parameter_add_od = one_row(parameter_increments, comparison="add_od_exposure")
    parameter_light_over_clock = one_row(parameter_increments, comparison="parameter_light_factorized_over_clock")
    parameter_eff_to_factorized = one_row(parameter_increments, comparison="add_efficiency_to_factorized_law")
    parameter_full_over_clock = one_row(parameter_increments, comparison="full_additive_over_clock")
    channel_efficiency = one_row(parameter_channel, score_id="S0_efficiency_only")
    channel_light_activation = one_row(parameter_channel, score_id="S4_parameter_light_activation")
    channel_full_activation = one_row(parameter_channel, score_id="S5_full_activation")
    event_decision_metrics = one_row(data["event_decision_metrics"])
    od_message_metrics = one_row(data["od_message_metrics"])
    od_message_scalar = one_row(data["od_message_summary"], model_id="O1_scalar_od_graph")
    od_message_scalar_plus = one_row(data["od_message_summary"], model_id="O3_scalar_plus_message")
    od_message_factorized = one_row(data["od_message_summary"], model_id="O4_factorized_low_dim")
    od_message_factorized_plus = one_row(data["od_message_summary"], model_id="O5_factorized_plus_message")
    od_message_message_only = one_row(data["od_message_summary"], model_id="O2_message_only_od")
    od_message_over_scalar = one_row(data["od_message_increments"], comparison="message_over_scalar_od")
    od_message_over_factorized = one_row(data["od_message_increments"], comparison="message_over_factorized")
    parameter_ensemble_metrics = one_row(data["parameter_ensemble_metrics"])
    parameter_lp_metrics = one_row(data["parameter_lp_metrics"])

    metrics: dict[str, Any] = {
        "n_action_tokens": safe_int(policy_capture["n_tokens"].max()) if "n_tokens" in policy_capture else None,
        "n_events": safe_int(len(event_law)) if not event_law.empty else None,
        "single_action_small_signal_spearman": safe_float(small_signal.get("spearman")),
        "single_action_small_signal_median_lp_to_label_ratio": safe_float(small_signal.get("median_lp_to_label_ratio")),
        "single_action_finite_area_spearman": safe_float(finite_label.get("spearman")),
        "leave_city_mean_spearman": safe_float(leave_city["spearman"].mean()) if "spearman" in leave_city else np.nan,
        "leave_city_mean_top5_capture": safe_float(leave_city["top_5pct_value_capture"].mean()) if "top_5pct_value_capture" in leave_city else np.nan,
        "leave_regime_mean_spearman": safe_float(leave_regime["spearman"].mean()) if "spearman" in leave_regime else np.nan,
        "law_top5_value_capture": safe_float(law_capture.get("top_5pct_value_capture")),
        "exposure_top5_value_capture": safe_float(exposure_capture.get("top_5pct_value_capture")),
        "deficit_top5_value_capture": safe_float(deficit_capture.get("top_5pct_value_capture")),
        "structure_top5_value_capture": safe_float(structure_capture.get("top_5pct_value_capture")),
        "base_lp_recoverable_fraction": safe_float(fixed_base_lp.get("mean_replay_recoverable_fraction")),
        "base_static_fraction_of_lp_gain": safe_float(fixed_base_law.get("mean_fraction_of_base_lp_gain")),
        "finite_gap_mean_static_fraction_of_lp_gain": safe_float(finite_event["greedy_fraction_of_lp_gain"].mean()) if "greedy_fraction_of_lp_gain" in finite_event else np.nan,
        "finite_gap_mean_action_cost_jaccard": safe_float(finite_event["action_cost_jaccard"].mean()) if "action_cost_jaccard" in finite_event else np.nan,
        "finite_gap_mean_lp_to_greedy_action_count_ratio": safe_float(finite_event["lp_to_greedy_action_count_ratio"].mean()) if "lp_to_greedy_action_count_ratio" in finite_event else np.nan,
        "base_residual_fraction_of_lp_gain": safe_float(residual_event["residual_fraction_of_lp_gain"].mean()) if "residual_fraction_of_lp_gain" in residual_event else np.nan,
        "base_residual_median_fraction_of_lp_gain": safe_float(residual_event["residual_fraction_of_lp_gain"].median()) if "residual_fraction_of_lp_gain" in residual_event else np.nan,
        "base_residual_improvement": safe_float(residual_event["residual_gain_improvement_over_static"].mean()) if "residual_gain_improvement_over_static" in residual_event else np.nan,
        "base_residual_positive_share": safe_float((residual_event["residual_gain_improvement_over_static"] > 1e-6).mean()) if "residual_gain_improvement_over_static" in residual_event else np.nan,
        "stress_base_residual_fraction": safe_float(stress_base.get("mean_residual_fraction_of_base_lp_gain")),
        "stress_low_residual_fraction": safe_float(stress_low.get("mean_residual_fraction_of_base_lp_gain")),
        "stress_delay4_residual_fraction": safe_float(stress_delay4.get("mean_residual_fraction_of_base_lp_gain")),
        "stress_scarce_residual_fraction": safe_float(stress_scarce.get("mean_residual_fraction_of_base_lp_gain")),
        "scenario_optimum_jobs": safe_int(len(optima)),
        "scenario_optimum_success_jobs": safe_int((optima["status"].astype(str) == "OPTIMAL").sum()) if "status" in optima else None,
        "scenario_optimum_timeout_jobs": safe_int((optima["status"].astype(str) == "ERROR").sum()) if "status" in optima else None,
        "scenario_static_fraction_of_lp_gain": safe_float(scenario_pivot["static_small_signal_greedy"].mean()) if not scenario_pivot.empty else np.nan,
        "scenario_residual_fraction_of_lp_gain": safe_float(scenario_pivot["residual_finite_greedy"].mean()) if not scenario_pivot.empty else np.nan,
        "scenario_residual_improvement": safe_float(scenario_pivot["residual_minus_static"].mean()) if not scenario_pivot.empty else np.nan,
        "scenario_residual_tie_or_win_share": safe_float((scenario_pivot["residual_minus_static"] >= -1e-6).mean()) if not scenario_pivot.empty else np.nan,
        "perturbed_solve_rows": safe_int(len(perturbed_solves)),
        "perturbed_success_rows": safe_int((perturbed_solves["status"].astype(str) == "OPTIMAL").sum()) if "status" in perturbed_solves else None,
        "perturbed_base_frequency_mass": policy_metric(perturbed_overlap, "base_lp", "frequency_mass_capture"),
        "perturbed_static_frequency_mass": policy_metric(perturbed_overlap, "static_small_signal_greedy", "frequency_mass_capture"),
        "perturbed_residual_frequency_mass": policy_metric(perturbed_overlap, "residual_finite_greedy", "frequency_mass_capture"),
        "perturbed_static_stable50_recall": policy_metric(perturbed_overlap, "static_small_signal_greedy", "stable50_recall"),
        "perturbed_residual_stable50_recall": policy_metric(perturbed_overlap, "residual_finite_greedy", "stable50_recall"),
        "event_mean_top5_value_share": safe_float(event_law["top_5pct_value_share"].mean()) if "top_5pct_value_share" in event_law else np.nan,
        "event_mean_marginal_value_gini": safe_float(event_law["marginal_value_gini"].mean()) if "marginal_value_gini" in event_law else np.nan,
        "event_loss_recoverable_spearman": safe_float(event_law[["baseline_objective", "recoverable_fraction"]].corr(method="spearman").iloc[0, 1]) if {"baseline_objective", "recoverable_fraction"}.issubset(event_law.columns) else np.nan,
        "event_top_tail_decision_spearman": safe_float(event_law[["top_5pct_value_share", "decision_criticality_score"]].corr(method="spearman").iloc[0, 1]) if {"top_5pct_value_share", "decision_criticality_score"}.issubset(event_law.columns) else np.nan,
        "symbolic_activated_mean_spearman": safe_float(activated_symbolic.get("mean_spearman")),
        "symbolic_activated_top5_capture": safe_float(activated_symbolic.get("mean_top_5pct_value_capture")),
        "symbolic_minimal_log_mean_spearman": safe_float(minimal_log_symbolic.get("mean_spearman")),
        "symbolic_minimal_log_top5_capture": safe_float(minimal_log_symbolic.get("mean_top_5pct_value_capture")),
        "symbolic_exposure_top5_capture": safe_float(exposure_symbolic.get("mean_top_5pct_value_capture")),
        "symbolic_full_ablation_top5_capture": safe_float(full_ablation.get("mean_top_5pct_value_capture")),
        "symbolic_without_od_top5_capture": safe_float(without_od.get("mean_top_5pct_value_capture")),
        "symbolic_largest_ablation_drop_group": str(largest_drop.get("removed_group", "")),
        "symbolic_largest_ablation_top5_drop": safe_float(largest_drop.get("top5_capture_drop_vs_full")),
        "symbolic_leave_city_rows": safe_int(len(symbolic_leave_city)),
        "budget_abs_random_peak_budget": str(budget_abs_random.get("peak_budget", "")),
        "budget_abs_random_interior_peak_supported": parse_bool(budget_abs_random.get("interior_peak_supported")),
        "budget_ratio_random_peak_budget": str(budget_ratio_random.get("peak_budget", "")),
        "budget_ratio_random_monotone_decreasing": parse_bool(budget_ratio_random.get("monotone_decreasing")),
        "budget_residual_per_cost_peak_budget": str(budget_per_cost.get("peak_budget", "")),
        "budget_low_proxy_leverage_vs_random": safe_float(budget_low.get("mean_proxy_leverage_vs_random_positive")),
        "budget_base_proxy_leverage_vs_random": safe_float(budget_base.get("mean_proxy_leverage_vs_random_positive")),
        "budget_high_proxy_leverage_vs_random": safe_float(budget_high.get("mean_proxy_leverage_vs_random_positive")),
        "budget_low_residual_static_per_cost": safe_float(budget_low.get("mean_residual_minus_static_per_cost")),
        "budget_base_residual_static_per_cost": safe_float(budget_base.get("mean_residual_minus_static_per_cost")),
        "budget_high_residual_static_per_cost": safe_float(budget_high.get("mean_residual_minus_static_per_cost")),
        "fine_budget_n_scales": safe_int(fine_budget_metrics.get("n_budget_scales")),
        "fine_budget_min_scale": safe_float(fine_budget_metrics.get("min_budget_scale")),
        "fine_budget_max_scale": safe_float(fine_budget_metrics.get("max_budget_scale")),
        "fine_budget_proxy_abs_peak_budget": safe_float(fine_budget_metrics.get("proxy_abs_peak_budget")),
        "fine_budget_proxy_abs_interior_peak_supported": parse_bool(fine_budget_metrics.get("proxy_abs_interior_peak_supported")),
        "fine_budget_proxy_abs_monotone_increasing": parse_bool(fine_budget_metrics.get("proxy_abs_monotone_increasing")),
        "fine_budget_replay_abs_peak_budget": safe_float(fine_budget_metrics.get("replay_abs_peak_budget")),
        "fine_budget_replay_abs_interior_peak_supported": parse_bool(fine_budget_metrics.get("replay_abs_interior_peak_supported")),
        "fine_budget_replay_abs_monotone_increasing": parse_bool(fine_budget_metrics.get("replay_abs_monotone_increasing")),
        "fine_budget_replay_per_budget_peak_budget": safe_float(fine_budget_metrics.get("replay_per_budget_peak_budget")),
        "fine_budget_replay_per_budget_monotone_decreasing": parse_bool(fine_budget_metrics.get("replay_per_budget_monotone_decreasing")),
        "fine_budget_replay_ratio_peak_budget": safe_float(fine_budget_metrics.get("replay_ratio_peak_budget")),
        "fine_budget_replay_ratio_monotone_decreasing": parse_bool(fine_budget_metrics.get("replay_ratio_monotone_decreasing")),
        "fine_budget_city_replay_abs_interior_peak_share": safe_float(fine_budget_metrics.get("city_replay_abs_interior_peak_share")),
        "fine_budget_base_law_fraction_of_oracle_replay_gain": safe_float(fine_budget_metrics.get("base_budget_law_fraction_of_oracle_replay_gain")),
        "fine_budget_base_replay_gain_leverage_vs_random": safe_float(fine_budget_metrics.get("base_budget_replay_gain_leverage_vs_random")),
        "early_decision_best_window": safe_int(early_decision_best.get("window_hours")),
        "early_decision_best_feature_group": str(early_decision_best.get("feature_group", "")),
        "early_decision_best_spearman": safe_float(early_decision_best.get("spearman")),
        "early_decision_best_top20_recall": safe_float(early_decision_best.get("top20_recall")),
        "early_decision_2h_all_spearman": safe_float(early_decision_2h.get("spearman")),
        "early_decision_2h_all_top20_recall": safe_float(early_decision_2h.get("top20_recall")),
        "early_recoverable_best_window": safe_int(early_recoverable_best.get("window_hours")),
        "early_recoverable_best_spearman": safe_float(early_recoverable_best.get("spearman")),
        "early_recoverable_2h_all_spearman": safe_float(early_recoverable_2h.get("spearman")),
        "early_decision_static_1h_spearman": safe_float(early_decision_static_1h.get("spearman")),
        "early_decision_speed_1h_spearman": safe_float(early_decision_speed_1h.get("spearman")),
        "early_decision_rain_1h_spearman": safe_float(early_decision_rain_1h.get("spearman")),
        "nonobvious_deficit_false_positive_share": safe_float(nonobvious_deficit.get("mean_false_positive_share")),
        "nonobvious_exposure_false_positive_share": safe_float(nonobvious_exposure.get("mean_false_positive_share")),
        "nonobvious_structure_false_positive_share": safe_float(nonobvious_structure.get("mean_false_positive_share")),
        "nonobvious_deficit_top5_relative_to_oracle": safe_float(nonobvious_deficit.get("mean_top5_relative_to_oracle")),
        "nonobvious_exposure_top5_relative_to_oracle": safe_float(nonobvious_exposure.get("mean_top5_relative_to_oracle")),
        "nonobvious_structure_top5_relative_to_oracle": safe_float(nonobvious_structure.get("mean_top5_relative_to_oracle")),
        "nonobvious_hidden_from_simple_top5_share": safe_float(nonobvious_hidden_all.get("hidden_from_all_simple_top5_share")),
        "nonobvious_hidden_from_simple_top20_share": safe_float(nonobvious_hidden_all.get("hidden_from_all_simple_top20_share")),
        "nonobvious_target_top5_low_structure_top20_share": safe_float(nonobvious_hidden_all.get("target_top5_low_structure_top20_share")),
        "nonobvious_deficit_failure_low_horizon_share": safe_float(nonobvious_deficit_reasons.get("below_median_future_horizon_share")),
        "nonobvious_deficit_failure_low_efficiency_share": safe_float(nonobvious_deficit_reasons.get("below_median_efficiency_share")),
        "nonobvious_structure_failure_low_exposure_share": safe_float(nonobvious_structure_reasons.get("below_median_exposure_share")),
        "nonobvious_structure_failure_low_efficiency_share": safe_float(nonobvious_structure_reasons.get("below_median_efficiency_share")),
        "nonobvious_peak_top5_relative_to_oracle": safe_float(nonobvious_peak.get("mean_top5_relative_to_oracle")),
        "nonobvious_remaining_area_top5_relative_to_oracle": safe_float(nonobvious_remaining.get("mean_top5_relative_to_oracle")),
        "factorized_deficit_top5_capture": safe_float(factorized_m1.get("mean_event_top_5pct_value_capture")),
        "factorized_deficit_event_spearman": safe_float(factorized_m1.get("mean_event_spearman")),
        "factorized_full_additive_top5_capture": safe_float(factorized_m5.get("mean_event_top_5pct_value_capture")),
        "factorized_full_additive_event_spearman": safe_float(factorized_m5.get("mean_event_spearman")),
        "factorized_full_interaction_top5_capture": safe_float(factorized_m6.get("mean_event_top_5pct_value_capture")),
        "factorized_full_interaction_event_spearman": safe_float(factorized_m6.get("mean_event_spearman")),
        "factorized_low_dim_top5_capture": safe_float(factorized_m7.get("mean_event_top_5pct_value_capture")),
        "factorized_low_dim_event_spearman": safe_float(factorized_m7.get("mean_event_spearman")),
        "factorized_low_dim_interaction_top5_capture": safe_float(factorized_m8.get("mean_event_top_5pct_value_capture")),
        "factorized_low_dim_interaction_event_spearman": safe_float(factorized_m8.get("mean_event_spearman")),
        "factorized_add_od_delta_top5_capture": safe_float(factorized_add_od.get("delta_top5_value_capture")),
        "factorized_add_time_delta_top5_capture": safe_float(factorized_add_time.get("delta_top5_value_capture")),
        "factorized_add_highdim_interaction_delta_top5_capture": safe_float(factorized_add_highdim_interaction.get("delta_top5_value_capture")),
        "factorized_add_lowdim_interaction_delta_top5_capture": safe_float(factorized_add_lowdim_interaction.get("delta_top5_value_capture")),
        "graph_no_graph_top5_capture": safe_float(graph_no_graph.get("mean_event_top_5pct_value_capture")),
        "graph_observed_full_top5_capture": safe_float(graph_observed_full.get("mean_event_top_5pct_value_capture")),
        "graph_shuffled_full_top5_capture": safe_float(graph_shuffled_full.get("mean_event_top_5pct_value_capture")),
        "graph_factorized_observed_top5_capture": safe_float(graph_factorized_observed.get("mean_event_top_5pct_value_capture")),
        "graph_factorized_shuffled_top5_capture": safe_float(graph_factorized_shuffled.get("mean_event_top_5pct_value_capture")),
        "graph_full_alignment_delta_top5_capture": safe_float(graph_full_alignment.get("delta_top5_capture")),
        "graph_full_alignment_delta_top5_ndcg": safe_float(graph_full_alignment.get("delta_top5_ndcg")),
        "graph_factorized_alignment_delta_top5_capture": safe_float(graph_factorized_alignment.get("delta_top5_capture")),
        "graph_factorized_alignment_delta_top5_ndcg": safe_float(graph_factorized_alignment.get("delta_top5_ndcg")),
        "graph_observed_over_no_graph_delta_top5_capture": safe_float(graph_observed_over_no_graph.get("delta_top5_capture")),
        "graph_observed_over_no_graph_delta_top5_ndcg": safe_float(graph_observed_over_no_graph.get("delta_top5_ndcg")),
        "regime_factorized_n_splits": safe_int(regime_factorized.get("n_splits")),
        "regime_factorized_mean_top5_capture": safe_float(regime_factorized.get("mean_top5_capture")),
        "regime_factorized_min_top5_capture": safe_float(regime_factorized.get("min_top5_capture")),
        "regime_factorized_mean_spearman": safe_float(regime_factorized.get("mean_spearman")),
        "regime_factorized_hardest_split_family": str(regime_factorized.get("hardest_split_family", "")),
        "regime_factorized_hardest_heldout": str(regime_factorized.get("hardest_heldout_regime", "")),
        "regime_full_mean_top5_capture": safe_float(regime_full.get("mean_top5_capture")),
        "regime_full_min_top5_capture": safe_float(regime_full.get("min_top5_capture")),
        "regime_interaction_mean_top5_capture": safe_float(regime_interaction.get("mean_top5_capture")),
        "regime_factorized_minus_full_mean_top5_delta": safe_float(regime_factorized_vs_full["delta_top5_capture"].mean()) if not regime_factorized_vs_full.empty else np.nan,
        "regime_factorized_minus_full_min_top5_delta": safe_float(regime_factorized_vs_full["delta_top5_capture"].min()) if not regime_factorized_vs_full.empty else np.nan,
        "regime_interaction_minus_full_mean_top5_delta": safe_float(regime_interaction_vs_full["delta_top5_capture"].mean()) if not regime_interaction_vs_full.empty else np.nan,
        "temporal_n_splits": safe_int(temporal_metrics.get("n_temporal_splits")),
        "temporal_main_n_splits": safe_int(temporal_metrics.get("n_main_within_city_splits")),
        "temporal_forward_n_splits": safe_int(temporal_metrics.get("n_forward_splits")),
        "temporal_year_n_splits": safe_int(temporal_metrics.get("n_year_confounded_splits")),
        "temporal_factorized_main_top5_capture": safe_float(temporal_factorized_main.get("mean_top5_capture")),
        "temporal_factorized_main_min_top5_capture": safe_float(temporal_factorized_main.get("min_top5_capture")),
        "temporal_factorized_main_spearman": safe_float(temporal_factorized_main.get("mean_spearman")),
        "temporal_factorized_main_hardest_split": str(temporal_factorized_main.get("hardest_split_family", "")),
        "temporal_factorized_main_hardest_heldout": str(temporal_factorized_main.get("hardest_heldout_period", "")),
        "temporal_full_main_top5_capture": safe_float(temporal_full_main.get("mean_top5_capture")),
        "temporal_full_main_min_top5_capture": safe_float(temporal_full_main.get("min_top5_capture")),
        "temporal_interaction_main_top5_capture": safe_float(temporal_interaction_main.get("mean_top5_capture")),
        "temporal_activated_main_top5_capture": safe_float(temporal_activated_main.get("mean_top5_capture")),
        "temporal_deficit_main_top5_capture": safe_float(temporal_deficit_main.get("mean_top5_capture")),
        "temporal_factorized_minus_full_main_top5_delta": safe_float(temporal_factorized_vs_full_main["delta_top5_capture"].mean()) if not temporal_factorized_vs_full_main.empty else np.nan,
        "temporal_activated_minus_deficit_main_top5_delta": safe_float(temporal_activated_vs_deficit_main["delta_top5_capture"].mean()) if not temporal_activated_vs_deficit_main.empty else np.nan,
        "temporal_factorized_forward_top5_capture": safe_float(temporal_factorized_forward.get("mean_top5_capture")),
        "temporal_factorized_forward_min_top5_capture": safe_float(temporal_factorized_forward.get("min_top5_capture")),
        "temporal_full_forward_top5_capture": safe_float(temporal_full_forward.get("mean_top5_capture")),
        "temporal_factorized_year_confounded_top5_capture": safe_float(temporal_factorized_year.get("mean_top5_capture")),
        "temporal_factorized_year_confounded_min_top5_capture": safe_float(temporal_factorized_year.get("min_top5_capture")),
        "temporal_full_year_confounded_top5_capture": safe_float(temporal_full_year.get("mean_top5_capture")),
        "temporal_year_holdout_design_note": str(temporal_metrics.get("year_holdout_design_note", "")),
        "neural_leave_city_factorized_mlp_top5_capture": safe_float(neural_metrics.get("leave_city_factorized_mlp_top5_capture")),
        "neural_leave_city_full_mlp_top5_capture": safe_float(neural_metrics.get("leave_city_full_mlp_top5_capture")),
        "neural_leave_city_factorized_ridge_top5_capture": safe_float(neural_metrics.get("leave_city_factorized_ridge_top5_capture")),
        "neural_leave_city_full_ridge_top5_capture": safe_float(neural_metrics.get("leave_city_full_ridge_top5_capture")),
        "neural_leave_city_full_mlp_minus_ridge_top5": safe_float(neural_metrics.get("leave_city_full_mlp_minus_ridge_top5")),
        "neural_leave_city_factorized_mlp_minus_ridge_top5": safe_float(neural_metrics.get("leave_city_factorized_mlp_minus_ridge_top5")),
        "neural_random_event_full_mlp_top5_capture": safe_float(neural_metrics.get("random_event_full_mlp_top5_capture")),
        "neural_random_event_full_ridge_top5_capture": safe_float(neural_metrics.get("random_event_full_ridge_top5_capture")),
        "neural_random_event_city_id_mlp_top5_capture": safe_float(neural_metrics.get("random_event_city_id_mlp_top5_capture")),
        "neural_random_event_city_id_minus_no_id_top5": safe_float(neural_metrics.get("random_event_city_id_minus_no_id_top5")),
        "neural_random_event_minus_leave_city_full_mlp_top5": safe_float(neural_metrics.get("random_event_minus_leave_city_full_mlp_top5")),
        "neural_token_random_full_mlp_spearman": safe_float(neural_metrics.get("token_random_full_mlp_spearman")),
        "neural_token_random_event_id_mlp_spearman": safe_float(neural_metrics.get("token_random_event_id_mlp_spearman")),
        "neural_token_random_event_id_minus_no_id_spearman": safe_float(neural_metrics.get("token_random_event_id_minus_no_id_spearman")),
        "objective_factorized_raw_top5_capture": safe_float(objective_factorized.get("raw_log_top5_capture")),
        "objective_factorized_top_tail_weighted_top5_capture": safe_float(objective_factorized.get("top_tail_weighted_top5_capture")),
        "objective_factorized_rank_top5_capture": safe_float(objective_factorized.get("rank_percentile_top5_capture")),
        "objective_factorized_best_objective": str(objective_factorized.get("best_objective_id", "")),
        "objective_factorized_best_top5_capture": safe_float(objective_factorized.get("best_top5_capture")),
        "objective_factorized_best_minus_raw_top5_capture": safe_float(objective_factorized.get("best_minus_raw_top5_capture")),
        "objective_factorized_raw_top5_regret": safe_float(objective_factorized_raw.get("mean_event_top_5pct_regret")),
        "objective_full_best_objective": str(objective_full.get("best_objective_id", "")),
        "objective_full_best_top5_capture": safe_float(objective_full.get("best_top5_capture")),
        "objective_full_best_minus_raw_top5_capture": safe_float(objective_full.get("best_minus_raw_top5_capture")),
        "objective_full_best_top5_regret": safe_float(objective_full_best.get("mean_event_top_5pct_regret")),
        "objective_interaction_best_top5_capture": safe_float(objective_interaction.get("best_top5_capture")),
        "objective_interaction_best_minus_raw_top5_capture": safe_float(objective_interaction.get("best_minus_raw_top5_capture")),
        "parameter_policy_clock_top5_capture": safe_float(parameter_clock.get("mean_event_top_5pct_value_capture")),
        "parameter_clock_plus_efficiency_top5_capture": safe_float(parameter_efficiency.get("mean_event_top_5pct_value_capture")),
        "parameter_add_od_exposure_top5_capture": safe_float(parameter_exposure.get("mean_event_top_5pct_value_capture")),
        "parameter_add_structure_top5_capture": safe_float(parameter_structure.get("mean_event_top_5pct_value_capture")),
        "parameter_light_factorized_top5_capture": safe_float(parameter_light_factorized.get("mean_event_top_5pct_value_capture")),
        "parameter_full_factorized_top5_capture": safe_float(parameter_full_factorized.get("mean_event_top_5pct_value_capture")),
        "parameter_full_additive_top5_capture": safe_float(parameter_full_additive.get("mean_event_top_5pct_value_capture")),
        "parameter_add_efficiency_delta_top5_capture": safe_float(parameter_add_efficiency.get("delta_top5_value_capture")),
        "parameter_add_od_delta_top5_capture": safe_float(parameter_add_od.get("delta_top5_value_capture")),
        "parameter_light_over_clock_delta_top5_capture": safe_float(parameter_light_over_clock.get("delta_top5_value_capture")),
        "parameter_efficiency_to_factorized_delta_top5_capture": safe_float(parameter_eff_to_factorized.get("delta_top5_value_capture")),
        "parameter_full_over_clock_delta_top5_capture": safe_float(parameter_full_over_clock.get("delta_top5_value_capture")),
        "parameter_channel_efficiency_top10_capture": safe_float(channel_efficiency.get("mean_top10_value_capture")),
        "parameter_channel_light_activation_top10_capture": safe_float(channel_light_activation.get("mean_top10_value_capture")),
        "parameter_channel_full_activation_top10_capture": safe_float(channel_full_activation.get("mean_top10_value_capture")),
        "parameter_channel_n_groups": safe_int(channel_full_activation.get("n_channels")),
        "event_decision_v21_decision_vs_loss_spearman": safe_float(event_decision_metrics.get("decision_vs_baseline_loss_spearman")),
        "event_decision_v21_decision_vs_top5_spearman": safe_float(event_decision_metrics.get("decision_vs_top5_share_spearman")),
        "event_decision_v21_decision_vs_gini_spearman": safe_float(event_decision_metrics.get("decision_vs_gini_spearman")),
        "event_decision_v21_top5_between_city_share": safe_float(event_decision_metrics.get("top5_between_city_share")),
        "event_decision_v21_gini_between_city_share": safe_float(event_decision_metrics.get("gini_between_city_share")),
        "event_decision_v21_severity_only_loco_spearman": safe_float(event_decision_metrics.get("severity_only_decision_loco_spearman")),
        "event_decision_v21_top_tail_loco_spearman": safe_float(event_decision_metrics.get("top_tail_decision_loco_spearman")),
        "event_decision_v21_high_loss_low_decision_count": safe_int(event_decision_metrics.get("high_loss_low_decision_count")),
        "event_decision_v21_moderate_loss_high_decision_count": safe_int(event_decision_metrics.get("moderate_loss_high_decision_count")),
        "event_decision_v21_high_rain_low_decision_count": safe_int(event_decision_metrics.get("high_rain_low_decision_count")),
        "od_message_scalar_od_top5_capture": safe_float(od_message_scalar.get("mean_event_top_5pct_value_capture")),
        "od_message_message_only_top5_capture": safe_float(od_message_message_only.get("mean_event_top_5pct_value_capture")),
        "od_message_scalar_plus_top5_capture": safe_float(od_message_scalar_plus.get("mean_event_top_5pct_value_capture")),
        "od_message_factorized_top5_capture": safe_float(od_message_factorized.get("mean_event_top_5pct_value_capture")),
        "od_message_factorized_plus_top5_capture": safe_float(od_message_factorized_plus.get("mean_event_top_5pct_value_capture")),
        "od_message_message_over_scalar_delta_top5_capture": safe_float(od_message_over_scalar.get("delta_top5_value_capture")),
        "od_message_message_over_factorized_delta_top5_capture": safe_float(od_message_over_factorized.get("delta_top5_value_capture")),
        "od_message_message_over_scalar_delta_event_spearman": safe_float(od_message_over_scalar.get("delta_event_spearman")),
        "od_message_message_over_factorized_delta_event_spearman": safe_float(od_message_over_factorized.get("delta_event_spearman")),
        "od_message_metrics_message_over_scalar_delta_top5": safe_float(od_message_metrics.get("message_over_scalar_od_delta_top5")),
        "od_message_metrics_message_over_factorized_delta_top5": safe_float(od_message_metrics.get("message_over_factorized_delta_top5")),
        "parameter_ensemble_n_scenarios": safe_int(parameter_ensemble_metrics.get("n_parameter_scenarios")),
        "parameter_ensemble_base_transfer_light_mean_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_parameter_light_mean_top5_capture")),
        "parameter_ensemble_base_transfer_light_min_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_parameter_light_min_top5_capture")),
        "parameter_ensemble_base_transfer_full_mean_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_factorized_mean_top5_capture")),
        "parameter_ensemble_base_transfer_full_min_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_factorized_min_top5_capture")),
        "parameter_ensemble_base_transfer_centered_mean_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_centered_factorized_mean_top5_capture")),
        "parameter_ensemble_base_transfer_centered_min_top5_capture": safe_float(parameter_ensemble_metrics.get("base_transfer_centered_factorized_min_top5_capture")),
        "parameter_ensemble_base_transfer_full_worst_scenario": str(parameter_ensemble_metrics.get("worst_base_transfer_factorized_scenario", "")),
        "parameter_ensemble_full_activation_score_mean_top5_capture": safe_float(parameter_ensemble_metrics.get("full_activation_score_mean_top5_capture")),
        "parameter_ensemble_light_activation_score_mean_top5_capture": safe_float(parameter_ensemble_metrics.get("light_activation_score_mean_top5_capture")),
        "parameter_lp_n_selected_events": safe_int(parameter_lp_metrics.get("n_selected_events")),
        "parameter_lp_n_parameter_scenarios": safe_int(parameter_lp_metrics.get("n_parameter_scenarios")),
        "parameter_lp_n_successful_lp_scenarios": safe_int(parameter_lp_metrics.get("n_successful_lp_scenarios")),
        "parameter_lp_mean_residual_fraction_of_scenario_lp_gain": safe_float(parameter_lp_metrics.get("mean_residual_fraction_of_scenario_lp_gain")),
        "parameter_lp_median_residual_fraction_of_scenario_lp_gain": safe_float(parameter_lp_metrics.get("median_residual_fraction_of_scenario_lp_gain")),
        "parameter_lp_mean_static_fraction_of_scenario_lp_gain": safe_float(parameter_lp_metrics.get("mean_static_fraction_of_scenario_lp_gain")),
        "parameter_lp_mean_residual_minus_static": safe_float(parameter_lp_metrics.get("mean_residual_minus_static")),
        "parameter_lp_positive_residual_improvement_share": safe_float(parameter_lp_metrics.get("positive_residual_improvement_share")),
        "parameter_lp_worst_residual_parameter_scenario": str(parameter_lp_metrics.get("worst_residual_parameter_scenario", "")),
        "parameter_lp_worst_residual_mean_fraction_of_scenario_lp_gain": safe_float(parameter_lp_metrics.get("worst_residual_mean_fraction_of_scenario_lp_gain")),
    }
    return metrics


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


def build_evidence_ladder(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "version": "V4/V5",
            "evidence_step": "Single-action LP validates label",
            "main_question": "What is the right marginal recovery-value label?",
            "key_metric": "small_signal_spearman",
            "value": metrics["single_action_small_signal_spearman"],
            "interpretation": "The first-segment derivative, not finite deficit area, matches direct single-action LP probes.",
        },
        {
            "version": "V5",
            "evidence_step": "Cross-city action-value field",
            "main_question": "Can normalized structural features recover the value ranking?",
            "key_metric": "leave_city_mean_spearman",
            "value": metrics["leave_city_mean_spearman"],
            "interpretation": "A simple surrogate generalizes action-value ranking across held-out cities.",
        },
        {
            "version": "V6",
            "evidence_step": "Finite-budget gap",
            "main_question": "Does a static first-order ranking solve the full budget problem?",
            "key_metric": "static_fraction_of_lp_gain",
            "value": metrics["finite_gap_mean_static_fraction_of_lp_gain"],
            "interpretation": "Static small-signal greedy captures substantial value but leaves a finite-budget interaction gap.",
        },
        {
            "version": "V7",
            "evidence_step": "Residual finite-budget law",
            "main_question": "Does re-scoring on the residual state close the base LP gap?",
            "key_metric": "residual_fraction_of_lp_gain",
            "value": metrics["base_residual_fraction_of_lp_gain"],
            "interpretation": "Residual replanning nearly closes the base-scenario LP optimum on average.",
        },
        {
            "version": "V8",
            "evidence_step": "Budget/delay stress test",
            "main_question": "Is the residual law stable under resource and response perturbations?",
            "key_metric": "delay4_residual_fraction_of_base_lp_gain",
            "value": metrics["stress_delay4_residual_fraction"],
            "interpretation": "Residual replanning remains systematically above static greedy under delayed response.",
        },
        {
            "version": "V9",
            "evidence_step": "Scenario-specific LP closure",
            "main_question": "Does the law still approach the true non-base scenario optimum?",
            "key_metric": "residual_fraction_of_scenario_lp_gain",
            "value": metrics["scenario_residual_fraction_of_lp_gain"],
            "interpretation": "Representative non-base LP solves show residual finite greedy close to scenario-specific optima.",
        },
        {
            "version": "V11",
            "evidence_step": "Perturbed-optimum stability",
            "main_question": "Are recovery-critical actions stable under small cost/effectiveness perturbations?",
            "key_metric": "residual_perturbed_frequency_mass_capture",
            "value": metrics["perturbed_residual_frequency_mass"],
            "interpretation": "Residual finite greedy captures more perturbed LP selection-frequency mass than static greedy, but stable action lists remain parameter-sensitive.",
        },
        {
            "version": "V12",
            "evidence_step": "Symbolic formula extraction and structure decoupling",
            "main_question": "Can the action-value field be compressed into a low-complexity law and stable feature groups?",
            "key_metric": "activated_symbolic_top5_capture",
            "value": metrics["symbolic_activated_top5_capture"],
            "interpretation": "The compact activated law sits on the formula Pareto frontier; OD exposure/structure is the largest feature-group contributor in ablation.",
        },
        {
            "version": "V13",
            "evidence_step": "Budget-leverage phase analysis",
            "main_question": "Is decision leverage highest at intermediate budget levels?",
            "key_metric": "interior_budget_peak_supported",
            "value": float(metrics["budget_abs_random_interior_peak_supported"]),
            "interpretation": "The current low/base/high scan does not support an interior absolute-leverage peak; absolute leverage rises with budget while per-budget leverage diminishes.",
        },
        {
            "version": "V14",
            "evidence_step": "Early predictability and hindsight boundary",
            "main_question": "Can decision-critical events be identified before the full 12-hour trajectory is known?",
            "key_metric": "2h_all_early_decision_spearman",
            "value": metrics["early_decision_2h_all_spearman"],
            "interpretation": "Decision-criticality is partly identifiable from early speed and static structure, but rainfall-only signals are insufficient; the main claim remains hindsight counterfactual recoverability.",
        },
        {
            "version": "V15",
            "evidence_step": "Non-obvious action-law failures",
            "main_question": "Are highest-deficit, highest-exposure, or highest-bottleneck actions sufficient recovery rules?",
            "key_metric": "structure_only_false_positive_share",
            "value": metrics["nonobvious_structure_false_positive_share"],
            "interpretation": "One-factor heuristics miss the activation mechanism: static bottlenecks become valuable only when future loss, OD exposure, feasibility, and efficiency align.",
        },
        {
            "version": "V16",
            "evidence_step": "Factorized surrogate and interaction ablation",
            "main_question": "Does recovery value require arbitrary high-dimensional interactions or a low-dimensional activated structure?",
            "key_metric": "low_dim_factorized_top5_capture",
            "value": metrics["factorized_low_dim_top5_capture"],
            "interpretation": "A low-dimensional factorized surrogate captures most top-tail value; OD exposure and time/feasibility add large gains, while unconstrained high-dimensional interactions do not improve leave-city performance.",
        },
        {
            "version": "V17",
            "evidence_step": "OD graph structure alignment ablation",
            "main_question": "Does observed OD graph alignment add value beyond local dynamics or shuffled city-level graph distributions?",
            "key_metric": "observed_vs_shuffled_od_graph_top5_capture_gap",
            "value": metrics["graph_full_alignment_delta_top5_capture"],
            "interpretation": "Observed OD graph alignment strongly outperforms a within-city shuffled graph, showing that spatial alignment between action location and OD exposure is part of the recoverability law.",
        },
        {
            "version": "V18",
            "evidence_step": "Event-regime generalization",
            "main_question": "Does the low-dimensional law remain useful when entire rainfall, impact, duration, loss, or time-of-day regimes are held out?",
            "key_metric": "factorized_min_heldout_regime_top5_capture",
            "value": metrics["regime_factorized_min_top5_capture"],
            "interpretation": "The factorized law keeps high top-tail capture across held-out event regimes, supporting a structural rather than regime-memorized law.",
        },
        {
            "version": "V19",
            "evidence_step": "Training-objective ablation",
            "main_question": "Does the recovered law require a special ranking or top-tail-weighted training objective?",
            "key_metric": "factorized_best_minus_raw_log_top5_capture",
            "value": metrics["objective_factorized_best_minus_raw_top5_capture"],
            "interpretation": "For the compact factorized law, ordinary log-value regression is already the best top-tail objective; ranking-aware variants are useful robustness checks rather than the source of the law.",
        },
        {
            "version": "V20",
            "evidence_step": "Parameter-deconfounded structure test",
            "main_question": "Is the law only a reflection of intervention timing, cost, and effectiveness parameters?",
            "key_metric": "parameter_light_factorized_over_clock_delta_top5_capture",
            "value": metrics["parameter_light_over_clock_delta_top5_capture"],
            "interpretation": "A factorized law without eta/cost still strongly outperforms policy-clock and intervention-type controls, showing that OD exposure and future loss alignment carry structural value beyond action mechanics.",
        },
        {
            "version": "V21",
            "evidence_step": "Event-level severity decoupling",
            "main_question": "Are the most severe rainfall-loss events automatically the most decision-critical?",
            "key_metric": "decision_vs_top5_share_spearman",
            "value": metrics["event_decision_v21_decision_vs_top5_spearman"],
            "interpretation": "Decision-criticality aligns with marginal-value top-tail concentration rather than loss magnitude; V21 also shows that the current top-tail signal is partly city-structural because event footprints are not yet region-specific.",
        },
        {
            "version": "V22",
            "evidence_step": "OD message-passing surrogate",
            "main_question": "Do explicit one-hop/two-hop OD-neighborhood messages add value beyond compact structural laws?",
            "key_metric": "message_over_scalar_od_delta_top5_capture",
            "value": metrics["od_message_message_over_scalar_delta_top5_capture"],
            "interpretation": "OD message-only features are informative, but adding them on top of scalar OD exposure/structure gives only a tiny top-tail gain and adding them to the low-dimensional factorized law slightly lowers top-5% capture; higher-order OD message passing is therefore not necessary for the current compact recoverability law.",
        },
        {
            "version": "V23",
            "evidence_step": "Parameter-ensemble action-law stability",
            "main_question": "Does the compact law survive changes in intervention effectiveness, cost, and response delay assumptions?",
            "key_metric": "base_trained_parameter_light_factorized_mean_top5_capture",
            "value": metrics["parameter_ensemble_base_transfer_light_mean_top5_capture"],
            "interpretation": "Across 11 eta/cost/delay scenarios, a base-trained parameter-light factorized law transfers with high top-tail capture; parameter-dependent ridge features are more scale-sensitive, so scenario augmentation is important for parameter-sensitive surrogates.",
        },
        {
            "version": "V24",
            "evidence_step": "Parameter-ensemble LP optimum closure",
            "main_question": "Does residual replanning remain close to full LP optima under eta/cost/delay/channel-favored parameter ensembles?",
            "key_metric": "residual_fraction_of_parameter_scenario_lp_gain",
            "value": metrics["parameter_lp_mean_residual_fraction_of_scenario_lp_gain"],
            "interpretation": "Across 20 representative parameter-scenario LP closures, residual finite greedy captures most scenario-specific LP gain and dominates static small-signal ranking in every tested event-scenario.",
        },
        {
            "version": "V25",
            "evidence_step": "Within-city temporal holdout robustness",
            "main_question": "Does the action-value law survive chronological event splits without relying on same-period leakage?",
            "key_metric": "factorized_within_city_temporal_top5_capture",
            "value": metrics["temporal_factorized_main_top5_capture"],
            "interpretation": "Within each city, early/middle/late and early/late event periods are held out; the compact factorized law keeps high top-tail capture, while year-holdout is reported only as city-confounded audit.",
        },
        {
            "version": "V26",
            "evidence_step": "Neural surrogate and identity-leakage audit",
            "main_question": "Does a lightweight neural surrogate reveal hidden nonlinear structure, or do random splits and identities inflate apparent performance?",
            "key_metric": "leave_city_full_mlp_minus_ridge_top5",
            "value": metrics["neural_leave_city_full_mlp_minus_ridge_top5"],
            "interpretation": "On strict leave-city top-tail evaluation, the MLP does not beat ridge; city identity adds only a tiny random-event gain, while event identity inflates token-random correlation. This supports compact laws and strict split design.",
        },
        {
            "version": "V27",
            "evidence_step": "Fine budget-leverage sweep",
            "main_question": "Does decision leverage peak at an intermediate budget once the budget grid is refined?",
            "key_metric": "fine_budget_replay_abs_interior_peak_supported",
            "value": float(metrics["fine_budget_replay_abs_interior_peak_supported"]),
            "interpretation": "Across a 10-point budget grid, absolute law-versus-random replay leverage peaks at the largest tested budget, while relative and per-budget leverage peak at the smallest tested budget and decline monotonically.",
        },
    ]
    return pd.DataFrame(rows)


def build_policy_closure_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    residual_event = data["residual_event"]
    if not residual_event.empty:
        rows.extend(
            [
                {
                    "scope": "base_all_105",
                    "comparison": "static_small_signal_vs_base_LP",
                    "n": len(residual_event),
                    "mean_fraction_of_lp_gain": residual_event["static_fraction_of_lp_gain"].mean(),
                    "median_fraction_of_lp_gain": residual_event["static_fraction_of_lp_gain"].median(),
                    "mean_improvement_over_static": 0.0,
                },
                {
                    "scope": "base_all_105",
                    "comparison": "residual_finite_greedy_vs_base_LP",
                    "n": len(residual_event),
                    "mean_fraction_of_lp_gain": residual_event["residual_fraction_of_lp_gain"].mean(),
                    "median_fraction_of_lp_gain": residual_event["residual_fraction_of_lp_gain"].median(),
                    "mean_improvement_over_static": residual_event["residual_gain_improvement_over_static"].mean(),
                },
            ]
        )
    stress = data["stress_scenario"]
    if not stress.empty:
        for row in stress.itertuples(index=False):
            rows.append(
                {
                    "scope": f"stress_{row.policy_scenario}_105",
                    "comparison": "residual_finite_greedy_vs_static_reference",
                    "n": int(row.n_events),
                    "mean_fraction_of_lp_gain": float(row.mean_residual_fraction_of_base_lp_gain),
                    "median_fraction_of_lp_gain": float(row.median_residual_fraction_of_base_lp_gain),
                    "mean_improvement_over_static": float(row.mean_residual_minus_static),
                }
            )
    scenario = data["scenario_policy"]
    if not scenario.empty:
        for policy, group in scenario.groupby("policy"):
            rows.append(
                {
                    "scope": "representative_nonbase_scenario_LP_23",
                    "comparison": f"{policy}_vs_scenario_LP",
                    "n": len(group),
                    "mean_fraction_of_lp_gain": group["fraction_of_scenario_lp_gain"].mean(),
                    "median_fraction_of_lp_gain": group["fraction_of_scenario_lp_gain"].median(),
                    "mean_improvement_over_static": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_city_closure_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    city = data["residual_city"].copy()
    if city.empty:
        return city
    keep = [
        "city",
        "n_events",
        "mean_static_fraction_of_lp_gain",
        "mean_residual_fraction_of_lp_gain",
        "mean_residual_gain_improvement",
        "mean_residual_gap_to_lp",
    ]
    city = city[[column for column in keep if column in city.columns]].copy()
    return city.sort_values("mean_residual_gain_improvement", ascending=False)


def build_decision_examples(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    event_law = data["event_law"].copy()
    if event_law.empty:
        return event_law
    examples = event_law.sort_values("decision_criticality_score", ascending=False).head(15).copy()
    keep = [
        "city",
        "event_id",
        "event_start",
        "baseline_objective",
        "recoverable_fraction",
        "top_5pct_value_share",
        "marginal_value_gini",
        "loss_magnitude_rank",
        "recoverable_rank",
        "top_tail_rank",
        "decision_criticality_score",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    return examples[[column for column in keep if column in examples.columns]]


def build_top_tail_correlations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    event_law = data["event_law"]
    if event_law.empty:
        return pd.DataFrame()
    targets = ["recoverable_fraction", "decision_criticality_score"]
    features = [
        "baseline_objective",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
        "top_1pct_value_share",
        "top_5pct_value_share",
        "top_10pct_value_share",
        "marginal_value_gini",
        "optimizer_selected_value_share",
    ]
    rows: list[dict[str, Any]] = []
    for target in targets:
        for feature in features:
            if target in event_law and feature in event_law:
                pair = event_law[[target, feature]].dropna()
                corr = pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else np.nan
                rows.append({"target": target, "feature": feature, "spearman": corr})
    return pd.DataFrame(rows).sort_values(["target", "spearman"], ascending=[True, False])


def build_limitations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    optima = data["scenario_optima"]
    timeout_count = int((optima["status"].astype(str) == "ERROR").sum()) if "status" in optima else 0
    return pd.DataFrame(
        [
            {
                "item": "scenario_optimum_coverage",
                "current_status": f"{len(optima) - timeout_count} successful LP closures; {timeout_count} time-limit/error rows",
                "implication": "V9 supports representative non-base closure, not full 105-event scenario-optimum closure.",
                "next_step": "Expand scenario-specific LP validation with resume mode or longer time limits for hard New York/Chicago/Philadelphia cases.",
            },
            {
                "item": "intervention_parameter_identification",
                "current_status": "R/C/S effectiveness, cost, caps, delays, and diminishing returns are recovery-regime assumptions; V20 adds parameter-deconfounded action-token tests, V23 adds first-order eta/cost/delay parameter-ensemble stability, and V24 adds representative full-LP parameter-ensemble closure.",
                "implication": "The law is still conditional on a management regime, but future-loss and OD-exposure alignment transfer across moderate parameter perturbations; residual replanning remains close to representative perturbed LP optima.",
                "next_step": "Expand full LP parameter ensembles across more city-events or incorporate observed intervention records if parameter identification becomes a central claim.",
            },
            {
                "item": "surrogate_architecture",
                "current_status": "V16 adds a factorized leave-one-city action-value surrogate; V17 adds observed-vs-shuffled OD graph alignment; V22 tests explicit one-hop/two-hop OD message features; V26 compares ridge and lightweight MLP surrogates under leave-city, random-event, and token-random splits with identity audits.",
                "implication": "The low-dimensional activated structure is strongly supported; higher-order OD messages add little beyond scalar OD alignment, and the MLP does not beat ridge under strict leave-city top-tail evaluation. Random token splits with event identity can inflate apparent fit.",
                "next_step": "Use neural graph or listwise surrogates only if the paper later makes higher-order spatial representation or operational prediction a central contribution.",
            },
            {
                "item": "graph_structure_scope",
                "current_status": "V17 tests OD-dependency graph features and within-city shuffled graph alignment; V22 tests deterministic OD-neighborhood message summaries, but not a unified road-adjacency graph.",
                "implication": "The paper can claim evidence for OD exposure alignment as city structure and show that simple OD message passing is not currently needed, but not yet claim full road-topology or GNN closure.",
                "next_step": "Add road adjacency, TMC-zone linkage, or graph neural baselines if higher-order physical topology becomes a central contribution.",
            },
            {
                "item": "event_regime_generalization_scope",
                "current_status": "V18 tests held-out event regimes for rain intensity, peak rain, speed impact, duration, baseline loss, recoverable fraction, time of day, and weekday/weekend.",
                "implication": "The factorized law is not only leave-city robust, but the year-based temporal split remains city-confounded in the current sample.",
                "next_step": "Add unconfounded multi-year observations within the same cities before claiming clean leave-time-period-out generalization.",
            },
            {
                "item": "temporal_generalization_scope",
                "current_status": "V25 adds within-city chronological holdouts over early/middle/late and early/late event-order periods, plus forward chronological audits and a separately labeled year-holdout audit.",
                "implication": "The law is robust to same-city event-order holdout, but calendar-year holdout still mixes temporal extrapolation with city composition.",
                "next_step": "Collect repeated months or years for the same cities so a future leave-time-period-out test can hold time fixed independently from city identity.",
            },
            {
                "item": "event_spatial_footprint_scope",
                "current_status": "V21 finds event-level top-tail concentration is strongly linked to decision-criticality, but top-5% value share has a sizable between-city variance component.",
                "implication": "The event-level law is currently a city-structure/top-tail law more than a fully event-specific spatial-footprint law.",
                "next_step": "Add zone-level speed/rainfall footprint mapping to test whether top-tail concentration varies strongly across events within the same city.",
            },
            {
                "item": "training_objective_scope",
                "current_status": "V19 compares ordinary log-value, event-centered, top-tail-weighted, rank-percentile, and event-zscore ridge objectives.",
                "implication": "Top-tail metrics are essential for evaluation, but the compact law does not require a special ranking-loss trick in the current data.",
                "next_step": "Use true pairwise/listwise neural ranking losses only if a later high-capacity surrogate is introduced.",
            },
            {
                "item": "perturbed_optimum_stability",
                "current_status": "Representative perturbation solves are available for 4 events with 3 cost/effectiveness perturbations each.",
                "implication": "V23 expands first-order action-value parameter stability and V24 adds 20 representative full-LP parameter-scenario closures, but exact action-list stability remains limited to representative samples.",
                "next_step": "Increase perturbation count and city-event coverage if action stability becomes a central claim.",
            },
            {
                "item": "budget_phase_coverage",
                "current_status": "V13 used low/base/high budget scales; V27 adds a 10-point budget sweep from 0.10 to 3.00 using action-value proxy allocation and LP replay dynamics.",
                "implication": "The refined sweep still rejects an interior absolute-leverage peak in the current proxy/replay setting: absolute law-versus-random leverage grows to the largest tested budget, while relative and per-budget leverage peak at the smallest budget and decline.",
                "next_step": "Use scenario-specific full LP optima over the same fine budget grid, plus multiple random-policy seeds, if budget phase shape becomes a central contribution.",
            },
            {
                "item": "online_predictability_scope",
                "current_status": "Early-window predictability is tested with leave-one-city-out ridge models using 1/2/3/6/12 hour aggregate features.",
                "implication": "Early decision-criticality signals are supplementary and should not be framed as a full online control policy.",
                "next_step": "Use rolling operational forecasts or causal nowcasting data before making real-time deployment claims.",
            },
            {
                "item": "nonobvious_action_scope",
                "current_status": "V15 compares simple action heuristics against optimizer-derived marginal-value tokens inside each observed city-event.",
                "implication": "The failure examples support the activated-law mechanism, but they are ranking diagnostics rather than independent causal evidence about real interventions.",
                "next_step": "Validate the same non-obvious patterns under parameter ensembles and, if available, observed intervention records.",
            },
        ]
    )


def make_figures(
    data: dict[str, pd.DataFrame],
    closure: pd.DataFrame,
    city_closure: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_evidence_ladder_figure(closure, figure_dir / "law_evidence_ladder.png")
    make_city_closure_figure(city_closure, figure_dir / "city_residual_closure.png")
    make_top_tail_phase_figure(data["event_law"], figure_dir / "event_top_tail_phase.png")


def make_evidence_ladder_figure(closure: pd.DataFrame, path: Path) -> None:
    if closure.empty:
        return
    selected = closure[closure["scope"].isin(["base_all_105", "representative_nonbase_scenario_LP_23"])].copy()
    selected["label"] = selected["comparison"].map(
        {
            "static_small_signal_vs_base_LP": "Base static",
            "residual_finite_greedy_vs_base_LP": "Base residual",
            "static_small_signal_greedy_vs_scenario_LP": "Scenario static",
            "residual_finite_greedy_vs_scenario_LP": "Scenario residual",
        }
    )
    selected = selected.dropna(subset=["label"])
    order = ["Base static", "Base residual", "Scenario static", "Scenario residual"]
    selected["label"] = pd.Categorical(selected["label"], categories=order, ordered=True)
    selected = selected.sort_values("label")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = ["#94a3b8" if "static" in label.lower() else "#2563eb" for label in selected["label"].astype(str)]
    ax.bar(selected["label"].astype(str), selected["mean_fraction_of_lp_gain"], color=colors, width=0.62)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Policy gain / LP gain")
    ax.set_title("From first-order ranking to residual finite-budget law")
    for idx, value in enumerate(selected["mean_fraction_of_lp_gain"]):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_closure_figure(city_closure: pd.DataFrame, path: Path) -> None:
    if city_closure.empty:
        return
    ordered = city_closure.sort_values("mean_residual_gain_improvement", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(y - 0.18, ordered["mean_static_fraction_of_lp_gain"], height=0.36, color="#94a3b8", label="static")
    ax.barh(y + 0.18, ordered["mean_residual_fraction_of_lp_gain"], height=0.36, color="#2563eb", label="residual")
    ax.axvline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y, ordered["city"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Base-scenario policy gain / LP gain")
    ax.set_title("Residual law closes the finite-budget gap across cities")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_top_tail_phase_figure(event_law: pd.DataFrame, path: Path) -> None:
    if event_law.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    scatter = ax.scatter(
        event_law["baseline_objective"],
        event_law["recoverable_fraction"],
        c=event_law["top_5pct_value_share"],
        s=52,
        cmap="viridis",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    top = event_law.sort_values("decision_criticality_score", ascending=False).head(6)
    for row in top.itertuples(index=False):
        ax.annotate(f"{row.city} {int(row.event_id)}", (row.baseline_objective, row.recoverable_fraction), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("No-intervention loss objective, log scale")
    ax.set_ylabel("Recoverable fraction under base LP")
    ax.set_title("Decision-criticality separates loss magnitude from recoverable top tail")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Top-5% marginal value share")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    evidence: pd.DataFrame,
    closure: pd.DataFrame,
    city_closure: pd.DataFrame,
    decision_examples: pd.DataFrame,
    top_tail_correlations: pd.DataFrame,
    limitations: pd.DataFrame,
) -> None:
    lines = [
        "# Recoverability Law Synthesis V27",
        "",
        "V27 adds a fine-grained budget-leverage sweep on top of the V26 neural-surrogate and identity-leakage audit. It tests whether the expected intermediate-budget leverage peak survives when the budget grid is refined from three points to ten points.",
        "",
        "## 本版做了什么",
        "",
        "V27 在 V26 neural surrogate / leakage audit 之后，补上更细的 budget-leverage phase 检验：预算从 0.10 到 3.00 共 10 个点，并在同一 action-value proxy 与 LP replay 动力学下比较 law、oracle、deficit-only、exposure-only、structure-only 和 random-positive 策略。",
        "",
        "## 当前可写入论文的 law",
        "",
        "1. **Small-signal activated recovery law**：第一小段资源的边际价值由 future recoverable horizon、OD exposure、intervention feasibility 和 efficiency 共同激活。",
        "",
        "2. **Residual finite-budget allocation law**：完整预算下，价值必须在每轮投放后按 residual state、remaining budget 和 remaining time 重新评分，以避免饱和和重叠效应。",
        "",
        "3. **Top-tail decision-criticality law**：事件是否 decision-critical 不只取决于 observed loss，而取决于 recoverable value 是否集中在少数高价值 action 上。",
        "",
        "4. **Non-obvious activation law**：高损失、高流量或高结构中心性都不是充分条件；静态城市结构只有被未来损失、OD 暴露、响应窗口和资源效率同时激活，才转化为 recovery value。",
        "",
        "5. **Factorized parsimony result**：跨城市 action-value surrogate 的最大增益来自 OD exposure 和 time/feasibility activation；不受约束的高维 interaction terms 没有提高 full model。",
        "",
        "6. **OD graph alignment law**：在城市内打乱 OD graph feature 与 unit 的对应关系会显著降低 top-tail value capture，说明城市结构不是只作为分布统计量起作用，真实空间对齐本身是 recoverability law 的一部分。",
        "",
        "7. **Event-regime stability result**：低维 factorized law 在整类雨强、速度冲击、持续时间、损失规模和时段被留出时仍保持较高 top-tail capture，说明它不是只记住某一类事件 regime。",
        "",
        "8. **Training-objective robustness result**：top-tail capture 和 regret 是核心评价指标，但低维 factorized law 不依赖特殊 ranking-loss trick；普通 log-value regression 已经给出最高 top-tail capture。",
        "",
        "9. **Parameter-deconfounded structure result**：只看 action mechanics 明显不足；即使移除 eta/cost efficiency，future-loss horizon 与 OD exposure 的 factorized law 仍显著优于 policy-clock baseline，说明城市结构对齐不是参数设定的同义反复。",
        "",
        "10. **Event-level severity-decoupling result**：decision-criticality 与 marginal-value top-tail concentration 强相关，而与 baseline loss magnitude 在当前样本中负相关；大损失事件不一定是最高管理价值事件，中等损失事件也可能因为 top-tail 集中而高度 decision-critical。",
        "",
        "11. **OD message-passing parsimony result**：一跳/两跳 OD message features 本身有信息，但在 scalar OD exposure/structure 已经进入模型后只带来极小 top-tail 增益；加到低维 factorized law 上还略微降低 top-5% capture。因此当前 law 需要真实 OD 空间对齐，但不需要把显式 message passing 作为主模型。",
        "",
        "12. **Parameter-ensemble stability result**：在 11 组 eta/cost/delay 扰动下，只用 delay、future horizon、OD exposure 和 intervention type 的 base-trained parameter-light law 仍保持较高 top-tail capture；含绝对 eta/cost 的 ridge surrogate 对参数尺度外推更敏感，说明后续参数敏感 surrogate 需要 scenario augmentation。",
        "",
        "13. **Parameter-ensemble LP closure result**：在 4 个代表事件、5 类参数扰动、20 个重新求解的完整 LP 场景中，residual finite greedy 平均捕获 0.9662 的 scenario-specific LP gain，显著高于 static small-signal greedy 的 0.7451；说明 residual law 不只是 first-order ranking，在有限预算和参数扰动下仍接近完整优化上界。",
        "",
        "14. **Temporal robustness result**：在 within-city chronological holdout 中，按每个城市的事件发生顺序留出 early/middle/late 或 early/late 时，compact factorized law 仍保持较高 top-tail capture；calendar-year holdout 由于和城市样本组成混杂，只能作为审计而不能单独声称干净时间外推。",
        "",
        "15. **Neural parsimony and leakage audit**: under strict leave-city top-tail evaluation, a lightweight MLP does not improve on ridge; random-event performance is slightly easier, city identity adds only a small top-tail gain, and event identity inflates token-random correlation. The scientific law should therefore be evaluated with strict city/event splits rather than token-level memorization checks.",
        "",
        "16. **Fine budget-leverage law**: a 10-point budget sweep does not support an interior peak in absolute decision leverage. Instead, absolute law-versus-random replay leverage grows to the largest tested budget, while law/random ratio and leverage per unit budget peak at the smallest tested budget and then decline.",
        "",
        "## 关键指标",
        "",
        f"- fine budget sweep: {metrics['fine_budget_n_scales']} budget scales from {metrics['fine_budget_min_scale']:.2f} to {metrics['fine_budget_max_scale']:.2f}; replay absolute law-random leverage peaks at {metrics['fine_budget_replay_abs_peak_budget']:.2f}; interior peak supported = {metrics['fine_budget_replay_abs_interior_peak_supported']}; replay per-budget leverage peaks at {metrics['fine_budget_replay_per_budget_peak_budget']:.2f} and monotone decreasing = {metrics['fine_budget_replay_per_budget_monotone_decreasing']}; law/random replay ratio peaks at {metrics['fine_budget_replay_ratio_peak_budget']:.2f} and monotone decreasing = {metrics['fine_budget_replay_ratio_monotone_decreasing']}; base-budget replay law-random gain = {metrics['fine_budget_base_replay_gain_leverage_vs_random']:.4f}",
        f"- neural surrogate/leakage audit: leave-city full MLP top-5% capture = {metrics['neural_leave_city_full_mlp_top5_capture']:.4f} vs full ridge = {metrics['neural_leave_city_full_ridge_top5_capture']:.4f}; factorized MLP = {metrics['neural_leave_city_factorized_mlp_top5_capture']:.4f} vs factorized ridge = {metrics['neural_leave_city_factorized_ridge_top5_capture']:.4f}; random-event full MLP = {metrics['neural_random_event_full_mlp_top5_capture']:.4f}; city-ID random-event gain = {metrics['neural_random_event_city_id_minus_no_id_top5']:+.4f}; event-ID token-random Spearman gain = {metrics['neural_token_random_event_id_minus_no_id_spearman']:+.4f}",
        f"- action tokens: {metrics['n_action_tokens']:,}",
        f"- city-event scenarios: {metrics['n_events']}",
        f"- single-action LP validation: small-signal Spearman = {metrics['single_action_small_signal_spearman']:.4f}, finite-area label Spearman = {metrics['single_action_finite_area_spearman']:.4f}",
        f"- leave-one-city-out surrogate: mean Spearman = {metrics['leave_city_mean_spearman']:.4f}, top-5% capture = {metrics['leave_city_mean_top5_capture']:.4f}",
        f"- base finite-budget closure: static greedy / LP gain = {metrics['base_static_fraction_of_lp_gain']:.4f}; residual greedy / LP gain = {metrics['base_residual_fraction_of_lp_gain']:.4f}",
        f"- representative non-base closure: static / scenario LP gain = {metrics['scenario_static_fraction_of_lp_gain']:.4f}; residual / scenario LP gain = {metrics['scenario_residual_fraction_of_lp_gain']:.4f}",
        f"- symbolic activated law top-5% capture = {metrics['symbolic_activated_top5_capture']:.4f}; largest feature ablation drop = {metrics['symbolic_largest_ablation_drop_group']} ({metrics['symbolic_largest_ablation_top5_drop']:.4f})",
        f"- non-obvious action law: false-positive shares are deficit-only {metrics['nonobvious_deficit_false_positive_share']:.1%}, exposure-only {metrics['nonobvious_exposure_false_positive_share']:.1%}, structure-only {metrics['nonobvious_structure_false_positive_share']:.1%}",
        f"- factorized surrogate: deficit-only top-5% capture = {metrics['factorized_deficit_top5_capture']:.4f}; full additive = {metrics['factorized_full_additive_top5_capture']:.4f}; low-dimensional factorized = {metrics['factorized_low_dim_top5_capture']:.4f}",
        f"- interaction ablation: adding OD exposure gives {metrics['factorized_add_od_delta_top5_capture']:+.4f}; adding time/feasibility gives {metrics['factorized_add_time_delta_top5_capture']:+.4f}; unrestricted high-dimensional interactions give {metrics['factorized_add_highdim_interaction_delta_top5_capture']:+.4f}",
        f"- graph structure ablation: no-graph top-5% capture = {metrics['graph_no_graph_top5_capture']:.4f}; observed OD graph = {metrics['graph_observed_full_top5_capture']:.4f}; shuffled OD graph = {metrics['graph_shuffled_full_top5_capture']:.4f}; observed-shuffled gap = {metrics['graph_full_alignment_delta_top5_capture']:+.4f}",
        f"- factorized graph alignment: observed OD = {metrics['graph_factorized_observed_top5_capture']:.4f}; shuffled OD = {metrics['graph_factorized_shuffled_top5_capture']:.4f}; gap = {metrics['graph_factorized_alignment_delta_top5_capture']:+.4f}",
        f"- OD message passing: message-only top-5% capture = {metrics['od_message_message_only_top5_capture']:.4f}; scalar OD = {metrics['od_message_scalar_od_top5_capture']:.4f}; scalar+message = {metrics['od_message_scalar_plus_top5_capture']:.4f}; message-over-scalar delta = {metrics['od_message_message_over_scalar_delta_top5_capture']:+.4f}; factorized+message delta = {metrics['od_message_message_over_factorized_delta_top5_capture']:+.4f}",
        f"- parameter ensemble stability: {metrics['parameter_ensemble_n_scenarios']} eta/cost/delay scenarios; base-trained parameter-light factorized mean/min top-5% capture = {metrics['parameter_ensemble_base_transfer_light_mean_top5_capture']:.4f}/{metrics['parameter_ensemble_base_transfer_light_min_top5_capture']:.4f}; full factorized mean/min = {metrics['parameter_ensemble_base_transfer_full_mean_top5_capture']:.4f}/{metrics['parameter_ensemble_base_transfer_full_min_top5_capture']:.4f}; centered-efficiency mean/min = {metrics['parameter_ensemble_base_transfer_centered_mean_top5_capture']:.4f}/{metrics['parameter_ensemble_base_transfer_centered_min_top5_capture']:.4f}; weakest full scenario = {metrics['parameter_ensemble_base_transfer_full_worst_scenario']}",
        f"- parameter LP closure: {metrics['parameter_lp_n_selected_events']} events x {metrics['parameter_lp_n_parameter_scenarios']} parameter scenarios = {metrics['parameter_lp_n_successful_lp_scenarios']} successful LP closures; residual / scenario LP gain = {metrics['parameter_lp_mean_residual_fraction_of_scenario_lp_gain']:.4f} mean and {metrics['parameter_lp_median_residual_fraction_of_scenario_lp_gain']:.4f} median; static = {metrics['parameter_lp_mean_static_fraction_of_scenario_lp_gain']:.4f}; residual-minus-static = {metrics['parameter_lp_mean_residual_minus_static']:+.4f}; weakest residual scenario = {metrics['parameter_lp_worst_residual_parameter_scenario']} at {metrics['parameter_lp_worst_residual_mean_fraction_of_scenario_lp_gain']:.4f}",
        f"- event-regime generalization: {metrics['regime_factorized_n_splits']} held-out regimes; factorized mean top-5% capture = {metrics['regime_factorized_mean_top5_capture']:.4f}; worst = {metrics['regime_factorized_hardest_split_family']} / {metrics['regime_factorized_hardest_heldout']} at {metrics['regime_factorized_min_top5_capture']:.4f}; full additive mean = {metrics['regime_full_mean_top5_capture']:.4f}",
        f"- temporal generalization: {metrics['temporal_main_n_splits']} within-city chronological splits; factorized mean/min top-5% capture = {metrics['temporal_factorized_main_top5_capture']:.4f}/{metrics['temporal_factorized_main_min_top5_capture']:.4f}; full additive mean/min = {metrics['temporal_full_main_top5_capture']:.4f}/{metrics['temporal_full_main_min_top5_capture']:.4f}; hardest factorized split = {metrics['temporal_factorized_main_hardest_split']} / {metrics['temporal_factorized_main_hardest_heldout']}; year audit note: {metrics['temporal_year_holdout_design_note']}",
        f"- training-objective ablation: factorized raw log-value capture = {metrics['objective_factorized_raw_top5_capture']:.4f}; best objective = {metrics['objective_factorized_best_objective']} at {metrics['objective_factorized_best_top5_capture']:.4f}; top-tail weighted = {metrics['objective_factorized_top_tail_weighted_top5_capture']:.4f}; rank-percentile = {metrics['objective_factorized_rank_top5_capture']:.4f}",
        f"- training-objective ablation: full additive best objective = {metrics['objective_full_best_objective']} at {metrics['objective_full_best_top5_capture']:.4f}, improvement over raw = {metrics['objective_full_best_minus_raw_top5_capture']:+.4f}",
        f"- parameter-deconfounded law: policy-clock only top-5% capture = {metrics['parameter_policy_clock_top5_capture']:.4f}; clock+efficiency = {metrics['parameter_clock_plus_efficiency_top5_capture']:.4f}; +OD exposure = {metrics['parameter_add_od_exposure_top5_capture']:.4f}; parameter-light factorized = {metrics['parameter_light_factorized_top5_capture']:.4f}",
        f"- parameter-deconfounded increments: adding eta/cost gives {metrics['parameter_add_efficiency_delta_top5_capture']:+.4f}; adding OD exposure gives {metrics['parameter_add_od_delta_top5_capture']:+.4f}; parameter-light factorized over clock gives {metrics['parameter_light_over_clock_delta_top5_capture']:+.4f}",
        f"- fixed-channel first-order diagnostic: {metrics['parameter_channel_n_groups']} event-time-intervention channels; efficiency-only top-10% capture = {metrics['parameter_channel_efficiency_top10_capture']:.4f}; no-eta horizon--OD activation = {metrics['parameter_channel_light_activation_top10_capture']:.4f}; full activation = {metrics['parameter_channel_full_activation_top10_capture']:.4f}",
        f"- event-level severity decoupling: decision-criticality vs loss Spearman = {metrics['event_decision_v21_decision_vs_loss_spearman']:.4f}; vs top-5% value share = {metrics['event_decision_v21_decision_vs_top5_spearman']:.4f}; vs marginal-value gini = {metrics['event_decision_v21_decision_vs_gini_spearman']:.4f}",
        f"- event-level counterexamples: high-loss/low-decision events = {metrics['event_decision_v21_high_loss_low_decision_count']}; moderate-loss/high-decision events = {metrics['event_decision_v21_moderate_loss_high_decision_count']}; high-rain/low-decision events = {metrics['event_decision_v21_high_rain_low_decision_count']}",
        f"- event-footprint boundary: top-5% value-share between-city variance share = {metrics['event_decision_v21_top5_between_city_share']:.4f}; gini between-city share = {metrics['event_decision_v21_gini_between_city_share']:.4f}; severity-only leave-city decision Spearman = {metrics['event_decision_v21_severity_only_loco_spearman']:.4f}; top-tail model = {metrics['event_decision_v21_top_tail_loco_spearman']:.4f}",
        f"- early decision-criticality: best Spearman = {metrics['early_decision_best_spearman']:.4f} at {metrics['early_decision_best_window']}h using {metrics['early_decision_best_feature_group']}; 2h all-early Spearman = {metrics['early_decision_2h_all_spearman']:.4f}",
        "",
        "## Evidence Ladder",
        "",
        table_to_markdown(evidence),
        "",
        "## Policy Closure",
        "",
        table_to_markdown(closure),
        "",
        "## City Closure",
        "",
        table_to_markdown(city_closure),
        "",
        "## Top Decision-Critical Events",
        "",
        table_to_markdown(decision_examples),
        "",
        "## Event-Level Correlations",
        "",
        table_to_markdown(top_tail_correlations),
        "",
        "## 当前边界与下一步",
        "",
        table_to_markdown(limitations),
        "",
        "## 论文写作含义",
        "",
        "现在 learning/law 部分可以写成一条更完整的证据链：优化模型产生 action-value field；single-action LP 验证 marginal label；cross-city surrogate、factorized surrogate 和 symbolic extraction 说明低维 activated law 可解释；residual greedy 说明有限预算需要动态重评分；event top-tail 说明 decision-criticality 不是 disruption magnitude；V15 给出反直觉证据；V17 说明 OD graph 的空间对齐本身有实证价值；V18 说明低维 law 在不同事件 regime 留出时仍能保持较高 top-tail capture；V19 说明低维 law 不是由特殊 ranking objective trick 造出来的；V25 说明同一城市内按事件时间顺序留出时 compact law 仍保持稳定；V26 说明轻量 neural surrogate 没有在严格 leave-city top-tail 上超过 ridge，并且 token-level identity split 会夸大 apparent fit；V27 将预算规律从三点扫描推进到 10 点扫描，支持 scale-dependent diminishing leverage 而不是中等预算绝对峰值。论文中仍需谨慎表述：当前 graph 证据是 OD-dependency graph 的 observed-vs-shuffled ablation，还不是完整 road-adjacency graph 或 GNN closure；calendar-year holdout 与 city composition 混杂，因此 clean leave-time-period-out 仍需要同城跨多年数据。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return
    lines = [
        "# Recoverability Law Synthesis V16",
        "",
        "## 这一版做了什么",
        "",
        "V16 把 factorized action-value surrogate 和 interaction ablation 接入 learning/law synthesis：在 V15 的非显然 action-law 诊断基础上，进一步检验恢复价值是否需要任意高维交互，还是可以由低维 activated components 稳定解释。",
        "",
        "## 当前可写入论文的 law",
        "",
        "1. **Small-signal activated recovery law**：第一小段资源的边际价值由 future recoverable horizon、OD exposure、intervention feasibility 和 efficiency 共同激活。",
        "",
        "2. **Residual finite-budget allocation law**：完整预算下，价值必须在每轮投放后按 residual state、remaining budget 和 remaining time 重新评分，以避免饱和和重叠效应。",
        "",
        "3. **Top-tail decision-criticality law**：事件是否 decision-critical 不只取决于 observed loss，而取决于 recoverable value 是否集中在少数高价值 action 上。",
        "",
        "4. **Non-obvious activation law**：高损失、高流量或高结构中心性都不是充分条件；静态城市结构只有被事件中的未来损失、OD 暴露、响应窗口和资源效率同时激活，才转化为 recovery value。",
        "",
        "5. **Factorized parsimony result**：跨城市 action-value surrogate 的最大增益来自 OD exposure 和 time/feasibility activation；不受约束的高维 interaction terms 没有提高 full model，支持低维可解释 law。",
        "",
        "## 关键指标",
        "",
        f"- action tokens: {metrics['n_action_tokens']:,}",
        f"- city-event scenarios: {metrics['n_events']}",
        f"- single-action LP validation: small-signal Spearman = {metrics['single_action_small_signal_spearman']:.4f}, finite-area label Spearman = {metrics['single_action_finite_area_spearman']:.4f}",
        f"- leave-one-city-out surrogate: mean Spearman = {metrics['leave_city_mean_spearman']:.4f}, top-5% capture = {metrics['leave_city_mean_top5_capture']:.4f}",
        f"- base finite-budget closure: static greedy / LP gain = {metrics['base_static_fraction_of_lp_gain']:.4f}; residual greedy / LP gain = {metrics['base_residual_fraction_of_lp_gain']:.4f}",
        f"- representative non-base closure: static / scenario LP gain = {metrics['scenario_static_fraction_of_lp_gain']:.4f}; residual / scenario LP gain = {metrics['scenario_residual_fraction_of_lp_gain']:.4f}",
        f"- symbolic activated law top-5% capture = {metrics['symbolic_activated_top5_capture']:.4f}; largest feature ablation drop = {metrics['symbolic_largest_ablation_drop_group']} ({metrics['symbolic_largest_ablation_top5_drop']:.4f})",
        f"- budget phase: interior absolute-leverage peak supported = {metrics['budget_abs_random_interior_peak_supported']}; residual-vs-static per-cost leverage falls from {metrics['budget_low_residual_static_per_cost']:.4f} to {metrics['budget_high_residual_static_per_cost']:.4f}",
        f"- early decision-criticality: best Spearman = {metrics['early_decision_best_spearman']:.4f} at {metrics['early_decision_best_window']}h using {metrics['early_decision_best_feature_group']}; 2h all-early Spearman = {metrics['early_decision_2h_all_spearman']:.4f}",
        f"- non-obvious action law: false-positive shares are deficit-only {metrics['nonobvious_deficit_false_positive_share']:.1%}, exposure-only {metrics['nonobvious_exposure_false_positive_share']:.1%}, structure-only {metrics['nonobvious_structure_false_positive_share']:.1%}",
        f"- non-obvious action law: target top-5% actions hidden from every simple top-5% rank = {metrics['nonobvious_hidden_from_simple_top5_share']:.1%}; target top-5% below structure-only top-20% = {metrics['nonobvious_target_top5_low_structure_top20_share']:.1%}",
        f"- factorized surrogate: deficit-only top-5% capture = {metrics['factorized_deficit_top5_capture']:.4f}; full additive = {metrics['factorized_full_additive_top5_capture']:.4f}; low-dimensional factorized = {metrics['factorized_low_dim_top5_capture']:.4f}",
        f"- interaction ablation: adding OD exposure gives {metrics['factorized_add_od_delta_top5_capture']:+.4f}; adding time/feasibility gives {metrics['factorized_add_time_delta_top5_capture']:+.4f}; unrestricted high-dimensional interactions give {metrics['factorized_add_highdim_interaction_delta_top5_capture']:+.4f}",
        "",
        "## Evidence Ladder",
        "",
        table_to_markdown(evidence),
        "",
        "## Policy Closure",
        "",
        table_to_markdown(closure),
        "",
        "## City Closure",
        "",
        table_to_markdown(city_closure),
        "",
        "## Top Decision-Critical Events",
        "",
        table_to_markdown(decision_examples),
        "",
        "## Event-Level Correlations",
        "",
        table_to_markdown(top_tail_correlations),
        "",
        "## 当前边界与下一步",
        "",
        table_to_markdown(limitations),
        "",
        "## 论文写作含义",
        "",
        "现在 learning/law 部分可以写成一条完整的证据链：优化模型产生 action-value field；single-action LP 验证 marginal label；cross-city surrogate、V16 factorized surrogate 和 symbolic extraction 说明低维结构可解释；residual greedy 说明有限预算需要动态重评分；event top-tail 说明 decision-criticality 不是 disruption magnitude；V15 进一步给出反直觉证据，说明城市结构不是静态优先级，而是需要被事件和干预条件激活的 latent leverage。V16 的负结果同样重要：高维显式交互项没有稳定提升留城表现，因此当前最稳妥的 law 是低维 activated structure，而不是任意复杂 surrogate。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return
    lines = [
        "# Recoverability Law Synthesis V14",
        "",
        "## 这版做了什么",
        "",
        "V14 把 V5-V13 的 learning/law 证据链和新的 early predictability analysis 合并成论文可用的 synthesis：从 action-level marginal value，到 finite-budget residual law，再到 event-level top-tail decision-criticality，并进一步给出公式复杂度、跨城市泛化、feature-group ablation、预算相图和早期可识别性证据。所有数字都从已有结果表重新读取生成。",
        "",
        "## 三条当前可写入论文的 law",
        "",
        "1. **Small-signal activated recovery law**：第一小段资源的边际价值主要由 active future horizon、OD exposure 或 destination importance、intervention efficiency per cost 共同决定，并按 passive event loss 归一化。它回答“第一单位资源投向哪里最值”。",
        "",
        "2. **Residual finite-budget allocation law**：完整预算下，价值必须写成 `value(segment | residual state, remaining budget, remaining time)`。每轮投放后要重新计算剩余 `b/rC/rS/ell`，用 `min(segment_effect_decay, residual_loss)` 截断后续边际收益。它回答“整组资源如何避免局部饱和”。",
        "",
        "3. **Top-tail decision-criticality law**：事件是否 decision-critical 不只取决于 observed loss 大小，而取决于 recoverable value 是否集中在少数高价值 action 上。高 recoverable fraction 与高 top-tail concentration 共同定义了管理决策的杠杆。",
        "",
        "## 关键指标",
        "",
        f"- action tokens: {metrics['n_action_tokens']:,}",
        f"- city-event scenarios: {metrics['n_events']}",
        f"- single-action LP validation: small-signal Spearman = {metrics['single_action_small_signal_spearman']:.4f}, median LP/label ratio = {metrics['single_action_small_signal_median_lp_to_label_ratio']:.4f}",
        f"- finite-area label Spearman = {metrics['single_action_finite_area_spearman']:.4f}",
        f"- leave-one-city-out mean Spearman = {metrics['leave_city_mean_spearman']:.4f}, top-5% capture = {metrics['leave_city_mean_top5_capture']:.4f}",
        f"- base static greedy / LP gain = {metrics['base_static_fraction_of_lp_gain']:.4f}",
        f"- base residual greedy / LP gain = {metrics['base_residual_fraction_of_lp_gain']:.4f}",
        f"- representative non-base static / scenario LP gain = {metrics['scenario_static_fraction_of_lp_gain']:.4f}",
        f"- representative non-base residual / scenario LP gain = {metrics['scenario_residual_fraction_of_lp_gain']:.4f}",
        f"- perturbed optimum residual frequency-mass capture = {metrics['perturbed_residual_frequency_mass']:.4f} versus static = {metrics['perturbed_static_frequency_mass']:.4f}",
        f"- symbolic activated law top-5% capture = {metrics['symbolic_activated_top5_capture']:.4f}; minimal log-law top-5% capture = {metrics['symbolic_minimal_log_top5_capture']:.4f}",
        f"- largest symbolic ablation drop = {metrics['symbolic_largest_ablation_drop_group']} ({metrics['symbolic_largest_ablation_top5_drop']:.4f} top-5 capture)",
        f"- budget phase: absolute law-vs-random leverage peaks at {metrics['budget_abs_random_peak_budget']}; interior peak supported = {metrics['budget_abs_random_interior_peak_supported']}",
        f"- budget phase: residual-vs-static per-cost leverage = {metrics['budget_low_residual_static_per_cost']:.4f} / {metrics['budget_base_residual_static_per_cost']:.4f} / {metrics['budget_high_residual_static_per_cost']:.4f} for low/base/high",
        f"- early decision-criticality: best leave-city Spearman = {metrics['early_decision_best_spearman']:.4f} at {metrics['early_decision_best_window']}h using {metrics['early_decision_best_feature_group']}; 2h all-early Spearman = {metrics['early_decision_2h_all_spearman']:.4f}",
        f"- early signal decomposition at 1h: static city = {metrics['early_decision_static_1h_spearman']:.4f}, speed = {metrics['early_decision_speed_1h_spearman']:.4f}, rain-only = {metrics['early_decision_rain_1h_spearman']:.4f}",
        f"- event mean top-5% value share = {metrics['event_mean_top5_value_share']:.4f}; marginal-value Gini = {metrics['event_mean_marginal_value_gini']:.4f}",
        "",
        "## Evidence Ladder",
        "",
        table_to_markdown(evidence),
        "",
        "## Policy Closure",
        "",
        table_to_markdown(closure),
        "",
        "## City Closure",
        "",
        table_to_markdown(city_closure),
        "",
        "## Top Decision-Critical Events",
        "",
        table_to_markdown(decision_examples),
        "",
        "## Event-Level Correlations",
        "",
        table_to_markdown(top_tail_correlations),
        "",
        "## 当前边界与下一步",
        "",
        table_to_markdown(limitations),
        "",
        "## 论文写作含义",
        "",
        "现在可以把 learning/law 部分从“未来要做 law extraction”改成“已经得到一个可复现实证链条”：action-level 的 activated marginal law、finite-budget 的 residual allocation law、event-level 的 top-tail decision-criticality law，以及 V12 的 formula extractor/structure decoupler。V13 修正了一个预期命题：当前 low/base/high 三点预算扫描不支持“中等预算绝对 decision leverage 最高”，而支持“绝对杠杆随预算增加、单位预算杠杆递减”。V14 进一步说明，decision-criticality 有一定早期可识别性，但主要来自静态城市结构和早期速度异常，不能把 hindsight counterfactual 直接写成在线控制。论文中需要谨慎表述的是：资源效率和 diminishing returns 仍是 recovery-regime 参数；V9/V11 是代表性验证，不是全量非 base 与全量 perturbation closure；完整 graph neural surrogate 仍可作为后续增强，而不是当前主结论的必要条件。",
    ]
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


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def policy_metric(df: pd.DataFrame, policy: str, column: str) -> float:
    if df.empty or "policy" not in df or column not in df:
        return float("nan")
    match = df[df["policy"].astype(str).eq(policy)]
    return safe_float(match[column].mean()) if not match.empty else float("nan")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()

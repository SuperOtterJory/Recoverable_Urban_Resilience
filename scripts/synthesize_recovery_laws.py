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
        "early_metrics": read_csv(results / "early_predictability" / "tables" / "early_predictability_metrics.csv"),
        "early_best": read_csv(results / "early_predictability" / "tables" / "early_best_metrics_by_target.csv"),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
                "current_status": "R/C/S effectiveness, cost, caps, delays, and diminishing returns are recovery-regime assumptions.",
                "implication": "The law is conditional on the specified management regime.",
                "next_step": "Run parameter ensembles or incorporate observed intervention records if available.",
            },
            {
                "item": "surrogate_architecture",
                "current_status": "Current surrogate is normalized ridge/ranking evidence plus V12 symbolic formula extraction, not a full graph neural model.",
                "implication": "The symbolic law is now explicit and reproducible, but the neural structure-extractor stage remains lightweight.",
                "next_step": "Train a factorized graph/action-value surrogate if the paper needs a stronger AI-law-discovery component.",
            },
            {
                "item": "perturbed_optimum_stability",
                "current_status": "Representative perturbation solves are available for 4 events with 3 cost/effectiveness perturbations each.",
                "implication": "The perturbation evidence supports stable value principles, but not yet full-sample action-list stability.",
                "next_step": "Increase perturbation count and city-event coverage if action stability becomes a central claim.",
            },
            {
                "item": "budget_phase_coverage",
                "current_status": "Budget-leverage phase analysis currently uses low/base/high budget scales from existing policy replay and proxy tables.",
                "implication": "The current evidence rejects an interior peak over these three scales, but a finer budget sweep would be needed to rule out a narrower nonmonotonic peak.",
                "next_step": "Run additional budget scales or scenario-specific LP closures if budget phase shape becomes a central contribution.",
            },
            {
                "item": "online_predictability_scope",
                "current_status": "Early-window predictability is tested with leave-one-city-out ridge models using 1/2/3/6/12 hour aggregate features.",
                "implication": "Early decision-criticality signals are supplementary and should not be framed as a full online control policy.",
                "next_step": "Use rolling operational forecasts or causal nowcasting data before making real-time deployment claims.",
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

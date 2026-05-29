"""Stress-test recovery laws across intervention-parameter ensembles.

The high-level learning plan asks whether recovered laws survive changes in
intervention effectiveness, cost, delay, and optimization parameterization. V20
showed that OD exposure and future loss are not reducible to action mechanics
inside the base regime. This script adds a complementary V23 test: perturb the
management-regime parameters on the existing action-token field and measure
whether compact structural laws keep their top-tail value capture.

The analysis is intentionally first-order and fast. It does not re-solve the
full LP for every parameter ensemble. Instead, it recomputes the small-signal
action-value target under each perturbed eta/cost/delay scenario:

    value = feasible(delay) * small_signal_effect_value * eta_per_cost

This is the right scope for testing the local action law. Full finite-budget
policy closure under parameter ensembles remains a separate LP validation task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_factorized_action_surrogate import (
    DEFICIT_FEATURES,
    EVENT_CONTEXT_FEATURES,
    EVENT_KEYS,
    EXPOSURE_FEATURES,
    FACTORIZED_BASE,
    STRUCTURE_FEATURES,
    SUBSTITUTION_FEATURES,
    TIME_FEASIBILITY_FEATURES,
    event_ndcg,
    event_precision,
    event_top_capture,
    fit_ridge,
    predict_ridge,
    prepare_tokens,
)
from recoverable_resilience.paths import find_repo_root


HORIZON = 12
RIDGE_ALPHA = 2.0
TOP_FRACS = (0.01, 0.05, 0.10)
EPS = 1e-12
INTERVENTIONS = ("R", "C", "S")

POLICY_CLOCK_FEATURES = [
    "time_remaining_frac",
    "delay_feasible",
    "delay_fraction",
    "intervention_R",
    "intervention_C",
    "intervention_S",
    "budget_fraction_of_baseline",
]

MODEL_SPECS = [
    {
        "model_id": "E1_parameter_light_factorized",
        "family": "factorized",
        "description": "delay, future horizon, OD exposure, and intervention type; no eta/cost",
        "features": [
            "delay_feasible",
            "log_active_weighted_horizon",
            "log_law_exposure",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
    },
    {
        "model_id": "E2_full_factorized",
        "family": "factorized",
        "description": "compact activated law with eta/cost",
        "features": FACTORIZED_BASE,
    },
    {
        "model_id": "E3_centered_efficiency_factorized",
        "family": "factorized",
        "description": "compact law with event-centered eta/cost for scale-invariant transfer",
        "features": [
            "delay_feasible",
            "log_active_weighted_horizon",
            "log_law_exposure",
            "log_eta_per_cost_event_centered",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
    },
]

SCORE_SPECS = [
    {
        "score_id": "S0_efficiency_only",
        "description": "delay feasibility times eta/cost only",
        "score_col": "ensemble_efficiency_score",
    },
    {
        "score_id": "S1_deficit_only",
        "description": "deficit-only rank",
        "score_col": "deficit_only_score",
    },
    {
        "score_id": "S2_exposure_only",
        "description": "OD exposure-only rank",
        "score_col": "exposure_only_score",
    },
    {
        "score_id": "S3_structure_only",
        "description": "static structure-only score",
        "score_col": "structure_only_score",
    },
    {
        "score_id": "S4_activation_no_efficiency",
        "description": "delay, future horizon, and OD exposure without eta/cost",
        "score_col": "ensemble_activation_no_efficiency",
    },
    {
        "score_id": "S5_full_activation",
        "description": "full activated bottleneck law under scenario eta/cost",
        "score_col": "activated_bottleneck_score",
    },
]

SCENARIOS = [
    {
        "parameter_scenario": "base",
        "description": "original calibrated eta/cost/delay",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "low_efficiency_all",
        "description": "all intervention effectiveness reduced by 25%",
        "eta_scale": {"R": 0.75, "C": 0.75, "S": 0.75},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "high_efficiency_all",
        "description": "all intervention effectiveness increased by 25%",
        "eta_scale": {"R": 1.25, "C": 1.25, "S": 1.25},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "expensive_all",
        "description": "all intervention costs increased by 25%",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.25, "C": 1.25, "S": 1.25},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "cheap_all",
        "description": "all intervention costs reduced by 25%",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 0.75, "C": 0.75, "S": 0.75},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "fast_response",
        "description": "each intervention can start one hour earlier when possible",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": -1, "C": -1, "S": -1},
    },
    {
        "parameter_scenario": "slow_response_2h",
        "description": "each intervention is delayed by two additional hours",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 2, "C": 2, "S": 2},
    },
    {
        "parameter_scenario": "slow_response_4h",
        "description": "each intervention is delayed by four additional hours",
        "eta_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "cost_scale": {"R": 1.0, "C": 1.0, "S": 1.0},
        "delay_add": {"R": 4, "C": 4, "S": 4},
    },
    {
        "parameter_scenario": "R_favored",
        "description": "durable restoration is relatively more efficient and cheaper",
        "eta_scale": {"R": 1.35, "C": 0.90, "S": 0.90},
        "cost_scale": {"R": 0.85, "C": 1.10, "S": 1.10},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "C_favored",
        "description": "temporary capacity is relatively more efficient and cheaper",
        "eta_scale": {"R": 0.90, "C": 1.35, "S": 0.90},
        "cost_scale": {"R": 1.10, "C": 0.85, "S": 1.10},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
    {
        "parameter_scenario": "S_favored",
        "description": "substitution/control is relatively more efficient and cheaper",
        "eta_scale": {"R": 0.90, "C": 0.90, "S": 1.35},
        "cost_scale": {"R": 1.10, "C": 1.10, "S": 0.85},
        "delay_add": {"R": 0, "C": 0, "S": 0},
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "parameter_ensemble_stability"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    base_tokens = load_tokens(root)
    scenario_defs = scenario_definition_table()
    leave_city_rows: list[pd.DataFrame] = []
    event_rows: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    token_summary_rows: list[dict[str, Any]] = []

    base_scenario = apply_parameter_scenario(base_tokens, SCENARIOS[0])
    for scenario in SCENARIOS:
        scenario_name = str(scenario["parameter_scenario"])
        print(f"Running parameter ensemble scenario: {scenario_name}", flush=True)
        tokens = base_scenario if scenario_name == "base" else apply_parameter_scenario(base_tokens, scenario)
        token_summary_rows.append(scenario_token_summary(tokens, scenario))

        transfer_leave, transfer_events = run_base_transfer_models(base_scenario, tokens)
        leave_city_rows.append(transfer_leave)
        event_rows.append(transfer_events)

        score_rows.extend(score_summary_rows(tokens, scenario_name))

    leave_city = pd.concat(leave_city_rows, ignore_index=True)
    event_metrics = pd.concat(event_rows, ignore_index=True)
    model_summary = summarize_models(leave_city, event_metrics)
    score_summary = pd.DataFrame(score_rows)
    token_summary = pd.DataFrame(token_summary_rows)
    diagnostics = build_diagnostics(model_summary, score_summary, token_summary)

    write_table(scenario_defs, table_dir / "parameter_ensemble_scenarios.csv")
    write_table(token_summary, table_dir / "parameter_ensemble_token_summary.csv")
    write_table(model_summary, table_dir / "parameter_ensemble_model_summary.csv")
    write_table(leave_city, table_dir / "parameter_ensemble_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "parameter_ensemble_event_metrics.csv")
    write_table(score_summary, table_dir / "parameter_ensemble_score_summary.csv")
    (table_dir / "parameter_ensemble_stability_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(model_summary, score_summary, token_summary, figure_dir)
    write_report(
        report_dir / "parameter_ensemble_stability_report_zh.md",
        diagnostics,
        model_summary,
        score_summary,
        token_summary,
        scenario_defs,
    )
    print(f"Wrote parameter-ensemble stability analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    tokens = prepare_tokens(pd.read_csv(path))
    tokens["unit"] = tokens["unit"].astype(str)
    tokens["event_id"] = pd.to_numeric(tokens["event_id"], errors="coerce").astype(int)
    return tokens


def scenario_definition_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        row: dict[str, Any] = {
            "parameter_scenario": scenario["parameter_scenario"],
            "description": scenario["description"],
        }
        for intervention in INTERVENTIONS:
            row[f"eta_scale_{intervention}"] = float(scenario["eta_scale"][intervention])
            row[f"cost_scale_{intervention}"] = float(scenario["cost_scale"][intervention])
            row[f"delay_add_{intervention}"] = int(scenario["delay_add"][intervention])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_parameter_scenario(tokens: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    df = tokens.copy()
    scenario_name = str(scenario["parameter_scenario"])
    df["parameter_scenario"] = scenario_name
    df["parameter_description"] = str(scenario["description"])
    df["base_delay_steps"] = np.rint(pd.to_numeric(df["delay_fraction"], errors="coerce").fillna(0.0) * HORIZON).astype(int)

    eta_scale = np.ones(len(df), dtype=float)
    cost_scale = np.ones(len(df), dtype=float)
    delay_add = np.zeros(len(df), dtype=int)
    for intervention in INTERVENTIONS:
        mask = df["intervention"].astype(str).eq(intervention).to_numpy()
        eta_scale[mask] = float(scenario["eta_scale"][intervention])
        cost_scale[mask] = float(scenario["cost_scale"][intervention])
        delay_add[mask] = int(scenario["delay_add"][intervention])
    df["eta_scale"] = eta_scale
    df["cost_scale"] = cost_scale
    df["delay_add_steps"] = delay_add
    df["scenario_delay_steps"] = np.clip(df["base_delay_steps"].to_numpy(dtype=int) + delay_add, 0, HORIZON)
    df["delay_feasible"] = (pd.to_numeric(df["t"], errors="coerce").fillna(0).to_numpy(dtype=int) >= df["scenario_delay_steps"].to_numpy(dtype=int)).astype(float)
    df["delay_fraction"] = df["scenario_delay_steps"] / HORIZON
    df["eta"] = pd.to_numeric(df["eta"], errors="coerce").fillna(0.0) * eta_scale
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0) * cost_scale
    df["eta_per_cost"] = df["eta"] / np.maximum(df["cost"], EPS)
    df["eta_per_cost_rank"] = (
        df.groupby(["city", "event_id", "t", "intervention"], sort=False)["eta_per_cost"]
        .rank(method="average", pct=True)
        .fillna(0.5)
    )
    df["log_eta_per_cost"] = np.log1p(df["eta_per_cost"].clip(lower=0.0))
    df["log_eta_per_cost_event_centered"] = (
        df["log_eta_per_cost"]
        - df.groupby(EVENT_KEYS, sort=False)["log_eta_per_cost"].transform("mean")
    )
    df["eta_per_cost_event_rank"] = (
        df.groupby(EVENT_KEYS, sort=False)["eta_per_cost"]
        .rank(method="average", pct=True)
        .fillna(0.5)
    )
    df["log_law_exposure"] = np.log1p(1_000.0 * df["law_exposure_term"].clip(lower=0.0))
    df["log_active_weighted_horizon"] = np.log1p(df["active_weighted_horizon"].clip(lower=0.0))

    df["target_value"] = (
        df["delay_feasible"]
        * pd.to_numeric(df["small_signal_effect_value"], errors="coerce").fillna(0.0).clip(lower=0.0)
        * df["eta_per_cost"].clip(lower=0.0)
    )
    df["marginal_resource_value"] = df["target_value"]
    df["target_log"] = np.log1p(1_000.0 * df["target_value"])
    df["activated_bottleneck_score"] = (
        df["delay_feasible"]
        * df["active_weighted_horizon"].clip(lower=0.0)
        * df["law_exposure_term"].clip(lower=0.0)
        * df["eta_per_cost"].clip(lower=0.0)
    )
    df["ensemble_activation_no_efficiency"] = (
        df["delay_feasible"]
        * df["active_weighted_horizon"].clip(lower=0.0)
        * df["law_exposure_term"].clip(lower=0.0)
    )
    df["ensemble_efficiency_score"] = df["delay_feasible"] * df["eta_per_cost"].clip(lower=0.0)

    df["log_horizon_x_log_efficiency"] = df["log_active_weighted_horizon"] * df["log_eta_per_cost"]
    df["log_exposure_x_log_efficiency"] = df["log_law_exposure"] * df["log_eta_per_cost"]
    df["log_horizon_x_log_exposure_x_log_efficiency"] = (
        df["log_active_weighted_horizon"] * df["log_law_exposure"] * df["log_eta_per_cost"]
    )
    df["delay_x_log_horizon"] = df["delay_feasible"] * df["log_active_weighted_horizon"]
    return df


def scenario_token_summary(tokens: pd.DataFrame, scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameter_scenario": scenario["parameter_scenario"],
        "description": scenario["description"],
        "n_tokens": int(len(tokens)),
        "n_events": int(tokens[EVENT_KEYS].drop_duplicates().shape[0]),
        "positive_value_share": float((tokens["target_value"] > EPS).mean()),
        "mean_target_value": float(tokens["target_value"].mean()),
        "total_target_value": float(tokens["target_value"].sum()),
        "mean_eta_per_cost": float(tokens["eta_per_cost"].mean()),
        "mean_delay_fraction": float(tokens["delay_fraction"].mean()),
    }


def run_leave_city_models(tokens: pd.DataFrame, *, train_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    scenario_name = str(tokens["parameter_scenario"].iloc[0])
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        validate_features(tokens, features)
        for heldout_city in sorted(tokens["city"].dropna().unique()):
            train = tokens[tokens["city"] != heldout_city].copy()
            test = tokens[tokens["city"] == heldout_city].copy()
            prediction = fit_predict(train, test, features)
            test = test.copy()
            test["predicted_value"] = prediction
            base = model_base_row(spec, train_mode, scenario_name, scenario_name, heldout_city)
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def run_base_transfer_models(base_tokens: pd.DataFrame, scenario_tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    scenario_name = str(scenario_tokens["parameter_scenario"].iloc[0])
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        validate_features(base_tokens, features)
        validate_features(scenario_tokens, features)
        for heldout_city in sorted(scenario_tokens["city"].dropna().unique()):
            train = base_tokens[base_tokens["city"] != heldout_city].copy()
            test = scenario_tokens[scenario_tokens["city"] == heldout_city].copy()
            prediction = fit_predict(train, test, features)
            test = test.copy()
            test["predicted_value"] = prediction
            base = model_base_row(spec, "base_train_transfer", "base", scenario_name, heldout_city)
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
    pred_log = predict_ridge(model, test[features])
    return np.expm1(pred_log) / 1_000.0


def validate_features(tokens: pd.DataFrame, features: list[str]) -> None:
    missing = [feature for feature in features if feature not in tokens]
    if missing:
        raise KeyError(f"Missing features: {missing}")


def model_base_row(
    spec: dict[str, Any],
    train_mode: str,
    train_parameter_scenario: str,
    test_parameter_scenario: str,
    heldout_city: str,
) -> dict[str, Any]:
    return {
        "model_id": spec["model_id"],
        "family": spec["family"],
        "description": spec["description"],
        "train_mode": train_mode,
        "train_parameter_scenario": train_parameter_scenario,
        "test_parameter_scenario": test_parameter_scenario,
        "heldout_city": heldout_city,
        "n_features": len(spec["features"]),
    }


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": corr(frame["target_value"], frame[score_col], method="pearson"),
        "spearman": corr(frame["target_value"], frame[score_col], method="spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in TOP_FRACS:
        label = f"top_{int(frac * 100)}pct"
        captures = [event_top_capture(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        ndcg = [event_ndcg(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        precision = [event_precision(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        out[f"{label}_value_capture"] = safe_nanmean(captures)
        out[f"{label}_ndcg"] = safe_nanmean(ndcg)
        out[f"{label}_precision"] = safe_nanmean(precision)
        out[f"{label}_regret"] = 1.0 - out[f"{label}_value_capture"]
    return out


def event_metric_rows(frame: pd.DataFrame, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        row = {
            **base,
            "city": city,
            "event_id": int(event_id),
            "n_tokens": int(len(group)),
            "total_value": float(group["target_value"].sum()),
            "spearman": corr(group["target_value"], group["predicted_value"], method="spearman"),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            row[f"{label}_value_capture"] = event_top_capture(group, "predicted_value", frac)
            row[f"{label}_ndcg"] = event_ndcg(group, "predicted_value", frac)
            row[f"{label}_precision"] = event_precision(group, "predicted_value", frac)
            row[f"{label}_regret"] = 1.0 - row[f"{label}_value_capture"]
        rows.append(row)
    return rows


def summarize_models(leave_city: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["train_mode", "train_parameter_scenario", "test_parameter_scenario", "model_id"]
    rows: list[dict[str, Any]] = []
    for keys, city_group in leave_city.groupby(group_cols, sort=True):
        train_mode, train_scenario, test_scenario, model_id = keys
        event_group = event_metrics[
            event_metrics["train_mode"].eq(train_mode)
            & event_metrics["train_parameter_scenario"].eq(train_scenario)
            & event_metrics["test_parameter_scenario"].eq(test_scenario)
            & event_metrics["model_id"].eq(model_id)
        ]
        spec = next(item for item in MODEL_SPECS if item["model_id"] == model_id)
        row = {
            "train_mode": train_mode,
            "train_parameter_scenario": train_scenario,
            "test_parameter_scenario": test_scenario,
            "model_id": model_id,
            "family": spec["family"],
            "description": spec["description"],
            "n_features": int(city_group["n_features"].iloc[0]),
            "n_cities": int(city_group["heldout_city"].nunique()),
            "n_events": int(event_group[EVENT_KEYS].drop_duplicates().shape[0]),
            "mean_city_spearman": safe_nanmean(city_group["spearman"].tolist()),
            "mean_event_spearman": safe_nanmean(event_group["spearman"].tolist()),
            "median_event_spearman": safe_nanmedian(event_group["spearman"].tolist()),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            for metric in ["value_capture", "ndcg", "precision", "regret"]:
                row[f"mean_event_{label}_{metric}"] = safe_nanmean(event_group[f"{label}_{metric}"].tolist())
                row[f"median_event_{label}_{metric}"] = safe_nanmedian(event_group[f"{label}_{metric}"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def score_summary_rows(tokens: pd.DataFrame, scenario_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SCORE_SPECS:
        score_col = spec["score_col"]
        row = {
            "parameter_scenario": scenario_name,
            "score_id": spec["score_id"],
            "description": spec["description"],
            "score_col": score_col,
            "n_tokens": int(len(tokens)),
            "n_events": int(tokens[EVENT_KEYS].drop_duplicates().shape[0]),
            "spearman": corr(tokens["target_value"], tokens[score_col], method="spearman"),
            "mean_event_spearman": mean_event_corr(tokens, score_col),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            row[f"mean_event_{label}_value_capture"] = safe_nanmean(
                [event_top_capture(group, score_col, frac) for _, group in tokens.groupby(EVENT_KEYS, sort=False)]
            )
            row[f"mean_event_{label}_ndcg"] = safe_nanmean(
                [event_ndcg(group, score_col, frac) for _, group in tokens.groupby(EVENT_KEYS, sort=False)]
            )
            row[f"mean_event_{label}_precision"] = safe_nanmean(
                [event_precision(group, score_col, frac) for _, group in tokens.groupby(EVENT_KEYS, sort=False)]
            )
        rows.append(row)
    return rows


def mean_event_corr(frame: pd.DataFrame, score_col: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby(EVENT_KEYS, sort=False):
        values.append(corr(group["target_value"], group[score_col], method="spearman"))
    return safe_nanmean(values)


def build_diagnostics(model_summary: pd.DataFrame, score_summary: pd.DataFrame, token_summary: pd.DataFrame) -> dict[str, Any]:
    transfer_factorized = model_summary[
        model_summary["train_mode"].eq("base_train_transfer")
        & model_summary["model_id"].eq("E2_full_factorized")
    ].copy()
    transfer_light = model_summary[
        model_summary["train_mode"].eq("base_train_transfer")
        & model_summary["model_id"].eq("E1_parameter_light_factorized")
    ].copy()
    transfer_centered = model_summary[
        model_summary["train_mode"].eq("base_train_transfer")
        & model_summary["model_id"].eq("E3_centered_efficiency_factorized")
    ].copy()
    score_full = score_summary[score_summary["score_id"].eq("S5_full_activation")].copy()
    score_light = score_summary[score_summary["score_id"].eq("S4_activation_no_efficiency")].copy()

    worst_transfer = (
        transfer_factorized.sort_values("mean_event_top_5pct_value_capture").head(1).iloc[0]
        if not transfer_factorized.empty
        else pd.Series(dtype=float)
    )
    return {
        "n_parameter_scenarios": int(token_summary["parameter_scenario"].nunique()) if not token_summary.empty else 0,
        "base_transfer_factorized_mean_top5_capture": safe_float(transfer_factorized["mean_event_top_5pct_value_capture"].mean()) if not transfer_factorized.empty else np.nan,
        "base_transfer_factorized_min_top5_capture": safe_float(transfer_factorized["mean_event_top_5pct_value_capture"].min()) if not transfer_factorized.empty else np.nan,
        "base_transfer_parameter_light_mean_top5_capture": safe_float(transfer_light["mean_event_top_5pct_value_capture"].mean()) if not transfer_light.empty else np.nan,
        "base_transfer_parameter_light_min_top5_capture": safe_float(transfer_light["mean_event_top_5pct_value_capture"].min()) if not transfer_light.empty else np.nan,
        "base_transfer_centered_factorized_mean_top5_capture": safe_float(transfer_centered["mean_event_top_5pct_value_capture"].mean()) if not transfer_centered.empty else np.nan,
        "base_transfer_centered_factorized_min_top5_capture": safe_float(transfer_centered["mean_event_top_5pct_value_capture"].min()) if not transfer_centered.empty else np.nan,
        "worst_base_transfer_factorized_scenario": str(worst_transfer.get("test_parameter_scenario", "")),
        "worst_base_transfer_factorized_top5_capture": safe_float(worst_transfer.get("mean_event_top_5pct_value_capture")),
        "full_activation_score_mean_top5_capture": safe_float(score_full["mean_event_top_5pct_value_capture"].mean()) if not score_full.empty else np.nan,
        "full_activation_score_min_top5_capture": safe_float(score_full["mean_event_top_5pct_value_capture"].min()) if not score_full.empty else np.nan,
        "light_activation_score_mean_top5_capture": safe_float(score_light["mean_event_top_5pct_value_capture"].mean()) if not score_light.empty else np.nan,
        "light_activation_score_min_top5_capture": safe_float(score_light["mean_event_top_5pct_value_capture"].min()) if not score_light.empty else np.nan,
    }


def make_figures(model_summary: pd.DataFrame, score_summary: pd.DataFrame, token_summary: pd.DataFrame, figure_dir: Path) -> None:
    make_base_transfer_figure(model_summary, figure_dir / "parameter_ensemble_base_transfer.png")
    make_score_stability_figure(score_summary, figure_dir / "parameter_ensemble_score_stability.png")
    make_positive_share_figure(token_summary, figure_dir / "parameter_ensemble_positive_value_share.png")


def make_model_stability_figure(model_summary: pd.DataFrame, path: Path) -> None:
    plot = model_summary[
        model_summary["train_mode"].eq("within_regime")
        & model_summary["model_id"].isin(["E1_parameter_light_factorized", "E2_full_factorized", "E3_full_additive"])
    ].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    scenarios = list(dict.fromkeys(plot["test_parameter_scenario"].tolist()))
    x = np.arange(len(scenarios))
    width = 0.25
    colors = {
        "E1_parameter_light_factorized": "#94a3b8",
        "E2_full_factorized": "#2563eb",
        "E3_full_additive": "#0f766e",
    }
    labels = {
        "E1_parameter_light_factorized": "factorized no eta/cost",
        "E2_full_factorized": "full factorized",
        "E3_full_additive": "full additive",
    }
    for idx, model_id in enumerate(labels):
        sub = plot[plot["model_id"].eq(model_id)].set_index("test_parameter_scenario").reindex(scenarios)
        ax.bar(x + (idx - 1) * width, sub["mean_event_top_5pct_value_capture"], width=width, color=colors[model_id], label=labels[model_id])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Within-regime leave-city top-5% capture")
    ax.set_xticks(x, scenarios, rotation=32, ha="right")
    ax.set_title("Parameter-ensemble stability of compact recovery laws")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_base_transfer_figure(model_summary: pd.DataFrame, path: Path) -> None:
    plot = model_summary[
        model_summary["train_mode"].eq("base_train_transfer")
        & model_summary["model_id"].isin([
            "E1_parameter_light_factorized",
            "E2_full_factorized",
            "E3_centered_efficiency_factorized",
        ])
    ].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for model_id, label, color in [
        ("E1_parameter_light_factorized", "factorized no eta/cost", "#94a3b8"),
        ("E2_full_factorized", "full factorized", "#2563eb"),
        ("E3_centered_efficiency_factorized", "centered eta/cost", "#0f766e"),
    ]:
        sub = plot[plot["model_id"].eq(model_id)]
        ax.plot(
            sub["test_parameter_scenario"],
            sub["mean_event_top_5pct_value_capture"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Base-trained leave-city top-5% capture")
    ax.set_title("Training on base parameters and testing perturbed regimes")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_score_stability_figure(score_summary: pd.DataFrame, path: Path) -> None:
    plot = score_summary[
        score_summary["score_id"].isin(["S4_activation_no_efficiency", "S5_full_activation", "S2_exposure_only", "S1_deficit_only"])
    ].copy()
    if plot.empty:
        return
    pivot = plot.pivot(index="parameter_scenario", columns="score_id", values="mean_event_top_5pct_value_capture")
    labels = {
        "S1_deficit_only": "deficit",
        "S2_exposure_only": "exposure",
        "S4_activation_no_efficiency": "activation no eta",
        "S5_full_activation": "full activation",
    }
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    for score_id, label in labels.items():
        if score_id in pivot:
            ax.plot(pivot.index, pivot[score_id], marker="o", linewidth=1.9, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Formula top-5% value capture")
    ax.set_title("Closed-form action scores under parameter ensembles")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_positive_share_figure(token_summary: pd.DataFrame, path: Path) -> None:
    if token_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(token_summary["parameter_scenario"], token_summary["positive_value_share"], color="#7c3aed")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Positive marginal-value token share")
    ax.set_title("Response delays change which action tokens remain feasible")
    ax.tick_params(axis="x", rotation=32)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    model_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    token_summary: pd.DataFrame,
    scenario_defs: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter-Ensemble Stability V23",
        "",
        "## 本版要回答的问题",
        "",
        "V20 已经说明 action mechanics alone 不足以解释恢复价值。V23 进一步问：如果改变干预效率、成本和响应延迟，低维结构 law 是否仍然稳定？这里不重新求每个参数情景的完整 LP，而是在现有 action-token field 上重算 small-signal target，因此它检验的是局部 action-value law，而不是完整 finite-budget policy closure。",
        "",
        "## 主要结果",
        "",
        f"- 参数情景数：{diagnostics['n_parameter_scenarios']}。",
        f"- base-trained parameter-light factorized mean top-5% capture = {diagnostics['base_transfer_parameter_light_mean_top5_capture']:.4f}，最弱情景 = {diagnostics['base_transfer_parameter_light_min_top5_capture']:.4f}。",
        f"- base-trained full factorized mean top-5% capture = {diagnostics['base_transfer_factorized_mean_top5_capture']:.4f}，最弱情景为 {diagnostics['worst_base_transfer_factorized_scenario']}，capture = {diagnostics['worst_base_transfer_factorized_top5_capture']:.4f}。",
        f"- base-trained centered-efficiency factorized mean top-5% capture = {diagnostics['base_transfer_centered_factorized_mean_top5_capture']:.4f}，最弱情景 = {diagnostics['base_transfer_centered_factorized_min_top5_capture']:.4f}。",
        f"- closed-form full activation score mean top-5% capture = {diagnostics['full_activation_score_mean_top5_capture']:.4f}，最弱情景 = {diagnostics['full_activation_score_min_top5_capture']:.4f}。",
        "",
        "## 解释",
        "",
        "如果 low-dimensional factorized law 在这些参数情景下仍保持高 top-tail capture，说明 law 的主体不是某一组 eta/cost/delay 数值的偶然结果，而是 future recoverable horizon 与 OD exposure 的结构对齐。一个重要细节是：直接使用绝对 log(eta/cost) 的 ridge surrogate 对 uniform scale shift 较敏感；但事件内中心化 eta/cost 也没有自动解决问题，尤其在 channel-favored 情景下会丢失真实的跨干预通道差异。因此后续若要训练参数敏感 surrogate，需要 scenario augmentation，而不是只靠简单归一化。closed-form full activation score 等于 1 是因为这里的 first-order target 正是由同一公式重算出来的，它应作为一致性检查，而不是独立因果验证。",
        "",
        "## Scenario Definitions",
        "",
        table_to_markdown(scenario_defs),
        "",
        "## Token Summary",
        "",
        table_to_markdown(token_summary),
        "",
        "## Model Summary",
        "",
        table_to_markdown(model_summary),
        "",
        "## Formula Score Summary",
        "",
        table_to_markdown(score_summary),
        "",
        "## 当前边界",
        "",
        "这一版仍然是 first-order action-token stability test。它不声称每组参数下的完整 LP allocation 都已经闭合，也没有重新验证 PWL diminishing returns 下的 finite-budget residual policy。若论文要把参数鲁棒性作为核心贡献，下一步应抽取代表性参数集合重新求 scenario-specific LP optimum；但作为 law discovery evidence，V23 已经说明低维 action-value law 对适度参数扰动是稳定的。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def corr(x: Any, y: Any, *, method: str) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(pair["x"].corr(pair["y"], method=method))


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def safe_nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def table_to_markdown(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    compact = frame.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


if __name__ == "__main__":
    main()

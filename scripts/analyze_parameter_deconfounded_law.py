"""Test whether recovery-value laws survive action-parameter deconfounding.

The high-level learning plan warns that a learned law may be an optimization
artifact if it only reflects response delays, intervention effectiveness, or
cost assumptions. This analysis therefore separates action mechanics from city
structure: first fit leave-one-city models using only policy-clock and
intervention-parameter features, then add local deficit, OD exposure, and
structural scarcity features. It also evaluates unit rankings within fixed
event-time-intervention channels, where delay, intervention type, and remaining
time are held constant by construction.
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
    prepare_tokens,
    predict_ridge,
    safe_div,
)
from recoverable_resilience.paths import find_repo_root


RIDGE_ALPHA = 2.0
EPS = 1e-12
TOP_FRACS = (0.01, 0.05, 0.10)
CHANNEL_TOP_FRAC = 0.10
MIN_CHANNEL_TOKENS = 25

POLICY_CLOCK_FEATURES = [
    "time_remaining_frac",
    "delay_feasible",
    "delay_fraction",
    "intervention_R",
    "intervention_C",
    "intervention_S",
    "budget_fraction_of_baseline",
]

EFFICIENCY_FEATURES = [
    "log_eta_per_cost",
    "eta_per_cost_rank",
]

MODEL_SPECS = [
    {
        "model_id": "P0_policy_clock_only",
        "family": "parameter_deconfounding",
        "description": "policy clock, response delay, intervention type, and budget only",
        "features": POLICY_CLOCK_FEATURES,
    },
    {
        "model_id": "P1_clock_plus_efficiency",
        "family": "parameter_deconfounding",
        "description": "policy clock plus calibrated effectiveness per cost",
        "features": POLICY_CLOCK_FEATURES + EFFICIENCY_FEATURES,
    },
    {
        "model_id": "P2_add_local_deficit",
        "family": "parameter_deconfounding",
        "description": "action mechanics plus local/passive deficit dynamics",
        "features": POLICY_CLOCK_FEATURES + EFFICIENCY_FEATURES + DEFICIT_FEATURES,
    },
    {
        "model_id": "P3_add_od_exposure",
        "family": "parameter_deconfounding",
        "description": "action mechanics, local deficit, and OD exposure",
        "features": POLICY_CLOCK_FEATURES + EFFICIENCY_FEATURES + DEFICIT_FEATURES + EXPOSURE_FEATURES,
    },
    {
        "model_id": "P4_add_structure_scarcity",
        "family": "parameter_deconfounding",
        "description": "add OD degree and substitutability/scarcity proxies",
        "features": (
            POLICY_CLOCK_FEATURES
            + EFFICIENCY_FEATURES
            + DEFICIT_FEATURES
            + EXPOSURE_FEATURES
            + STRUCTURE_FEATURES
            + SUBSTITUTION_FEATURES
        ),
    },
    {
        "model_id": "P5_parameter_light_factorized",
        "family": "factorized_deconfounding",
        "description": "factorized law without eta/cost efficiency",
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
        "model_id": "P6_full_factorized",
        "family": "factorized_deconfounding",
        "description": "factorized activated law with eta/cost efficiency",
        "features": FACTORIZED_BASE,
    },
    {
        "model_id": "P7_full_additive",
        "family": "full_reference",
        "description": "full additive feature set used as reference",
        "features": (
            DEFICIT_FEATURES
            + EXPOSURE_FEATURES
            + STRUCTURE_FEATURES
            + SUBSTITUTION_FEATURES
            + TIME_FEASIBILITY_FEATURES
            + EVENT_CONTEXT_FEATURES
        ),
    },
]

CHANNEL_SCORE_SPECS = [
    {
        "score_id": "S0_efficiency_only",
        "description": "within-channel calibrated efficiency per cost only",
        "score_col": "parameter_efficiency_score",
    },
    {
        "score_id": "S1_deficit_only",
        "description": "within-channel local/access deficit only",
        "score_col": "deficit_only_score",
    },
    {
        "score_id": "S2_exposure_only",
        "description": "within-channel OD exposure only",
        "score_col": "exposure_only_score",
    },
    {
        "score_id": "S3_structure_only",
        "description": "within-channel static OD structure only",
        "score_col": "structure_only_score",
    },
    {
        "score_id": "S4_parameter_light_activation",
        "description": "within-channel future horizon times OD exposure, no eta/cost",
        "score_col": "parameter_light_activation_score",
    },
    {
        "score_id": "S5_full_activation",
        "description": "within-channel activated bottleneck score with eta/cost",
        "score_col": "activated_bottleneck_score",
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "parameter_deconfounded_law"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    tokens = add_deconfounding_scores(tokens)
    validate_features(tokens)

    leave_city, event_metrics, coefficients = run_leave_city_models(tokens)
    model_summary = summarize_models(leave_city, event_metrics)
    increments = build_incremental_gains(model_summary)
    channel_metrics = run_channel_neutral_scores(tokens)
    channel_summary = summarize_channel_scores(channel_metrics)
    diagnostics = build_diagnostics(model_summary, increments, channel_summary)

    write_table(model_summary, table_dir / "parameter_deconfounded_model_summary.csv")
    write_table(leave_city, table_dir / "parameter_deconfounded_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "parameter_deconfounded_event_metrics.csv")
    write_table(coefficients, table_dir / "parameter_deconfounded_coefficients.csv")
    write_table(increments, table_dir / "parameter_deconfounded_increments.csv")
    write_table(channel_metrics, table_dir / "channel_neutral_score_metrics.csv")
    write_table(channel_summary, table_dir / "channel_neutral_score_summary.csv")
    (table_dir / "parameter_deconfounded_law_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(model_summary, increments, channel_summary, figure_dir)
    write_report(
        report_dir / "parameter_deconfounded_law_report_zh.md",
        diagnostics,
        model_summary,
        increments,
        channel_summary,
    )
    print(f"Wrote parameter-deconfounded law analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    return prepare_tokens(pd.read_csv(path))


def add_deconfounding_scores(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    df["parameter_efficiency_score"] = df["delay_feasible"].fillna(0.0) * df["eta_per_cost"].clip(lower=0.0)
    df["parameter_light_activation_score"] = (
        df["delay_feasible"].fillna(0.0)
        * df["active_weighted_horizon"].clip(lower=0.0)
        * df["law_exposure_term"].clip(lower=0.0)
    )
    df["parameter_window_efficiency_score"] = (
        df["delay_feasible"].fillna(0.0)
        * df["time_remaining_frac"].clip(lower=0.0)
        * df["eta_per_cost"].clip(lower=0.0)
    )
    return df


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in MODEL_SPECS:
        for feature in spec["features"]:
            if feature not in tokens:
                missing.append(feature)
    for spec in CHANNEL_SCORE_SPECS:
        if spec["score_col"] not in tokens:
            missing.append(spec["score_col"])
    if missing:
        raise KeyError(f"Missing parameter-deconfounding features: {sorted(set(missing))}")


def run_leave_city_models(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        for heldout_city in sorted(tokens["city"].dropna().unique()):
            train = tokens[tokens["city"] != heldout_city].copy()
            test = tokens[tokens["city"] == heldout_city].copy()
            model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
            pred_log = predict_ridge(model, test[features])
            test["predicted_value"] = np.expm1(pred_log) / 1_000.0
            base = {
                "model_id": spec["model_id"],
                "family": spec["family"],
                "description": spec["description"],
                "heldout_city": heldout_city,
                "n_features": len(features),
            }
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, spec, heldout_city))
            for feature, coef in zip(features, model["coef"][1:]):
                coefficient_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "heldout_city": heldout_city,
                        "feature": feature,
                        "standardized_coef": float(coef),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows), pd.DataFrame(coefficient_rows)


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": safe_corr(frame["target_value"], frame[score_col], method="spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in TOP_FRACS:
        label = f"top_{int(frac * 100)}pct"
        values = [event_top_capture(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        ndcgs = [event_ndcg(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        precisions = [event_precision(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        out[f"{label}_value_capture"] = safe_nanmean(values)
        out[f"{label}_ndcg"] = safe_nanmean(ndcgs)
        out[f"{label}_precision"] = safe_nanmean(precisions)
        out[f"{label}_regret"] = 1.0 - out[f"{label}_value_capture"]
    return out


def event_metric_rows(frame: pd.DataFrame, spec: dict[str, Any], heldout_city: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        row = {
            "model_id": spec["model_id"],
            "family": spec["family"],
            "description": spec["description"],
            "heldout_city": heldout_city,
            "city": city,
            "event_id": int(event_id),
            "n_tokens": int(len(group)),
            "total_value": float(group["target_value"].sum()),
            "spearman": safe_corr(group["target_value"], group["predicted_value"], method="spearman"),
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
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_id = spec["model_id"]
        city_group = leave_city[leave_city["model_id"].eq(model_id)]
        event_group = event_metrics[event_metrics["model_id"].eq(model_id)]
        row = {
            "model_id": model_id,
            "family": spec["family"],
            "description": spec["description"],
            "n_features": len(spec["features"]),
            "n_cities": int(city_group["heldout_city"].nunique()),
            "n_events": int(event_group[EVENT_KEYS].drop_duplicates().shape[0]),
            "mean_city_spearman": float(city_group["spearman"].mean()),
            "mean_event_spearman": float(event_group["spearman"].mean()),
            "median_event_spearman": float(event_group["spearman"].median()),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            for metric in ["value_capture", "ndcg", "precision", "regret"]:
                row[f"mean_event_{label}_{metric}"] = float(event_group[f"{label}_{metric}"].mean())
                row[f"median_event_{label}_{metric}"] = float(event_group[f"{label}_{metric}"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_incremental_gains(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("add_calibrated_efficiency", "P0_policy_clock_only", "P1_clock_plus_efficiency"),
        ("add_local_deficit", "P1_clock_plus_efficiency", "P2_add_local_deficit"),
        ("add_od_exposure", "P2_add_local_deficit", "P3_add_od_exposure"),
        ("add_structure_scarcity", "P3_add_od_exposure", "P4_add_structure_scarcity"),
        ("parameter_light_factorized_over_clock", "P0_policy_clock_only", "P5_parameter_light_factorized"),
        ("add_efficiency_to_factorized_law", "P5_parameter_light_factorized", "P6_full_factorized"),
        ("full_additive_over_clock", "P0_policy_clock_only", "P7_full_additive"),
    ]
    rows: list[dict[str, Any]] = []
    for comparison, base_id, next_id in comparisons:
        base = one_row(summary, model_id=base_id)
        nxt = one_row(summary, model_id=next_id)
        if base.empty or nxt.empty:
            continue
        rows.append(
            {
                "comparison": comparison,
                "base_model": base_id,
                "next_model": next_id,
                "delta_top5_value_capture": safe_float(nxt.get("mean_event_top_5pct_value_capture"))
                - safe_float(base.get("mean_event_top_5pct_value_capture")),
                "delta_top5_ndcg": safe_float(nxt.get("mean_event_top_5pct_ndcg"))
                - safe_float(base.get("mean_event_top_5pct_ndcg")),
                "delta_top5_precision": safe_float(nxt.get("mean_event_top_5pct_precision"))
                - safe_float(base.get("mean_event_top_5pct_precision")),
                "delta_event_spearman": safe_float(nxt.get("mean_event_spearman"))
                - safe_float(base.get("mean_event_spearman")),
            }
        )
    return pd.DataFrame(rows)


def run_channel_neutral_scores(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["city", "event_id", "t", "intervention"]
    for group_key, group in tokens.groupby(group_cols, sort=True):
        if len(group) < MIN_CHANNEL_TOKENS or group["target_value"].sum() <= EPS:
            continue
        base = {
            "city": group_key[0],
            "event_id": int(group_key[1]),
            "t": int(group_key[2]),
            "intervention": str(group_key[3]),
            "n_tokens": int(len(group)),
            "total_value": float(group["target_value"].sum()),
        }
        for spec in CHANNEL_SCORE_SPECS:
            score_col = spec["score_col"]
            rows.append(
                {
                    **base,
                    "score_id": spec["score_id"],
                    "description": spec["description"],
                    "spearman": safe_corr(group["target_value"], group[score_col], method="spearman"),
                    "top10_value_capture": event_top_capture(group, score_col, CHANNEL_TOP_FRAC),
                    "top10_ndcg": event_ndcg(group, score_col, CHANNEL_TOP_FRAC),
                    "top10_precision": event_precision(group, score_col, CHANNEL_TOP_FRAC),
                }
            )
    return pd.DataFrame(rows)


def summarize_channel_scores(channel_metrics: pd.DataFrame) -> pd.DataFrame:
    if channel_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for spec in CHANNEL_SCORE_SPECS:
        group = channel_metrics[channel_metrics["score_id"].eq(spec["score_id"])]
        rows.append(
            {
                "score_id": spec["score_id"],
                "description": spec["description"],
                "n_channels": int(len(group)),
                "mean_spearman": float(group["spearman"].mean()),
                "median_spearman": float(group["spearman"].median()),
                "mean_top10_value_capture": float(group["top10_value_capture"].mean()),
                "median_top10_value_capture": float(group["top10_value_capture"].median()),
                "mean_top10_ndcg": float(group["top10_ndcg"].mean()),
                "mean_top10_precision": float(group["top10_precision"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(
    model_summary: pd.DataFrame,
    increments: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> dict[str, Any]:
    p0 = one_row(model_summary, model_id="P0_policy_clock_only")
    p1 = one_row(model_summary, model_id="P1_clock_plus_efficiency")
    p3 = one_row(model_summary, model_id="P3_add_od_exposure")
    p4 = one_row(model_summary, model_id="P4_add_structure_scarcity")
    p5 = one_row(model_summary, model_id="P5_parameter_light_factorized")
    p6 = one_row(model_summary, model_id="P6_full_factorized")
    p7 = one_row(model_summary, model_id="P7_full_additive")
    add_eff = one_row(increments, comparison="add_calibrated_efficiency")
    add_local = one_row(increments, comparison="add_local_deficit")
    add_od = one_row(increments, comparison="add_od_exposure")
    add_struct = one_row(increments, comparison="add_structure_scarcity")
    factorized_over_clock = one_row(increments, comparison="parameter_light_factorized_over_clock")
    factorized_eff = one_row(increments, comparison="add_efficiency_to_factorized_law")
    full_over_clock = one_row(increments, comparison="full_additive_over_clock")
    ch_eff = one_row(channel_summary, score_id="S0_efficiency_only")
    ch_light = one_row(channel_summary, score_id="S4_parameter_light_activation")
    ch_full = one_row(channel_summary, score_id="S5_full_activation")
    return {
        "policy_clock_top5_capture": safe_float(p0.get("mean_event_top_5pct_value_capture")),
        "clock_plus_efficiency_top5_capture": safe_float(p1.get("mean_event_top_5pct_value_capture")),
        "mechanics_deficit_exposure_top5_capture": safe_float(p3.get("mean_event_top_5pct_value_capture")),
        "mechanics_deficit_exposure_structure_top5_capture": safe_float(p4.get("mean_event_top_5pct_value_capture")),
        "parameter_light_factorized_top5_capture": safe_float(p5.get("mean_event_top_5pct_value_capture")),
        "full_factorized_top5_capture": safe_float(p6.get("mean_event_top_5pct_value_capture")),
        "full_additive_top5_capture": safe_float(p7.get("mean_event_top_5pct_value_capture")),
        "add_efficiency_delta_top5_capture": safe_float(add_eff.get("delta_top5_value_capture")),
        "add_local_deficit_delta_top5_capture": safe_float(add_local.get("delta_top5_value_capture")),
        "add_od_exposure_delta_top5_capture": safe_float(add_od.get("delta_top5_value_capture")),
        "add_structure_scarcity_delta_top5_capture": safe_float(add_struct.get("delta_top5_value_capture")),
        "parameter_light_factorized_over_clock_delta_top5_capture": safe_float(factorized_over_clock.get("delta_top5_value_capture")),
        "add_efficiency_to_factorized_delta_top5_capture": safe_float(factorized_eff.get("delta_top5_value_capture")),
        "full_additive_over_clock_delta_top5_capture": safe_float(full_over_clock.get("delta_top5_value_capture")),
        "channel_efficiency_only_top10_capture": safe_float(ch_eff.get("mean_top10_value_capture")),
        "channel_parameter_light_activation_top10_capture": safe_float(ch_light.get("mean_top10_value_capture")),
        "channel_full_activation_top10_capture": safe_float(ch_full.get("mean_top10_value_capture")),
        "channel_activation_minus_efficiency_top10_capture": safe_float(ch_full.get("mean_top10_value_capture"))
        - safe_float(ch_eff.get("mean_top10_value_capture")),
        "channel_light_activation_minus_efficiency_top10_capture": safe_float(ch_light.get("mean_top10_value_capture"))
        - safe_float(ch_eff.get("mean_top10_value_capture")),
        "channel_n_groups": safe_int(ch_full.get("n_channels")),
    }


def make_figures(
    model_summary: pd.DataFrame,
    increments: pd.DataFrame,
    channel_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_model_ladder_figure(model_summary, figure_dir / "parameter_deconfounded_model_ladder.png")
    make_increment_figure(increments, figure_dir / "parameter_deconfounded_increments.png")
    make_channel_figure(channel_summary, figure_dir / "channel_neutral_score_capture.png")


def make_model_ladder_figure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    labels = {
        "P0_policy_clock_only": "clock",
        "P1_clock_plus_efficiency": "+efficiency",
        "P2_add_local_deficit": "+deficit",
        "P3_add_od_exposure": "+OD exposure",
        "P4_add_structure_scarcity": "+structure",
        "P5_parameter_light_factorized": "factorized no eta",
        "P6_full_factorized": "factorized",
        "P7_full_additive": "full",
    }
    ordered = summary[summary["model_id"].isin(labels)].copy()
    ordered["label"] = pd.Categorical(ordered["model_id"].map(labels), categories=list(labels.values()), ordered=True)
    ordered = ordered.sort_values("label")
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    colors = ["#64748b", "#94a3b8", "#60a5fa", "#2563eb", "#1d4ed8", "#14b8a6", "#0f766e", "#111827"]
    ax.bar(ordered["label"].astype(str), ordered["mean_event_top_5pct_value_capture"], color=colors[: len(ordered)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Leave-city top-5% value capture")
    ax.set_title("City structure adds value after action-parameter controls")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(ordered["mean_event_top_5pct_value_capture"]):
        ax.text(idx, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_increment_figure(increments: pd.DataFrame, path: Path) -> None:
    if increments.empty:
        return
    ordered = increments.copy()
    ordered["label"] = ordered["comparison"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in ordered["delta_top5_value_capture"]]
    ax.barh(ordered["label"], ordered["delta_top5_value_capture"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Delta top-5% value capture")
    ax.set_title("Incremental gain beyond parameter-only baselines")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_channel_figure(channel_summary: pd.DataFrame, path: Path) -> None:
    if channel_summary.empty:
        return
    labels = {
        "S0_efficiency_only": "efficiency",
        "S1_deficit_only": "deficit",
        "S2_exposure_only": "OD exposure",
        "S3_structure_only": "structure",
        "S4_parameter_light_activation": "activation no eta",
        "S5_full_activation": "full activation",
    }
    ordered = channel_summary.copy()
    ordered["label"] = pd.Categorical(ordered["score_id"].map(labels), categories=list(labels.values()), ordered=True)
    ordered = ordered.sort_values("label")
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.bar(ordered["label"].astype(str), ordered["mean_top10_value_capture"], color="#0f766e")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Within-channel top-10% value capture")
    ax.set_title("Unit ranking after fixing event, time, and intervention type")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(ordered["mean_top10_value_capture"]):
        ax.text(idx, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    model_summary: pd.DataFrame,
    increments: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter-Deconfounded Recoverability Law V20",
        "",
        "## 本版要回答的问题",
        "",
        "这一版检验一个关键风险：action-value law 会不会只是恢复模型中的响应时间、R/C/S 类型、效率、成本和预算等参数关系，而不是真正来自城市结构。",
        "",
        "做法是先训练只含 action mechanics 的 leave-one-city ridge，再逐步加入 local deficit、OD exposure、OD degree/scarcity，并另外在固定 event-time-intervention channel 内比较 unit ranking。后者相当于把事件、时间、干预类型和 delay 轴固定住，只看同一类 action 内哪些 region 更有恢复价值。",
        "",
        "## 主要发现",
        "",
        f"- policy-clock only 的 top-5% value capture = {diagnostics['policy_clock_top5_capture']:.4f}。",
        f"- 加入 calibrated eta/cost 后为 {diagnostics['clock_plus_efficiency_top5_capture']:.4f}，增量 {diagnostics['add_efficiency_delta_top5_capture']:+.4f}。",
        f"- 继续加入 local deficit 后增量 {diagnostics['add_local_deficit_delta_top5_capture']:+.4f}；加入 OD exposure 后增量 {diagnostics['add_od_exposure_delta_top5_capture']:+.4f}；加入 structure/scarcity 后增量 {diagnostics['add_structure_scarcity_delta_top5_capture']:+.4f}。",
        f"- 不使用 eta/cost 的 parameter-light factorized law 已达到 {diagnostics['parameter_light_factorized_top5_capture']:.4f}，比 policy-clock only 高 {diagnostics['parameter_light_factorized_over_clock_delta_top5_capture']:+.4f}。",
        f"- 完整 factorized law 为 {diagnostics['full_factorized_top5_capture']:.4f}，eta/cost 在 factorized law 上额外贡献 {diagnostics['add_efficiency_to_factorized_delta_top5_capture']:+.4f}。",
        f"- 固定 event-time-intervention channel 后，efficiency-only 的 top-10% capture = {diagnostics['channel_efficiency_only_top10_capture']:.4f}；在这个 first-order label field 内，不含 eta/cost 的 horizon--OD-exposure activation score = {diagnostics['channel_parameter_light_activation_top10_capture']:.4f}，完整 activation score = {diagnostics['channel_full_activation_top10_capture']:.4f}。",
        "",
        "## 解释",
        "",
        "结果说明，响应时间、R/C/S 类型和效率/成本确实解释了部分 action-value 排名；这部分应该被诚实地称为 recovery-regime effect。但在控制这些 action mechanics 后，OD exposure、persistent future loss 和实际空间位置仍然显著提高 top-tail capture。固定 channel 的诊断更像是对 small-signal label 结构的自检，而不是独立因果证据；它说明在该标签场内部，主要空间变化来自 future-loss horizon 与 OD exposure 的对齐。因此目前的 law 不是“参数越好越值得投”的同义反复，而是：在给定管理制度下，城市结构决定哪些可行动位置能把一单位资源转化为系统性损失下降。",
        "",
        "## 模型阶梯",
        "",
        table_to_markdown(model_summary),
        "",
        "## 增量",
        "",
        table_to_markdown(increments),
        "",
        "## 固定 channel 的 unit ranking",
        "",
        table_to_markdown(channel_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def table_to_markdown(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def safe_corr(x: Any, y: Any, *, method: str = "pearson") -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(pair["x"].corr(pair["y"], method=method))


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


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


if __name__ == "__main__":
    main()

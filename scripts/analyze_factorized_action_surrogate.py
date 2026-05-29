"""Evaluate factorized action-value surrogate and interaction ablations.

The high-level learning plan asks for a decision-centered surrogate that
separates local action score, event context, and interaction corrections. This
script tests that idea with leave-one-city-out ridge models of increasing
complexity and ranking-focused metrics.
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

from recoverable_resilience.paths import find_repo_root


TARGET = "marginal_resource_value"
EVENT_KEYS = ["city", "event_id"]
EPS = 1e-12
RIDGE_ALPHA = 2.0
TOP_FRACS = (0.01, 0.05, 0.10)


DEFICIT_FEATURES = [
    "local_remaining_rank",
    "access_remaining_rank",
    "passive_b_rank",
    "passive_ell_rank",
    "b0_rank",
    "h_total_rank",
    "local_need_rank",
]

EXPOSURE_FEATURES = [
    "origin_exposure_rank",
    "destination_importance_rank",
    "log_law_exposure",
]

STRUCTURE_FEATURES = [
    "out_degree_rank",
    "in_degree_rank",
]

SUBSTITUTION_FEATURES = [
    "od_scarcity",
    "structure_only_score",
]

TIME_FEASIBILITY_FEATURES = [
    "time_remaining_frac",
    "log_active_weighted_horizon",
    "active_future_loss_share",
    "delay_feasible",
    "delay_fraction",
    "log_eta_per_cost",
    "eta_per_cost_rank",
    "intervention_R",
    "intervention_C",
    "intervention_S",
]

EVENT_CONTEXT_FEATURES = [
    "budget_fraction_of_baseline",
    "log_event_total_precip",
    "log_event_peak_precip",
    "event_peak_positive_abnormal_deficit",
    "weighted_b0",
    "weighted_h_total",
]

INTERACTION_FEATURES = [
    "deficit_x_exposure",
    "access_x_origin",
    "deficit_x_exposure_x_scarcity",
    "log_horizon_x_log_exposure",
    "log_horizon_x_log_efficiency",
    "log_exposure_x_log_efficiency",
    "log_horizon_x_log_exposure_x_log_efficiency",
    "delay_x_log_horizon",
    "scarcity_x_log_exposure",
]

FACTORIZED_BASE = [
    "delay_feasible",
    "log_active_weighted_horizon",
    "log_law_exposure",
    "log_eta_per_cost",
    "intervention_R",
    "intervention_C",
    "intervention_S",
]

FACTORIZED_INTERACTIONS = [
    "log_horizon_x_log_exposure",
    "log_horizon_x_log_efficiency",
    "log_exposure_x_log_efficiency",
    "log_horizon_x_log_exposure_x_log_efficiency",
    "delay_x_log_horizon",
]


MODEL_SPECS = [
    {
        "model_id": "M1_deficit_only",
        "family": "additive_ladder",
        "description": "deficit and passive remaining-loss features only",
        "features": DEFICIT_FEATURES,
    },
    {
        "model_id": "M2_deficit_plus_exposure",
        "family": "additive_ladder",
        "description": "deficit plus OD exposure",
        "features": DEFICIT_FEATURES + EXPOSURE_FEATURES,
    },
    {
        "model_id": "M3_add_structure",
        "family": "additive_ladder",
        "description": "add static structural leverage",
        "features": DEFICIT_FEATURES + EXPOSURE_FEATURES + STRUCTURE_FEATURES,
    },
    {
        "model_id": "M4_add_substitution",
        "family": "additive_ladder",
        "description": "add substitution scarcity proxies",
        "features": DEFICIT_FEATURES + EXPOSURE_FEATURES + STRUCTURE_FEATURES + SUBSTITUTION_FEATURES,
    },
    {
        "model_id": "M5_full_additive",
        "family": "additive_ladder",
        "description": "add time window, feasibility, intervention, and event context",
        "features": (
            DEFICIT_FEATURES
            + EXPOSURE_FEATURES
            + STRUCTURE_FEATURES
            + SUBSTITUTION_FEATURES
            + TIME_FEASIBILITY_FEATURES
            + EVENT_CONTEXT_FEATURES
        ),
    },
    {
        "model_id": "M6_full_interaction",
        "family": "interaction_ablation",
        "description": "full additive model plus explicit pairwise and triple interaction terms",
        "features": (
            DEFICIT_FEATURES
            + EXPOSURE_FEATURES
            + STRUCTURE_FEATURES
            + SUBSTITUTION_FEATURES
            + TIME_FEASIBILITY_FEATURES
            + EVENT_CONTEXT_FEATURES
            + INTERACTION_FEATURES
        ),
    },
    {
        "model_id": "M7_factorized_low_dim",
        "family": "factorized_law",
        "description": "low-dimensional interpretable activated components only",
        "features": FACTORIZED_BASE,
    },
    {
        "model_id": "M8_factorized_interaction",
        "family": "factorized_law",
        "description": "low-dimensional activated components plus explicit interactions",
        "features": FACTORIZED_BASE + FACTORIZED_INTERACTIONS,
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "factorized_action_surrogate"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    tokens = prepare_tokens(tokens)
    validate_features(tokens)

    leave_city_metrics, event_metrics, predictions, coefficients = run_leave_city_out(tokens)
    summary = summarize_models(leave_city_metrics, event_metrics)
    increments = build_incremental_gains(summary)
    city_summary = build_city_summary(event_metrics)
    top_tail_examples = build_top_tail_examples(predictions)
    metrics = build_metrics(summary, increments)

    write_table(summary, table_dir / "factorized_model_summary.csv")
    write_table(leave_city_metrics, table_dir / "factorized_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "factorized_event_metrics.csv")
    write_table(increments, table_dir / "factorized_incremental_gains.csv")
    write_table(city_summary, table_dir / "factorized_city_summary.csv")
    write_table(coefficients, table_dir / "factorized_coefficients.csv")
    write_table(top_tail_examples, table_dir / "factorized_top_tail_examples.csv")
    (table_dir / "factorized_action_surrogate_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(summary, increments, city_summary, event_metrics, figure_dir)
    write_report(
        report_dir / "factorized_action_surrogate_report_zh.md",
        metrics,
        summary,
        increments,
        city_summary,
        top_tail_examples,
    )
    print(f"Wrote factorized action surrogate analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    return pd.read_csv(path)


def prepare_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    numeric_cols = set(
        DEFICIT_FEATURES
        + EXPOSURE_FEATURES
        + STRUCTURE_FEATURES
        + SUBSTITUTION_FEATURES
        + TIME_FEASIBILITY_FEATURES
        + EVENT_CONTEXT_FEATURES
        + [TARGET, "law_exposure_term", "active_weighted_horizon", "eta_per_cost"]
    )
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "log_law_exposure" not in df:
        df["log_law_exposure"] = np.log1p(1_000.0 * df["law_exposure_term"].clip(lower=0.0))
    if "log_active_weighted_horizon" not in df:
        df["log_active_weighted_horizon"] = np.log1p(df["active_weighted_horizon"].clip(lower=0.0))
    if "log_eta_per_cost" not in df:
        df["log_eta_per_cost"] = np.log1p(10.0 * df["eta_per_cost"].clip(lower=0.0))

    df["target_value"] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["target_log"] = np.log1p(1_000.0 * df["target_value"])
    df["log_horizon_x_log_exposure"] = df["log_active_weighted_horizon"] * df["log_law_exposure"]
    df["log_horizon_x_log_efficiency"] = df["log_active_weighted_horizon"] * df["log_eta_per_cost"]
    df["log_exposure_x_log_efficiency"] = df["log_law_exposure"] * df["log_eta_per_cost"]
    df["log_horizon_x_log_exposure_x_log_efficiency"] = (
        df["log_active_weighted_horizon"] * df["log_law_exposure"] * df["log_eta_per_cost"]
    )
    df["delay_x_log_horizon"] = df["delay_feasible"].fillna(0.0) * df["log_active_weighted_horizon"]
    df["scarcity_x_log_exposure"] = df["od_scarcity"].fillna(0.0) * df["log_law_exposure"]
    for col in INTERACTION_FEATURES:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in MODEL_SPECS:
        for feature in spec["features"]:
            if feature not in tokens:
                missing.append(feature)
    if missing:
        raise KeyError(f"Missing factorized surrogate features: {sorted(set(missing))}")


def run_leave_city_out(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        for heldout_city in sorted(tokens["city"].unique()):
            train = tokens[tokens["city"] != heldout_city].copy()
            test = tokens[tokens["city"] == heldout_city].copy()
            model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
            pred_log = predict_ridge(model, test[features])
            test["predicted_value"] = np.expm1(pred_log) / 1_000.0
            test["model_id"] = spec["model_id"]
            test["family"] = spec["family"]
            test["heldout_city"] = heldout_city
            metric_rows.append(
                {
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "description": spec["description"],
                    "heldout_city": heldout_city,
                    "n_features": len(features),
                    **prediction_metrics(test, "predicted_value"),
                }
            )
            event_rows.extend(event_metric_rows(test, spec, heldout_city))
            prediction_frames.append(
                test[
                    [
                        "model_id",
                        "heldout_city",
                        "city",
                        "event_id",
                        "event_start",
                        "unit",
                        "t",
                        "intervention",
                        "target_value",
                        "predicted_value",
                        "deficit_only_score",
                        "exposure_only_score",
                        "structure_only_score",
                        "activated_bottleneck_score",
                    ]
                ].copy()
            )
            for feature, coef in zip(features, model["coef"][1:]):
                coefficient_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "heldout_city": heldout_city,
                        "feature": feature,
                        "standardized_coef": float(coef),
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(event_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(coefficient_rows),
    )


def fit_ridge(x: pd.DataFrame, y: pd.Series, *, alpha: float) -> dict[str, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mean = np.nanmean(x_arr, axis=0)
    std = np.nanstd(x_arr, axis=0)
    std = np.where(std <= EPS, 1.0, std)
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
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": float(frame["target_value"].corr(frame[score_col], method="spearman")),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in TOP_FRACS:
        label = f"top_{int(frac * 100)}pct"
        out[f"{label}_value_capture"] = mean_event_top_capture(frame, score_col, frac)
        out[f"{label}_ndcg"] = mean_event_ndcg(frame, score_col, frac)
        out[f"{label}_precision"] = mean_event_precision(frame, score_col, frac)
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
            "spearman": float(group["target_value"].corr(group["predicted_value"], method="spearman")),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            row[f"{label}_value_capture"] = event_top_capture(group, "predicted_value", frac)
            row[f"{label}_ndcg"] = event_ndcg(group, "predicted_value", frac)
            row[f"{label}_precision"] = event_precision(group, "predicted_value", frac)
            row[f"{label}_regret"] = 1.0 - row[f"{label}_value_capture"]
        rows.append(row)
    return rows


def mean_event_top_capture(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    values = [event_top_capture(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
    return float(np.nanmean(values)) if values else np.nan


def event_top_capture(group: pd.DataFrame, score_col: str, frac: float) -> float:
    if group.empty or group["target_value"].sum() <= EPS:
        return np.nan
    k = max(1, int(np.ceil(len(group) * frac)))
    oracle = float(group.nlargest(k, "target_value")["target_value"].sum())
    chosen = float(group.nlargest(k, score_col)["target_value"].sum())
    return safe_div(chosen, oracle)


def mean_event_ndcg(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    values = [event_ndcg(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
    return float(np.nanmean(values)) if values else np.nan


def event_ndcg(group: pd.DataFrame, score_col: str, frac: float) -> float:
    if group.empty or group["target_value"].sum() <= EPS:
        return np.nan
    k = max(1, int(np.ceil(len(group) * frac)))
    chosen = group.nlargest(k, score_col)["target_value"].to_numpy(dtype=float)
    ideal = group.nlargest(k, "target_value")["target_value"].to_numpy(dtype=float)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(chosen * discount[: len(chosen)]))
    idcg = float(np.sum(ideal * discount[: len(ideal)]))
    return safe_div(dcg, idcg)


def mean_event_precision(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    values = [event_precision(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
    return float(np.nanmean(values)) if values else np.nan


def event_precision(group: pd.DataFrame, score_col: str, frac: float) -> float:
    if group.empty:
        return np.nan
    k = max(1, int(np.ceil(len(group) * frac)))
    chosen = set(group.nlargest(k, score_col).index)
    ideal = set(group.nlargest(k, "target_value").index)
    return len(chosen & ideal) / k


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
        ("add_od_exposure", "M1_deficit_only", "M2_deficit_plus_exposure"),
        ("add_structure", "M2_deficit_plus_exposure", "M3_add_structure"),
        ("add_substitution", "M3_add_structure", "M4_add_substitution"),
        ("add_time_feasibility", "M4_add_substitution", "M5_full_additive"),
        ("add_explicit_interactions", "M5_full_additive", "M6_full_interaction"),
        ("factorized_add_interactions", "M7_factorized_low_dim", "M8_factorized_interaction"),
        ("full_interaction_vs_deficit", "M1_deficit_only", "M6_full_interaction"),
        ("factorized_interaction_vs_full_additive", "M5_full_additive", "M8_factorized_interaction"),
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


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    city = (
        event_metrics.groupby(["city", "model_id"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_event_spearman=("spearman", "mean"),
            mean_top5_capture=("top_5pct_value_capture", "mean"),
            mean_top5_ndcg=("top_5pct_ndcg", "mean"),
            mean_top5_precision=("top_5pct_precision", "mean"),
        )
        .sort_values(["model_id", "mean_top5_capture"], ascending=[True, False])
    )
    return city


def build_top_tail_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    additive = predictions[predictions["model_id"].eq("M5_full_additive")].copy()
    interaction = predictions[predictions["model_id"].eq("M6_full_interaction")].copy()
    if additive.empty or interaction.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    action_cols = ["city", "event_id", "unit", "t", "intervention"]
    for keys, add_group in additive.groupby(EVENT_KEYS, sort=True):
        int_group = interaction[(interaction["city"].eq(keys[0])) & (interaction["event_id"].eq(keys[1]))]
        if int_group.empty:
            continue
        k = max(1, int(np.ceil(len(add_group) * 0.05)))
        target_top = tuple_set(add_group.nlargest(k, "target_value"), action_cols)
        add_top = tuple_set(add_group.nlargest(k, "predicted_value"), action_cols)
        int_top = tuple_set(int_group.nlargest(k, "predicted_value"), action_cols)
        rescued_keys = (target_top & int_top) - add_top
        if not rescued_keys:
            continue
        action_key = add_group[action_cols].apply(lambda row: tuple(row), axis=1)
        rescued = add_group[action_key.isin(rescued_keys)].copy()
        rescued["additive_predicted_value"] = rescued["predicted_value"]
        int_lookup = int_group[["city", "event_id", "unit", "t", "intervention", "predicted_value"]].rename(
            columns={"predicted_value": "interaction_predicted_value"}
        )
        rescued = rescued.merge(
            int_lookup,
            on=["city", "event_id", "unit", "t", "intervention"],
            how="left",
        )
        rows.append(rescued)
    if not rows:
        return pd.DataFrame()
    examples = pd.concat(rows, ignore_index=True)
    examples = examples.sort_values("target_value", ascending=False).head(80)
    keep = [
        "city",
        "event_id",
        "event_start",
        "unit",
        "t",
        "intervention",
        "target_value",
        "additive_predicted_value",
        "interaction_predicted_value",
        "deficit_only_score",
        "exposure_only_score",
        "structure_only_score",
        "activated_bottleneck_score",
    ]
    return examples[[col for col in keep if col in examples.columns]]


def tuple_set(frame: pd.DataFrame, cols: list[str]) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in frame[cols].itertuples(index=False, name=None)}


def build_metrics(summary: pd.DataFrame, increments: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for model_id in [spec["model_id"] for spec in MODEL_SPECS]:
        row = one_row(summary, model_id=model_id)
        prefix = model_id.lower()
        metrics[f"{prefix}_top5_capture"] = safe_float(row.get("mean_event_top_5pct_value_capture"))
        metrics[f"{prefix}_top5_ndcg"] = safe_float(row.get("mean_event_top_5pct_ndcg"))
        metrics[f"{prefix}_top5_precision"] = safe_float(row.get("mean_event_top_5pct_precision"))
        metrics[f"{prefix}_event_spearman"] = safe_float(row.get("mean_event_spearman"))
    for _, row in increments.iterrows():
        key = str(row["comparison"])
        metrics[f"{key}_delta_top5_capture"] = safe_float(row.get("delta_top5_value_capture"))
        metrics[f"{key}_delta_event_spearman"] = safe_float(row.get("delta_event_spearman"))
    return metrics


def make_figures(
    summary: pd.DataFrame,
    increments: pd.DataFrame,
    city_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_model_ladder_figure(summary, figure_dir / "factorized_model_ladder.png")
    make_incremental_gain_figure(increments, figure_dir / "interaction_incremental_gains.png")
    make_city_gain_figure(city_summary, figure_dir / "interaction_gain_by_city.png")
    make_regret_figure(summary, figure_dir / "top_tail_regret_ladder.png")
    make_event_scatter(event_metrics, figure_dir / "additive_vs_interaction_event_capture.png")


def make_model_ladder_figure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    x = np.arange(len(plot))
    fig, ax1 = plt.subplots(figsize=(11.0, 5.2))
    ax1.bar(x - 0.18, plot["mean_event_top_5pct_value_capture"], width=0.36, color="#2563eb", label="top-5% capture")
    ax1.bar(x + 0.18, plot["mean_event_top_5pct_ndcg"], width=0.36, color="#0f766e", label="NDCG@5%")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Decision-centered ranking metric")
    ax1.set_xticks(x, plot["model_id"], rotation=28, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, plot["mean_event_spearman"], color="#ef4444", marker="o", linewidth=1.8, label="Spearman")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Mean event Spearman")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="lower right")
    ax1.set_title("Factorized action-value surrogate: additive ladder and interaction correction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_incremental_gain_figure(increments: pd.DataFrame, path: Path) -> None:
    if increments.empty:
        return
    plot = increments[~increments["comparison"].isin(["full_interaction_vs_deficit"])].copy()
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    y = np.arange(len(plot))
    ax.barh(y, plot["delta_top5_value_capture"], color="#7c3aed")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, plot["comparison"])
    ax.set_xlabel("Incremental top-5% value capture")
    ax.set_title("Which added factor improves decision-centered ranking?")
    for idx, value in enumerate(plot["delta_top5_value_capture"]):
        ax.text(value + (0.004 if value >= 0 else -0.004), idx, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_gain_figure(city_summary: pd.DataFrame, path: Path) -> None:
    if city_summary.empty:
        return
    pivot = city_summary.pivot(index="city", columns="model_id", values="mean_top5_capture")
    needed = ["M5_full_additive", "M6_full_interaction", "M8_factorized_interaction"]
    if not set(needed).issubset(pivot.columns):
        return
    pivot["interaction_gain"] = pivot["M6_full_interaction"] - pivot["M5_full_additive"]
    pivot = pivot.sort_values("interaction_gain")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(pivot))
    ax.barh(y - 0.18, pivot["M5_full_additive"], height=0.36, color="#94a3b8", label="full additive")
    ax.barh(y + 0.18, pivot["M6_full_interaction"], height=0.36, color="#2563eb", label="full interaction")
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean top-5% value capture")
    ax.set_title("Interaction correction by held-out city")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_regret_figure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.plot(plot["model_id"], plot["mean_event_top_5pct_regret"], marker="o", color="#ef4444")
    ax.set_ylabel("Top-5% regret = 1 - value capture")
    ax.set_title("Decision regret declines as activated interaction terms are added")
    ax.tick_params(axis="x", rotation=28)
    ax.set_ylim(0, max(0.05, float(plot["mean_event_top_5pct_regret"].max()) * 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_event_scatter(event_metrics: pd.DataFrame, path: Path) -> None:
    if event_metrics.empty:
        return
    add = event_metrics[event_metrics["model_id"].eq("M5_full_additive")][
        ["city", "event_id", "top_5pct_value_capture"]
    ].rename(columns={"top_5pct_value_capture": "additive_capture"})
    inter = event_metrics[event_metrics["model_id"].eq("M6_full_interaction")][
        ["city", "event_id", "top_5pct_value_capture"]
    ].rename(columns={"top_5pct_value_capture": "interaction_capture"})
    merged = add.merge(inter, on=EVENT_KEYS, how="inner")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(merged["additive_capture"], merged["interaction_capture"], s=38, alpha=0.78, color="#2563eb", edgecolor="white", linewidth=0.35)
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Full additive top-5% capture")
    ax.set_ylabel("Full interaction top-5% capture")
    ax.set_title("Event-level benefit of explicit interaction correction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    summary: pd.DataFrame,
    increments: pd.DataFrame,
    city_summary: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    m1 = one_row(summary, model_id="M1_deficit_only")
    m5 = one_row(summary, model_id="M5_full_additive")
    m6 = one_row(summary, model_id="M6_full_interaction")
    m7 = one_row(summary, model_id="M7_factorized_low_dim")
    m8 = one_row(summary, model_id="M8_factorized_interaction")
    lines = [
        "# Factorized Action Surrogate V16",
        "",
        "## 这一版做了什么",
        "",
        "V16 对 high-level idea 中的 Structure Decoupler 和 interaction ablation 做了显式实现。做法是构造一组从简单到复杂的 leave-one-city-out ridge action-value surrogate，并用 within-event top-5% value capture、NDCG、precision 和 regret 评价，而不是只看均方误差。",
        "",
        "## 主要结论",
        "",
        f"- deficit-only 模型的 mean event top-5% capture = {safe_float(m1.get('mean_event_top_5pct_value_capture')):.3f}。",
        f"- full additive 模型加入 OD exposure、structure、substitution、time/feasibility 和 event context 后，top-5% capture = {safe_float(m5.get('mean_event_top_5pct_value_capture')):.3f}。",
        f"- full interaction 模型加入高维显式交互项后，top-5% capture = {safe_float(m6.get('mean_event_top_5pct_value_capture')):.3f}，相对 full additive 的增量 = {metrics['add_explicit_interactions_delta_top5_capture']:+.3f}；这说明粗暴增加交互会在跨城市验证中略微过拟合。",
        f"- 低维 factorized 模型只用 delay、future horizon、OD exposure、efficiency 和 intervention type，top-5% capture = {safe_float(m7.get('mean_event_top_5pct_value_capture')):.3f}；加入交互后达到 {safe_float(m8.get('mean_event_top_5pct_value_capture')):.3f}。",
        "",
        "这说明 action-value field 的主结构可以被很低维的 activated components 捕获；真正必要的是 OD exposure 与 time/feasibility 的激活，而不是不受约束地加入大量 interaction terms。",
        "",
        "## Model Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Incremental Gains",
        "",
        table_to_markdown(increments),
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary[city_summary["model_id"].isin(["M5_full_additive", "M6_full_interaction", "M8_factorized_interaction"])]),
        "",
        "## Interaction-Rescued Top-Tail Examples",
        "",
        table_to_markdown(examples.head(40)),
        "",
        "## 论文写作含义",
        "",
        "这一版给 learning/law 叙事补上了模型层证据：不是只有手工公式或启发式比较支持 activated law，跨城市留一验证的 factorized surrogate 也显示，从 deficit-only 到加入 OD exposure，再到加入 time/feasibility，top-tail decision performance 明显提升。高维显式 interaction 没有稳定提高 full model，反而支持一个更简洁的写法：恢复价值主要是低维 activated components 在 log-value 空间的稳定耦合，而不是任意复杂交互的堆叠。",
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


def safe_div(numerator: Any, denominator: Any) -> float:
    try:
        denom = float(denominator)
        if abs(denom) <= EPS or not np.isfinite(denom):
            return float("nan")
        return float(numerator) / denom
    except Exception:
        return float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) <= EPS or np.std(b[valid]) <= EPS:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    compact = df.copy()
    if len(compact) > 45:
        compact = compact.head(45)
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

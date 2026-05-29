"""Evaluate observed versus shuffled OD-graph structure in action-value learning.

The current data provide OD-zone dependence and derived structural features, but
not a full road-adjacency graph for every city. This script implements the graph
structure ablation that is possible with the current evidence: compare local
dynamic/action features, observed OD graph features, and graph features shuffled
across units while preserving each city's graph-feature distribution.
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
RANDOM_SEED = 271828
TOP_FRACS = (0.01, 0.05, 0.10)


LOCAL_STATE_FEATURES = [
    "local_remaining_rank",
    "access_remaining_rank",
    "passive_b_rank",
    "passive_ell_rank",
    "b0_rank",
    "h_total_rank",
    "local_need_rank",
]

ACTION_TIME_FEATURES = [
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

GRAPH_EXPOSURE_FEATURES = [
    "origin_exposure_rank",
    "destination_importance_rank",
    "log_law_exposure",
]

GRAPH_STRUCTURE_FEATURES = [
    "out_degree_rank",
    "in_degree_rank",
    "od_scarcity",
    "structure_only_score",
]

SHUFFLED_EXPOSURE_FEATURES = [
    "shuffled_origin_exposure_rank",
    "shuffled_destination_importance_rank",
    "shuffled_log_law_exposure",
]

SHUFFLED_STRUCTURE_FEATURES = [
    "shuffled_out_degree_rank",
    "shuffled_in_degree_rank",
    "shuffled_od_scarcity",
    "shuffled_structure_only_score",
]

GRAPH_UNIT_COLUMNS = [
    "origin_exposure",
    "destination_importance",
    "origin_exposure_rank",
    "destination_importance_rank",
    "out_degree_rank",
    "in_degree_rank",
    "od_scarcity",
]


MODEL_SPECS = [
    {
        "model_id": "G0_action_time_only",
        "family": "no_graph",
        "description": "action timing, feasibility, intervention type, and event context only",
        "features": ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES,
    },
    {
        "model_id": "G1_local_dynamic_no_graph",
        "family": "no_graph",
        "description": "local/passive deficit dynamics plus action mechanics, no OD graph",
        "features": LOCAL_STATE_FEATURES + ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES,
    },
    {
        "model_id": "G2_graph_only_observed",
        "family": "observed_graph",
        "description": "observed OD exposure and structural features plus action mechanics",
        "features": GRAPH_EXPOSURE_FEATURES + GRAPH_STRUCTURE_FEATURES + ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES,
    },
    {
        "model_id": "G3_graph_only_shuffled",
        "family": "shuffled_graph",
        "description": "shuffled OD exposure and structural features plus action mechanics",
        "features": SHUFFLED_EXPOSURE_FEATURES + SHUFFLED_STRUCTURE_FEATURES + ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES,
    },
    {
        "model_id": "G4_local_plus_observed_exposure",
        "family": "observed_graph",
        "description": "local dynamics plus observed OD exposure",
        "features": LOCAL_STATE_FEATURES + ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES + GRAPH_EXPOSURE_FEATURES,
    },
    {
        "model_id": "G5_local_plus_shuffled_exposure",
        "family": "shuffled_graph",
        "description": "local dynamics plus shuffled OD exposure",
        "features": LOCAL_STATE_FEATURES + ACTION_TIME_FEATURES + EVENT_CONTEXT_FEATURES + SHUFFLED_EXPOSURE_FEATURES,
    },
    {
        "model_id": "G6_local_plus_observed_od_graph",
        "family": "observed_graph",
        "description": "local dynamics plus observed OD exposure, degree, and scarcity",
        "features": (
            LOCAL_STATE_FEATURES
            + ACTION_TIME_FEATURES
            + EVENT_CONTEXT_FEATURES
            + GRAPH_EXPOSURE_FEATURES
            + GRAPH_STRUCTURE_FEATURES
        ),
    },
    {
        "model_id": "G7_local_plus_shuffled_od_graph",
        "family": "shuffled_graph",
        "description": "local dynamics plus shuffled OD exposure, degree, and scarcity",
        "features": (
            LOCAL_STATE_FEATURES
            + ACTION_TIME_FEATURES
            + EVENT_CONTEXT_FEATURES
            + SHUFFLED_EXPOSURE_FEATURES
            + SHUFFLED_STRUCTURE_FEATURES
        ),
    },
    {
        "model_id": "G8_factorized_observed_od",
        "family": "observed_graph",
        "description": "low-dimensional activated law with observed OD exposure",
        "features": [
            "delay_feasible",
            "log_active_weighted_horizon",
            "log_law_exposure",
            "log_eta_per_cost",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
    },
    {
        "model_id": "G9_factorized_shuffled_od",
        "family": "shuffled_graph",
        "description": "low-dimensional activated law with shuffled OD exposure",
        "features": [
            "delay_feasible",
            "log_active_weighted_horizon",
            "shuffled_log_law_exposure",
            "log_eta_per_cost",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
    },
]


SHUFFLE_COMPARISONS = [
    ("graph_only_alignment", "G2_graph_only_observed", "G3_graph_only_shuffled"),
    ("exposure_alignment", "G4_local_plus_observed_exposure", "G5_local_plus_shuffled_exposure"),
    ("full_od_graph_alignment", "G6_local_plus_observed_od_graph", "G7_local_plus_shuffled_od_graph"),
    ("factorized_od_alignment", "G8_factorized_observed_od", "G9_factorized_shuffled_od"),
    ("observed_graph_over_no_graph", "G6_local_plus_observed_od_graph", "G1_local_dynamic_no_graph"),
    ("factorized_observed_over_no_graph", "G8_factorized_observed_od", "G1_local_dynamic_no_graph"),
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "graph_structure_ablation"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    tokens = prepare_tokens(tokens)
    validate_features(tokens)

    leave_city_metrics, event_metrics, coefficients = run_leave_city_out(tokens)
    summary = summarize_models(leave_city_metrics, event_metrics)
    gaps = build_shuffle_gaps(summary)
    city_gaps = build_city_gaps(event_metrics)
    event_gaps = build_event_gaps(event_metrics)
    metrics = build_metrics(summary, gaps)

    write_table(summary, table_dir / "graph_structure_model_summary.csv")
    write_table(leave_city_metrics, table_dir / "graph_structure_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "graph_structure_event_metrics.csv")
    write_table(gaps, table_dir / "graph_structure_shuffle_gaps.csv")
    write_table(city_gaps, table_dir / "graph_structure_city_gaps.csv")
    write_table(event_gaps, table_dir / "graph_structure_event_gaps.csv")
    write_table(coefficients, table_dir / "graph_structure_coefficients.csv")
    (table_dir / "graph_structure_ablation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(summary, gaps, city_gaps, event_gaps, figure_dir)
    write_report(report_dir / "graph_structure_ablation_report_zh.md", metrics, summary, gaps, city_gaps, event_gaps)
    print(f"Wrote graph structure ablation to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    return pd.read_csv(path)


def prepare_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    numeric_cols = set(
        LOCAL_STATE_FEATURES
        + ACTION_TIME_FEATURES
        + EVENT_CONTEXT_FEATURES
        + GRAPH_EXPOSURE_FEATURES
        + GRAPH_STRUCTURE_FEATURES
        + GRAPH_UNIT_COLUMNS
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
    df = add_shuffled_graph_features(df)
    return df


def add_shuffled_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    shuffled_unit_tables: list[pd.DataFrame] = []
    for city, group in out.groupby("city", sort=True):
        unit = (
            group[["city", "unit", *GRAPH_UNIT_COLUMNS]]
            .drop_duplicates(["city", "unit"])
            .sort_values("unit")
            .reset_index(drop=True)
        )
        rng = np.random.default_rng(RANDOM_SEED + stable_city_seed(str(city)))
        perm = rng.permutation(len(unit))
        shuffled = unit[["city", "unit"]].copy()
        for col in GRAPH_UNIT_COLUMNS:
            shuffled[f"shuffled_{col}"] = unit[col].to_numpy()[perm]
        shuffled_unit_tables.append(shuffled)
    shuffled_units = pd.concat(shuffled_unit_tables, ignore_index=True)
    out = out.merge(shuffled_units, on=["city", "unit"], how="left")
    out["shuffled_law_exposure_term"] = np.where(
        out["intervention"].astype(str).eq("S"),
        out["shuffled_origin_exposure"],
        out["shuffled_destination_importance"],
    )
    out["shuffled_log_law_exposure"] = np.log1p(1_000.0 * out["shuffled_law_exposure_term"].clip(lower=0.0))
    out["shuffled_structure_only_score"] = np.where(
        out["intervention"].astype(str).eq("S"),
        out["shuffled_origin_exposure_rank"] * (1.0 - out["shuffled_out_degree_rank"]),
        out["shuffled_destination_importance_rank"] * (1.0 - out["shuffled_out_degree_rank"]),
    )
    return out


def stable_city_seed(city: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(city)) % 100_000


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in MODEL_SPECS:
        for feature in spec["features"]:
            if feature not in tokens:
                missing.append(feature)
    if missing:
        raise KeyError(f"Missing graph ablation features: {sorted(set(missing))}")


def run_leave_city_out(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        for heldout_city in sorted(tokens["city"].unique()):
            train = tokens[tokens["city"] != heldout_city].copy()
            test = tokens[tokens["city"] == heldout_city].copy()
            model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
            test["predicted_value"] = np.expm1(predict_ridge(model, test[features])) / 1_000.0
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
    return safe_div(np.sum(chosen * discount[: len(chosen)]), np.sum(ideal * discount[: len(ideal)]))


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


def build_shuffle_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison, observed_id, shuffled_id in SHUFFLE_COMPARISONS:
        observed = one_row(summary, model_id=observed_id)
        shuffled = one_row(summary, model_id=shuffled_id)
        if observed.empty or shuffled.empty:
            continue
        rows.append(
            {
                "comparison": comparison,
                "observed_model": observed_id,
                "baseline_or_shuffled_model": shuffled_id,
                "observed_top5_capture": safe_float(observed.get("mean_event_top_5pct_value_capture")),
                "baseline_top5_capture": safe_float(shuffled.get("mean_event_top_5pct_value_capture")),
                "delta_top5_capture": safe_float(observed.get("mean_event_top_5pct_value_capture"))
                - safe_float(shuffled.get("mean_event_top_5pct_value_capture")),
                "observed_top5_ndcg": safe_float(observed.get("mean_event_top_5pct_ndcg")),
                "baseline_top5_ndcg": safe_float(shuffled.get("mean_event_top_5pct_ndcg")),
                "delta_top5_ndcg": safe_float(observed.get("mean_event_top_5pct_ndcg"))
                - safe_float(shuffled.get("mean_event_top_5pct_ndcg")),
                "delta_event_spearman": safe_float(observed.get("mean_event_spearman"))
                - safe_float(shuffled.get("mean_event_spearman")),
            }
        )
    return pd.DataFrame(rows)


def build_city_gaps(event_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison, observed_id, shuffled_id in SHUFFLE_COMPARISONS:
        observed = event_metrics[event_metrics["model_id"].eq(observed_id)]
        shuffled = event_metrics[event_metrics["model_id"].eq(shuffled_id)]
        merged = observed.merge(
            shuffled,
            on=EVENT_KEYS,
            suffixes=("_observed", "_baseline"),
            how="inner",
        )
        if merged.empty:
            continue
        merged["delta_top5_capture"] = merged["top_5pct_value_capture_observed"] - merged["top_5pct_value_capture_baseline"]
        merged["delta_top5_ndcg"] = merged["top_5pct_ndcg_observed"] - merged["top_5pct_ndcg_baseline"]
        for city, group in merged.groupby("city", sort=True):
            rows.append(
                {
                    "comparison": comparison,
                    "city": city,
                    "n_events": int(group["event_id"].nunique()),
                    "mean_delta_top5_capture": float(group["delta_top5_capture"].mean()),
                    "median_delta_top5_capture": float(group["delta_top5_capture"].median()),
                    "mean_delta_top5_ndcg": float(group["delta_top5_ndcg"].mean()),
                    "positive_delta_share": float((group["delta_top5_capture"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_event_gaps(event_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for comparison, observed_id, shuffled_id in SHUFFLE_COMPARISONS:
        observed = event_metrics[event_metrics["model_id"].eq(observed_id)]
        shuffled = event_metrics[event_metrics["model_id"].eq(shuffled_id)]
        merged = observed.merge(
            shuffled,
            on=EVENT_KEYS,
            suffixes=("_observed", "_baseline"),
            how="inner",
        )
        if merged.empty:
            continue
        merged["comparison"] = comparison
        merged["observed_model"] = observed_id
        merged["baseline_or_shuffled_model"] = shuffled_id
        merged["delta_top5_capture"] = merged["top_5pct_value_capture_observed"] - merged["top_5pct_value_capture_baseline"]
        merged["delta_top5_ndcg"] = merged["top_5pct_ndcg_observed"] - merged["top_5pct_ndcg_baseline"]
        keep = [
            "comparison",
            "observed_model",
            "baseline_or_shuffled_model",
            "city",
            "event_id",
            "top_5pct_value_capture_observed",
            "top_5pct_value_capture_baseline",
            "delta_top5_capture",
            "top_5pct_ndcg_observed",
            "top_5pct_ndcg_baseline",
            "delta_top5_ndcg",
        ]
        rows.append(merged[keep])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_metrics(summary: pd.DataFrame, gaps: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        row = one_row(summary, model_id=spec["model_id"])
        prefix = spec["model_id"].lower()
        metrics[f"{prefix}_top5_capture"] = safe_float(row.get("mean_event_top_5pct_value_capture"))
        metrics[f"{prefix}_top5_ndcg"] = safe_float(row.get("mean_event_top_5pct_ndcg"))
        metrics[f"{prefix}_event_spearman"] = safe_float(row.get("mean_event_spearman"))
    for _, row in gaps.iterrows():
        comparison = str(row["comparison"])
        metrics[f"{comparison}_delta_top5_capture"] = safe_float(row.get("delta_top5_capture"))
        metrics[f"{comparison}_delta_top5_ndcg"] = safe_float(row.get("delta_top5_ndcg"))
    return metrics


def make_figures(
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    city_gaps: pd.DataFrame,
    event_gaps: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_model_summary_figure(summary, figure_dir / "graph_ablation_model_ladder.png")
    make_shuffle_gap_figure(gaps, figure_dir / "observed_vs_shuffled_graph_gaps.png")
    make_city_gap_figure(city_gaps, figure_dir / "graph_alignment_gap_by_city.png")
    make_event_gap_scatter(event_gaps, figure_dir / "observed_vs_shuffled_event_capture.png")


def make_model_summary_figure(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    plot = summary.copy()
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    colors = plot["family"].map({"no_graph": "#94a3b8", "observed_graph": "#2563eb", "shuffled_graph": "#ef4444"}).fillna("#64748b")
    ax.bar(x, plot["mean_event_top_5pct_value_capture"], color=colors, width=0.68)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean event top-5% value capture")
    ax.set_xticks(x, plot["model_id"], rotation=28, ha="right")
    ax.set_title("OD-graph structure ablation under leave-one-city-out validation")
    for idx, value in enumerate(plot["mean_event_top_5pct_value_capture"]):
        ax.text(idx, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_shuffle_gap_figure(gaps: pd.DataFrame, path: Path) -> None:
    if gaps.empty:
        return
    plot = gaps.copy()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(plot))
    ax.barh(y, plot["delta_top5_capture"], color=np.where(plot["delta_top5_capture"] >= 0, "#2563eb", "#ef4444"))
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_yticks(y, plot["comparison"])
    ax.set_xlabel("Observed model minus shuffled/no-graph baseline")
    ax.set_title("Does spatial graph alignment add top-tail value?")
    for idx, value in enumerate(plot["delta_top5_capture"]):
        ax.text(value + (0.004 if value >= 0 else -0.004), idx, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_gap_figure(city_gaps: pd.DataFrame, path: Path) -> None:
    plot = city_gaps[city_gaps["comparison"].eq("full_od_graph_alignment")].copy()
    if plot.empty:
        return
    plot = plot.sort_values("mean_delta_top5_capture")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    y = np.arange(len(plot))
    ax.barh(y, plot["mean_delta_top5_capture"], color=np.where(plot["mean_delta_top5_capture"] >= 0, "#2563eb", "#ef4444"))
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_yticks(y, plot["city"])
    ax.set_xlabel("Observed full OD graph - shuffled graph top-5% capture")
    ax.set_title("OD graph alignment gap by city")
    for idx, value in enumerate(plot["mean_delta_top5_capture"]):
        ax.text(value + (0.004 if value >= 0 else -0.004), idx, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_event_gap_scatter(event_gaps: pd.DataFrame, path: Path) -> None:
    plot = event_gaps[event_gaps["comparison"].eq("factorized_od_alignment")].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    ax.scatter(
        plot["top_5pct_value_capture_baseline"],
        plot["top_5pct_value_capture_observed"],
        s=36,
        alpha=0.78,
        color="#2563eb",
        edgecolor="white",
        linewidth=0.35,
    )
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Shuffled OD exposure top-5% capture")
    ax.set_ylabel("Observed OD exposure top-5% capture")
    ax.set_title("Event-level observed versus shuffled OD alignment")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    city_gaps: pd.DataFrame,
    event_gaps: pd.DataFrame,
) -> None:
    g1 = one_row(summary, model_id="G1_local_dynamic_no_graph")
    g6 = one_row(summary, model_id="G6_local_plus_observed_od_graph")
    g7 = one_row(summary, model_id="G7_local_plus_shuffled_od_graph")
    g8 = one_row(summary, model_id="G8_factorized_observed_od")
    g9 = one_row(summary, model_id="G9_factorized_shuffled_od")
    lines = [
        "# Graph Structure Ablation V17",
        "",
        "## 这一版做了什么",
        "",
        "V17 实现 high-level idea 中的 graph structure ablation。由于当前全量城市样本没有统一 road-adjacency graph，本版聚焦现有可验证的 OD-dependency graph：比较 no-graph local dynamics、observed OD exposure/structure、以及在城市内打乱 unit 对齐但保留分布的 shuffled OD graph。",
        "",
        "## 主要结论",
        "",
        f"- local dynamic no-graph 模型 top-5% capture = {safe_float(g1.get('mean_event_top_5pct_value_capture')):.3f}。",
        f"- local + observed OD graph top-5% capture = {safe_float(g6.get('mean_event_top_5pct_value_capture')):.3f}；local + shuffled OD graph = {safe_float(g7.get('mean_event_top_5pct_value_capture')):.3f}。",
        f"- factorized observed OD 模型 top-5% capture = {safe_float(g8.get('mean_event_top_5pct_value_capture')):.3f}；factorized shuffled OD = {safe_float(g9.get('mean_event_top_5pct_value_capture')):.3f}。",
        f"- full OD graph alignment gap = {metrics['full_od_graph_alignment_delta_top5_capture']:+.3f}；factorized OD alignment gap = {metrics['factorized_od_alignment_delta_top5_capture']:+.3f}。",
        "",
        "解释：如果 observed-shuffled gap 为正，说明 OD graph 的空间对齐本身有价值；如果 gap 很小或为负，则说明当前低维 law 主要依赖 OD exposure 的分布和 action feasibility，而不是需要更复杂的图拓扑表示。",
        "",
        "## Model Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Shuffle / No-Graph Gaps",
        "",
        table_to_markdown(gaps),
        "",
        "## City Alignment Gaps",
        "",
        table_to_markdown(city_gaps[city_gaps["comparison"].isin(["full_od_graph_alignment", "factorized_od_alignment"])]),
        "",
        "## Top Event Alignment Gaps",
        "",
        table_to_markdown(event_gaps.sort_values("delta_top5_capture", ascending=False).head(30)),
        "",
        "## 论文写作含义",
        "",
        "这版给 graph 需求划了一条清楚边界：当前证据已经能检验 OD-dependency graph 的低维结构和空间对齐，但还不能声称测试了完整 road graph 或 road-adjacency message passing。论文里应写成：OD graph exposure 是 activated law 的核心结构变量；是否需要更高阶 road graph 表示仍是下一步，而不是当前结论的必要前提。",
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

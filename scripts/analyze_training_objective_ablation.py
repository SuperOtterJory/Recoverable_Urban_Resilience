"""Compare value-regression, ranking, and top-tail training objectives.

The learning plan emphasizes that action-value learning should be evaluated by
within-event ranking and top-K regret, not only by absolute-value regression.
This script keeps the same interpretable feature families but changes the
training target/weights to test whether the recovered law depends on a specific
regression objective.
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
    INTERACTION_FEATURES,
    STRUCTURE_FEATURES,
    SUBSTITUTION_FEATURES,
    TIME_FEASIBILITY_FEATURES,
    prepare_tokens,
)
from recoverable_resilience.paths import find_repo_root


RIDGE_ALPHA = 2.0
TOP_FRACS = (0.01, 0.05, 0.10)
FEATURE_SPECS = [
    {
        "feature_set": "factorized_low_dim",
        "description": "seven-feature activated law components",
        "features": FACTORIZED_BASE,
    },
    {
        "feature_set": "full_additive",
        "description": "full additive action-value features",
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
        "feature_set": "full_interaction",
        "description": "full additive features plus explicit interaction terms",
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
]

OBJECTIVE_SPECS = [
    {
        "objective_id": "O1_log_value",
        "description": "ordinary log marginal-value regression",
        "target_col": "target_log",
        "weight_col": None,
    },
    {
        "objective_id": "O2_event_centered_log",
        "description": "within-event centered log-value regression",
        "target_col": "event_centered_target_log",
        "weight_col": None,
    },
    {
        "objective_id": "O3_top_tail_weighted_log",
        "description": "log-value regression with top-tail action weights",
        "target_col": "target_log",
        "weight_col": "top_tail_training_weight",
    },
    {
        "objective_id": "O4_rank_percentile",
        "description": "within-event target rank-percentile regression",
        "target_col": "target_rank_percentile",
        "weight_col": None,
    },
    {
        "objective_id": "O5_event_zscore_log",
        "description": "within-event standardized log-value regression",
        "target_col": "event_zscore_target_log",
        "weight_col": None,
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "training_objective_ablation"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    tokens = add_objective_targets(tokens)
    validate_features(tokens)

    leave_city, event_metrics = run_leave_city_objectives(tokens)
    summary = summarize_objectives(leave_city, event_metrics)
    improvements = build_improvement_summary(summary)
    diagnostics = build_diagnostics(summary, improvements)

    write_table(summary, table_dir / "objective_model_summary.csv")
    write_table(leave_city, table_dir / "objective_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "objective_event_metrics.csv")
    write_table(improvements, table_dir / "objective_improvement_summary.csv")
    (table_dir / "training_objective_ablation_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(summary, leave_city, improvements, figure_dir)
    write_report(
        report_dir / "training_objective_ablation_report_zh.md",
        diagnostics,
        summary,
        improvements,
    )
    print(f"Wrote training-objective ablation to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    return prepare_tokens(pd.read_csv(path))


def add_objective_targets(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    event_group = df.groupby(EVENT_KEYS, sort=False)
    df["event_mean_target_log"] = event_group["target_log"].transform("mean")
    df["event_std_target_log"] = event_group["target_log"].transform("std").replace(0.0, np.nan)
    df["event_centered_target_log"] = df["target_log"] - df["event_mean_target_log"]
    df["event_zscore_target_log"] = (df["target_log"] - df["event_mean_target_log"]) / df["event_std_target_log"].fillna(1.0)
    df["target_rank_percentile"] = event_group["target_value"].rank(method="average", pct=True)
    df["target_desc_rank_percentile"] = 1.0 - event_group["target_value"].rank(method="average", pct=True, ascending=False)
    top5 = event_group["target_value"].rank(method="first", pct=True, ascending=False) <= 0.05
    top20 = event_group["target_value"].rank(method="first", pct=True, ascending=False) <= 0.20
    positive = pd.to_numeric(df["target_value"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
    df["top_tail_training_weight"] = 1.0 + 8.0 * top5.astype(float) + 3.0 * top20.astype(float) + 1.0 * positive.astype(float)
    return df


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in FEATURE_SPECS:
        for feature in spec["features"]:
            if feature not in tokens:
                missing.append(feature)
    for spec in OBJECTIVE_SPECS:
        if spec["target_col"] not in tokens:
            missing.append(spec["target_col"])
        weight_col = spec.get("weight_col")
        if weight_col and weight_col not in tokens:
            missing.append(weight_col)
    if missing:
        raise KeyError(f"Missing objective-ablation features: {sorted(set(missing))}")


def run_leave_city_objectives(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for feature_spec in FEATURE_SPECS:
        features = list(feature_spec["features"])
        for objective in OBJECTIVE_SPECS:
            target_col = str(objective["target_col"])
            weight_col = objective.get("weight_col")
            for heldout_city in sorted(tokens["city"].unique()):
                train = tokens[tokens["city"] != heldout_city].copy()
                test = tokens[tokens["city"] == heldout_city].copy()
                weights = train[str(weight_col)] if weight_col else None
                model = fit_weighted_ridge(train[features], train[target_col], weights=weights, alpha=RIDGE_ALPHA)
                test["predicted_score"] = predict_linear(model, test[features])
                base = {
                    "feature_set": feature_spec["feature_set"],
                    "feature_description": feature_spec["description"],
                    "objective_id": objective["objective_id"],
                    "objective_description": objective["description"],
                    "heldout_city": heldout_city,
                    "n_features": len(features),
                }
                metric_rows.append({**base, **prediction_metrics(test, "predicted_score")})
                event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def fit_weighted_ridge(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    weights: pd.Series | None,
    alpha: float,
) -> dict[str, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    if weights is None:
        w = np.ones(len(y_arr), dtype=float)
    else:
        w = pd.to_numeric(weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        w = np.clip(w, 1e-6, np.inf)
    w = w / max(float(np.mean(w)), 1e-12)
    mean = weighted_mean(x_arr, w)
    std = weighted_std(x_arr, w, mean)
    std = np.where(std <= 1e-12, 1.0, std)
    x_std = np.nan_to_num((x_arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    sqrt_w = np.sqrt(w)
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y_arr * sqrt_w
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(weighted_design.T @ weighted_design + penalty, weighted_design.T @ weighted_y)
    return {"coef": coef, "mean": mean, "std": std}


def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    safe_x = np.where(valid, x, 0.0)
    denom = np.sum(np.where(valid, w[:, None], 0.0), axis=0)
    return np.divide(np.sum(safe_x * w[:, None], axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 0)


def weighted_std(x: np.ndarray, w: np.ndarray, mean: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    centered = np.where(valid, x - mean, 0.0)
    denom = np.sum(np.where(valid, w[:, None], 0.0), axis=0)
    var = np.divide(np.sum((centered**2) * w[:, None], axis=0), denom, out=np.ones(x.shape[1]), where=denom > 0)
    return np.sqrt(np.maximum(var, 0.0))


def predict_linear(model: dict[str, np.ndarray], x: pd.DataFrame) -> np.ndarray:
    x_arr = x.to_numpy(dtype=float)
    x_std = np.nan_to_num((x_arr - model["mean"]) / model["std"], nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return design @ model["coef"]


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    events = [event_top_metrics(group, score_col) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
    event_df = pd.DataFrame(events)
    return {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "spearman": safe_float(frame["target_value"].corr(frame[score_col], method="spearman")),
        "top_1pct_value_capture": safe_float(event_df["top_1pct_value_capture"].mean()),
        "top_5pct_value_capture": safe_float(event_df["top_5pct_value_capture"].mean()),
        "top_10pct_value_capture": safe_float(event_df["top_10pct_value_capture"].mean()),
        "top_5pct_ndcg": safe_float(event_df["top_5pct_ndcg"].mean()),
        "top_5pct_precision": safe_float(event_df["top_5pct_precision"].mean()),
        "top_5pct_regret": 1.0 - safe_float(event_df["top_5pct_value_capture"].mean()),
    }


def event_metric_rows(frame: pd.DataFrame, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        rows.append(
            {
                **base,
                "city": city,
                "event_id": int(event_id),
                "n_tokens": int(len(group)),
                "spearman": safe_float(group["target_value"].corr(group["predicted_score"], method="spearman")),
                **event_top_metrics(group, "predicted_score"),
            }
        )
    return rows


def event_top_metrics(group: pd.DataFrame, score_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for frac in TOP_FRACS:
        k = max(1, int(np.ceil(len(group) * frac)))
        if group.empty or group["target_value"].sum() <= 1e-12:
            capture = ndcg = precision = np.nan
        else:
            chosen = group.nlargest(k, score_col)
            ideal = group.nlargest(k, "target_value")
            chosen_values = chosen["target_value"].to_numpy(dtype=float)
            ideal_values = ideal["target_value"].to_numpy(dtype=float)
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            capture = safe_div(float(chosen_values.sum()), float(ideal_values.sum()))
            ndcg = safe_div(float(np.sum(chosen_values * discount[: len(chosen_values)])), float(np.sum(ideal_values * discount[: len(ideal_values)])))
            precision = len(set(chosen.index) & set(ideal.index)) / k
        label = f"top_{int(frac * 100)}pct"
        out[f"{label}_value_capture"] = capture
        out[f"{label}_ndcg"] = ndcg
        out[f"{label}_precision"] = precision
        out[f"{label}_regret"] = 1.0 - capture if np.isfinite(capture) else np.nan
    return out


def summarize_objectives(leave_city: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_spec in FEATURE_SPECS:
        for objective in OBJECTIVE_SPECS:
            city_group = leave_city[
                leave_city["feature_set"].eq(feature_spec["feature_set"])
                & leave_city["objective_id"].eq(objective["objective_id"])
            ]
            event_group = event_metrics[
                event_metrics["feature_set"].eq(feature_spec["feature_set"])
                & event_metrics["objective_id"].eq(objective["objective_id"])
            ]
            if city_group.empty:
                continue
            rows.append(
                {
                    "feature_set": feature_spec["feature_set"],
                    "feature_description": feature_spec["description"],
                    "objective_id": objective["objective_id"],
                    "objective_description": objective["description"],
                    "n_features": len(feature_spec["features"]),
                    "n_cities": int(city_group["heldout_city"].nunique()),
                    "n_events": int(event_group[EVENT_KEYS].drop_duplicates().shape[0]),
                    "mean_city_spearman": safe_float(city_group["spearman"].mean()),
                    "mean_event_spearman": safe_float(event_group["spearman"].mean()),
                    "mean_event_top_1pct_value_capture": safe_float(event_group["top_1pct_value_capture"].mean()),
                    "mean_event_top_5pct_value_capture": safe_float(event_group["top_5pct_value_capture"].mean()),
                    "mean_event_top_10pct_value_capture": safe_float(event_group["top_10pct_value_capture"].mean()),
                    "mean_event_top_5pct_ndcg": safe_float(event_group["top_5pct_ndcg"].mean()),
                    "mean_event_top_5pct_precision": safe_float(event_group["top_5pct_precision"].mean()),
                    "mean_event_top_5pct_regret": 1.0 - safe_float(event_group["top_5pct_value_capture"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_set", "mean_event_top_5pct_value_capture"], ascending=[True, False])


def build_improvement_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set, group in summary.groupby("feature_set", sort=True):
        raw = one_row(group, objective_id="O1_log_value")
        best = group.sort_values("mean_event_top_5pct_value_capture", ascending=False).iloc[0]
        top_weighted = one_row(group, objective_id="O3_top_tail_weighted_log")
        rank = one_row(group, objective_id="O4_rank_percentile")
        rows.append(
            {
                "feature_set": feature_set,
                "raw_log_top5_capture": safe_float(raw.get("mean_event_top_5pct_value_capture")),
                "top_tail_weighted_top5_capture": safe_float(top_weighted.get("mean_event_top_5pct_value_capture")),
                "rank_percentile_top5_capture": safe_float(rank.get("mean_event_top_5pct_value_capture")),
                "best_objective_id": str(best["objective_id"]),
                "best_top5_capture": safe_float(best["mean_event_top_5pct_value_capture"]),
                "best_minus_raw_top5_capture": safe_float(best["mean_event_top_5pct_value_capture"]) - safe_float(raw.get("mean_event_top_5pct_value_capture")),
                "top_tail_weighted_minus_raw_top5_capture": safe_float(top_weighted.get("mean_event_top_5pct_value_capture")) - safe_float(raw.get("mean_event_top_5pct_value_capture")),
                "rank_percentile_minus_raw_top5_capture": safe_float(rank.get("mean_event_top_5pct_value_capture")) - safe_float(raw.get("mean_event_top_5pct_value_capture")),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(summary: pd.DataFrame, improvements: pd.DataFrame) -> dict[str, Any]:
    factorized = improvements[improvements["feature_set"].eq("factorized_low_dim")].iloc[0]
    full = improvements[improvements["feature_set"].eq("full_additive")].iloc[0]
    interaction = improvements[improvements["feature_set"].eq("full_interaction")].iloc[0]
    factorized_best = one_row(summary, feature_set="factorized_low_dim", objective_id=str(factorized["best_objective_id"]))
    return {
        "factorized_raw_log_top5_capture": safe_float(factorized["raw_log_top5_capture"]),
        "factorized_top_tail_weighted_top5_capture": safe_float(factorized["top_tail_weighted_top5_capture"]),
        "factorized_rank_percentile_top5_capture": safe_float(factorized["rank_percentile_top5_capture"]),
        "factorized_best_objective_id": str(factorized["best_objective_id"]),
        "factorized_best_top5_capture": safe_float(factorized["best_top5_capture"]),
        "factorized_best_minus_raw_top5_capture": safe_float(factorized["best_minus_raw_top5_capture"]),
        "factorized_best_top5_regret": safe_float(factorized_best.get("mean_event_top_5pct_regret")),
        "full_additive_best_objective_id": str(full["best_objective_id"]),
        "full_additive_best_top5_capture": safe_float(full["best_top5_capture"]),
        "full_additive_best_minus_raw_top5_capture": safe_float(full["best_minus_raw_top5_capture"]),
        "full_interaction_best_objective_id": str(interaction["best_objective_id"]),
        "full_interaction_best_top5_capture": safe_float(interaction["best_top5_capture"]),
        "full_interaction_best_minus_raw_top5_capture": safe_float(interaction["best_minus_raw_top5_capture"]),
    }


def make_figures(summary: pd.DataFrame, leave_city: pd.DataFrame, improvements: pd.DataFrame, figure_dir: Path) -> None:
    make_objective_ladder(summary, figure_dir / "objective_top5_capture.png")
    make_improvement_figure(improvements, figure_dir / "objective_improvement_vs_raw.png")
    make_city_regret_figure(leave_city, figure_dir / "objective_regret_by_city.png")


def make_objective_ladder(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot_table(
        index="objective_id",
        columns="feature_set",
        values="mean_event_top_5pct_value_capture",
        aggfunc="first",
    ).loc[[spec["objective_id"] for spec in OBJECTIVE_SPECS]]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = {"factorized_low_dim": "#0f766e", "full_additive": "#2563eb", "full_interaction": "#7c3aed"}
    offsets = np.linspace(-width, width, len(pivot.columns))
    for offset, column in zip(offsets, pivot.columns):
        ax.bar(x + offset, pivot[column], width=width, label=column, color=colors.get(column, "#64748b"))
    ax.set_xticks(x, pivot.index, rotation=25, ha="right")
    ax.set_ylim(0.84, 1.0)
    ax.set_ylabel("Leave-city mean event top-5% value capture")
    ax.set_title("Training objective ablation for action-value ranking")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_improvement_figure(improvements: pd.DataFrame, path: Path) -> None:
    plot = improvements.copy()
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(y - 0.18, plot["top_tail_weighted_minus_raw_top5_capture"], height=0.34, color="#2563eb", label="top-tail weighted")
    ax.barh(y + 0.18, plot["rank_percentile_minus_raw_top5_capture"], height=0.34, color="#0f766e", label="rank percentile")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, plot["feature_set"])
    ax.set_xlabel("Delta top-5% capture versus ordinary log-value regression")
    ax.set_title("Does a ranking-aware objective reduce top-tail regret?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_regret_figure(leave_city: pd.DataFrame, path: Path) -> None:
    keep = leave_city[
        leave_city["feature_set"].eq("factorized_low_dim")
        & leave_city["objective_id"].isin(["O1_log_value", "O3_top_tail_weighted_log", "O4_rank_percentile"])
    ].copy()
    pivot = keep.pivot_table(index="heldout_city", columns="objective_id", values="top_5pct_regret", aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(pivot.index))
    for objective_id, color in [("O1_log_value", "#94a3b8"), ("O3_top_tail_weighted_log", "#2563eb"), ("O4_rank_percentile", "#0f766e")]:
        ax.plot(x, pivot[objective_id], marker="o", linewidth=1.8, label=objective_id, color=color)
    ax.set_xticks(x, pivot.index, rotation=25, ha="right")
    ax.set_ylabel("Top-5% regret = 1 - value capture")
    ax.set_title("Factorized law regret by held-out city")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, diagnostics: dict[str, Any], summary: pd.DataFrame, improvements: pd.DataFrame) -> None:
    lines = [
        "# Training Objective Ablation V19",
        "",
        "## 这一版做了什么",
        "",
        "V19 检验 high-level idea 中的 training objective 问题：学习目标是否应该只是 marginal value regression，还是需要 within-event ranking / top-K regret 风格的目标。具体比较普通 log-value regression、event-centered log regression、top-tail weighted regression、rank-percentile regression、event-zscore regression。",
        "",
        "## 主要结论",
        "",
        f"- factorized ordinary log-value top-5% capture = {diagnostics['factorized_raw_log_top5_capture']:.4f}。",
        f"- factorized top-tail weighted top-5% capture = {diagnostics['factorized_top_tail_weighted_top5_capture']:.4f}；rank-percentile = {diagnostics['factorized_rank_percentile_top5_capture']:.4f}。",
        f"- factorized best objective = {diagnostics['factorized_best_objective_id']}，top-5% capture = {diagnostics['factorized_best_top5_capture']:.4f}，delta vs raw = {diagnostics['factorized_best_minus_raw_top5_capture']:+.4f}。",
        f"- full additive best objective = {diagnostics['full_additive_best_objective_id']}，top-5% capture = {diagnostics['full_additive_best_top5_capture']:.4f}，delta vs raw = {diagnostics['full_additive_best_minus_raw_top5_capture']:+.4f}。",
        f"- full interaction best objective = {diagnostics['full_interaction_best_objective_id']}，top-5% capture = {diagnostics['full_interaction_best_top5_capture']:.4f}，delta vs raw = {diagnostics['full_interaction_best_minus_raw_top5_capture']:+.4f}。",
        "",
        "解释：如果 ranking-aware 或 top-tail-weighted objective 明显提升 top-5% capture，就说明后续 surrogate 应以 ranking/policy-regret 为训练目标；如果提升很小或为负，则说明当前 log-value regression 已经足够贴近 top-tail ranking，而 law 的科学结论主要来自 feature structure，而不是训练 objective trick。",
        "",
        "## Objective Model Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Improvement Summary",
        "",
        table_to_markdown(improvements),
        "",
        "## 论文写作含义",
        "",
        "这一版可以补一句方法论边界：本文用 top-tail capture 和 regret 作为核心评价指标，但在当前 action-token 数据上，简单 log-value regression 已经是很强的 decision-centered objective；ranking-aware 变体用于稳健性检查，而不是改变 law 结论的必要条件。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 1e-12 else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()

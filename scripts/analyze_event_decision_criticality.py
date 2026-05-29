"""Event-level severity decoupling for recoverability laws.

This script strengthens the event-level law in the learning plan: large
rainfall disruptions are not necessarily decision-critical; events become
decision-critical when recoverable value is concentrated in a top tail of
candidate actions. It also audits an important boundary of the current data:
because event deficits are spatially distributed through OD vulnerability, much
of the top-tail concentration signal is city-structural rather than a fully
event-specific rainfall footprint.
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


EPS = 1e-12
RIDGE_ALPHA = 1.0
TARGETS = [
    "decision_criticality_score",
    "recoverable_fraction",
    "lp_minus_random",
    "law_minus_random",
]
SEVERITY_FEATURES = [
    "log_baseline_objective",
    "log_event_total_precip",
    "event_peak_positive_abnormal_deficit",
]
TOP_TAIL_FEATURES = [
    "top_5pct_value_share",
    "marginal_value_gini",
    "optimizer_selected_value_share",
]
PREDICTOR_SPECS = [
    {
        "model_id": "E0_severity_only",
        "description": "rainfall and no-intervention loss severity only",
        "features": SEVERITY_FEATURES,
    },
    {
        "model_id": "E1_top_tail_structure",
        "description": "marginal-value top-tail concentration and optimizer selected-value share",
        "features": TOP_TAIL_FEATURES,
    },
    {
        "model_id": "E2_severity_plus_top_tail",
        "description": "severity features plus top-tail concentration",
        "features": SEVERITY_FEATURES + TOP_TAIL_FEATURES,
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "event_decision_criticality"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    events = build_event_dataset(root)
    correlations = build_correlations(events)
    variance = build_variance_decomposition(events)
    model_metrics, predictions = run_leave_city_models(events)
    examples = build_counterexamples(events)
    phase_summary = build_phase_summary(events)
    diagnostics = build_diagnostics(events, correlations, variance, model_metrics, examples)

    write_table(events, table_dir / "event_decision_criticality_dataset.csv")
    write_table(correlations, table_dir / "event_decision_correlations.csv")
    write_table(variance, table_dir / "event_variance_decomposition.csv")
    write_table(model_metrics, table_dir / "event_model_comparison.csv")
    write_table(predictions, table_dir / "event_model_predictions.csv")
    write_table(examples, table_dir / "event_counterexample_examples.csv")
    write_table(phase_summary, table_dir / "event_phase_summary.csv")
    (table_dir / "event_decision_criticality_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(events, correlations, variance, model_metrics, examples, figure_dir)
    write_report(
        report_dir / "event_decision_criticality_report_zh.md",
        diagnostics,
        correlations,
        variance,
        model_metrics,
        examples,
        phase_summary,
    )
    print(f"Wrote event decision-criticality analysis to {output_dir}")


def build_event_dataset(root: Path) -> pd.DataFrame:
    event_law = pd.read_csv(root / "results" / "law_learning" / "tables" / "event_level_top_tail_law.csv")
    replay = pd.read_csv(root / "results" / "law_learning" / "tables" / "fixed_policy_replay.csv")
    residual = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv")

    event_law["event_id"] = pd.to_numeric(event_law["event_id"], errors="coerce").astype(int)
    replay["event_id"] = pd.to_numeric(replay["event_id"], errors="coerce").astype(int)
    residual["event_id"] = pd.to_numeric(residual["event_id"], errors="coerce").astype(int)

    base_replay = replay[replay["policy_scenario"].eq("base")].copy()
    replay_pivot = base_replay.pivot_table(
        index=["city", "event_id"],
        columns="policy_score",
        values="replay_recoverable_fraction",
        aggfunc="first",
    ).reset_index()
    replay_pivot.columns.name = None
    simple_policies = ["random_positive", "deficit_only", "exposure_only", "structure_only"]
    available_simple = [col for col in simple_policies if col in replay_pivot]
    replay_pivot["best_simple_recoverable_fraction"] = replay_pivot[available_simple].max(axis=1)
    replay_pivot["lp_minus_random"] = replay_pivot["lp_optimizer_replay"] - replay_pivot["random_positive"]
    replay_pivot["lp_minus_best_simple"] = replay_pivot["lp_optimizer_replay"] - replay_pivot["best_simple_recoverable_fraction"]
    replay_pivot["law_minus_random"] = replay_pivot["activated_bottleneck_law"] - replay_pivot["random_positive"]
    replay_pivot["law_minus_best_simple"] = replay_pivot["activated_bottleneck_law"] - replay_pivot["best_simple_recoverable_fraction"]

    residual_keep = [
        "city",
        "event_id",
        "static_fraction_of_lp_gain",
        "residual_fraction_of_lp_gain",
        "residual_gain_improvement_over_static",
        "residual_gap_to_lp",
        "static_gap_to_lp",
    ]
    events = event_law.merge(replay_pivot, on=["city", "event_id"], how="left")
    events = events.merge(residual[[col for col in residual_keep if col in residual]], on=["city", "event_id"], how="left")
    events["log_baseline_objective"] = np.log1p(events["baseline_objective"].clip(lower=0.0))
    events["log_event_total_precip"] = np.log1p(events["event_total_precip"].clip(lower=0.0))
    add_ranks(events)
    add_phase_flags(events)
    return events.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)


def add_ranks(events: pd.DataFrame) -> None:
    rank_cols = [
        "baseline_objective",
        "event_total_precip",
        "event_peak_positive_abnormal_deficit",
        "recoverable_fraction",
        "decision_criticality_score",
        "lp_minus_random",
        "lp_minus_best_simple",
        "law_minus_random",
        "top_5pct_value_share",
        "marginal_value_gini",
    ]
    for col in rank_cols:
        if col in events:
            events[f"{col}_rank"] = events[col].rank(method="average", pct=True)


def add_phase_flags(events: pd.DataFrame) -> None:
    events["high_loss_low_decision"] = (
        (events["baseline_objective_rank"] >= 0.75)
        & (events["decision_criticality_score_rank"] <= 0.50)
    )
    events["moderate_loss_high_decision"] = (
        (events["baseline_objective_rank"] <= 0.50)
        & (events["decision_criticality_score_rank"] >= 0.75)
    )
    events["high_rain_low_decision"] = (
        (events["event_total_precip_rank"] >= 0.75)
        & (events["decision_criticality_score_rank"] <= 0.50)
    )
    events["high_speed_impact_low_decision"] = (
        (events["event_peak_positive_abnormal_deficit_rank"] >= 0.75)
        & (events["decision_criticality_score_rank"] <= 0.50)
    )
    events["high_decision_not_high_loss"] = (
        (events["decision_criticality_score_rank"] >= 0.80)
        & (events["baseline_objective_rank"] < 0.80)
    )


def build_correlations(events: pd.DataFrame) -> pd.DataFrame:
    features = [
        "baseline_objective",
        "event_total_precip",
        "event_peak_positive_abnormal_deficit",
        "top_1pct_value_share",
        "top_5pct_value_share",
        "marginal_value_gini",
        "optimizer_selected_value_share",
    ]
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for feature in features:
            if target not in events or feature not in events:
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "overall_spearman": spearman(events[target], events[feature]),
                    "within_city_demeaned_corr": within_city_demeaned_corr(events, target, feature),
                    "n_events": int(events[[target, feature]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "overall_spearman"], ascending=[True, False])


def build_variance_decomposition(events: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "baseline_objective",
        "event_total_precip",
        "event_peak_positive_abnormal_deficit",
        "top_5pct_value_share",
        "marginal_value_gini",
        "recoverable_fraction",
        "decision_criticality_score",
        "lp_minus_random",
        "law_minus_random",
    ]
    rows: list[dict[str, Any]] = []
    for variable in variables:
        frame = events[["city", variable]].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            continue
        y = frame[variable].to_numpy(dtype=float)
        grand = float(y.mean())
        city_mean = frame.groupby("city")[variable].transform("mean").to_numpy(dtype=float)
        total = float(np.mean((y - grand) ** 2))
        between = float(np.mean((city_mean - grand) ** 2))
        within = float(np.mean((y - city_mean) ** 2))
        rows.append(
            {
                "variable": variable,
                "n_events": int(len(frame)),
                "n_cities": int(frame["city"].nunique()),
                "total_variance": total,
                "between_city_variance": between,
                "within_city_variance": within,
                "between_city_share": between / total if total > EPS else np.nan,
                "within_city_share": within / total if total > EPS else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_leave_city_models(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for target in TARGETS:
        for spec in PREDICTOR_SPECS:
            preds: list[pd.DataFrame] = []
            for heldout_city in sorted(events["city"].dropna().unique()):
                train = events[events["city"] != heldout_city].copy()
                test = events[events["city"] == heldout_city].copy()
                features = list(spec["features"])
                model = fit_ridge(train[features], train[target], alpha=RIDGE_ALPHA)
                test["predicted"] = predict_ridge(model, test[features])
                test["target"] = target
                test["model_id"] = spec["model_id"]
                test["heldout_city"] = heldout_city
                preds.append(test[["city", "event_id", "target", "model_id", "heldout_city", target, "predicted"]])
            pred = pd.concat(preds, ignore_index=True)
            prediction_rows.append(pred.rename(columns={target: "observed"}))
            metric_rows.append(
                {
                    "target": target,
                    "model_id": spec["model_id"],
                    "description": spec["description"],
                    "n_events": int(len(pred)),
                    "n_cities": int(pred["heldout_city"].nunique()),
                    "pooled_spearman": spearman(pred[target], pred["predicted"]),
                    "pooled_pearson": pearson(pred[target], pred["predicted"]),
                    "rmse": rmse(pred[target], pred["predicted"]),
                    "mae": mae(pred[target], pred["predicted"]),
                    "mean_city_spearman": mean_city_spearman(pred, target),
                }
            )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def build_counterexamples(events: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("high_loss_low_decision", "baseline_objective_rank", False),
        ("moderate_loss_high_decision", "decision_criticality_score_rank", False),
        ("high_rain_low_decision", "event_total_precip_rank", False),
        ("high_speed_impact_low_decision", "event_peak_positive_abnormal_deficit_rank", False),
        ("high_decision_not_high_loss", "decision_criticality_score_rank", False),
    ]
    rows: list[pd.DataFrame] = []
    keep = [
        "city",
        "event_id",
        "event_start",
        "baseline_objective",
        "baseline_objective_rank",
        "event_total_precip",
        "event_total_precip_rank",
        "event_peak_positive_abnormal_deficit",
        "event_peak_positive_abnormal_deficit_rank",
        "recoverable_fraction",
        "top_5pct_value_share",
        "marginal_value_gini",
        "decision_criticality_score",
        "decision_criticality_score_rank",
        "lp_minus_random",
        "law_minus_random",
    ]
    for category, sort_col, ascending in groups:
        subset = events[events[category]].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(sort_col, ascending=ascending).head(12)
        subset.insert(0, "category", category)
        rows.append(subset[["category", *[col for col in keep if col in subset]]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["category", *keep])


def build_phase_summary(events: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "high_loss_low_decision",
        "moderate_loss_high_decision",
        "high_rain_low_decision",
        "high_speed_impact_low_decision",
        "high_decision_not_high_loss",
    ]
    rows = []
    for flag in flags:
        subset = events[events[flag]].copy()
        rows.append(
            {
                "phase": flag,
                "n_events": int(len(subset)),
                "event_share": float(len(subset) / max(len(events), 1)),
                "mean_loss_rank": float(subset["baseline_objective_rank"].mean()) if not subset.empty else np.nan,
                "mean_decision_rank": float(subset["decision_criticality_score_rank"].mean()) if not subset.empty else np.nan,
                "mean_recoverable_fraction": float(subset["recoverable_fraction"].mean()) if not subset.empty else np.nan,
                "mean_top5_value_share": float(subset["top_5pct_value_share"].mean()) if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(
    events: pd.DataFrame,
    correlations: pd.DataFrame,
    variance: pd.DataFrame,
    model_metrics: pd.DataFrame,
    examples: pd.DataFrame,
) -> dict[str, Any]:
    decision_baseline = one_row(correlations, target="decision_criticality_score", feature="baseline_objective")
    decision_top5 = one_row(correlations, target="decision_criticality_score", feature="top_5pct_value_share")
    decision_gini = one_row(correlations, target="decision_criticality_score", feature="marginal_value_gini")
    recoverable_baseline = one_row(correlations, target="recoverable_fraction", feature="baseline_objective")
    recoverable_gini = one_row(correlations, target="recoverable_fraction", feature="marginal_value_gini")
    top5_var = one_row(variance, variable="top_5pct_value_share")
    gini_var = one_row(variance, variable="marginal_value_gini")
    decision_var = one_row(variance, variable="decision_criticality_score")
    severity_decision = one_row(model_metrics, target="decision_criticality_score", model_id="E0_severity_only")
    top_tail_decision = one_row(model_metrics, target="decision_criticality_score", model_id="E1_top_tail_structure")
    combined_decision = one_row(model_metrics, target="decision_criticality_score", model_id="E2_severity_plus_top_tail")
    return {
        "n_events": int(len(events)),
        "n_cities": int(events["city"].nunique()),
        "decision_vs_baseline_loss_spearman": safe_float(decision_baseline.get("overall_spearman")),
        "decision_vs_top5_share_spearman": safe_float(decision_top5.get("overall_spearman")),
        "decision_vs_gini_spearman": safe_float(decision_gini.get("overall_spearman")),
        "recoverable_vs_baseline_loss_spearman": safe_float(recoverable_baseline.get("overall_spearman")),
        "recoverable_vs_gini_spearman": safe_float(recoverable_gini.get("overall_spearman")),
        "top5_between_city_share": safe_float(top5_var.get("between_city_share")),
        "gini_between_city_share": safe_float(gini_var.get("between_city_share")),
        "decision_between_city_share": safe_float(decision_var.get("between_city_share")),
        "severity_only_decision_loco_spearman": safe_float(severity_decision.get("pooled_spearman")),
        "top_tail_decision_loco_spearman": safe_float(top_tail_decision.get("pooled_spearman")),
        "combined_decision_loco_spearman": safe_float(combined_decision.get("pooled_spearman")),
        "high_loss_low_decision_count": int(events["high_loss_low_decision"].sum()),
        "moderate_loss_high_decision_count": int(events["moderate_loss_high_decision"].sum()),
        "high_rain_low_decision_count": int(events["high_rain_low_decision"].sum()),
        "high_speed_impact_low_decision_count": int(events["high_speed_impact_low_decision"].sum()),
        "high_decision_not_high_loss_count": int(events["high_decision_not_high_loss"].sum()),
        "counterexample_rows": int(len(examples)),
    }


def make_figures(
    events: pd.DataFrame,
    correlations: pd.DataFrame,
    variance: pd.DataFrame,
    model_metrics: pd.DataFrame,
    examples: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_phase_figure(events, examples, figure_dir / "severity_decision_phase.png")
    make_correlation_figure(correlations, figure_dir / "severity_vs_top_tail_correlations.png")
    make_variance_figure(variance, figure_dir / "event_variance_decomposition.png")
    make_model_figure(model_metrics, figure_dir / "event_model_comparison.png")


def make_phase_figure(events: pd.DataFrame, examples: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    scatter = ax.scatter(
        events["baseline_objective_rank"],
        events["decision_criticality_score_rank"],
        c=events["top_5pct_value_share"],
        s=42 + 160 * events["recoverable_fraction"].clip(lower=0.0),
        cmap="viridis",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.75, color="#64748b", linestyle="--", linewidth=1)
    ax.axhline(0.75, color="#64748b", linestyle="--", linewidth=1)
    ax.axhline(0.50, color="#cbd5e1", linestyle=":", linewidth=1)
    ax.set_xlabel("No-intervention loss rank")
    ax.set_ylabel("Decision-criticality rank")
    ax.set_title("Large loss and decision-criticality are decoupled")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Top-5% value share")
    if not examples.empty:
        label_rows = examples[
            examples["category"].isin(["high_loss_low_decision", "moderate_loss_high_decision"])
        ].head(8)
        for row in label_rows.itertuples(index=False):
            ax.annotate(
                f"{row.city} {int(row.event_id)}",
                (row.baseline_objective_rank, row.decision_criticality_score_rank),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_correlation_figure(correlations: pd.DataFrame, path: Path) -> None:
    selected_features = [
        "baseline_objective",
        "event_total_precip",
        "event_peak_positive_abnormal_deficit",
        "top_5pct_value_share",
        "marginal_value_gini",
    ]
    selected_targets = ["decision_criticality_score", "recoverable_fraction", "law_minus_random"]
    plot = correlations[
        correlations["feature"].isin(selected_features)
        & correlations["target"].isin(selected_targets)
    ].copy()
    if plot.empty:
        return
    labels = {
        "baseline_objective": "loss",
        "event_total_precip": "rain",
        "event_peak_positive_abnormal_deficit": "speed impact",
        "top_5pct_value_share": "top-tail",
        "marginal_value_gini": "gini",
    }
    targets = list(selected_targets)
    x = np.arange(len(selected_features))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    colors = ["#2563eb", "#0f766e", "#9333ea"]
    for idx, target in enumerate(targets):
        values = []
        for feature in selected_features:
            row = one_row(plot, target=target, feature=feature)
            values.append(safe_float(row.get("overall_spearman")))
        ax.bar(x + (idx - 1) * width, values, width=width, label=target, color=colors[idx])
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, [labels[f] for f in selected_features], rotation=20)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Top-tail structure separates decision value from event severity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_variance_figure(variance: pd.DataFrame, path: Path) -> None:
    variables = [
        "baseline_objective",
        "event_total_precip",
        "top_5pct_value_share",
        "marginal_value_gini",
        "recoverable_fraction",
        "decision_criticality_score",
    ]
    labels = ["loss", "rain", "top-tail", "gini", "recoverable", "decision"]
    plot = variance[variance["variable"].isin(variables)].copy()
    if plot.empty:
        return
    plot["label"] = pd.Categorical(plot["variable"].map(dict(zip(variables, labels))), categories=labels, ordered=True)
    plot = plot.sort_values("label")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(plot["label"].astype(str), plot["between_city_share"], color="#2563eb", label="between-city")
    ax.bar(
        plot["label"].astype(str),
        plot["within_city_share"],
        bottom=plot["between_city_share"],
        color="#93c5fd",
        label="within-city",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Variance share")
    ax.set_title("Top-tail concentration is currently mostly city-structural")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_model_figure(model_metrics: pd.DataFrame, path: Path) -> None:
    plot = model_metrics[model_metrics["target"].isin(["decision_criticality_score", "law_minus_random"])].copy()
    if plot.empty:
        return
    labels = {
        "E0_severity_only": "severity",
        "E1_top_tail_structure": "top-tail",
        "E2_severity_plus_top_tail": "combined",
    }
    targets = list(plot["target"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for idx, target in enumerate(targets):
        values = []
        for model_id in labels:
            row = one_row(plot, target=target, model_id=model_id)
            values.append(safe_float(row.get("pooled_spearman")))
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=target)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, list(labels.values()))
    ax.set_ylabel("Leave-city pooled Spearman")
    ax.set_title("Event-level prediction: severity versus top-tail structure")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    correlations: pd.DataFrame,
    variance: pd.DataFrame,
    model_metrics: pd.DataFrame,
    examples: pd.DataFrame,
    phase_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Event-Level Decision-Criticality V21",
        "",
        "## 本版要回答的问题",
        "",
        "V21 检验事件级 Law B：降雨事件是否 decision-critical，不应只由雨量、速度冲击或无干预损失大小决定，而应由 recoverable value 是否形成 top-tail 决定。同时，本版也审视当前数据边界：由于事件冲击目前主要通过城市级速度异常再按 OD vulnerability 分配，top-tail concentration 在不少城市内接近常数，因此它更像城市结构信号，而不是完整的事件足迹信号。",
        "",
        "## 主要发现",
        "",
        f"- decision-criticality 与 baseline loss 的 Spearman = {diagnostics['decision_vs_baseline_loss_spearman']:.4f}，与 top-5% value share 的 Spearman = {diagnostics['decision_vs_top5_share_spearman']:.4f}，与 marginal-value gini 的 Spearman = {diagnostics['decision_vs_gini_spearman']:.4f}。",
        f"- recoverable fraction 与 baseline loss 的 Spearman = {diagnostics['recoverable_vs_baseline_loss_spearman']:.4f}，与 marginal-value gini 的 Spearman = {diagnostics['recoverable_vs_gini_spearman']:.4f}。",
        f"- top-5% value share 的 between-city variance share = {diagnostics['top5_between_city_share']:.4f}；marginal-value gini 的 between-city share = {diagnostics['gini_between_city_share']:.4f}。",
        f"- high-loss but low-decision events: {diagnostics['high_loss_low_decision_count']}；moderate-loss but high-decision events: {diagnostics['moderate_loss_high_decision_count']}；high-rain but low-decision events: {diagnostics['high_rain_low_decision_count']}。",
        "",
        "## 解释",
        "",
        "这些结果支持一个更精确的事件级表述：当前数据中，decision-criticality 不是 severity law，而是 city-structure/top-tail law。大损失或强降雨事件可能因为损失扩散、可恢复 top-tail 不集中而不具备高决策杠杆；较小或中等事件如果落在高暴露、低替代性的结构位置上，则可能更 decision-critical。",
        "",
        "但这个结论必须带上边界：当前 top-tail concentration 的城市间方差占比较高，说明它主要来自 OD 结构和当前空间标定，而不是精细观测到的事件空间 footprint。若后续有 zone-level speed/rainfall footprint，应重新检验 top-tail 是否在同一城市不同事件间显著变化。",
        "",
        "## Correlations",
        "",
        table_to_markdown(correlations, max_rows=30),
        "",
        "## Variance Decomposition",
        "",
        table_to_markdown(variance),
        "",
        "## Leave-City Event Models",
        "",
        table_to_markdown(model_metrics),
        "",
        "## Counterexamples",
        "",
        table_to_markdown(examples, max_rows=30),
        "",
        "## Phase Summary",
        "",
        table_to_markdown(phase_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    return design @ model["coef"]


def spearman(x: Any, y: Any) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(pair["x"].corr(pair["y"], method="spearman"))


def pearson(x: Any, y: Any) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(pair["x"].corr(pair["y"], method="pearson"))


def within_city_demeaned_corr(events: pd.DataFrame, target: str, feature: str) -> float:
    frame = events[["city", target, feature]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if frame.empty:
        return np.nan
    frame["target_resid"] = frame[target] - frame.groupby("city")[target].transform("mean")
    frame["feature_resid"] = frame[feature] - frame.groupby("city")[feature].transform("mean")
    return pearson(frame["target_resid"], frame["feature_resid"])


def mean_city_spearman(pred: pd.DataFrame, target: str) -> float:
    values = []
    for _, group in pred.groupby("heldout_city"):
        values.append(spearman(group[target], group["predicted"]))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def rmse(x: Any, y: Any) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return np.nan
    return float(np.sqrt(np.mean((pair["x"] - pair["y"]) ** 2)))


def mae(x: Any, y: Any) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return np.nan
    return float(np.mean(np.abs(pair["x"] - pair["y"])))


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


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


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def table_to_markdown(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


if __name__ == "__main__":
    main()

"""Test early-window predictability of hindsight recoverability laws."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


WINDOW_HOURS = [1, 2, 3, 6, 12]
TARGETS = [
    "recoverable_fraction",
    "decision_criticality_score",
    "top_5pct_value_share",
    "marginal_value_gini",
]
EPS = 1e-12


FEATURE_GROUPS: dict[str, list[str]] = {
    "early_rain": [
        "early_precip_sum",
        "early_precip_max",
        "early_precip_mean",
        "early_rainy_hour_share",
        "early_last_precip",
    ],
    "early_speed": [
        "early_positive_deficit_start",
        "early_positive_deficit_last",
        "early_positive_deficit_max",
        "early_positive_deficit_mean",
        "early_positive_deficit_sum",
        "early_positive_deficit_slope",
        "early_abnormal_deficit_mean",
        "early_p90_deficit_max",
        "early_mean_deficit_mean",
    ],
    "static_city": [
        "n_units",
        "q_density",
        "mean_a_retention",
        "dynamic_rain_kernel_sum",
        "delay_R",
        "delay_C",
        "delay_S",
    ],
    "start_state": [
        "weighted_b0",
        "city_signal_b0",
        "event_peak_precip_observed_so_far",
    ],
}


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "early_predictability"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root)
    features = build_early_feature_panel(data)
    metrics, predictions = run_leave_city_models(features)
    correlations = build_feature_correlations(features)
    best_by_target = best_metrics_by_target(metrics)

    write_table(features, table_dir / "early_feature_panel.csv")
    write_table(metrics, table_dir / "early_predictability_metrics.csv")
    write_table(predictions, table_dir / "early_predictability_predictions.csv")
    write_table(correlations, table_dir / "early_feature_correlations.csv")
    write_table(best_by_target, table_dir / "early_best_metrics_by_target.csv")

    make_figures(metrics, predictions, correlations, figure_dir)
    write_report(report_dir / "early_predictability_report_zh.md", metrics, best_by_target, correlations)
    print(f"Wrote early predictability analysis to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    results = root / "results"
    return {
        "event_law": pd.read_csv(results / "law_learning" / "tables" / "event_level_top_tail_law.csv", parse_dates=["event_start"]),
        "abnormal": pd.read_csv(results / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv", parse_dates=["hour"]),
        "calibration": pd.read_csv(results / "event_optimization" / "tables" / "event_calibration_summary.csv", parse_dates=["event_start"]),
    }


def build_early_feature_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    event_law = data["event_law"].copy()
    abnormal = data["abnormal"].copy()
    calibration = data["calibration"].copy()

    calibration = calibration[calibration["scenario"].astype(str).eq("base")].copy()
    cal_cols = [
        "city",
        "event_id",
        "n_units",
        "q_density",
        "weighted_b0",
        "mean_a_retention",
        "dynamic_rain_kernel_sum",
        "delay_R",
        "delay_C",
        "delay_S",
        "city_signal_b0",
        "event_peak_precip",
    ]
    calibration = calibration[[col for col in cal_cols if col in calibration.columns]].drop_duplicates(["city", "event_id"])
    event_law = event_law.merge(calibration, on=["city", "event_id"], how="left")

    rows: list[dict[str, Any]] = []
    grouped_abnormal = {
        city: group.sort_values("hour").set_index("hour")
        for city, group in abnormal.groupby("city", sort=False)
    }
    for event in event_law.itertuples(index=False):
        city = str(event.city)
        city_hourly = grouped_abnormal.get(city, pd.DataFrame())
        for window in WINDOW_HOURS:
            start = pd.Timestamp(event.event_start)
            end = start + pd.Timedelta(hours=int(window))
            if city_hourly.empty:
                hourly = pd.DataFrame()
            else:
                hourly = city_hourly[(city_hourly.index >= start) & (city_hourly.index < end)].copy()
            row = {
                "city": city,
                "event_id": int(event.event_id),
                "event_start": str(start),
                "window_hours": int(window),
            }
            for target in TARGETS + ["baseline_objective", "event_peak_positive_abnormal_deficit", "event_total_precip"]:
                if hasattr(event, target):
                    row[target] = getattr(event, target)
            for feature in FEATURE_GROUPS["static_city"] + ["weighted_b0", "city_signal_b0"]:
                row[feature] = getattr(event, feature, np.nan)
            row.update(early_window_features(hourly, event))
            rows.append(row)
    panel = pd.DataFrame(rows)
    for column in panel.columns:
        if column not in {"city", "event_start"}:
            try:
                panel[column] = pd.to_numeric(panel[column])
            except (TypeError, ValueError):
                pass
    return add_transforms(panel)


def early_window_features(hourly: pd.DataFrame, event: Any) -> dict[str, float]:
    if hourly.empty:
        base = {
            "early_observed_hours": 0,
            "early_precip_sum": np.nan,
            "early_precip_max": np.nan,
            "early_precip_mean": np.nan,
            "early_rainy_hour_share": np.nan,
            "early_last_precip": np.nan,
            "event_peak_precip_observed_so_far": np.nan,
            "early_positive_deficit_start": np.nan,
            "early_positive_deficit_last": np.nan,
            "early_positive_deficit_max": np.nan,
            "early_positive_deficit_mean": np.nan,
            "early_positive_deficit_sum": np.nan,
            "early_positive_deficit_slope": np.nan,
            "early_abnormal_deficit_mean": np.nan,
            "early_p90_deficit_max": np.nan,
            "early_mean_deficit_mean": np.nan,
            "early_observation_count_sum": np.nan,
        }
        return base
    precip = pd.to_numeric(hourly["precipitation"], errors="coerce").fillna(0.0)
    positive = pd.to_numeric(hourly["positive_abnormal_deficit"], errors="coerce").fillna(0.0)
    abnormal = pd.to_numeric(hourly["abnormal_deficit"], errors="coerce")
    p90 = pd.to_numeric(hourly["p90_deficit"], errors="coerce")
    mean_deficit = pd.to_numeric(hourly["mean_deficit"], errors="coerce")
    observation_count = pd.to_numeric(hourly["observation_count"], errors="coerce")
    return {
        "early_observed_hours": int(len(hourly)),
        "early_precip_sum": float(precip.sum()),
        "early_precip_max": float(precip.max()),
        "early_precip_mean": float(precip.mean()),
        "early_rainy_hour_share": float((precip > 0).mean()),
        "early_last_precip": float(precip.iloc[-1]),
        "event_peak_precip_observed_so_far": float(max(precip.max(), getattr(event, "event_peak_precip", 0.0) if len(hourly) == 0 else 0.0)),
        "early_positive_deficit_start": float(positive.iloc[0]),
        "early_positive_deficit_last": float(positive.iloc[-1]),
        "early_positive_deficit_max": float(positive.max()),
        "early_positive_deficit_mean": float(positive.mean()),
        "early_positive_deficit_sum": float(positive.sum()),
        "early_positive_deficit_slope": float((positive.iloc[-1] - positive.iloc[0]) / max(len(positive) - 1, 1)),
        "early_abnormal_deficit_mean": float(abnormal.mean()),
        "early_p90_deficit_max": float(p90.max()),
        "early_mean_deficit_mean": float(mean_deficit.mean()),
        "early_observation_count_sum": float(observation_count.sum()),
    }


def add_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for col in [
        "early_precip_sum",
        "early_precip_max",
        "early_positive_deficit_sum",
        "early_positive_deficit_max",
        "early_p90_deficit_max",
        "baseline_objective",
        "n_units",
    ]:
        if col in out:
            out[f"log_{col}"] = np.log1p(pd.to_numeric(out[col], errors="coerce").clip(lower=0.0))
    return out


def run_leave_city_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    all_features = sorted({feature for values in FEATURE_GROUPS.values() for feature in values})
    feature_sets = {
        "early_rain": FEATURE_GROUPS["early_rain"],
        "early_speed": FEATURE_GROUPS["early_speed"],
        "static_city": FEATURE_GROUPS["static_city"],
        "start_state": FEATURE_GROUPS["start_state"],
        "rain_plus_speed": FEATURE_GROUPS["early_rain"] + FEATURE_GROUPS["early_speed"],
        "speed_plus_static": FEATURE_GROUPS["early_speed"] + FEATURE_GROUPS["static_city"] + FEATURE_GROUPS["start_state"],
        "all_early": all_features,
    }
    for window in WINDOW_HOURS:
        window_panel = panel[panel["window_hours"].eq(window)].copy()
        for target in TARGETS:
            for feature_group, features in feature_sets.items():
                available = [feature for feature in features if feature in window_panel.columns]
                predictions = leave_city_predictions(window_panel, target, available)
                if predictions.empty:
                    continue
                metric_rows.append(
                    {
                        "window_hours": int(window),
                        "target": target,
                        "feature_group": feature_group,
                        "n_features": len(available),
                        **prediction_metrics(predictions, target, "prediction"),
                    }
                )
                predictions["window_hours"] = int(window)
                predictions["target"] = target
                predictions["feature_group"] = feature_group
                prediction_rows.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def leave_city_predictions(panel: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    if not features or target not in panel:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for city in sorted(panel["city"].unique()):
        train = panel[panel["city"] != city].copy()
        test = panel[panel["city"] == city].copy()
        train = train.dropna(subset=[target])
        test = test.dropna(subset=[target])
        if len(train) < len(features) + 5 or test.empty:
            continue
        model = fit_ridge(train[features], train[target], alpha=1.0)
        pred = predict_ridge(model, test[features])
        part = test[["city", "event_id", "event_start", target]].copy()
        part["prediction"] = pred
        part["heldout_city"] = city
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_ridge(x: pd.DataFrame, y: pd.Series, *, alpha: float) -> dict[str, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    col_mean = np.nanmean(x_arr, axis=0)
    col_std = np.nanstd(x_arr, axis=0)
    col_std = np.where(col_std <= 1e-12, 1.0, col_std)
    x_std = np.nan_to_num((x_arr - col_mean) / col_std, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
    return {"coef": coef, "mean": col_mean, "std": col_std}


def predict_ridge(model: dict[str, np.ndarray], x: pd.DataFrame) -> np.ndarray:
    x_arr = x.to_numpy(dtype=float)
    x_std = np.nan_to_num((x_arr - model["mean"]) / model["std"], nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return design @ model["coef"]


def prediction_metrics(frame: pd.DataFrame, target: str, score_col: str) -> dict[str, float]:
    y = pd.to_numeric(frame[target], errors="coerce")
    pred = pd.to_numeric(frame[score_col], errors="coerce")
    valid = y.notna() & pred.notna()
    work = frame.loc[valid].copy()
    if work.empty:
        return {"n_events": 0, "pearson": np.nan, "spearman": np.nan, "mae": np.nan, "top20_recall": np.nan, "top20_precision": np.nan}
    y = work[target].to_numpy(dtype=float)
    pred = work[score_col].to_numpy(dtype=float)
    k = max(1, int(np.ceil(0.20 * len(work))))
    true_top = set(np.argsort(y)[::-1][:k])
    pred_top = set(np.argsort(pred)[::-1][:k])
    hit = len(true_top & pred_top)
    return {
        "n_events": int(len(work)),
        "pearson": safe_corr(work[target], work[score_col], "pearson"),
        "spearman": safe_corr(work[target], work[score_col], "spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
        "top20_recall": float(hit / max(len(true_top), 1)),
        "top20_precision": float(hit / max(len(pred_top), 1)),
    }


def build_feature_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["window_hours"].isin([1, 2, 3, 6, 12])].copy()
    feature_candidates = sorted({feature for values in FEATURE_GROUPS.values() for feature in values if feature in base.columns})
    rows: list[dict[str, Any]] = []
    for window in WINDOW_HOURS:
        subset = base[base["window_hours"].eq(window)]
        for target in TARGETS:
            for feature in feature_candidates:
                pair = subset[[target, feature]].dropna()
                rows.append(
                    {
                        "window_hours": int(window),
                        "target": target,
                        "feature": feature,
                        "spearman": pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target", "window_hours", "spearman"], ascending=[True, True, False])


def best_metrics_by_target(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.sort_values(["target", "spearman", "top20_recall"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def make_figures(metrics: pd.DataFrame, predictions: pd.DataFrame, correlations: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_spearman_curve(metrics, figure_dir / "early_predictability_spearman.png")
    make_top20_curve(metrics, figure_dir / "early_predictability_top20_recall.png")
    make_prediction_scatter(predictions, metrics, figure_dir / "early_prediction_scatter.png")
    make_feature_correlation_figure(correlations, figure_dir / "early_feature_correlations.png")


def make_spearman_curve(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics[metrics["target"].isin(["decision_criticality_score", "recoverable_fraction"])].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for (target, group_name), group in subset.groupby(["target", "feature_group"]):
        if group_name not in {"early_speed", "static_city", "start_state", "speed_plus_static", "all_early"}:
            continue
        label = f"{target.replace('_', ' ')} / {group_name}"
        ax.plot(group["window_hours"], group["spearman"], marker="o", label=label)
    ax.axhline(0.0, color="#111827", linewidth=1, alpha=0.45)
    ax.set_xlabel("Available hours after event start")
    ax.set_ylabel("Leave-city Spearman")
    ax.set_title("Early predictability of final recoverability laws")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_top20_curve(metrics: pd.DataFrame, path: Path) -> None:
    subset = metrics[metrics["target"].eq("decision_criticality_score")].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for group_name, group in subset.groupby("feature_group"):
        ax.plot(group["window_hours"], group["top20_recall"], marker="o", label=group_name)
    ax.set_xlabel("Available hours after event start")
    ax.set_ylabel("Top-20% decision-critical recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Can early windows identify decision-critical events?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_prediction_scatter(predictions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    choice = metrics[
        metrics["target"].eq("decision_criticality_score")
        & metrics["feature_group"].eq("all_early")
        & metrics["window_hours"].eq(2)
    ]
    if choice.empty:
        choice = metrics[metrics["target"].eq("decision_criticality_score")].sort_values("spearman", ascending=False).head(1)
    if choice.empty:
        return
    row = choice.iloc[0]
    subset = predictions[
        predictions["target"].eq(row["target"])
        & predictions["feature_group"].eq(row["feature_group"])
        & predictions["window_hours"].eq(row["window_hours"])
    ].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(subset[row["target"]], subset["prediction"], color="#2563eb", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Final decision-criticality score")
    ax.set_ylabel("Early prediction")
    ax.set_title(f"{int(row['window_hours'])}h early prediction, {row['feature_group']}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_feature_correlation_figure(correlations: pd.DataFrame, path: Path) -> None:
    subset = correlations[
        correlations["target"].eq("decision_criticality_score")
        & correlations["feature"].isin(
            [
                "early_positive_deficit_max",
                "early_positive_deficit_sum",
                "early_precip_sum",
                "early_precip_max",
                "weighted_b0",
                "q_density",
                "mean_a_retention",
            ]
        )
    ].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(index="feature", columns="window_hours", values="spearman", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Available hours after event start")
    ax.set_title("Early feature correlations with final decision-criticality")
    fig.colorbar(image, ax=ax, label="Spearman")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, metrics: pd.DataFrame, best_by_target: pd.DataFrame, correlations: pd.DataFrame) -> None:
    decision = metrics[metrics["target"].eq("decision_criticality_score")].sort_values("spearman", ascending=False)
    recoverable = metrics[metrics["target"].eq("recoverable_fraction")].sort_values("spearman", ascending=False)
    decision_best = decision.iloc[0] if not decision.empty else pd.Series(dtype=float)
    recoverable_best = recoverable.iloc[0] if not recoverable.empty else pd.Series(dtype=float)
    decision_2h = one_row(metrics, target="decision_criticality_score", feature_group="all_early", window_hours=2)
    decision_static_1h = one_row(metrics, target="decision_criticality_score", feature_group="static_city", window_hours=1)
    decision_speed_1h = one_row(metrics, target="decision_criticality_score", feature_group="early_speed", window_hours=1)
    decision_rain_1h = one_row(metrics, target="decision_criticality_score", feature_group="early_rain", window_hours=1)
    recoverable_2h = one_row(metrics, target="recoverable_fraction", feature_group="all_early", window_hours=2)
    top_corr = (
        correlations[
            correlations["target"].eq("decision_criticality_score")
            & correlations["window_hours"].eq(2)
        ]
        .sort_values("spearman", ascending=False)
        .head(8)
    )
    lines = [
        "# Early Predictability Analysis V14",
        "",
        "## 这一版回答什么问题",
        "",
        "主分析是 hindsight counterfactual recoverability：用完整 12 小时事实轨迹问“如果管理者知道后来发生了什么，理论上能恢复多少”。V14 检验一个更谨慎的问题：只用事件开始后前 1/2/3/6 小时的 rainfall 和 abnormal speed 信息，加上静态城市结构，能否在 leave-one-city-out 设置下预测最终 recoverability 与 decision-criticality。",
        "",
        "## 主要结果",
        "",
        f"- best decision-criticality early model: window = {safe_int(decision_best.get('window_hours'))}h, group = {decision_best.get('feature_group', '')}, Spearman = {safe_float(decision_best.get('spearman')):.4f}, top-20 recall = {safe_float(decision_best.get('top20_recall')):.4f}",
        f"- 2h all-early decision-criticality: Spearman = {safe_float(decision_2h.get('spearman')):.4f}, top-20 recall = {safe_float(decision_2h.get('top20_recall')):.4f}",
        f"- 1h decision-criticality controls: static-city Spearman = {safe_float(decision_static_1h.get('spearman')):.4f}, early-speed Spearman = {safe_float(decision_speed_1h.get('spearman')):.4f}, early-rain Spearman = {safe_float(decision_rain_1h.get('spearman')):.4f}",
        f"- best recoverable-fraction early model: window = {safe_int(recoverable_best.get('window_hours'))}h, group = {recoverable_best.get('feature_group', '')}, Spearman = {safe_float(recoverable_best.get('spearman')):.4f}, top-20 recall = {safe_float(recoverable_best.get('top20_recall')):.4f}",
        f"- 2h all-early recoverable fraction: Spearman = {safe_float(recoverable_2h.get('spearman')):.4f}, top-20 recall = {safe_float(recoverable_2h.get('top20_recall')):.4f}",
        "",
        "## Best Metrics By Target",
        "",
        table_to_markdown(best_by_target),
        "",
        "## All Metrics",
        "",
        table_to_markdown(metrics),
        "",
        "## 2h Feature Correlations With Decision-Criticality",
        "",
        table_to_markdown(top_corr),
        "",
        "## 科学解释",
        "",
        "这版结果用于给论文加边界：decision-criticality 在很早窗口已有一定可识别性，但主要来自静态城市结构与早期速度异常的组合，而不是 rainfall-only signal。recoverable fraction 更依赖后续轨迹，2 小时窗口只是中等强度预测。因此主文仍应坚持 hindsight counterfactual framing；早期模型可以作为 supplementary evidence，说明哪些事件可能较早显露 decision-criticality，但不能等同于完整在线控制策略。",
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


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    try:
        return float(a.corr(b, method=method))
    except Exception:
        return float("nan")


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


if __name__ == "__main__":
    main()

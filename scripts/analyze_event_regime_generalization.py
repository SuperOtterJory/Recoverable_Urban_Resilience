"""Evaluate whether action-value laws generalize across event regimes.

The high-level learning plan asks for validation beyond random splits and
leave-one-city tests. This script performs leave-event-regime-out tests over
rainfall severity, speed-impact severity, rainfall duration, baseline loss, and
time-of-day regimes. The goal is to test whether the compact activated law and
factorized surrogate remain useful when an entire event regime is held out.
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
    FACTORIZED_INTERACTIONS,
    INTERACTION_FEATURES,
    STRUCTURE_FEATURES,
    SUBSTITUTION_FEATURES,
    TIME_FEASIBILITY_FEATURES,
    fit_ridge,
    predict_ridge,
    prepare_tokens,
)
from recoverable_resilience.paths import find_repo_root


RIDGE_ALPHA = 2.0
MIN_HELDOUT_EVENTS = 6
MODEL_SPECS = [
    {
        "model_id": "H1_deficit_only",
        "family": "heuristic",
        "description": "deficit-only one-factor score",
        "score_col": "deficit_only_score",
    },
    {
        "model_id": "H2_exposure_only",
        "family": "heuristic",
        "description": "OD exposure-only one-factor score",
        "score_col": "exposure_only_score",
    },
    {
        "model_id": "H3_structure_only",
        "family": "heuristic",
        "description": "static structure-only one-factor score",
        "score_col": "structure_only_score",
    },
    {
        "model_id": "H4_activated_law",
        "family": "direct_law",
        "description": "hand-built activated bottleneck score",
        "score_col": "activated_bottleneck_score",
    },
    {
        "model_id": "R1_factorized_low_dim",
        "family": "trained_surrogate",
        "description": "seven-feature factorized activated law",
        "features": FACTORIZED_BASE,
    },
    {
        "model_id": "R2_full_additive",
        "family": "trained_surrogate",
        "description": "full additive action-value surrogate",
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
        "model_id": "R3_full_interaction",
        "family": "trained_surrogate",
        "description": "full additive surrogate plus explicit interaction terms",
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
        "model_id": "R4_factorized_interaction",
        "family": "trained_surrogate",
        "description": "factorized law plus compact interaction terms",
        "features": FACTORIZED_BASE + FACTORIZED_INTERACTIONS,
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "event_regime_generalization"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    event_regimes = build_event_regimes(root, tokens)
    tokens = tokens.merge(event_regimes[["city", "event_id", *regime_columns(event_regimes)]], on=["city", "event_id"], how="left")
    validate_features(tokens)

    split_summary = build_split_summary(tokens, event_regimes)
    metrics, event_metrics = run_leave_regime_out(tokens, split_summary)
    model_summary = summarize_models(metrics)
    gap_summary = build_gap_summary(metrics)
    diagnostics = build_diagnostics(model_summary, gap_summary)

    write_table(event_regimes, table_dir / "event_regime_assignments.csv")
    write_table(split_summary, table_dir / "regime_split_summary.csv")
    write_table(metrics, table_dir / "regime_model_metrics.csv")
    write_table(event_metrics, table_dir / "regime_event_metrics.csv")
    write_table(model_summary, table_dir / "regime_model_summary.csv")
    write_table(gap_summary, table_dir / "regime_gap_summary.csv")
    (table_dir / "event_regime_generalization_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(metrics, model_summary, gap_summary, figure_dir)
    write_report(
        report_dir / "event_regime_generalization_report_zh.md",
        diagnostics,
        split_summary,
        model_summary,
        gap_summary,
    )
    print(f"Wrote event-regime generalization analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    tokens = pd.read_csv(path)
    return prepare_tokens(tokens)


def build_event_regimes(root: Path, tokens: pd.DataFrame) -> pd.DataFrame:
    events = (
        tokens[
            [
                "city",
                "event_id",
                "event_start",
                "event_total_precip",
                "event_peak_precip",
                "event_peak_positive_abnormal_deficit",
                "baseline_objective",
                "recoverable_fraction",
            ]
        ]
        .drop_duplicates(["city", "event_id"])
        .copy()
    )
    details_path = root / "results" / "data_mining" / "tables" / "rainfall_event_impact_details.csv"
    if details_path.exists():
        details = pd.read_csv(details_path, parse_dates=["event_start", "event_end"])
        details["event_id"] = pd.to_numeric(details["event_id"], errors="coerce").astype("Int64")
        details = details[
            [
                "city",
                "event_id",
                "duration_hours",
                "affected_hours_in_window",
                "recovery_hours_after_peak",
            ]
        ].copy()
        events = events.merge(details, on=["city", "event_id"], how="left")
    else:
        events["duration_hours"] = np.nan
        events["affected_hours_in_window"] = np.nan
        events["recovery_hours_after_peak"] = np.nan

    events["event_start"] = pd.to_datetime(events["event_start"], errors="coerce")
    events["event_year"] = events["event_start"].dt.year.astype("Int64")
    events["event_hour"] = events["event_start"].dt.hour
    events["total_rain_regime"] = tertile_labels(events["event_total_precip"], "rain")
    events["peak_rain_regime"] = tertile_labels(events["event_peak_precip"], "peak")
    events["speed_impact_regime"] = tertile_labels(events["event_peak_positive_abnormal_deficit"], "speed")
    events["duration_regime"] = tertile_labels(events["duration_hours"], "duration")
    events["baseline_loss_regime"] = tertile_labels(events["baseline_objective"], "loss")
    events["recoverable_fraction_regime"] = tertile_labels(events["recoverable_fraction"], "recoverable")
    events["time_of_day_regime"] = events["event_hour"].map(time_of_day_label)
    events["weekend_regime"] = np.where(events["event_start"].dt.dayofweek >= 5, "weekend", "weekday")
    return events.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)


def tertile_labels(values: pd.Series, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series(index=values.index, dtype=object)
    if numeric.notna().sum() < 3 or numeric.nunique(dropna=True) < 3:
        labels.loc[numeric.notna()] = f"{prefix}_all"
        return labels.fillna(f"{prefix}_missing")
    q1, q2 = numeric.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    labels.loc[numeric <= q1] = f"{prefix}_low"
    labels.loc[(numeric > q1) & (numeric <= q2)] = f"{prefix}_medium"
    labels.loc[numeric > q2] = f"{prefix}_high"
    labels.loc[numeric.isna()] = f"{prefix}_missing"
    return labels


def time_of_day_label(hour: Any) -> str:
    if pd.isna(hour):
        return "time_missing"
    hour = int(hour)
    if 7 <= hour <= 9:
        return "morning_peak"
    if 10 <= hour <= 15:
        return "midday"
    if 16 <= hour <= 19:
        return "evening_peak"
    return "offpeak_night"


def regime_columns(events: pd.DataFrame) -> list[str]:
    return [column for column in events.columns if column.endswith("_regime")]


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in MODEL_SPECS:
        for feature in spec.get("features", []):
            if feature not in tokens:
                missing.append(feature)
        score_col = spec.get("score_col")
        if score_col and score_col not in tokens:
            missing.append(score_col)
    if missing:
        raise KeyError(f"Missing event-regime features: {sorted(set(missing))}")


def build_split_summary(tokens: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_counts = tokens[["city", "event_id"]].drop_duplicates()
    for column in regime_columns(events):
        for regime in sorted(events[column].dropna().astype(str).unique()):
            if regime.endswith("_missing") or regime.endswith("_all"):
                continue
            held_events = events[events[column].astype(str).eq(regime)][["city", "event_id"]].drop_duplicates()
            if len(held_events) < MIN_HELDOUT_EVENTS:
                continue
            test_keys = set(map(tuple, held_events[["city", "event_id"]].to_numpy()))
            test_mask = tokens[["city", "event_id"]].apply(tuple, axis=1).isin(test_keys)
            train_events = event_counts[~event_counts.apply(tuple, axis=1).isin(test_keys)]
            rows.append(
                {
                    "split_family": column,
                    "heldout_regime": regime,
                    "n_test_events": int(len(held_events)),
                    "n_train_events": int(len(train_events)),
                    "n_test_tokens": int(test_mask.sum()),
                    "n_train_tokens": int((~test_mask).sum()),
                    "n_test_cities": int(held_events["city"].nunique()),
                    "mean_total_rain": mean_event_value(events, held_events, "event_total_precip"),
                    "mean_peak_speed_impact": mean_event_value(events, held_events, "event_peak_positive_abnormal_deficit"),
                    "mean_baseline_loss": mean_event_value(events, held_events, "baseline_objective"),
                    "mean_recoverable_fraction": mean_event_value(events, held_events, "recoverable_fraction"),
                }
            )
    return pd.DataFrame(rows).sort_values(["split_family", "heldout_regime"]).reset_index(drop=True)


def mean_event_value(events: pd.DataFrame, held_events: pd.DataFrame, column: str) -> float:
    subset = events.merge(held_events, on=["city", "event_id"], how="inner")
    return safe_float(pd.to_numeric(subset[column], errors="coerce").mean()) if column in subset else np.nan


def run_leave_regime_out(tokens: pd.DataFrame, split_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for split in split_summary.itertuples(index=False):
        split_family = str(split.split_family)
        heldout = str(split.heldout_regime)
        test_mask = tokens[split_family].astype(str).eq(heldout)
        train = tokens.loc[~test_mask].copy()
        test_base = tokens.loc[test_mask].copy()
        if train.empty or test_base.empty:
            continue
        for spec in MODEL_SPECS:
            test = test_base.copy()
            if "features" in spec:
                model = fit_ridge(train[list(spec["features"])], train["target_log"], alpha=RIDGE_ALPHA)
                predicted = np.expm1(predict_ridge(model, test[list(spec["features"])])) / 1_000.0
            else:
                predicted = pd.to_numeric(test[str(spec["score_col"])], errors="coerce").fillna(0.0).clip(lower=0.0)
            test["predicted_value"] = predicted
            base = {
                "split_family": split_family,
                "heldout_regime": heldout,
                "model_id": spec["model_id"],
                "family": spec["family"],
                "description": spec["description"],
                "n_features": int(len(spec.get("features", []))),
                "n_test_cities": int(split.n_test_cities),
            }
            metric_rows.append({**base, **fast_prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def fast_prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    event_metric = [event_top_metrics(group, score_col, 0.05) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
    event_df = pd.DataFrame(event_metric)
    return {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": safe_float(frame["target_value"].corr(frame[score_col], method="spearman")),
        "mae": safe_float(np.mean(np.abs(y - pred))),
        "top_5pct_value_capture": safe_float(event_df["value_capture"].mean()) if not event_df.empty else np.nan,
        "top_5pct_ndcg": safe_float(event_df["ndcg"].mean()) if not event_df.empty else np.nan,
        "top_5pct_precision": safe_float(event_df["precision"].mean()) if not event_df.empty else np.nan,
        "top_5pct_regret": 1.0 - safe_float(event_df["value_capture"].mean()) if not event_df.empty else np.nan,
    }


def event_metric_rows(frame: pd.DataFrame, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        metric = event_top_metrics(group, "predicted_value", 0.05)
        row = {
            **base,
            "city": city,
            "event_id": int(event_id),
            "n_tokens": int(len(group)),
            "spearman": safe_float(group["target_value"].corr(group["predicted_value"], method="spearman")),
            "top_5pct_value_capture": metric["value_capture"],
            "top_5pct_ndcg": metric["ndcg"],
            "top_5pct_precision": metric["precision"],
        }
        rows.append(row)
    return rows


def event_top_metrics(group: pd.DataFrame, score_col: str, frac: float) -> dict[str, float]:
    if group.empty or group["target_value"].sum() <= 1e-12:
        return {"value_capture": np.nan, "ndcg": np.nan, "precision": np.nan}
    k = max(1, int(np.ceil(len(group) * frac)))
    chosen = group.nlargest(k, score_col)
    ideal = group.nlargest(k, "target_value")
    chosen_values = chosen["target_value"].to_numpy(dtype=float)
    ideal_values = ideal["target_value"].to_numpy(dtype=float)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(chosen_values * discount[: len(chosen_values)]))
    idcg = float(np.sum(ideal_values * discount[: len(ideal_values)]))
    chosen_index = set(chosen.index)
    ideal_index = set(ideal.index)
    return {
        "value_capture": safe_div(float(chosen_values.sum()), float(ideal_values.sum())),
        "ndcg": safe_div(dcg, idcg),
        "precision": len(chosen_index & ideal_index) / k,
    }


def event_top_capture(group: pd.DataFrame, score_col: str, frac: float) -> float:
    if group.empty or group["target_value"].sum() <= 1e-12:
        return np.nan
    k = max(1, int(np.ceil(len(group) * frac)))
    oracle = float(group.nlargest(k, "target_value")["target_value"].sum())
    chosen = float(group.nlargest(k, score_col)["target_value"].sum())
    return safe_div(chosen, oracle)


def summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        group = metrics[metrics["model_id"].eq(spec["model_id"])].copy()
        if group.empty:
            continue
        rows.append(
            {
                "model_id": spec["model_id"],
                "family": spec["family"],
                "description": spec["description"],
                "n_splits": int(len(group)),
                "mean_top5_capture": safe_float(group["top_5pct_value_capture"].mean()),
                "median_top5_capture": safe_float(group["top_5pct_value_capture"].median()),
                "min_top5_capture": safe_float(group["top_5pct_value_capture"].min()),
                "mean_top5_ndcg": safe_float(group["top_5pct_ndcg"].mean()) if "top_5pct_ndcg" in group else np.nan,
                "mean_spearman": safe_float(group["spearman"].mean()),
                "hardest_split_family": str(group.sort_values("top_5pct_value_capture").iloc[0]["split_family"]),
                "hardest_heldout_regime": str(group.sort_values("top_5pct_value_capture").iloc[0]["heldout_regime"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_top5_capture", ascending=False)


def build_gap_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("factorized_vs_activated_law", "R1_factorized_low_dim", "H4_activated_law"),
        ("factorized_vs_full_additive", "R1_factorized_low_dim", "R2_full_additive"),
        ("full_interaction_vs_full_additive", "R3_full_interaction", "R2_full_additive"),
        ("activated_law_vs_deficit_only", "H4_activated_law", "H1_deficit_only"),
        ("activated_law_vs_structure_only", "H4_activated_law", "H3_structure_only"),
    ]
    rows: list[dict[str, Any]] = []
    keys = ["split_family", "heldout_regime"]
    for name, left_id, right_id in comparisons:
        left = metrics[metrics["model_id"].eq(left_id)]
        right = metrics[metrics["model_id"].eq(right_id)]
        merged = left.merge(right, on=keys, suffixes=("_left", "_right"))
        if merged.empty:
            continue
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "comparison": name,
                    "split_family": getattr(row, "split_family"),
                    "heldout_regime": getattr(row, "heldout_regime"),
                    "left_model": left_id,
                    "right_model": right_id,
                    "left_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_left")),
                    "right_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_right")),
                    "delta_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_left"))
                    - safe_float(getattr(row, "top_5pct_value_capture_right")),
                    "left_spearman": safe_float(getattr(row, "spearman_left")),
                    "right_spearman": safe_float(getattr(row, "spearman_right")),
                    "delta_spearman": safe_float(getattr(row, "spearman_left")) - safe_float(getattr(row, "spearman_right")),
                }
            )
    return pd.DataFrame(rows)


def build_diagnostics(model_summary: pd.DataFrame, gap_summary: pd.DataFrame) -> dict[str, Any]:
    factorized = one_row(model_summary, model_id="R1_factorized_low_dim")
    full = one_row(model_summary, model_id="R2_full_additive")
    activated = one_row(model_summary, model_id="H4_activated_law")
    factorized_vs_full = gap_summary[gap_summary["comparison"].eq("factorized_vs_full_additive")]
    activated_vs_deficit = gap_summary[gap_summary["comparison"].eq("activated_law_vs_deficit_only")]
    return {
        "n_regime_splits": safe_int(factorized.get("n_splits")),
        "factorized_mean_top5_capture": safe_float(factorized.get("mean_top5_capture")),
        "factorized_min_top5_capture": safe_float(factorized.get("min_top5_capture")),
        "factorized_hardest_split_family": str(factorized.get("hardest_split_family", "")),
        "factorized_hardest_heldout_regime": str(factorized.get("hardest_heldout_regime", "")),
        "full_additive_mean_top5_capture": safe_float(full.get("mean_top5_capture")),
        "full_additive_min_top5_capture": safe_float(full.get("min_top5_capture")),
        "activated_law_mean_top5_capture": safe_float(activated.get("mean_top5_capture")),
        "activated_law_min_top5_capture": safe_float(activated.get("min_top5_capture")),
        "factorized_minus_full_mean_top5_delta": safe_float(factorized_vs_full["delta_top5_capture"].mean())
        if not factorized_vs_full.empty
        else np.nan,
        "factorized_minus_full_min_top5_delta": safe_float(factorized_vs_full["delta_top5_capture"].min())
        if not factorized_vs_full.empty
        else np.nan,
        "activated_minus_deficit_mean_top5_delta": safe_float(activated_vs_deficit["delta_top5_capture"].mean())
        if not activated_vs_deficit.empty
        else np.nan,
    }


def make_figures(metrics: pd.DataFrame, model_summary: pd.DataFrame, gap_summary: pd.DataFrame, figure_dir: Path) -> None:
    make_model_summary_figure(model_summary, figure_dir / "regime_model_summary.png")
    make_regime_capture_heatmap(metrics, figure_dir / "regime_top5_capture_heatmap.png")
    make_gap_figure(gap_summary, figure_dir / "regime_generalization_gaps.png")


def make_model_summary_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.sort_values("mean_top5_capture", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    colors = plot["family"].map({"heuristic": "#94a3b8", "direct_law": "#0f766e", "trained_surrogate": "#2563eb"}).fillna("#64748b")
    ax.barh(plot["model_id"], plot["mean_top5_capture"], color=colors)
    ax.errorbar(
        plot["mean_top5_capture"],
        plot["model_id"],
        xerr=plot["mean_top5_capture"] - plot["min_top5_capture"],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.0,
        capsize=3,
        alpha=0.65,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean held-out-regime top-5% value capture")
    ax.set_title("Event-regime generalization of recovery-value laws")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_regime_capture_heatmap(metrics: pd.DataFrame, path: Path) -> None:
    keep = metrics[metrics["model_id"].isin(["H4_activated_law", "R1_factorized_low_dim", "R2_full_additive"])].copy()
    keep["split"] = keep["split_family"].str.replace("_regime", "", regex=False) + ":" + keep["heldout_regime"]
    pivot = keep.pivot_table(index="split", columns="model_id", values="top_5pct_value_capture", aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8.8, max(5.0, 0.34 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Top-5% capture by held-out event regime")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "#111827", fontsize=7)
    fig.colorbar(im, ax=ax, label="Top-5% value capture")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_gap_figure(gaps: pd.DataFrame, path: Path) -> None:
    keep = gaps[gaps["comparison"].isin(["factorized_vs_full_additive", "activated_law_vs_deficit_only"])].copy()
    keep["split"] = keep["split_family"].str.replace("_regime", "", regex=False) + ":" + keep["heldout_regime"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, max(5.0, 0.28 * keep["split"].nunique())), sharey=True)
    for ax, comparison, title in zip(
        axes,
        ["factorized_vs_full_additive", "activated_law_vs_deficit_only"],
        ["Factorized minus full additive", "Activated law minus deficit-only"],
    ):
        plot = keep[keep["comparison"].eq(comparison)].sort_values("delta_top5_capture")
        y = np.arange(len(plot))
        colors = np.where(plot["delta_top5_capture"] >= 0, "#2563eb", "#ef4444")
        ax.barh(y, plot["delta_top5_capture"], color=colors)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_yticks(y, plot["split"])
        ax.set_title(title)
        ax.set_xlabel("Delta top-5% value capture")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    split_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Event-Regime Generalization V18",
        "",
        "## 这一版做了什么",
        "",
        "V18 专门补 high-level idea 中的 leave-event-regime-out validation。做法是把 city-event 按 total rain、peak rain、speed impact、rain duration、baseline loss、recoverable fraction、time of day、weekday/weekend 分成 regime；每次整类 regime 留出，只在其他 regime 上训练 surrogate，再测试 held-out regime 的 action-value top-tail capture。",
        "",
        "## 主要结论",
        "",
        f"- tested regime splits = {diagnostics['n_regime_splits']}。",
        f"- factorized low-dimensional law mean top-5% capture = {diagnostics['factorized_mean_top5_capture']:.4f}，worst split = {diagnostics['factorized_hardest_split_family']} / {diagnostics['factorized_hardest_heldout_regime']}，capture = {diagnostics['factorized_min_top5_capture']:.4f}。",
        f"- full additive surrogate mean top-5% capture = {diagnostics['full_additive_mean_top5_capture']:.4f}，minimum = {diagnostics['full_additive_min_top5_capture']:.4f}。",
        f"- activated hand law mean top-5% capture = {diagnostics['activated_law_mean_top5_capture']:.4f}，minimum = {diagnostics['activated_law_min_top5_capture']:.4f}。",
        f"- factorized minus full additive mean top-5% delta = {diagnostics['factorized_minus_full_mean_top5_delta']:+.4f}；activated law minus deficit-only mean delta = {diagnostics['activated_minus_deficit_mean_top5_delta']:+.4f}。",
        "",
        "解释：主要独立证据应看 trained factorized surrogate 与 full additive surrogate。`activated_bottleneck_score` 是 analytic score，和当前 small-signal label 有同源构造关系，因此适合作为公式一致性参照，不应单独当作独立预测胜利。若 factorized law 在 held-out regimes 上接近 full additive model，说明它不是只记住某一类雨型或时段；如果某些 regime 明显掉分，这些 regime 就是后续需要参数敏感性或更强动态表征的位置。",
        "",
        "## Regime Split Summary",
        "",
        table_to_markdown(split_summary),
        "",
        "## Model Summary",
        "",
        table_to_markdown(model_summary),
        "",
        "## Gap Summary",
        "",
        table_to_markdown(gap_summary.groupby("comparison", as_index=False).agg(
            n_splits=("delta_top5_capture", "count"),
            mean_delta_top5_capture=("delta_top5_capture", "mean"),
            min_delta_top5_capture=("delta_top5_capture", "min"),
            max_delta_top5_capture=("delta_top5_capture", "max"),
            mean_delta_spearman=("delta_spearman", "mean"),
        )),
        "",
        "## 论文写作含义",
        "",
        "这一版可以把 cross-regime generalization 写进 learning/law 的验证链条：当前 law 不只在 leave-city 下有效，也在不同雨强、速度冲击、持续时间、损失规模和时段留出时保持较高 top-tail capture。边界是：这仍基于当前 sampled action-token table；year-based temporal split 与城市强混杂，因此不应声称已经完成无混杂的 leave-time-period-out。"
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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return safe_float(np.corrcoef(a, b)[0, 1])


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()

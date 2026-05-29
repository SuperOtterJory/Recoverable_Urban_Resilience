"""Temporal holdout tests for recoverability action-value laws.

The high-level learning plan asks for leave-time-period-out validation. In the
current empirical sample, calendar year is strongly confounded with city: 2019
contains New York/Chicago/Dallas/Houston, 2023 contains Philadelphia, and 2024
contains Austin/San Antonio. This script therefore treats within-city
chronological splits as the main temporal robustness test and keeps year
holdout as a clearly labeled confounded audit.
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
MIN_TRAIN_EVENTS = 12
MIN_TEST_EVENTS = 6

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
    output_dir = root / "results" / "temporal_generalization"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    event_table = build_event_table(tokens)
    split_summary = build_splits(event_table, tokens)
    validate_features(tokens)

    metrics, event_metrics = run_temporal_holdouts(tokens, split_summary)
    model_summary = summarize_models(metrics)
    gap_summary = build_gap_summary(metrics)
    diagnostics = build_diagnostics(model_summary, gap_summary, split_summary)

    write_table(event_table, table_dir / "temporal_split_assignments.csv")
    write_table(split_summary, table_dir / "temporal_split_summary.csv")
    write_table(metrics, table_dir / "temporal_model_metrics.csv")
    write_table(event_metrics, table_dir / "temporal_event_metrics.csv")
    write_table(model_summary, table_dir / "temporal_model_summary.csv")
    write_table(gap_summary, table_dir / "temporal_gap_summary.csv")
    (table_dir / "temporal_generalization_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(metrics, model_summary, gap_summary, figure_dir)
    write_report(
        report_dir / "temporal_generalization_report_zh.md",
        diagnostics,
        split_summary,
        model_summary,
        gap_summary,
    )
    print(f"Wrote temporal generalization analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    tokens = pd.read_csv(path)
    tokens = prepare_tokens(tokens)
    tokens["event_key"] = event_key(tokens)
    return tokens


def build_event_table(tokens: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "city",
        "event_id",
        "event_start",
        "event_total_precip",
        "event_peak_precip",
        "event_peak_positive_abnormal_deficit",
        "baseline_objective",
        "recoverable_fraction",
    ]
    events = tokens[keep].drop_duplicates(["city", "event_id"]).copy()
    events["event_start"] = pd.to_datetime(events["event_start"], errors="coerce")
    events["event_year"] = events["event_start"].dt.year.astype("Int64")
    events["event_month"] = events["event_start"].dt.to_period("M").astype(str)
    events["event_day"] = events["event_start"].dt.day
    events["event_hour"] = events["event_start"].dt.hour
    events = events.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)
    events["event_key"] = event_key(events)

    city_counts = events.groupby("city")["event_id"].transform("count")
    city_order = events.groupby("city").cumcount()
    events["within_city_event_count"] = city_counts.astype(int)
    events["within_city_order_index"] = city_order.astype(int)
    events["within_city_order_rank"] = (city_order + 0.5) / city_counts
    events["within_city_order_tertile"] = events["within_city_order_rank"].map(order_tertile)
    events["within_city_order_half"] = np.where(
        events["within_city_order_index"] < events["within_city_event_count"] / 2.0,
        "early_half",
        "late_half",
    )
    events["calendar_day_phase"] = events["event_day"].map(day_phase)
    events["calendar_year_group"] = "year_" + events["event_year"].astype(str)
    events["temporal_confounding_note"] = np.where(
        events["calendar_year_group"].eq("year_2019"),
        "year_2019_contains_chicago_dallas_houston_new_york",
        np.where(
            events["calendar_year_group"].eq("year_2023"),
            "year_2023_contains_philadelphia_only",
            "year_2024_contains_austin_san_antonio",
        ),
    )
    return events


def order_tertile(value: float) -> str:
    if value <= 1 / 3:
        return "early_tertile"
    if value <= 2 / 3:
        return "middle_tertile"
    return "late_tertile"


def day_phase(day: Any) -> str:
    if pd.isna(day):
        return "day_missing"
    day = int(day)
    if day <= 10:
        return "early_month"
    if day <= 20:
        return "middle_month"
    return "late_month"


def build_splits(events: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_keys = set(events["event_key"])
    token_counts = tokens.groupby("event_key").size()

    for column, role, description in [
        (
            "within_city_order_tertile",
            "main_within_city_chronological",
            "Hold out early/middle/late event order bins inside each city.",
        ),
        (
            "within_city_order_half",
            "main_within_city_chronological",
            "Hold out early or late half of each city's event sequence.",
        ),
        (
            "calendar_day_phase",
            "supporting_calendar_phase",
            "Hold out early/middle/late day-of-month phases.",
        ),
        (
            "calendar_year_group",
            "confounded_year_audit",
            "Hold out calendar years; this is city-confounded in the current data.",
        ),
    ]:
        for label in sorted(events[column].dropna().astype(str).unique()):
            if label.endswith("_missing"):
                continue
            test_keys = set(events.loc[events[column].astype(str).eq(label), "event_key"])
            train_keys = event_keys - test_keys
            rows.append(
                split_row(
                    events,
                    token_counts,
                    split_family=column,
                    heldout_period=label,
                    split_role=role,
                    train_rule=f"all_events_except_{label}",
                    description=description,
                    train_keys=train_keys,
                    test_keys=test_keys,
                )
            )

    for cutoff in [0.50, 0.67]:
        train_keys = set(events.loc[events["within_city_order_rank"] <= cutoff, "event_key"])
        test_keys = set(events.loc[events["within_city_order_rank"] > cutoff, "event_key"])
        rows.append(
            split_row(
                events,
                token_counts,
                split_family="within_city_forward_cut",
                heldout_period=f"after_{int(round(cutoff * 100))}pct",
                split_role="forward_chronological_audit",
                train_rule=f"train_within_city_order_rank_le_{cutoff:.2f}",
                description="Train on earlier events within each city and test on later events.",
                train_keys=train_keys,
                test_keys=test_keys,
            )
        )

    splits = pd.DataFrame(rows)
    splits = splits[
        (splits["n_train_events"] >= MIN_TRAIN_EVENTS)
        & (splits["n_test_events"] >= MIN_TEST_EVENTS)
        & (splits["n_train_tokens"] > 0)
        & (splits["n_test_tokens"] > 0)
    ].copy()
    return splits.sort_values(["split_role", "split_family", "heldout_period"]).reset_index(drop=True)


def split_row(
    events: pd.DataFrame,
    token_counts: pd.Series,
    *,
    split_family: str,
    heldout_period: str,
    split_role: str,
    train_rule: str,
    description: str,
    train_keys: set[str],
    test_keys: set[str],
) -> dict[str, Any]:
    train_events = events[events["event_key"].isin(train_keys)]
    test_events = events[events["event_key"].isin(test_keys)]
    return {
        "split_family": split_family,
        "heldout_period": heldout_period,
        "split_role": split_role,
        "train_rule": train_rule,
        "description": description,
        "train_event_keys": ";".join(sorted(train_keys)),
        "test_event_keys": ";".join(sorted(test_keys)),
        "n_train_events": int(len(train_events)),
        "n_test_events": int(len(test_events)),
        "n_train_cities": int(train_events["city"].nunique()),
        "n_test_cities": int(test_events["city"].nunique()),
        "n_train_tokens": int(token_counts.reindex(list(train_keys), fill_value=0).sum()),
        "n_test_tokens": int(token_counts.reindex(list(test_keys), fill_value=0).sum()),
        "test_start": str(test_events["event_start"].min()),
        "test_end": str(test_events["event_start"].max()),
        "mean_test_total_rain": safe_float(test_events["event_total_precip"].mean()),
        "mean_test_peak_speed_impact": safe_float(test_events["event_peak_positive_abnormal_deficit"].mean()),
        "mean_test_baseline_loss": safe_float(test_events["baseline_objective"].mean()),
        "mean_test_recoverable_fraction": safe_float(test_events["recoverable_fraction"].mean()),
    }


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
        raise KeyError(f"Missing temporal-generalization features: {sorted(set(missing))}")


def run_temporal_holdouts(tokens: pd.DataFrame, split_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for split in split_summary.itertuples(index=False):
        train_keys = set(str(split.train_event_keys).split(";"))
        test_keys = set(str(split.test_event_keys).split(";"))
        train = tokens[tokens["event_key"].isin(train_keys)].copy()
        test_base = tokens[tokens["event_key"].isin(test_keys)].copy()
        if train.empty or test_base.empty:
            continue
        for spec in MODEL_SPECS:
            test = test_base.copy()
            if "features" in spec:
                features = list(spec["features"])
                model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
                predicted = np.expm1(predict_ridge(model, test[features])) / 1_000.0
            else:
                predicted = pd.to_numeric(test[str(spec["score_col"])], errors="coerce").fillna(0.0).clip(lower=0.0)
            test["predicted_value"] = predicted
            base = {
                "split_family": str(split.split_family),
                "heldout_period": str(split.heldout_period),
                "split_role": str(split.split_role),
                "model_id": spec["model_id"],
                "family": spec["family"],
                "description": spec["description"],
                "n_features": int(len(spec.get("features", []))),
                "n_train_events": int(split.n_train_events),
                "n_test_events": int(split.n_test_events),
                "n_train_cities": int(split.n_train_cities),
                "n_test_cities": int(split.n_test_cities),
            }
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, base))
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
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
        rows.append(
            {
                **base,
                "city": city,
                "event_id": int(event_id),
                "event_start": str(group["event_start"].iloc[0]) if "event_start" in group else "",
                "n_tokens": int(len(group)),
                "spearman": safe_float(group["target_value"].corr(group["predicted_value"], method="spearman")),
                "top_5pct_value_capture": metric["value_capture"],
                "top_5pct_ndcg": metric["ndcg"],
                "top_5pct_precision": metric["precision"],
            }
        )
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
    return {
        "value_capture": safe_div(float(chosen_values.sum()), float(ideal_values.sum())),
        "ndcg": safe_div(dcg, idcg),
        "precision": len(set(chosen.index) & set(ideal.index)) / k,
    }


def summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    role_filters = {
        "all": metrics,
        "main": metrics[metrics["split_role"].eq("main_within_city_chronological")],
        "forward": metrics[metrics["split_role"].eq("forward_chronological_audit")],
        "year_confounded": metrics[metrics["split_role"].eq("confounded_year_audit")],
    }
    for spec in MODEL_SPECS:
        for scope, group_all in role_filters.items():
            group = group_all[group_all["model_id"].eq(spec["model_id"])].copy()
            if group.empty:
                continue
            hardest = group.sort_values("top_5pct_value_capture").iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "model_id": spec["model_id"],
                    "family": spec["family"],
                    "description": spec["description"],
                    "n_splits": int(len(group)),
                    "mean_top5_capture": safe_float(group["top_5pct_value_capture"].mean()),
                    "median_top5_capture": safe_float(group["top_5pct_value_capture"].median()),
                    "min_top5_capture": safe_float(group["top_5pct_value_capture"].min()),
                    "mean_top5_ndcg": safe_float(group["top_5pct_ndcg"].mean()),
                    "mean_spearman": safe_float(group["spearman"].mean()),
                    "hardest_split_family": str(hardest["split_family"]),
                    "hardest_heldout_period": str(hardest["heldout_period"]),
                    "hardest_split_role": str(hardest["split_role"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "mean_top5_capture"], ascending=[True, False])


def build_gap_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("factorized_vs_full_additive", "R1_factorized_low_dim", "R2_full_additive"),
        ("full_interaction_vs_full_additive", "R3_full_interaction", "R2_full_additive"),
        ("factorized_vs_activated_law", "R1_factorized_low_dim", "H4_activated_law"),
        ("activated_law_vs_deficit_only", "H4_activated_law", "H1_deficit_only"),
        ("activated_law_vs_structure_only", "H4_activated_law", "H3_structure_only"),
    ]
    rows: list[dict[str, Any]] = []
    keys = ["split_family", "heldout_period", "split_role"]
    for comparison, left_id, right_id in comparisons:
        left = metrics[metrics["model_id"].eq(left_id)]
        right = metrics[metrics["model_id"].eq(right_id)]
        merged = left.merge(right, on=keys, suffixes=("_left", "_right"))
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "comparison": comparison,
                    "split_family": getattr(row, "split_family"),
                    "heldout_period": getattr(row, "heldout_period"),
                    "split_role": getattr(row, "split_role"),
                    "left_model": left_id,
                    "right_model": right_id,
                    "left_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_left")),
                    "right_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_right")),
                    "delta_top5_capture": safe_float(getattr(row, "top_5pct_value_capture_left"))
                    - safe_float(getattr(row, "top_5pct_value_capture_right")),
                    "left_spearman": safe_float(getattr(row, "spearman_left")),
                    "right_spearman": safe_float(getattr(row, "spearman_right")),
                    "delta_spearman": safe_float(getattr(row, "spearman_left"))
                    - safe_float(getattr(row, "spearman_right")),
                }
            )
    return pd.DataFrame(rows)


def build_diagnostics(
    model_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> dict[str, Any]:
    factorized_main = one_row(model_summary, scope="main", model_id="R1_factorized_low_dim")
    full_main = one_row(model_summary, scope="main", model_id="R2_full_additive")
    interaction_main = one_row(model_summary, scope="main", model_id="R3_full_interaction")
    activated_main = one_row(model_summary, scope="main", model_id="H4_activated_law")
    deficit_main = one_row(model_summary, scope="main", model_id="H1_deficit_only")
    factorized_forward = one_row(model_summary, scope="forward", model_id="R1_factorized_low_dim")
    full_forward = one_row(model_summary, scope="forward", model_id="R2_full_additive")
    factorized_year = one_row(model_summary, scope="year_confounded", model_id="R1_factorized_low_dim")
    full_year = one_row(model_summary, scope="year_confounded", model_id="R2_full_additive")
    factorized_vs_full_main = gap_summary[
        gap_summary["comparison"].eq("factorized_vs_full_additive")
        & gap_summary["split_role"].eq("main_within_city_chronological")
    ]
    activated_vs_deficit_main = gap_summary[
        gap_summary["comparison"].eq("activated_law_vs_deficit_only")
        & gap_summary["split_role"].eq("main_within_city_chronological")
    ]
    year_splits = split_summary[split_summary["split_role"].eq("confounded_year_audit")]
    return {
        "n_temporal_splits": safe_int(len(split_summary)),
        "n_main_within_city_splits": safe_int(
            (split_summary["split_role"].eq("main_within_city_chronological")).sum()
        ),
        "n_forward_splits": safe_int((split_summary["split_role"].eq("forward_chronological_audit")).sum()),
        "n_year_confounded_splits": safe_int((split_summary["split_role"].eq("confounded_year_audit")).sum()),
        "factorized_main_mean_top5_capture": safe_float(factorized_main.get("mean_top5_capture")),
        "factorized_main_min_top5_capture": safe_float(factorized_main.get("min_top5_capture")),
        "factorized_main_mean_spearman": safe_float(factorized_main.get("mean_spearman")),
        "factorized_main_hardest_split_family": str(factorized_main.get("hardest_split_family", "")),
        "factorized_main_hardest_heldout": str(factorized_main.get("hardest_heldout_period", "")),
        "full_main_mean_top5_capture": safe_float(full_main.get("mean_top5_capture")),
        "full_main_min_top5_capture": safe_float(full_main.get("min_top5_capture")),
        "interaction_main_mean_top5_capture": safe_float(interaction_main.get("mean_top5_capture")),
        "activated_main_mean_top5_capture": safe_float(activated_main.get("mean_top5_capture")),
        "deficit_main_mean_top5_capture": safe_float(deficit_main.get("mean_top5_capture")),
        "factorized_minus_full_main_mean_top5_delta": safe_float(factorized_vs_full_main["delta_top5_capture"].mean())
        if not factorized_vs_full_main.empty
        else np.nan,
        "activated_minus_deficit_main_mean_top5_delta": safe_float(
            activated_vs_deficit_main["delta_top5_capture"].mean()
        )
        if not activated_vs_deficit_main.empty
        else np.nan,
        "factorized_forward_mean_top5_capture": safe_float(factorized_forward.get("mean_top5_capture")),
        "factorized_forward_min_top5_capture": safe_float(factorized_forward.get("min_top5_capture")),
        "full_forward_mean_top5_capture": safe_float(full_forward.get("mean_top5_capture")),
        "factorized_year_confounded_mean_top5_capture": safe_float(factorized_year.get("mean_top5_capture")),
        "factorized_year_confounded_min_top5_capture": safe_float(factorized_year.get("min_top5_capture")),
        "full_year_confounded_mean_top5_capture": safe_float(full_year.get("mean_top5_capture")),
        "year_holdout_design_note": "Calendar year is city-confounded: "
        + "; ".join(
            f"{row.heldout_period} tests {row.n_test_events} events from {row.n_test_cities} city group(s)"
            for row in year_splits.itertuples(index=False)
        ),
    }


def make_figures(
    metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_model_summary_figure(model_summary, figure_dir / "temporal_model_summary.png")
    make_capture_heatmap(metrics, figure_dir / "temporal_top5_capture_heatmap.png")
    make_gap_figure(gap_summary, figure_dir / "temporal_generalization_gaps.png")


def make_model_summary_figure(summary: pd.DataFrame, path: Path) -> None:
    main = summary[summary["scope"].eq("main")].sort_values("mean_top5_capture", ascending=True).copy()
    if main.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    colors = main["family"].map({"heuristic": "#94a3b8", "direct_law": "#0f766e", "trained_surrogate": "#2563eb"})
    ax.barh(main["model_id"], main["mean_top5_capture"], color=colors.fillna("#64748b"))
    ax.errorbar(
        main["mean_top5_capture"],
        main["model_id"],
        xerr=main["mean_top5_capture"] - main["min_top5_capture"],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.0,
        capsize=3,
        alpha=0.65,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean within-city temporal-holdout top-5% value capture")
    ax.set_title("Temporal robustness of recovery-value laws")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_capture_heatmap(metrics: pd.DataFrame, path: Path) -> None:
    keep = metrics[metrics["model_id"].isin(["H4_activated_law", "R1_factorized_low_dim", "R2_full_additive"])].copy()
    if keep.empty:
        return
    keep["split"] = keep["split_role"].str.replace("_", " ", regex=False) + ":" + keep["heldout_period"]
    pivot = keep.pivot_table(index="split", columns="model_id", values="top_5pct_value_capture", aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8.8, max(5.0, 0.36 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Top-5% capture by temporal holdout")
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
    if keep.empty:
        return
    keep["split"] = keep["split_role"].str.replace("_", " ", regex=False) + ":" + keep["heldout_period"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, max(5.2, 0.30 * keep["split"].nunique())), sharey=True)
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
    gap_rollup = (
        gap_summary.groupby(["comparison", "split_role"], as_index=False)
        .agg(
            n_splits=("delta_top5_capture", "count"),
            mean_delta_top5_capture=("delta_top5_capture", "mean"),
            min_delta_top5_capture=("delta_top5_capture", "min"),
            max_delta_top5_capture=("delta_top5_capture", "max"),
            mean_delta_spearman=("delta_spearman", "mean"),
        )
        .sort_values(["comparison", "split_role"])
    )
    lines = [
        "# Temporal Generalization V25",
        "",
        "## 这一版做了什么",
        "",
        "V25 专门补 high-level idea 中的 leave-time-period-out validation。由于当前样本的年份和城市高度绑定，严格的 year-based holdout 会同时测试城市外推和时间外推，不能作为干净时间泛化证据。因此主检验改为城市内部事件顺序留出：在每个城市内部按事件发生顺序划分 early/middle/late 或 early/late，再把某个时期整体留出。另设 forward chronological audit 和 confounded year audit，但论文中必须明确它们的证据强度不同。",
        "",
        "## 主要结论",
        "",
        f"- temporal splits = {diagnostics['n_temporal_splits']}，其中 within-city main splits = {diagnostics['n_main_within_city_splits']}，forward audits = {diagnostics['n_forward_splits']}，year-confounded audits = {diagnostics['n_year_confounded_splits']}。",
        f"- 主检验中，低维 factorized law mean top-5% capture = {diagnostics['factorized_main_mean_top5_capture']:.4f}，minimum = {diagnostics['factorized_main_min_top5_capture']:.4f}；最难 split = {diagnostics['factorized_main_hardest_split_family']} / {diagnostics['factorized_main_hardest_heldout']}。",
        f"- full additive surrogate mean top-5% capture = {diagnostics['full_main_mean_top5_capture']:.4f}，minimum = {diagnostics['full_main_min_top5_capture']:.4f}；full interaction mean = {diagnostics['interaction_main_mean_top5_capture']:.4f}。",
        f"- activated hand law mean top-5% capture = {diagnostics['activated_main_mean_top5_capture']:.4f}，deficit-only mean = {diagnostics['deficit_main_mean_top5_capture']:.4f}，activated minus deficit-only mean delta = {diagnostics['activated_minus_deficit_main_mean_top5_delta']:+.4f}。",
        f"- factorized minus full additive mean delta = {diagnostics['factorized_minus_full_main_mean_top5_delta']:+.4f}。如果它略低于 full additive，含义是 full model 捕捉到少量额外时间/事件上下文；如果仍接近，则说明 compact law 的主体结构没有依赖同一时期事件记忆。",
        f"- forward audit 中，factorized mean/min top-5% capture = {diagnostics['factorized_forward_mean_top5_capture']:.4f}/{diagnostics['factorized_forward_min_top5_capture']:.4f}，full additive mean = {diagnostics['full_forward_mean_top5_capture']:.4f}。",
        f"- year audit 中，factorized mean/min top-5% capture = {diagnostics['factorized_year_confounded_mean_top5_capture']:.4f}/{diagnostics['factorized_year_confounded_min_top5_capture']:.4f}，full additive mean = {diagnostics['full_year_confounded_mean_top5_capture']:.4f}。注意：{diagnostics['year_holdout_design_note']}。",
        "",
        "## 如何解读",
        "",
        "这不是最终的干净多年时间外推，因为每个城市只有一个主要速度覆盖月份；但它回答了一个更适合当前数据的问题：同一城市内不同事件顺序段被留出时，行动价值 law 是否仍能找到 top-tail recovery value。若 within-city temporal holdout 稳定，而 year audit 更弱或更强，都不能单独解释为时间效应，因为 year audit 混入了城市结构差异。",
        "",
        "因此论文里最稳妥的写法是：当前 law 已通过 leave-city、leave-event-regime 和 within-city chronological holdout；clean leave-time-period-out 仍需要同一城市跨多年、多月份事件数据。",
        "",
        "## Split Summary",
        "",
        table_to_markdown(split_summary.drop(columns=["train_event_keys", "test_event_keys"], errors="ignore")),
        "",
        "## Model Summary",
        "",
        table_to_markdown(model_summary),
        "",
        "## Gap Summary",
        "",
        table_to_markdown(gap_rollup),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def event_key(frame: pd.DataFrame) -> pd.Series:
    return frame["city"].astype(str) + "||" + pd.to_numeric(frame["event_id"], errors="coerce").fillna(-1).astype(int).astype(str)


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


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return safe_float(np.corrcoef(left, right)[0, 1])


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else float("nan")


if __name__ == "__main__":
    main()

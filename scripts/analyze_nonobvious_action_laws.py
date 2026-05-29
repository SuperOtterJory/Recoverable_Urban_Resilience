"""Analyze where simple action-ranking heuristics fail.

This script turns the high-level "activated bottleneck" idea into a
diagnostic evidence layer: highest deficit, highest exposure, and highest
structural leverage are compared with the optimizer-derived marginal recovery
value field inside each city-event.
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


EVENT_KEYS = ["city", "event_id"]
TARGET = "marginal_resource_value"
EPS = 1e-12

HEURISTICS = [
    {
        "heuristic": "deficit_only",
        "score_col": "deficit_only_score",
        "label": "Highest remaining deficit",
        "interpretation": "large local/access remaining loss only",
    },
    {
        "heuristic": "exposure_only",
        "score_col": "exposure_only_score",
        "label": "Highest OD exposure",
        "interpretation": "large demand exposure only",
    },
    {
        "heuristic": "structure_only",
        "score_col": "structure_only_score",
        "label": "Highest structural bottleneck",
        "interpretation": "OD importance with low outgoing degree only",
    },
    {
        "heuristic": "activated_law",
        "score_col": "activated_bottleneck_score",
        "label": "Activated law reference",
        "interpretation": "future loss x exposure x feasibility x efficiency",
    },
]

SIMPLE_HEURISTICS = ["deficit_only", "exposure_only", "structure_only"]
TOP_FRACS = {"top5": 0.05, "top20": 0.20}

REASON_COLUMNS = [
    ("delay_blocked", "delay_feasible", "le", 0.0),
    ("below_median_future_horizon", "active_weighted_horizon", "median", None),
    ("below_median_exposure", "law_exposure_term", "median", None),
    ("below_median_efficiency", "eta_per_cost", "median", None),
    ("short_remaining_window", "time_remaining_frac", "le", 0.50),
]

PERSISTENCE_FEATURES = [
    ("instantaneous_deficit", "passive_b_t"),
    ("peak_event_disturbance", "h_peak"),
    ("total_event_disturbance", "h_total"),
    ("remaining_local_area", "local_remaining_area"),
    ("remaining_access_area", "access_remaining_area"),
    ("active_weighted_horizon", "active_weighted_horizon"),
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "nonobvious_action_laws"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    tokens = prepare_tokens(tokens)

    event_metrics = build_event_heuristic_metrics(tokens)
    summary = build_heuristic_summary(event_metrics)
    city_summary = build_city_summary(event_metrics)
    failure_reasons = build_failure_reason_summary(tokens)
    hidden_summary, hidden_city_summary, hidden_examples = build_hidden_gem_tables(tokens)
    failure_examples = build_failure_examples(tokens)
    intervention_profile = build_intervention_profile(tokens)
    persistence = build_persistence_table(tokens)
    metrics = build_metrics(summary, hidden_summary, failure_reasons, persistence)

    write_table(event_metrics, table_dir / "heuristic_failure_event_metrics.csv")
    write_table(summary, table_dir / "heuristic_failure_summary.csv")
    write_table(city_summary, table_dir / "heuristic_failure_city_summary.csv")
    write_table(failure_reasons, table_dir / "heuristic_failure_reason_summary.csv")
    write_table(hidden_summary, table_dir / "hidden_gem_summary.csv")
    write_table(hidden_city_summary, table_dir / "hidden_gem_city_summary.csv")
    write_table(hidden_examples, table_dir / "hidden_gem_examples.csv")
    write_table(failure_examples, table_dir / "heuristic_failure_examples.csv")
    write_table(intervention_profile, table_dir / "intervention_top_value_profile.csv")
    write_table(persistence, table_dir / "persistence_vs_peak_summary.csv")
    (table_dir / "nonobvious_action_law_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(summary, city_summary, failure_reasons, hidden_city_summary, persistence, figure_dir)
    write_report(
        report_dir / "nonobvious_action_laws_report_zh.md",
        metrics,
        summary,
        city_summary,
        failure_reasons,
        hidden_summary,
        failure_examples,
        hidden_examples,
        intervention_profile,
        persistence,
    )
    print(f"Wrote non-obvious action-law analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    return pd.read_csv(path)


def prepare_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    out = tokens.copy()
    numeric_cols = {
        TARGET,
        "deficit_only_score",
        "exposure_only_score",
        "structure_only_score",
        "activated_bottleneck_score",
        "delay_feasible",
        "active_weighted_horizon",
        "active_future_loss_share",
        "law_exposure_term",
        "eta_per_cost",
        "time_remaining_frac",
        "passive_b_t",
        "passive_ell_t",
        "local_remaining_area",
        "access_remaining_area",
        "h_peak",
        "h_total",
        "origin_exposure",
        "destination_importance",
        "od_scarcity",
        "baseline_objective",
        "recoverable_fraction",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    }
    for col in numeric_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["target_value"] = out[TARGET].fillna(0.0).clip(lower=0.0)

    for spec in HEURISTICS:
        score_col = spec["score_col"]
        if score_col not in out:
            raise KeyError(f"Missing score column {score_col}")
        out[score_col] = out[score_col].fillna(0.0)
        out[f"{spec['heuristic']}_rank_pct"] = out.groupby(EVENT_KEYS)[score_col].rank(
            method="average",
            pct=True,
        )

    out["target_rank_pct"] = out.groupby(EVENT_KEYS)["target_value"].rank(method="average", pct=True)
    for flag_name, frac in TOP_FRACS.items():
        out[f"target_{flag_name}"] = mark_top_fraction(out, "target_value", frac)
        for spec in HEURISTICS:
            out[f"{spec['heuristic']}_{flag_name}"] = mark_top_fraction(out, spec["score_col"], frac)

    for reason, col, mode, value in REASON_COLUMNS:
        if mode == "median":
            median = out.groupby(EVENT_KEYS)[col].transform("median")
            out[f"reason_{reason}"] = out[col].fillna(0.0) <= median.fillna(0.0)
        elif mode == "le":
            out[f"reason_{reason}"] = out[col].fillna(0.0) <= float(value)
        else:
            raise ValueError(f"Unsupported reason mode {mode}")
    return out


def mark_top_fraction(frame: pd.DataFrame, score_col: str, frac: float) -> pd.Series:
    flags = pd.Series(False, index=frame.index)
    sort_cols = [score_col, "unit", "t", "intervention"]
    ascending = [False, True, True, True]
    for _, group in frame.groupby(EVENT_KEYS, sort=False):
        k = max(1, int(np.ceil(len(group) * frac)))
        selected = group.sort_values(sort_cols, ascending=ascending).head(k).index
        flags.loc[selected] = True
    return flags


def build_event_heuristic_metrics(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in tokens.groupby(EVENT_KEYS, sort=True):
        total_value = float(group["target_value"].sum())
        oracle_top5 = group["target_top5"]
        oracle_value = float(group.loc[oracle_top5, "target_value"].sum())
        target_top5_count = int(oracle_top5.sum())
        target_top20 = group["target_top20"]
        event_meta = event_metadata(group)
        for spec in HEURISTICS:
            heuristic = spec["heuristic"]
            top5 = group[f"{heuristic}_top5"]
            top20 = group[f"{heuristic}_top20"]
            top5_count = int(top5.sum())
            top5_value = float(group.loc[top5, "target_value"].sum())
            false_positive = top5 & ~target_top20
            zero_value = top5 & (group["target_value"] <= EPS)
            rows.append(
                {
                    "city": city,
                    "event_id": int(event_id),
                    **event_meta,
                    "heuristic": heuristic,
                    "heuristic_label": spec["label"],
                    "heuristic_interpretation": spec["interpretation"],
                    "candidate_action_count": int(len(group)),
                    "target_top5_count": target_top5_count,
                    "heuristic_top5_count": top5_count,
                    "total_value": total_value,
                    "oracle_top5_value": oracle_value,
                    "heuristic_top5_value": top5_value,
                    "top5_relative_to_oracle": safe_div(top5_value, oracle_value),
                    "top5_share_of_total_value": safe_div(top5_value, total_value),
                    "oracle_top5_share_of_total_value": safe_div(oracle_value, total_value),
                    "false_positive_count": int(false_positive.sum()),
                    "false_positive_share": safe_div(false_positive.sum(), top5_count),
                    "zero_value_count": int(zero_value.sum()),
                    "zero_value_share": safe_div(zero_value.sum(), top5_count),
                    "target_top5_hit_count": int((top5 & oracle_top5).sum()),
                    "target_top5_precision": safe_div((top5 & oracle_top5).sum(), top5_count),
                    "target_top5_recall": safe_div((top5 & oracle_top5).sum(), target_top5_count),
                    "target_top5_missed_by_top20_count": int((oracle_top5 & ~top20).sum()),
                    "target_top5_missed_by_top20_share": safe_div((oracle_top5 & ~top20).sum(), target_top5_count),
                }
            )
    return pd.DataFrame(rows)


def event_metadata(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    keep = [
        "event_start",
        "baseline_objective",
        "recoverable_fraction",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
        "weighted_b0",
        "weighted_h_total",
        "budget_fraction_of_baseline",
    ]
    return {col: first.get(col, np.nan) for col in keep if col in group.columns}


def build_heuristic_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return event_metrics
    rows: list[dict[str, Any]] = []
    for heuristic, group in event_metrics.groupby("heuristic", sort=False):
        top5_total = float(group["heuristic_top5_count"].sum())
        rows.append(
            {
                "heuristic": heuristic,
                "heuristic_label": group["heuristic_label"].iloc[0],
                "n_events": int(len(group)),
                "mean_top5_relative_to_oracle": float(group["top5_relative_to_oracle"].mean()),
                "median_top5_relative_to_oracle": float(group["top5_relative_to_oracle"].median()),
                "mean_top5_share_of_total_value": float(group["top5_share_of_total_value"].mean()),
                "mean_false_positive_share": float(group["false_positive_share"].mean()),
                "weighted_false_positive_share": safe_div(group["false_positive_count"].sum(), top5_total),
                "mean_zero_value_share": float(group["zero_value_share"].mean()),
                "weighted_zero_value_share": safe_div(group["zero_value_count"].sum(), top5_total),
                "mean_target_top5_precision": float(group["target_top5_precision"].mean()),
                "mean_target_top5_recall": float(group["target_top5_recall"].mean()),
                "mean_target_top5_missed_by_top20_share": float(group["target_top5_missed_by_top20_share"].mean()),
            }
        )
    order = {spec["heuristic"]: idx for idx, spec in enumerate(HEURISTICS)}
    return pd.DataFrame(rows).sort_values("heuristic", key=lambda s: s.map(order))


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return event_metrics
    summary = (
        event_metrics.groupby(["city", "heuristic"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            mean_top5_relative_to_oracle=("top5_relative_to_oracle", "mean"),
            mean_false_positive_share=("false_positive_share", "mean"),
            mean_zero_value_share=("zero_value_share", "mean"),
            mean_target_top5_missed_by_top20_share=("target_top5_missed_by_top20_share", "mean"),
            mean_recoverable_fraction=("recoverable_fraction", "mean"),
            mean_baseline_objective=("baseline_objective", "mean"),
        )
        .sort_values(["heuristic", "mean_false_positive_share"], ascending=[True, False])
    )
    return summary


def build_failure_reason_summary(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_top20 = tokens["target_top20"]
    reason_cols = [f"reason_{name}" for name, *_ in REASON_COLUMNS]
    for spec in HEURISTICS:
        heuristic = spec["heuristic"]
        if heuristic == "activated_law":
            continue
        top5 = tokens[f"{heuristic}_top5"]
        failure = tokens[top5 & ~target_top20].copy()
        row: dict[str, Any] = {
            "heuristic": heuristic,
            "heuristic_label": spec["label"],
            "failure_action_count": int(len(failure)),
        }
        for col in reason_cols:
            row[col.replace("reason_", "") + "_share"] = float(failure[col].mean()) if len(failure) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_hidden_gem_tables(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    simple_top5 = combine_rank_flags(tokens, SIMPLE_HEURISTICS, 0.95)
    simple_top20 = combine_rank_flags(tokens, SIMPLE_HEURISTICS, 0.80)
    target_top5 = tokens["target_top5"]
    hidden_top5 = target_top5 & ~simple_top5
    hard_hidden = target_top5 & ~simple_top20
    rows = [
        {
            "scope": "all_events",
            "target_top5_count": int(target_top5.sum()),
            "hidden_from_all_simple_top5_count": int(hidden_top5.sum()),
            "hidden_from_all_simple_top5_share": safe_div(hidden_top5.sum(), target_top5.sum()),
            "hidden_from_all_simple_top20_count": int(hard_hidden.sum()),
            "hidden_from_all_simple_top20_share": safe_div(hard_hidden.sum(), target_top5.sum()),
            "target_top5_low_deficit_top20_share": safe_div((target_top5 & (tokens["deficit_only_rank_pct"] < 0.80)).sum(), target_top5.sum()),
            "target_top5_low_exposure_top20_share": safe_div((target_top5 & (tokens["exposure_only_rank_pct"] < 0.80)).sum(), target_top5.sum()),
            "target_top5_low_structure_top20_share": safe_div((target_top5 & (tokens["structure_only_rank_pct"] < 0.80)).sum(), target_top5.sum()),
        }
    ]
    hidden_summary = pd.DataFrame(rows)

    city_rows: list[dict[str, Any]] = []
    for city, group in tokens.groupby("city", sort=True):
        city_target = group["target_top5"]
        city_simple_top5 = combine_rank_flags(group, SIMPLE_HEURISTICS, 0.95)
        city_simple_top20 = combine_rank_flags(group, SIMPLE_HEURISTICS, 0.80)
        city_rows.append(
            {
                "city": city,
                "target_top5_count": int(city_target.sum()),
                "hidden_from_all_simple_top5_share": safe_div((city_target & ~city_simple_top5).sum(), city_target.sum()),
                "hidden_from_all_simple_top20_share": safe_div((city_target & ~city_simple_top20).sum(), city_target.sum()),
                "target_top5_low_deficit_top20_share": safe_div((city_target & (group["deficit_only_rank_pct"] < 0.80)).sum(), city_target.sum()),
                "target_top5_low_exposure_top20_share": safe_div((city_target & (group["exposure_only_rank_pct"] < 0.80)).sum(), city_target.sum()),
                "target_top5_low_structure_top20_share": safe_div((city_target & (group["structure_only_rank_pct"] < 0.80)).sum(), city_target.sum()),
            }
        )
    hidden_city = pd.DataFrame(city_rows).sort_values("hidden_from_all_simple_top5_share", ascending=False)

    examples = tokens[hard_hidden].copy()
    hidden_definition = "target_top5_not_any_simple_top20"
    if examples.empty:
        examples = tokens[hidden_top5].copy()
        hidden_definition = "target_top5_not_any_simple_top5"
    examples["hidden_definition"] = hidden_definition
    examples = examples.sort_values("target_value", ascending=False).head(60)
    keep = ["hidden_definition", *example_columns()]
    return hidden_summary, hidden_city, examples[[col for col in keep if col in examples.columns]]


def combine_flags(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(False, index=frame.index)
    combined = pd.Series(False, index=frame.index)
    for col in cols:
        combined = combined | frame[col].astype(bool)
    return combined


def combine_rank_flags(frame: pd.DataFrame, heuristics: list[str], threshold: float) -> pd.Series:
    combined = pd.Series(False, index=frame.index)
    for heuristic in heuristics:
        combined = combined | (frame[f"{heuristic}_rank_pct"].fillna(0.0) >= threshold)
    return combined


def build_failure_examples(tokens: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in HEURISTICS:
        heuristic = spec["heuristic"]
        if heuristic == "activated_law":
            continue
        score_col = spec["score_col"]
        fail = tokens[tokens[f"{heuristic}_top5"] & ~tokens["target_top20"]].copy()
        if fail.empty:
            continue
        fail["heuristic"] = heuristic
        fail["heuristic_label"] = spec["label"]
        fail["heuristic_score"] = fail[score_col]
        fail["reason_tags"] = fail.apply(reason_tags, axis=1)
        fail = fail.sort_values(["heuristic_score", "target_value"], ascending=[False, True]).head(40)
        frames.append(fail)
    if not frames:
        return pd.DataFrame()
    examples = pd.concat(frames, ignore_index=True)
    keep = ["heuristic", "heuristic_label", "heuristic_score", *example_columns(), "reason_tags"]
    return examples[[col for col in keep if col in examples.columns]]


def example_columns() -> list[str]:
    return [
        "city",
        "event_id",
        "event_start",
        "unit",
        "t",
        "intervention",
        "target_value",
        "target_rank_pct",
        "delay_feasible",
        "active_weighted_horizon",
        "active_future_loss_share",
        "law_exposure_term",
        "eta_per_cost",
        "deficit_only_score",
        "deficit_only_rank_pct",
        "exposure_only_score",
        "exposure_only_rank_pct",
        "structure_only_score",
        "structure_only_rank_pct",
        "origin_exposure",
        "destination_importance",
        "od_scarcity",
        "baseline_objective",
        "recoverable_fraction",
    ]


def reason_tags(row: pd.Series) -> str:
    tags: list[str] = []
    for name, *_ in REASON_COLUMNS:
        if bool(row.get(f"reason_{name}", False)):
            tags.append(name)
    return ";".join(tags)


def build_intervention_profile(tokens: pd.DataFrame) -> pd.DataFrame:
    action_sets = {
        "target_top5": tokens["target_top5"],
        "hidden_target_top5": tokens["target_top5"]
        & ~combine_rank_flags(tokens, SIMPLE_HEURISTICS, 0.95),
        "deficit_top5": tokens["deficit_only_top5"],
        "exposure_top5": tokens["exposure_only_top5"],
        "structure_top5": tokens["structure_only_top5"],
    }
    rows: list[dict[str, Any]] = []
    for action_set, mask in action_sets.items():
        subset = tokens[mask].copy()
        total_count = max(len(subset), 1)
        total_value = float(subset["target_value"].sum())
        for intervention, group in subset.groupby("intervention", sort=True):
            rows.append(
                {
                    "action_set": action_set,
                    "intervention": intervention,
                    "action_count": int(len(group)),
                    "count_share": len(group) / total_count,
                    "target_value_share": safe_div(group["target_value"].sum(), total_value),
                    "mean_target_value": float(group["target_value"].mean()),
                    "mean_t": float(pd.to_numeric(group["t"], errors="coerce").mean()),
                    "mean_delay_feasible": float(group["delay_feasible"].mean()),
                    "mean_law_exposure_term": float(group["law_exposure_term"].mean()),
                    "mean_active_weighted_horizon": float(group["active_weighted_horizon"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_persistence_table(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name, col in PERSISTENCE_FEATURES:
        if col not in tokens:
            continue
        correlations: list[float] = []
        captures: list[float] = []
        for _, group in tokens.groupby(EVENT_KEYS, sort=True):
            if group[col].nunique(dropna=True) < 2 or group["target_value"].nunique(dropna=True) < 2:
                continue
            correlations.append(float(group["target_value"].corr(group[col], method="spearman")))
            k = max(1, int(np.ceil(len(group) * 0.05)))
            oracle = float(group.nlargest(k, "target_value")["target_value"].sum())
            chosen = float(group.nlargest(k, col)["target_value"].sum())
            captures.append(safe_div(chosen, oracle))
        rows.append(
            {
                "feature": feature_name,
                "column": col,
                "n_events": int(len(correlations)),
                "mean_event_spearman": float(np.nanmean(correlations)) if correlations else np.nan,
                "median_event_spearman": float(np.nanmedian(correlations)) if correlations else np.nan,
                "mean_top5_relative_to_oracle": float(np.nanmean(captures)) if captures else np.nan,
                "median_top5_relative_to_oracle": float(np.nanmedian(captures)) if captures else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_top5_relative_to_oracle", ascending=False)


def build_metrics(
    summary: pd.DataFrame,
    hidden_summary: pd.DataFrame,
    failure_reasons: pd.DataFrame,
    persistence: pd.DataFrame,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for heuristic in SIMPLE_HEURISTICS + ["activated_law"]:
        row = one_row(summary, heuristic=heuristic)
        metrics[f"{heuristic}_mean_top5_relative_to_oracle"] = safe_float(row.get("mean_top5_relative_to_oracle"))
        metrics[f"{heuristic}_mean_false_positive_share"] = safe_float(row.get("mean_false_positive_share"))
        metrics[f"{heuristic}_mean_zero_value_share"] = safe_float(row.get("mean_zero_value_share"))
        metrics[f"{heuristic}_mean_target_top5_missed_by_top20_share"] = safe_float(
            row.get("mean_target_top5_missed_by_top20_share")
        )
    hidden = hidden_summary.iloc[0] if not hidden_summary.empty else pd.Series(dtype=float)
    metrics["hidden_from_all_simple_top5_share"] = safe_float(hidden.get("hidden_from_all_simple_top5_share"))
    metrics["hidden_from_all_simple_top20_share"] = safe_float(hidden.get("hidden_from_all_simple_top20_share"))
    metrics["target_top5_low_deficit_top20_share"] = safe_float(hidden.get("target_top5_low_deficit_top20_share"))
    metrics["target_top5_low_exposure_top20_share"] = safe_float(hidden.get("target_top5_low_exposure_top20_share"))
    metrics["target_top5_low_structure_top20_share"] = safe_float(hidden.get("target_top5_low_structure_top20_share"))
    for heuristic in SIMPLE_HEURISTICS:
        row = one_row(failure_reasons, heuristic=heuristic)
        metrics[f"{heuristic}_failure_delay_blocked_share"] = safe_float(row.get("delay_blocked_share"))
        metrics[f"{heuristic}_failure_low_horizon_share"] = safe_float(row.get("below_median_future_horizon_share"))
        metrics[f"{heuristic}_failure_low_exposure_share"] = safe_float(row.get("below_median_exposure_share"))
    peak = one_row(persistence, feature="peak_event_disturbance")
    remaining = one_row(persistence, feature="remaining_local_area")
    active = one_row(persistence, feature="active_weighted_horizon")
    metrics["peak_disturbance_top5_relative_to_oracle"] = safe_float(peak.get("mean_top5_relative_to_oracle"))
    metrics["remaining_local_area_top5_relative_to_oracle"] = safe_float(remaining.get("mean_top5_relative_to_oracle"))
    metrics["active_horizon_top5_relative_to_oracle"] = safe_float(active.get("mean_top5_relative_to_oracle"))
    return metrics


def make_figures(
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    failure_reasons: pd.DataFrame,
    hidden_city: pd.DataFrame,
    persistence: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_false_positive_figure(summary, figure_dir / "heuristic_false_positive_rates.png")
    make_value_capture_figure(summary, figure_dir / "heuristic_value_capture.png")
    make_failure_reason_figure(failure_reasons, figure_dir / "failure_reason_breakdown.png")
    make_hidden_city_figure(hidden_city, figure_dir / "hidden_gem_city_share.png")
    make_persistence_figure(persistence, figure_dir / "persistence_vs_peak_capture.png")
    make_city_false_positive_figure(city_summary, figure_dir / "city_heuristic_failure_rates.png")


def make_false_positive_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[summary["heuristic"].isin(SIMPLE_HEURISTICS)].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["mean_false_positive_share"], width=0.36, color="#ef4444", label="outside target top-20%")
    ax.bar(x + 0.18, plot["mean_zero_value_share"], width=0.36, color="#f59e0b", label="zero marginal value")
    ax.set_xticks(x, plot["heuristic_label"], rotation=12, ha="right")
    ax.set_ylim(0, max(0.05, float(plot["mean_false_positive_share"].max()) * 1.22))
    ax.set_ylabel("Share among top-5% heuristic actions")
    ax.set_title("Simple high-score heuristics create false-positive recovery targets")
    ax.legend(frameon=False)
    for idx, value in enumerate(plot["mean_false_positive_share"]):
        ax.text(idx - 0.18, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_value_capture_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    colors = ["#64748b", "#0f766e", "#8b5cf6", "#2563eb"]
    ax.bar(plot["heuristic_label"], plot["mean_top5_relative_to_oracle"], color=colors[: len(plot)], width=0.62)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.55)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Value captured by top-5% / oracle top-5%")
    ax.set_title("Activation is stronger than one-factor action rankings")
    ax.tick_params(axis="x", rotation=12)
    for idx, value in enumerate(plot["mean_top5_relative_to_oracle"]):
        ax.text(idx, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_failure_reason_figure(failure_reasons: pd.DataFrame, path: Path) -> None:
    if failure_reasons.empty:
        return
    reason_cols = [col for col in failure_reasons.columns if col.endswith("_share") and col != "failure_action_count_share"]
    plot = failure_reasons.set_index("heuristic")[reason_cols].copy()
    if plot.empty:
        return
    labels = [col.replace("_share", "").replace("_", "\n") for col in reason_cols]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(plot.to_numpy(dtype=float), cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(reason_cols)), labels, fontsize=8)
    ax.set_yticks(np.arange(len(plot.index)), plot.index)
    ax.set_title("Why false-positive heuristic targets fail")
    for i in range(plot.shape[0]):
        for j in range(plot.shape[1]):
            value = plot.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="share of false positives")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_hidden_city_figure(hidden_city: pd.DataFrame, path: Path) -> None:
    if hidden_city.empty:
        return
    plot = hidden_city.sort_values("hidden_from_all_simple_top5_share", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(plot["city"], plot["hidden_from_all_simple_top5_share"], color="#0f766e")
    ax.set_xlim(0, max(0.05, float(plot["hidden_from_all_simple_top5_share"].max()) * 1.18))
    ax.set_xlabel("Share of target top-5% actions not in any simple top-5%")
    ax.set_title("High-value actions hidden from one-factor heuristics")
    for idx, value in enumerate(plot["hidden_from_all_simple_top5_share"]):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_persistence_figure(persistence: pd.DataFrame, path: Path) -> None:
    if persistence.empty:
        return
    plot = persistence.sort_values("mean_top5_relative_to_oracle", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(plot["feature"], plot["mean_top5_relative_to_oracle"], color="#2563eb")
    ax.set_xlim(0, max(0.05, float(plot["mean_top5_relative_to_oracle"].max()) * 1.18))
    ax.set_xlabel("Top-5% value captured / oracle top-5%")
    ax.set_title("Remaining-area signals versus peak or instantaneous deficit")
    for idx, value in enumerate(plot["mean_top5_relative_to_oracle"]):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_false_positive_figure(city_summary: pd.DataFrame, path: Path) -> None:
    plot = city_summary[city_summary["heuristic"].isin(SIMPLE_HEURISTICS)].copy()
    if plot.empty:
        return
    pivot = plot.pivot(index="city", columns="heuristic", values="mean_false_positive_share")
    pivot = pivot.sort_values("structure_only", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(pivot))
    width = 0.24
    colors = {"deficit_only": "#64748b", "exposure_only": "#0f766e", "structure_only": "#8b5cf6"}
    for offset, heuristic in zip([-width, 0, width], SIMPLE_HEURISTICS):
        if heuristic in pivot:
            ax.barh(y + offset, pivot[heuristic], height=width, label=heuristic, color=colors[heuristic])
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("Mean false-positive share")
    ax.set_title("Heuristic failure is city-structure dependent")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    failure_reasons: pd.DataFrame,
    hidden_summary: pd.DataFrame,
    failure_examples: pd.DataFrame,
    hidden_examples: pd.DataFrame,
    intervention_profile: pd.DataFrame,
    persistence: pd.DataFrame,
) -> None:
    deficit = one_row(summary, heuristic="deficit_only")
    exposure = one_row(summary, heuristic="exposure_only")
    structure = one_row(summary, heuristic="structure_only")
    hidden = hidden_summary.iloc[0] if not hidden_summary.empty else pd.Series(dtype=float)
    top_city_fail = (
        city_summary[city_summary["heuristic"].eq("structure_only")]
        .sort_values("mean_false_positive_share", ascending=False)
        .head(5)
        if not city_summary.empty
        else pd.DataFrame()
    )
    target_profile = intervention_profile[intervention_profile["action_set"].eq("target_top5")]
    lines = [
        "# Non-Obvious Action Law Analysis V15",
        "",
        "## 这一版做了什么",
        "",
        "本版本专门检验 high-level idea 中的非显然命题：最高速度损失、最高 OD 活跃度、最高结构瓶颈并不自动等于最高恢复优先级。分析单位是每个 city-event 内的 action token，比较 top 5% 简单启发式动作与 optimizer-derived marginal recovery value 的 top-tail。",
        "",
        "## 核心结论",
        "",
        f"- highest-deficit top-5% 动作平均只能捕获 oracle top-5% value 的 {safe_float(deficit.get('mean_top5_relative_to_oracle')):.3f}，其中 {safe_float(deficit.get('mean_false_positive_share')):.1%} 不在真实 value top-20%。",
        f"- highest-exposure top-5% 动作平均捕获 {safe_float(exposure.get('mean_top5_relative_to_oracle')):.3f} 的 oracle top-5% value，但仍有 {safe_float(exposure.get('mean_false_positive_share')):.1%} 是 false positive。",
        f"- structure-only top-5% 动作表现最不稳定，只捕获 {safe_float(structure.get('mean_top5_relative_to_oracle')):.3f} 的 oracle top-5% value，false-positive share 达 {safe_float(structure.get('mean_false_positive_share')):.1%}。",
        f"- 真实 target top-5% 动作中，{safe_float(hidden.get('hidden_from_all_simple_top5_share')):.1%} 没有出现在任一简单启发式的 top-5% 中；这说明高价值动作往往来自多因素共同激活，而不是单一指标最大。",
        f"- target top-5% 动作中，{safe_float(hidden.get('target_top5_low_structure_top20_share')):.1%} 甚至不在 structure-only top-20%，说明静态瓶颈必须被事件损失、OD 暴露、时间窗口和资源效率激活后才有恢复价值。",
        "",
        "## 失败原因",
        "",
        table_to_markdown(failure_reasons),
        "",
        "上表的各列可以重叠：同一个 false positive 可能同时因为响应延迟、未来损失窗口不足、OD exposure 不足或效率偏低而失败。它不是概率分解，而是失败机制诊断。",
        "",
        "## Heuristic Summary",
        "",
        table_to_markdown(summary),
        "",
        "## City Structure Differences",
        "",
        "structure-only 失败率最高的城市事件组如下，说明静态结构指标在不同城市里被事件激活的程度差异很大：",
        "",
        table_to_markdown(top_city_fail),
        "",
        "## Hidden High-Value Actions",
        "",
        table_to_markdown(hidden_summary),
        "",
        "## Intervention Mix",
        "",
        "target top-5% action 的资源类别分布如下。它描述的是高价值 action token 倾向于落在哪类 primitive 上，而不是最终 LP 的完整预算分配：",
        "",
        table_to_markdown(target_profile),
        "",
        "## Persistence Versus Peak",
        "",
        "当前事件标定把城市级 event signal 按 destination vulnerability 投影到 region，因此 `h_peak` 与 `h_total` 在空间上高度同源，二者不能严格区分“峰值冲击”和“累计冲击”。更有信息的是 action-time 层面的 remaining area 与 active horizon；它们直接进入未来可恢复损失窗口。",
        "",
        table_to_markdown(persistence),
        "",
        "## Failure Examples",
        "",
        table_to_markdown(failure_examples.head(30)),
        "",
        "## Hidden Gem Examples",
        "",
        table_to_markdown(hidden_examples.head(30)),
        "",
        "## 论文写作含义",
        "",
        "这版结果把 activated-bottleneck law 的非显然性补强了：城市结构不是一个静态 ranking，而是 latent leverage。只有当 persistent future loss、OD exposure、intervention feasibility 和 efficiency 同时重叠时，结构瓶颈才转化为 recovery value。论文里应避免写成“恢复最拥堵/最中心的区域”，而应写成“恢复被事件激活的、需求暴露且仍有未来可恢复损失的结构位置”。",
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
    if len(compact) > 40:
        compact = compact.head(40)
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

"""Analyze why small-signal action laws do not fully match finite-budget LP allocations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.paths import find_repo_root


INTERVENTIONS = ("R", "C", "S")
EVENT_KEYS = ["city", "event_id"]
ACTION_KEYS = ["city", "event_id", "unit", "t", "intervention"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/finite_budget_gap")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root)
    action_features = prepare_action_features(data["tokens"])
    greedy_segments = prepare_greedy_segments(data["greedy"])
    lp_segments, lp_match_summary = segmentize_lp_allocations(
        data["lp_interventions"],
        action_features,
        config,
    )
    event_metrics = build_event_gap_metrics(
        data["replay"],
        greedy_segments,
        lp_segments,
        lp_match_summary,
    )
    correlations = mechanism_correlations(event_metrics)
    city_summary = summarize_by_city(event_metrics)
    allocation_examples = top_allocation_differences(action_features, greedy_segments, lp_segments, event_metrics)

    write_table(event_metrics, table_dir / "finite_budget_gap_event_metrics.csv")
    write_table(city_summary, table_dir / "finite_budget_gap_city_summary.csv")
    write_table(correlations, table_dir / "finite_budget_gap_correlations.csv")
    write_table(allocation_examples, table_dir / "finite_budget_gap_top_action_differences.csv")
    write_table(lp_segments, table_dir / "lp_segmentized_allocations.csv.gz")
    write_table(greedy_segments, table_dir / "greedy_segment_allocations.csv.gz")

    make_figures(event_metrics, city_summary, figure_dir)
    write_report(
        report_dir / "finite_budget_gap_report_zh.md",
        event_metrics,
        city_summary,
        correlations,
        allocation_examples,
    )
    print(f"Wrote finite-budget gap analysis to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    law_tables = root / "results" / "law_learning" / "tables"
    return {
        "tokens": pd.read_csv(law_tables / "action_value_tokens.csv.gz"),
        "greedy": pd.read_csv(law_tables / "greedy_oracle_actions.csv.gz"),
        "replay": pd.read_csv(law_tables / "fixed_policy_replay.csv"),
        "lp_interventions": pd.read_csv(root / "results" / "event_optimization" / "tables" / "event_optimization_interventions.csv"),
    }


def prepare_action_features(tokens: pd.DataFrame) -> pd.DataFrame:
    cols = [
        *ACTION_KEYS,
        "event_start",
        "cost",
        "u_cap",
        "marginal_resource_value",
        "small_signal_marginal_value",
        "finite_deficit_area_value",
        "activated_bottleneck_score",
        "active_weighted_horizon",
        "active_future_loss_share",
        "origin_exposure_rank",
        "destination_importance_rank",
        "local_remaining_rank",
        "access_remaining_rank",
        "eta_per_cost_rank",
        "optimized_cost",
    ]
    available = [col for col in cols if col in tokens.columns]
    out = tokens[available].copy()
    normalize_key_columns(out)
    out = out.drop_duplicates(ACTION_KEYS, keep="first")
    return out


def prepare_greedy_segments(greedy: pd.DataFrame) -> pd.DataFrame:
    out = greedy.copy()
    normalize_key_columns(out)
    out["segment"] = pd.to_numeric(out["segment"], errors="coerce").fillna(0).astype(int)
    out["allocated_cost"] = pd.to_numeric(out["allocated_cost"], errors="coerce").fillna(0.0)
    out["value_proxy"] = pd.to_numeric(out["value_proxy"], errors="coerce").fillna(0.0)
    out["oracle_value_per_cost"] = pd.to_numeric(out["oracle_value_per_cost"], errors="coerce").fillna(0.0)
    out["allocation_source"] = "small_signal_greedy"
    return out[out["allocated_cost"] > 1e-12].copy()


def segmentize_lp_allocations(
    lp_interventions: pd.DataFrame,
    action_features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lp = lp_interventions.copy()
    lp = lp[lp["scenario"].eq("base")].copy()
    normalize_key_columns(lp)
    lp["u"] = pd.to_numeric(lp["u"], errors="coerce").fillna(0.0)
    lp["effective_cost"] = pd.to_numeric(lp["effective_cost"], errors="coerce").fillna(0.0)
    lp = lp[(lp["u"] > 1e-12) | (lp["effective_cost"] > 1e-12)].copy()
    feature_cols = [
        *ACTION_KEYS,
        "cost",
        "u_cap",
        "marginal_resource_value",
        "finite_deficit_area_value",
        "activated_bottleneck_score",
        "active_weighted_horizon",
    ]
    feature_cols = [col for col in feature_cols if col in action_features.columns]
    merged = lp.merge(action_features[feature_cols], on=ACTION_KEYS, how="left", suffixes=("", "_feature"))
    merged["feature_matched"] = merged["marginal_resource_value"].notna()
    match_summary = (
        merged.groupby(EVENT_KEYS, as_index=False)
        .agg(
            lp_action_rows=("u", "count"),
            lp_feature_matched_rows=("feature_matched", "sum"),
            lp_cost_total=("effective_cost", "sum"),
            lp_matched_cost=("effective_cost", lambda x: float(x[merged.loc[x.index, "feature_matched"]].sum())),
        )
    )
    match_summary["lp_feature_match_rate"] = match_summary["lp_feature_matched_rows"] / match_summary["lp_action_rows"].replace(0, np.nan)
    match_summary["lp_matched_cost_share"] = match_summary["lp_matched_cost"] / match_summary["lp_cost_total"].replace(0.0, np.nan)

    pwl = config["interventions"].get("pwl_diminishing_returns", {})
    if bool(pwl.get("enabled", False)):
        segment_shares = np.asarray(pwl["segment_cap_shares"], dtype=float)
        segment_shares = segment_shares / segment_shares.sum()
        multipliers = {key: np.asarray(pwl["effectiveness_multipliers"][key], dtype=float) for key in INTERVENTIONS}
    else:
        segment_shares = np.array([1.0], dtype=float)
        multipliers = {key: np.array([1.0], dtype=float) for key in INTERVENTIONS}

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        remaining_u = max(float(getattr(row, "u")), 0.0)
        if remaining_u <= 1e-12:
            continue
        cost = finite_float(getattr(row, "cost", np.nan), finite_float(getattr(row, "cost_feature", np.nan), np.nan))
        if not np.isfinite(cost) or cost <= 0:
            cost = float(getattr(row, "effective_cost", 0.0)) / max(remaining_u, 1e-12)
        u_cap = finite_float(getattr(row, "u_cap", np.nan), remaining_u)
        value_per_cost = finite_float(getattr(row, "marginal_resource_value", np.nan), 0.0)
        for segment_id, share in enumerate(segment_shares):
            if remaining_u <= 1e-12:
                break
            cap_u = max(float(u_cap) * float(share), 0.0)
            used_u = min(remaining_u, cap_u) if cap_u > 0 else remaining_u
            if used_u <= 1e-12:
                continue
            multiplier = float(multipliers[str(row.intervention)][segment_id])
            allocated_cost = used_u * cost
            rows.append(
                {
                    "city": row.city,
                    "event_id": int(row.event_id),
                    "event_start": getattr(row, "event_start", ""),
                    "scenario": "base",
                    "unit": str(row.unit),
                    "t": int(row.t),
                    "intervention": str(row.intervention),
                    "segment": int(segment_id),
                    "allocated_cost": float(allocated_cost),
                    "allocated_u": float(used_u),
                    "value_proxy": float(allocated_cost * value_per_cost * multiplier),
                    "oracle_value_per_cost": float(value_per_cost * multiplier),
                    "law_value_score": float(finite_float(getattr(row, "activated_bottleneck_score", np.nan), 0.0) * multiplier),
                    "segment_effectiveness_multiplier": multiplier,
                    "finite_deficit_area_value": finite_float(getattr(row, "finite_deficit_area_value", np.nan), 0.0),
                    "active_weighted_horizon": finite_float(getattr(row, "active_weighted_horizon", np.nan), 0.0),
                    "allocation_source": "lp_optimizer",
                }
            )
            remaining_u -= used_u
    return pd.DataFrame(rows), match_summary


def build_event_gap_metrics(
    replay: pd.DataFrame,
    greedy_segments: pd.DataFrame,
    lp_segments: pd.DataFrame,
    lp_match_summary: pd.DataFrame,
) -> pd.DataFrame:
    base = replay[replay["policy_scenario"].eq("base")].copy()
    lp_replay = base[base["policy_score"].eq("lp_optimizer_replay")].copy()
    greedy_replay = base[base["policy_score"].eq("greedy_oracle")].copy()
    replay_cols = [
        *EVENT_KEYS,
        "event_start",
        "baseline_objective",
        "optimized_objective",
        "lp_recoverable_fraction",
        "replay_gain",
        "replay_recoverable_fraction",
        "replay_fraction_of_base_lp_gain",
        "selected_action_count",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    lp_replay = lp_replay[replay_cols].rename(
        columns={
            "replay_gain": "lp_replay_gain",
            "replay_recoverable_fraction": "lp_replay_recoverable_fraction",
            "selected_action_count": "lp_selected_action_count",
        }
    )
    greedy_replay = greedy_replay[replay_cols].rename(
        columns={
            "replay_gain": "greedy_replay_gain",
            "replay_recoverable_fraction": "greedy_replay_recoverable_fraction",
            "replay_fraction_of_base_lp_gain": "greedy_fraction_of_lp_gain",
            "selected_action_count": "greedy_selected_action_count",
        }
    )
    metrics = lp_replay.merge(greedy_replay, on=[*EVENT_KEYS, "event_start"], suffixes=("", "_greedy"))
    metrics["replay_gap_recoverable_fraction"] = metrics["lp_replay_recoverable_fraction"] - metrics["greedy_replay_recoverable_fraction"]
    metrics["replay_gap_fraction_of_lp_gain"] = 1.0 - metrics["greedy_fraction_of_lp_gain"]

    lp_summary = allocation_summary(lp_segments, "lp")
    greedy_summary = allocation_summary(greedy_segments, "greedy")
    overlap = allocation_overlap(lp_segments, greedy_segments)
    metrics = metrics.merge(lp_summary, on=EVENT_KEYS, how="left")
    metrics = metrics.merge(greedy_summary, on=EVENT_KEYS, how="left")
    metrics = metrics.merge(overlap, on=EVENT_KEYS, how="left")
    metrics = metrics.merge(lp_match_summary, on=EVENT_KEYS, how="left")
    metrics = add_difference_metrics(metrics)
    return metrics.sort_values(["replay_gap_fraction_of_lp_gain", "replay_gap_recoverable_fraction"], ascending=False)


def allocation_summary(segments: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=EVENT_KEYS)
    rows = []
    for key, group in segments.groupby(EVENT_KEYS):
        total_cost = float(group["allocated_cost"].sum())
        row: dict[str, Any] = {
            "city": key[0],
            "event_id": int(key[1]),
            f"{prefix}_allocated_cost": total_cost,
            f"{prefix}_first_order_value_proxy": float(group["value_proxy"].sum()),
            f"{prefix}_segment_rows": int(len(group)),
            f"{prefix}_action_count": int(group[ACTION_KEYS].drop_duplicates().shape[0]),
            f"{prefix}_unit_count": int(group[["city", "event_id", "unit"]].drop_duplicates().shape[0]),
            f"{prefix}_cost_gini": gini(group["allocated_cost"].to_numpy(dtype=float)),
        }
        for intervention in INTERVENTIONS:
            share = cost_share(group, group["intervention"].eq(intervention), total_cost)
            row[f"{prefix}_cost_share_{intervention}"] = share
        for segment in sorted(group["segment"].dropna().astype(int).unique()):
            row[f"{prefix}_cost_share_segment_{segment}"] = cost_share(group, group["segment"].astype(int).eq(segment), total_cost)
        for name, mask in time_bin_masks(group["t"].to_numpy(dtype=int)).items():
            row[f"{prefix}_cost_share_time_{name}"] = cost_share(group, mask, total_cost)
        rows.append(row)
    return pd.DataFrame(rows)


def allocation_overlap(lp_segments: pd.DataFrame, greedy_segments: pd.DataFrame) -> pd.DataFrame:
    lp = lp_segments.groupby(ACTION_KEYS, as_index=False).agg(lp_action_cost=("allocated_cost", "sum"))
    greedy = greedy_segments.groupby(ACTION_KEYS, as_index=False).agg(greedy_action_cost=("allocated_cost", "sum"))
    merged = lp.merge(greedy, on=ACTION_KEYS, how="outer").fillna({"lp_action_cost": 0.0, "greedy_action_cost": 0.0})
    merged["overlap_cost"] = np.minimum(merged["lp_action_cost"], merged["greedy_action_cost"])
    rows = []
    for key, group in merged.groupby(EVENT_KEYS):
        lp_cost = float(group["lp_action_cost"].sum())
        greedy_cost = float(group["greedy_action_cost"].sum())
        overlap_cost = float(group["overlap_cost"].sum())
        union_cost = lp_cost + greedy_cost - overlap_cost
        rows.append(
            {
                "city": key[0],
                "event_id": int(key[1]),
                "action_cost_overlap_share_lp": overlap_cost / lp_cost if lp_cost > 0 else np.nan,
                "action_cost_overlap_share_greedy": overlap_cost / greedy_cost if greedy_cost > 0 else np.nan,
                "action_cost_jaccard": overlap_cost / union_cost if union_cost > 0 else np.nan,
                "lp_only_cost_share": max(lp_cost - overlap_cost, 0.0) / lp_cost if lp_cost > 0 else np.nan,
                "greedy_only_cost_share": max(greedy_cost - overlap_cost, 0.0) / greedy_cost if greedy_cost > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_difference_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["greedy_proxy_over_realized"] = out["greedy_first_order_value_proxy"] - out["greedy_replay_recoverable_fraction"]
    out["lp_proxy_over_realized"] = out["lp_first_order_value_proxy"] - out["lp_replay_recoverable_fraction"]
    out["greedy_minus_lp_proxy"] = out["greedy_first_order_value_proxy"] - out["lp_first_order_value_proxy"]
    out["lp_to_greedy_action_count_ratio"] = out["lp_action_count"] / out["greedy_action_count"].replace(0, np.nan)
    out["lp_to_greedy_unit_count_ratio"] = out["lp_unit_count"] / out["greedy_unit_count"].replace(0, np.nan)
    out["intervention_share_l1"] = l1_share_distance(out, [f"cost_share_{k}" for k in INTERVENTIONS])
    out["time_share_l1"] = l1_share_distance(out, ["cost_share_time_early", "cost_share_time_mid", "cost_share_time_late"])
    out["segment_share_l1"] = l1_share_distance(out, ["cost_share_segment_0", "cost_share_segment_1", "cost_share_segment_2"])
    return out


def l1_share_distance(frame: pd.DataFrame, suffixes: list[str]) -> pd.Series:
    distance = pd.Series(0.0, index=frame.index)
    for suffix in suffixes:
        lp_col = f"lp_{suffix}"
        greedy_col = f"greedy_{suffix}"
        if lp_col in frame and greedy_col in frame:
            distance += (frame[lp_col].fillna(0.0) - frame[greedy_col].fillna(0.0)).abs()
    return 0.5 * distance


def mechanism_correlations(event_metrics: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["replay_gap_fraction_of_lp_gain", "replay_gap_recoverable_fraction"]
    feature_cols = [
        "action_cost_jaccard",
        "lp_only_cost_share",
        "greedy_only_cost_share",
        "intervention_share_l1",
        "time_share_l1",
        "segment_share_l1",
        "greedy_proxy_over_realized",
        "lp_proxy_over_realized",
        "greedy_minus_lp_proxy",
        "lp_to_greedy_action_count_ratio",
        "lp_to_greedy_unit_count_ratio",
        "lp_cost_gini",
        "greedy_cost_gini",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    rows = []
    for target in target_cols:
        for feature in feature_cols:
            if target not in event_metrics or feature not in event_metrics:
                continue
            x = event_metrics[feature]
            y = event_metrics[target]
            if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                corr = np.nan
            else:
                corr = x.corr(y, method="spearman")
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman": corr,
                    "abs_spearman": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "abs_spearman"], ascending=[True, False])


def summarize_by_city(event_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "lp_replay_recoverable_fraction",
        "greedy_replay_recoverable_fraction",
        "greedy_fraction_of_lp_gain",
        "replay_gap_fraction_of_lp_gain",
        "action_cost_jaccard",
        "intervention_share_l1",
        "time_share_l1",
        "segment_share_l1",
        "greedy_proxy_over_realized",
        "lp_to_greedy_action_count_ratio",
        "lp_cost_gini",
        "greedy_cost_gini",
    ]
    summary = event_metrics.groupby("city", as_index=False).agg(
        n_events=("event_id", "count"),
        **{f"mean_{col}": (col, "mean") for col in numeric_cols if col in event_metrics},
    )
    return summary.sort_values("mean_replay_gap_fraction_of_lp_gain", ascending=False)


def top_allocation_differences(
    action_features: pd.DataFrame,
    greedy_segments: pd.DataFrame,
    lp_segments: pd.DataFrame,
    event_metrics: pd.DataFrame,
    n_events: int = 8,
    n_actions_per_event: int = 20,
) -> pd.DataFrame:
    top_events = event_metrics.head(n_events)[EVENT_KEYS]
    lp = lp_segments.groupby(ACTION_KEYS, as_index=False).agg(lp_cost=("allocated_cost", "sum"))
    greedy = greedy_segments.groupby(ACTION_KEYS, as_index=False).agg(greedy_cost=("allocated_cost", "sum"))
    merged = lp.merge(greedy, on=ACTION_KEYS, how="outer").fillna({"lp_cost": 0.0, "greedy_cost": 0.0})
    merged = merged.merge(top_events, on=EVENT_KEYS, how="inner")
    merged["absolute_cost_difference"] = (merged["lp_cost"] - merged["greedy_cost"]).abs()
    merged["allocation_preference"] = np.where(merged["lp_cost"] > merged["greedy_cost"], "lp_more", "greedy_more")
    feature_cols = [
        *ACTION_KEYS,
        "marginal_resource_value",
        "finite_deficit_area_value",
        "active_weighted_horizon",
        "origin_exposure_rank",
        "destination_importance_rank",
        "local_remaining_rank",
        "access_remaining_rank",
        "eta_per_cost_rank",
    ]
    available = [col for col in feature_cols if col in action_features.columns]
    merged = merged.merge(action_features[available], on=ACTION_KEYS, how="left")
    return (
        merged.sort_values([*EVENT_KEYS, "absolute_cost_difference"], ascending=[True, True, False])
        .groupby(EVENT_KEYS, as_index=False)
        .head(n_actions_per_event)
        .reset_index(drop=True)
    )


def make_figures(event_metrics: pd.DataFrame, city_summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(
        event_metrics["action_cost_jaccard"],
        event_metrics["greedy_fraction_of_lp_gain"],
        c=event_metrics["replay_gap_fraction_of_lp_gain"],
        cmap="magma_r",
        s=58,
        alpha=0.82,
    )
    ax.set_xlabel("LP-greedy action-cost Jaccard overlap")
    ax.set_ylabel("Greedy replay gain / LP gain")
    ax.set_title("Finite-budget gap and allocation overlap")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "gap_vs_allocation_overlap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ordered = city_summary.sort_values("mean_greedy_fraction_of_lp_gain")
    ax.barh(ordered["city"], ordered["mean_greedy_fraction_of_lp_gain"], color="#2563eb")
    ax.axvline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean greedy replay gain / LP gain")
    ax.set_title("Finite-budget law gap by city")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "gap_by_city.png", dpi=180)
    plt.close(fig)

    mean_shares = []
    for source in ["lp", "greedy"]:
        for intervention in INTERVENTIONS:
            col = f"{source}_cost_share_{intervention}"
            if col in event_metrics:
                mean_shares.append({"source": source, "category": intervention, "share": event_metrics[col].mean(), "panel": "intervention"})
        for segment in [0, 1, 2]:
            col = f"{source}_cost_share_segment_{segment}"
            if col in event_metrics:
                mean_shares.append({"source": source, "category": f"segment {segment}", "share": event_metrics[col].mean(), "panel": "segment"})
        for time_bin in ["early", "mid", "late"]:
            col = f"{source}_cost_share_time_{time_bin}"
            if col in event_metrics:
                mean_shares.append({"source": source, "category": time_bin, "share": event_metrics[col].mean(), "panel": "time"})
    share_df = pd.DataFrame(mean_shares)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for ax, panel in zip(axes, ["intervention", "segment", "time"], strict=True):
        sub = share_df[share_df["panel"].eq(panel)].copy()
        categories = sub["category"].drop_duplicates().tolist()
        x = np.arange(len(categories))
        width = 0.35
        for offset, source in [(-width / 2, "lp"), (width / 2, "greedy")]:
            vals = sub[sub["source"].eq(source)].set_index("category")["share"].reindex(categories).fillna(0.0)
            ax.bar(x + offset, vals, width=width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=20)
        ax.set_ylim(0, max(0.05, float(sub["share"].max()) * 1.2))
        ax.set_title(panel.title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean cost share")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "lp_vs_greedy_allocation_mix.png", dpi=180)
    plt.close(fig)

    top = event_metrics.head(12).copy()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels = top["city"] + " #" + top["event_id"].astype(str)
    x = np.arange(len(top))
    width = 0.38
    ax.bar(x - width / 2, top["lp_replay_recoverable_fraction"], width=width, label="LP", color="#0f766e")
    ax.bar(x + width / 2, top["greedy_replay_recoverable_fraction"], width=width, label="Small-signal greedy", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Recoverable fraction")
    ax.set_title("Largest finite-budget gaps")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "top_gap_events.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    event_metrics: pd.DataFrame,
    city_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    allocation_examples: pd.DataFrame,
) -> None:
    base_gain = float(event_metrics["greedy_fraction_of_lp_gain"].mean())
    median_gain = float(event_metrics["greedy_fraction_of_lp_gain"].median())
    mean_gap = float(event_metrics["replay_gap_fraction_of_lp_gain"].mean())
    mean_jaccard = float(event_metrics["action_cost_jaccard"].mean())
    mean_proxy_over = float(event_metrics["greedy_proxy_over_realized"].mean())
    mean_lp_spread = float(event_metrics["lp_to_greedy_action_count_ratio"].mean())
    top_corr = correlations[correlations["target"].eq("replay_gap_fraction_of_lp_gain")].head(10)
    top_events = event_metrics.head(12)[
        [
            "city",
            "event_id",
            "greedy_fraction_of_lp_gain",
            "replay_gap_fraction_of_lp_gain",
            "action_cost_jaccard",
            "intervention_share_l1",
            "time_share_l1",
            "segment_share_l1",
            "greedy_proxy_over_realized",
            "lp_to_greedy_action_count_ratio",
        ]
    ]
    lines = [
        "# Finite-Budget Allocation Gap V6",
        "",
        "## 这一版回答什么问题",
        "",
        "V5 已经说明：对 single-action 第一小段资源，small-signal activated law 与 LP 一阶边际值基本一致。但完整预算下，按一阶值贪心分配的 fixed-policy replay 仍低于 LP optimum。因此 V6 专门分析这个剩余差距，目标是从 action-level marginal law 进入 finite-budget allocation law。",
        "",
        "核心结论是：一阶边际排序解决的是“第一单位资源投向哪里最值”；LP optimum 解决的是“在总预算、单期预算、部署上限、diminishing returns 和 R/C/S 互相替代下，整组资源如何组合”。二者不是同一个 law。",
        "",
        "## 总体结果",
        "",
        f"- small-signal greedy 平均获得 LP gain 的 {base_gain:.4f}，中位数为 {median_gain:.4f}",
        f"- 平均剩余 gap 为 LP gain 的 {mean_gap:.4f}",
        f"- LP 与 greedy 的 action-cost Jaccard overlap 平均为 {mean_jaccard:.4f}",
        f"- greedy first-order proxy 平均比真实 replay recoverable fraction 高 {mean_proxy_over:.4f}",
        f"- LP 使用的 action 数约为 greedy 的 {mean_lp_spread:.4f} 倍",
        "",
        "## 机制解释",
        "",
        "1. **small-signal proxy 会高估有限预算收益。** 它假设每个 segment 的边际效果都在 passive trajectory 上独立发挥作用；但当多个资源作用到相同或相邻的 loss channel 时，后投放的资源会被已经降低的 `b`、`d`、`ell` 截断。",
        "",
        "2. **LP 比一阶 greedy 更会分散组合。** Greedy 会持续吃掉当前最高一阶值 segment；LP 会在容量和互相替代作用下，把资源扩散到更多 action，以避免局部饱和。",
        "",
        "3. **剩余 law 不是新的局部一阶 law，而是 allocation interaction law。** 它需要刻画同一 unit/time 上 R、C、S 的替代关系、同一未来 loss channel 被多次处理后的边际下降、以及 period budget 导致的跨时间挤出。",
        "",
        "## Gap 相关性最高的机制变量",
        "",
        dataframe_to_markdown(top_corr),
        "",
        "## 城市层面汇总",
        "",
        dataframe_to_markdown(city_summary),
        "",
        "## Gap 最大事件",
        "",
        dataframe_to_markdown(top_events),
        "",
        "## 高差距事件中的 allocation 差异样例",
        "",
        dataframe_to_markdown(allocation_examples.head(40)),
        "",
        "## 下一步",
        "",
        "下一版应把这组发现反馈到 policy construction：不再只按 first-order value 排序，而是实现 residual greedy 或 interaction-aware greedy。每选择一个 segment 后，重新用当前 replay state 更新剩余 `b/rC/rS/ell`，再计算下一轮边际值。若 residual greedy 能显著缩小 82% 到 100% 的差距，就能把 finite-budget law 写成“activated value under residual loss state”。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_key_columns(df: pd.DataFrame) -> None:
    df["city"] = df["city"].astype(str)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(int)
    df["unit"] = df["unit"].astype(str)
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(int)
    df["intervention"] = df["intervention"].astype(str)


def cost_share(group: pd.DataFrame, mask: pd.Series | np.ndarray, total_cost: float) -> float:
    if total_cost <= 0:
        return np.nan
    return float(group.loc[mask, "allocated_cost"].sum() / total_cost)


def time_bin_masks(t: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "early": t <= 3,
        "mid": (t >= 4) & (t <= 7),
        "late": t >= 8,
    }


def finite_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    return out if np.isfinite(out) else fallback


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    values = np.sort(np.maximum(values, 0.0))
    total = values.sum()
    if total <= 0:
        return np.nan
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * total) - (n + 1) / n)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    df.to_csv(path, index=False, float_format="%.10g", compression=compression)


if __name__ == "__main__":
    main()

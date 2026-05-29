"""Scan explicit recovery-footprint trade-offs for residual policies.

V39 showed that footprint-aware residual re-scoring is not a free recovery-gain
improvement. This V40 diagnostic treats footprint coverage as an explicit
secondary objective by scanning a single footprint weight:

    policy_score = residual_recovery_score * (1 + lambda * footprint_rank)

The output is a Pareto-style frontier between recovery gain and observed
event-footprint coverage on the representative hybrid-footprint LP events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_footprint_aware_law_frontier import add_score_columns
from analyze_hybrid_absorption_mechanisms import footprint_weights, hybrid_summary_row, prepare_interventions_with_caps
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from learn_recovery_laws import build_event_action_frame, replay_optimizer_solution, replay_policy_allocations
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from validate_footprint_aware_policy import build_policy_segments, policy_row
from validate_residual_footprint_policy import allocate_adaptive_residual


EPS = 1e-12
DEFAULT_LAMBDAS = (
    0.0,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
    6.00,
    8.00,
    12.00,
)
LOSS_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/residual_footprint_tradeoff")
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
    parser.add_argument("--replan-budget-share", type=float, default=0.05)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--lambda-grid", nargs="*", type=float, default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    lambdas = sorted({float(value) for value in args.lambda_grid if np.isfinite(value) and value >= 0})
    if 0.0 not in lambdas:
        lambdas.insert(0, 0.0)

    policy, allocations = run_tradeoff_scan(
        root,
        config,
        hybrid_lp_dir=root / args.hybrid_lp_dir,
        lambdas=lambdas,
        footprint_blend=float(args.footprint_blend),
        footprint_floor=float(args.footprint_floor),
        max_relative=float(args.max_relative),
        replan_budget_share=float(args.replan_budget_share),
        max_replans=int(args.max_replans),
    )
    summary = build_tradeoff_summary(policy, lambdas)
    frontier = build_pareto_frontier(summary)
    event_best = build_event_best(policy)
    metrics = build_metrics(summary, frontier, event_best)

    write_table(policy, table_dir / "residual_footprint_tradeoff_event_metrics.csv")
    write_table(summary, table_dir / "residual_footprint_tradeoff_summary.csv")
    write_table(frontier, table_dir / "residual_footprint_tradeoff_frontier.csv")
    write_table(event_best, table_dir / "residual_footprint_tradeoff_event_best.csv")
    write_table(allocations, table_dir / "residual_footprint_tradeoff_allocations.csv.gz")
    (table_dir / "residual_footprint_tradeoff_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(summary, frontier, event_best, policy, figure_dir)
    write_report(report_dir / "residual_footprint_tradeoff_report_zh.md", metrics, summary, frontier, event_best)
    print(f"Wrote residual footprint trade-off scan to {output_dir}")


def run_tradeoff_scan(
    root: Path,
    config: dict[str, Any],
    *,
    hybrid_lp_dir: Path,
    lambdas: list[float],
    footprint_blend: float,
    footprint_floor: float,
    max_relative: float,
    replan_budget_share: float,
    max_replans: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_inputs(root)
    lp_dir = hybrid_lp_dir / "tables"
    event_metrics = pd.read_csv(lp_dir / "hybrid_lp_event_metrics.csv", parse_dates=["event_start"])
    selected_events = pd.read_csv(lp_dir / "hybrid_lp_selected_events.csv", parse_dates=["event_start"])
    hybrid_selected_actions = pd.read_csv(lp_dir / "hybrid_lp_selected_actions.csv")
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"].eq("OPTIMAL")) & (base_summary["scenario"].eq("base"))].copy()

    for frame in [event_metrics, selected_events, hybrid_selected_actions, base_summary, data["events"], data["footprint_zone"]]:
        if "event_id" in frame:
            frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")

    event_lookup = {
        (row.city, int(row.event_id)): row
        for row in data["events"].dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    base_summary_lookup = {
        (row.city, int(row.event_id)): row
        for row in base_summary.dropna(subset=["event_id"]).itertuples(index=False)
    }
    selected_lookup = {
        (row.city, int(row.event_id)): row
        for row in selected_events.dropna(subset=["event_id"]).itertuples(index=False)
    }
    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = footprint["event_id"].astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }
    hybrid_prepared = prepare_interventions_with_caps(hybrid_selected_actions, scenario="base")
    ok_events = event_metrics[
        event_metrics["hybrid_status"].astype(str).eq("OPTIMAL")
        & event_metrics["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)

    policy_rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    for idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{len(ok_events)}] Residual footprint trade-off for {city} event {event_id}", flush=True)
        event_row = event_lookup[(city, event_id)]
        base_row = base_summary_lookup[(city, event_id)]
        selected_row = selected_lookup[(city, event_id)]
        footprint_group = footprint_groups[(city, event_id)]

        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=data["abnormal"],
            root=root,
        )
        hybrid_params, _ = build_hybrid_params(
            base_params,
            footprint_group,
            footprint_blend=footprint_blend,
            footprint_floor=footprint_floor,
            max_relative=max_relative,
        )
        hybrid_summary = hybrid_summary_row(row, base_row, hybrid_params, event_row)
        hybrid_actions = hybrid_prepared[
            hybrid_prepared["city"].astype(str).eq(city)
            & hybrid_prepared["event_id"].eq(event_id)
            & hybrid_prepared["scenario"].astype(str).eq("base")
        ].copy()
        full = build_event_action_frame(hybrid_params, hybrid_summary, event_row, hybrid_actions)
        weights = footprint_weights(footprint_group, hybrid_params.units)
        full = add_score_columns(full, weights)
        segments = build_policy_segments(full, config, score_ids=[])
        baseline_objective = no_intervention_objective(hybrid_params)
        optimized_objective = float(getattr(row, "hybrid_optimized_objective"))
        hybrid_lp_gain = baseline_objective - optimized_objective
        optimizer_replay = replay_optimizer_solution(full, hybrid_params)

        lp_row = policy_row(
            city,
            event_id,
            row,
            "hybrid_lp_replay",
            allocations=pd.DataFrame(),
            replay_objective=float(optimizer_replay["objective"]),
            baseline_objective=baseline_objective,
            optimized_objective=optimized_objective,
            hybrid_lp_gain=hybrid_lp_gain,
            weights=weights,
            full=full,
            v34_delta=safe_float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)),
            v35_delta=safe_float(getattr(row, "delta_selected_unit_footprint_mass", np.nan)),
        )
        lp_row["lambda_footprint"] = np.nan
        lp_row["tradeoff_family"] = "hybrid_lp"
        policy_rows.append(lp_row)

        for lambda_footprint in lambdas:
            score_id = lambda_policy_id(lambda_footprint)
            result = allocate_adaptive_residual(
                segments,
                hybrid_params,
                baseline_objective=baseline_objective,
                replan_budget_share=replan_budget_share,
                max_replans=max_replans,
                footprint_lambda=lambda_footprint,
                finite_lambda=0.0,
                mode="additive",
            )
            allocations = result["allocations"]
            if not allocations.empty:
                allocations = allocations.copy()
                allocations["policy_score"] = score_id
                allocations["lambda_footprint"] = float(lambda_footprint)
                allocation_frames.append(allocations)
            replay = replay_policy_allocations(allocations, hybrid_params)
            row_out = policy_row(
                city,
                event_id,
                row,
                score_id,
                allocations=allocations,
                replay_objective=float(replay["objective"]),
                baseline_objective=baseline_objective,
                optimized_objective=optimized_objective,
                hybrid_lp_gain=hybrid_lp_gain,
                weights=weights,
                full=full,
                v34_delta=safe_float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)),
                v35_delta=safe_float(getattr(row, "delta_selected_unit_footprint_mass", np.nan)),
            )
            row_out["lambda_footprint"] = float(lambda_footprint)
            row_out["tradeoff_family"] = "residual_lambda"
            policy_rows.append(row_out)

    policy = pd.DataFrame(policy_rows)
    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    return policy, allocations


def build_tradeoff_summary(policy: pd.DataFrame, lambdas: list[float]) -> pd.DataFrame:
    frame = policy[policy["tradeoff_family"].eq("residual_lambda")].copy()
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby(["policy_score", "lambda_footprint"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_fraction_of_hybrid_lp_gain=("fraction_of_hybrid_lp_gain", "mean"),
            median_fraction_of_hybrid_lp_gain=("fraction_of_hybrid_lp_gain", "median"),
            mean_replay_recoverable_fraction=("replay_recoverable_fraction", "mean"),
            mean_allocated_unit_footprint_mass=("allocated_unit_footprint_mass", "mean"),
            mean_allocated_top5pct_unit_footprint_mass=("allocated_top5pct_unit_footprint_mass", "mean"),
            mean_cost_weighted_footprint_score=("allocated_cost_weighted_footprint_score", "mean"),
            mean_unit_jaccard_with_hybrid_lp=("selected_unit_jaccard_with_hybrid_lp", "mean"),
            mean_action_cost_jaccard_with_hybrid_lp=("action_cost_jaccard_with_hybrid_lp", "mean"),
            mean_action_cost_overlap_share_lp=("action_cost_overlap_share_lp", "mean"),
            mean_selected_action_count=("selected_action_count", "mean"),
            mean_selected_unit_count=("selected_unit_count", "mean"),
        )
        .sort_values("lambda_footprint")
        .reset_index(drop=True)
    )
    base = summary[np.isclose(summary["lambda_footprint"], 0.0)]
    if not base.empty:
        base_row = base.iloc[0]
        summary["delta_fraction_vs_lambda0"] = (
            summary["mean_fraction_of_hybrid_lp_gain"] - float(base_row["mean_fraction_of_hybrid_lp_gain"])
        )
        summary["gain_loss_vs_lambda0"] = -summary["delta_fraction_vs_lambda0"]
        summary["delta_top5_footprint_vs_lambda0"] = (
            summary["mean_allocated_top5pct_unit_footprint_mass"]
            - float(base_row["mean_allocated_top5pct_unit_footprint_mass"])
        )
        summary["delta_cost_weighted_footprint_vs_lambda0"] = (
            summary["mean_cost_weighted_footprint_score"] - float(base_row["mean_cost_weighted_footprint_score"])
        )
        summary["gain_loss_per_0p01_footprint"] = np.where(
            summary["delta_top5_footprint_vs_lambda0"].abs() > EPS,
            summary["gain_loss_vs_lambda0"] / (summary["delta_top5_footprint_vs_lambda0"] / 0.01),
            np.nan,
        )
    summary["grid_order"] = summary["lambda_footprint"].map({value: idx for idx, value in enumerate(lambdas)})
    return summary.sort_values(["grid_order", "lambda_footprint"]).drop(columns=["grid_order"])


def build_pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    frame = summary.copy()
    footprint = frame["mean_allocated_top5pct_unit_footprint_mass"].to_numpy(dtype=float)
    gain = frame["mean_fraction_of_hybrid_lp_gain"].to_numpy(dtype=float)
    efficient = np.ones(len(frame), dtype=bool)
    for idx in range(len(frame)):
        dominates = (
            (footprint >= footprint[idx] - EPS)
            & (gain >= gain[idx] - EPS)
            & ((footprint > footprint[idx] + EPS) | (gain > gain[idx] + EPS))
        )
        if dominates.any():
            efficient[idx] = False
    frontier = frame.loc[efficient].copy().sort_values("mean_allocated_top5pct_unit_footprint_mass")
    frontier["frontier_step"] = np.arange(len(frontier))
    frontier["frontier_delta_gain"] = frontier["mean_fraction_of_hybrid_lp_gain"].diff()
    frontier["frontier_delta_footprint"] = frontier["mean_allocated_top5pct_unit_footprint_mass"].diff()
    frontier["frontier_gain_loss_per_0p01_footprint"] = np.where(
        frontier["frontier_delta_footprint"].abs() > EPS,
        -frontier["frontier_delta_gain"] / (frontier["frontier_delta_footprint"] / 0.01),
        np.nan,
    )
    return frontier


def build_event_best(policy: pd.DataFrame) -> pd.DataFrame:
    frame = policy[policy["tradeoff_family"].eq("residual_lambda")].copy()
    if frame.empty:
        return pd.DataFrame()
    base = frame[np.isclose(frame["lambda_footprint"], 0.0)][
        [
            "city",
            "event_id",
            "fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass",
            "allocated_cost_weighted_footprint_score",
        ]
    ].rename(
        columns={
            "fraction_of_hybrid_lp_gain": "lambda0_fraction_of_hybrid_lp_gain",
            "allocated_top5pct_unit_footprint_mass": "lambda0_top5_footprint_mass",
            "allocated_cost_weighted_footprint_score": "lambda0_cost_weighted_footprint_score",
        }
    )
    enriched = frame.merge(base, on=["city", "event_id"], how="left")
    enriched["delta_fraction_vs_lambda0"] = (
        enriched["fraction_of_hybrid_lp_gain"] - enriched["lambda0_fraction_of_hybrid_lp_gain"]
    )
    enriched["delta_top5_footprint_vs_lambda0"] = (
        enriched["allocated_top5pct_unit_footprint_mass"] - enriched["lambda0_top5_footprint_mass"]
    )
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in enriched.groupby(["city", "event_id"], sort=True):
        row: dict[str, Any] = {"city": city, "event_id": int(event_id)}
        base_row = group[np.isclose(group["lambda_footprint"], 0.0)].iloc[0]
        row["lambda0_fraction"] = float(base_row["lambda0_fraction_of_hybrid_lp_gain"])
        row["lambda0_top5_footprint_mass"] = float(base_row["lambda0_top5_footprint_mass"])
        for threshold in LOSS_THRESHOLDS:
            eligible = group[group["delta_fraction_vs_lambda0"] >= -threshold].copy()
            best = best_footprint_row(eligible)
            suffix = threshold_suffix(threshold)
            row[f"best_lambda_loss_le_{suffix}"] = safe_float(best.get("lambda_footprint"))
            row[f"best_fraction_loss_le_{suffix}"] = safe_float(best.get("fraction_of_hybrid_lp_gain"))
            row[f"best_delta_fraction_loss_le_{suffix}"] = safe_float(best.get("delta_fraction_vs_lambda0"))
            row[f"best_top5_footprint_loss_le_{suffix}"] = safe_float(best.get("allocated_top5pct_unit_footprint_mass"))
            row[f"best_delta_top5_footprint_loss_le_{suffix}"] = safe_float(best.get("delta_top5_footprint_vs_lambda0"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["city", "event_id"])


def build_metrics(summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame) -> dict[str, Any]:
    base = one_row(summary, lambda_footprint=0.0)
    max_gain = summary.sort_values("mean_fraction_of_hybrid_lp_gain", ascending=False).head(1) if not summary.empty else pd.DataFrame()
    max_footprint = summary.sort_values("mean_allocated_top5pct_unit_footprint_mass", ascending=False).head(1) if not summary.empty else pd.DataFrame()
    metrics: dict[str, Any] = {
        "n_lambdas": safe_int(summary["lambda_footprint"].nunique()) if not summary.empty else 0,
        "n_events": safe_int(base.get("n_events")),
        "lambda0_fraction": safe_float(base.get("mean_fraction_of_hybrid_lp_gain")),
        "lambda0_top5_footprint_mass": safe_float(base.get("mean_allocated_top5pct_unit_footprint_mass")),
        "lambda0_cost_weighted_footprint": safe_float(base.get("mean_cost_weighted_footprint_score")),
        "max_gain_lambda": safe_first(max_gain, "lambda_footprint"),
        "max_gain_fraction": safe_first(max_gain, "mean_fraction_of_hybrid_lp_gain"),
        "max_gain_top5_footprint_mass": safe_first(max_gain, "mean_allocated_top5pct_unit_footprint_mass"),
        "max_footprint_lambda": safe_first(max_footprint, "lambda_footprint"),
        "max_footprint_fraction": safe_first(max_footprint, "mean_fraction_of_hybrid_lp_gain"),
        "max_footprint_top5_mass": safe_first(max_footprint, "mean_allocated_top5pct_unit_footprint_mass"),
        "pareto_frontier_points": safe_int(len(frontier)),
    }
    for threshold in LOSS_THRESHOLDS:
        suffix = threshold_suffix(threshold)
        eligible = summary[summary["delta_fraction_vs_lambda0"] >= -threshold].copy() if not summary.empty else pd.DataFrame()
        best = best_footprint_row(eligible)
        metrics[f"best_lambda_loss_le_{suffix}"] = safe_float(best.get("lambda_footprint"))
        metrics[f"best_fraction_loss_le_{suffix}"] = safe_float(best.get("mean_fraction_of_hybrid_lp_gain"))
        metrics[f"best_delta_fraction_loss_le_{suffix}"] = safe_float(best.get("delta_fraction_vs_lambda0"))
        metrics[f"best_top5_footprint_loss_le_{suffix}"] = safe_float(best.get("mean_allocated_top5pct_unit_footprint_mass"))
        metrics[f"best_delta_top5_footprint_loss_le_{suffix}"] = safe_float(best.get("delta_top5_footprint_vs_lambda0"))
        metrics[f"best_gain_loss_per_0p01_footprint_loss_le_{suffix}"] = safe_float(best.get("gain_loss_per_0p01_footprint"))
        event_col = f"best_delta_top5_footprint_loss_le_{suffix}"
        metrics[f"event_mean_delta_top5_loss_le_{suffix}"] = (
            safe_float(event_best[event_col].mean()) if not event_best.empty and event_col in event_best else np.nan
        )
        metrics[f"event_positive_delta_share_loss_le_{suffix}"] = (
            safe_float((event_best[event_col] > EPS).mean()) if not event_best.empty and event_col in event_best else np.nan
        )
    return metrics


def make_figures(
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    event_best: pd.DataFrame,
    policy: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    scatter = ax.scatter(
        summary["mean_allocated_top5pct_unit_footprint_mass"],
        summary["mean_fraction_of_hybrid_lp_gain"],
        c=summary["lambda_footprint"],
        cmap="viridis",
        s=72,
        alpha=0.88,
    )
    if not frontier.empty:
        ax.plot(
            frontier["mean_allocated_top5pct_unit_footprint_mass"],
            frontier["mean_fraction_of_hybrid_lp_gain"],
            color="#111827",
            linewidth=1.4,
            alpha=0.72,
            label="Pareto frontier",
        )
    base = summary[np.isclose(summary["lambda_footprint"], 0.0)]
    if not base.empty:
        ax.axhline(float(base.iloc[0]["mean_fraction_of_hybrid_lp_gain"]), color="#6b7280", linestyle="--", linewidth=1)
        ax.axvline(float(base.iloc[0]["mean_allocated_top5pct_unit_footprint_mass"]), color="#6b7280", linestyle="--", linewidth=1)
    for _, row in summary.iterrows():
        label = f"{float(row['lambda_footprint']):g}"
        if row["lambda_footprint"] in {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0}:
            ax.annotate(label, (row["mean_allocated_top5pct_unit_footprint_mass"], row["mean_fraction_of_hybrid_lp_gain"]), fontsize=7)
    ax.set_xlabel("Allocated top-5% unit footprint mass")
    ax.set_ylabel("Replay gain / hybrid LP gain")
    ax.set_title("Explicit residual recovery-footprint trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="footprint lambda")
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_footprint_tradeoff_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(summary["lambda_footprint"], summary["gain_loss_vs_lambda0"], marker="o", color="#b91c1c", label="gain loss")
    ax.plot(
        summary["lambda_footprint"],
        summary["delta_top5_footprint_vs_lambda0"],
        marker="s",
        color="#2563eb",
        label="footprint gain",
    )
    ax.set_xscale("symlog", linthresh=0.05)
    ax.axhline(0, color="#111827", linewidth=1, alpha=0.4)
    ax.set_xlabel("Footprint lambda")
    ax.set_ylabel("Delta vs lambda=0")
    ax.set_title("Footprint gain and recovery-gain cost")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "lambda_path_gain_loss.png", dpi=180)
    plt.close(fig)

    if not event_best.empty:
        col = "best_delta_top5_footprint_loss_le_0p01"
        frac_col = "best_delta_fraction_loss_le_0p01"
        plot = event_best.sort_values(col, ascending=True).copy()
        labels = plot["city"].astype(str) + " #" + plot["event_id"].astype(str)
        y = np.arange(len(plot))
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        ax.barh(y, plot[col], color="#0f766e", label="footprint delta")
        ax.scatter(plot[frac_col], y, color="#b91c1c", label="gain delta")
        ax.axvline(0, color="#111827", linewidth=1, alpha=0.45)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Event-level delta under <=0.01 gain loss")
        ax.set_title("Event heterogeneity in near-loss footprint gains")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "event_near_loss_footprint_gain.png", dpi=180)
        plt.close(fig)

    event_frame = policy[policy["tradeoff_family"].eq("residual_lambda")].copy()
    if not event_frame.empty:
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        for (city, event_id), group in event_frame.groupby(["city", "event_id"]):
            group = group.sort_values("lambda_footprint")
            ax.plot(
                group["allocated_top5pct_unit_footprint_mass"],
                group["fraction_of_hybrid_lp_gain"],
                marker="o",
                linewidth=1.0,
                alpha=0.65,
                label=f"{city} #{int(event_id)}",
            )
        ax.set_xlabel("Allocated top-5% unit footprint mass")
        ax.set_ylabel("Replay gain / hybrid LP gain")
        ax.set_title("Event-level trade-off curves")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / "event_tradeoff_curves.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    event_best: pd.DataFrame,
) -> None:
    lines = [
        "# Residual Footprint Trade-Off V40",
        "",
        "本版把 V39 的 adaptive residual footprint 结果改写成显式多目标问题：不再问 footprint 是否能免费提高 recovery gain，而是扫描 `lambda_footprint`，观察每一点 footprint 覆盖需要付出多少 recovery-gain 代价。",
        "",
        "策略分数为：",
        "",
        "```text",
        "score = residual_recovery_score * (1 + lambda_footprint * footprint_rank)",
        "```",
        "",
        "其中 `lambda=0` 是纯 residual finite greedy；`lambda` 越大，越愿意在 residual recovery score 相近时偏向 observed event footprint 区域。",
        "",
        "## 关键结论",
        "",
        f"- 覆盖事件数：{metrics['n_events']}；lambda 网格点数：{metrics['n_lambdas']}；Pareto frontier 点数：{metrics['pareto_frontier_points']}。",
        f"- 纯 residual (`lambda=0`) gain / hybrid LP gain = {fmt(metrics['lambda0_fraction'])}，top-5% footprint mass = {fmt(metrics['lambda0_top5_footprint_mass'])}。",
        f"- 最大 recovery gain 出现在 lambda = {fmt(metrics['max_gain_lambda'])}：gain / LP = {fmt(metrics['max_gain_fraction'])}，footprint mass = {fmt(metrics['max_gain_top5_footprint_mass'])}。",
        f"- 最大 footprint 覆盖出现在 lambda = {fmt(metrics['max_footprint_lambda'])}：gain / LP = {fmt(metrics['max_footprint_fraction'])}，footprint mass = {fmt(metrics['max_footprint_top5_mass'])}。",
        f"- 若允许 gain / LP 损失不超过 0.005，最佳 lambda = {fmt(metrics['best_lambda_loss_le_0p005'])}，footprint mass = {fmt(metrics['best_top5_footprint_loss_le_0p005'])}，footprint delta = {fmt(metrics['best_delta_top5_footprint_loss_le_0p005'])}，gain delta = {fmt(metrics['best_delta_fraction_loss_le_0p005'])}。",
        f"- 若允许 gain / LP 损失不超过 0.01，最佳 lambda = {fmt(metrics['best_lambda_loss_le_0p01'])}，footprint mass = {fmt(metrics['best_top5_footprint_loss_le_0p01'])}，footprint delta = {fmt(metrics['best_delta_top5_footprint_loss_le_0p01'])}，gain delta = {fmt(metrics['best_delta_fraction_loss_le_0p01'])}。",
        f"- 在 <=0.01 gain loss 下，事件层面平均 footprint delta = {fmt(metrics['event_mean_delta_top5_loss_le_0p01'])}，正向 footprint 改善事件占比 = {fmt(metrics['event_positive_delta_share_loss_le_0p01'])}。",
        "",
        "## 解释",
        "",
        "这版的核心 law 更适合写成 trade-off law：observed footprint 不是恢复收益的免费来源，而是一个可显式定价的空间偏好。小 lambda 对 recovery gain 的损失很小，但只能带来有限 footprint 改善；大 lambda 会把资源推向 footprint 区域，同时明显牺牲 residual recovery value。",
        "",
        "因此，如果论文主张是 recoverability law，footprint 应写作 secondary spatial preference；如果论文主张是 footprint fairness 或 event-footprint coverage，就应把它作为显式多目标项，而不是隐藏在 recovery-only objective 里。",
        "",
        "## Lambda Summary",
        "",
        table_to_markdown(summary, max_rows=40),
        "",
        "## Pareto Frontier",
        "",
        table_to_markdown(frontier, max_rows=40),
        "",
        "## Event Best Under Loss Budgets",
        "",
        table_to_markdown(event_best, max_rows=30),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_footprint_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.sort_values(
        ["mean_allocated_top5pct_unit_footprint_mass", "mean_fraction_of_hybrid_lp_gain"]
        if "mean_allocated_top5pct_unit_footprint_mass" in frame
        else ["allocated_top5pct_unit_footprint_mass", "fraction_of_hybrid_lp_gain"],
        ascending=[False, False],
    ).iloc[0]


def lambda_policy_id(value: float) -> str:
    if abs(value) <= EPS:
        return "lambda_0_residual"
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return f"lambda_{text}_footprint"


def threshold_suffix(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def one_row(df: pd.DataFrame, **filters: Any) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if column not in df:
            return pd.Series(dtype=float)
        if isinstance(value, float):
            mask &= np.isclose(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float), value)
        else:
            mask &= df[column].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=float)
    return df.loc[mask].iloc[0]


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return np.nan
    return number if np.isfinite(number) else np.nan


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def safe_first(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return safe_float(frame.iloc[0][column])


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if not np.isfinite(number) else f"{number:.4f}"


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
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

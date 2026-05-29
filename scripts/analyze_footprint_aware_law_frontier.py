"""Test whether observed footprints can improve the hybrid action law.

V36 showed that footprint-rich units can have high finite-value rank while
remaining weak in small-signal OD action value.  This script asks the next
question: can a simple footprint-aware composite score improve footprint
coverage without losing alignment with the representative hybrid LP support?

The analysis reuses the V35 hybrid LP selected actions and rebuilds the
hybrid-calibrated action-token field.  It evaluates a family of ranking scores
on two axes:

1. selected-support alignment: fraction of hybrid LP selected cost captured by
   the score's top-20 percent actions;
2. footprint alignment: observed footprint mass covered by the score's top-5
   percent units.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_hybrid_absorption_mechanisms import footprint_weights, hybrid_summary_row, prepare_interventions_with_caps, rank_pct
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs
from learn_recovery_laws import build_event_action_frame
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--hybrid-lp-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/footprint_aware_law_frontier")
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
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
    lp_dir = root / args.hybrid_lp_dir / "tables"
    event_metrics = pd.read_csv(lp_dir / "hybrid_lp_event_metrics.csv", parse_dates=["event_start"])
    hybrid_selected_actions = pd.read_csv(lp_dir / "hybrid_lp_selected_actions.csv")
    selected_events = pd.read_csv(lp_dir / "hybrid_lp_selected_events.csv", parse_dates=["event_start"])
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"].eq("OPTIMAL")) & (base_summary["scenario"].eq("base"))].copy()

    for frame in [event_metrics, selected_events, hybrid_selected_actions, base_summary, data["events"], data["footprint_zone"]]:
        if "event_id" in frame:
            frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")

    event_lookup = {
        (row.city, int(row.event_id)): row for row in data["events"].dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    base_summary_lookup = {
        (row.city, int(row.event_id)): row for row in base_summary.dropna(subset=["event_id"]).itertuples(index=False)
    }
    selected_lookup = {
        (row.city, int(row.event_id)): row for row in selected_events.dropna(subset=["event_id"]).itertuples(index=False)
    }
    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = footprint["event_id"].astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }
    hybrid_prepared = prepare_interventions_with_caps(hybrid_selected_actions, scenario="base")

    score_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []

    ok_events = event_metrics[
        event_metrics["hybrid_status"].astype(str).eq("OPTIMAL")
        & event_metrics["error"].fillna("").astype(str).eq("")
    ].copy()
    ok_events["event_id"] = ok_events["event_id"].astype(int)
    for idx, row in enumerate(ok_events.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{len(ok_events)}] Evaluating footprint-aware scores for {city} event {event_id}", flush=True)
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
            footprint_blend=float(args.footprint_blend),
            footprint_floor=float(args.footprint_floor),
            max_relative=float(args.max_relative),
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
        score_defs = build_scores(full)
        event_score_rows = [evaluate_score(full, weights, score_id, score_values) for score_id, score_values in score_defs.items()]
        for metric in event_score_rows:
            metric.update(
                {
                    "city": city,
                    "event_id": event_id,
                    "event_start": str(getattr(row, "event_start", "")),
                    "v34_delta_finite_top5pct_units_footprint_mass": float(getattr(selected_row, "delta_finite_top5pct_units_footprint_mass", np.nan)),
                    "v35_delta_selected_unit_footprint_mass": float(getattr(row, "delta_selected_unit_footprint_mass", np.nan)),
                }
            )
        score_rows.extend(event_score_rows)
        event_rows.extend(event_best_rows(event_score_rows))

    score_metrics = pd.DataFrame(score_rows)
    score_summary = build_score_summary(score_metrics)
    frontier = pareto_frontier(score_summary)
    frontier_rows.extend(frontier.to_dict("records"))
    event_best = pd.DataFrame(event_rows)
    metrics = build_metrics(score_summary, frontier, event_best)

    write_table(score_metrics, table_dir / "footprint_aware_score_event_metrics.csv")
    write_table(score_summary, table_dir / "footprint_aware_score_summary.csv")
    write_table(frontier, table_dir / "footprint_aware_pareto_frontier.csv")
    write_table(event_best, table_dir / "footprint_aware_event_best_scores.csv")
    (table_dir / "footprint_aware_law_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(score_summary, frontier, event_best, figure_dir)
    write_report(report_dir / "footprint_aware_law_frontier_report_zh.md", metrics, score_summary, frontier, event_best)
    print(f"Wrote footprint-aware law frontier to {output_dir}")


def add_score_columns(full: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    out = full.copy()
    out["unit"] = out["unit"].astype(str)
    out["footprint_weight"] = weights.reindex(out["unit"], fill_value=0.0).to_numpy(dtype=float)
    unit_weight = weights.reindex(sorted(weights.index.astype(str)), fill_value=0.0)
    footprint_rank = pd.Series(rank_pct(unit_weight.to_numpy(dtype=float)), index=unit_weight.index)
    out["footprint_unit_rank"] = footprint_rank.reindex(out["unit"], fill_value=0.0).to_numpy(dtype=float)
    unit_finite = out.groupby("unit")["finite_deficit_area_value"].sum()
    finite_rank = pd.Series(rank_pct(unit_finite.to_numpy(dtype=float)), index=unit_finite.index)
    out["finite_unit_rank"] = finite_rank.reindex(out["unit"], fill_value=0.0).to_numpy(dtype=float)
    unit_small = out.groupby("unit")["marginal_resource_value"].sum()
    small_rank = pd.Series(rank_pct(unit_small.to_numpy(dtype=float)), index=unit_small.index)
    out["small_signal_unit_rank"] = small_rank.reindex(out["unit"], fill_value=0.0).to_numpy(dtype=float)
    return out


def build_scores(full: pd.DataFrame) -> dict[str, np.ndarray]:
    small = positive(full["marginal_resource_value"])
    finite = positive(full["finite_deficit_area_value"])
    footprint_rank = positive(full["footprint_unit_rank"])
    finite_rank = positive(full["finite_unit_rank"])
    small_rank = positive(full["small_signal_unit_rank"])
    exposure = positive(full["law_exposure_term"])
    horizon = positive(full["active_weighted_horizon"])
    eta_cost = positive(full["eta_per_cost"])

    scores: dict[str, np.ndarray] = {
        "small_signal": small,
        "finite_value": finite,
        "footprint_only": footprint_rank * positive(full["delay_feasible"]),
        "small_x_finite_rank": small * finite_rank,
        "small_x_footprint_rank": small * footprint_rank,
        "finite_x_small_rank": finite * small_rank,
        "activation_no_eta": horizon * exposure * positive(full["delay_feasible"]),
        "activation_x_footprint": horizon * exposure * footprint_rank * positive(full["delay_feasible"]),
    }
    for lam in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        suffix = str(lam).replace(".", "p")
        scores[f"small_plus_{suffix}_footprint"] = small * (1.0 + lam * footprint_rank)
        scores[f"small_plus_{suffix}_finite"] = small * (1.0 + lam * finite_rank)
        scores[f"activation_plus_{suffix}_footprint"] = horizon * exposure * eta_cost * positive(full["delay_feasible"]) * (1.0 + lam * footprint_rank)
    return scores


def evaluate_score(full: pd.DataFrame, weights: pd.Series, score_id: str, score_values: np.ndarray) -> dict[str, Any]:
    frame = full.copy()
    frame["score"] = np.asarray(score_values, dtype=float)
    frame["score"] = frame["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_actions = len(frame)
    top20_n = max(1, int(math.ceil(0.20 * n_actions)))
    top5_n = max(1, int(math.ceil(0.05 * n_actions)))
    n_units = frame["unit"].nunique()
    top5_unit_n = max(1, int(math.ceil(0.05 * n_units)))
    selected_cost_total = float(frame["optimized_cost"].sum())
    selected_action_count = int((frame["optimized_cost"] > EPS).sum())

    ranked = frame.sort_values("score", ascending=False)
    top20 = ranked.head(top20_n)
    top5 = ranked.head(top5_n)
    unit_score = frame.groupby("unit", as_index=False)["score"].sum().sort_values("score", ascending=False)
    top_units = set(unit_score.head(top5_unit_n)["unit"].astype(str))
    selected_units = set(frame.loc[frame["optimized_cost"] > EPS, "unit"].astype(str))

    finite_values = positive(frame["finite_deficit_area_value"])
    small_values = positive(frame["marginal_resource_value"])
    top20_idx = set(top20.index)
    top5_idx = set(top5.index)
    return {
        "score_id": score_id,
        "selected_cost_capture_top20pct_actions": float(top20["optimized_cost"].sum()) / max(selected_cost_total, EPS),
        "selected_action_recall_top20pct_actions": safe_div(float((top20["optimized_cost"] > EPS).sum()), selected_action_count),
        "selected_cost_capture_top5pct_actions": float(top5["optimized_cost"].sum()) / max(selected_cost_total, EPS),
        "finite_value_capture_top20pct_actions": float(finite_values[list(top20_idx)].sum()) / max(float(finite_values.sum()), EPS),
        "small_signal_value_capture_top20pct_actions": float(small_values[list(top20_idx)].sum()) / max(float(small_values.sum()), EPS),
        "top5pct_units_footprint_mass": set_mass(weights, top_units),
        "top5pct_units_selected_share": safe_div(len(top_units & selected_units), len(top_units)),
        "top5pct_units_count": int(len(top_units)),
        "mean_top5pct_unit_footprint_rank": safe_mean(frame[frame["unit"].isin(top_units)], "footprint_unit_rank"),
        "mean_top5pct_unit_small_signal_rank": safe_mean(frame[frame["unit"].isin(top_units)], "small_signal_unit_rank"),
        "mean_top5pct_unit_finite_rank": safe_mean(frame[frame["unit"].isin(top_units)], "finite_unit_rank"),
    }


def build_score_summary(score_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        score_metrics.groupby("score_id", as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_selected_cost_capture_top20pct_actions=("selected_cost_capture_top20pct_actions", "mean"),
            mean_selected_action_recall_top20pct_actions=("selected_action_recall_top20pct_actions", "mean"),
            mean_selected_cost_capture_top5pct_actions=("selected_cost_capture_top5pct_actions", "mean"),
            mean_finite_value_capture_top20pct_actions=("finite_value_capture_top20pct_actions", "mean"),
            mean_small_signal_value_capture_top20pct_actions=("small_signal_value_capture_top20pct_actions", "mean"),
            mean_top5pct_units_footprint_mass=("top5pct_units_footprint_mass", "mean"),
            mean_top5pct_units_selected_share=("top5pct_units_selected_share", "mean"),
            mean_top5pct_unit_footprint_rank=("mean_top5pct_unit_footprint_rank", "mean"),
            mean_top5pct_unit_small_signal_rank=("mean_top5pct_unit_small_signal_rank", "mean"),
            mean_top5pct_unit_finite_rank=("mean_top5pct_unit_finite_rank", "mean"),
        )
        .copy()
    )
    base = grouped[grouped["score_id"].eq("small_signal")]
    if not base.empty:
        base_row = base.iloc[0]
        grouped["delta_selected_cost_capture_vs_small_signal"] = (
            grouped["mean_selected_cost_capture_top20pct_actions"] - float(base_row["mean_selected_cost_capture_top20pct_actions"])
        )
        grouped["delta_footprint_mass_vs_small_signal"] = (
            grouped["mean_top5pct_units_footprint_mass"] - float(base_row["mean_top5pct_units_footprint_mass"])
        )
    grouped["frontier_utility"] = (
        grouped["mean_selected_cost_capture_top20pct_actions"]
        + grouped["mean_top5pct_units_footprint_mass"]
        - 0.5 * np.maximum(-grouped.get("delta_selected_cost_capture_vs_small_signal", 0.0), 0.0)
    )
    return grouped.sort_values(
        ["mean_selected_cost_capture_top20pct_actions", "mean_top5pct_units_footprint_mass"],
        ascending=[False, False],
    )


def pareto_frontier(score_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in score_summary.itertuples(index=False):
        dominated = False
        for other in score_summary.itertuples(index=False):
            if other.score_id == row.score_id:
                continue
            better_or_equal = (
                other.mean_selected_cost_capture_top20pct_actions >= row.mean_selected_cost_capture_top20pct_actions - EPS
                and other.mean_top5pct_units_footprint_mass >= row.mean_top5pct_units_footprint_mass - EPS
            )
            strictly_better = (
                other.mean_selected_cost_capture_top20pct_actions > row.mean_selected_cost_capture_top20pct_actions + EPS
                or other.mean_top5pct_units_footprint_mass > row.mean_top5pct_units_footprint_mass + EPS
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            out = row._asdict()
            out["pareto_frontier"] = True
            rows.append(out)
    return pd.DataFrame(rows).sort_values(
        ["mean_selected_cost_capture_top20pct_actions", "mean_top5pct_units_footprint_mass"],
        ascending=[False, False],
    )


def event_best_rows(event_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(event_score_rows)
    rows: list[dict[str, Any]] = []
    for metric, label in [
        ("selected_cost_capture_top20pct_actions", "best_selected_support"),
        ("top5pct_units_footprint_mass", "best_footprint_mass"),
    ]:
        best = frame.sort_values(metric, ascending=False).head(1)
        if best.empty:
            continue
        row = best.iloc[0].to_dict()
        row["best_for"] = label
        rows.append(row)
    return rows


def build_metrics(score_summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame) -> dict[str, Any]:
    small = one_row(score_summary, score_id="small_signal")
    finite = one_row(score_summary, score_id="finite_value")
    footprint = one_row(score_summary, score_id="footprint_only")
    nonbaseline = score_summary[~score_summary["score_id"].isin(["small_signal"])]
    no_loss = nonbaseline[nonbaseline["delta_selected_cost_capture_vs_small_signal"] >= -0.01] if "delta_selected_cost_capture_vs_small_signal" in nonbaseline else pd.DataFrame()
    best_no_loss = no_loss.sort_values(["mean_top5pct_units_footprint_mass", "mean_selected_cost_capture_top20pct_actions"], ascending=[False, False]).head(1)
    best_footprint = score_summary.sort_values("mean_top5pct_units_footprint_mass", ascending=False).head(1)
    best_support = score_summary.sort_values("mean_selected_cost_capture_top20pct_actions", ascending=False).head(1)
    best_utility = score_summary.sort_values("frontier_utility", ascending=False).head(1)
    return {
        "n_scores": int(len(score_summary)),
        "n_frontier_scores": int(len(frontier)),
        "small_signal_selected_cost_capture_top20": safe_float(small.get("mean_selected_cost_capture_top20pct_actions")),
        "small_signal_footprint_mass_top5_units": safe_float(small.get("mean_top5pct_units_footprint_mass")),
        "finite_value_selected_cost_capture_top20": safe_float(finite.get("mean_selected_cost_capture_top20pct_actions")),
        "finite_value_footprint_mass_top5_units": safe_float(finite.get("mean_top5pct_units_footprint_mass")),
        "footprint_only_selected_cost_capture_top20": safe_float(footprint.get("mean_selected_cost_capture_top20pct_actions")),
        "footprint_only_footprint_mass_top5_units": safe_float(footprint.get("mean_top5pct_units_footprint_mass")),
        "best_no_loss_score_id": str(best_no_loss.iloc[0]["score_id"]) if not best_no_loss.empty else "",
        "best_no_loss_selected_cost_capture_top20": safe_first(best_no_loss, "mean_selected_cost_capture_top20pct_actions"),
        "best_no_loss_footprint_mass_top5_units": safe_first(best_no_loss, "mean_top5pct_units_footprint_mass"),
        "best_no_loss_delta_selected_capture": safe_first(best_no_loss, "delta_selected_cost_capture_vs_small_signal"),
        "best_no_loss_delta_footprint_mass": safe_first(best_no_loss, "delta_footprint_mass_vs_small_signal"),
        "best_footprint_score_id": str(best_footprint.iloc[0]["score_id"]) if not best_footprint.empty else "",
        "best_footprint_selected_cost_capture_top20": safe_first(best_footprint, "mean_selected_cost_capture_top20pct_actions"),
        "best_footprint_mass_top5_units": safe_first(best_footprint, "mean_top5pct_units_footprint_mass"),
        "best_footprint_delta_selected_capture": safe_first(best_footprint, "delta_selected_cost_capture_vs_small_signal"),
        "best_support_score_id": str(best_support.iloc[0]["score_id"]) if not best_support.empty else "",
        "best_support_selected_cost_capture_top20": safe_first(best_support, "mean_selected_cost_capture_top20pct_actions"),
        "best_support_footprint_mass_top5_units": safe_first(best_support, "mean_top5pct_units_footprint_mass"),
        "best_utility_score_id": str(best_utility.iloc[0]["score_id"]) if not best_utility.empty else "",
        "best_utility_selected_cost_capture_top20": safe_first(best_utility, "mean_selected_cost_capture_top20pct_actions"),
        "best_utility_footprint_mass_top5_units": safe_first(best_utility, "mean_top5pct_units_footprint_mass"),
        "event_best_support_small_signal_share": safe_share(event_best, "best_selected_support", "small_signal"),
        "event_best_footprint_small_signal_share": safe_share(event_best, "best_footprint_mass", "small_signal"),
    }


def make_figures(score_summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame, figure_dir: Path) -> None:
    if score_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        score_summary["mean_top5pct_units_footprint_mass"],
        score_summary["mean_selected_cost_capture_top20pct_actions"],
        s=55,
        color="#94a3b8",
        alpha=0.75,
        label="scores",
    )
    if not frontier.empty:
        ax.scatter(
            frontier["mean_top5pct_units_footprint_mass"],
            frontier["mean_selected_cost_capture_top20pct_actions"],
            s=90,
            color="#2563eb",
            label="Pareto frontier",
        )
    for _, row in score_summary.sort_values("frontier_utility", ascending=False).head(8).iterrows():
        ax.annotate(str(row["score_id"]), (row["mean_top5pct_units_footprint_mass"], row["mean_selected_cost_capture_top20pct_actions"]), fontsize=7)
    ax.set_xlabel("Top-5% unit footprint mass")
    ax.set_ylabel("LP selected-cost capture by top-20% actions")
    ax.set_title("Footprint-aware score frontier")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "footprint_aware_score_frontier.png", dpi=180)
    plt.close(fig)

    ordered = score_summary.sort_values("mean_selected_cost_capture_top20pct_actions", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(ordered))
    width = 0.38
    ax.bar(x - width / 2, ordered["mean_selected_cost_capture_top20pct_actions"], width=width, color="#2563eb", label="selected support")
    ax.bar(x + width / 2, ordered["mean_top5pct_units_footprint_mass"], width=width, color="#f59e0b", label="footprint mass")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["score_id"], rotation=35, ha="right")
    ax.set_ylabel("Mean metric")
    ax.set_title("Top support-aligned scores and footprint trade-off")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "support_vs_footprint_top_scores.png", dpi=180)
    plt.close(fig)

    if not event_best.empty:
        pivot = event_best.groupby(["best_for", "score_id"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for idx, (best_for, group) in enumerate(pivot.groupby("best_for")):
            ax.barh(group["score_id"] + f" ({best_for})", group["count"], color="#0f766e" if idx == 0 else "#7c3aed")
        ax.set_xlabel("Event count")
        ax.set_title("Which score wins per representative event?")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "event_best_score_counts.png", dpi=180)
        plt.close(fig)


def write_report(path: Path, metrics: dict[str, Any], score_summary: pd.DataFrame, frontier: pd.DataFrame, event_best: pd.DataFrame) -> None:
    if metrics["best_no_loss_score_id"]:
        no_loss_line = (
            f"- 在 selected-cost capture 允许最多 0.01 损失的候选中，最好的 footprint-aware score 是 "
            f"`{metrics['best_no_loss_score_id']}`：selected-cost capture = {fmt(metrics['best_no_loss_selected_cost_capture_top20'])}，"
            f"top-5% unit footprint mass = {fmt(metrics['best_no_loss_footprint_mass_top5_units'])}，"
            f"相对 small-signal 的 footprint gain = {fmt(metrics['best_no_loss_delta_footprint_mass'])}。"
        )
    else:
        no_loss_line = "- 没有 footprint-aware score 能在 selected-cost capture 损失小于 0.01 的条件下改善 footprint mass。"
    lines = [
        "# Footprint-Aware Law Frontier V37",
        "",
        "本版测试一个关键问题：observed event footprint 是否可以作为复合项进入主 recovery-action law，或者它只能作为 magnitude-aware 边界信号。",
        "",
        "## 主要结果",
        "",
        f"- 共评估 {metrics['n_scores']} 个候选 score，其中 Pareto frontier 上有 {metrics['n_frontier_scores']} 个。",
        f"- `small_signal` 的 selected-cost top-20% capture = {fmt(metrics['small_signal_selected_cost_capture_top20'])}，top-5% unit footprint mass = {fmt(metrics['small_signal_footprint_mass_top5_units'])}。",
        f"- `finite_value` 的 selected-cost capture = {fmt(metrics['finite_value_selected_cost_capture_top20'])}，footprint mass = {fmt(metrics['finite_value_footprint_mass_top5_units'])}。",
        f"- `footprint_only` 的 selected-cost capture = {fmt(metrics['footprint_only_selected_cost_capture_top20'])}，footprint mass = {fmt(metrics['footprint_only_footprint_mass_top5_units'])}。",
        no_loss_line,
        f"- footprint mass 最高的 score 是 `{metrics['best_footprint_score_id']}`，footprint mass = {fmt(metrics['best_footprint_mass_top5_units'])}，但 selected-cost capture 相对 small-signal 变化 {fmt(metrics['best_footprint_delta_selected_capture'])}。",
        f"- selected support 捕获最高的 score 是 `{metrics['best_support_score_id']}`，selected-cost capture = {fmt(metrics['best_support_selected_cost_capture_top20'])}，footprint mass = {fmt(metrics['best_support_footprint_mass_top5_units'])}。",
        "",
        "## 解释",
        "",
        "V37 的判据不是单纯让 score 更靠近 footprint，而是要求它同时保留对 hybrid LP selected support 的解释力。这个 frontier 因此直接测试 footprint 能否成为主 action law 的一部分。",
        "",
        "如果 footprint-aware score 只能通过牺牲 selected-support capture 来提高 footprint mass，那么它更适合作为 event-specific finite-magnitude diagnostic，而不是替代 small-signal OD activation law 的主排序变量。",
        "",
        "## Score Summary",
        "",
        table_to_markdown(score_summary, max_rows=40),
        "",
        "## Pareto Frontier",
        "",
        table_to_markdown(frontier, max_rows=30),
        "",
        "## Event Winners",
        "",
        table_to_markdown(event_best, max_rows=30),
        "",
        "## 写作含义",
        "",
        "这版结果用于决定论文中的 footprint 位置：若存在 near-no-loss footprint-aware score，可把 footprint 写成主 law 的 magnitude-aware refinement；若不存在，则应把 footprint 写成数据中真实存在、但尚未闭合为 recovery-action law 的空间信号。后一种结果同样重要，因为它防止把观测损失 footprint 误写成最优恢复 footprint。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


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


def positive(values: Any) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return np.clip(arr, 0.0, None)


def set_mass(weights: pd.Series, units: set[str]) -> float:
    if not units:
        return 0.0
    return float(weights.reindex(pd.Index([str(unit) for unit in units]), fill_value=0.0).sum())


def safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else np.nan


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return np.nan
    return number if np.isfinite(number) else np.nan


def safe_first(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return safe_float(frame.iloc[0][column])


def safe_share(frame: pd.DataFrame, best_for: str, score_id: str) -> float:
    if frame.empty:
        return np.nan
    sub = frame[frame["best_for"].astype(str).eq(best_for)]
    if sub.empty:
        return np.nan
    return float(sub["score_id"].astype(str).eq(score_id).mean())


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if not np.isfinite(number) else f"{number:.4f}"


if __name__ == "__main__":
    main()

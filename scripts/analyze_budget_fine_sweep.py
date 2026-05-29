"""Run a fine-grained budget sweep for managed-recoverability leverage laws."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learn_recovery_laws import (
    INTERVENTIONS,
    allocate_greedy_policy,
    build_budget_segments,
    build_event_action_frame,
    load_inputs,
    prepare_interventions,
    replay_policy_allocations,
    replay_row,
)
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


POLICY_SCORES = {
    "greedy_oracle": "oracle_value_per_cost",
    "activated_bottleneck_law": "law_value_score",
    "exposure_only": "exposure_policy_score",
    "deficit_only": "deficit_policy_score",
    "structure_only": "structure_policy_score",
    "random_positive": "random_policy_score",
}
BASELINES = ["random_positive", "exposure_only", "deficit_only", "structure_only"]
DEFAULT_BUDGET_SCALES = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
EPS = 1e-12
RNG_SEED = 20260529


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/budget_fine_sweep")
    parser.add_argument("--budget-scales", nargs="*", type=float, default=DEFAULT_BUDGET_SCALES)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    policy, replay = run_fine_sweep(
        root,
        config,
        sorted(set(float(scale) for scale in args.budget_scales)),
        max_events=args.max_events,
    )
    policy = add_relative_proxy_metrics(policy)
    replay = add_relative_replay_metrics(replay)
    event_metrics = build_event_metrics(policy, replay, root)
    summary = build_summary(event_metrics)
    city_summary = build_city_summary(event_metrics)
    phase_tests = build_phase_tests(summary)
    city_phase = build_city_phase_tests(city_summary)
    diagnostics = build_diagnostics(summary, phase_tests, city_phase)

    write_table(policy, table_dir / "fine_budget_policy_simulation.csv")
    write_table(replay, table_dir / "fine_budget_policy_replay.csv")
    write_table(event_metrics, table_dir / "fine_budget_event_metrics.csv")
    write_table(summary, table_dir / "fine_budget_summary.csv")
    write_table(city_summary, table_dir / "fine_budget_city_summary.csv")
    write_table(phase_tests, table_dir / "fine_budget_phase_tests.csv")
    write_table(city_phase, table_dir / "fine_budget_city_phase_tests.csv")
    (table_dir / "fine_budget_metrics.json").write_text(
        pd.Series(diagnostics).to_json(indent=2),
        encoding="utf-8",
    )

    make_figures(summary, city_summary, event_metrics, phase_tests, figure_dir)
    write_report(report_dir / "budget_fine_sweep_report_zh.md", diagnostics, summary, city_summary, phase_tests, city_phase)
    print(f"Wrote fine budget sweep to {output_dir}")


def run_fine_sweep(
    root: Path,
    config: dict[str, Any],
    budget_scales: list[float],
    *,
    max_events: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_inputs(root)
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    summary = summary.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)
    if max_events is not None:
        summary = summary.head(max_events).copy()

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()
    rng = np.random.default_rng(RNG_SEED)

    policy_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    total_events = len(summary)
    for idx, row in enumerate(summary.itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        print(f"[{idx}/{total_events}] Fine budget sweep for {city} event {event_id}", flush=True)
        event_row = event_lookup.get((city, event_id))
        if event_row is None or city not in dynamic_lookup:
            continue
        params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        event_interventions = interventions[
            (interventions["city"] == city)
            & (interventions["event_id"] == event_id)
            & (interventions["scenario"] == "base")
        ]
        full = build_event_action_frame(params, row, event_row, event_interventions)
        segments = build_budget_segments(full, config, rng)
        feasible_by_delay = delay_feasible_mask(segments, params)
        baseline_objective = float(full["baseline_objective"].iloc[0])
        optimized_objective = float(full["optimized_objective"].iloc[0])
        lp_recoverable_fraction = float(full["recoverable_fraction"].iloc[0])
        for budget_scale in budget_scales:
            period_budget = params.period_budget * float(budget_scale)
            total_budget = float(params.total_budget) * float(budget_scale)
            scenario_segments = segments.loc[feasible_by_delay & (segments["oracle_value_per_cost"] > 0.0)].copy()
            for policy_score, score_col in POLICY_SCORES.items():
                result = allocate_greedy_policy(
                    scenario_segments,
                    score_col,
                    period_budget=period_budget,
                    total_budget=total_budget,
                )
                policy_rows.append(
                    {
                        "city": city,
                        "event_id": event_id,
                        "event_start": str(full["event_start"].iloc[0]),
                        "budget_scale": float(budget_scale),
                        "policy_score": policy_score,
                        "allocated_cost": float(result["allocated_cost"]),
                        "value_proxy": float(result["value_proxy"]),
                        "selected_segment_count": int(result["selected_segment_count"]),
                        "selected_action_count": int(result["selected_action_count"]),
                        "baseline_objective": baseline_objective,
                        "recoverable_fraction": lp_recoverable_fraction,
                        "event_peak_positive_abnormal_deficit": float(full["event_peak_positive_abnormal_deficit"].iloc[0]),
                        "event_total_precip": float(full["event_total_precip"].iloc[0]),
                    }
                )
                replay = replay_policy_allocations(result["allocations"], params)
                replay_rows.append(
                    replay_row(
                        full,
                        policy_scenario=f"budget_{budget_scale:g}",
                        budget_scale=float(budget_scale),
                        delay_add_hours=0,
                        policy_score=policy_score,
                        allocated_cost=float(result["allocated_cost"]),
                        value_proxy=float(result["value_proxy"]),
                        selected_action_count=int(result["selected_action_count"]),
                        replay_objective=replay["objective"],
                        baseline_objective=baseline_objective,
                        optimized_objective=optimized_objective,
                        lp_recoverable_fraction=lp_recoverable_fraction,
                    )
                )
    return pd.DataFrame(policy_rows), pd.DataFrame(replay_rows)


def delay_feasible_mask(segments: pd.DataFrame, params: Any) -> np.ndarray:
    feasible = np.zeros(len(segments), dtype=bool)
    for intervention in INTERVENTIONS:
        delay = int(params.delays.get(intervention, 0))
        mask = segments["intervention"].eq(intervention).to_numpy()
        feasible[mask] = segments.loc[mask, "t"].to_numpy(dtype=int) >= delay
    return feasible


def add_relative_proxy_metrics(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    keys = ["city", "event_id", "budget_scale"]
    oracle = out[out["policy_score"].eq("greedy_oracle")][keys + ["value_proxy"]].rename(columns={"value_proxy": "oracle_value_proxy"})
    law = out[out["policy_score"].eq("activated_bottleneck_law")][keys + ["value_proxy", "allocated_cost"]].rename(
        columns={"value_proxy": "law_value_proxy", "allocated_cost": "law_allocated_cost"}
    )
    out = out.merge(oracle, on=keys, how="left").merge(law, on=keys, how="left")
    out["relative_to_oracle_proxy"] = out["value_proxy"] / out["oracle_value_proxy"].replace(0.0, np.nan)
    out["law_minus_policy_proxy"] = out["law_value_proxy"] - out["value_proxy"]
    out["law_to_policy_proxy_ratio"] = out["law_value_proxy"] / out["value_proxy"].replace(0.0, np.nan)
    out["law_minus_policy_proxy_per_cost"] = out["law_minus_policy_proxy"] / out["law_allocated_cost"].replace(0.0, np.nan)
    return out


def add_relative_replay_metrics(replay: pd.DataFrame) -> pd.DataFrame:
    out = replay.copy()
    keys = ["city", "event_id", "budget_scale"]
    oracle = out[out["policy_score"].eq("greedy_oracle")][keys + ["replay_gain", "replay_recoverable_fraction"]].rename(
        columns={"replay_gain": "oracle_replay_gain", "replay_recoverable_fraction": "oracle_replay_recoverable_fraction"}
    )
    law = out[out["policy_score"].eq("activated_bottleneck_law")][keys + ["replay_gain", "replay_recoverable_fraction", "allocated_cost"]].rename(
        columns={
            "replay_gain": "law_replay_gain",
            "replay_recoverable_fraction": "law_replay_recoverable_fraction",
            "allocated_cost": "law_allocated_cost",
        }
    )
    out = out.merge(oracle, on=keys, how="left").merge(law, on=keys, how="left")
    out["relative_to_oracle_replay_gain"] = out["replay_gain"] / out["oracle_replay_gain"].replace(0.0, np.nan)
    out["law_minus_policy_replay_gain"] = out["law_replay_gain"] - out["replay_gain"]
    out["law_minus_policy_recoverable_fraction"] = out["law_replay_recoverable_fraction"] - out["replay_recoverable_fraction"]
    out["law_to_policy_replay_gain_ratio"] = out["law_replay_gain"] / out["replay_gain"].replace(0.0, np.nan)
    out["law_minus_policy_replay_gain_per_cost"] = out["law_minus_policy_replay_gain"] / out["law_allocated_cost"].replace(0.0, np.nan)
    return out


def build_event_metrics(policy: pd.DataFrame, replay: pd.DataFrame, root: Path) -> pd.DataFrame:
    keys = ["city", "event_id", "event_start", "budget_scale"]
    policy_pivot = policy.pivot_table(index=keys, columns="policy_score", values="value_proxy", aggfunc="first").reset_index()
    replay_pivot = replay.pivot_table(index=["city", "event_id", "budget_scale"], columns="policy_score", values="replay_gain", aggfunc="first").reset_index()
    replay_pivot = replay_pivot.rename(columns={col: f"replay_gain_{col}" for col in replay_pivot.columns if col not in {"city", "event_id", "budget_scale"}})
    out = policy_pivot.merge(replay_pivot, on=["city", "event_id", "budget_scale"], how="left")
    for baseline in BASELINES:
        out[f"proxy_leverage_vs_{baseline}"] = out["activated_bottleneck_law"] - out[baseline]
        out[f"proxy_ratio_vs_{baseline}"] = out["activated_bottleneck_law"] / out[baseline].replace(0.0, np.nan)
        out[f"replay_gain_leverage_vs_{baseline}"] = out["replay_gain_activated_bottleneck_law"] - out[f"replay_gain_{baseline}"]
        out[f"replay_gain_ratio_vs_{baseline}"] = out["replay_gain_activated_bottleneck_law"] / out[f"replay_gain_{baseline}"].replace(0.0, np.nan)
    out["law_fraction_of_oracle_proxy"] = out["activated_bottleneck_law"] / out["greedy_oracle"].replace(0.0, np.nan)
    out["law_fraction_of_oracle_replay_gain"] = out["replay_gain_activated_bottleneck_law"] / out["replay_gain_greedy_oracle"].replace(0.0, np.nan)
    event_law_path = root / "results" / "law_learning" / "tables" / "event_level_top_tail_law.csv"
    event_law = pd.read_csv(event_law_path)
    event_cols = [
        "city",
        "event_id",
        "baseline_objective",
        "recoverable_fraction",
        "top_5pct_value_share",
        "marginal_value_gini",
        "decision_criticality_score",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    event_cols = [col for col in event_cols if col in event_law]
    return out.merge(event_law[event_cols], on=["city", "event_id"], how="left").sort_values(["budget_scale", "city", "event_id"])


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget_scale, group in events.groupby("budget_scale", sort=True):
        row: dict[str, Any] = {
            "budget_scale": float(budget_scale),
            "n_events": int(len(group)),
            "mean_law_value_proxy": safe_mean(group["activated_bottleneck_law"]),
            "mean_oracle_value_proxy": safe_mean(group["greedy_oracle"]),
            "mean_law_fraction_of_oracle_proxy": safe_mean(group["law_fraction_of_oracle_proxy"]),
            "mean_law_replay_gain": safe_mean(group["replay_gain_activated_bottleneck_law"]),
            "mean_oracle_replay_gain": safe_mean(group["replay_gain_greedy_oracle"]),
            "mean_law_fraction_of_oracle_replay_gain": safe_mean(group["law_fraction_of_oracle_replay_gain"]),
        }
        for baseline in BASELINES:
            row[f"mean_proxy_leverage_vs_{baseline}"] = safe_mean(group[f"proxy_leverage_vs_{baseline}"])
            row[f"mean_proxy_ratio_vs_{baseline}"] = safe_mean(group[f"proxy_ratio_vs_{baseline}"])
            row[f"mean_replay_gain_leverage_vs_{baseline}"] = safe_mean(group[f"replay_gain_leverage_vs_{baseline}"])
            row[f"mean_replay_gain_ratio_vs_{baseline}"] = safe_mean(group[f"replay_gain_ratio_vs_{baseline}"])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("budget_scale").reset_index(drop=True)
    for baseline in BASELINES:
        out[f"mean_proxy_leverage_per_budget_vs_{baseline}"] = out[f"mean_proxy_leverage_vs_{baseline}"] / out["budget_scale"].replace(0.0, np.nan)
        out[f"mean_replay_gain_leverage_per_budget_vs_{baseline}"] = out[f"mean_replay_gain_leverage_vs_{baseline}"] / out["budget_scale"].replace(0.0, np.nan)
    return add_incremental_slopes(out)


def add_incremental_slopes(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    scales = out["budget_scale"].to_numpy(dtype=float)
    for col in [
        "mean_proxy_leverage_vs_random_positive",
        "mean_replay_gain_leverage_vs_random_positive",
        "mean_law_replay_gain",
        "mean_law_value_proxy",
    ]:
        values = out[col].to_numpy(dtype=float)
        slope = np.full(len(out), np.nan)
        if len(out) > 1:
            slope[1:] = np.diff(values) / np.maximum(np.diff(scales), EPS)
        out[f"incremental_{col}"] = slope
    return out


def build_city_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, budget_scale), group in events.groupby(["city", "budget_scale"], sort=True):
        rows.append(
            {
                "city": city,
                "budget_scale": float(budget_scale),
                "n_events": int(len(group)),
                "mean_proxy_leverage_vs_random": safe_mean(group["proxy_leverage_vs_random_positive"]),
                "mean_proxy_leverage_per_budget_vs_random": safe_mean(group["proxy_leverage_vs_random_positive"]) / max(float(budget_scale), EPS),
                "mean_replay_gain_leverage_vs_random": safe_mean(group["replay_gain_leverage_vs_random_positive"]),
                "mean_replay_gain_leverage_per_budget_vs_random": safe_mean(group["replay_gain_leverage_vs_random_positive"]) / max(float(budget_scale), EPS),
                "mean_law_fraction_of_oracle_proxy": safe_mean(group["law_fraction_of_oracle_proxy"]),
                "mean_law_fraction_of_oracle_replay_gain": safe_mean(group["law_fraction_of_oracle_replay_gain"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["city", "budget_scale"])


def build_phase_tests(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_proxy_leverage_vs_random_positive",
        "mean_proxy_leverage_per_budget_vs_random_positive",
        "mean_proxy_ratio_vs_random_positive",
        "mean_replay_gain_leverage_vs_random_positive",
        "mean_replay_gain_leverage_per_budget_vs_random_positive",
        "mean_replay_gain_ratio_vs_random_positive",
        "mean_law_fraction_of_oracle_proxy",
        "mean_law_fraction_of_oracle_replay_gain",
        "incremental_mean_proxy_leverage_vs_random_positive",
        "incremental_mean_replay_gain_leverage_vs_random_positive",
    ]
    rows = [phase_row(summary, metric) for metric in metrics if metric in summary]
    return pd.DataFrame(rows)


def build_city_phase_tests(city_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city, group in city_summary.groupby("city", sort=True):
        for metric in [
            "mean_proxy_leverage_vs_random",
            "mean_proxy_leverage_per_budget_vs_random",
            "mean_replay_gain_leverage_vs_random",
            "mean_replay_gain_leverage_per_budget_vs_random",
        ]:
            row = phase_row(group, metric)
            row["city"] = city
            rows.append(row)
    return pd.DataFrame(rows)


def phase_row(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    work = frame[["budget_scale", metric]].dropna().sort_values("budget_scale")
    if work.empty:
        return {"metric": metric}
    values = work[metric].to_numpy(dtype=float)
    scales = work["budget_scale"].to_numpy(dtype=float)
    peak_pos = int(np.nanargmax(values))
    peak_scale = float(scales[peak_pos])
    interior = 0 < peak_pos < len(values) - 1
    diffs = np.diff(values)
    return {
        "metric": metric,
        "peak_budget_scale": peak_scale,
        "peak_value": float(values[peak_pos]),
        "min_budget_value": float(values[0]),
        "max_budget_value": float(values[-1]),
        "interior_peak_supported": bool(interior),
        "monotone_increasing": bool(np.all(diffs >= -1e-10)) if len(diffs) else False,
        "monotone_decreasing": bool(np.all(diffs <= 1e-10)) if len(diffs) else False,
        "first_to_peak_gain": float(values[peak_pos] - values[0]),
        "peak_to_last_change": float(values[-1] - values[peak_pos]),
    }


def build_diagnostics(summary: pd.DataFrame, phase_tests: pd.DataFrame, city_phase: pd.DataFrame) -> dict[str, Any]:
    def get_phase(metric: str) -> pd.Series:
        match = phase_tests[phase_tests["metric"].eq(metric)]
        return match.iloc[0] if not match.empty else pd.Series(dtype=object)

    proxy_abs = get_phase("mean_proxy_leverage_vs_random_positive")
    proxy_per_budget = get_phase("mean_proxy_leverage_per_budget_vs_random_positive")
    replay_abs = get_phase("mean_replay_gain_leverage_vs_random_positive")
    replay_per_budget = get_phase("mean_replay_gain_leverage_per_budget_vs_random_positive")
    ratio = get_phase("mean_replay_gain_ratio_vs_random_positive")
    city_abs = city_phase[city_phase["metric"].eq("mean_replay_gain_leverage_vs_random")]
    interior_city_share = safe_mean(city_abs["interior_peak_supported"].astype(float)) if not city_abs.empty else np.nan
    base_row = nearest_budget_row(summary, 1.0)
    return {
        "n_budget_scales": int(summary["budget_scale"].nunique()),
        "min_budget_scale": float(summary["budget_scale"].min()),
        "max_budget_scale": float(summary["budget_scale"].max()),
        "proxy_abs_peak_budget": safe_float(proxy_abs.get("peak_budget_scale")),
        "proxy_abs_interior_peak_supported": bool(proxy_abs.get("interior_peak_supported", False)),
        "proxy_abs_monotone_increasing": bool(proxy_abs.get("monotone_increasing", False)),
        "proxy_per_budget_peak_budget": safe_float(proxy_per_budget.get("peak_budget_scale")),
        "proxy_per_budget_monotone_decreasing": bool(proxy_per_budget.get("monotone_decreasing", False)),
        "replay_abs_peak_budget": safe_float(replay_abs.get("peak_budget_scale")),
        "replay_abs_interior_peak_supported": bool(replay_abs.get("interior_peak_supported", False)),
        "replay_abs_monotone_increasing": bool(replay_abs.get("monotone_increasing", False)),
        "replay_per_budget_peak_budget": safe_float(replay_per_budget.get("peak_budget_scale")),
        "replay_per_budget_monotone_decreasing": bool(replay_per_budget.get("monotone_decreasing", False)),
        "replay_ratio_peak_budget": safe_float(ratio.get("peak_budget_scale")),
        "replay_ratio_monotone_decreasing": bool(ratio.get("monotone_decreasing", False)),
        "city_replay_abs_interior_peak_share": interior_city_share,
        "base_budget_law_fraction_of_oracle_proxy": safe_float(base_row.get("mean_law_fraction_of_oracle_proxy")),
        "base_budget_law_fraction_of_oracle_replay_gain": safe_float(base_row.get("mean_law_fraction_of_oracle_replay_gain")),
        "base_budget_replay_gain_leverage_vs_random": safe_float(base_row.get("mean_replay_gain_leverage_vs_random_positive")),
        "base_budget_replay_gain_leverage_per_budget_vs_random": safe_float(base_row.get("mean_replay_gain_leverage_per_budget_vs_random_positive")),
    }


def nearest_budget_row(summary: pd.DataFrame, budget_scale: float) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    idx = (summary["budget_scale"].astype(float) - budget_scale).abs().idxmin()
    return summary.loc[idx]


def make_figures(
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    phase_tests: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_budget_curve(summary, figure_dir / "fine_budget_leverage_curve.png")
    make_per_budget_curve(summary, figure_dir / "fine_budget_per_budget_leverage.png")
    make_city_heatmap(city_summary, figure_dir / "fine_budget_city_heatmap.png")
    make_top_tail_scatter(event_metrics, figure_dir / "fine_budget_top_tail_relation.png")
    make_phase_test_figure(phase_tests, figure_dir / "fine_budget_phase_tests.png")


def make_budget_curve(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = summary["budget_scale"].to_numpy(dtype=float)
    ax.plot(x, summary["mean_proxy_leverage_vs_random_positive"], marker="o", label="proxy: law - random")
    ax.plot(x, summary["mean_replay_gain_leverage_vs_random_positive"], marker="o", label="replay: law - random")
    ax.plot(x, summary["mean_replay_gain_leverage_vs_exposure_only"], marker="o", label="replay: law - exposure")
    ax.set_xlabel("Budget scale")
    ax.set_ylabel("Absolute decision leverage")
    ax.set_title("Fine budget sweep: absolute leverage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_per_budget_curve(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = summary["budget_scale"].to_numpy(dtype=float)
    ax.plot(x, summary["mean_proxy_leverage_per_budget_vs_random_positive"], marker="o", label="proxy per budget")
    ax.plot(x, summary["mean_replay_gain_leverage_per_budget_vs_random_positive"], marker="o", label="replay per budget")
    ax.plot(x, summary["mean_replay_gain_ratio_vs_random_positive"], marker="s", linestyle="--", label="replay law/random ratio")
    ax.set_xlabel("Budget scale")
    ax.set_ylabel("Relative or per-budget leverage")
    ax.set_title("Fine budget sweep: diminishing leverage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_heatmap(city_summary: pd.DataFrame, path: Path) -> None:
    pivot = city_summary.pivot_table(index="city", columns="budget_scale", values="mean_replay_gain_leverage_vs_random")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{col:g}" for col in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Budget scale")
    ax.set_title("City heterogeneity in replay leverage versus random")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Law minus random replay gain")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_top_tail_scatter(event_metrics: pd.DataFrame, path: Path) -> None:
    base = event_metrics[np.isclose(event_metrics["budget_scale"].astype(float), 1.0)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(
        base["top_5pct_value_share"],
        base["replay_gain_leverage_vs_random_positive"],
        c=base["baseline_objective"],
        cmap="viridis",
        s=58,
        alpha=0.84,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Top-5% marginal value share")
    ax.set_ylabel("Law minus random replay gain")
    ax.set_title("Budget leverage tracks value concentration at base budget")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("No-intervention loss objective")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_phase_test_figure(phase_tests: pd.DataFrame, path: Path) -> None:
    keep = phase_tests[
        phase_tests["metric"].isin(
            [
                "mean_proxy_leverage_vs_random_positive",
                "mean_proxy_leverage_per_budget_vs_random_positive",
                "mean_replay_gain_leverage_vs_random_positive",
                "mean_replay_gain_leverage_per_budget_vs_random_positive",
                "mean_replay_gain_ratio_vs_random_positive",
            ]
        )
    ].copy()
    if keep.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels = [
        label.replace("mean_", "").replace("_vs_random_positive", "").replace("_", " ")
        for label in keep["metric"]
    ]
    colors = np.where(keep["interior_peak_supported"].astype(bool), "#2563eb", "#94a3b8")
    ax.barh(labels, keep["peak_budget_scale"], color=colors)
    ax.set_xlabel("Peak budget scale")
    ax.set_title("Peak location of budget-leverage metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    phase_tests: pd.DataFrame,
    city_phase: pd.DataFrame,
) -> None:
    base_city = city_summary[np.isclose(city_summary["budget_scale"].astype(float), 1.0)].sort_values(
        "mean_replay_gain_leverage_vs_random",
        ascending=False,
    )
    lines = [
        "# Fine Budget-Leverage Sweep V27",
        "",
        "## 这一版做了什么",
        "",
        "V27 将预算从原来的 low/base/high 三点扩展为更细的预算网格，并在每个 city-event 上重放 greedy-oracle、activated law、deficit-only、exposure-only、structure-only 和 random-positive 策略。它不重新求解每个预算下的完整 LP，而是使用同一套 action-value field 与 LP replay 动力学来区分三件事：绝对决策杠杆、相对优势，以及单位预算杠杆。",
        "",
        "## 主要结论",
        "",
        f"- budget scales: {diagnostics['n_budget_scales']} points from {diagnostics['min_budget_scale']:.2f} to {diagnostics['max_budget_scale']:.2f}.",
        f"- proxy absolute law-random leverage peaks at budget scale {diagnostics['proxy_abs_peak_budget']:.2f}; interior peak supported = {diagnostics['proxy_abs_interior_peak_supported']}.",
        f"- replay absolute law-random leverage peaks at budget scale {diagnostics['replay_abs_peak_budget']:.2f}; interior peak supported = {diagnostics['replay_abs_interior_peak_supported']}.",
        f"- replay law/random ratio peaks at budget scale {diagnostics['replay_ratio_peak_budget']:.2f}; monotone decreasing = {diagnostics['replay_ratio_monotone_decreasing']}.",
        f"- replay leverage per budget peaks at budget scale {diagnostics['replay_per_budget_peak_budget']:.2f}; monotone decreasing = {diagnostics['replay_per_budget_monotone_decreasing']}.",
        f"- at base budget, law captures {diagnostics['base_budget_law_fraction_of_oracle_replay_gain']:.4f} of greedy-oracle replay gain and exceeds random by {diagnostics['base_budget_replay_gain_leverage_vs_random']:.4f} replay-gain units.",
        "",
        "这版把 high-level idea 里的“中等预算可能最有 decision leverage”改写得更精确：当前数据不支持绝对 law-random 杠杆在中等预算出现内点峰值；更稳健的规律是 scale-dependent diminishing leverage。预算越高，law 相比 random/简单规则能多恢复的绝对量仍会上升或进入平台，但相对优势和单位预算优势在低预算附近最高，并随预算增加下降。",
        "",
        "## Budget Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Phase Tests",
        "",
        table_to_markdown(phase_tests),
        "",
        "## Base-Budget City Ranking",
        "",
        table_to_markdown(base_city),
        "",
        "## City Phase Tests",
        "",
        table_to_markdown(city_phase),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if len(values.dropna()) else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def table_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_empty_"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

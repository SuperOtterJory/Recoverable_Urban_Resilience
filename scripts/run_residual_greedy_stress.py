"""Stress-test the residual finite-budget recovery law across budget and delay scenarios."""

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
    POLICY_SCENARIOS,
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
from run_residual_greedy_policy import allocate_residual_greedy


SCENARIO_ORDER = ["low_budget", "base", "high_budget", "delay_2h", "delay_4h", "scarce_and_late"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/residual_greedy_stress")
    parser.add_argument("--replan-budget-share", type=float, default=0.10)
    parser.add_argument("--max-replans", type=int, default=40)
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

    data = load_inputs(root)
    reference = pd.read_csv(root / "results" / "law_learning" / "tables" / "fixed_policy_replay.csv")
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    if args.max_events is not None:
        summary = summary.sort_values(["city", "event_start", "event_id"]).head(args.max_events).copy()

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()

    replay_rows: list[dict[str, Any]] = []
    allocation_frames: list[pd.DataFrame] = []
    pass_frames: list[pd.DataFrame] = []
    total_jobs = len(summary) * len(POLICY_SCENARIOS)
    job_idx = 0
    rng = np.random.default_rng(20260529)
    for _, row in summary.sort_values(["city", "event_start", "event_id"]).iterrows():
        city = str(row["city"])
        event_id = int(row["event_id"])
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

        for scenario in POLICY_SCENARIOS:
            job_idx += 1
            scenario_name = str(scenario["policy_scenario"])
            budget_scale = float(scenario["budget_scale"])
            delay_add = int(scenario["delay_add_hours"])
            print(f"[{job_idx}/{total_jobs}] Residual stress {city} event {event_id} / {scenario_name}", flush=True)
            scenario_delays = {key: int(value) + delay_add for key, value in params.delays.items()}
            scenario_params = params.copy_with_budget(budget_scale, delays=scenario_delays)
            result = allocate_residual_greedy(
                segments,
                scenario_params,
                baseline_objective=float(row["baseline_objective"]),
                replan_budget_share=float(args.replan_budget_share),
                max_replans=int(args.max_replans),
            )
            allocations = result["allocations"]
            if not allocations.empty:
                allocations = allocations.copy()
                allocations["policy_scenario"] = scenario_name
                allocations["budget_scale"] = budget_scale
                allocations["delay_add_hours"] = delay_add
                allocation_frames.append(allocations)
            passes = pd.DataFrame(result["pass_rows"])
            if not passes.empty:
                passes["policy_scenario"] = scenario_name
                passes["budget_scale"] = budget_scale
                passes["delay_add_hours"] = delay_add
                pass_frames.append(passes)
            replay = replay_policy_allocations(allocations, scenario_params)
            replay_rows.append(
                replay_row(
                    full,
                    policy_scenario=scenario_name,
                    budget_scale=budget_scale,
                    delay_add_hours=delay_add,
                    policy_score="residual_finite_greedy",
                    allocated_cost=float(result["allocated_cost"]),
                    value_proxy=float(result["value_proxy"]),
                    selected_action_count=int(result["selected_action_count"]),
                    replay_objective=replay["objective"],
                    baseline_objective=float(row["baseline_objective"]),
                    optimized_objective=float(row["optimized_objective"]),
                    lp_recoverable_fraction=float(row["recoverable_fraction"]),
                )
            )

    residual_replay = pd.DataFrame(replay_rows)
    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    passes = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
    event_metrics = build_stress_event_metrics(reference, residual_replay)
    scenario_summary = summarize_by_scenario(event_metrics)
    city_scenario_summary = summarize_by_city_scenario(event_metrics)
    pass_summary = summarize_passes(passes)

    write_table(residual_replay, table_dir / "residual_stress_replay.csv")
    write_table(event_metrics, table_dir / "residual_stress_event_metrics.csv")
    write_table(scenario_summary, table_dir / "residual_stress_scenario_summary.csv")
    write_table(city_scenario_summary, table_dir / "residual_stress_city_scenario_summary.csv")
    write_table(pass_summary, table_dir / "residual_stress_pass_summary.csv")
    write_table(allocations, table_dir / "residual_stress_allocations.csv.gz")
    write_table(passes, table_dir / "residual_stress_passes.csv.gz")

    make_figures(event_metrics, scenario_summary, city_scenario_summary, figure_dir)
    write_report(
        report_dir / "residual_greedy_stress_report_zh.md",
        event_metrics,
        scenario_summary,
        city_scenario_summary,
        pass_summary,
        float(args.replan_budget_share),
    )
    print(f"Wrote residual greedy stress test to {output_dir}")


def build_stress_event_metrics(reference: pd.DataFrame, residual_replay: pd.DataFrame) -> pd.DataFrame:
    static = reference[
        reference["policy_score"].eq("greedy_oracle")
        & reference["policy_scenario"].isin(SCENARIO_ORDER)
    ].copy()
    static["event_id"] = pd.to_numeric(static["event_id"], errors="coerce").astype(int)
    residual = residual_replay.copy()
    residual["event_id"] = pd.to_numeric(residual["event_id"], errors="coerce").astype(int)
    keys = ["city", "event_id", "event_start", "policy_scenario", "budget_scale", "delay_add_hours"]
    static_cols = keys + [
        "baseline_objective",
        "optimized_objective",
        "lp_recoverable_fraction",
        "replay_gain",
        "replay_recoverable_fraction",
        "replay_fraction_of_base_lp_gain",
        "selected_action_count",
        "allocated_cost",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    residual_cols = keys + [
        "replay_gain",
        "replay_recoverable_fraction",
        "replay_fraction_of_base_lp_gain",
        "selected_action_count",
        "allocated_cost",
        "value_proxy",
    ]
    out = static[static_cols].merge(
        residual[residual_cols],
        on=keys,
        how="inner",
        suffixes=("_static", "_residual"),
    )
    out = out.rename(
        columns={
            "replay_gain_static": "static_replay_gain",
            "replay_recoverable_fraction_static": "static_replay_recoverable_fraction",
            "replay_fraction_of_base_lp_gain_static": "static_fraction_of_base_lp_gain",
            "selected_action_count_static": "static_selected_action_count",
            "allocated_cost_static": "static_allocated_cost",
            "replay_gain_residual": "residual_replay_gain",
            "replay_recoverable_fraction_residual": "residual_replay_recoverable_fraction",
            "replay_fraction_of_base_lp_gain_residual": "residual_fraction_of_base_lp_gain",
            "selected_action_count_residual": "residual_selected_action_count",
            "allocated_cost_residual": "residual_allocated_cost",
            "value_proxy": "residual_value_proxy",
        }
    )
    out["residual_minus_static_fraction_of_base_lp_gain"] = (
        out["residual_fraction_of_base_lp_gain"] - out["static_fraction_of_base_lp_gain"]
    )
    out["residual_minus_static_recoverable_fraction"] = (
        out["residual_replay_recoverable_fraction"] - out["static_replay_recoverable_fraction"]
    )
    out["residual_relative_to_static_gain"] = out["residual_replay_gain"] / out["static_replay_gain"].replace(0.0, np.nan)
    out["scenario_order"] = out["policy_scenario"].map({name: idx for idx, name in enumerate(SCENARIO_ORDER)})
    return out.sort_values(["scenario_order", "city", "event_start", "event_id"]).drop(columns=["scenario_order"])


def summarize_by_scenario(event_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        event_metrics.groupby(["policy_scenario", "budget_scale", "delay_add_hours"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_static_fraction_of_base_lp_gain=("static_fraction_of_base_lp_gain", "mean"),
            mean_residual_fraction_of_base_lp_gain=("residual_fraction_of_base_lp_gain", "mean"),
            median_residual_fraction_of_base_lp_gain=("residual_fraction_of_base_lp_gain", "median"),
            mean_residual_minus_static=("residual_minus_static_fraction_of_base_lp_gain", "mean"),
            median_residual_minus_static=("residual_minus_static_fraction_of_base_lp_gain", "median"),
            positive_improvement_share=("residual_minus_static_fraction_of_base_lp_gain", lambda x: float((x > 1e-6).mean())),
            mean_static_recoverable_fraction=("static_replay_recoverable_fraction", "mean"),
            mean_residual_recoverable_fraction=("residual_replay_recoverable_fraction", "mean"),
            mean_residual_selected_actions=("residual_selected_action_count", "mean"),
            mean_residual_allocated_cost=("residual_allocated_cost", "mean"),
        )
    )
    summary["scenario_order"] = summary["policy_scenario"].map({name: idx for idx, name in enumerate(SCENARIO_ORDER)})
    return summary.sort_values("scenario_order").drop(columns=["scenario_order"])


def summarize_by_city_scenario(event_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        event_metrics.groupby(["city", "policy_scenario"], as_index=False)
        .agg(
            n_events=("event_id", "count"),
            mean_static_fraction_of_base_lp_gain=("static_fraction_of_base_lp_gain", "mean"),
            mean_residual_fraction_of_base_lp_gain=("residual_fraction_of_base_lp_gain", "mean"),
            mean_residual_minus_static=("residual_minus_static_fraction_of_base_lp_gain", "mean"),
            positive_improvement_share=("residual_minus_static_fraction_of_base_lp_gain", lambda x: float((x > 1e-6).mean())),
        )
    )
    summary["scenario_order"] = summary["policy_scenario"].map({name: idx for idx, name in enumerate(SCENARIO_ORDER)})
    return summary.sort_values(["scenario_order", "city"]).drop(columns=["scenario_order"])


def summarize_passes(passes: pd.DataFrame) -> pd.DataFrame:
    if passes.empty:
        return pd.DataFrame()
    return (
        passes.groupby(["city", "event_id", "policy_scenario", "budget_scale", "delay_add_hours"], as_index=False)
        .agg(
            n_replans=("pass_id", "count"),
            mean_pass_cost=("allocated_cost", "mean"),
            final_remaining_total=("remaining_total", "last"),
        )
        .sort_values(["policy_scenario", "city", "event_id"])
    )


def make_figures(event_metrics: pd.DataFrame, scenario_summary: pd.DataFrame, city_scenario_summary: pd.DataFrame, figure_dir: Path) -> None:
    scenario_summary = scenario_summary.copy()
    scenario_summary["policy_scenario"] = pd.Categorical(scenario_summary["policy_scenario"], categories=SCENARIO_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(scenario_summary))
    width = 0.36
    ax.bar(
        x - width / 2,
        scenario_summary["mean_static_fraction_of_base_lp_gain"],
        width=width,
        label="static small-signal greedy",
        color="#94a3b8",
    )
    ax.bar(
        x + width / 2,
        scenario_summary["mean_residual_fraction_of_base_lp_gain"],
        width=width,
        label="residual finite greedy",
        color="#2563eb",
    )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_summary["policy_scenario"].astype(str), rotation=20)
    ax.set_ylabel("Mean replay gain / base LP gain")
    ax.set_title("Residual law under budget and delay stress")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_stress_by_scenario.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(
        scenario_summary["policy_scenario"].astype(str),
        scenario_summary["mean_residual_minus_static"],
        marker="o",
        linewidth=2.4,
        color="#0f766e",
    )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_ylabel("Residual improvement over static\n(fraction of base LP gain)")
    ax.set_title("Residual replanning improvement by scenario")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_stress_improvement.png", dpi=180)
    plt.close(fig)

    pivot = city_scenario_summary.pivot(index="city", columns="policy_scenario", values="mean_residual_minus_static")
    pivot = pivot.reindex(columns=SCENARIO_ORDER)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Residual improvement across cities and scenarios")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Improvement over static")
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_stress_city_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    base = event_metrics[event_metrics["policy_scenario"].eq("base")]
    delay = event_metrics[event_metrics["policy_scenario"].eq("delay_4h")]
    merged = base[["city", "event_id", "residual_fraction_of_base_lp_gain"]].merge(
        delay[["city", "event_id", "residual_fraction_of_base_lp_gain"]],
        on=["city", "event_id"],
        suffixes=("_base", "_delay4"),
    )
    ax.scatter(
        merged["residual_fraction_of_base_lp_gain_base"],
        merged["residual_fraction_of_base_lp_gain_delay4"],
        s=55,
        alpha=0.78,
        color="#7c3aed",
    )
    ax.plot([0, max(1.05, merged.max(numeric_only=True).max())], [0, max(1.05, merged.max(numeric_only=True).max())], "--", color="#111827", alpha=0.4)
    ax.set_xlabel("Base residual gain / base LP gain")
    ax.set_ylabel("Delay-4h residual gain / base LP gain")
    ax.set_title("Event stability under delayed response")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_base_vs_delay4.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    event_metrics: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    city_scenario_summary: pd.DataFrame,
    pass_summary: pd.DataFrame,
    replan_budget_share: float,
) -> None:
    base = scenario_summary[scenario_summary["policy_scenario"].eq("base")].iloc[0]
    low = scenario_summary[scenario_summary["policy_scenario"].eq("low_budget")].iloc[0]
    high = scenario_summary[scenario_summary["policy_scenario"].eq("high_budget")].iloc[0]
    delay4 = scenario_summary[scenario_summary["policy_scenario"].eq("delay_4h")].iloc[0]
    lines = [
        "# Residual Greedy Stress Test V8",
        "",
        "## 这一版做了什么",
        "",
        "V7 证明 residual finite greedy 在 base scenario 中几乎闭合 LP optimum。V8 进一步检验这个 finite-budget law 是否稳定：同一批 105 个 city-event，在 low/high budget、2/4 小时额外响应延迟、以及 scarce-and-late 条件下重新运行 residual replanning，并与 V5 的 static small-signal greedy replay 对比。",
        "",
        f"本版每次 replan 最多使用总预算的 {replan_budget_share:.2%}，因此它不是一次性排序，而是在同一个 scenario 内反复更新 residual state。",
        "",
        "## 关键结果",
        "",
        f"- base: static = {base.mean_static_fraction_of_base_lp_gain:.4f}, residual = {base.mean_residual_fraction_of_base_lp_gain:.4f}, improvement = {base.mean_residual_minus_static:.4f}",
        f"- low budget: static = {low.mean_static_fraction_of_base_lp_gain:.4f}, residual = {low.mean_residual_fraction_of_base_lp_gain:.4f}, improvement = {low.mean_residual_minus_static:.4f}",
        f"- high budget: static = {high.mean_static_fraction_of_base_lp_gain:.4f}, residual = {high.mean_residual_fraction_of_base_lp_gain:.4f}, improvement = {high.mean_residual_minus_static:.4f}",
        f"- delay 4h: static = {delay4.mean_static_fraction_of_base_lp_gain:.4f}, residual = {delay4.mean_residual_fraction_of_base_lp_gain:.4f}, improvement = {delay4.mean_residual_minus_static:.4f}",
        "",
        "注意：非 base scenario 没有重新求解 Gurobi optimum，因此表中的 gain 都以 base LP gain 为归一化参照。它检验的是 law-guided policy 在参数扰动下相对 static greedy 是否稳定，而不是宣称这些新 scenario 下已经达到各自的 LP optimum。",
        "",
        "## Scenario Summary",
        "",
        dataframe_to_markdown(scenario_summary),
        "",
        "## City-Scenario Summary",
        "",
        dataframe_to_markdown(city_scenario_summary, max_rows=60),
        "",
        "## Replan Summary",
        "",
        dataframe_to_markdown(pass_summary.head(30)),
        "",
        "## 科学解释",
        "",
        "Residual law 在低预算、延迟和 scarce-and-late 场景下仍系统性优于 static small-signal greedy，说明 V7 发现不是 base 参数的偶然结果。它支持一个更稳定的 finite-budget law：恢复价值应当写成 `value(segment | residual state, remaining budget, remaining time)`，而不是只写成事件开始时的固定 action score。",
        "",
        "高预算场景中 residual gain 可以超过 base LP gain，这是正常的，因为归一化分母仍然是 base LP optimized gain，而不是 high-budget 重新优化后的 optimum。下一步如果要做完全闭合的 robustness，需要对若干代表性 budget/delay scenario 重新求解 Gurobi LP，比较 residual policy 与各自 scenario optimum。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

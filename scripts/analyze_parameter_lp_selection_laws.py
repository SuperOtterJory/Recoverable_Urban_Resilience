"""Audit full-LP selected actions under parameter-ensemble scenarios."""

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

from learn_recovery_laws import build_event_action_frame, load_inputs
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters, solve_recovery_lp
from validate_parameter_ensemble_optimum import PARAMETER_SCENARIOS, apply_parameter_scenario


EPS = 1e-12
SCORE_SPECS = [
    ("target", "target_value"),
    ("activated", "activated_bottleneck_score"),
    ("deficit", "deficit_only_score"),
    ("exposure", "exposure_only_score"),
    ("structure", "structure_only_score"),
]
FRACTIONS = [0.05, 0.10, 0.20]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--input-dir", default="results/parameter_ensemble_optimum_validation")
    parser.add_argument("--output-dir", default="results/parameter_lp_selection_laws")
    parser.add_argument("--time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))

    input_dir = root / args.input_dir / "tables"
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_actions": table_dir / "parameter_lp_selected_actions.csv",
        "event_metrics": table_dir / "parameter_lp_selection_event_metrics.csv",
        "scenario_summary": table_dir / "parameter_lp_selection_scenario_summary.csv",
        "city_summary": table_dir / "parameter_lp_selection_city_summary.csv",
        "metrics": table_dir / "parameter_lp_selection_metrics.json",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    selected_events = pd.read_csv(input_dir / "selected_events.csv", parse_dates=["event_start"])
    scenario_rows = pd.read_csv(input_dir / "parameter_lp_scenarios.csv")
    scenarios = scenario_dicts(scenario_rows)
    data = load_inputs(root)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    abnormal = data["abnormal"].copy()
    empty_interventions = empty_optimized_interventions()
    completed = completed_keys(paths["event_metrics"]) if args.resume else set()

    total_jobs = len(selected_events) * len(scenarios)
    job_idx = 0
    for _, base_row in selected_events.iterrows():
        city = str(base_row["city"])
        event_id = int(base_row["event_id"])
        event_row = event_lookup.get((city, event_id))
        if event_row is None or city not in dynamic_lookup:
            continue
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        for scenario in scenarios:
            job_idx += 1
            scenario_name = str(scenario["parameter_scenario"])
            key = (city, event_id, scenario_name)
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed {city} event {event_id} / {scenario_name}", flush=True)
                continue
            print(f"[{job_idx}/{total_jobs}] Solving LP selection audit {city} event {event_id} / {scenario_name}", flush=True)
            scenario_params = apply_parameter_scenario(base_params, scenario)
            optimized = solve_recovery_lp(
                scenario_params,
                output_flag=bool(solver.get("output_flag", False)),
                method=method,
                time_limit_seconds=float(args.time_limit_seconds),
            )
            full = build_event_action_frame(scenario_params, base_row, event_row, empty_interventions)
            full["scenario"] = scenario_name
            full["parameter_scenario"] = scenario_name
            full["parameter_description"] = str(scenario["description"])
            full["target_value"] = full["marginal_resource_value"].fillna(0.0).clip(lower=0.0)
            add_top_flags(full)

            selected_actions, event_metrics = analyze_lp_selection(
                full,
                optimized.interventions,
                base_row,
                scenario,
                optimized_objective=float(optimized.objective),
                lp_status=str(optimized.status),
                runtime_seconds=float(optimized.runtime_seconds),
            )
            append_csv(selected_actions, paths["selected_actions"])
            append_csv(pd.DataFrame([event_metrics]), paths["event_metrics"])

    event_metrics_all = pd.read_csv(paths["event_metrics"]) if paths["event_metrics"].exists() else pd.DataFrame()
    selected_actions_all = pd.read_csv(paths["selected_actions"]) if paths["selected_actions"].exists() else pd.DataFrame()
    scenario_summary = build_scenario_summary(event_metrics_all)
    city_summary = build_city_summary(event_metrics_all)
    diagnostics = build_diagnostics(event_metrics_all, scenario_summary, selected_actions_all)
    write_table(scenario_summary, paths["scenario_summary"])
    write_table(city_summary, paths["city_summary"])
    paths["metrics"].write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(event_metrics_all, scenario_summary, figure_dir)
    write_report(
        report_dir / "parameter_lp_selection_laws_report_zh.md",
        diagnostics,
        scenario_summary,
        city_summary,
        event_metrics_all,
        selected_actions_all,
    )
    print(f"Wrote parameter LP selection-law audit to {output_dir}")


def scenario_dicts(scenario_rows: pd.DataFrame) -> list[dict[str, Any]]:
    available = {str(row["parameter_scenario"]): dict(row) for row in PARAMETER_SCENARIOS}
    scenarios: list[dict[str, Any]] = []
    for name in scenario_rows["parameter_scenario"].astype(str).tolist():
        if name not in available:
            raise ValueError(f"Scenario {name} is not defined in PARAMETER_SCENARIOS.")
        scenarios.append(available[name])
    return scenarios


def empty_optimized_interventions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["city", "event_id", "scenario", "unit", "t", "intervention", "optimized_u", "optimized_e", "optimized_cost"]
    )


def add_top_flags(full: pd.DataFrame) -> None:
    for score_name, score_col in SCORE_SPECS:
        for frac in FRACTIONS:
            flag = f"{score_name}_top{int(frac * 100)}"
            full[flag] = mark_top_fraction(full, score_col, frac)
    full["simple_union_top5"] = full[[f"{score}_top5" for score in ["deficit", "exposure", "structure"]]].any(axis=1)
    full["simple_union_top10"] = full[[f"{score}_top10" for score in ["deficit", "exposure", "structure"]]].any(axis=1)
    full["simple_union_top20"] = full[[f"{score}_top20" for score in ["deficit", "exposure", "structure"]]].any(axis=1)
    for score_name, score_col in SCORE_SPECS:
        full[f"{score_name}_rank_pct"] = full[score_col].rank(method="average", pct=True).fillna(0.0)


def mark_top_fraction(frame: pd.DataFrame, score_col: str, frac: float) -> pd.Series:
    scores = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0)
    k = max(1, int(np.ceil(len(frame) * frac)))
    top_index = scores.sort_values(ascending=False).head(k).index
    out = pd.Series(False, index=frame.index)
    out.loc[top_index] = True
    return out


def analyze_lp_selection(
    full: pd.DataFrame,
    lp_interventions: pd.DataFrame,
    base_row: pd.Series,
    scenario: dict[str, Any],
    *,
    optimized_objective: float,
    lp_status: str,
    runtime_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lp = lp_interventions.copy()
    lp["unit"] = lp["unit"].astype(str)
    lp["t"] = pd.to_numeric(lp["t"], errors="coerce").astype(int)
    lp["lp_u"] = pd.to_numeric(lp["u"], errors="coerce").fillna(0.0)
    lp["lp_e"] = pd.to_numeric(lp["e"], errors="coerce").fillna(0.0)
    lp["lp_effective_cost"] = pd.to_numeric(lp["effective_cost"], errors="coerce").fillna(0.0)
    lp = lp[["unit", "t", "intervention", "lp_u", "lp_e", "lp_effective_cost"]]
    merged = full.merge(lp, on=["unit", "t", "intervention"], how="left")
    for col in ["lp_u", "lp_e", "lp_effective_cost"]:
        merged[col] = merged[col].fillna(0.0)
    merged["lp_selected"] = (merged["lp_u"] > EPS) | (merged["lp_e"] > EPS) | (merged["lp_effective_cost"] > EPS)
    selected = merged[merged["lp_selected"]].copy()
    total_cost = float(selected["lp_effective_cost"].sum())
    total_effect = float(selected["lp_e"].sum())
    baseline_objective = float(base_row["baseline_objective"])
    scenario_lp_gain = max(baseline_objective - optimized_objective, EPS)

    row: dict[str, Any] = {
        "city": str(base_row["city"]),
        "event_id": int(base_row["event_id"]),
        "event_start": str(base_row["event_start"]),
        "parameter_scenario": str(scenario["parameter_scenario"]),
        "parameter_description": str(scenario["description"]),
        "lp_status": lp_status,
        "runtime_seconds": runtime_seconds,
        "baseline_objective": baseline_objective,
        "scenario_optimized_objective": optimized_objective,
        "scenario_lp_gain": scenario_lp_gain,
        "scenario_lp_recoverable_fraction": optimized_objective_to_fraction(baseline_objective, optimized_objective),
        "lp_selected_action_count": int(len(selected)),
        "lp_selected_total_cost": total_cost,
        "lp_selected_total_effect": total_effect,
    }
    for score_name, _ in SCORE_SPECS:
        row[f"{score_name}_selected_cost_weighted_rank"] = weighted_mean(selected[f"{score_name}_rank_pct"], selected["lp_effective_cost"])
        row[f"{score_name}_selected_effect_weighted_rank"] = weighted_mean(selected[f"{score_name}_rank_pct"], selected["lp_e"])
        for frac in FRACTIONS:
            flag = f"{score_name}_top{int(frac * 100)}"
            row[f"{score_name}_top{int(frac * 100)}_selected_cost_share"] = weighted_share(selected, flag, "lp_effective_cost", total_cost)
            row[f"{score_name}_top{int(frac * 100)}_selected_effect_share"] = weighted_share(selected, flag, "lp_e", total_effect)
            row[f"{score_name}_top{int(frac * 100)}_selected_count_share"] = count_share(selected, flag)
    for frac in FRACTIONS:
        flag = f"simple_union_top{int(frac * 100)}"
        row[f"{flag}_selected_cost_share"] = weighted_share(selected, flag, "lp_effective_cost", total_cost)
        row[f"{flag}_selected_effect_share"] = weighted_share(selected, flag, "lp_e", total_effect)
        row[f"hidden_from_{flag}_selected_cost_share"] = 1.0 - row[f"{flag}_selected_cost_share"]
    row["target_top20_hidden_from_simple_top20_selected_cost_share"] = weighted_share(
        selected,
        selected["target_top20"] & ~selected["simple_union_top20"],
        "lp_effective_cost",
        total_cost,
    )
    row["target_top20_hidden_from_simple_top5_selected_cost_share"] = weighted_share(
        selected,
        selected["target_top20"] & ~selected["simple_union_top5"],
        "lp_effective_cost",
        total_cost,
    )
    row["below_structure_top20_selected_cost_share"] = weighted_share(
        selected,
        ~selected["structure_top20"],
        "lp_effective_cost",
        total_cost,
    )
    row["best_simple_top20_selected_cost_share"] = max(
        row["deficit_top20_selected_cost_share"],
        row["exposure_top20_selected_cost_share"],
        row["structure_top20_selected_cost_share"],
    )
    row["activated_minus_best_simple_top20_selected_cost_share"] = (
        row["activated_top20_selected_cost_share"] - row["best_simple_top20_selected_cost_share"]
    )
    for key in INTERVENTIONS:
        mask = selected["intervention"].eq(key)
        row[f"lp_cost_share_{key}"] = float(selected.loc[mask, "lp_effective_cost"].sum() / total_cost) if total_cost > EPS else np.nan
        row[f"lp_count_share_{key}"] = float(mask.mean()) if len(selected) else np.nan

    selected_cols = [
        "city",
        "event_id",
        "event_start",
        "parameter_scenario",
        "parameter_description",
        "unit",
        "t",
        "intervention",
        "lp_u",
        "lp_e",
        "lp_effective_cost",
        "target_value",
        "activated_bottleneck_score",
        "deficit_only_score",
        "exposure_only_score",
        "structure_only_score",
        "target_rank_pct",
        "activated_rank_pct",
        "deficit_rank_pct",
        "exposure_rank_pct",
        "structure_rank_pct",
        "target_top20",
        "activated_top20",
        "deficit_top20",
        "exposure_top20",
        "structure_top20",
        "simple_union_top20",
        "active_weighted_horizon",
        "law_exposure_term",
        "eta_per_cost",
        "time_remaining_frac",
        "delay_feasible",
    ]
    return selected[selected_cols].copy(), row


def build_scenario_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in event_metrics.columns
        if col not in {"city", "event_start", "parameter_scenario", "parameter_description", "lp_status"}
        and pd.api.types.is_numeric_dtype(event_metrics[col])
    ]
    agg = event_metrics.groupby(["parameter_scenario", "parameter_description"], as_index=False)[numeric_cols].mean()
    counts = event_metrics.groupby(["parameter_scenario", "parameter_description"], as_index=False).agg(
        n_event_scenarios=("event_id", "count"),
        n_cities=("city", "nunique"),
    )
    return counts.merge(agg, on=["parameter_scenario", "parameter_description"], how="left")


def build_city_summary(event_metrics: pd.DataFrame) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    keep = [
        "city",
        "lp_selected_action_count",
        "activated_top20_selected_cost_share",
        "deficit_top20_selected_cost_share",
        "exposure_top20_selected_cost_share",
        "structure_top20_selected_cost_share",
        "simple_union_top20_selected_cost_share",
        "hidden_from_simple_union_top20_selected_cost_share",
        "below_structure_top20_selected_cost_share",
        "lp_cost_share_R",
        "lp_cost_share_C",
        "lp_cost_share_S",
    ]
    available = [col for col in keep if col in event_metrics]
    return (
        event_metrics[available]
        .groupby("city", as_index=False)
        .mean()
        .sort_values("hidden_from_simple_union_top20_selected_cost_share", ascending=False)
    )


def build_diagnostics(event_metrics: pd.DataFrame, scenario_summary: pd.DataFrame, selected_actions: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "n_event_scenarios": int(len(event_metrics)),
        "n_cities": int(event_metrics["city"].nunique()) if "city" in event_metrics else 0,
        "n_parameter_scenarios": int(event_metrics["parameter_scenario"].nunique()) if "parameter_scenario" in event_metrics else 0,
        "n_lp_selected_actions": int(len(selected_actions)),
    }
    if event_metrics.empty:
        return diagnostics
    metric_cols = [
        "activated_top20_selected_cost_share",
        "deficit_top20_selected_cost_share",
        "exposure_top20_selected_cost_share",
        "structure_top20_selected_cost_share",
        "simple_union_top20_selected_cost_share",
        "hidden_from_simple_union_top20_selected_cost_share",
        "target_top20_hidden_from_simple_top20_selected_cost_share",
        "below_structure_top20_selected_cost_share",
        "activated_minus_best_simple_top20_selected_cost_share",
        "lp_cost_share_R",
        "lp_cost_share_C",
        "lp_cost_share_S",
    ]
    for col in metric_cols:
        if col in event_metrics:
            diagnostics[f"mean_{col}"] = safe_float(event_metrics[col].mean())
            diagnostics[f"min_{col}"] = safe_float(event_metrics[col].min())
            diagnostics[f"max_{col}"] = safe_float(event_metrics[col].max())
    if not scenario_summary.empty and "hidden_from_simple_union_top20_selected_cost_share" in scenario_summary:
        worst = scenario_summary.sort_values("hidden_from_simple_union_top20_selected_cost_share", ascending=False).iloc[0]
        diagnostics["worst_hidden_selected_scenario"] = str(worst["parameter_scenario"])
        diagnostics["worst_hidden_selected_scenario_value"] = safe_float(worst["hidden_from_simple_union_top20_selected_cost_share"])
    return diagnostics


def make_figures(event_metrics: pd.DataFrame, scenario_summary: pd.DataFrame, figure_dir: Path) -> None:
    if event_metrics.empty or scenario_summary.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = scenario_summary["parameter_scenario"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(len(order))
    width = 0.18
    bars = [
        ("activated_top20_selected_cost_share", "activated", "#111827"),
        ("deficit_top20_selected_cost_share", "deficit", "#2563eb"),
        ("exposure_top20_selected_cost_share", "exposure", "#0f766e"),
        ("structure_top20_selected_cost_share", "structure", "#ef4444"),
    ]
    for idx, (col, label, color) in enumerate(bars):
        ax.bar(x + (idx - 1.5) * width, scenario_summary[col], width=width, label=label, color=color)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, order, rotation=30, ha="right")
    ax.set_ylabel("LP-selected cost share in score top-20%")
    ax.set_title("Exact LP selected actions are not recovered by one-factor ranks")
    ax.legend(frameon=False, ncols=4)
    fig.tight_layout()
    fig.savefig(figure_dir / "lp_selected_cost_top20_capture.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar(order, scenario_summary["hidden_from_simple_union_top20_selected_cost_share"], color="#7c3aed", alpha=0.82)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of LP-selected cost")
    ax.set_title("LP-selected cost hidden from all simple top-20% rankings")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "lp_selected_hidden_from_simple.png", dpi=180)
    plt.close(fig)

    channel_cols = ["lp_cost_share_R", "lp_cost_share_C", "lp_cost_share_S"]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bottom = np.zeros(len(order))
    colors = {"lp_cost_share_R": "#2563eb", "lp_cost_share_C": "#0f766e", "lp_cost_share_S": "#f59e0b"}
    labels = {"lp_cost_share_R": "R", "lp_cost_share_C": "C", "lp_cost_share_S": "S"}
    for col in channel_cols:
        values = scenario_summary[col].to_numpy(dtype=float)
        ax.bar(order, values, bottom=bottom, color=colors[col], label=labels[col])
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("LP-selected cost share")
    ax.set_title("Selected intervention channels under parameter scenarios")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "lp_selected_channel_mix.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    scenario_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    event_metrics: pd.DataFrame,
    selected_actions: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter-Ensemble LP Selection Laws V30",
        "",
        "## 本版回答的问题",
        "",
        "V29 说明单因子规则在 first-order parameter ensemble target 中仍会产生 false priority。V30 进一步重解 V24 的代表性 full LP parameter scenarios，保存 LP 实际正投放的 unit-time-intervention actions，并检查这些 LP-selected actions 是否能被 deficit-only、exposure-only、structure-only 的 top ranks 捕获。",
        "",
        "这个诊断更接近完整优化器行为，但仍然是代表性 closure：覆盖 4 个 city-events、5 个 parameter scenarios，共 20 个 full LP，而不是全 105 events 的完整参数网格。",
        "",
        "## 核心结果",
        "",
        f"- event-scenarios: {diagnostics['n_event_scenarios']}; cities: {diagnostics['n_cities']}; parameter scenarios: {diagnostics['n_parameter_scenarios']}",
        f"- LP-selected positive actions: {diagnostics['n_lp_selected_actions']}",
        f"- activated top-20% captures {diagnostics['mean_activated_top20_selected_cost_share']:.1%} of LP-selected cost on average.",
        f"- simple union top-20% captures {diagnostics['mean_simple_union_top20_selected_cost_share']:.1%}; hidden from all simple top-20% = {diagnostics['mean_hidden_from_simple_union_top20_selected_cost_share']:.1%}.",
        f"- one-factor top-20% captures: deficit {diagnostics['mean_deficit_top20_selected_cost_share']:.1%}, exposure {diagnostics['mean_exposure_top20_selected_cost_share']:.1%}, structure {diagnostics['mean_structure_top20_selected_cost_share']:.1%}.",
        f"- below structure top-20% selected cost share = {diagnostics['mean_below_structure_top20_selected_cost_share']:.1%}.",
        f"- worst hidden-selected scenario = {diagnostics.get('worst_hidden_selected_scenario', '')} ({diagnostics.get('worst_hidden_selected_scenario_value', float('nan')):.1%}).",
        "",
        "解释：V30 的结论比 V29 更有边界感。完整 LP 的 selected support 并不是大量隐藏在所有简单规则之外；deficit 与 exposure 的并集已经能覆盖多数实际投放成本。这说明有限预算 LP 在选 support 时仍会使用可见的损失和暴露信号。真正稳定失败的是单独使用某一个因子，尤其是 structure-only：静态结构中心性本身几乎不能解释 LP 把资源投到哪里。activated score 不是完整 LP 的充分替代，因为 LP 还会考虑饱和、period budget 和 R/C/S 替代；但它比任何单因子规则更接近 LP selected support。",
        "",
        "## Scenario Summary",
        "",
        table_to_markdown(scenario_summary),
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary),
        "",
        "## Event Metrics",
        "",
        table_to_markdown(event_metrics),
    ]
    if not selected_actions.empty:
        examples = selected_actions.sort_values("lp_effective_cost", ascending=False).head(60)
        lines.extend(["", "## Largest Selected Actions", "", table_to_markdown(examples)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def weighted_share(frame: pd.DataFrame, flag: str | pd.Series, weight_col: str, total: float) -> float:
    if frame.empty or total <= EPS:
        return float("nan")
    mask = frame[flag].astype(bool) if isinstance(flag, str) else flag.astype(bool)
    return float(frame.loc[mask, weight_col].sum() / total)


def count_share(frame: pd.DataFrame, flag: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame[flag].astype(bool).mean())


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if total <= EPS:
        return float("nan")
    return float(np.sum(values * weights) / total)


def optimized_objective_to_fraction(baseline_objective: float, objective: float) -> float:
    return float(1.0 - objective / baseline_objective) if baseline_objective > EPS else np.nan


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "parameter_scenario"}.issubset(existing.columns):
        return set()
    return {
        (str(row.city), int(row.event_id), str(row.parameter_scenario))
        for row in existing[["city", "event_id", "parameter_scenario"]].itertuples(index=False)
    }


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


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.10g")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")


if __name__ == "__main__":
    main()

"""Validate analytic action-value labels with representative single-action LPs."""

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
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters, solve_recovery_lp


RNG_SEED = 20260529


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/law_learning_lp_validation")
    parser.add_argument("--actions-per-city", type=int, default=6)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(root)
    selected = select_validation_actions(inputs, args.actions_per_city, cities=args.cities)
    rows = run_single_action_validations(root, config, inputs, selected, args.time_limit_seconds)
    validation = pd.DataFrame(rows)
    summary = summarize_validation(validation)

    write_table(validation, table_dir / "single_action_lp_marginal_validation.csv")
    write_table(summary, table_dir / "single_action_lp_marginal_summary.csv")
    make_figures(validation, figure_dir)
    write_report(report_dir / "single_action_lp_marginal_validation_report_zh.md", validation, summary)
    print(f"Wrote single-action LP marginal validation outputs to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "results"
    return {
        "tokens": pd.read_csv(tables / "law_learning" / "tables" / "action_value_tokens.csv.gz"),
        "event_law": pd.read_csv(tables / "law_learning" / "tables" / "event_level_top_tail_law.csv"),
        "summary": pd.read_csv(tables / "event_optimization" / "tables" / "event_optimization_summary.csv"),
        "events": pd.read_csv(tables / "data_mining" / "tables" / "rainfall_event_impact_details.csv", parse_dates=["event_start", "event_end"]),
        "dynamics": pd.read_csv(tables / "event_calibration" / "tables" / "event_dynamic_calibration_summary.csv"),
        "abnormal": pd.read_csv(tables / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv", parse_dates=["hour"]),
    }


def select_validation_actions(
    inputs: dict[str, pd.DataFrame],
    actions_per_city: int,
    *,
    cities: list[str] | None,
) -> pd.DataFrame:
    tokens = inputs["tokens"].copy()
    tokens["unit"] = tokens["unit"].astype(str)
    tokens["event_id"] = pd.to_numeric(tokens["event_id"], errors="coerce").astype(int)
    tokens = tokens[(tokens["delay_feasible"] > 0) & (tokens["marginal_resource_value"] > 0)].copy()
    event_law = inputs["event_law"].copy()
    event_law["event_id"] = pd.to_numeric(event_law["event_id"], errors="coerce").astype(int)
    if cities:
        event_law = event_law[event_law["city"].isin(cities)].copy()
    representative_events = (
        event_law.sort_values(["city", "decision_criticality_score"], ascending=[True, False])
        .groupby("city", as_index=False)
        .head(1)[["city", "event_id"]]
    )
    chosen_frames: list[pd.DataFrame] = []
    for event in representative_events.itertuples(index=False):
        frame = tokens[(tokens["city"] == event.city) & (tokens["event_id"] == int(event.event_id))].copy()
        if frame.empty:
            continue
        chosen = choose_event_actions(frame, actions_per_city)
        chosen_frames.append(chosen)
    if not chosen_frames:
        return pd.DataFrame()
    out = pd.concat(chosen_frames, ignore_index=True)
    out = out.drop_duplicates(["city", "event_id", "unit", "t", "intervention"], keep="first")
    out["validation_action_id"] = np.arange(len(out))
    return out


def choose_event_actions(frame: pd.DataFrame, actions_per_city: int) -> pd.DataFrame:
    selectors: list[tuple[str, str]] = [
        ("analytic_value_top", "marginal_resource_value"),
        ("law_score_top", "activated_bottleneck_score"),
        ("exposure_top", "exposure_only_score"),
        ("deficit_top", "deficit_only_score"),
        ("optimizer_selected_top", "optimized_cost"),
        ("greedy_selected_top", "greedy_oracle_value_proxy"),
    ]
    rows: list[pd.DataFrame] = []
    per_selector = max(1, int(np.ceil(actions_per_city / len(selectors))))
    for source, score_col in selectors:
        candidates = frame[frame[score_col].fillna(0.0) > 0.0].copy()
        if candidates.empty:
            continue
        picked = candidates.nlargest(per_selector, score_col).copy()
        picked["validation_source"] = source
        picked["validation_score_column"] = score_col
        rows.append(picked)
    if not rows:
        return frame.nlargest(actions_per_city, "marginal_resource_value").copy()
    chosen = pd.concat(rows, ignore_index=True)
    chosen = chosen.sort_values("marginal_resource_value", ascending=False)
    chosen = chosen.drop_duplicates(["unit", "t", "intervention"], keep="first")
    if len(chosen) < actions_per_city:
        filler = frame.nlargest(actions_per_city * 2, "marginal_resource_value").copy()
        filler["validation_source"] = "analytic_value_filler"
        filler["validation_score_column"] = "marginal_resource_value"
        chosen = pd.concat([chosen, filler], ignore_index=True)
        chosen = chosen.drop_duplicates(["unit", "t", "intervention"], keep="first")
    return chosen.head(actions_per_city).copy()


def run_single_action_validations(
    root: Path,
    config: dict[str, Any],
    inputs: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    time_limit_seconds: float,
) -> list[dict[str, Any]]:
    if selected.empty:
        return []
    summary = inputs["summary"].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary_lookup = {
        (row.city, int(row.event_id)): row for row in summary.itertuples(index=False)
    }
    events = inputs["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamics = {row["city"]: row for _, row in inputs["dynamics"].iterrows()}
    abnormal = inputs["abnormal"].copy()

    rows: list[dict[str, Any]] = []
    total = len(selected)
    for idx, action in enumerate(selected.itertuples(index=False), start=1):
        city = str(action.city)
        event_id = int(action.event_id)
        print(
            f"[{idx}/{total}] Single-action LP {city} event {event_id} "
            f"{action.intervention} unit {action.unit} t={int(action.t)}",
            flush=True,
        )
        try:
            params = calibrate_observed_event_city(
                city,
                config,
                pd.Series(event_lookup[(city, event_id)]._asdict()),
                dynamics[city],
                abnormal_hourly=abnormal,
                root=root,
            )
            summary_row = summary_lookup[(city, event_id)]
            probe_params, probe_meta = build_single_action_probe_params(params, action)
            solution = solve_recovery_lp(
                probe_params,
                output_flag=False,
                method=int(config.get("solver", {}).get("method", -1)),
                time_limit_seconds=time_limit_seconds,
            )
            lp_used_cost = float(solution.interventions["effective_cost"].sum())
            lp_total_e = float(solution.interventions["e"].sum())
            baseline_objective = float(summary_row.baseline_objective)
            gain = baseline_objective - float(solution.objective)
            normalized_gain_per_cost = gain / max(baseline_objective * lp_used_cost, 1e-12)
            analytic = float(action.marginal_resource_value)
            derivative = small_signal_derivative_value(action, config)
            rows.append(
                {
                    **action_metadata(action),
                    **probe_meta,
                    "status": solution.status,
                    "runtime_seconds": solution.runtime_seconds,
                    "baseline_objective": baseline_objective,
                    "lp_objective": float(solution.objective),
                    "lp_gain": gain,
                    "lp_used_cost": lp_used_cost,
                    "lp_total_e": lp_total_e,
                    "lp_normalized_gain_per_cost": normalized_gain_per_cost,
                    "analytic_marginal_resource_value": analytic,
                    "small_signal_derivative_value": derivative,
                    "law_score": float(action.activated_bottleneck_score),
                    "lp_to_analytic_ratio": normalized_gain_per_cost / analytic if analytic > 1e-12 else np.nan,
                    "lp_to_derivative_ratio": normalized_gain_per_cost / derivative if derivative > 1e-12 else np.nan,
                    "abs_log_error": abs(np.log(max(normalized_gain_per_cost, 1e-12)) - np.log(max(analytic, 1e-12))),
                    "derivative_abs_log_error": abs(np.log(max(normalized_gain_per_cost, 1e-12)) - np.log(max(derivative, 1e-12))),
                    "base_lp_recoverable_fraction": float(summary_row.recoverable_fraction),
                    "base_lp_optimized_objective": float(summary_row.optimized_objective),
                }
            )
        except Exception as exc:  # pragma: no cover - batch diagnostics
            rows.append({**action_metadata(action), "status": "ERROR", "error": str(exc)})
    return rows


def small_signal_derivative_value(action: Any, config: dict[str, Any]) -> float:
    horizon = int(config["calibration"]["horizon_steps"])
    steps = max(horizon - int(action.t), 0)
    if steps <= 0 or float(action.delay_feasible) <= 0:
        return 0.0
    k = np.arange(steps, dtype=float)
    intervention = str(action.intervention)
    if intervention == "R":
        exposure = float(action.destination_importance)
        decay_sum = float(np.sum(float(action.a_retention) ** k))
    elif intervention == "C":
        exposure = float(action.destination_importance)
        decay_sum = float(np.sum((1.0 - float(config["interventions"]["delta_C"])) ** k))
    elif intervention == "S":
        exposure = float(action.origin_exposure)
        decay_sum = float(np.sum((1.0 - float(config["interventions"]["delta_S"])) ** k))
    else:
        return np.nan
    return exposure * decay_sum * float(action.eta_per_cost) / max(float(action.baseline_objective), 1e-12)


def build_single_action_probe_params(params: RecoveryLPParameters, action: Any) -> tuple[RecoveryLPParameters, dict[str, Any]]:
    unit = str(action.unit)
    intervention = str(action.intervention)
    t = int(action.t)
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention {intervention}.")
    unit_to_idx = {str(unit_id): idx for idx, unit_id in enumerate(params.units)}
    if unit not in unit_to_idx:
        raise ValueError(f"Unit {unit} is not in calibrated params for {params.city}.")
    i = unit_to_idx[unit]
    u_cap = {key: np.zeros_like(value) for key, value in params.u_cap.items()}
    u_segment_cap = None
    segment_effectiveness = None
    segment_id = 0
    segment_multiplier = 1.0
    if params.u_segment_cap is not None and params.segment_effectiveness is not None:
        u_segment_cap = {key: np.zeros_like(value) for key, value in params.u_segment_cap.items()}
        segment_cap = float(params.u_segment_cap[intervention][i, t, segment_id])
        u_cap[intervention][i, t] = segment_cap
        u_segment_cap[intervention][i, t, segment_id] = segment_cap
        segment_effectiveness = {key: value.copy() for key, value in params.segment_effectiveness.items()}
        segment_multiplier = float(segment_effectiveness[intervention][segment_id])
    else:
        segment_cap = float(params.u_cap[intervention][i, t]) * 0.25
        u_cap[intervention][i, t] = segment_cap
    probe_cost = float(params.cost[intervention][i, t]) * segment_cap
    if probe_cost <= 1e-12:
        raise ValueError("Probe action has zero available cost cap.")
    period_budget = np.zeros(params.horizon, dtype=float)
    period_budget[t] = probe_cost
    probe_params = RecoveryLPParameters(
        city=params.city,
        units=list(params.units),
        p=params.p.copy(),
        q=params.q.copy(),
        b0=params.b0.copy(),
        a=params.a.copy(),
        h=params.h.copy(),
        eta={key: value.copy() for key, value in params.eta.items()},
        cost={key: value.copy() for key, value in params.cost.items()},
        u_cap=u_cap,
        u_segment_cap=u_segment_cap,
        segment_effectiveness=segment_effectiveness,
        period_budget=period_budget,
        total_budget=probe_cost,
        delays=dict(params.delays),
        delta_c=params.delta_c,
        delta_s=params.delta_s,
        delta_t=params.delta_t,
        metadata={**params.metadata, "probe_type": "single_action_first_segment"},
    )
    return probe_params, {
        "probe_u_cap": segment_cap,
        "probe_cost_cap": probe_cost,
        "probe_segment_id": segment_id,
        "probe_segment_multiplier": segment_multiplier,
    }


def action_metadata(action: Any) -> dict[str, Any]:
    return {
        "validation_action_id": int(action.validation_action_id),
        "validation_source": str(action.validation_source),
        "city": str(action.city),
        "event_id": int(action.event_id),
        "event_start": str(action.event_start),
        "unit": str(action.unit),
        "t": int(action.t),
        "intervention": str(action.intervention),
        "event_peak_positive_abnormal_deficit": float(action.event_peak_positive_abnormal_deficit),
        "recoverable_fraction": float(action.recoverable_fraction),
    }


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    valid = validation[validation["status"].isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"])].copy()
    valid = valid[np.isfinite(valid["lp_normalized_gain_per_cost"])]
    rows: list[dict[str, Any]] = []
    label_specs = [
        ("finite_deficit_area_label", "analytic_marginal_resource_value", "lp_to_analytic_ratio", "abs_log_error"),
        ("small_signal_derivative_label", "small_signal_derivative_value", "lp_to_derivative_ratio", "derivative_abs_log_error"),
    ]
    for label_name, label_col, ratio_col, error_col in label_specs:
        label_valid = valid[np.isfinite(valid[label_col]) & (valid[label_col] > 0)].copy()
        groups: list[tuple[str, pd.DataFrame]] = [("all", label_valid)]
        groups.extend((str(name), group) for name, group in label_valid.groupby("validation_source"))
        for name, group in groups:
            if group.empty:
                continue
            rows.append(
                {
                    "label": label_name,
                    "group": name,
                    "n_actions": int(len(group)),
                    "n_cities": int(group["city"].nunique()),
                    "pearson": safe_corr(group[label_col], group["lp_normalized_gain_per_cost"]),
                    "spearman": group[label_col].corr(group["lp_normalized_gain_per_cost"], method="spearman"),
                    "median_lp_to_label_ratio": float(group[ratio_col].median()),
                    "mean_lp_to_label_ratio": float(group[ratio_col].mean()),
                    "median_abs_log_error": float(group[error_col].median()),
                    "mean_abs_log_error": float(group[error_col].mean()),
                    "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                    "mean_probe_cost_cap": float(group["probe_cost_cap"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_figures(validation: pd.DataFrame, figure_dir: Path) -> None:
    valid = validation[validation["status"].isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"])].copy()
    valid = valid[(valid["analytic_marginal_resource_value"] > 0) & (valid["lp_normalized_gain_per_cost"] > 0)]
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    specs = [
        ("analytic_marginal_resource_value", "Finite-deficit-area label"),
        ("small_signal_derivative_value", "Small-signal derivative label"),
    ]
    for ax, (label_col, title) in zip(axes, specs, strict=True):
        for source, group in valid.groupby("validation_source"):
            ax.scatter(
                group[label_col],
                group["lp_normalized_gain_per_cost"],
                s=52,
                alpha=0.8,
                label=source,
            )
        lo = min(valid[label_col].min(), valid["lp_normalized_gain_per_cost"].min())
        hi = max(valid[label_col].max(), valid["lp_normalized_gain_per_cost"].max())
        ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.2, linestyle="--", label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Single-action LP normalized gain per cost")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "single_action_lp_vs_analytic_value.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ordered = valid.assign(ratio_clipped=valid["lp_to_analytic_ratio"].clip(0, 3))
    ordered.boxplot(column="ratio_clipped", by="validation_source", ax=ax, grid=False, rot=25)
    ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", alpha=0.45)
    ax.set_title("LP / analytic marginal-value ratio")
    fig.suptitle("")
    ax.set_ylabel("Ratio clipped at 3")
    fig.tight_layout()
    fig.savefig(figure_dir / "single_action_lp_ratio_by_source.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, validation: pd.DataFrame, summary: pd.DataFrame) -> None:
    valid = validation[validation["status"].isin(["OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"])].copy()
    finite_overall = (
        summary[(summary["label"] == "finite_deficit_area_label") & (summary["group"] == "all")].iloc[0].to_dict()
        if not summary.empty and ((summary["label"] == "finite_deficit_area_label") & (summary["group"] == "all")).any()
        else {}
    )
    derivative_overall = (
        summary[(summary["label"] == "small_signal_derivative_label") & (summary["group"] == "all")].iloc[0].to_dict()
        if not summary.empty and ((summary["label"] == "small_signal_derivative_label") & (summary["group"] == "all")).any()
        else {}
    )
    lines = [
        "# Single-Action LP Marginal Validation V4",
        "",
        "## 本版本做了什么",
        "",
        "V4 用代表性的 single-action LP 检查 V1-V3 中的解析 action-value label。做法是：每个城市选择一个 decision-critical event；在该 event 中选择若干高 analytic value、高 law score、optimizer-selected、greedy-selected 或 simple-baseline action；然后构造一个只允许该 action 第一段 PWL deployment 的 LP。",
        "",
        "这个验证回答的问题是：`marginal_resource_value` 作为 action-level 学习标签，是否和真实 LP objective gain 的方向一致？如果一致，那么后续 surrogate/law 学到的不是纯粹的公式自洽，而是和 LP 目标函数有可验证的连接。",
        "",
        "## 样本规模",
        "",
        f"- validation actions attempted: {len(validation):,}",
        f"- solved/feasible actions: {len(valid):,}",
        f"- cities: {valid['city'].nunique() if not valid.empty else 0}",
        "",
        "## 总体结果",
        "",
        "Finite-deficit-area label 是 V1-V3 使用的标签思想；small-signal derivative label 是这次根据 single-action LP 暴露出的导数版本。",
        "",
        f"- finite-label Spearman: {finite_overall.get('spearman', np.nan):.4f}",
        f"- finite-label median LP/label ratio: {finite_overall.get('median_lp_to_label_ratio', np.nan):.4f}",
        f"- finite-label median abs log error: {finite_overall.get('median_abs_log_error', np.nan):.4f}",
        f"- derivative-label Spearman: {derivative_overall.get('spearman', np.nan):.4f}",
        f"- derivative-label median LP/label ratio: {derivative_overall.get('median_lp_to_label_ratio', np.nan):.4f}",
        f"- derivative-label median abs log error: {derivative_overall.get('median_abs_log_error', np.nan):.4f}",
        "",
        "## Summary by Source",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Validation Rows",
        "",
        dataframe_to_markdown(validation.head(30)),
        "",
        "## 如何解读",
        "",
        "如果 ratio 接近 1，说明解析 marginal value 与单 action LP 的单位成本收益接近；如果 Spearman 高但 ratio 偏离 1，说明排序可靠但尺度需要校准；如果二者都低，则说明解析 label 需要重构。",
        "",
        "这次特别区分了两类 label：V1-V3 的 finite-deficit-area label 更像“有限资源能吃到多少剩余损失面积”的保守估计；single-action first segment LP 更像小信号导数，因此 derivative label 可能在尺度上更接近 LP。这个发现会指导下一版把 action label 拆成两个头：ranking/finite-cap value 与 small-signal marginal value。",
        "",
        "这版仍然只是 first-segment single-action check。它没有验证多 action 之间的 residual interaction，也没有验证 perturbed optimum stability。下一步应在少量代表 event 上做 greedy residual LP 或 near-optimal perturbation，检验多个 action 同时存在时 law 是否仍稳定。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
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

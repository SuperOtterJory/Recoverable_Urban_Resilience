"""Validate non-obvious action laws under intervention-parameter ensembles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_nonobvious_action_laws import (
    HEURISTICS,
    SIMPLE_HEURISTICS,
    build_event_heuristic_metrics,
    build_failure_reason_summary,
    build_heuristic_summary,
    build_hidden_gem_tables,
    prepare_tokens as prepare_nonobvious_tokens,
)
from analyze_parameter_ensemble_stability import (
    SCENARIOS,
    apply_parameter_scenario,
    load_tokens as load_base_tokens,
)
from recoverable_resilience.paths import find_repo_root


SIMPLE_LABELS = {
    "deficit_only": "deficit",
    "exposure_only": "exposure",
    "structure_only": "structure",
}


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "parameter_nonobvious_laws"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    base_tokens = load_base_tokens(root)
    event_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    reason_frames: list[pd.DataFrame] = []
    hidden_frames: list[pd.DataFrame] = []

    for scenario in SCENARIOS:
        scenario_name = str(scenario["parameter_scenario"])
        print(f"Analyzing non-obvious laws under {scenario_name}", flush=True)
        scenario_tokens = apply_parameter_scenario(base_tokens, scenario)
        tokens = prepare_nonobvious_tokens(scenario_tokens)
        event_metrics = build_event_heuristic_metrics(tokens)
        summary = build_heuristic_summary(event_metrics)
        reasons = build_failure_reason_summary(tokens)
        hidden_summary, _, _ = build_hidden_gem_tables(tokens)

        for frame in [event_metrics, summary, reasons, hidden_summary]:
            frame.insert(0, "parameter_scenario", scenario_name)
            frame.insert(1, "parameter_description", str(scenario["description"]))
        event_frames.append(event_metrics)
        summary_frames.append(summary)
        reason_frames.append(reasons)
        hidden_frames.append(hidden_summary)

    event_metrics_all = pd.concat(event_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    reasons_all = pd.concat(reason_frames, ignore_index=True)
    hidden_all = pd.concat(hidden_frames, ignore_index=True)
    scenario_metrics = build_scenario_metrics(summary_all, hidden_all)
    diagnostics = build_diagnostics(summary_all, hidden_all, reasons_all, scenario_metrics)

    write_table(event_metrics_all, table_dir / "parameter_nonobvious_event_metrics.csv")
    write_table(summary_all, table_dir / "parameter_nonobvious_summary.csv")
    write_table(reasons_all, table_dir / "parameter_nonobvious_reason_summary.csv")
    write_table(hidden_all, table_dir / "parameter_nonobvious_hidden_summary.csv")
    write_table(scenario_metrics, table_dir / "parameter_nonobvious_scenario_metrics.csv")
    (table_dir / "parameter_nonobvious_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(summary_all, hidden_all, reasons_all, scenario_metrics, figure_dir)
    write_report(
        report_dir / "parameter_nonobvious_laws_report_zh.md",
        diagnostics,
        summary_all,
        hidden_all,
        reasons_all,
        scenario_metrics,
    )
    print(f"Wrote parameter non-obvious law analysis to {output_dir}")


def build_scenario_metrics(summary: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, group in summary.groupby("parameter_scenario", sort=False):
        row: dict[str, Any] = {
            "parameter_scenario": scenario,
            "parameter_description": group["parameter_description"].iloc[0],
        }
        for heuristic in SIMPLE_HEURISTICS + ["activated_law"]:
            h = group[group["heuristic"].eq(heuristic)]
            if h.empty:
                continue
            prefix = SIMPLE_LABELS.get(heuristic, heuristic)
            first = h.iloc[0]
            row[f"{prefix}_top5_relative_to_oracle"] = float(first["mean_top5_relative_to_oracle"])
            row[f"{prefix}_false_positive_share"] = float(first["mean_false_positive_share"])
            row[f"{prefix}_zero_value_share"] = float(first["mean_zero_value_share"])
            row[f"{prefix}_target_top5_precision"] = float(first["mean_target_top5_precision"])
        hrow = hidden[hidden["parameter_scenario"].eq(scenario)]
        if not hrow.empty:
            first_hidden = hrow.iloc[0]
            row["hidden_from_all_simple_top5_share"] = float(first_hidden["hidden_from_all_simple_top5_share"])
            row["hidden_from_all_simple_top20_share"] = float(first_hidden["hidden_from_all_simple_top20_share"])
            row["target_top5_low_structure_top20_share"] = float(first_hidden["target_top5_low_structure_top20_share"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_diagnostics(
    summary: pd.DataFrame,
    hidden: pd.DataFrame,
    reasons: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "n_parameter_scenarios": int(summary["parameter_scenario"].nunique()) if not summary.empty else 0,
        "n_events_per_scenario": int(summary["n_events"].max()) if not summary.empty and "n_events" in summary else 0,
    }
    base = scenario_metrics[scenario_metrics["parameter_scenario"].eq("base")]
    for heuristic, short in SIMPLE_LABELS.items():
        rows = summary[summary["heuristic"].eq(heuristic)]
        if rows.empty:
            continue
        diagnostics[f"{short}_mean_false_positive_share"] = safe_float(rows["mean_false_positive_share"].mean())
        diagnostics[f"{short}_min_false_positive_share"] = safe_float(rows["mean_false_positive_share"].min())
        diagnostics[f"{short}_max_false_positive_share"] = safe_float(rows["mean_false_positive_share"].max())
        diagnostics[f"{short}_mean_top5_relative_to_oracle"] = safe_float(rows["mean_top5_relative_to_oracle"].mean())
        diagnostics[f"{short}_min_top5_relative_to_oracle"] = safe_float(rows["mean_top5_relative_to_oracle"].min())
        if not base.empty and f"{short}_false_positive_share" in base:
            diagnostics[f"{short}_base_false_positive_share"] = safe_float(base.iloc[0][f"{short}_false_positive_share"])
            diagnostics[f"{short}_base_top5_relative_to_oracle"] = safe_float(base.iloc[0][f"{short}_top5_relative_to_oracle"])
    activated = summary[summary["heuristic"].eq("activated_law")]
    if not activated.empty:
        diagnostics["activated_law_mean_top5_relative_to_oracle"] = safe_float(activated["mean_top5_relative_to_oracle"].mean())
        diagnostics["activated_law_min_top5_relative_to_oracle"] = safe_float(activated["mean_top5_relative_to_oracle"].min())
    diagnostics["hidden_from_simple_top5_mean_share"] = safe_float(hidden["hidden_from_all_simple_top5_share"].mean()) if not hidden.empty else np.nan
    diagnostics["hidden_from_simple_top5_min_share"] = safe_float(hidden["hidden_from_all_simple_top5_share"].min()) if not hidden.empty else np.nan
    diagnostics["hidden_from_simple_top5_max_share"] = safe_float(hidden["hidden_from_all_simple_top5_share"].max()) if not hidden.empty else np.nan
    diagnostics["target_top5_low_structure_top20_mean_share"] = safe_float(hidden["target_top5_low_structure_top20_share"].mean()) if not hidden.empty else np.nan
    diagnostics["target_top5_low_structure_top20_min_share"] = safe_float(hidden["target_top5_low_structure_top20_share"].min()) if not hidden.empty else np.nan
    if not scenario_metrics.empty:
        for col in ["deficit_false_positive_share", "exposure_false_positive_share", "structure_false_positive_share"]:
            if col in scenario_metrics:
                worst = scenario_metrics.sort_values(col, ascending=False).head(1).iloc[0]
                diagnostics[f"worst_{col}_scenario"] = str(worst["parameter_scenario"])
                diagnostics[f"worst_{col}_value"] = safe_float(worst[col])
    if not reasons.empty:
        simple_reasons = reasons[reasons["heuristic"].isin(SIMPLE_HEURISTICS)].copy()
        for col in [c for c in simple_reasons.columns if c.endswith("_share") and c not in {"mean_false_positive_share"}]:
            diagnostics[f"reason_{col}_mean"] = safe_float(simple_reasons[col].mean())
    return diagnostics


def make_figures(
    summary: pd.DataFrame,
    hidden: pd.DataFrame,
    reasons: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_false_positive_figure(summary, figure_dir / "parameter_nonobvious_false_positive.png")
    make_capture_figure(summary, figure_dir / "parameter_nonobvious_top5_capture.png")
    make_hidden_figure(hidden, figure_dir / "parameter_nonobvious_hidden_share.png")
    make_reason_figure(reasons, figure_dir / "parameter_nonobvious_failure_reasons.png")
    make_tradeoff_figure(scenario_metrics, figure_dir / "parameter_nonobvious_capture_vs_false_positive.png")


def make_false_positive_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[summary["heuristic"].isin(SIMPLE_HEURISTICS)].copy()
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    scenarios = list(dict.fromkeys(plot["parameter_scenario"].tolist()))
    x = np.arange(len(scenarios))
    width = 0.25
    colors = {"deficit_only": "#2563eb", "exposure_only": "#0f766e", "structure_only": "#ef4444"}
    for idx, heuristic in enumerate(SIMPLE_HEURISTICS):
        sub = plot[plot["heuristic"].eq(heuristic)].set_index("parameter_scenario").reindex(scenarios)
        ax.bar(x + (idx - 1) * width, sub["mean_false_positive_share"], width=width, label=heuristic.replace("_", " "), color=colors[heuristic])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("False-positive share of heuristic top-5%")
    ax.set_title("Simple action heuristics remain unreliable under parameter ensembles")
    ax.set_xticks(x, scenarios, rotation=35, ha="right")
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_capture_figure(summary: pd.DataFrame, path: Path) -> None:
    plot = summary[summary["heuristic"].isin(SIMPLE_HEURISTICS + ["activated_law"])].copy()
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    for heuristic, color in [
        ("deficit_only", "#2563eb"),
        ("exposure_only", "#0f766e"),
        ("structure_only", "#ef4444"),
        ("activated_law", "#111827"),
    ]:
        sub = plot[plot["heuristic"].eq(heuristic)]
        ax.plot(
            sub["parameter_scenario"],
            sub["mean_top5_relative_to_oracle"],
            marker="o",
            linewidth=2.0,
            label=heuristic.replace("_", " "),
            color=color,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Top-5% value capture relative to oracle")
    ax.set_title("Activation remains the stable top-tail explanation")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_hidden_figure(hidden: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    x = np.arange(len(hidden))
    ax.bar(x, hidden["hidden_from_all_simple_top5_share"], color="#7c3aed", alpha=0.82, label="hidden from simple top-5%")
    ax.plot(x, hidden["target_top5_low_structure_top20_share"], color="#ef4444", marker="o", label="below structure top-20%")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of true top-5% actions")
    ax.set_title("High-value actions remain hidden from one-factor rankings")
    ax.set_xticks(x, hidden["parameter_scenario"], rotation=35, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_reason_figure(reasons: pd.DataFrame, path: Path) -> None:
    plot = reasons[reasons["heuristic"].isin(SIMPLE_HEURISTICS)].copy()
    reason_cols = [
        "delay_blocked_share",
        "below_median_future_horizon_share",
        "below_median_exposure_share",
        "below_median_efficiency_share",
        "short_remaining_window_share",
    ]
    aggregate = plot.groupby("heuristic")[reason_cols].mean().reindex(SIMPLE_HEURISTICS)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(aggregate.to_numpy(dtype=float), aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(aggregate.index)), [idx.replace("_", " ") for idx in aggregate.index])
    ax.set_xticks(np.arange(len(reason_cols)), [col.replace("_share", "").replace("_", " ") for col in reason_cols], rotation=30, ha="right")
    ax.set_title("Why simple top-ranked actions fail across parameter scenarios")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean share among false positives")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_tradeoff_figure(scenario_metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for short, label, color in [
        ("deficit", "deficit", "#2563eb"),
        ("exposure", "exposure", "#0f766e"),
        ("structure", "structure", "#ef4444"),
    ]:
        ax.scatter(
            scenario_metrics[f"{short}_false_positive_share"],
            scenario_metrics[f"{short}_top5_relative_to_oracle"],
            s=70,
            alpha=0.82,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("False-positive share")
    ax.set_ylabel("Top-5% value capture")
    ax.set_title("One-factor heuristics trade capture for false priorities")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    hidden: pd.DataFrame,
    reasons: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter-Ensemble Non-Obvious Action Laws V29",
        "",
        "## 这一版做了什么",
        "",
        "V29 把 V15 的非显然 action-law 诊断放到 11 个 eta/cost/delay 参数扰动场景下重做。目标不是重新求完整 LP，而是在 V23 的 first-order parameter ensemble target 上检查：最高 deficit、最高 exposure、最高 structure 这些一因子规则是否仍会产生 false priority，以及 activated law 是否仍是最稳定的 top-tail 解释。",
        "",
        "## 核心结论",
        "",
        f"- parameter scenarios: {diagnostics['n_parameter_scenarios']}; events per scenario: {diagnostics['n_events_per_scenario']}",
        f"- deficit-only false-positive share: base {diagnostics['deficit_base_false_positive_share']:.1%}, ensemble mean {diagnostics['deficit_mean_false_positive_share']:.1%}, range {diagnostics['deficit_min_false_positive_share']:.1%}-{diagnostics['deficit_max_false_positive_share']:.1%}.",
        f"- exposure-only false-positive share: base {diagnostics['exposure_base_false_positive_share']:.1%}, ensemble mean {diagnostics['exposure_mean_false_positive_share']:.1%}, range {diagnostics['exposure_min_false_positive_share']:.1%}-{diagnostics['exposure_max_false_positive_share']:.1%}.",
        f"- structure-only false-positive share: base {diagnostics['structure_base_false_positive_share']:.1%}, ensemble mean {diagnostics['structure_mean_false_positive_share']:.1%}, range {diagnostics['structure_min_false_positive_share']:.1%}-{diagnostics['structure_max_false_positive_share']:.1%}.",
        f"- hidden true top-5% actions from all simple top-5% rankings: mean {diagnostics['hidden_from_simple_top5_mean_share']:.1%}, range {diagnostics['hidden_from_simple_top5_min_share']:.1%}-{diagnostics['hidden_from_simple_top5_max_share']:.1%}.",
        f"- activated law relative top-5% capture: mean {diagnostics['activated_law_mean_top5_relative_to_oracle']:.4f}, min {diagnostics['activated_law_min_top5_relative_to_oracle']:.4f}.",
        "",
        "解释：如果一因子规则的 false-positive share 在参数扰动后仍然较高，而 activated law 仍保持 oracle top-tail capture，那么“非显然性”就不是 base 管理参数下的偶然现象。恢复价值需要 future-loss horizon、OD exposure、delay feasibility 和 channel efficiency 同时激活。",
        "",
        "## Scenario Metrics",
        "",
        table_to_markdown(scenario_metrics),
        "",
        "## Heuristic Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Hidden High-Value Actions",
        "",
        table_to_markdown(hidden),
        "",
        "## Failure Reasons",
        "",
        table_to_markdown(reasons),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

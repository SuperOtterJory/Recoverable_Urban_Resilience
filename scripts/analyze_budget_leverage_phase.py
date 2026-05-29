"""Analyze how decision leverage changes across budget levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


BUDGET_ORDER = ["low_budget", "base", "high_budget"]
BUDGET_LABELS = {0.5: "low", 1.0: "base", 2.0: "high"}
BASELINES = ["random_positive", "exposure_only", "deficit_only", "structure_only"]
LAW_POLICY = "activated_bottleneck_law"
EPS = 1e-12


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "budget_leverage_phase"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root)
    proxy_event = build_proxy_event_metrics(data["budget_policy"])
    residual_event = build_residual_event_metrics(data["residual_stress"])
    event_metrics = merge_event_metrics(proxy_event, residual_event, data["event_law"])
    summary = build_budget_summary(event_metrics)
    city_summary = build_city_summary(event_metrics)
    correlations = build_correlations(event_metrics)
    phase_tests = build_phase_tests(summary)

    write_table(event_metrics, table_dir / "budget_leverage_event_metrics.csv")
    write_table(summary, table_dir / "budget_leverage_summary.csv")
    write_table(city_summary, table_dir / "budget_leverage_city_summary.csv")
    write_table(correlations, table_dir / "budget_leverage_correlations.csv")
    write_table(phase_tests, table_dir / "budget_phase_tests.csv")

    make_figures(summary, city_summary, event_metrics, figure_dir)
    write_report(report_dir / "budget_leverage_phase_report_zh.md", summary, city_summary, correlations, phase_tests)
    print(f"Wrote budget leverage phase analysis to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    results = root / "results"
    return {
        "budget_policy": pd.read_csv(results / "law_learning" / "tables" / "budget_policy_simulation.csv"),
        "residual_stress": pd.read_csv(results / "residual_greedy_stress" / "tables" / "residual_stress_event_metrics.csv"),
        "event_law": pd.read_csv(results / "law_learning" / "tables" / "event_level_top_tail_law.csv"),
    }


def budget_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["policy_scenario"].isin(BUDGET_ORDER) & frame["delay_add_hours"].eq(0)].copy()


def build_proxy_event_metrics(policy: pd.DataFrame) -> pd.DataFrame:
    work = budget_only(policy)
    value = work.pivot_table(
        index=["city", "event_id", "event_start", "budget_scale", "policy_scenario"],
        columns="policy_score",
        values="value_proxy",
        aggfunc="first",
    ).reset_index()
    cost = work.pivot_table(
        index=["city", "event_id", "budget_scale"],
        columns="policy_score",
        values="allocated_cost",
        aggfunc="first",
    ).reset_index()
    cost = cost.rename(columns={LAW_POLICY: "law_allocated_cost"})
    out = value.merge(cost[["city", "event_id", "budget_scale", "law_allocated_cost"]], on=["city", "event_id", "budget_scale"], how="left")
    out["budget_label"] = out["budget_scale"].map(BUDGET_LABELS).fillna(out["budget_scale"].astype(str))
    out["law_value_proxy"] = out[LAW_POLICY]
    for baseline in BASELINES:
        out[f"proxy_leverage_vs_{baseline}"] = out[LAW_POLICY] - out[baseline]
        out[f"proxy_ratio_vs_{baseline}"] = out[LAW_POLICY] / out[baseline].replace(0.0, np.nan)
        out[f"proxy_leverage_per_cost_vs_{baseline}"] = out[f"proxy_leverage_vs_{baseline}"] / out["law_allocated_cost"].replace(0.0, np.nan)
    return out


def build_residual_event_metrics(stress: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "city",
        "event_id",
        "budget_scale",
        "residual_minus_static_fraction_of_base_lp_gain",
        "residual_minus_static_recoverable_fraction",
        "residual_relative_to_static_gain",
        "static_fraction_of_base_lp_gain",
        "residual_fraction_of_base_lp_gain",
        "static_replay_recoverable_fraction",
        "residual_replay_recoverable_fraction",
        "residual_allocated_cost",
    ]
    out = budget_only(stress)[cols].copy()
    out["residual_minus_static_per_cost"] = (
        out["residual_minus_static_recoverable_fraction"] / out["residual_allocated_cost"].replace(0.0, np.nan)
    )
    return out


def merge_event_metrics(proxy: pd.DataFrame, residual: pd.DataFrame, event_law: pd.DataFrame) -> pd.DataFrame:
    out = proxy.merge(residual, on=["city", "event_id", "budget_scale"], how="left")
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
    event_cols = [col for col in event_cols if col in event_law.columns]
    out = out.merge(event_law[event_cols], on=["city", "event_id"], how="left")
    return out.sort_values(["budget_scale", "city", "event_id"]).reset_index(drop=True)


def build_budget_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget_scale, group in events.groupby("budget_scale", sort=True):
        row: dict[str, Any] = {
            "budget_scale": float(budget_scale),
            "budget_label": BUDGET_LABELS.get(float(budget_scale), str(budget_scale)),
            "n_events": int(len(group)),
            "mean_law_value_proxy": safe_mean(group["law_value_proxy"]),
            "mean_law_value_per_cost": safe_mean(group["law_value_proxy"] / group["law_allocated_cost"].replace(0.0, np.nan)),
            "mean_residual_minus_static_fraction_of_base_lp_gain": safe_mean(group["residual_minus_static_fraction_of_base_lp_gain"]),
            "mean_residual_minus_static_recoverable_fraction": safe_mean(group["residual_minus_static_recoverable_fraction"]),
            "mean_residual_minus_static_per_cost": safe_mean(group["residual_minus_static_per_cost"]),
            "mean_residual_relative_to_static_gain": safe_mean(group["residual_relative_to_static_gain"]),
        }
        for baseline in BASELINES:
            row[f"mean_proxy_leverage_vs_{baseline}"] = safe_mean(group[f"proxy_leverage_vs_{baseline}"])
            row[f"median_proxy_leverage_vs_{baseline}"] = safe_median(group[f"proxy_leverage_vs_{baseline}"])
            row[f"mean_proxy_ratio_vs_{baseline}"] = safe_mean(group[f"proxy_ratio_vs_{baseline}"])
            row[f"mean_proxy_leverage_per_cost_vs_{baseline}"] = safe_mean(group[f"proxy_leverage_per_cost_vs_{baseline}"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("budget_scale")


def build_city_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (city, budget_scale), group in events.groupby(["city", "budget_scale"], sort=True):
        rows.append(
            {
                "city": city,
                "budget_scale": float(budget_scale),
                "budget_label": BUDGET_LABELS.get(float(budget_scale), str(budget_scale)),
                "n_events": int(len(group)),
                "mean_proxy_leverage_vs_random": safe_mean(group["proxy_leverage_vs_random_positive"]),
                "mean_proxy_ratio_vs_random": safe_mean(group["proxy_ratio_vs_random_positive"]),
                "mean_proxy_leverage_vs_exposure": safe_mean(group["proxy_leverage_vs_exposure_only"]),
                "mean_residual_minus_static_fraction_of_base_lp_gain": safe_mean(group["residual_minus_static_fraction_of_base_lp_gain"]),
                "mean_residual_minus_static_recoverable_fraction": safe_mean(group["residual_minus_static_recoverable_fraction"]),
            }
        )
    out = pd.DataFrame(rows)
    phase_rows = []
    for city, group in out.groupby("city", sort=True):
        phase_rows.append(
            {
                "city": city,
                "absolute_proxy_random_peak_budget": peak_budget(group, "mean_proxy_leverage_vs_random"),
                "relative_proxy_random_peak_budget": peak_budget(group, "mean_proxy_ratio_vs_random"),
                "residual_static_peak_budget": peak_budget(group, "mean_residual_minus_static_fraction_of_base_lp_gain"),
            }
        )
    return out.merge(pd.DataFrame(phase_rows), on="city", how="left")


def build_correlations(events: pd.DataFrame) -> pd.DataFrame:
    base = events[events["budget_scale"].eq(1.0)].copy()
    targets = [
        "proxy_leverage_vs_random_positive",
        "proxy_leverage_vs_exposure_only",
        "proxy_ratio_vs_random_positive",
        "residual_minus_static_fraction_of_base_lp_gain",
        "residual_minus_static_recoverable_fraction",
    ]
    features = [
        "top_5pct_value_share",
        "marginal_value_gini",
        "decision_criticality_score",
        "baseline_objective",
        "recoverable_fraction",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    rows: list[dict[str, Any]] = []
    for target in targets:
        for feature in features:
            if target in base and feature in base:
                pair = base[[target, feature]].dropna()
                rows.append(
                    {
                        "budget_scale": 1.0,
                        "target": target,
                        "feature": feature,
                        "spearman": pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target", "spearman"], ascending=[True, False])


def build_phase_tests(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_proxy_leverage_vs_random_positive",
        "mean_proxy_ratio_vs_random_positive",
        "mean_proxy_leverage_vs_exposure_only",
        "mean_proxy_ratio_vs_exposure_only",
        "mean_proxy_leverage_per_cost_vs_random_positive",
        "mean_residual_minus_static_fraction_of_base_lp_gain",
        "mean_residual_minus_static_recoverable_fraction",
        "mean_residual_minus_static_per_cost",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = {
            float(row.budget_scale): float(getattr(row, metric))
            for row in summary[["budget_scale", metric]].itertuples(index=False)
            if np.isfinite(float(getattr(row, metric)))
        }
        low, base, high = values.get(0.5, np.nan), values.get(1.0, np.nan), values.get(2.0, np.nan)
        rows.append(
            {
                "metric": metric,
                "low": low,
                "base": base,
                "high": high,
                "peak_budget": budget_label(max(values, key=values.get)) if values else "",
                "interior_peak_supported": bool(base > low and base > high) if np.all(np.isfinite([low, base, high])) else False,
                "monotone_increasing": bool(low <= base <= high) if np.all(np.isfinite([low, base, high])) else False,
                "monotone_decreasing": bool(low >= base >= high) if np.all(np.isfinite([low, base, high])) else False,
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, city_summary: pd.DataFrame, events: pd.DataFrame, figure_dir: Path) -> None:
    make_budget_curve(summary, figure_dir / "budget_leverage_curve.png")
    make_city_phase(city_summary, figure_dir / "budget_leverage_city_phase.png")
    make_top_tail_relation(events, figure_dir / "budget_leverage_top_tail_relation.png")


def make_budget_curve(summary: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.6, 5.4))
    x = summary["budget_scale"].to_numpy(dtype=float)
    ax1.plot(x, summary["mean_proxy_leverage_vs_random_positive"], marker="o", color="#2563eb", label="law - random proxy")
    ax1.plot(x, summary["mean_proxy_leverage_vs_exposure_only"], marker="o", color="#0f766e", label="law - exposure proxy")
    ax1.plot(x, summary["mean_residual_minus_static_recoverable_fraction"], marker="o", color="#7c3aed", label="residual - static replay")
    ax1.set_xlabel("Budget scale")
    ax1.set_ylabel("Absolute decision leverage")
    ax2 = ax1.twinx()
    ax2.plot(x, summary["mean_proxy_ratio_vs_random_positive"], marker="s", linestyle="--", color="#ef4444", label="law / random ratio")
    ax2.set_ylabel("Relative leverage ratio")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.set_title("Budget phase: absolute leverage grows while relative leverage decays")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_city_phase(city_summary: pd.DataFrame, path: Path) -> None:
    subset = city_summary[city_summary["budget_scale"].eq(1.0)].copy()
    subset = subset.sort_values("mean_residual_minus_static_fraction_of_base_lp_gain")
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(subset))
    ax.barh(y, subset["mean_residual_minus_static_fraction_of_base_lp_gain"], color="#2563eb", alpha=0.84)
    ax.set_yticks(y, subset["city"])
    ax.set_xlabel("Residual minus static gain / base LP gain")
    ax.set_title("Budget leverage at base budget varies sharply by city")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_top_tail_relation(events: pd.DataFrame, path: Path) -> None:
    base = events[events["budget_scale"].eq(1.0)].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    scatter = ax.scatter(
        base["top_5pct_value_share"],
        base["proxy_leverage_vs_random_positive"],
        c=base["baseline_objective"],
        cmap="viridis",
        s=54,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Top-5% marginal value share")
    ax.set_ylabel("Law proxy value minus random proxy value")
    ax.set_title("Decision leverage is tied to top-tail concentration, not just event size")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("No-intervention loss objective")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    phase_tests: pd.DataFrame,
) -> None:
    absolute = phase_tests[phase_tests["metric"].eq("mean_proxy_leverage_vs_random_positive")].iloc[0]
    relative = phase_tests[phase_tests["metric"].eq("mean_proxy_ratio_vs_random_positive")].iloc[0]
    residual = phase_tests[phase_tests["metric"].eq("mean_residual_minus_static_recoverable_fraction")].iloc[0]
    base_city = city_summary[city_summary["budget_scale"].eq(1.0)].sort_values("mean_residual_minus_static_fraction_of_base_lp_gain", ascending=False)
    lines = [
        "# Budget-Leverage Phase Analysis V13",
        "",
        "## 这一版回答什么问题",
        "",
        "High-level idea 里有一个预期：decision leverage 可能在中等预算最高。V13 用已有 low/base/high budget policy replay 与 action-value proxy 直接检验这个命题，并区分两类量：绝对决策收益（law 比 naive 多拿到多少 value）和单位预算/相对收益（每单位资源或相对 naive 的优势）。",
        "",
        "## 主要结论",
        "",
        f"- law-vs-random absolute proxy leverage peak: {absolute.peak_budget}; interior peak supported = {absolute.interior_peak_supported}",
        f"- law/random relative ratio peak: {relative.peak_budget}; monotone decreasing = {relative.monotone_decreasing}",
        f"- residual-vs-static replay leverage peak: {residual.peak_budget}; interior peak supported = {residual.interior_peak_supported}",
        "",
        "当前三点预算扫描不支持“中等预算绝对 decision leverage 最高”。更准确的结论是：预算越高，law 相对 naive 的绝对额外收益继续增加；但相对优势和单位预算杠杆递减。这说明智能分配的收益存在 diminishing leverage，而不是简单的 interior nonmonotonic peak。",
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
        "## Correlations at Base Budget",
        "",
        table_to_markdown(correlations),
        "",
        "## 科学含义",
        "",
        "这个结果让论文叙事更精确：managed recoverability 的预算规律不是单纯“预算越多越好”，也不是当前样本中已经证明的“中等预算最高”。从现有数据看，绝对可恢复收益随预算增加，但每单位预算产生的额外决策杠杆下降；因此预算 law 应表述为 scale-dependent diminishing leverage。城市差异也很强，Dallas 的 residual-vs-static gap 接近零，而 Chicago、Philadelphia、San Antonio 的 gap 明显更大，说明预算杠杆仍由城市结构和 action-value top tail 决定。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def peak_budget(group: pd.DataFrame, metric: str) -> str:
    if metric not in group or group.empty:
        return ""
    row = group.sort_values(metric, ascending=False).iloc[0]
    return budget_label(float(row["budget_scale"]))


def budget_label(value: float) -> str:
    return BUDGET_LABELS.get(float(value), str(value))


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if len(values.dropna()) else np.nan


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.median()) if len(values.dropna()) else np.nan


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


if __name__ == "__main__":
    main()

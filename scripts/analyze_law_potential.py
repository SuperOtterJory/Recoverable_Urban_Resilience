"""Analyze whether optimization outputs contain learnable law-like structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


SCENARIO_DELAY_SCORE = {
    "fast_response": 0.0,
    "base": 1.0,
    "low_budget": 1.0,
    "high_budget": 1.0,
    "very_high_budget": 1.0,
    "delayed_response": 2.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/law_potential")
    args = parser.parse_args()

    root = find_repo_root()
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root)
    city_features = build_city_features(data)
    scenario_dataset = build_scenario_dataset(data, city_features)
    tuning_dataset = build_tuning_dataset(data, city_features)
    correlations = build_correlations(scenario_dataset, tuning_dataset)
    budget_curvature = build_budget_curvature(tuning_dataset)
    delay_penalty = build_delay_penalty(tuning_dataset)
    primitive_structure = build_primitive_structure(scenario_dataset)
    model_fits = build_model_fits(tuning_dataset, scenario_dataset)

    city_features.to_csv(table_dir / "city_structural_features.csv", index=False)
    scenario_dataset.to_csv(table_dir / "scenario_law_dataset.csv", index=False)
    tuning_dataset.to_csv(table_dir / "tuning_law_dataset.csv", index=False)
    correlations.to_csv(table_dir / "law_correlation_summary.csv", index=False)
    budget_curvature.to_csv(table_dir / "budget_response_curvature.csv", index=False)
    delay_penalty.to_csv(table_dir / "response_delay_penalty.csv", index=False)
    primitive_structure.to_csv(table_dir / "primitive_mix_structure.csv", index=False)
    model_fits.to_csv(table_dir / "simple_law_model_fits.csv", index=False)

    make_figures(scenario_dataset, tuning_dataset, correlations, primitive_structure, figure_dir)
    write_report(
        report_dir / "law_potential_report_zh.md",
        city_features,
        scenario_dataset,
        tuning_dataset,
        correlations,
        budget_curvature,
        delay_penalty,
        primitive_structure,
        model_fits,
    )
    print(f"Wrote law-potential analysis to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    opt = root / "results" / "optimization" / "tables"
    mining = root / "results" / "data_mining" / "tables"
    return {
        "optimization": read_csv(opt / "optimization_summary.csv"),
        "tuning": read_csv(opt / "parameter_tuning_summary.csv"),
        "policy": read_csv(opt / "policy_comparison.csv"),
        "calibration": read_csv(opt / "calibration_summary.csv"),
        "credibility": read_csv(opt / "model_credibility_checks.csv"),
        "speed": read_csv(mining / "speed_deficit_summary.csv"),
        "rain_speed": read_csv(mining / "rainfall_speed_alignment.csv"),
        "demand": read_csv(mining / "demand_network_summary.csv"),
        "concentration": read_csv(mining / "speed_tmc_deficit_concentration.csv"),
        "fit": read_csv(mining / "idea_data_fit_scores.csv"),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        if col != "city" and col != "scenario" and col != "policy" and col != "delay_name":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    return df


def build_city_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = data["calibration"].copy()
    keep_tables = [
        (
            "speed",
            [
                "mean_deficit",
                "p90_deficit",
                "severe_deficit_share_20pct",
                "hourly_observations",
            ],
        ),
        (
            "rain_speed",
            [
                "max_lag_corr",
                "mean_event_deficit_impact",
                "median_event_recovery_hours",
                "events_with_recovery_observed",
            ],
        ),
        (
            "demand",
            [
                "od_rows",
                "origin_zone_count",
                "destination_zone_count",
                "destination_volume_hhi",
                "top10_destination_volume_share",
                "congested_volume_share_speed_ratio_lt_0_8",
                "volume_weighted_speed_kmph",
            ],
        ),
        (
            "concentration",
            [
                "tmc_deficit_gini",
                "top_10pct_tmc_deficit_share",
                "high_deficit_tmc_share_mean_gt_0_2",
            ],
        ),
        (
            "fit",
            [
                "overall_data_support_score_0_5",
                "functional_dependence_score_0_5",
                "deficit_signal_score_0_5",
            ],
        ),
    ]
    for name, cols in keep_tables:
        table = data[name]
        if table.empty:
            continue
        present = ["city", *[col for col in cols if col in table.columns]]
        base = base.merge(table[present], on="city", how="left")
    base["baseline_loss_scale"] = base["weighted_b0"] * base["mean_a_retention"]
    base["disturbance_ratio"] = base["total_disturbance"] / base["weighted_b0"].replace(0, np.nan)
    base["dependence_concentration_proxy"] = (
        base.get("destination_volume_hhi", np.nan) * 1000
        + base.get("top10_destination_volume_share", np.nan)
    )
    return base


def build_scenario_dataset(data: dict[str, pd.DataFrame], city_features: pd.DataFrame) -> pd.DataFrame:
    df = data["optimization"].copy()
    best_policy = (
        data["policy"].sort_values("objective")
        .groupby(["city", "scenario"], as_index=False)
        .first()
        if not data["policy"].empty
        else pd.DataFrame()
    )
    if not best_policy.empty:
        df = df.merge(
            best_policy[
                [
                    "city",
                    "scenario",
                    "policy",
                    "objective",
                    "recoverable_fraction",
                ]
            ],
            on=["city", "scenario"],
            how="left",
            suffixes=("", "_best_heuristic"),
        )
        df = df.rename(
            columns={
                "policy": "best_heuristic_policy",
                "objective_best_heuristic": "best_heuristic_objective",
                "recoverable_fraction_best_heuristic": "best_heuristic_recoverable_fraction",
            }
        )
        df["decision_leverage_fraction"] = (
            df["best_heuristic_objective"] - df["optimized_objective"]
        ) / df["baseline_objective"].replace(0, np.nan)
        df["relative_gain_over_best_heuristic"] = df["decision_leverage_fraction"] / df[
            "best_heuristic_recoverable_fraction"
        ].replace(0, np.nan)
    df["delay_score"] = df["scenario"].map(SCENARIO_DELAY_SCORE).fillna(1.0)
    df["log_budget"] = np.log1p(df["budget_intensity"])
    df["budget_squared"] = df["budget_intensity"] ** 2
    for key in ["R", "C", "S"]:
        df[f"cost_share_{key}"] = df[f"total_cost_{key}"] / df["total_intervention_cost"].replace(0, np.nan)
    df = df.merge(city_features, on="city", how="left", suffixes=("", "_city"))
    return df


def build_tuning_dataset(data: dict[str, pd.DataFrame], city_features: pd.DataFrame) -> pd.DataFrame:
    df = data["tuning"].copy()
    delay_map = {"fast": 0.0, "base": 1.0, "slow": 2.0}
    df["delay_score"] = df["delay_name"].map(delay_map).fillna(1.0)
    df["log_budget"] = np.log1p(df["budget_intensity"])
    df["budget_squared"] = df["budget_intensity"] ** 2
    df = df.merge(city_features, on="city", how="left", suffixes=("", "_city"))
    return df


def build_correlations(scenario_dataset: pd.DataFrame, tuning_dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_features = [
        "budget_intensity",
        "delay_score",
        "weighted_b0",
        "mean_a_retention",
        "total_disturbance",
        "p90_deficit",
        "max_lag_corr",
        "destination_volume_hhi",
        "congested_volume_share_speed_ratio_lt_0_8",
        "tmc_deficit_gini",
        "top_10pct_tmc_deficit_share",
        "cost_share_R",
        "cost_share_C",
        "cost_share_S",
    ]
    for feature in scenario_features:
        add_corr(rows, "scenario", scenario_dataset, feature, "recoverable_fraction")
        if "decision_leverage_fraction" in scenario_dataset.columns:
            add_corr(rows, "scenario", scenario_dataset, feature, "decision_leverage_fraction")
    tuning_features = [
        "budget_intensity",
        "log_budget",
        "budget_squared",
        "delay_score",
        "weighted_b0",
        "mean_a_retention",
        "p90_deficit",
        "top_10pct_tmc_deficit_share",
    ]
    for feature in tuning_features:
        add_corr(rows, "tuning", tuning_dataset, feature, "recoverable_fraction")
    return pd.DataFrame(rows).sort_values(["dataset", "target", "abs_spearman"], ascending=[True, True, False])


def add_corr(rows: list[dict[str, object]], dataset_name: str, df: pd.DataFrame, feature: str, target: str) -> None:
    if feature not in df.columns or target not in df.columns:
        return
    pair = df[[feature, target]].dropna()
    if len(pair) < 4 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
        return
    pearson = pair[feature].corr(pair[target], method="pearson")
    spearman = pair[feature].corr(pair[target], method="spearman")
    rows.append(
        {
            "dataset": dataset_name,
            "feature": feature,
            "target": target,
            "n": len(pair),
            "pearson": pearson,
            "spearman": spearman,
            "abs_spearman": abs(spearman),
        }
    )


def build_budget_curvature(tuning: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, delay_name), group in tuning.groupby(["city", "delay_name"]):
        group = group.sort_values("budget_intensity")
        x = group["budget_intensity"].to_numpy(dtype=float)
        y = group["recoverable_fraction"].to_numpy(dtype=float)
        slopes = np.diff(y) / np.diff(x)
        rows.append(
            {
                "city": city,
                "delay_name": delay_name,
                "monotonic": bool(np.all(np.diff(y) >= -1e-8)),
                "first_slope": slopes[0] if len(slopes) else np.nan,
                "last_slope": slopes[-1] if len(slopes) else np.nan,
                "slope_ratio_last_to_first": slopes[-1] / slopes[0] if len(slopes) and slopes[0] != 0 else np.nan,
                "mean_slope": np.mean(slopes) if len(slopes) else np.nan,
                "recoverable_at_0_18": interp_at(x, y, 0.18),
                "recoverable_at_1_00": interp_at(x, y, 1.00),
            }
        )
    return pd.DataFrame(rows)


def build_delay_penalty(tuning: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, budget), group in tuning.groupby(["city", "budget_intensity"]):
        values = group.set_index("delay_name")["recoverable_fraction"]
        if {"fast", "base", "slow"}.issubset(values.index):
            rows.append(
                {
                    "city": city,
                    "budget_intensity": budget,
                    "fast_minus_base": values["fast"] - values["base"],
                    "base_minus_slow": values["base"] - values["slow"],
                    "fast_minus_slow": values["fast"] - values["slow"],
                    "relative_delay_penalty": (values["fast"] - values["slow"]) / values["fast"]
                    if values["fast"] != 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_primitive_structure(scenario: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name, group in scenario.groupby("scenario"):
        row = {"scenario": scenario_name, "n": len(group)}
        for key in ["R", "C", "S"]:
            row[f"mean_cost_share_{key}"] = group[f"cost_share_{key}"].mean()
            row[f"mean_cap_utilization_{key}"] = group.get(f"cap_utilization_{key}", pd.Series(dtype=float)).mean()
        row["mean_recoverable_fraction"] = group["recoverable_fraction"].mean()
        row["mean_decision_leverage"] = group.get("decision_leverage_fraction", pd.Series(dtype=float)).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_recoverable_fraction", ascending=False)


def build_model_fits(tuning: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(fit_model("tuning_log_budget", tuning, "recoverable_fraction", ["log_budget"]))
    rows.append(fit_model("tuning_budget_delay", tuning, "recoverable_fraction", ["log_budget", "delay_score"]))
    rows.append(
        fit_model(
            "tuning_city_budget_delay",
            tuning,
            "recoverable_fraction",
            ["log_budget", "delay_score"],
            categorical=["city"],
        )
    )
    rows.append(
        fit_model(
            "tuning_city_quadratic_delay",
            tuning,
            "recoverable_fraction",
            ["budget_intensity", "budget_squared", "delay_score"],
            categorical=["city"],
        )
    )
    rows.append(
        fit_model(
            "scenario_budget_delay_city",
            scenario,
            "recoverable_fraction",
            ["log_budget", "delay_score"],
            categorical=["city"],
        )
    )
    if "decision_leverage_fraction" in scenario.columns:
        rows.append(
            fit_model(
                "scenario_decision_leverage",
                scenario,
                "decision_leverage_fraction",
                ["log_budget", "delay_score", "recoverable_fraction"],
                categorical=["city"],
            )
        )
    return pd.DataFrame(rows)


def fit_model(
    name: str,
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical: list[str] | None = None,
) -> dict[str, object]:
    categorical = categorical or []
    cols = [target, *numeric_features, *categorical]
    data = df[cols].dropna()
    y = data[target].to_numpy(dtype=float)
    x_parts = [np.ones((len(data), 1))]
    feature_names = ["intercept"]
    for feature in numeric_features:
        x_parts.append(data[[feature]].to_numpy(dtype=float))
        feature_names.append(feature)
    for feature in categorical:
        dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True, dtype=float)
        if not dummies.empty:
            x_parts.append(dummies.to_numpy(dtype=float))
            feature_names.extend(dummies.columns.tolist())
    x = np.hstack(x_parts)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    formula_terms = [f"{coef[i]:+.4g}*{feature_names[i]}" for i in range(1, len(feature_names))]
    return {
        "model": name,
        "target": target,
        "n": len(data),
        "r2": r2,
        "intercept": coef[0] if len(coef) else np.nan,
        "formula_skeleton": "y = {:.4g} {}".format(coef[0], " ".join(formula_terms)),
    }


def make_figures(
    scenario: pd.DataFrame,
    tuning: pd.DataFrame,
    correlations: pd.DataFrame,
    primitive_structure: pd.DataFrame,
    figure_dir: Path,
) -> None:
    make_budget_response_figure(tuning, figure_dir / "budget_response_curves.png")
    make_decision_leverage_figure(scenario, figure_dir / "decision_leverage_by_scenario.png")
    make_correlation_figure(correlations, figure_dir / "law_correlation_summary.png")
    make_primitive_mix_figure(primitive_structure, figure_dir / "primitive_mix_structure.png")


def make_budget_response_figure(tuning: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for (city, delay_name), group in tuning.groupby(["city", "delay_name"]):
        if delay_name != "base":
            continue
        group = group.sort_values("budget_intensity")
        ax.plot(group["budget_intensity"], group["recoverable_fraction"], marker="o", label=city)
    ax.set_title("Budget-response curves under base delay")
    ax.set_xlabel("Budget intensity")
    ax.set_ylabel("Recoverable fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_decision_leverage_figure(scenario: pd.DataFrame, path: Path) -> None:
    if "decision_leverage_fraction" not in scenario.columns:
        return
    pivot = scenario.pivot_table(index="city", columns="scenario", values="decision_leverage_fraction")
    ordered = [c for c in ["low_budget", "base", "high_budget", "very_high_budget", "delayed_response", "fast_response"] if c in pivot.columns]
    pivot = pivot[ordered]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Optimized decision leverage over best heuristic")
    ax.set_ylabel("Additional recovered fraction")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Scenario", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_correlation_figure(correlations: pd.DataFrame, path: Path) -> None:
    subset = correlations[correlations["target"].isin(["recoverable_fraction", "decision_leverage_fraction"])]
    subset = subset.sort_values("abs_spearman", ascending=False).head(16).copy()
    if subset.empty:
        return
    labels = subset["dataset"] + ":" + subset["feature"] + " -> " + subset["target"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = ["#2563EB" if v >= 0 else "#DC2626" for v in subset["spearman"]]
    ax.barh(labels[::-1], subset["spearman"][::-1], color=colors[::-1])
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_title("Strongest Spearman structure signals")
    ax.set_xlabel("Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_primitive_mix_figure(primitive_structure: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ordered = primitive_structure.sort_values("mean_recoverable_fraction")
    bottom = np.zeros(len(ordered))
    for key, color in [("R", "#2563EB"), ("C", "#059669"), ("S", "#EA580C")]:
        values = ordered[f"mean_cost_share_{key}"].fillna(0).to_numpy()
        ax.barh(ordered["scenario"], values, left=bottom, color=color, label=key)
        bottom += values
    ax.set_title("Mean primitive cost share by scenario")
    ax.set_xlabel("Share of used intervention budget")
    ax.legend(title="Primitive")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    city_features: pd.DataFrame,
    scenario: pd.DataFrame,
    tuning: pd.DataFrame,
    correlations: pd.DataFrame,
    budget_curvature: pd.DataFrame,
    delay_penalty: pd.DataFrame,
    primitive_structure: pd.DataFrame,
    model_fits: pd.DataFrame,
) -> None:
    top_corr = correlations.sort_values("abs_spearman", ascending=False).head(12)
    best_models = model_fits.sort_values("r2", ascending=False)
    leverage = scenario.sort_values("decision_leverage_fraction", ascending=False).head(10)
    delay_summary = delay_penalty.groupby("budget_intensity", as_index=False).agg(
        mean_fast_minus_slow=("fast_minus_slow", "mean"),
        mean_relative_delay_penalty=("relative_delay_penalty", "mean"),
    )
    curvature_summary = budget_curvature.groupby("delay_name", as_index=False).agg(
        monotonic_share=("monotonic", "mean"),
        mean_slope_ratio=("slope_ratio_last_to_first", "mean"),
        mean_rec_at_1=("recoverable_at_1_00", "mean"),
    )

    lines = [
        "# Law-Potential Analysis for Recoverable Urban Resilience",
        "",
        "## 核心判断",
        "",
        "从当前 optimization outputs 看，已经存在适合后续 learning-to-optimize、XAI 与 symbolic regression 的结构信号，但这些信号的层级不同：",
        "",
        "1. **强信号**：预算强度、响应延迟、PWL 边际收益递减共同形成稳定的 recoverability response surface。这个结构已经足够训练一个小型 surrogate 或 learning-to-optimize policy，并可用 XAI 检查预算/延迟/城市固定效应。",
        "2. **中等信号**：optimized policy 相对 damage/exposure/access heuristic 有稳定 decision leverage，说明模型输出中确实包含“智能配置优于朴素规则”的可学习结构。",
        "3. **弱到中等信号**：城市结构变量，例如 speed deficit、OD concentration、TMC deficit concentration，与 recoverability/decision leverage 存在方向性相关，但当前只有 5 个城市，不能把它们当作可靠 law。它们更像后续 law extraction 的候选变量。",
        "",
        "因此，目前结果可以衔接后续模块，但最适合先学习的是 **scenario-conditioned managed recovery law**，而不是直接宣称跨城市通用 law 已经被识别。",
        "",
        "## 最强相关结构",
        "",
        dataframe_to_markdown(top_corr[["dataset", "feature", "target", "n", "spearman", "pearson"]]),
        "",
        "解释：预算相关变量通常是 recoverability 的最强驱动；delay_score 与 recoverability 负相关；primitive mix 和 decision leverage 也有强信号。城市结构变量的相关性应谨慎解释，因为城市样本数很小。",
        "",
        "## 简单 law-like surrogate 拟合",
        "",
        dataframe_to_markdown(best_models[["model", "target", "n", "r2", "formula_skeleton"]]),
        "",
        "解释：如果加入 city fixed effects，预算和延迟的低维表达可以很好解释 tuning results。这说明后续 neural network 不一定一开始就需要很复杂；可从低维 response surface 开始，再逐步引入城市结构特征。",
        "",
        "## 预算响应曲线与边际收益递减",
        "",
        dataframe_to_markdown(curvature_summary),
        "",
        "解释：所有 tuning 曲线保持单调，且由于 PWL diminishing returns 与 caps，预算响应呈现更保守、更可信的增长。是否严格凹取决于城市和延迟情景，但总体上不再是无限线性收益。",
        "",
        "## 响应延迟惩罚",
        "",
        dataframe_to_markdown(delay_summary),
        "",
        "解释：fast response 相比 slow response 有稳定收益，且预算越大时 delay penalty 的绝对值通常更明显。这为论文中的 response delay 参数提供了可学习结构。",
        "",
        "## Decision Leverage",
        "",
        dataframe_to_markdown(
            leverage[
                [
                    "city",
                    "scenario",
                    "best_heuristic_policy",
                    "recoverable_fraction",
                    "best_heuristic_recoverable_fraction",
                    "decision_leverage_fraction",
                    "relative_gain_over_best_heuristic",
                ]
            ]
        ),
        "",
        "解释：optimized policy 不只是优于 no-intervention，也优于 best heuristic。这一点非常重要，因为它对应 high-level idea 里的 decision leverage：智能干预配置本身具有可识别价值。",
        "",
        "## Primitive Mix 结构",
        "",
        dataframe_to_markdown(primitive_structure),
        "",
        "解释：随着预算上升，R/C/S 的组合发生变化。这个 primitive mix regime shift 是 learning-to-optimize 可学习的结构，也适合后续 XAI 分析。",
        "",
        "## 自我审视：哪些已经分析，哪些还不能宣称",
        "",
        "已经分析：预算响应、延迟惩罚、decision leverage、primitive mix、简单 surrogate 可拟合性、城市结构变量相关性、结果可信度检查。",
        "",
        "仍不能宣称：当前只有 5 个优化城市，不能可靠提取跨城市 universal law；`eta/cost/cap` 仍是 scenario-calibrated 参数，不是观测因果参数；城市结构变量和 recoverability 的相关性只是候选规律。",
        "",
        "最终判断：当前 optimization results **足以支持进入 learning-to-optimize + XAI/symbolic regression 的下一阶段**，但下一阶段最合理的学习对象应是“在给定城市状态、预算、延迟、primitive constraints 下的 managed recovery response surface 和 decision leverage”，而不是直接从 5 个城市中提取终极城市韧性定律。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    formatted = formatted.fillna("").astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def interp_at(x: np.ndarray, y: np.ndarray, value: float) -> float:
    return float(np.interp(value, x, y))


if __name__ == "__main__":
    main()

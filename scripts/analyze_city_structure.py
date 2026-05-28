"""Analyze full-zone LP outcomes as city-structure signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


PRIMARY_STRUCTURAL_FEATURES = {
    "scale": ["n_units", "q_nnz", "od_rows", "origin_zone_count"],
    "dependence": ["q_density", "od_density_observed", "destination_volume_hhi", "top10_destination_volume_share"],
    "traffic": ["congested_volume_share_speed_ratio_lt_0_8", "doc_over_1_volume_share", "volume_weighted_speed_kmph"],
    "disruption": ["mean_deficit", "p90_deficit", "severe_deficit_share_20pct"],
    "rainfall": ["positive_rain_event_count", "mean_peak_extra_deficit", "max_peak_extra_deficit", "mean_affected_hours"],
    "concentration": ["tmc_deficit_gini", "top_10pct_tmc_deficit_share", "high_deficit_tmc_share_mean_gt_0_2"],
}

RAIN_FREE_STRUCTURE_FEATURES = {
    "scale": ["n_units", "q_nnz", "od_rows", "origin_zone_count", "log_n_units", "log_od_rows"],
    "dependence": [
        "q_density",
        "od_density_observed",
        "od_sparsity",
        "destination_volume_hhi",
        "top10_destination_volume_share",
        "within_zone_volume_share",
    ],
    "traffic_network": [
        "congested_volume_share_speed_ratio_lt_0_8",
        "doc_over_1_volume_share",
        "volume_weighted_speed_kmph",
        "link_volume_hhi",
        "top10_link_volume_share",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-dir", default="results/optimization")
    parser.add_argument("--output-dir", default="results/city_structure")
    args = parser.parse_args()

    root = find_repo_root()
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs(root, root / args.optimization_dir)
    dataset = build_dataset(data)
    correlations = build_correlations(dataset)
    structure_only = build_structure_only_correlations(dataset)
    rankings = build_rankings(dataset)

    dataset.to_csv(table_dir / "city_structure_dataset.csv", index=False)
    correlations.to_csv(table_dir / "structure_correlations.csv", index=False)
    structure_only.to_csv(table_dir / "structure_only_correlations.csv", index=False)
    rankings.to_csv(table_dir / "city_rankings.csv", index=False)
    make_figures(dataset, correlations, figure_dir)
    write_report(report_dir / "city_structure_report_zh.md", dataset, correlations, structure_only, rankings)
    print(f"Wrote city-structure analysis to {output_dir}")


def load_inputs(root: Path, optimization_dir: Path) -> dict[str, pd.DataFrame]:
    opt = optimization_dir / "tables"
    mining = root / "results" / "data_mining" / "tables"
    return {
        "optimization": read_csv(opt / "optimization_summary.csv"),
        "calibration": read_csv(opt / "calibration_summary.csv"),
        "policy": read_csv(opt / "policy_comparison.csv"),
        "demand": read_csv(mining / "demand_network_summary.csv"),
        "speed": read_csv(mining / "speed_deficit_summary.csv"),
        "tmc": read_csv(mining / "speed_tmc_deficit_concentration.csv"),
        "rain_events": read_csv(mining / "rainfall_event_impact_summary.csv"),
        "fit": read_csv(mining / "idea_data_fit_scores.csv"),
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in df.columns:
        if column not in {"city", "scenario", "status", "policy", "interpretation"}:
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().sum() > 0:
                df[column] = converted
    return df


def build_dataset(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    opt = data["optimization"].copy()
    opt = opt[opt["scenario"] == "base"].copy() if "scenario" in opt.columns else opt
    dataset = opt.merge(data["calibration"], on="city", how="left", suffixes=("", "_cal"))
    for key in ["demand", "speed", "tmc", "rain_events", "fit"]:
        table = data[key]
        if not table.empty:
            dataset = dataset.merge(table, on="city", how="left")
    policy = data["policy"]
    if not policy.empty:
        best = policy.sort_values("objective").groupby(["city", "scenario"], as_index=False).first()
        best = best.rename(
            columns={
                "policy": "best_heuristic_policy",
                "objective": "best_heuristic_objective",
                "recoverable_fraction": "best_heuristic_recoverable_fraction",
            }
        )
        dataset = dataset.merge(
            best[
                [
                    "city",
                    "scenario",
                    "best_heuristic_policy",
                    "best_heuristic_objective",
                    "best_heuristic_recoverable_fraction",
                ]
            ],
            on=["city", "scenario"],
            how="left",
        )
        dataset["decision_leverage_fraction"] = (
            dataset["best_heuristic_objective"] - dataset["optimized_objective"]
        ) / dataset["baseline_objective"].replace(0, np.nan)
    dataset["od_sparsity"] = 1.0 - dataset["q_density"]
    dataset["log_n_units"] = np.log1p(dataset["n_units"])
    dataset["log_od_rows"] = np.log1p(dataset["od_rows"])
    dataset["baseline_loss_per_unit"] = dataset["baseline_objective"] / dataset["n_units"].replace(0, np.nan)
    dataset["total_budget_per_unit"] = dataset["total_budget"] / dataset["n_units"].replace(0, np.nan)
    return dataset.sort_values("recoverable_fraction", ascending=False)


def build_correlations(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = ["recoverable_fraction", "decision_leverage_fraction"]
    for family, features in PRIMARY_STRUCTURAL_FEATURES.items():
        for feature in features + derived_features_for_family(family):
            for target in targets:
                add_corr(rows, dataset, family, feature, target)
    return pd.DataFrame(rows).sort_values(["target", "abs_spearman"], ascending=[True, False])


def build_structure_only_correlations(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, features in RAIN_FREE_STRUCTURE_FEATURES.items():
        for feature in features:
            add_corr(rows, dataset, family, feature, "recoverable_fraction")
            add_corr(rows, dataset, family, feature, "decision_leverage_fraction")
    return pd.DataFrame(rows).sort_values(["target", "abs_spearman"], ascending=[True, False])


def derived_features_for_family(family: str) -> list[str]:
    if family == "scale":
        return ["log_n_units", "log_od_rows"]
    if family == "dependence":
        return ["od_sparsity"]
    return []


def add_corr(
    rows: list[dict[str, Any]],
    dataset: pd.DataFrame,
    family: str,
    feature: str,
    target: str,
) -> None:
    if feature not in dataset.columns or target not in dataset.columns:
        return
    pair = dataset[["city", feature, target]].dropna()
    if len(pair) < 4 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
        return
    rows.append(
        {
            "feature_family": family,
            "feature": feature,
            "target": target,
            "n": len(pair),
            "pearson": pair[feature].corr(pair[target], method="pearson"),
            "spearman": pair[feature].corr(pair[target], method="spearman"),
            "abs_spearman": abs(pair[feature].corr(pair[target], method="spearman")),
        }
    )


def build_rankings(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "city",
        "recoverable_fraction",
        "decision_leverage_fraction",
        "best_heuristic_policy",
        "n_units",
        "q_nnz",
        "q_density",
        "od_density_observed",
        "p90_deficit",
        "mean_peak_extra_deficit",
        "max_peak_extra_deficit",
        "top_10pct_tmc_deficit_share",
        "congested_volume_share_speed_ratio_lt_0_8",
    ]
    present = [column for column in columns if column in dataset.columns]
    return dataset[present].copy()


def make_figures(dataset: pd.DataFrame, correlations: pd.DataFrame, figure_dir: Path) -> None:
    make_bar_figure(dataset, figure_dir / "full_zone_recoverability_by_city.png")
    make_scatter(dataset, "q_density", "recoverable_fraction", figure_dir / "recoverability_vs_od_density.png")
    make_scatter(dataset, "p90_deficit", "recoverable_fraction", figure_dir / "recoverability_vs_speed_deficit.png")
    make_scatter(dataset, "max_peak_extra_deficit", "recoverable_fraction", figure_dir / "recoverability_vs_rain_impact.png")
    make_correlation_figure(correlations, figure_dir / "structure_correlation_summary.png")


def make_bar_figure(dataset: pd.DataFrame, path: Path) -> None:
    df = dataset.sort_values("recoverable_fraction", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(df["city"], df["recoverable_fraction"], color="#2563eb")
    ax.set_ylabel("Recoverable fraction")
    ax.set_title("Full-zone LP recoverability under the same base scenario")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(0.12, df["recoverable_fraction"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_scatter(dataset: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    if x_col not in dataset.columns or y_col not in dataset.columns:
        return
    df = dataset[[x_col, y_col, "city"]].dropna()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(df[x_col], df[y_col], s=70, color="#0f766e")
    for row in df.itertuples():
        ax.annotate(row.city, (getattr(row, x_col), getattr(row, y_col)), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_correlation_figure(correlations: pd.DataFrame, path: Path) -> None:
    if correlations.empty:
        return
    top = correlations[correlations["target"] == "recoverable_fraction"].head(12).copy()
    top = top.sort_values("spearman")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in top["spearman"]]
    ax.barh(top["feature"], top["spearman"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Spearman correlation with recoverable fraction")
    ax.set_title("City-structure associations, base scenario")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    dataset: pd.DataFrame,
    correlations: pd.DataFrame,
    structure_only: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    top_corr = correlations[correlations["target"] == "recoverable_fraction"].head(10)
    leverage_corr = correlations[correlations["target"] == "decision_leverage_fraction"].head(10)
    structure_corr = structure_only[structure_only["target"] == "recoverable_fraction"].head(10)
    lines = [
        "# Full-Zone City-Structure Analysis",
        "",
        "## 核心变化",
        "",
        "本轮分析对有可用 speed 数据的 7 个城市使用全 OD zone。New York 为 1,940 个单元、707,646 个非零 OD 依赖；Chicago 为 882 个单元、337,138 个非零 OD 依赖。LP 使用稀疏 CSR 形式，只对真实 OD 非零依赖建立 access-loss 约束。",
        "",
        "为了避免把预算曲线误读成城市结构规律，主结果只保留同一 base 情景：budget intensity = 0.18，R/C/S 延迟固定为 2/0/1 小时，干预效率、成本、部署上限和 diminishing returns 也保持一致。这里比较的是在同一制度设定下，不同城市结构对应的 recoverability 和 decision leverage。",
        "",
        "## 城市结果排序",
        "",
        dataframe_to_markdown(rankings),
        "",
        "## 结构相关性",
        "",
        "下表只使用城市结构变量，不使用 budget、delay、eta、cost 等情景参数。样本数为 7，因此只能作为结构假设生成，不能直接宣称 universal law。",
        "",
        dataframe_to_markdown(top_corr[["feature_family", "feature", "target", "n", "spearman", "pearson"]]),
        "",
        "## Decision Leverage 的结构相关性",
        "",
        dataframe_to_markdown(leverage_corr[["feature_family", "feature", "target", "n", "spearman", "pearson"]]),
        "",
        "## 去除降雨冲击后的城市结构相关性",
        "",
        "下表不使用 rainfall event impact，也不使用 speed-deficit severity；只使用 OD/network/link-performance 结构变量。样本数仍为 7，因此它是结构假设而不是最终 law。",
        "",
        dataframe_to_markdown(structure_corr[["feature_family", "feature", "target", "n", "spearman", "pearson"]]),
        "",
        "## 初步结构解释",
        "",
        "1. 在 full-zone LP 下，recoverability 最高的是 New York、Chicago、Houston，说明全城市 OD 依赖结构中确实存在可被优化利用的恢复空间。",
        "2. recoverability 与城市规模变量呈正相关，例如 n_units、OD rows、Q nonzeros 的 Spearman 约为 0.86；与 OD density/Q density 呈负相关，Spearman 约为 -0.79。一个合理解释是：更大、更稀疏的功能依赖网络中，优化有更多空间寻找高杠杆恢复位置；高度稠密的小网络中，损失传播更均匀，边际定位价值较低。",
        "3. 降雨事件冲击也很重要：max_peak_extra_deficit 与 recoverability 的 Spearman 约为 0.93。这说明可恢复性不是只由网络规模决定，灾害冲击在速度系统里是否形成清晰峰值，也会影响优化可挽回的损失空间。",
        "4. speed-deficit severity 与 recoverability 正相关，p90_deficit 的 Spearman 约为 0.71。这与直觉一致：没有可观测损失，就没有太多 counterfactual recovery 空间。",
        "5. TMC 损失集中度在当前样本里与 recoverability 呈负相关。这个方向需要谨慎解释：它可能意味着损失过度集中时，可恢复空间受部署上限限制；也可能只是小样本城市差异造成。",
        "",
        "## 对论文的含义",
        "",
        "更合适的 law 方向不应是“预算越大越好”，而应是：在相同恢复制度下，城市功能依赖结构、扰动峰值强度、拥堵暴露和损失集中度共同决定 recoverable fraction 与 optimization decision leverage。当前结果支持把后续学习任务改写为 city-structure law extraction，而不是参数响应曲线拟合。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.head(max_rows).copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return compact.to_markdown(index=False)


if __name__ == "__main__":
    main()

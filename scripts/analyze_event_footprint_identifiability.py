"""Audit whether event-level top-tail laws identify spatial footprints.

The current observed-event calibration distributes each event's city-level
abnormal speed signal over zones through an OD vulnerability template. This
script quantifies the consequence: top-tail concentration can become a
city-template signal unless zone-level rainfall/speed footprints add spatial
variation within the same city.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


EPS = 1e-12
KEY_METRICS = [
    "top_1pct_value_share",
    "top_5pct_value_share",
    "top_10pct_value_share",
    "marginal_value_gini",
    "optimizer_selected_value_share",
    "recoverable_fraction",
    "baseline_objective",
    "event_total_precip",
    "event_peak_positive_abnormal_deficit",
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "event_footprint_identifiability"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    concentration = read_csv(root / "results" / "law_learning" / "tables" / "event_value_concentration.csv")
    event_law = read_csv(root / "results" / "law_learning" / "tables" / "event_level_top_tail_law.csv")
    greedy = read_csv(root / "results" / "law_learning" / "tables" / "greedy_oracle_actions.csv.gz")

    event_table = build_event_table(concentration, event_law)
    metric_variation = build_metric_variation(event_table)
    city_summary = build_city_summary(metric_variation, event_table)
    pairwise_units, pairwise_summary = build_pairwise_unit_stability(greedy)
    metrics = build_metrics(event_table, city_summary, pairwise_summary)

    write_table(event_table, table_dir / "event_footprint_event_table.csv")
    write_table(metric_variation, table_dir / "event_footprint_metric_variation.csv")
    write_table(city_summary, table_dir / "event_footprint_city_summary.csv")
    write_table(pairwise_units, table_dir / "event_footprint_pairwise_unit_jaccard.csv")
    write_table(pairwise_summary, table_dir / "event_footprint_unit_jaccard_summary.csv")
    (table_dir / "event_footprint_identifiability_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(event_table, city_summary, pairwise_summary, figure_dir)
    write_report(
        report_dir / "event_footprint_identifiability_report_zh.md",
        metrics,
        city_summary,
        metric_variation,
        pairwise_summary,
    )
    print(f"Wrote event footprint identifiability audit to {output_dir}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def build_event_table(concentration: pd.DataFrame, event_law: pd.DataFrame) -> pd.DataFrame:
    if concentration.empty:
        raise FileNotFoundError("event_value_concentration.csv is required.")
    df = concentration.copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(int)
    if not event_law.empty:
        event_law = event_law.copy()
        event_law["event_id"] = pd.to_numeric(event_law["event_id"], errors="coerce").astype(int)
        keep = [
            col
            for col in [
                "city",
                "event_id",
                "decision_criticality_score",
                "decision_criticality_rank",
                "loss_magnitude_rank",
                "recoverable_rank",
                "top_tail_rank",
            ]
            if col in event_law.columns
        ]
        df = df.merge(event_law[keep], on=["city", "event_id"], how="left")
    df = df.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)
    df["city_event_order"] = df.groupby("city").cumcount() + 1
    return df


def build_metric_variation(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city, group in events.groupby("city", sort=True):
        for metric in KEY_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            metric_range = float(values.max() - values.min())
            rows.append(
                {
                    "city": city,
                    "metric": metric,
                    "n_events": int(len(values)),
                    "n_unique_rounded_1e10": int(values.round(10).nunique()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": metric_range,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "cv_abs": float(values.std(ddof=0) / max(abs(values.mean()), EPS)),
                    "zero_within_city_variance": bool(metric_range <= 1e-10),
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "city"])


def build_city_summary(metric_variation: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for city, group in events.groupby("city", sort=True):
        row: dict[str, Any] = {
            "city": city,
            "n_events": int(len(group)),
        }
        for metric in [
            "top_5pct_value_share",
            "marginal_value_gini",
            "optimizer_selected_value_share",
            "recoverable_fraction",
            "baseline_objective",
        ]:
            match = metric_variation[(metric_variation["city"] == city) & (metric_variation["metric"] == metric)]
            if match.empty:
                continue
            item = match.iloc[0]
            row[f"{metric}_unique"] = int(item["n_unique_rounded_1e10"])
            row[f"{metric}_range"] = float(item["range"])
            row[f"{metric}_std"] = float(item["std"])
            row[f"{metric}_zero_variance"] = bool(item["zero_within_city_variance"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_unit_stability(greedy: pd.DataFrame, top_k_values: tuple[int, ...] = (5, 10, 20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    if greedy.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = greedy.copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(int)
    df["unit"] = df["unit"].astype(str)
    df["value_proxy"] = pd.to_numeric(df["value_proxy"], errors="coerce").fillna(0.0)
    unit_value = (
        df.groupby(["city", "event_id", "unit"], as_index=False)
        .agg(total_value_proxy=("value_proxy", "sum"), total_allocated_cost=("allocated_cost", "sum"))
        .sort_values(["city", "event_id", "total_value_proxy"], ascending=[True, True, False])
    )

    top_sets: dict[tuple[str, int, int], set[str]] = {}
    for (city, event_id), group in unit_value.groupby(["city", "event_id"]):
        group = group.sort_values("total_value_proxy", ascending=False)
        for top_k in top_k_values:
            top_sets[(str(city), int(event_id), top_k)] = set(group.head(top_k)["unit"].astype(str))

    rows: list[dict[str, Any]] = []
    for city, city_events in unit_value[["city", "event_id"]].drop_duplicates().groupby("city"):
        event_ids = sorted(int(value) for value in city_events["event_id"].unique())
        for top_k in top_k_values:
            for event_a, event_b in combinations(event_ids, 2):
                set_a = top_sets.get((str(city), event_a, top_k), set())
                set_b = top_sets.get((str(city), event_b, top_k), set())
                union = set_a | set_b
                inter = set_a & set_b
                rows.append(
                    {
                        "city": city,
                        "top_k_units": top_k,
                        "event_id_a": event_a,
                        "event_id_b": event_b,
                        "jaccard": float(len(inter) / len(union)) if union else np.nan,
                        "intersection_count": int(len(inter)),
                        "union_count": int(len(union)),
                    }
                )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pd.DataFrame()
    summary = (
        pairwise.groupby(["city", "top_k_units"], as_index=False)
        .agg(
            n_event_pairs=("jaccard", "count"),
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            min_jaccard=("jaccard", "min"),
            max_jaccard=("jaccard", "max"),
        )
        .sort_values(["top_k_units", "city"])
    )
    return pairwise, summary


def build_metrics(events: pd.DataFrame, city_summary: pd.DataFrame, pairwise_summary: pd.DataFrame) -> dict[str, Any]:
    top5_zero = city_summary[city_summary.get("top_5pct_value_share_zero_variance", False).astype(bool)]
    gini_zero = city_summary[city_summary.get("marginal_value_gini_zero_variance", False).astype(bool)]
    top5_variable = city_summary[~city_summary.get("top_5pct_value_share_zero_variance", False).astype(bool)]
    top10_pairs = pairwise_summary[pairwise_summary["top_k_units"] == 10] if "top_k_units" in pairwise_summary else pd.DataFrame()
    event_share_zero = float(top5_zero["n_events"].sum() / max(len(events), 1)) if not top5_zero.empty else 0.0
    metrics = {
        "n_events": int(len(events)),
        "n_cities": int(events["city"].nunique()) if "city" in events else 0,
        "top5_zero_variance_city_count": int(len(top5_zero)),
        "top5_zero_variance_event_share": event_share_zero,
        "top5_variable_cities": "; ".join(top5_variable["city"].astype(str).tolist()),
        "gini_zero_variance_city_count": int(len(gini_zero)),
        "gini_zero_variance_event_share": float(gini_zero["n_events"].sum() / max(len(events), 1)) if not gini_zero.empty else 0.0,
        "top5_max_within_city_range": float(city_summary["top_5pct_value_share_range"].max()),
        "gini_max_within_city_range": float(city_summary["marginal_value_gini_range"].max()),
        "optimizer_selected_value_share_mean_range": float(city_summary["optimizer_selected_value_share_range"].mean()),
        "optimizer_selected_value_share_zero_variance_city_count": int(city_summary["optimizer_selected_value_share_zero_variance"].sum()),
        "top10_unit_pairwise_mean_jaccard": float(top10_pairs["mean_jaccard"].mean()) if not top10_pairs.empty else np.nan,
        "top10_unit_pairwise_min_city_jaccard": float(top10_pairs["mean_jaccard"].min()) if not top10_pairs.empty else np.nan,
        "top10_unit_pairwise_max_city_jaccard": float(top10_pairs["mean_jaccard"].max()) if not top10_pairs.empty else np.nan,
        "interpretation": (
            "Current top-tail concentration is largely a city-template signal; "
            "zone-level event footprints are needed before claiming fully resolved within-city footprint laws."
        ),
    }
    return metrics


def make_figures(
    events: pd.DataFrame,
    city_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_range_figure(city_summary, figure_dir / "top_tail_within_city_range.png")
    make_event_line_figure(events, figure_dir / "top_tail_event_lines.png")
    make_jaccard_figure(pairwise_summary, figure_dir / "greedy_top_unit_jaccard.png")


def make_range_figure(city_summary: pd.DataFrame, path: Path) -> None:
    frame = city_summary.sort_values("top_5pct_value_share_range", ascending=False)
    x = np.arange(len(frame))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width, frame["top_5pct_value_share_range"], width, label="top-5% value share range", color="#2563eb")
    ax.bar(x, frame["marginal_value_gini_range"], width, label="marginal-value Gini range", color="#16a34a")
    ax.bar(x + width, frame["optimizer_selected_value_share_range"], width, label="optimizer selected share range", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["city"], rotation=35, ha="right")
    ax.set_ylabel("Within-city range across events")
    ax.set_title("Event-footprint variation currently visible in top-tail metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_event_line_figure(events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for city, group in events.groupby("city", sort=True):
        group = group.sort_values("city_event_order")
        ax.plot(group["city_event_order"], group["top_5pct_value_share"], marker="o", linewidth=1.5, markersize=3, label=city)
    ax.set_xlabel("Event order within city")
    ax.set_ylabel("Top-5% marginal-value share")
    ax.set_title("Within-city event variation in top-tail concentration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_jaccard_figure(pairwise_summary: pd.DataFrame, path: Path) -> None:
    if pairwise_summary.empty:
        return
    frame = pairwise_summary[pairwise_summary["top_k_units"] == 10].sort_values("mean_jaccard", ascending=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(frame["city"], frame["mean_jaccard"], color="#7c3aed", alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean pairwise Jaccard")
    ax.set_title("Stability of top greedy-recovery units across events")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    city_summary: pd.DataFrame,
    metric_variation: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
) -> None:
    key_variation = metric_variation[
        metric_variation["metric"].isin(["top_5pct_value_share", "marginal_value_gini", "optimizer_selected_value_share"])
    ].copy()
    lines = [
        "# Event Spatial Footprint Identifiability V32",
        "",
        "## Question",
        "",
        "This audit asks whether the current event-level top-tail law is identifying event-specific spatial footprints, or whether it is mostly identifying a city-level OD vulnerability template.",
        "",
        "## Main Findings",
        "",
        f"- events/cities audited: {metrics['n_events']} / {metrics['n_cities']}",
        f"- cities with zero within-city variation in top-5% value share: {metrics['top5_zero_variance_city_count']} ({metrics['top5_zero_variance_event_share']:.1%} of events)",
        f"- cities with event-specific top-tail variation: {metrics['top5_variable_cities'] or 'none'}",
        f"- cities with zero within-city variation in marginal-value Gini: {metrics['gini_zero_variance_city_count']} ({metrics['gini_zero_variance_event_share']:.1%} of events)",
        f"- mean within-city range of optimizer-selected value share: {metrics['optimizer_selected_value_share_mean_range']:.4f}",
        f"- top-10 greedy unit mean pairwise Jaccard across cities: {metrics['top10_unit_pairwise_mean_jaccard']:.4f}",
        "",
        "Interpretation: the current top-tail concentration is strongly constrained by the spatial template used in calibration. It remains useful as a city-structure/top-tail law, but not yet as a fully resolved within-city event-footprint law.",
        "",
        "## City Summary",
        "",
        table_to_markdown(city_summary),
        "",
        "## Key Metric Variation",
        "",
        table_to_markdown(key_variation),
        "",
        "## Top Unit Stability",
        "",
        table_to_markdown(pairwise_summary),
        "",
        "## Writing Implication",
        "",
        "The paper should keep the event-level law, but phrase it carefully: decision-criticality is a top-tail law under the current OD-vulnerability spatial calibration. A stronger future claim about rainfall-footprint-specific recovery laws requires zone-level speed/rainfall footprints or a calibrated spatial footprint augmentation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

"""Evaluate explicit OD-message-passing features for action-value laws.

V17 showed that observed OD graph alignment matters, but it used scalar graph
features such as exposure, degree, and scarcity. This script asks a sharper
architecture question from the high-level learning plan: do one-hop and two-hop
OD message features add predictive value beyond the compact activated law and
the existing scalar OD graph features?

The analysis remains intentionally interpretable. It builds deterministic
message-passing summaries from each calibrated city-event OD matrix and then
uses the same leave-one-city ridge evaluation as earlier surrogate tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_factorized_action_surrogate import (
    DEFICIT_FEATURES,
    EVENT_CONTEXT_FEATURES,
    EVENT_KEYS,
    EXPOSURE_FEATURES,
    FACTORIZED_BASE,
    INTERACTION_FEATURES,
    STRUCTURE_FEATURES,
    SUBSTITUTION_FEATURES,
    TIME_FEASIBILITY_FEATURES,
    event_ndcg,
    event_precision,
    event_top_capture,
    fit_ridge,
    prepare_tokens,
    predict_ridge,
)
from learn_recovery_laws import load_inputs
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


RIDGE_ALPHA = 2.0
TOP_FRACS = (0.01, 0.05, 0.10)
EPS = 1e-12

MESSAGE_FEATURES = [
    "mp_out_b0_rank",
    "mp_out_h_total_rank",
    "mp_out_need_rank",
    "mp_out_destination_importance_rank",
    "mp_in_origin_loss_rank",
    "mp_in_origin_h_rank",
    "mp_twohop_destination_importance_rank",
    "mp_twohop_origin_exposure_rank",
    "log_mp_out_b0",
    "log_mp_in_origin_loss",
    "log_mp_twohop_destination_importance",
]

LOCAL_ACTION_FEATURES = DEFICIT_FEATURES + TIME_FEASIBILITY_FEATURES + EVENT_CONTEXT_FEATURES
SCALAR_OD_FEATURES = EXPOSURE_FEATURES + STRUCTURE_FEATURES + SUBSTITUTION_FEATURES

MODEL_SPECS = [
    {
        "model_id": "O0_local_action_no_od",
        "family": "no_message",
        "description": "local dynamics, action mechanics, and event context; no OD graph",
        "features": LOCAL_ACTION_FEATURES,
    },
    {
        "model_id": "O1_scalar_od_graph",
        "family": "scalar_od",
        "description": "local/action features plus scalar OD exposure, degree, and scarcity",
        "features": LOCAL_ACTION_FEATURES + SCALAR_OD_FEATURES,
    },
    {
        "model_id": "O2_message_only_od",
        "family": "message_od",
        "description": "local/action features plus OD message-passing summaries only",
        "features": LOCAL_ACTION_FEATURES + MESSAGE_FEATURES,
    },
    {
        "model_id": "O3_scalar_plus_message",
        "family": "message_od",
        "description": "scalar OD graph features plus OD message-passing summaries",
        "features": LOCAL_ACTION_FEATURES + SCALAR_OD_FEATURES + MESSAGE_FEATURES,
    },
    {
        "model_id": "O4_factorized_low_dim",
        "family": "factorized_law",
        "description": "compact activated law components",
        "features": FACTORIZED_BASE,
    },
    {
        "model_id": "O5_factorized_plus_message",
        "family": "factorized_law",
        "description": "compact activated law plus OD message-passing summaries",
        "features": FACTORIZED_BASE + MESSAGE_FEATURES,
    },
    {
        "model_id": "O6_full_interaction_reference",
        "family": "full_reference",
        "description": "full additive feature set plus explicit interaction terms",
        "features": LOCAL_ACTION_FEATURES + SCALAR_OD_FEATURES + INTERACTION_FEATURES,
    },
]


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "od_message_passing_surrogate"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(root)
    message_features = build_message_features(root, tokens)
    tokens = tokens.merge(message_features, on=["city", "event_id", "unit"], how="left")
    tokens = fill_message_features(tokens)
    validate_features(tokens)

    leave_city, event_metrics, coefficients = run_leave_city_out(tokens)
    model_summary = summarize_models(leave_city, event_metrics)
    increments = build_incremental_gains(model_summary)
    diagnostics = build_diagnostics(model_summary, increments)

    write_table(message_features, table_dir / "od_message_unit_features.csv.gz")
    write_table(model_summary, table_dir / "od_message_model_summary.csv")
    write_table(leave_city, table_dir / "od_message_leave_city_metrics.csv")
    write_table(event_metrics, table_dir / "od_message_event_metrics.csv")
    write_table(coefficients, table_dir / "od_message_coefficients.csv")
    write_table(increments, table_dir / "od_message_incremental_gains.csv")
    (table_dir / "od_message_passing_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_figures(model_summary, increments, coefficients, figure_dir)
    write_report(
        report_dir / "od_message_passing_report_zh.md",
        diagnostics,
        model_summary,
        increments,
    )
    print(f"Wrote OD message-passing surrogate analysis to {output_dir}")


def load_tokens(root: Path) -> pd.DataFrame:
    path = root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing action-token table: {path}")
    tokens = pd.read_csv(path)
    tokens = prepare_tokens(tokens)
    tokens["unit"] = tokens["unit"].astype(str)
    tokens["event_id"] = pd.to_numeric(tokens["event_id"], errors="coerce").astype(int)
    return tokens


def build_message_features(root: Path, tokens: pd.DataFrame) -> pd.DataFrame:
    config = load_yaml(root / "configs" / "optimization.yml")
    data = load_inputs(root)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    abnormal = data["abnormal"].copy()

    rows: list[pd.DataFrame] = []
    unique_events = tokens[["city", "event_id"]].drop_duplicates().sort_values(["city", "event_id"])
    total = len(unique_events)
    for idx, row in enumerate(unique_events.itertuples(index=False), start=1):
        city = str(row.city)
        event_id = int(row.event_id)
        event_row = event_lookup.get((city, event_id))
        if event_row is None or city not in dynamic_lookup:
            continue
        print(f"[{idx}/{total}] Building OD messages for {city} event {event_id}", flush=True)
        params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        rows.append(message_frame_for_params(params, event_id))
    if not rows:
        return pd.DataFrame(columns=["city", "event_id", "unit", *MESSAGE_FEATURES])
    return pd.concat(rows, ignore_index=True)


def message_frame_for_params(params: Any, event_id: int) -> pd.DataFrame:
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    p = np.asarray(params.p, dtype=float)
    b0 = np.asarray(params.b0, dtype=float)
    h_total = np.asarray(params.h.sum(axis=1), dtype=float)
    local_need = np.clip(b0 + h_total, 0.02, 1.0)
    destination_importance = np.asarray(q.T @ p).ravel()
    origin_exposure = p

    out_b0 = np.asarray(q @ b0).ravel()
    out_h = np.asarray(q @ h_total).ravel()
    out_need = np.asarray(q @ local_need).ravel()
    out_destination_importance = np.asarray(q @ destination_importance).ravel()

    upstream_weight = np.asarray(q.T @ p).ravel()
    in_origin_loss = safe_divide(np.asarray(q.T @ (p * b0)).ravel(), upstream_weight)
    in_origin_h = safe_divide(np.asarray(q.T @ (p * h_total)).ravel(), upstream_weight)

    twohop_destination_importance = np.asarray(q.T @ destination_importance).ravel()
    twohop_origin_exposure = np.asarray(q @ origin_exposure).ravel()

    frame = pd.DataFrame(
        {
            "city": params.city,
            "event_id": int(event_id),
            "unit": np.asarray(params.units, dtype=str),
            "mp_out_b0": out_b0,
            "mp_out_h_total": out_h,
            "mp_out_need": out_need,
            "mp_out_destination_importance": out_destination_importance,
            "mp_in_origin_loss": in_origin_loss,
            "mp_in_origin_h": in_origin_h,
            "mp_twohop_destination_importance": twohop_destination_importance,
            "mp_twohop_origin_exposure": twohop_origin_exposure,
        }
    )
    for col in [
        "mp_out_b0",
        "mp_out_h_total",
        "mp_out_need",
        "mp_out_destination_importance",
        "mp_in_origin_loss",
        "mp_in_origin_h",
        "mp_twohop_destination_importance",
        "mp_twohop_origin_exposure",
    ]:
        frame[f"{col}_rank"] = rank_pct(frame[col].to_numpy(dtype=float))
    frame["log_mp_out_b0"] = np.log1p(100.0 * frame["mp_out_b0"].clip(lower=0.0))
    frame["log_mp_in_origin_loss"] = np.log1p(100.0 * frame["mp_in_origin_loss"].clip(lower=0.0))
    frame["log_mp_twohop_destination_importance"] = np.log1p(
        1_000.0 * frame["mp_twohop_destination_importance"].clip(lower=0.0)
    )
    return frame


def fill_message_features(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    for col in MESSAGE_FEATURES:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def validate_features(tokens: pd.DataFrame) -> None:
    missing: list[str] = []
    for spec in MODEL_SPECS:
        for feature in spec["features"]:
            if feature not in tokens:
                missing.append(feature)
    if missing:
        raise KeyError(f"Missing OD message features: {sorted(set(missing))}")


def run_leave_city_out(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        features = list(spec["features"])
        for heldout_city in sorted(tokens["city"].dropna().unique()):
            train = tokens[tokens["city"] != heldout_city].copy()
            test = tokens[tokens["city"] == heldout_city].copy()
            model = fit_ridge(train[features], train["target_log"], alpha=RIDGE_ALPHA)
            pred_log = predict_ridge(model, test[features])
            test["predicted_value"] = np.expm1(pred_log) / 1_000.0
            base = {
                "model_id": spec["model_id"],
                "family": spec["family"],
                "description": spec["description"],
                "heldout_city": heldout_city,
                "n_features": len(features),
            }
            metric_rows.append({**base, **prediction_metrics(test, "predicted_value")})
            event_rows.extend(event_metric_rows(test, spec, heldout_city))
            for feature, coef in zip(features, model["coef"][1:]):
                coef_rows.append(
                    {
                        "model_id": spec["model_id"],
                        "heldout_city": heldout_city,
                        "feature": feature,
                        "standardized_coef": float(coef),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows), pd.DataFrame(coef_rows)


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["target_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[EVENT_KEYS].drop_duplicates().shape[0]),
        "pearson": corr(frame["target_value"], frame[score_col], method="pearson"),
        "spearman": corr(frame["target_value"], frame[score_col], method="spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in TOP_FRACS:
        label = f"top_{int(frac * 100)}pct"
        values = [event_top_capture(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        ndcg = [event_ndcg(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        precision = [event_precision(group, score_col, frac) for _, group in frame.groupby(EVENT_KEYS, sort=False)]
        out[f"{label}_value_capture"] = safe_nanmean(values)
        out[f"{label}_ndcg"] = safe_nanmean(ndcg)
        out[f"{label}_precision"] = safe_nanmean(precision)
        out[f"{label}_regret"] = 1.0 - out[f"{label}_value_capture"]
    return out


def event_metric_rows(frame: pd.DataFrame, spec: dict[str, Any], heldout_city: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (city, event_id), group in frame.groupby(EVENT_KEYS, sort=True):
        row = {
            "model_id": spec["model_id"],
            "family": spec["family"],
            "description": spec["description"],
            "heldout_city": heldout_city,
            "city": city,
            "event_id": int(event_id),
            "n_tokens": int(len(group)),
            "total_value": float(group["target_value"].sum()),
            "spearman": corr(group["target_value"], group["predicted_value"], method="spearman"),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            row[f"{label}_value_capture"] = event_top_capture(group, "predicted_value", frac)
            row[f"{label}_ndcg"] = event_ndcg(group, "predicted_value", frac)
            row[f"{label}_precision"] = event_precision(group, "predicted_value", frac)
            row[f"{label}_regret"] = 1.0 - row[f"{label}_value_capture"]
        rows.append(row)
    return rows


def summarize_models(leave_city: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_id = spec["model_id"]
        city_group = leave_city[leave_city["model_id"].eq(model_id)]
        event_group = event_metrics[event_metrics["model_id"].eq(model_id)]
        row = {
            "model_id": model_id,
            "family": spec["family"],
            "description": spec["description"],
            "n_features": len(spec["features"]),
            "n_cities": int(city_group["heldout_city"].nunique()),
            "n_events": int(event_group[EVENT_KEYS].drop_duplicates().shape[0]),
            "mean_city_spearman": float(city_group["spearman"].mean()),
            "mean_event_spearman": float(event_group["spearman"].mean()),
            "median_event_spearman": float(event_group["spearman"].median()),
        }
        for frac in TOP_FRACS:
            label = f"top_{int(frac * 100)}pct"
            for metric in ["value_capture", "ndcg", "precision", "regret"]:
                row[f"mean_event_{label}_{metric}"] = float(event_group[f"{label}_{metric}"].mean())
                row[f"median_event_{label}_{metric}"] = float(event_group[f"{label}_{metric}"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_incremental_gains(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("scalar_od_over_local", "O0_local_action_no_od", "O1_scalar_od_graph"),
        ("message_od_over_local", "O0_local_action_no_od", "O2_message_only_od"),
        ("message_over_scalar_od", "O1_scalar_od_graph", "O3_scalar_plus_message"),
        ("message_over_factorized", "O4_factorized_low_dim", "O5_factorized_plus_message"),
        ("factorized_over_local", "O0_local_action_no_od", "O4_factorized_low_dim"),
        ("full_reference_over_scalar_message", "O3_scalar_plus_message", "O6_full_interaction_reference"),
    ]
    rows: list[dict[str, Any]] = []
    for comparison, base_id, next_id in comparisons:
        base = one_row(summary, model_id=base_id)
        nxt = one_row(summary, model_id=next_id)
        if base.empty or nxt.empty:
            continue
        rows.append(
            {
                "comparison": comparison,
                "base_model": base_id,
                "next_model": next_id,
                "delta_top5_value_capture": safe_float(nxt.get("mean_event_top_5pct_value_capture"))
                - safe_float(base.get("mean_event_top_5pct_value_capture")),
                "delta_top5_ndcg": safe_float(nxt.get("mean_event_top_5pct_ndcg"))
                - safe_float(base.get("mean_event_top_5pct_ndcg")),
                "delta_top5_precision": safe_float(nxt.get("mean_event_top_5pct_precision"))
                - safe_float(base.get("mean_event_top_5pct_precision")),
                "delta_event_spearman": safe_float(nxt.get("mean_event_spearman"))
                - safe_float(base.get("mean_event_spearman")),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(summary: pd.DataFrame, increments: pd.DataFrame) -> dict[str, Any]:
    local = one_row(summary, model_id="O0_local_action_no_od")
    scalar = one_row(summary, model_id="O1_scalar_od_graph")
    message = one_row(summary, model_id="O2_message_only_od")
    scalar_message = one_row(summary, model_id="O3_scalar_plus_message")
    factorized = one_row(summary, model_id="O4_factorized_low_dim")
    factorized_message = one_row(summary, model_id="O5_factorized_plus_message")
    full = one_row(summary, model_id="O6_full_interaction_reference")
    msg_over_scalar = one_row(increments, comparison="message_over_scalar_od")
    msg_over_factorized = one_row(increments, comparison="message_over_factorized")
    msg_over_local = one_row(increments, comparison="message_od_over_local")
    scalar_over_local = one_row(increments, comparison="scalar_od_over_local")
    return {
        "local_top5_capture": safe_float(local.get("mean_event_top_5pct_value_capture")),
        "scalar_od_top5_capture": safe_float(scalar.get("mean_event_top_5pct_value_capture")),
        "message_od_top5_capture": safe_float(message.get("mean_event_top_5pct_value_capture")),
        "scalar_plus_message_top5_capture": safe_float(scalar_message.get("mean_event_top_5pct_value_capture")),
        "factorized_top5_capture": safe_float(factorized.get("mean_event_top_5pct_value_capture")),
        "factorized_plus_message_top5_capture": safe_float(factorized_message.get("mean_event_top_5pct_value_capture")),
        "full_reference_top5_capture": safe_float(full.get("mean_event_top_5pct_value_capture")),
        "scalar_od_over_local_delta_top5": safe_float(scalar_over_local.get("delta_top5_value_capture")),
        "message_od_over_local_delta_top5": safe_float(msg_over_local.get("delta_top5_value_capture")),
        "message_over_scalar_od_delta_top5": safe_float(msg_over_scalar.get("delta_top5_value_capture")),
        "message_over_factorized_delta_top5": safe_float(msg_over_factorized.get("delta_top5_value_capture")),
    }


def make_figures(summary: pd.DataFrame, increments: pd.DataFrame, coefficients: pd.DataFrame, figure_dir: Path) -> None:
    make_model_ladder(summary, figure_dir / "od_message_model_ladder.png")
    make_increment_figure(increments, figure_dir / "od_message_incremental_gains.png")
    make_coefficient_figure(coefficients, figure_dir / "od_message_coefficients.png")


def make_model_ladder(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    labels = {
        "O0_local_action_no_od": "local",
        "O1_scalar_od_graph": "scalar OD",
        "O2_message_only_od": "message OD",
        "O3_scalar_plus_message": "scalar+message",
        "O4_factorized_low_dim": "factorized",
        "O5_factorized_plus_message": "factorized+message",
        "O6_full_interaction_reference": "full reference",
    }
    ordered = summary[summary["model_id"].isin(labels)].copy()
    ordered["label"] = pd.Categorical(ordered["model_id"].map(labels), categories=list(labels.values()), ordered=True)
    ordered = ordered.sort_values("label")
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    colors = ["#94a3b8", "#2563eb", "#38bdf8", "#0f766e", "#9333ea", "#7c3aed", "#111827"]
    ax.bar(ordered["label"].astype(str), ordered["mean_event_top_5pct_value_capture"], color=colors[: len(ordered)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Leave-city top-5% value capture")
    ax.set_title("Explicit OD message passing versus compact recovery law")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(ordered["mean_event_top_5pct_value_capture"]):
        ax.text(idx, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_increment_figure(increments: pd.DataFrame, path: Path) -> None:
    if increments.empty:
        return
    plot = increments.copy()
    plot["label"] = plot["comparison"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in plot["delta_top5_value_capture"]]
    ax.barh(plot["label"], plot["delta_top5_value_capture"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Delta top-5% value capture")
    ax.set_title("Does OD message passing add decision-centered signal?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_coefficient_figure(coefficients: pd.DataFrame, path: Path) -> None:
    if coefficients.empty:
        return
    subset = coefficients[
        coefficients["model_id"].isin(["O3_scalar_plus_message", "O5_factorized_plus_message"])
        & coefficients["feature"].isin(MESSAGE_FEATURES)
    ].copy()
    if subset.empty:
        return
    summary = (
        subset.groupby(["model_id", "feature"], as_index=False)["standardized_coef"]
        .mean()
        .sort_values("standardized_coef", key=lambda s: s.abs(), ascending=False)
        .head(18)
    )
    summary["label"] = summary["model_id"] + ": " + summary["feature"]
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in summary["standardized_coef"]]
    ax.barh(summary["label"], summary["standardized_coef"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Mean standardized coefficient")
    ax.set_title("Largest OD-message coefficients under leave-city fits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    diagnostics: dict[str, Any],
    summary: pd.DataFrame,
    increments: pd.DataFrame,
) -> None:
    msg_scalar_delta = diagnostics["message_over_scalar_od_delta_top5"]
    msg_factorized_delta = diagnostics["message_over_factorized_delta_top5"]
    if max(abs(msg_scalar_delta), abs(msg_factorized_delta)) < 0.02:
        interpretation = (
            "OD message features 的边际增益很小，说明在当前大规模 all-zone action labels 下，"
            "一跳/两跳 OD 邻域信息主要已经被 OD exposure、future-loss horizon、feasibility "
            "和 eta/cost 这些低维 activated terms 捕捉。显式 message-passing 目前更适合作为"
            "稳健性检验，不是主 law 的必要条件。"
        )
    elif msg_scalar_delta > 0 or msg_factorized_delta > 0:
        interpretation = (
            "OD message features 带来正向增益，说明单点 exposure、degree、scarcity 还不能完全概括"
            "高阶 OD 邻域中的 recoverable value。后续如果继续提高模型复杂度，应优先把 OD graph encoder "
            "作为 neural surrogate 的核心模块。"
        )
    else:
        interpretation = (
            "OD message features 没有改善 top-tail ranking，可能是因为这些 message summaries 与已有"
            "低维结构变量高度共线，或者 leave-city 条件下额外图特征增加了方差。当前结论应偏向 compact law。"
        )

    lines = [
        "# OD Message-Passing Surrogate V22",
        "",
        "## 本版要回答的问题",
        "",
        "V17 已经证明 observed OD graph alignment 很重要，但那里主要使用 exposure、degree、scarcity 等 scalar graph features。V22 进一步构造一跳和两跳 OD message-passing summaries，检验显式 OD-neighborhood 信息是否能超越低维 activated law。",
        "",
        "## 主要结果",
        "",
        f"- local/action no-OD model top-5% capture = {diagnostics['local_top5_capture']:.4f}。",
        f"- scalar OD graph features = {diagnostics['scalar_od_top5_capture']:.4f}，相对 local 增量 {diagnostics['scalar_od_over_local_delta_top5']:+.4f}。",
        f"- OD message-only features = {diagnostics['message_od_top5_capture']:.4f}，相对 local 增量 {diagnostics['message_od_over_local_delta_top5']:+.4f}。",
        f"- scalar + message = {diagnostics['scalar_plus_message_top5_capture']:.4f}，message 相对 scalar OD 的增量 {diagnostics['message_over_scalar_od_delta_top5']:+.4f}。",
        f"- low-dimensional factorized law = {diagnostics['factorized_top5_capture']:.4f}，factorized + message = {diagnostics['factorized_plus_message_top5_capture']:.4f}，增量 {diagnostics['message_over_factorized_delta_top5']:+.4f}。",
        f"- full interaction reference top-5% capture = {diagnostics['full_reference_top5_capture']:.4f}。",
        "",
        "## 解释",
        "",
        interpretation,
        "",
        "## Model Summary",
        "",
        table_to_markdown(summary),
        "",
        "## Incremental Gains",
        "",
        table_to_markdown(increments),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=np.abs(den) > EPS)


def rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), 0.5, dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def corr(x: Any, y: Any, *, method: str) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(pair["x"].corr(pair["y"], method=method))


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


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


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def table_to_markdown(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


if __name__ == "__main__":
    main()

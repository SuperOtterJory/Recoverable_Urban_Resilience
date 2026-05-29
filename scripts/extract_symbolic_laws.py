"""Extract compact symbolic recoverability laws from action-value tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


TARGET = "marginal_resource_value"
EPS = 1e-12


CANDIDATE_FORMULAS: tuple[dict[str, Any], ...] = (
    {
        "formula_id": "F0_delay_feasible",
        "formula": "delay_feasible",
        "score_col": "delay_feasible",
        "complexity": 1,
        "family": "single_factor",
    },
    {
        "formula_id": "F1_future_horizon",
        "formula": "delay_feasible * active_weighted_horizon",
        "score_col": "formula_future_horizon",
        "complexity": 2,
        "family": "single_factor",
    },
    {
        "formula_id": "F2_exposure",
        "formula": "delay_feasible * law_exposure_term",
        "score_col": "formula_exposure",
        "complexity": 2,
        "family": "single_factor",
    },
    {
        "formula_id": "F3_efficiency",
        "formula": "delay_feasible * eta_per_cost",
        "score_col": "formula_efficiency",
        "complexity": 2,
        "family": "single_factor",
    },
    {
        "formula_id": "F4_horizon_x_exposure",
        "formula": "delay_feasible * active_weighted_horizon * law_exposure_term",
        "score_col": "formula_horizon_x_exposure",
        "complexity": 3,
        "family": "interaction",
    },
    {
        "formula_id": "F5_horizon_x_efficiency",
        "formula": "delay_feasible * active_weighted_horizon * eta_per_cost",
        "score_col": "formula_horizon_x_efficiency",
        "complexity": 3,
        "family": "interaction",
    },
    {
        "formula_id": "F6_exposure_x_efficiency",
        "formula": "delay_feasible * law_exposure_term * eta_per_cost",
        "score_col": "formula_exposure_x_efficiency",
        "complexity": 3,
        "family": "interaction",
    },
    {
        "formula_id": "F7_activated_recovery_law",
        "formula": "delay_feasible * active_weighted_horizon * law_exposure_term * eta_per_cost",
        "score_col": "activated_bottleneck_score",
        "complexity": 4,
        "family": "symbolic_law",
    },
    {
        "formula_id": "F8_rank_interaction",
        "formula": "delay_feasible * active_future_loss_share * exposure_rank * eta_per_cost_rank",
        "score_col": "formula_rank_interaction",
        "complexity": 4,
        "family": "rank_proxy",
    },
)


REGRESSION_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "formula_id": "R1_log_horizon",
        "formula": "linear(log active horizon)",
        "features": ["log_active_weighted_horizon", "delay_feasible"],
        "complexity": 2,
    },
    {
        "formula_id": "R2_log_exposure",
        "formula": "linear(log exposure)",
        "features": ["log_law_exposure", "delay_feasible"],
        "complexity": 2,
    },
    {
        "formula_id": "R3_log_efficiency",
        "formula": "linear(log eta/cost)",
        "features": ["log_eta_per_cost", "delay_feasible"],
        "complexity": 2,
    },
    {
        "formula_id": "R4_log_horizon_exposure",
        "formula": "linear(log active horizon + log exposure)",
        "features": ["log_active_weighted_horizon", "log_law_exposure", "delay_feasible"],
        "complexity": 3,
    },
    {
        "formula_id": "R5_log_horizon_efficiency",
        "formula": "linear(log active horizon + log eta/cost)",
        "features": ["log_active_weighted_horizon", "log_eta_per_cost", "delay_feasible"],
        "complexity": 3,
    },
    {
        "formula_id": "R6_log_exposure_efficiency",
        "formula": "linear(log exposure + log eta/cost)",
        "features": ["log_law_exposure", "log_eta_per_cost", "delay_feasible"],
        "complexity": 3,
    },
    {
        "formula_id": "R7_minimal_log_law",
        "formula": "linear(log horizon + log exposure + log eta/cost + delay)",
        "features": ["log_active_weighted_horizon", "log_law_exposure", "log_eta_per_cost", "delay_feasible"],
        "complexity": 4,
    },
    {
        "formula_id": "R8_minimal_plus_intervention",
        "formula": "minimal log law + intervention type",
        "features": [
            "log_active_weighted_horizon",
            "log_law_exposure",
            "log_eta_per_cost",
            "delay_feasible",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
        "complexity": 7,
    },
    {
        "formula_id": "R9_augmented_structure",
        "formula": "minimal log law + ranks + scarcity",
        "features": [
            "log_active_weighted_horizon",
            "log_law_exposure",
            "log_eta_per_cost",
            "delay_feasible",
            "local_remaining_rank",
            "access_remaining_rank",
            "origin_exposure_rank",
            "destination_importance_rank",
            "od_scarcity",
            "eta_per_cost_rank",
        ],
        "complexity": 10,
    },
)


FEATURE_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group_id": "state_horizon",
        "description": "remaining loss state and active future horizon",
        "features": [
            "t_frac",
            "time_remaining_frac",
            "log_active_weighted_horizon",
            "active_future_loss_share",
            "local_remaining_rank",
            "access_remaining_rank",
            "passive_b_rank",
            "passive_ell_rank",
        ],
    },
    {
        "group_id": "od_exposure_structure",
        "description": "OD exposure, destination importance, and structural scarcity",
        "features": [
            "log_law_exposure",
            "origin_exposure_rank",
            "destination_importance_rank",
            "out_degree_rank",
            "in_degree_rank",
            "od_scarcity",
        ],
    },
    {
        "group_id": "intervention_feasibility",
        "description": "response delay, intervention type, and effectiveness per cost",
        "features": [
            "delay_feasible",
            "delay_fraction",
            "log_eta_per_cost",
            "eta_per_cost_rank",
            "intervention_R",
            "intervention_C",
            "intervention_S",
        ],
    },
    {
        "group_id": "event_context",
        "description": "event rain severity, event loss scale, and budget context",
        "features": [
            "budget_fraction_of_baseline",
            "log_event_total_precip",
            "log_event_peak_precip",
            "event_peak_positive_abnormal_deficit",
            "weighted_b0",
            "weighted_h_total",
        ],
    },
    {
        "group_id": "interaction_terms",
        "description": "pre-built deficit/exposure/scarcity interactions",
        "features": [
            "deficit_x_exposure",
            "access_x_origin",
            "deficit_x_exposure_x_scarcity",
        ],
    },
)


def main() -> None:
    root = find_repo_root()
    output_dir = root / "results" / "symbolic_law_extraction"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_csv(root / "results" / "law_learning" / "tables" / "action_value_tokens.csv.gz")
    tokens = prepare_tokens(tokens)
    direct_metrics = evaluate_direct_formulas(tokens)
    regression_metrics, coefficients, leave_city_metrics = evaluate_regression_formulas(tokens)
    group_ablation = evaluate_feature_group_ablation(tokens)
    combined = pd.concat([direct_metrics, regression_metrics], ignore_index=True)
    combined = combined.sort_values(["mean_top_5pct_value_capture", "mean_spearman"], ascending=False)
    pareto = pareto_frontier(combined)
    event_metrics = event_level_formula_metrics(tokens)

    write_table(direct_metrics, table_dir / "direct_formula_metrics.csv")
    write_table(regression_metrics, table_dir / "regression_formula_metrics.csv")
    write_table(coefficients, table_dir / "regression_formula_coefficients.csv")
    write_table(leave_city_metrics, table_dir / "formula_leave_city_metrics.csv")
    write_table(group_ablation, table_dir / "feature_group_ablation.csv")
    write_table(combined, table_dir / "symbolic_formula_metrics.csv")
    write_table(pareto, table_dir / "symbolic_formula_pareto_frontier.csv")
    write_table(event_metrics, table_dir / "event_formula_metrics.csv")

    make_figures(combined, pareto, coefficients, group_ablation, event_metrics, figure_dir)
    write_report(
        report_dir / "symbolic_law_extraction_report_zh.md",
        tokens,
        combined,
        pareto,
        coefficients,
        leave_city_metrics,
        group_ablation,
        event_metrics,
    )
    print(f"Wrote symbolic law extraction to {output_dir}")


def prepare_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    df["law_exposure_term"] = pd.to_numeric(df["law_exposure_term"], errors="coerce").fillna(0.0)
    df["active_weighted_horizon"] = pd.to_numeric(df["active_weighted_horizon"], errors="coerce").fillna(0.0)
    df["eta_per_cost"] = pd.to_numeric(df["eta_per_cost"], errors="coerce").fillna(0.0)
    df["delay_feasible"] = pd.to_numeric(df["delay_feasible"], errors="coerce").fillna(0.0)
    df["log_law_exposure"] = np.log1p(1_000.0 * df["law_exposure_term"].clip(lower=0.0))
    if "log_active_weighted_horizon" not in df:
        df["log_active_weighted_horizon"] = np.log1p(df["active_weighted_horizon"].clip(lower=0.0))
    if "log_eta_per_cost" not in df:
        df["log_eta_per_cost"] = np.log1p(df["eta_per_cost"].clip(lower=0.0))
    for spec in FEATURE_GROUPS:
        for column in spec["features"]:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    exposure_rank = np.where(
        df["intervention"].astype(str).eq("S"),
        df["origin_exposure_rank"],
        df["destination_importance_rank"],
    )
    df["formula_future_horizon"] = df["delay_feasible"] * df["active_weighted_horizon"]
    df["formula_exposure"] = df["delay_feasible"] * df["law_exposure_term"]
    df["formula_efficiency"] = df["delay_feasible"] * df["eta_per_cost"]
    df["formula_horizon_x_exposure"] = df["delay_feasible"] * df["active_weighted_horizon"] * df["law_exposure_term"]
    df["formula_horizon_x_efficiency"] = df["delay_feasible"] * df["active_weighted_horizon"] * df["eta_per_cost"]
    df["formula_exposure_x_efficiency"] = df["delay_feasible"] * df["law_exposure_term"] * df["eta_per_cost"]
    df["formula_rank_interaction"] = (
        df["delay_feasible"]
        * df["active_future_loss_share"]
        * exposure_rank
        * df["eta_per_cost_rank"]
    )
    df["target_log"] = np.log1p(1_000.0 * df[TARGET].clip(lower=0.0))
    return df


def evaluate_direct_formulas(tokens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in CANDIDATE_FORMULAS:
        metrics = aggregate_formula_metrics(tokens, spec["score_col"])
        rows.append(
            {
                "formula_id": spec["formula_id"],
                "model_type": "direct_symbolic",
                "family": spec["family"],
                "formula": spec["formula"],
                "complexity": int(spec["complexity"]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_regression_formulas(tokens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    leave_city_rows: list[dict[str, Any]] = []
    for spec in REGRESSION_CANDIDATES:
        prediction_frames = []
        split_coef_rows = []
        for city in sorted(tokens["city"].unique()):
            train = tokens[tokens["city"] != city].copy()
            test = tokens[tokens["city"] == city].copy()
            features = available_features(tokens, spec["features"])
            model = fit_linear(train[features], train["target_log"], alpha=1.0)
            pred_log = predict_linear(model, test[features])
            pred = np.expm1(pred_log) / 1_000.0
            part = test[["city", "event_id", TARGET]].copy()
            part["score"] = np.maximum(pred, 0.0)
            prediction_frames.append(part)
            split_metrics = aggregate_formula_metrics(part, "score")
            leave_city_rows.append(
                {
                    "formula_id": spec["formula_id"],
                    "heldout_city": city,
                    "model_type": "leave_city_log_linear",
                    "formula": spec["formula"],
                    "complexity": int(spec["complexity"]),
                    **split_metrics,
                }
            )
            for feature, coef in zip(["intercept", *features], model["coef"], strict=True):
                split_coef_rows.append(
                    {
                        "formula_id": spec["formula_id"],
                        "heldout_city": city,
                        "feature": feature,
                        "coefficient": float(coef),
                    }
                )
        predictions = pd.concat(prediction_frames, ignore_index=True)
        metrics = aggregate_formula_metrics(predictions, "score")
        metric_rows.append(
            {
                "formula_id": spec["formula_id"],
                "model_type": "leave_city_log_linear",
                "family": "formula_extractor",
                "formula": spec["formula"],
                "complexity": int(spec["complexity"]),
                **metrics,
            }
        )
        coef_rows.extend(split_coef_rows)
    coefficients = summarize_coefficients(pd.DataFrame(coef_rows))
    return pd.DataFrame(metric_rows), coefficients, pd.DataFrame(leave_city_rows)


def evaluate_feature_group_ablation(tokens: pd.DataFrame) -> pd.DataFrame:
    groups = {
        spec["group_id"]: available_features(tokens, spec["features"])
        for spec in FEATURE_GROUPS
    }
    groups = {key: value for key, value in groups.items() if value}
    all_features = sorted({feature for features in groups.values() for feature in features})
    rows: list[dict[str, Any]] = []

    full_metrics = leave_city_model_metrics(tokens, all_features)
    rows.append(
        {
            "ablation_id": "full_interpretable",
            "mode": "all_groups",
            "removed_group": "",
            "included_groups": ",".join(groups),
            "n_features": len(all_features),
            **full_metrics,
        }
    )

    for group_id, features in groups.items():
        only_metrics = leave_city_model_metrics(tokens, features)
        rows.append(
            {
                "ablation_id": f"only_{group_id}",
                "mode": "single_group_only",
                "removed_group": "",
                "included_groups": group_id,
                "n_features": len(features),
                **only_metrics,
            }
        )

        remaining = [feature for feature in all_features if feature not in set(features)]
        without_metrics = leave_city_model_metrics(tokens, remaining)
        rows.append(
            {
                "ablation_id": f"without_{group_id}",
                "mode": "leave_one_group_out",
                "removed_group": group_id,
                "included_groups": ",".join(key for key in groups if key != group_id),
                "n_features": len(remaining),
                **without_metrics,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        full_top5 = float(out.loc[out["ablation_id"].eq("full_interpretable"), "mean_top_5pct_value_capture"].iloc[0])
        full_spearman = float(out.loc[out["ablation_id"].eq("full_interpretable"), "mean_spearman"].iloc[0])
        out["top5_capture_drop_vs_full"] = full_top5 - out["mean_top_5pct_value_capture"]
        out["spearman_drop_vs_full"] = full_spearman - out["mean_spearman"]
    return out


def leave_city_model_metrics(tokens: pd.DataFrame, features: list[str]) -> dict[str, float]:
    prediction_frames = []
    for city in sorted(tokens["city"].unique()):
        train = tokens[tokens["city"] != city].copy()
        test = tokens[tokens["city"] == city].copy()
        model = fit_linear(train[features], train["target_log"], alpha=1.0)
        pred_log = predict_linear(model, test[features])
        pred = np.expm1(pred_log) / 1_000.0
        part = test[["city", "event_id", TARGET]].copy()
        part["score"] = np.maximum(pred, 0.0)
        prediction_frames.append(part)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return aggregate_formula_metrics(predictions, "score")


def available_features(tokens: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in tokens.columns]


def fit_linear(x: pd.DataFrame, y: pd.Series, *, alpha: float) -> dict[str, np.ndarray]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    mean = np.nanmean(x_arr, axis=0)
    std = np.nanstd(x_arr, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    x_std = np.nan_to_num((x_arr - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_arr)
    return {"coef": coef, "mean": mean, "std": std}


def predict_linear(model: dict[str, np.ndarray], x: pd.DataFrame) -> np.ndarray:
    x_arr = x.to_numpy(dtype=float)
    x_std = np.nan_to_num((x_arr - model["mean"]) / model["std"], nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return np.maximum(design @ model["coef"], 0.0)


def summarize_coefficients(coefs: pd.DataFrame) -> pd.DataFrame:
    if coefs.empty:
        return pd.DataFrame()
    return (
        coefs.groupby(["formula_id", "feature"], as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            sd_coefficient=("coefficient", "std"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
        )
        .sort_values(["formula_id", "feature"])
    )


def aggregate_formula_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    score = pd.to_numeric(frame[score_col], errors="coerce")
    valid = y.notna() & score.notna()
    work = frame.loc[valid, ["city", "event_id", TARGET]].copy()
    work["score"] = score.loc[valid].to_numpy(dtype=float)
    return {
        "n_tokens": int(len(work)),
        "mean_spearman": mean_event_spearman(work, "score"),
        "global_spearman": safe_corr(work[TARGET], work["score"], method="spearman"),
        "mean_top_1pct_value_capture": mean_event_top_capture(work, "score", 0.01),
        "mean_top_5pct_value_capture": mean_event_top_capture(work, "score", 0.05),
        "mean_top_10pct_value_capture": mean_event_top_capture(work, "score", 0.10),
        "positive_score_share": float((work["score"] > 0).mean()),
    }


def mean_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    values = []
    for _, group in frame.groupby(["city", "event_id"]):
        if group[TARGET].nunique() <= 1 or group[score_col].nunique() <= 1:
            continue
        values.append(group[TARGET].corr(group[score_col], method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def mean_event_top_capture(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    captures = []
    for _, group in frame.groupby(["city", "event_id"]):
        values = group[TARGET].to_numpy(dtype=float)
        values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        total = float(values.sum())
        if total <= EPS:
            continue
        k = max(1, int(np.ceil(len(group) * frac)))
        order = np.argsort(group[score_col].to_numpy(dtype=float))[::-1][:k]
        captures.append(float(values[order].sum() / total))
    return float(np.mean(captures)) if captures else np.nan


def safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    try:
        return float(a.corr(b, method=method))
    except Exception:
        return float("nan")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else float("nan")
    except Exception:
        return float("nan")


def pareto_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = metrics.sort_values(["complexity", "mean_top_5pct_value_capture"], ascending=[True, False])
    best = -np.inf
    for _, row in ordered.iterrows():
        value = float(row["mean_top_5pct_value_capture"])
        if value > best + 1e-9:
            rows.append(row)
            best = value
    return pd.DataFrame(rows)


def event_level_formula_metrics(tokens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best = "activated_bottleneck_score"
    for (city, event_id), group in tokens.groupby(["city", "event_id"]):
        values = group[TARGET].to_numpy(dtype=float)
        score = group[best].to_numpy(dtype=float)
        total = np.maximum(values.clip(min=0).sum(), EPS)
        k = max(1, int(np.ceil(len(group) * 0.05)))
        top = np.argsort(score)[::-1][:k]
        rows.append(
            {
                "city": city,
                "event_id": int(event_id),
                "n_tokens": int(len(group)),
                "positive_value_share": float((values > 0).mean()),
                "activated_law_top_5pct_value_capture": float(values[top].clip(min=0).sum() / total),
                "activated_law_spearman": group[TARGET].corr(group[best], method="spearman")
                if group[TARGET].nunique() > 1 and group[best].nunique() > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    combined: pd.DataFrame,
    pareto: pd.DataFrame,
    coefficients: pd.DataFrame,
    group_ablation: pd.DataFrame,
    event_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    make_complexity_capture_figure(combined, pareto, figure_dir / "formula_complexity_capture.png")
    make_formula_spearman_figure(combined, figure_dir / "formula_spearman_ranking.png")
    make_minimal_coefficients_figure(coefficients, figure_dir / "minimal_log_law_coefficients.png")
    make_feature_group_ablation_figure(group_ablation, figure_dir / "feature_group_ablation.png")
    make_event_capture_histogram(event_metrics, figure_dir / "activated_law_event_capture_histogram.png")


def make_complexity_capture_figure(combined: pd.DataFrame, pareto: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    color_map = {"direct_symbolic": "#64748b", "leave_city_log_linear": "#2563eb"}
    for model_type, group in combined.groupby("model_type"):
        ax.scatter(
            group["complexity"],
            group["mean_top_5pct_value_capture"],
            s=74,
            alpha=0.78,
            label=model_type.replace("_", " "),
            color=color_map.get(model_type, "#7c3aed"),
        )
    if not pareto.empty:
        ordered = pareto.sort_values("complexity")
        ax.plot(ordered["complexity"], ordered["mean_top_5pct_value_capture"], color="#ef4444", linewidth=2.0, label="Pareto frontier")
    for _, row in combined.sort_values("mean_top_5pct_value_capture", ascending=False).head(5).iterrows():
        ax.annotate(row["formula_id"], (row["complexity"], row["mean_top_5pct_value_capture"]), fontsize=8)
    ax.set_xlabel("Formula complexity")
    ax.set_ylabel("Mean top-5% value capture")
    ax.set_title("Formula extractor: accuracy versus simplicity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_formula_spearman_figure(combined: pd.DataFrame, path: Path) -> None:
    ordered = combined.sort_values("mean_spearman", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    colors = np.where(ordered["model_type"].eq("direct_symbolic"), "#64748b", "#2563eb")
    ax.barh(ordered["formula_id"], ordered["mean_spearman"], color=colors)
    ax.set_xlabel("Mean within-event Spearman")
    ax.set_title("Formula ranking agreement with marginal recovery value")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_minimal_coefficients_figure(coefficients: pd.DataFrame, path: Path) -> None:
    subset = coefficients[coefficients["formula_id"].eq("R7_minimal_log_law") & ~coefficients["feature"].eq("intercept")].copy()
    if subset.empty:
        return
    subset = subset.sort_values("mean_coefficient")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(subset["feature"], subset["mean_coefficient"], color="#2563eb", alpha=0.84)
    ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Mean standardized coefficient")
    ax.set_title("Minimal log-law coefficients across leave-city splits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_feature_group_ablation_figure(group_ablation: pd.DataFrame, path: Path) -> None:
    if group_ablation.empty:
        return
    subset = group_ablation[group_ablation["mode"].eq("leave_one_group_out")].copy()
    if subset.empty:
        return
    subset = subset.sort_values("top5_capture_drop_vs_full")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.barh(subset["removed_group"], subset["top5_capture_drop_vs_full"], color="#0f766e", alpha=0.84)
    ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Top-5% capture drop after removing group")
    ax.set_title("Structure decoupler: feature-group ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_event_capture_histogram(event_metrics: pd.DataFrame, path: Path) -> None:
    if event_metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(event_metrics["activated_law_top_5pct_value_capture"], bins=12, color="#2563eb", edgecolor="white", alpha=0.82)
    ax.set_xlabel("Activated law top-5% value capture by event")
    ax.set_ylabel("Event count")
    ax.set_title("Event-level stability of activated symbolic law")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    tokens: pd.DataFrame,
    combined: pd.DataFrame,
    pareto: pd.DataFrame,
    coefficients: pd.DataFrame,
    leave_city_metrics: pd.DataFrame,
    group_ablation: pd.DataFrame,
    event_metrics: pd.DataFrame,
) -> None:
    best = combined.sort_values("mean_top_5pct_value_capture", ascending=False).iloc[0]
    minimal = combined[combined["formula_id"].eq("R7_minimal_log_law")].iloc[0]
    activated = combined[combined["formula_id"].eq("F7_activated_recovery_law")].iloc[0]
    exposure = combined[combined["formula_id"].eq("F2_exposure")].iloc[0]
    ablation_rows = group_ablation[group_ablation["mode"].eq("leave_one_group_out")].copy()
    most_important_group = (
        ablation_rows.sort_values("top5_capture_drop_vs_full", ascending=False).iloc[0]
        if not ablation_rows.empty
        else pd.Series(dtype=float)
    )
    lines = [
        "# Symbolic Law Extraction V12",
        "",
        "## 这一版回答什么问题",
        "",
        "V12 把 high-level idea 中的 Formula Extractor 显式化：构造一组候选符号公式和 leave-one-city log-linear 公式，比较它们的复杂度、跨事件排序能力、top-tail value capture，并形成 accuracy-simplicity 的 Pareto frontier。它还加入一个轻量 structure decoupler：按 feature group 做 single-group 与 leave-one-group-out ablation，检查 action recovery value 主要由哪些低维因素支撑。",
        "",
        "## 数据规模",
        "",
        f"- sampled action tokens: {len(tokens):,}",
        f"- city-event scenarios: {tokens[['city', 'event_id']].drop_duplicates().shape[0]}",
        "",
        "## 关键结果",
        "",
        f"- best formula by top-5% capture: {best.formula_id}, capture = {best.mean_top_5pct_value_capture:.4f}, Spearman = {best.mean_spearman:.4f}",
        f"- direct activated recovery law: top-5% capture = {activated.mean_top_5pct_value_capture:.4f}, Spearman = {activated.mean_spearman:.4f}",
        f"- minimal log-law extractor: top-5% capture = {minimal.mean_top_5pct_value_capture:.4f}, Spearman = {minimal.mean_spearman:.4f}",
        f"- exposure-only symbolic term: top-5% capture = {exposure.mean_top_5pct_value_capture:.4f}, Spearman = {exposure.mean_spearman:.4f}",
        f"- largest leave-group-out drop: {most_important_group.get('removed_group', '')}, top-5% capture drop = {safe_float(most_important_group.get('top5_capture_drop_vs_full')):.4f}",
        "",
        "## Formula Metrics",
        "",
        table_to_markdown(combined),
        "",
        "## Leave-City Metrics",
        "",
        table_to_markdown(leave_city_metrics),
        "",
        "## Feature-Group Ablation",
        "",
        table_to_markdown(group_ablation),
        "",
        "## Pareto Frontier",
        "",
        table_to_markdown(pareto),
        "",
        "## Minimal Log-Law Coefficients",
        "",
        table_to_markdown(coefficients[coefficients["formula_id"].eq("R7_minimal_log_law")]),
        "",
        "## Event-Level Activated-Law Metrics",
        "",
        table_to_markdown(event_metrics.describe().reset_index()),
        "",
        "## 科学解释",
        "",
        "Formula extractor 支持一个简洁结论：action-level recoverability 主要来自 delay feasibility、active future horizon、OD exposure、intervention efficiency per cost 的共同激活。单独变量可以解释一部分排序，其中 OD exposure 是最强单项；但完整 activated law 在 Pareto frontier 上仍然提供最高 top-tail capture。log-linear extractor 在 leave-one-city 设置下接近这个显式 law，说明该关系不是单个城市身份记忆。",
        "",
        "Ablation 的作用是把公式里的变量进一步解耦。若移除某一组变量导致 top-tail capture 明显下降，说明它不是装饰性参数，而是 recoverability value field 的必要结构来源。当前结果尤其提醒我们：这套 law 不应被表述为“降雨越强越值得恢复”或“预算越高越好”这类模型参数规律，而应表述为“未来仍可作用的损失、OD 暴露和资源效率在同一 action 上重合时，恢复价值才被激活”。",
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

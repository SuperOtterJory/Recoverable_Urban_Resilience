"""Build action-token learning data and first-pass recoverability laws."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root


INTERVENTIONS = ("R", "C", "S")
RNG_SEED = 20260529


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/law_learning")
    parser.add_argument("--top-actions-per-event", type=int, default=700)
    parser.add_argument("--random-actions-per-event", type=int, default=500)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
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
    action_tokens, concentration = build_action_token_dataset(
        root,
        config,
        data,
        top_actions_per_event=args.top_actions_per_event,
        random_actions_per_event=args.random_actions_per_event,
    )
    action_tokens = add_learning_features(action_tokens)
    loco_metrics, predictions = run_leave_city_out_models(action_tokens, ridge_alpha=args.ridge_alpha)
    regime_metrics = run_leave_regime_out_models(action_tokens, ridge_alpha=args.ridge_alpha)
    policy_capture = evaluate_policy_scores(action_tokens)
    event_law = build_event_level_law_table(data["summary"], concentration)

    legacy_action_tokens = table_dir / "action_value_tokens.csv"
    if legacy_action_tokens.exists():
        legacy_action_tokens.unlink()
    write_table(action_tokens, table_dir / "action_value_tokens.csv.gz")
    write_table(concentration, table_dir / "event_value_concentration.csv")
    write_table(loco_metrics, table_dir / "leave_city_out_metrics.csv")
    write_table(regime_metrics, table_dir / "leave_regime_out_metrics.csv")
    write_table(predictions, table_dir / "action_value_predictions.csv")
    write_table(policy_capture, table_dir / "policy_score_value_capture.csv")
    write_table(event_law, table_dir / "event_level_top_tail_law.csv")
    make_figures(action_tokens, loco_metrics, policy_capture, event_law, figure_dir)
    write_report(
        report_dir / "law_learning_report_zh.md",
        action_tokens,
        concentration,
        loco_metrics,
        regime_metrics,
        policy_capture,
        event_law,
    )
    print(f"Wrote law-learning outputs to {output_dir}")


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "results"
    return {
        "summary": pd.read_csv(tables / "event_optimization" / "tables" / "event_optimization_summary.csv"),
        "interventions": pd.read_csv(tables / "event_optimization" / "tables" / "event_optimization_interventions.csv"),
        "events": pd.read_csv(tables / "data_mining" / "tables" / "rainfall_event_impact_details.csv", parse_dates=["event_start", "event_end"]),
        "dynamics": pd.read_csv(tables / "event_calibration" / "tables" / "event_dynamic_calibration_summary.csv"),
        "abnormal": pd.read_csv(tables / "data_mining" / "tables" / "speed_hourly_abnormal_deficit.csv", parse_dates=["hour"]),
    }


def build_action_token_dataset(
    root: Path,
    config: dict[str, Any],
    data: dict[str, pd.DataFrame],
    *,
    top_actions_per_event: int,
    random_actions_per_event: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    summary = data["summary"].copy()
    summary = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    summary["event_id"] = pd.to_numeric(summary["event_id"], errors="coerce").astype(int)
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {
        (row.city, int(row.event_id)): row for row in events.itertuples(index=False)
    }
    dynamics = data["dynamics"]
    dynamic_lookup = {row["city"]: row for _, row in dynamics.iterrows()}
    interventions = prepare_interventions(data["interventions"])
    abnormal = data["abnormal"].copy()

    token_frames: list[pd.DataFrame] = []
    concentration_rows: list[dict[str, Any]] = []
    total_events = len(summary)
    for idx, row in enumerate(summary.sort_values(["city", "event_start", "event_id"]).itertuples(index=False), start=1):
        city = row.city
        event_id = int(row.event_id)
        print(f"[{idx}/{total_events}] Building action tokens for {city} event {event_id}", flush=True)
        event_row = event_lookup.get((city, event_id))
        if event_row is None:
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
        concentration_rows.append(event_concentration(row, full))
        keep = choose_token_sample(full, top_actions_per_event, random_actions_per_event, rng)
        token_frames.append(full.loc[keep].copy())

    tokens = pd.concat(token_frames, ignore_index=True)
    concentration = pd.DataFrame(concentration_rows).sort_values(["city", "event_start", "event_id"])
    return tokens, concentration


def prepare_interventions(interventions: pd.DataFrame) -> pd.DataFrame:
    df = interventions.copy()
    df["unit"] = df["unit"].astype(str)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype(int)
    grouped = df.groupby(["city", "event_id", "scenario", "unit", "t", "intervention"], as_index=False).agg(
        optimized_u=("u", "sum"),
        optimized_e=("e", "sum"),
        optimized_cost=("effective_cost", "sum"),
    )
    grouped["t"] = pd.to_numeric(grouped["t"], errors="coerce").astype(int)
    return grouped


def build_event_action_frame(
    params: Any,
    summary_row: Any,
    event_row: Any,
    event_interventions: pd.DataFrame,
) -> pd.DataFrame:
    n = params.n_units
    horizon = params.horizon
    q = params.q.tocsr() if sparse.issparse(params.q) else sparse.csr_matrix(params.q)
    p = np.asarray(params.p, dtype=float)
    destination_importance = np.asarray(q.T @ p).ravel()
    out_degree = np.diff(q.indptr).astype(float)
    in_degree = np.diff(q.tocsc().indptr).astype(float)
    passive_b, passive_ell = passive_no_intervention(params)
    local_remaining = remaining_area(passive_b)
    access_remaining = remaining_area(passive_ell)
    h_total = params.h.sum(axis=1)
    h_peak = params.h.max(axis=1)
    local_need = np.clip(params.b0 + h_total, 0.02, 1.0)
    value_eff = effective_output_values(params, passive_b, passive_ell, destination_importance)

    base = pd.DataFrame(
        {
            "unit": np.asarray(params.units, dtype=str),
            "origin_exposure": p,
            "destination_importance": destination_importance,
            "b0": params.b0,
            "h_total": h_total,
            "h_peak": h_peak,
            "a_retention": params.a,
            "local_need": local_need,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "origin_exposure_rank": rank_pct(p),
            "destination_importance_rank": rank_pct(destination_importance),
            "b0_rank": rank_pct(params.b0),
            "h_total_rank": rank_pct(h_total),
            "out_degree_rank": rank_pct(out_degree),
            "in_degree_rank": rank_pct(in_degree),
            "local_need_rank": rank_pct(local_need),
        }
    )
    rows: list[pd.DataFrame] = []
    for t in range(horizon):
        time_remaining_frac = (horizon - t) / horizon
        t_base = base.copy()
        t_base["t"] = t
        t_base["t_frac"] = t / max(horizon - 1, 1)
        t_base["time_remaining_frac"] = time_remaining_frac
        t_base["passive_b_t"] = passive_b[:, t]
        t_base["passive_ell_t"] = passive_ell[:, t]
        t_base["local_remaining_area"] = local_remaining[:, t]
        t_base["access_remaining_area"] = access_remaining[:, t]
        t_base["local_remaining_rank"] = rank_pct(local_remaining[:, t])
        t_base["access_remaining_rank"] = rank_pct(access_remaining[:, t])
        t_base["passive_b_rank"] = rank_pct(passive_b[:, t])
        t_base["passive_ell_rank"] = rank_pct(passive_ell[:, t])
        for intervention in INTERVENTIONS:
            k_base = t_base.copy()
            k_base["intervention"] = intervention
            k_base["intervention_R"] = 1.0 if intervention == "R" else 0.0
            k_base["intervention_C"] = 1.0 if intervention == "C" else 0.0
            k_base["intervention_S"] = 1.0 if intervention == "S" else 0.0
            feasible = t >= int(params.delays.get(intervention, 0))
            k_base["delay_feasible"] = float(feasible)
            k_base["delay_fraction"] = int(params.delays.get(intervention, 0)) / horizon
            eta = params.eta[intervention][:, t]
            cost = params.cost[intervention][:, t]
            k_base["eta"] = eta
            k_base["cost"] = cost
            k_base["eta_per_cost"] = eta / np.maximum(cost, 1e-12)
            k_base["eta_per_cost_rank"] = rank_pct(k_base["eta_per_cost"].to_numpy(dtype=float))
            k_base["u_cap"] = params.u_cap[intervention][:, t]
            eff_value = value_eff[intervention][:, t]
            k_base["marginal_effect_value"] = eff_value / max(float(summary_row.baseline_objective), 1e-12)
            k_base["marginal_resource_value"] = np.where(
                feasible,
                eff_value * eta / np.maximum(cost, 1e-12) / max(float(summary_row.baseline_objective), 1e-12),
                0.0,
            )
            k_base["law_future_deficit_area"] = np.where(
                intervention == "S",
                k_base["access_remaining_area"],
                k_base["local_remaining_area"],
            )
            k_base["law_exposure_term"] = np.where(
                intervention == "S",
                k_base["origin_exposure"],
                k_base["destination_importance"],
            )
            k_base["activated_bottleneck_score"] = activated_bottleneck_score(k_base, intervention)
            k_base["deficit_only_score"] = np.where(
                intervention == "S",
                k_base["access_remaining_rank"],
                k_base["local_remaining_rank"],
            )
            k_base["exposure_only_score"] = np.where(
                intervention == "S",
                k_base["origin_exposure_rank"],
                k_base["destination_importance_rank"],
            )
            k_base["structure_only_score"] = np.where(
                intervention == "S",
                k_base["origin_exposure_rank"] * (1.0 - k_base["out_degree_rank"]),
                k_base["destination_importance_rank"] * (1.0 - k_base["out_degree_rank"]),
            )
            rows.append(k_base)
    full = pd.concat(rows, ignore_index=True)
    full.insert(0, "city", params.city)
    full.insert(1, "event_id", int(summary_row.event_id))
    full.insert(2, "event_start", str(summary_row.event_start))
    full["scenario"] = "base"
    full["event_total_precip"] = float(summary_row.event_total_precip)
    full["event_peak_precip"] = float(summary_row.event_peak_precip)
    full["event_peak_positive_abnormal_deficit"] = float(summary_row.event_peak_positive_abnormal_deficit)
    full["weighted_b0"] = float(summary_row.weighted_b0)
    full["weighted_h_total"] = float(summary_row.weighted_h_total)
    full["baseline_objective"] = float(summary_row.baseline_objective)
    full["recoverable_fraction"] = float(summary_row.recoverable_fraction)
    full["total_budget"] = float(summary_row.total_budget)
    full["budget_fraction_of_baseline"] = float(summary_row.total_budget) / max(float(summary_row.baseline_objective), 1e-12)
    full = full.merge(event_interventions, on=["city", "event_id", "scenario", "unit", "t", "intervention"], how="left")
    for col in ["optimized_u", "optimized_e", "optimized_cost"]:
        full[col] = full[col].fillna(0.0)
    full["selected_by_optimizer"] = full["optimized_u"] > 1e-12
    full["optimized_value_proxy"] = full["optimized_e"] * full["marginal_effect_value"]
    full["rain_regime"] = event_regime(float(summary_row.event_peak_positive_abnormal_deficit))
    return full


def passive_no_intervention(params: Any) -> tuple[np.ndarray, np.ndarray]:
    n = params.n_units
    horizon = params.horizon
    b = np.zeros((n, horizon + 1), dtype=float)
    ell = np.zeros((n, horizon + 1), dtype=float)
    b[:, 0] = params.b0
    for t in range(horizon + 1):
        ell[:, t] = np.clip(params.q @ b[:, t], 0.0, 1.0)
        if t == horizon:
            break
        b[:, t + 1] = np.clip(params.a * b[:, t] + params.h[:, t + 1], 0.0, 1.0)
    return b, ell


def remaining_area(values: np.ndarray) -> np.ndarray:
    n, total_steps = values.shape
    horizon = total_steps - 1
    out = np.zeros((n, horizon), dtype=float)
    for t in range(horizon):
        out[:, t] = values[:, t + 1 :].sum(axis=1)
    return out


def effective_output_values(
    params: Any,
    passive_b: np.ndarray,
    passive_ell: np.ndarray,
    destination_importance: np.ndarray,
) -> dict[str, np.ndarray]:
    n = params.n_units
    horizon = params.horizon
    values = {key: np.zeros((n, horizon), dtype=float) for key in INTERVENTIONS}
    for t in range(horizon):
        steps = horizon - t
        k = np.arange(steps, dtype=float)
        r_decay = params.a[:, None] ** k[None, :]
        c_decay = (1.0 - params.delta_c) ** k
        s_decay = (1.0 - params.delta_s) ** k
        future_b = passive_b[:, t + 1 :]
        future_ell = passive_ell[:, t + 1 :]
        values["R"][:, t] = destination_importance * np.minimum(r_decay, future_b).sum(axis=1)
        values["C"][:, t] = destination_importance * np.minimum(c_decay[None, :], future_b).sum(axis=1)
        values["S"][:, t] = params.p * np.minimum(s_decay[None, :], future_ell).sum(axis=1)
    return values


def activated_bottleneck_score(frame: pd.DataFrame, intervention: str) -> np.ndarray:
    future_deficit_area = frame["law_future_deficit_area"].to_numpy(dtype=float)
    exposure = frame["law_exposure_term"].to_numpy(dtype=float)
    eta_per_cost = frame["eta_per_cost"].to_numpy(dtype=float)
    feasible = frame["delay_feasible"].to_numpy(dtype=float)
    return feasible * future_deficit_area * exposure * eta_per_cost


def choose_token_sample(
    full: pd.DataFrame,
    top_actions_per_event: int,
    random_actions_per_event: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = full.index[full["selected_by_optimizer"].to_numpy(dtype=bool)].to_numpy()
    top = full["marginal_resource_value"].nlargest(min(top_actions_per_event, len(full))).index.to_numpy()
    remaining = np.setdiff1d(full.index.to_numpy(), np.union1d(selected, top), assume_unique=False)
    random_n = min(random_actions_per_event, len(remaining))
    random = rng.choice(remaining, size=random_n, replace=False) if random_n > 0 else np.array([], dtype=int)
    return np.unique(np.concatenate([selected, top, random]))


def event_concentration(summary_row: Any, full: pd.DataFrame) -> dict[str, Any]:
    values = full["marginal_resource_value"].to_numpy(dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = float(values.sum())
    row = {
        "city": summary_row.city,
        "event_id": int(summary_row.event_id),
        "event_start": str(summary_row.event_start),
        "candidate_action_count": int(len(values)),
        "positive_action_count": int((values > 0).sum()),
        "total_marginal_value_proxy": total,
        "marginal_value_gini": gini(values),
        "top_1pct_value_share": top_share(values, 0.01),
        "top_5pct_value_share": top_share(values, 0.05),
        "top_10pct_value_share": top_share(values, 0.10),
        "recoverable_fraction": float(summary_row.recoverable_fraction),
        "baseline_objective": float(summary_row.baseline_objective),
        "event_peak_positive_abnormal_deficit": float(summary_row.event_peak_positive_abnormal_deficit),
        "event_total_precip": float(summary_row.event_total_precip),
    }
    for intervention in INTERVENTIONS:
        mask = full["intervention"].to_numpy() == intervention
        row[f"top_5pct_value_share_{intervention}"] = top_share(values[mask], 0.05)
        row[f"total_value_share_{intervention}"] = float(values[mask].sum() / total) if total > 0 else np.nan
    selected = full["selected_by_optimizer"].to_numpy(dtype=bool)
    row["optimizer_selected_value_share"] = float(values[selected].sum() / total) if total > 0 else np.nan
    row["optimizer_selected_action_share"] = float(selected.mean())
    return row


def add_learning_features(tokens: pd.DataFrame) -> pd.DataFrame:
    df = tokens.copy()
    df["log_event_total_precip"] = np.log1p(df["event_total_precip"])
    df["log_event_peak_precip"] = np.log1p(df["event_peak_precip"])
    df["od_scarcity"] = np.clip(1.0 - df["out_degree_rank"], 0.0, 1.0)
    df["deficit_x_exposure"] = df["local_remaining_rank"] * df["destination_importance_rank"]
    df["access_x_origin"] = df["access_remaining_rank"] * df["origin_exposure_rank"]
    df["deficit_x_exposure_x_scarcity"] = df["deficit_x_exposure"] * (0.1 + df["od_scarcity"])
    df["law_score_log"] = np.log1p(1000.0 * df["activated_bottleneck_score"])
    df["target_log"] = np.log1p(1000.0 * df["marginal_resource_value"])
    return df


MODEL_FEATURES = [
    "t_frac",
    "time_remaining_frac",
    "delay_fraction",
    "delay_feasible",
    "intervention_R",
    "intervention_C",
    "intervention_S",
    "origin_exposure_rank",
    "destination_importance_rank",
    "b0_rank",
    "h_total_rank",
    "local_remaining_rank",
    "access_remaining_rank",
    "passive_b_rank",
    "passive_ell_rank",
    "out_degree_rank",
    "in_degree_rank",
    "od_scarcity",
    "local_need_rank",
    "eta_per_cost_rank",
    "a_retention",
    "budget_fraction_of_baseline",
    "log_event_total_precip",
    "log_event_peak_precip",
    "event_peak_positive_abnormal_deficit",
    "weighted_b0",
    "weighted_h_total",
    "deficit_x_exposure",
    "access_x_origin",
    "deficit_x_exposure_x_scarcity",
    "activated_bottleneck_score",
    "law_score_log",
]


def run_leave_city_out_models(tokens: pd.DataFrame, *, ridge_alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for city in sorted(tokens["city"].unique()):
        train = tokens[tokens["city"] != city].copy()
        test = tokens[tokens["city"] == city].copy()
        model = fit_ridge(train[MODEL_FEATURES], train["target_log"], alpha=ridge_alpha)
        pred_log = predict_ridge(model, test[MODEL_FEATURES])
        test["predicted_value_surrogate"] = np.expm1(pred_log) / 1000.0
        prediction_frames.append(
            test[
                [
                    "city",
                    "event_id",
                    "unit",
                    "t",
                    "intervention",
                    "marginal_resource_value",
                    "activated_bottleneck_score",
                    "predicted_value_surrogate",
                    "selected_by_optimizer",
                ]
            ].copy()
        )
        metric_rows.append({"split": "leave_city_out", "heldout": city, **prediction_metrics(test, "predicted_value_surrogate")})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return pd.DataFrame(metric_rows), predictions


def run_leave_regime_out_models(tokens: pd.DataFrame, *, ridge_alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in ["low", "medium", "high"]:
        train = tokens[tokens["rain_regime"] != regime].copy()
        test = tokens[tokens["rain_regime"] == regime].copy()
        if train.empty or test.empty:
            continue
        model = fit_ridge(train[MODEL_FEATURES], train["target_log"], alpha=ridge_alpha)
        pred_log = predict_ridge(model, test[MODEL_FEATURES])
        test["predicted_value_surrogate"] = np.expm1(pred_log) / 1000.0
        rows.append({"split": "leave_regime_out", "heldout": regime, **prediction_metrics(test, "predicted_value_surrogate")})
    return pd.DataFrame(rows)


def fit_ridge(x: pd.DataFrame, y: pd.Series, *, alpha: float) -> dict[str, np.ndarray]:
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


def predict_ridge(model: dict[str, np.ndarray], x: pd.DataFrame) -> np.ndarray:
    x_arr = x.to_numpy(dtype=float)
    x_std = np.nan_to_num((x_arr - model["mean"]) / model["std"], nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std])
    return np.maximum(design @ model["coef"], 0.0)


def prediction_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    y = frame["marginal_resource_value"].to_numpy(dtype=float)
    pred = frame[score_col].to_numpy(dtype=float)
    out = {
        "n_tokens": int(len(frame)),
        "n_events": int(frame[["city", "event_id"]].drop_duplicates().shape[0]),
        "pearson": safe_corr(y, pred),
        "spearman": frame["marginal_resource_value"].corr(frame[score_col], method="spearman"),
        "mae": float(np.mean(np.abs(y - pred))),
    }
    for frac in [0.01, 0.05, 0.10]:
        out[f"top_{int(frac*100)}pct_value_capture"] = mean_event_top_capture(frame, score_col, frac)
    return out


def evaluate_policy_scores(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_cols = {
        "surrogate_placeholder": None,
        "activated_bottleneck_law": "activated_bottleneck_score",
        "deficit_only": "deficit_only_score",
        "exposure_only": "exposure_only_score",
        "structure_only": "structure_only_score",
        "optimizer_selected": "selected_by_optimizer",
    }
    for name, score_col in score_cols.items():
        if score_col is None:
            continue
        metrics = {"policy_score": name, "n_tokens": int(len(tokens))}
        for frac in [0.01, 0.05, 0.10]:
            metrics[f"top_{int(frac*100)}pct_value_capture"] = mean_event_top_capture(tokens, score_col, frac)
        metrics["mean_spearman_by_event"] = mean_event_spearman(tokens, score_col)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("top_5pct_value_capture", ascending=False)


def build_event_level_law_table(summary: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    opt = summary[(summary["status"] == "OPTIMAL") & (summary["scenario"] == "base")].copy()
    opt["event_id"] = pd.to_numeric(opt["event_id"], errors="coerce").astype(int)
    merged = opt.merge(concentration, on=["city", "event_id", "event_start"], how="left", suffixes=("", "_conc"))
    merged["loss_magnitude_rank"] = rank_pct(merged["baseline_objective"].to_numpy(dtype=float))
    merged["recoverable_rank"] = rank_pct(merged["recoverable_fraction"].to_numpy(dtype=float))
    merged["top_tail_rank"] = rank_pct(merged["top_5pct_value_share"].to_numpy(dtype=float))
    merged["decision_criticality_score"] = (
        merged["recoverable_fraction"] * merged["top_5pct_value_share"] * merged["marginal_value_gini"]
    )
    merged["decision_criticality_rank"] = rank_pct(merged["decision_criticality_score"].to_numpy(dtype=float))
    cols = [
        "city",
        "event_id",
        "event_start",
        "baseline_objective",
        "recoverable_fraction",
        "top_1pct_value_share",
        "top_5pct_value_share",
        "top_10pct_value_share",
        "marginal_value_gini",
        "optimizer_selected_value_share",
        "loss_magnitude_rank",
        "recoverable_rank",
        "top_tail_rank",
        "decision_criticality_score",
        "decision_criticality_rank",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    return merged[cols].sort_values("decision_criticality_score", ascending=False)


def mean_event_top_capture(frame: pd.DataFrame, score_col: str, frac: float) -> float:
    captures = []
    for _, group in frame.groupby(["city", "event_id"]):
        values = group["marginal_resource_value"].to_numpy(dtype=float)
        scores = group[score_col].astype(float).to_numpy()
        if values.sum() <= 0 or len(group) == 0:
            continue
        k = max(1, int(np.ceil(len(group) * frac)))
        best = np.sort(values)[-k:].sum()
        chosen = values[np.argsort(scores)[-k:]].sum()
        captures.append(float(chosen / best)) if best > 0 else None
    return float(np.mean(captures)) if captures else np.nan


def mean_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    values = []
    for _, group in frame.groupby(["city", "event_id"]):
        if group["marginal_resource_value"].nunique() < 2 or group[score_col].nunique() < 2:
            continue
        values.append(group["marginal_resource_value"].corr(group[score_col], method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def make_figures(
    tokens: pd.DataFrame,
    loco_metrics: pd.DataFrame,
    policy_capture: pd.DataFrame,
    event_law: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ordered = policy_capture.sort_values("top_5pct_value_capture", ascending=True)
    ax.barh(ordered["policy_score"], ordered["top_5pct_value_capture"], color="#2563eb")
    ax.set_xlabel("Mean top-5% value capture")
    ax.set_title("Law score versus simple action-ranking baselines")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_score_value_capture.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(tokens["activated_bottleneck_score"], tokens["marginal_resource_value"], s=7, alpha=0.25, color="#0f766e")
    ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel("Activated-bottleneck law score")
    ax.set_ylabel("Marginal resource value proxy")
    ax.set_title("Local law score and action value field")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "activated_law_vs_action_value.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(event_law["baseline_objective"], event_law["top_5pct_value_share"], c=event_law["recoverable_fraction"], cmap="viridis", s=70)
    ax.set_xlabel("Observed/passive loss proxy")
    ax.set_ylabel("Top-5% recovery-value concentration")
    ax.set_title("Event-level top-tail decision criticality")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "event_top_tail_law.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(loco_metrics["heldout"], loco_metrics["top_5pct_value_capture"], color="#7c3aed")
    ax.set_ylabel("Top-5% value capture")
    ax.set_title("Leave-one-city-out surrogate performance")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "leave_city_out_capture.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    tokens: pd.DataFrame,
    concentration: pd.DataFrame,
    loco_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    policy_capture: pd.DataFrame,
    event_law: pd.DataFrame,
) -> None:
    loco_top5_mean = float(loco_metrics["top_5pct_value_capture"].mean()) if not loco_metrics.empty else np.nan
    loco_spearman_mean = float(loco_metrics["spearman"].mean()) if not loco_metrics.empty else np.nan
    law_row = policy_capture[policy_capture["policy_score"] == "activated_bottleneck_law"]
    law_top5 = float(law_row["top_5pct_value_capture"].iloc[0]) if not law_row.empty else np.nan
    law_spearman = float(law_row["mean_spearman_by_event"].iloc[0]) if not law_row.empty else np.nan
    event_top5_mean = float(concentration["top_5pct_value_share"].mean()) if not concentration.empty else np.nan
    event_gini_mean = float(concentration["marginal_value_gini"].mean()) if not concentration.empty else np.nan
    lines = [
        "# Learning and Law Discovery V1",
        "",
        "## 本版本做了什么",
        "",
        "这一版把 event-level optimization outputs 转换成 action-token 学习问题。每个 token 表示 `city-event-unit-time-intervention`。目标不是直接学习 optimizer 是否选择该 token，而是构造一个可解释的 marginal recovery-value proxy：单位资源投到该 token 后，沿着无干预的被动恢复轨迹，估计它能减少多少未来加权功能损失。",
        "",
        "这仍然是 V1：action value 是解析近似标签，不是逐 token 重新求解 single-action LP，也不是 perturbed-optimum stability。它的作用是建立 learning/law pipeline 的可复现骨架，并检验一个可解释 law 是否能跨城市排序高价值 action。",
        "",
        "## 数据规模",
        "",
        f"- sampled action tokens: {len(tokens):,}",
        f"- city-event scenarios: {tokens[['city', 'event_id']].drop_duplicates().shape[0]}",
        f"- full candidate-action concentration rows: {len(concentration)}",
        f"- mean event top-5% value share: {event_top5_mean:.4f}",
        f"- mean event marginal-value gini: {event_gini_mean:.4f}",
        "",
        "## Action Value Label",
        "",
        "每个 action 的 label 不是 optimizer 的 0/1 选择，而是 marginal resource value。直观上，如果一个 action 作用于未来仍会持续存在的损失、又覆盖高 OD 暴露区域、且单位成本效率高，那么它的恢复价值就高。",
        "",
        "对 `R` 和 `C` 类 action，价值主要来自某个 region 本地 deficit 被降低后，通过 OD dependence `Q` 减少其他 origins 的 accessibility loss；因此 exposure 使用 destination importance。对 `S` 类 action，价值直接作用在 origin 的 experienced loss 上；因此 exposure 使用 origin exposure。",
        "",
        "```text",
        "marginal_resource_value(i,t,k)",
        "  = future_effect_value(i,t,k)",
        "    * eta(i,t,k) / cost(i,t,k)",
        "    / passive_event_loss",
        "```",
        "",
        "## Interpretable Law Score",
        "",
        "V1 的可解释 law score 保留 action label 的核心结构，但不直接使用 optimizer 选择结果：",
        "",
        "```text",
        "activated_bottleneck_score(i,t,k)",
        "  ≈ delay_feasible(i,t,k)",
        "    × future_deficit_area(i,t,k)",
        "    × OD_exposure_or_destination_importance(i,k)",
        "    × intervention_efficiency_per_cost(i,t,k)",
        "```",
        "",
        "这里的 `future_deficit_area` 已经包含剩余时间窗口，因此不再额外乘一个简单的 time rank。早期草稿里我把 `1 - out_degree_rank` 当作 substitutability scarcity 强行乘进去，结果明显拉低排序表现；这一版先移除这个不稳定项，把 substitutability 留到后续用更可靠的替代路径或网络冗余指标刻画。",
        "",
        "## 关键结果概览",
        "",
        f"- Leave-one-city-out mean Spearman: {loco_spearman_mean:.4f}",
        f"- Leave-one-city-out mean top-5% value capture: {loco_top5_mean:.4f}",
        f"- Activated-bottleneck law top-5% value capture: {law_top5:.4f}",
        f"- Activated-bottleneck law mean event Spearman: {law_spearman:.4f}",
        "",
        "## Leave-One-City-Out Surrogate",
        "",
        dataframe_to_markdown(loco_metrics),
        "",
        "## Leave-Regime-Out Surrogate",
        "",
        dataframe_to_markdown(regime_metrics),
        "",
        "## Law Score 与 Baselines",
        "",
        dataframe_to_markdown(policy_capture),
        "",
        "解释：`optimizer_selected` 在 action-value 排序里不一定最高，因为 optimizer 选择受到总预算、单期预算、部署上限、分段边际收益和替代 action 的共同约束；而 law score 评价的是“单个 action 的边际价值排序”。因此这里更应该看 law score 是否能捕捉 value field 的 top tail，而不是是否复刻 optimizer 的最终稀疏解。",
        "",
        "## Event-Level Top-Tail Law",
        "",
        dataframe_to_markdown(event_law.head(15)),
        "",
        "## 当前可读出的初步 law",
        "",
        "Local activated-bottleneck law 的第一版可写成：",
        "",
        "```text",
        "recovery_value(action)",
        "  ≈ persistent_future_deficit_area",
        "    × exposed_OD_importance_or_origin_exposure",
        "    × intervention_efficiency_per_cost",
        "    × response_feasibility",
        "```",
        "",
        "Event top-tail law 的第一版可写成：",
        "",
        "```text",
        "decision_criticality(event)",
        "  ≈ recoverable_fraction",
        "    × top_tail_concentration_of_action_values",
        "    × inequality_of_recovery_value_field",
        "```",
        "",
        "## 需要继续改进的地方",
        "",
        "1. 下一版应生成更强的 action-level oracle label：single-action marginal LP、greedy residual marginal value 或 perturbed optimum stability。",
        "2. 需要加入更多 scenario augmentation，尤其是 budget 和 delay 变化，否则 law 仍主要来自 base scenario。",
        "3. 当前 surrogate 是 ridge baseline，不是最终神经模型；后续可升级为 factorized action-value scorer 或 graph surrogate。",
        "4. 当前 substitutability 没有被可靠刻画。简单 out-degree scarcity 在本数据中会损害排序，后续应加入替代路径、网络冗余或 OD rerouting proxy。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), 0.5, dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def top_share(values: np.ndarray, pct: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0 or values.sum() <= 0:
        return np.nan
    n = max(1, int(np.ceil(len(values) * pct)))
    return float(np.sort(values)[-n:].sum() / values.sum())


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    values = np.sort(np.maximum(values, 0.0))
    total = values.sum()
    if total <= 0:
        return np.nan
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * total) - (n + 1) / n)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def event_regime(peak_positive_abnormal_deficit: float) -> str:
    if peak_positive_abnormal_deficit < 0.015:
        return "low"
    if peak_positive_abnormal_deficit < 0.035:
        return "medium"
    return "high"


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

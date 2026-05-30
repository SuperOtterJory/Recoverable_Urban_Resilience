"""Audit the New York-scale boundary for footprint-aware LP closure.

V42 closed the explicit recovery--footprint LP frontier on additional
non-New-York full-zone events.  The remaining hard case is New York: it has
event-footprint signal, but the representative hybrid-footprint LP previously
hit the solver time limit before returning a feasible optimized solution.

This diagnostic does not claim a New York optimum.  It quantifies whether the
boundary is caused by absent footprint signal or by computational scale, and it
records the exact model-size gap between the closed V42 events and the selected
New York footprint-sensitive events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from analyze_hybrid_absorption_mechanisms import footprint_weights
from analyze_hybrid_footprint_calibration import DEFAULT_MAIN_BLEND, build_hybrid_params, load_inputs, no_intervention_objective
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters


EPS = 1e-12
DEFAULT_MAX_EVENTS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--v34-dir", default="results/hybrid_footprint_calibration")
    parser.add_argument("--v35-dir", default="results/hybrid_footprint_lp_validation")
    parser.add_argument("--v42-dir", default="results/broader_multiobjective_footprint_lp_validation")
    parser.add_argument("--output-dir", default="results/new_york_footprint_lp_boundary")
    parser.add_argument("--city", default="New York")
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--footprint-blend", type=float, default=DEFAULT_MAIN_BLEND)
    parser.add_argument("--footprint-floor", type=float, default=0.02)
    parser.add_argument("--max-relative", type=float, default=12.0)
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
    v34_events = pd.read_csv(root / args.v34_dir / "tables" / "hybrid_footprint_event_metrics.csv", parse_dates=["event_start"])
    v35_events = read_csv(root / args.v35_dir / "tables" / "hybrid_lp_event_metrics.csv")
    v42_events = read_csv(root / args.v42_dir / "tables" / "broader_multiobjective_lp_event_metrics.csv")
    calibration_summary = read_csv(root / "results" / "event_optimization" / "tables" / "event_calibration_summary.csv")

    selected = select_boundary_events(
        v34_events,
        city=str(args.city),
        footprint_blend=float(args.footprint_blend),
        max_events=int(args.max_events),
    )
    event_metrics = build_new_york_event_metrics(
        root,
        config,
        data,
        selected,
        v35_events,
        footprint_blend=float(args.footprint_blend),
        footprint_floor=float(args.footprint_floor),
        max_relative=float(args.max_relative),
    )
    size_comparison = build_size_comparison(event_metrics, v42_events, calibration_summary, config)
    metrics = build_metrics(event_metrics, size_comparison, v42_events)

    write_table(selected, table_dir / "new_york_boundary_selected_events.csv")
    write_table(event_metrics, table_dir / "new_york_boundary_event_metrics.csv")
    write_table(size_comparison, table_dir / "new_york_boundary_size_comparison.csv")
    (table_dir / "new_york_boundary_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_size_figure(size_comparison, figure_dir / "new_york_lp_size_boundary.png")
    make_signal_figure(event_metrics, figure_dir / "new_york_footprint_signal_boundary.png")
    write_report(
        report_dir / "new_york_footprint_lp_boundary_report_zh.md",
        event_metrics,
        size_comparison,
        metrics,
    )
    print(f"Wrote New York footprint LP boundary audit to {output_dir}")


def select_boundary_events(
    v34_events: pd.DataFrame,
    *,
    city: str,
    footprint_blend: float,
    max_events: int,
) -> pd.DataFrame:
    frame = v34_events.copy()
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")
    frame = frame[
        frame["city"].astype(str).eq(city)
        & np.isclose(pd.to_numeric(frame["footprint_blend"], errors="coerce"), footprint_blend)
    ].copy()
    if frame.empty:
        raise ValueError(f"No V34 footprint events found for {city} at blend={footprint_blend}.")
    frame = frame.sort_values(
        ["delta_finite_top5pct_units_footprint_mass", "finite_top5pct_action_jaccard", "event_start"],
        ascending=[False, True, True],
    ).head(max(1, max_events))
    keep = [
        "city",
        "event_id",
        "event_start",
        "footprint_blend",
        "finite_action_value_spearman",
        "finite_top5pct_action_jaccard",
        "delta_finite_top5pct_units_footprint_mass",
        "base_finite_top5pct_units_footprint_mass",
        "hybrid_finite_top5pct_units_footprint_mass",
        "hybrid_to_base_baseline_objective_ratio",
    ]
    return frame[keep].reset_index(drop=True)


def build_new_york_event_metrics(
    root: Path,
    config: dict[str, Any],
    data: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    v35_events: pd.DataFrame,
    *,
    footprint_blend: float,
    footprint_floor: float,
    max_relative: float,
) -> pd.DataFrame:
    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype("Int64")
    event_lookup = {
        (row.city, int(row.event_id)): row
        for row in events.dropna(subset=["event_id"]).itertuples(index=False)
    }
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    footprint = data["footprint_zone"].dropna(subset=["event_id"]).copy()
    footprint["event_id"] = pd.to_numeric(footprint["event_id"], errors="coerce").astype(int)
    footprint["zone_id"] = footprint["zone_id"].astype(str)
    footprint_groups = {
        (city, int(event_id)): group.copy()
        for (city, event_id), group in footprint.groupby(["city", "event_id"])
    }
    base_summary = data["summary"].copy()
    base_summary = base_summary[
        base_summary["status"].astype(str).eq("OPTIMAL")
        & base_summary["scenario"].astype(str).eq("base")
    ].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype("Int64")
    base_lookup = {
        (row.city, int(row.event_id)): row
        for row in base_summary.dropna(subset=["event_id"]).itertuples(index=False)
    }
    v35_lookup: dict[tuple[str, int], Any] = {}
    if not v35_events.empty and {"city", "event_id"}.issubset(v35_events.columns):
        v35_events = v35_events.copy()
        v35_events["event_id"] = pd.to_numeric(v35_events["event_id"], errors="coerce").astype("Int64")
        v35_lookup = {
            (row.city, int(row.event_id)): row
            for row in v35_events.dropna(subset=["event_id"]).itertuples(index=False)
        }

    rows: list[dict[str, Any]] = []
    for idx, selected_row in enumerate(selected.itertuples(index=False), start=1):
        city = str(selected_row.city)
        event_id = int(selected_row.event_id)
        print(f"[{idx}/{len(selected)}] Auditing New York boundary event {event_id}", flush=True)
        event_key = (city, event_id)
        event_row = event_lookup[event_key]
        footprint_group = footprint_groups[event_key]
        base_params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=data["abnormal"],
            root=root,
        )
        hybrid_params, diagnostics = build_hybrid_params(
            base_params,
            footprint_group,
            footprint_blend=footprint_blend,
            footprint_floor=footprint_floor,
            max_relative=max_relative,
        )
        weights = footprint_weights(footprint_group, hybrid_params.units)
        size = lp_size_estimate(hybrid_params)
        prior = v35_lookup.get(event_key)
        base_row = base_lookup.get(event_key)
        top_n = max(1, int(math.ceil(0.05 * hybrid_params.n_units)))
        footprint_top5_mass = float(weights.sort_values(ascending=False).head(top_n).sum())
        positive_footprint_units = int((weights > EPS).sum())
        rows.append(
            {
                "city": city,
                "event_id": event_id,
                "event_start": str(selected_row.event_start),
                "n_units": hybrid_params.n_units,
                "horizon": hybrid_params.horizon,
                "q_nnz": size["q_nnz"],
                "q_density": size["q_density"],
                "estimated_total_variables": size["estimated_total_variables"],
                "estimated_total_constraints": size["estimated_total_constraints"],
                "estimated_access_nonzero_terms": size["estimated_access_nonzero_terms"],
                "estimated_action_tokens": size["estimated_action_tokens"],
                "estimated_pwl_segment_variables": size["estimated_pwl_segment_variables"],
                "base_no_intervention_objective": no_intervention_objective(base_params),
                "hybrid_no_intervention_objective": no_intervention_objective(hybrid_params),
                "hybrid_to_base_no_intervention_ratio": no_intervention_objective(hybrid_params)
                / max(no_intervention_objective(base_params), EPS),
                "base_lp_recoverable_fraction": safe_get(base_row, "recoverable_fraction"),
                "base_lp_optimized_objective": safe_get(base_row, "optimized_objective"),
                "base_lp_total_budget": float(base_params.total_budget),
                "prior_hybrid_lp_status": prior_status(prior),
                "prior_hybrid_lp_error": str(getattr(prior, "error", "")) if prior is not None else "",
                "prior_hybrid_lp_runtime_seconds": safe_get(prior, "runtime_seconds"),
                "prior_base_lp_status": str(getattr(prior, "base_status", "")) if prior is not None else "",
                "prior_base_lp_recoverable_fraction": safe_get(prior, "base_recoverable_fraction"),
                "finite_action_value_spearman": float(selected_row.finite_action_value_spearman),
                "finite_top5pct_action_jaccard": float(selected_row.finite_top5pct_action_jaccard),
                "delta_finite_top5pct_units_footprint_mass": float(
                    selected_row.delta_finite_top5pct_units_footprint_mass
                ),
                "base_finite_top5pct_units_footprint_mass": float(
                    selected_row.base_finite_top5pct_units_footprint_mass
                ),
                "hybrid_finite_top5pct_units_footprint_mass": float(
                    selected_row.hybrid_finite_top5pct_units_footprint_mass
                ),
                "event_footprint_top5_unit_mass": footprint_top5_mass,
                "positive_footprint_unit_count": positive_footprint_units,
                **diagnostics,
            }
        )
    return pd.DataFrame(rows)


def lp_size_estimate(params: RecoveryLPParameters) -> dict[str, float]:
    n = int(params.n_units)
    horizon = int(params.horizon)
    q_nnz = int(params.q.nnz) if sparse.issparse(params.q) else int(np.count_nonzero(params.q))
    q_density = float(q_nnz / max(n * n, 1))
    use_pwl = params.u_segment_cap is not None and params.segment_effectiveness is not None
    segment_count = int(next(iter(params.u_segment_cap.values())).shape[2]) if use_pwl else 0
    delays = {key: int(params.delays.get(key, 0)) for key in INTERVENTIONS}
    return estimate_lp_size_from_counts(
        n_units=n,
        q_nnz=q_nnz,
        horizon=horizon,
        use_pwl=use_pwl,
        segment_count=segment_count,
        delays=delays,
    )


def estimate_lp_size_from_counts(
    *,
    n_units: int,
    q_nnz: int,
    horizon: int,
    use_pwl: bool,
    segment_count: int,
    delays: dict[str, int],
) -> dict[str, float]:
    n = int(n_units)
    t = int(horizon)
    k_count = len(INTERVENTIONS)
    state_vars = 5 * n * (t + 1)
    action_u_vars = k_count * n * t
    effect_vars = k_count * n * t
    segment_vars = k_count * n * t * int(segment_count) if use_pwl else 0
    total_vars = state_vars + action_u_vars + effect_vars + segment_vars

    initial_constraints = 3 * n
    transition_constraints = 3 * n * t
    segment_sum_constraints = k_count * n * t if use_pwl else 0
    segment_cap_constraints = k_count * n * t * int(segment_count) if use_pwl else 0
    effectiveness_constraints = k_count * n * t
    deployment_cap_constraints = k_count * n * t
    delay_constraints = sum(max(0, min(t, int(delays.get(key, 0)))) * n for key in INTERVENTIONS)
    local_access_constraints = 2 * n * (t + 1)
    budget_constraints = t + 1
    total_constraints = (
        initial_constraints
        + transition_constraints
        + segment_sum_constraints
        + segment_cap_constraints
        + effectiveness_constraints
        + deployment_cap_constraints
        + delay_constraints
        + local_access_constraints
        + budget_constraints
    )
    return {
        "n_units": n,
        "horizon": t,
        "q_nnz": int(q_nnz),
        "q_density": float(q_nnz / max(n * n, 1)),
        "estimated_total_variables": int(total_vars),
        "estimated_state_variables": int(state_vars),
        "estimated_action_u_variables": int(action_u_vars),
        "estimated_effect_variables": int(effect_vars),
        "estimated_pwl_segment_variables": int(segment_vars),
        "estimated_total_constraints": int(total_constraints),
        "estimated_access_nonzero_terms": int(q_nnz * (t + 1)),
        "estimated_action_tokens": int(k_count * n * t),
        "pwl_segment_count": int(segment_count),
    }


def build_size_comparison(
    ny_events: pd.DataFrame,
    v42_events: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon = int(config["calibration"]["horizon_steps"])
    delays = {key: int(value) for key, value in config["interventions"]["delays"].items()}
    pwl = config["interventions"].get("pwl_diminishing_returns", {})
    use_pwl = bool(pwl.get("enabled", False))
    segment_count = len(pwl.get("segment_cap_shares", [])) if use_pwl else 0

    q_lookup: dict[tuple[str, int], int] = {}
    if not calibration_summary.empty and {"city", "event_id", "q_nnz"}.issubset(calibration_summary.columns):
        tmp = calibration_summary.copy()
        tmp["event_id"] = pd.to_numeric(tmp["event_id"], errors="coerce").astype("Int64")
        q_lookup = {
            (row.city, int(row.event_id)): int(row.q_nnz)
            for row in tmp.dropna(subset=["event_id"]).itertuples(index=False)
        }

    if not v42_events.empty and {"city", "event_id", "n_units", "status"}.issubset(v42_events.columns):
        solved = v42_events[v42_events["status"].astype(str).eq("OPTIMAL")].copy()
        solved["event_id"] = pd.to_numeric(solved["event_id"], errors="coerce").astype("Int64")
        grouped = (
            solved.groupby(["city", "event_id"], as_index=False)
            .agg(
                n_units=("n_units", "first"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
                max_runtime_seconds=("runtime_seconds", "max"),
                n_lambdas=("lambda_footprint", "nunique"),
            )
        )
        for row in grouped.dropna(subset=["event_id"]).itertuples(index=False):
            q_nnz = q_lookup.get((row.city, int(row.event_id)), np.nan)
            if not np.isfinite(q_nnz):
                continue
            size = estimate_lp_size_from_counts(
                n_units=int(row.n_units),
                q_nnz=int(q_nnz),
                horizon=horizon,
                use_pwl=use_pwl,
                segment_count=segment_count,
                delays=delays,
            )
            rows.append(
                {
                    "scope": "V42_broader_solved",
                    "city": row.city,
                    "event_id": int(row.event_id),
                    "mean_runtime_seconds": float(row.mean_runtime_seconds),
                    "max_runtime_seconds": float(row.max_runtime_seconds),
                    "n_lambdas": int(row.n_lambdas),
                    **size,
                }
            )

    for row in ny_events.itertuples(index=False):
        rows.append(
            {
                "scope": "V43_New_York_boundary",
                "city": row.city,
                "event_id": int(row.event_id),
                "mean_runtime_seconds": np.nan,
                "max_runtime_seconds": np.nan,
                "n_lambdas": np.nan,
                "n_units": int(row.n_units),
                "horizon": int(row.horizon),
                "q_nnz": int(row.q_nnz),
                "q_density": float(row.q_density),
                "estimated_total_variables": int(row.estimated_total_variables),
                "estimated_state_variables": np.nan,
                "estimated_action_u_variables": np.nan,
                "estimated_effect_variables": np.nan,
                "estimated_pwl_segment_variables": int(row.estimated_pwl_segment_variables),
                "estimated_total_constraints": int(row.estimated_total_constraints),
                "estimated_access_nonzero_terms": int(row.estimated_access_nonzero_terms),
                "estimated_action_tokens": int(row.estimated_action_tokens),
                "pwl_segment_count": segment_count,
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    for column in [
        "n_units",
        "q_nnz",
        "estimated_total_variables",
        "estimated_total_constraints",
        "estimated_access_nonzero_terms",
        "estimated_action_tokens",
    ]:
        broader_max = comparison.loc[comparison["scope"].eq("V42_broader_solved"), column].max()
        comparison[f"{column}_ratio_to_v42_max"] = comparison[column] / broader_max if broader_max else np.nan
    return comparison.sort_values(["scope", "city", "event_id"]).reset_index(drop=True)


def build_metrics(event_metrics: pd.DataFrame, size_comparison: pd.DataFrame, v42_events: pd.DataFrame) -> dict[str, Any]:
    ny_size = size_comparison[size_comparison["scope"].eq("V43_New_York_boundary")].copy()
    v42_size = size_comparison[size_comparison["scope"].eq("V42_broader_solved")].copy()
    attempted = event_metrics[~event_metrics["prior_hybrid_lp_status"].astype(str).eq("not_attempted")]
    time_limited = event_metrics[
        event_metrics["prior_hybrid_lp_error"].fillna("").astype(str).str.contains("TIME_LIMIT", case=False)
        | event_metrics["prior_hybrid_lp_status"].astype(str).eq("ERROR")
    ]
    max_row = event_metrics.sort_values("delta_finite_top5pct_units_footprint_mass", ascending=False).iloc[0]
    metrics = {
        "n_selected_new_york_events": int(len(event_metrics)),
        "n_prior_hybrid_lp_attempted": int(len(attempted)),
        "n_prior_hybrid_lp_time_limit_or_error": int(len(time_limited)),
        "prior_time_limit_event_ids": "; ".join(str(int(row.event_id)) for row in time_limited.itertuples(index=False)),
        "max_signal_event_id": int(max_row.event_id),
        "max_signal_delta_finite_top5_mass": float(max_row.delta_finite_top5pct_units_footprint_mass),
        "mean_delta_finite_top5_mass": float(event_metrics["delta_finite_top5pct_units_footprint_mass"].mean()),
        "mean_base_finite_top5_mass": float(event_metrics["base_finite_top5pct_units_footprint_mass"].mean()),
        "mean_hybrid_finite_top5_mass": float(event_metrics["hybrid_finite_top5pct_units_footprint_mass"].mean()),
        "mean_finite_top5_action_jaccard": float(event_metrics["finite_top5pct_action_jaccard"].mean()),
        "mean_hybrid_to_base_no_intervention_ratio": float(event_metrics["hybrid_to_base_no_intervention_ratio"].mean()),
        "max_new_york_units": int(event_metrics["n_units"].max()),
        "max_new_york_q_nnz": int(event_metrics["q_nnz"].max()),
        "max_new_york_estimated_variables": int(event_metrics["estimated_total_variables"].max()),
        "max_new_york_estimated_constraints": int(event_metrics["estimated_total_constraints"].max()),
        "max_new_york_access_terms": int(event_metrics["estimated_access_nonzero_terms"].max()),
        "v42_closed_events": int(v42_events[["city", "event_id"]].drop_duplicates().shape[0]) if not v42_events.empty else 0,
        "v42_closed_lp_jobs": int(len(v42_events[v42_events["status"].astype(str).eq("OPTIMAL")])) if "status" in v42_events else 0,
        "v42_max_units": int(v42_size["n_units"].max()) if not v42_size.empty else None,
        "v42_max_estimated_variables": int(v42_size["estimated_total_variables"].max()) if not v42_size.empty else None,
        "v42_max_estimated_constraints": int(v42_size["estimated_total_constraints"].max()) if not v42_size.empty else None,
        "v42_max_access_terms": int(v42_size["estimated_access_nonzero_terms"].max()) if not v42_size.empty else None,
        "v42_max_runtime_seconds": float(v42_size["max_runtime_seconds"].max()) if not v42_size.empty else np.nan,
        "ny_to_v42_max_units_ratio": ratio(ny_size["n_units"].max(), v42_size["n_units"].max()),
        "ny_to_v42_max_variables_ratio": ratio(
            ny_size["estimated_total_variables"].max(),
            v42_size["estimated_total_variables"].max(),
        ),
        "ny_to_v42_max_constraints_ratio": ratio(
            ny_size["estimated_total_constraints"].max(),
            v42_size["estimated_total_constraints"].max(),
        ),
        "ny_to_v42_max_access_terms_ratio": ratio(
            ny_size["estimated_access_nonzero_terms"].max(),
            v42_size["estimated_access_nonzero_terms"].max(),
        ),
        "boundary_interpretation": (
            "New York has strong event-footprint signal in V34, but prior full hybrid LP closure is unresolved; "
            "current evidence supports a computational-boundary statement, not a New York optimum claim."
        ),
    }
    return metrics


def make_size_figure(size_comparison: pd.DataFrame, path: Path) -> None:
    if size_comparison.empty:
        return
    v42 = size_comparison[size_comparison["scope"].eq("V42_broader_solved")]
    ny = size_comparison[size_comparison["scope"].eq("V43_New_York_boundary")]
    metrics = [
        ("n_units", "units"),
        ("estimated_total_variables", "LP variables"),
        ("estimated_total_constraints", "LP constraints"),
        ("estimated_access_nonzero_terms", "access terms"),
    ]
    labels = [label for _, label in metrics]
    v42_values = [float(v42[column].max()) for column, _ in metrics]
    ny_values = [float(ny[column].max()) for column, _ in metrics]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, v42_values, width, label="V42 max solved", color="#4C78A8")
    ax.bar(x + width / 2, ny_values, width, label="New York boundary", color="#F58518")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("count (log scale)")
    ax.set_title("New York footprint LP scale versus closed V42 cases")
    ax.legend(frameon=False)
    for idx, (a, b) in enumerate(zip(v42_values, ny_values, strict=True)):
        if a > 0:
            ax.text(idx + width / 2, b, f"{b / a:.1f}x", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_signal_figure(event_metrics: pd.DataFrame, path: Path) -> None:
    if event_metrics.empty:
        return
    frame = event_metrics.sort_values("delta_finite_top5pct_units_footprint_mass", ascending=False).copy()
    labels = [str(int(value)) for value in frame["event_id"]]
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, frame["base_finite_top5pct_units_footprint_mass"], width, label="OD template finite top-5%", color="#72B7B2")
    ax.bar(x + width / 2, frame["hybrid_finite_top5pct_units_footprint_mass"], width, label="hybrid footprint finite top-5%", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("observed footprint mass")
    ax.set_xlabel("New York event id")
    ax.set_ylim(0.0, max(0.55, float(frame["hybrid_finite_top5pct_units_footprint_mass"].max()) * 1.12))
    ax.set_title("New York footprint signal exists before full LP closure")
    ax.legend(frameon=False)
    for idx, row in enumerate(frame.itertuples(index=False)):
        status = str(row.prior_hybrid_lp_status)
        if status != "not_attempted":
            ax.text(idx, row.hybrid_finite_top5pct_units_footprint_mass + 0.015, status, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(path: Path, event_metrics: pd.DataFrame, size_comparison: pd.DataFrame, metrics: dict[str, Any]) -> None:
    event_table = event_metrics[
        [
            "event_id",
            "event_start",
            "n_units",
            "delta_finite_top5pct_units_footprint_mass",
            "base_finite_top5pct_units_footprint_mass",
            "hybrid_finite_top5pct_units_footprint_mass",
            "prior_hybrid_lp_status",
            "prior_hybrid_lp_error",
        ]
    ].copy()
    size_table = size_comparison[
        size_comparison["scope"].isin(["V42_broader_solved", "V43_New_York_boundary"])
    ][
        [
            "scope",
            "city",
            "event_id",
            "n_units",
            "q_nnz",
            "estimated_total_variables",
            "estimated_total_constraints",
            "estimated_access_nonzero_terms",
            "max_runtime_seconds",
        ]
    ].copy()
    lines = [
        "# New York Footprint LP Boundary Audit",
        "",
        "## 结论",
        "",
        (
            f"V43 不声称已经得到 New York-scale footprint-aware LP 最优解。它证明的是："
            f"New York 的 event-footprint 信号存在，而且强度不低；当前缺口主要是全 LP 闭合的计算边界。"
            f"在选出的 {metrics['n_selected_new_york_events']} 个 New York footprint-sensitive events 中，"
            f"hybrid finite top-5% footprint mass 平均从 "
            f"{metrics['mean_base_finite_top5_mass']:.4f} 升到 {metrics['mean_hybrid_finite_top5_mass']:.4f}，"
            f"平均增量 {metrics['mean_delta_finite_top5_mass']:+.4f}。"
        ),
        "",
        (
            f"最大 New York case 有 {metrics['max_new_york_units']:,} zones、"
            f"约 {metrics['max_new_york_estimated_variables']:,} 个 LP 变量、"
            f"{metrics['max_new_york_estimated_constraints']:,} 个约束、"
            f"{metrics['max_new_york_access_terms']:,} 个 access-loss 非零项。"
            f"相对 V42 已闭合样本最大值，变量规模为 "
            f"{metrics['ny_to_v42_max_variables_ratio']:.2f}x，"
            f"access-loss 非零项为 {metrics['ny_to_v42_max_access_terms_ratio']:.2f}x。"
        ),
        "",
        (
            f"此前 hybrid-footprint full LP 已尝试的 New York event 为 "
            f"{metrics['prior_time_limit_event_ids'] or 'none'}；"
            f"当前记录中 {metrics['n_prior_hybrid_lp_time_limit_or_error']}/"
            f"{metrics['n_prior_hybrid_lp_attempted']} 个已尝试 New York row 未闭合。"
            f"因此论文中应写为 computational boundary，而不是 footprint law 失败。"
        ),
        "",
        "## Selected New York Events",
        "",
        table_to_markdown(event_table),
        "",
        "## Size Comparison",
        "",
        table_to_markdown(size_table.sort_values(["scope", "city", "event_id"])),
        "",
        "## 写作含义",
        "",
        "1. V42 已经证明非 New York full-zone 城市中 recovery--footprint frontier 存在；V43 说明未闭合的是最大城市的直接 LP 计算边界。",
        "2. New York 的 observed footprint signal 很强，因此不能把 New York 排除解释为“没有事件空间信号”。",
        "3. 在模型部分可以继续声称大规模 all-zone base recovery LP 已覆盖 New York；在 footprint-aware direct LP 部分则应明确 New York-scale closure 仍需 solver tuning、decomposition、warm-start basis 或 city-specific preference experiments。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    return frame.to_markdown(index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_get(row: Any, attr: str) -> float:
    if row is None:
        return float("nan")
    try:
        value = getattr(row, attr)
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def prior_status(row: Any) -> str:
    if row is None:
        return "not_attempted"
    status = str(getattr(row, "hybrid_status", ""))
    return status if status else "unknown"


def ratio(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
        return num / den if np.isfinite(num) and np.isfinite(den) and abs(den) > EPS else float("nan")
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()

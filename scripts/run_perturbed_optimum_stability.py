"""Measure near-optimal action stability under small LP parameter perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learn_recovery_laws import load_inputs, prepare_interventions
from recoverable_resilience.calibration import load_yaml
from recoverable_resilience.event_calibration import calibrate_observed_event_city
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import INTERVENTIONS, RecoveryLPParameters, solve_recovery_lp


EPS = 1e-12
ACTION_KEYS = ["city", "event_id", "unit", "t", "intervention"]
POLICY_NAMES = ("base_lp", "static_small_signal_greedy", "residual_finite_greedy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--output-dir", default="results/perturbed_optimum_stability")
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--events-per-city", type=int, default=1)
    parser.add_argument("--max-reference-runtime-seconds", type=float, default=40.0)
    parser.add_argument("--num-perturbations", type=int, default=4)
    parser.add_argument("--eta-sigma", type=float, default=0.08)
    parser.add_argument("--cost-sigma", type=float, default=0.08)
    parser.add_argument("--multiplier-clip", type=float, default=0.20)
    parser.add_argument("--time-limit-seconds", type=float, default=180.0)
    parser.add_argument("--method", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    solver = config.get("solver", {})
    method = int(args.method if args.method is not None else solver.get("method", -1))

    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_events": table_dir / "selected_events.csv",
        "solves": table_dir / "perturbed_solve_summary.csv",
        "actions": table_dir / "perturbed_selected_actions.csv",
        "frequency": table_dir / "perturbed_action_frequency.csv",
        "overlap": table_dir / "perturbed_policy_overlap.csv",
        "event_summary": table_dir / "perturbed_event_stability_summary.csv",
    }
    if not args.resume:
        for path in paths.values():
            if path.exists():
                path.unlink()

    data = load_inputs(root)
    base_summary = data["summary"].copy()
    base_summary = base_summary[(base_summary["status"] == "OPTIMAL") & (base_summary["scenario"] == "base")].copy()
    base_summary["event_id"] = pd.to_numeric(base_summary["event_id"], errors="coerce").astype(int)
    residual_metrics = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_event_metrics.csv")
    selected_events = select_events(
        residual_metrics,
        base_summary,
        events_per_city=int(args.events_per_city),
        max_events=int(args.max_events),
        max_reference_runtime_seconds=float(args.max_reference_runtime_seconds),
    )
    write_table(selected_events, paths["selected_events"])

    events = data["events"].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    event_lookup = {(row.city, int(row.event_id)): row for row in events.itertuples(index=False)}
    dynamic_lookup = {row["city"]: row for _, row in data["dynamics"].iterrows()}
    abnormal = data["abnormal"].copy()
    completed = completed_keys(paths["solves"]) if args.resume else set()
    rng = np.random.default_rng(20260529)

    total_jobs = len(selected_events) * int(args.num_perturbations)
    job_idx = 0
    for _, row in selected_events.iterrows():
        city = str(row["city"])
        event_id = int(row["event_id"])
        event_row = event_lookup.get((city, event_id))
        if event_row is None or city not in dynamic_lookup:
            continue
        params = calibrate_observed_event_city(
            city,
            config,
            pd.Series(event_row._asdict()),
            dynamic_lookup[city],
            abnormal_hourly=abnormal,
            root=root,
        )
        for perturbation_id in range(int(args.num_perturbations)):
            job_idx += 1
            key = (city, event_id, perturbation_id)
            if key in completed:
                print(f"[{job_idx}/{total_jobs}] Skipping completed perturbation {city} event {event_id} #{perturbation_id}", flush=True)
                continue
            print(f"[{job_idx}/{total_jobs}] Perturbed LP {city} event {event_id} #{perturbation_id}", flush=True)
            perturbed = perturb_params(
                params,
                rng,
                eta_sigma=float(args.eta_sigma),
                cost_sigma=float(args.cost_sigma),
                multiplier_clip=float(args.multiplier_clip),
                perturbation_id=perturbation_id,
            )
            try:
                solution = solve_recovery_lp(
                    perturbed,
                    output_flag=bool(solver.get("output_flag", False)),
                    method=method,
                    time_limit_seconds=float(args.time_limit_seconds),
                )
                baseline_objective = float(row["baseline_objective"])
                solve_row = {
                    **event_metadata(row),
                    "perturbation_id": int(perturbation_id),
                    "status": solution.status,
                    "runtime_seconds": float(solution.runtime_seconds),
                    "perturbed_objective": float(solution.objective),
                    "perturbed_recoverable_fraction": fraction_recovered(baseline_objective, float(solution.objective)),
                    "total_intervention_cost": float(solution.interventions["effective_cost"].sum()),
                    "selected_action_count": int(selected_action_count(solution.interventions)),
                    "eta_sigma": float(args.eta_sigma),
                    "cost_sigma": float(args.cost_sigma),
                    "multiplier_clip": float(args.multiplier_clip),
                    "error": "",
                }
                append_csv(pd.DataFrame([solve_row]), paths["solves"])
                actions = positive_actions(solution.interventions, row, perturbation_id)
                append_csv(actions, paths["actions"])
            except Exception as exc:  # pragma: no cover - long batch diagnostics
                print(f"ERROR {city} event {event_id} #{perturbation_id}: {exc}", flush=True)
                error_row = {
                    **event_metadata(row),
                    "perturbation_id": int(perturbation_id),
                    "status": "ERROR",
                    "runtime_seconds": np.nan,
                    "perturbed_objective": np.nan,
                    "perturbed_recoverable_fraction": np.nan,
                    "total_intervention_cost": np.nan,
                    "selected_action_count": 0,
                    "eta_sigma": float(args.eta_sigma),
                    "cost_sigma": float(args.cost_sigma),
                    "multiplier_clip": float(args.multiplier_clip),
                    "error": str(exc),
                }
                append_csv(pd.DataFrame([error_row]), paths["solves"])

    solves = pd.read_csv(paths["solves"]) if paths["solves"].exists() else pd.DataFrame()
    actions = pd.read_csv(paths["actions"]) if paths["actions"].exists() else pd.DataFrame()
    policy_sets = load_policy_action_sets(root)
    frequency = build_action_frequency(actions, solves, policy_sets)
    overlap = build_policy_overlap(frequency, policy_sets)
    event_summary = build_event_summary(solves, frequency, overlap)
    write_table(frequency, paths["frequency"])
    write_table(overlap, paths["overlap"])
    write_table(event_summary, paths["event_summary"])
    make_figures(frequency, overlap, event_summary, figure_dir)
    write_report(
        report_dir / "perturbed_optimum_stability_report_zh.md",
        selected_events,
        solves,
        frequency,
        overlap,
        event_summary,
        args,
    )
    print(f"Wrote perturbed optimum stability to {output_dir}")


def select_events(
    residual_metrics: pd.DataFrame,
    base_summary: pd.DataFrame,
    *,
    events_per_city: int,
    max_events: int,
    max_reference_runtime_seconds: float,
) -> pd.DataFrame:
    residual = residual_metrics.copy()
    residual["event_id"] = pd.to_numeric(residual["event_id"], errors="coerce").astype(int)
    base_cols = [
        "city",
        "event_id",
        "event_start",
        "n_units",
        "runtime_seconds",
        "baseline_objective",
        "optimized_objective",
        "recoverable_fraction",
        "total_budget",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    merged = residual.merge(base_summary[base_cols], on=["city", "event_id", "event_start"], how="left", suffixes=("", "_base"))
    merged = merged[merged["runtime_seconds"].fillna(np.inf) <= max_reference_runtime_seconds].copy()
    merged = merged.sort_values(["city", "residual_gain_improvement_over_static"], ascending=[True, False])
    per_city = merged.groupby("city", as_index=False).head(max(events_per_city, 1)).copy()
    per_city = per_city.sort_values("residual_gain_improvement_over_static", ascending=False).head(max_events).copy()
    keep = [
        "city",
        "event_id",
        "event_start",
        "n_units",
        "runtime_seconds",
        "baseline_objective",
        "optimized_objective",
        "recoverable_fraction",
        "static_fraction_of_lp_gain",
        "residual_fraction_of_lp_gain",
        "residual_gain_improvement_over_static",
        "residual_gap_to_lp",
        "total_budget",
        "event_peak_positive_abnormal_deficit",
        "event_total_precip",
    ]
    return per_city[[column for column in keep if column in per_city.columns]].reset_index(drop=True)


def perturb_params(
    params: RecoveryLPParameters,
    rng: np.random.Generator,
    *,
    eta_sigma: float,
    cost_sigma: float,
    multiplier_clip: float,
    perturbation_id: int,
) -> RecoveryLPParameters:
    eta: dict[str, np.ndarray] = {}
    cost: dict[str, np.ndarray] = {}
    for key in INTERVENTIONS:
        eta[key] = params.eta[key] * smooth_multiplier(
            rng,
            params.n_units,
            params.horizon,
            sigma=eta_sigma,
            clip=multiplier_clip,
        )
        cost[key] = params.cost[key] * smooth_multiplier(
            rng,
            params.n_units,
            params.horizon,
            sigma=cost_sigma,
            clip=multiplier_clip,
        )
    return RecoveryLPParameters(
        city=params.city,
        units=list(params.units),
        p=params.p.copy(),
        q=params.q.copy(),
        b0=params.b0.copy(),
        a=params.a.copy(),
        h=params.h.copy(),
        eta=eta,
        cost=cost,
        u_cap={key: value.copy() for key, value in (params.u_cap or {}).items()},
        u_segment_cap={key: value.copy() for key, value in (params.u_segment_cap or {}).items()} or None,
        segment_effectiveness={key: value.copy() for key, value in (params.segment_effectiveness or {}).items()} or None,
        period_budget=params.period_budget.copy(),
        total_budget=float(params.total_budget),
        delays=dict(params.delays),
        delta_c=float(params.delta_c),
        delta_s=float(params.delta_s),
        delta_t=float(params.delta_t),
        metadata={
            **dict(params.metadata or {}),
            "perturbation_id": int(perturbation_id),
            "eta_sigma": float(eta_sigma),
            "cost_sigma": float(cost_sigma),
            "multiplier_clip": float(multiplier_clip),
        },
    )


def smooth_multiplier(
    rng: np.random.Generator,
    n_units: int,
    horizon: int,
    *,
    sigma: float,
    clip: float,
) -> np.ndarray:
    if sigma <= 0:
        return np.ones((n_units, horizon), dtype=float)
    unit_noise = rng.normal(0.0, sigma, size=(n_units, 1))
    time_noise = rng.normal(0.0, sigma * 0.5, size=(1, horizon))
    raw = 1.0 + unit_noise + time_noise
    return np.clip(raw, max(0.05, 1.0 - clip), 1.0 + clip)


def event_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "city": str(row["city"]),
        "event_id": int(row["event_id"]),
        "event_start": str(row["event_start"]),
        "n_units": int(row["n_units"]),
        "baseline_objective": float(row["baseline_objective"]),
        "base_optimized_objective": float(row["optimized_objective"]),
        "base_recoverable_fraction": float(row["recoverable_fraction"]),
        "base_static_fraction_of_lp_gain": float(row["static_fraction_of_lp_gain"]),
        "base_residual_fraction_of_lp_gain": float(row["residual_fraction_of_lp_gain"]),
        "base_residual_improvement": float(row["residual_gain_improvement_over_static"]),
        "base_runtime_seconds": float(row["runtime_seconds"]),
        "event_peak_positive_abnormal_deficit": float(row["event_peak_positive_abnormal_deficit"]),
        "event_total_precip": float(row["event_total_precip"]),
    }


def selected_action_count(interventions: pd.DataFrame) -> int:
    positive = interventions[
        (interventions["u"] > 1e-10)
        | (interventions["e"] > 1e-10)
        | (interventions["effective_cost"] > 1e-10)
    ].copy()
    if positive.empty:
        return 0
    return int(positive[["unit", "t", "intervention"]].drop_duplicates().shape[0])


def positive_actions(solution_interventions: pd.DataFrame, event_row: pd.Series, perturbation_id: int) -> pd.DataFrame:
    positive = solution_interventions[
        (solution_interventions["u"] > 1e-10)
        | (solution_interventions["e"] > 1e-10)
        | (solution_interventions["effective_cost"] > 1e-10)
    ].copy()
    if positive.empty:
        return pd.DataFrame()
    grouped = (
        positive.assign(unit=positive["unit"].astype(str), t=pd.to_numeric(positive["t"], errors="coerce").astype(int))
        .groupby(["unit", "t", "intervention"], as_index=False)
        .agg(
            perturbed_u=("u", "sum"),
            perturbed_e=("e", "sum"),
            perturbed_cost=("effective_cost", "sum"),
        )
    )
    grouped.insert(0, "city", str(event_row["city"]))
    grouped.insert(1, "event_id", int(event_row["event_id"]))
    grouped.insert(2, "event_start", str(event_row["event_start"]))
    grouped.insert(3, "perturbation_id", int(perturbation_id))
    return grouped


def load_policy_action_sets(root: Path) -> pd.DataFrame:
    base_lp = pd.read_csv(root / "results" / "event_optimization" / "tables" / "event_optimization_interventions.csv")
    static = pd.read_csv(root / "results" / "law_learning" / "tables" / "greedy_oracle_actions.csv.gz")
    residual = pd.read_csv(root / "results" / "residual_greedy_policy" / "tables" / "residual_greedy_allocations.csv.gz")
    frames = [
        policy_actions(base_lp, "base_lp", cost_col="effective_cost"),
        policy_actions(static, "static_small_signal_greedy", cost_col="allocated_cost"),
        policy_actions(residual, "residual_finite_greedy", cost_col="allocated_cost"),
    ]
    return pd.concat(frames, ignore_index=True)


def policy_actions(df: pd.DataFrame, policy: str, *, cost_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*ACTION_KEYS, "policy", "policy_cost"])
    work = df.copy()
    work["unit"] = work["unit"].astype(str)
    work["event_id"] = pd.to_numeric(work["event_id"], errors="coerce").astype(int)
    work["t"] = pd.to_numeric(work["t"], errors="coerce").astype(int)
    cost = pd.to_numeric(work.get(cost_col, 0.0), errors="coerce").fillna(0.0)
    if cost_col in work:
        work = work[cost > 1e-10].copy()
    grouped = work.groupby(ACTION_KEYS, as_index=False).agg(policy_cost=(cost_col, "sum"))
    grouped["policy"] = policy
    return grouped[[*ACTION_KEYS, "policy", "policy_cost"]]


def build_action_frequency(
    actions: pd.DataFrame,
    solves: pd.DataFrame,
    policy_sets: pd.DataFrame,
) -> pd.DataFrame:
    if actions.empty or solves.empty:
        return pd.DataFrame()
    actions = normalize_action_keys(actions)
    policy_sets = normalize_action_keys(policy_sets)
    ok = solves[solves["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    success_counts = ok.groupby(["city", "event_id"], as_index=False).agg(n_successful_perturbations=("perturbation_id", "nunique"))
    grouped = (
        actions.groupby(ACTION_KEYS, as_index=False)
        .agg(
            selected_count=("perturbation_id", "nunique"),
            mean_perturbed_cost=("perturbed_cost", "mean"),
            total_perturbed_cost=("perturbed_cost", "sum"),
        )
        .merge(success_counts, on=["city", "event_id"], how="left")
    )
    grouped["selection_frequency"] = grouped["selected_count"] / grouped["n_successful_perturbations"].replace(0, np.nan)
    for policy in POLICY_NAMES:
        selected = policy_sets[policy_sets["policy"].eq(policy)][ACTION_KEYS].drop_duplicates()
        selected[f"selected_by_{policy}"] = True
        grouped = grouped.merge(selected, on=ACTION_KEYS, how="left")
        grouped[f"selected_by_{policy}"] = grouped[f"selected_by_{policy}"].eq(True)
    return grouped.sort_values(["selection_frequency", "total_perturbed_cost"], ascending=False)


def build_policy_overlap(frequency: pd.DataFrame, policy_sets: pd.DataFrame) -> pd.DataFrame:
    if frequency.empty:
        return pd.DataFrame()
    frequency = normalize_action_keys(frequency)
    policy_sets = normalize_action_keys(policy_sets)
    rows: list[dict[str, Any]] = []
    event_keys = frequency[["city", "event_id"]].drop_duplicates()
    for event in event_keys.itertuples(index=False):
        city = str(event.city)
        event_id = int(event.event_id)
        event_freq = frequency[(frequency["city"] == city) & (frequency["event_id"] == event_id)].copy()
        all_freq_mass = float(event_freq["selection_frequency"].sum())
        any_perturbed = action_tuple_set(event_freq)
        stable50 = action_tuple_set(event_freq[event_freq["selection_frequency"] >= 0.50])
        stable80 = action_tuple_set(event_freq[event_freq["selection_frequency"] >= 0.80])
        for policy in POLICY_NAMES:
            policy_frame = policy_sets[
                policy_sets["policy"].eq(policy)
                & policy_sets["city"].astype(str).eq(city)
                & (pd.to_numeric(policy_sets["event_id"], errors="coerce").astype(int) == event_id)
            ].copy()
            policy_set = action_tuple_set(policy_frame)
            policy_freq = event_freq.merge(policy_frame[ACTION_KEYS].drop_duplicates(), on=ACTION_KEYS, how="inner")
            rows.append(
                {
                    "city": city,
                    "event_id": event_id,
                    "policy": policy,
                    "n_perturbed_actions": len(any_perturbed),
                    "n_stable50_actions": len(stable50),
                    "n_stable80_actions": len(stable80),
                    "policy_action_count": len(policy_set),
                    "jaccard_any_perturbed": jaccard(policy_set, any_perturbed),
                    "jaccard_stable50": jaccard(policy_set, stable50),
                    "jaccard_stable80": jaccard(policy_set, stable80),
                    "stable50_recall": recall(policy_set, stable50),
                    "stable80_recall": recall(policy_set, stable80),
                    "stable50_precision": precision(policy_set, stable50),
                    "stable80_precision": precision(policy_set, stable80),
                    "frequency_mass_capture": float(policy_freq["selection_frequency"].sum() / max(all_freq_mass, EPS)),
                    "mean_selected_frequency": float(policy_freq["selection_frequency"].mean()) if not policy_freq.empty else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "frequency_mass_capture"], ascending=[True, False])


def build_event_summary(solves: pd.DataFrame, frequency: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    if solves.empty:
        return pd.DataFrame()
    ok = solves[solves["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    event = ok.groupby(["city", "event_id"], as_index=False).agg(
        n_successful_perturbations=("perturbation_id", "nunique"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_perturbed_recoverable_fraction=("perturbed_recoverable_fraction", "mean"),
        mean_selected_action_count=("selected_action_count", "mean"),
        base_residual_improvement=("base_residual_improvement", "first"),
    )
    if not frequency.empty:
        freq_summary = frequency.groupby(["city", "event_id"], as_index=False).agg(
            n_any_perturbed_actions=("selection_frequency", "count"),
            n_stable50_actions=("selection_frequency", lambda s: int((s >= 0.50).sum())),
            n_stable80_actions=("selection_frequency", lambda s: int((s >= 0.80).sum())),
            mean_selection_frequency=("selection_frequency", "mean"),
            max_selection_frequency=("selection_frequency", "max"),
        )
        event = event.merge(freq_summary, on=["city", "event_id"], how="left")
    if not overlap.empty:
        pivot = overlap.pivot_table(
            index=["city", "event_id"],
            columns="policy",
            values=["frequency_mass_capture", "stable50_recall", "stable80_recall"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{policy}" for metric, policy in pivot.columns]
        event = event.merge(pivot.reset_index(), on=["city", "event_id"], how="left")
    return event


def action_tuple_set(df: pd.DataFrame) -> set[tuple[str, int, str]]:
    if df.empty:
        return set()
    return {
        (str(row.unit), int(row.t), str(row.intervention))
        for row in df[["unit", "t", "intervention"]].drop_duplicates().itertuples(index=False)
    }


def normalize_action_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "city" in out:
        out["city"] = out["city"].astype(str)
    if "event_id" in out:
        out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype(int)
    if "unit" in out:
        out["unit"] = out["unit"].astype(str)
    if "t" in out:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype(int)
    if "intervention" in out:
        out["intervention"] = out["intervention"].astype(str)
    return out


def jaccard(a: set[Any], b: set[Any]) -> float:
    union = len(a | b)
    return float(len(a & b) / union) if union else np.nan


def recall(policy: set[Any], target: set[Any]) -> float:
    return float(len(policy & target) / len(target)) if target else np.nan


def precision(policy: set[Any], target: set[Any]) -> float:
    return float(len(policy & target) / len(policy)) if policy else np.nan


def fraction_recovered(baseline_objective: float, objective: float) -> float:
    return float(1.0 - objective / baseline_objective) if baseline_objective > EPS else np.nan


def completed_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path)
    if existing.empty or not {"city", "event_id", "perturbation_id", "status"}.issubset(existing.columns):
        return set()
    valid = existing[existing["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])].copy()
    return {
        (str(row.city), int(row.event_id), int(row.perturbation_id))
        for row in valid[["city", "event_id", "perturbation_id"]].itertuples(index=False)
    }


def make_figures(frequency: pd.DataFrame, overlap: pd.DataFrame, event_summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not overlap.empty:
        make_policy_mass_figure(overlap, figure_dir / "perturbed_policy_frequency_mass.png")
        make_stable_recall_figure(overlap, figure_dir / "perturbed_stable_core_recall.png")
    if not frequency.empty:
        make_frequency_histogram(frequency, figure_dir / "perturbed_action_frequency_histogram.png")
    if not event_summary.empty:
        make_event_stability_figure(event_summary, figure_dir / "perturbed_event_stability.png")


def make_policy_mass_figure(overlap: pd.DataFrame, path: Path) -> None:
    summary = overlap.groupby("policy", as_index=False).agg(
        mean_frequency_mass_capture=("frequency_mass_capture", "mean"),
        mean_jaccard_stable50=("jaccard_stable50", "mean"),
    )
    order = [policy for policy in POLICY_NAMES if policy in set(summary["policy"])]
    summary["policy"] = pd.Categorical(summary["policy"], categories=order, ordered=True)
    summary = summary.sort_values("policy")
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    colors = {"base_lp": "#64748b", "static_small_signal_greedy": "#94a3b8", "residual_finite_greedy": "#2563eb"}
    x = np.arange(len(summary))
    labels = summary["policy"].astype(str).tolist()
    ax.bar(x, summary["mean_frequency_mass_capture"], color=[colors[p] for p in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Perturbed selection-frequency mass captured")
    ax.set_title("Policy overlap with perturbed LP action stability")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels])
    for idx, value in enumerate(summary["mean_frequency_mass_capture"]):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_stable_recall_figure(overlap: pd.DataFrame, path: Path) -> None:
    summary = overlap.groupby("policy", as_index=False).agg(
        stable50_recall=("stable50_recall", "mean"),
        stable80_recall=("stable80_recall", "mean"),
    )
    order = [policy for policy in POLICY_NAMES if policy in set(summary["policy"])]
    x = np.arange(len(order))
    width = 0.36
    summary = summary.set_index("policy").reindex(order)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.bar(x - width / 2, summary["stable50_recall"], width=width, label="frequency >= 0.5", color="#38bdf8")
    ax.bar(x + width / 2, summary["stable80_recall"], width=width, label="frequency >= 0.8", color="#1d4ed8")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [label.replace("_", "\n") for label in order])
    ax.set_ylabel("Stable-core recall")
    ax.set_title("How much of the perturbed stable core is recovered?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_frequency_histogram(frequency: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(frequency["selection_frequency"], bins=np.linspace(0, 1, 11), color="#2563eb", alpha=0.78, edgecolor="white")
    ax.set_xlabel("Perturbed LP selection frequency")
    ax.set_ylabel("Action count")
    ax.set_title("Near-optimal action stability distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_event_stability_figure(event_summary: pd.DataFrame, path: Path) -> None:
    if "frequency_mass_capture_residual_finite_greedy" not in event_summary:
        return
    labels = event_summary.apply(lambda row: f"{row['city']} {int(row['event_id'])}", axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(event_summary))
    ax.barh(y, event_summary["frequency_mass_capture_residual_finite_greedy"], color="#2563eb", alpha=0.86)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Residual policy frequency-mass capture")
    ax.set_title("Residual law alignment by perturbed event")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    selected_events: pd.DataFrame,
    solves: pd.DataFrame,
    frequency: pd.DataFrame,
    overlap: pd.DataFrame,
    event_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    success = solves[solves["status"].astype(str).isin(["OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"])] if not solves.empty else pd.DataFrame()
    mean_overlap = overlap.groupby("policy", as_index=False).agg(
        mean_frequency_mass_capture=("frequency_mass_capture", "mean"),
        mean_stable50_recall=("stable50_recall", "mean"),
        mean_stable80_recall=("stable80_recall", "mean"),
        mean_jaccard_stable50=("jaccard_stable50", "mean"),
    ) if not overlap.empty else pd.DataFrame()
    lines = [
        "# Perturbed-Optimum Stability V11",
        "",
        "## 这一版回答什么问题",
        "",
        "V5-V10 已经给出 marginal recovery law 和 residual finite-budget law。V11 补上 high-level idea 中的 perturbed-optimum selection frequency：在轻微扰动 intervention effectiveness 和 cost 后重新求解 LP，观察哪些 action 在近邻优化问题中反复被选中，并比较 base LP、static small-signal greedy、residual finite greedy 与这些稳定 action 的重合程度。",
        "",
        "扰动方式：对每个 intervention 的 unit-time `eta` 和 `cost` 施加平滑乘法扰动，默认 sigma = "
        f"{args.eta_sigma:.3f}/{args.cost_sigma:.3f}，并裁剪在 +/-{args.multiplier_clip:.2f} 内。扰动不改变 OD、速度损失、自然恢复和预算，因此它主要检验 resource-regime 附近的 near-optimal action stability。",
        "",
        "## 求解覆盖",
        "",
        f"- selected events: {len(selected_events)}",
        f"- perturbations per event: {args.num_perturbations}",
        f"- solve rows: {len(solves)}",
        f"- successful rows: {len(success)}",
    ]
    if not solves.empty and "status" in solves:
        lines.append(f"- status counts: {solves['status'].astype(str).value_counts().to_dict()}")
    if not success.empty:
        lines.append(f"- mean runtime seconds: {success['runtime_seconds'].mean():.2f}")
        lines.append(f"- mean perturbed recoverable fraction: {success['perturbed_recoverable_fraction'].mean():.4f}")
    lines.extend(["", "## Selected Events", "", table_to_markdown(selected_events)])
    if not mean_overlap.empty:
        lines.extend(
            [
                "",
                "## Policy Alignment With Perturbed Stable Actions",
                "",
                table_to_markdown(mean_overlap),
            ]
        )
    if not event_summary.empty:
        lines.extend(["", "## Event Summary", "", table_to_markdown(event_summary)])
    if not frequency.empty:
        top = frequency.sort_values(["selection_frequency", "total_perturbed_cost"], ascending=False).head(20)
        lines.extend(["", "## Top Stable Actions", "", table_to_markdown(top)])
    lines.extend(
        [
            "",
            "## 科学解释",
            "",
            "如果 residual finite greedy 捕获了更高的 perturbed selection-frequency mass 或 stable-core recall，说明它不仅接近单个 base optimum，也更接近一族近邻 LP optimum 的稳定 action core。这能缓解“优化器选择只是某个参数点的偶然 tie-breaking”这一风险。若 base LP 本身对 perturbed stable core 的捕获也有限，则说明最优 action set 对资源参数敏感，论文中应把 law 表述为稳定的 value principle，而不是固定 action list。",
        ]
    )
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


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    if path.exists() and path.stat().st_size > 0:
        columns = pd.read_csv(path, nrows=0, compression=compression).columns.tolist()
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.10g", compression=compression)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    df.to_csv(path, index=False, float_format="%.10g", compression=compression)


if __name__ == "__main__":
    main()

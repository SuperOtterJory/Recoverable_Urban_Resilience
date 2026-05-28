"""Run calibrated recoverable-resilience LP scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recoverable_resilience.calibration import calibrate_city, load_yaml, save_params
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.policy_eval import evaluate_default_policies
from recoverable_resilience.recovery_lp import solve_with_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / config["project"]["output_dir"]
    table_dir = output_dir / "tables"
    scenario_dir = output_dir / "scenarios"
    table_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    cities = args.cities or config["calibration"]["cities"]
    scenarios = config["scenarios"]
    if args.scenarios:
        scenarios = [s for s in scenarios if s["name"] in set(args.scenarios)]

    summary_rows = []
    trajectory_frames = []
    intervention_frames = []
    policy_frames = []
    policy_trajectory_frames = []
    solver = config.get("solver", {})
    outputs = config.get("outputs", {})
    save_trajectories = bool(outputs.get("save_trajectories", True))
    save_interventions = bool(outputs.get("save_interventions", True))
    evaluate_policies = bool(outputs.get("evaluate_policies", True))
    save_policy_trajectories = bool(outputs.get("save_policy_trajectories", True))
    save_calibrated_params = bool(outputs.get("save_calibrated_params", True))
    for city in cities:
        for scenario in scenarios:
            scenario_name = scenario["name"]
            print(f"Solving {city} / {scenario_name}")
            params = calibrate_city(city, config, scenario_override=scenario, root=root)
            if save_calibrated_params:
                save_params(params, scenario_dir / f"{city.replace(' ', '_')}_{scenario_name}.json")
            baseline, optimized = solve_with_baseline(
                params,
                output_flag=bool(solver.get("output_flag", False)),
                method=int(solver.get("method", -1)),
                time_limit_seconds=float(solver.get("time_limit_seconds", 120)),
            )
            row = dict(optimized.summary)
            row["scenario"] = scenario_name
            row["baseline_objective"] = baseline.objective
            row["optimized_objective"] = optimized.objective
            row["recoverable_fraction"] = optimized.recoverable_fraction
            row["budget_intensity"] = scenario.get("budget_intensity", config["interventions"]["budget_intensity"])
            summary_rows.append(row)
            if evaluate_policies:
                policies, policy_trajectories = evaluate_default_policies(params, baseline_objective=baseline.objective)
                policies["scenario"] = scenario_name
                policy_frames.append(policies)
                if save_policy_trajectories:
                    policy_trajectories["scenario"] = scenario_name
                    policy_trajectory_frames.append(policy_trajectories)
            if save_trajectories:
                traj = optimized.trajectory.copy()
                traj["scenario"] = scenario_name
                trajectory_frames.append(traj)
            if save_interventions:
                interventions = optimized.interventions.copy()
                interventions["scenario"] = scenario_name
                intervention_frames.append(interventions)

    pd.DataFrame(summary_rows).to_csv(table_dir / "optimization_summary.csv", index=False)
    if trajectory_frames:
        pd.concat(trajectory_frames, ignore_index=True).to_csv(table_dir / "optimization_trajectories.csv", index=False)
    if intervention_frames:
        pd.concat(intervention_frames, ignore_index=True).to_csv(table_dir / "optimization_interventions.csv", index=False)
    if policy_frames:
        pd.concat(policy_frames, ignore_index=True).to_csv(table_dir / "policy_comparison.csv", index=False)
    if policy_trajectory_frames:
        pd.concat(policy_trajectory_frames, ignore_index=True).to_csv(table_dir / "policy_trajectories.csv", index=False)
    print(f"Wrote optimization outputs to {output_dir}")


if __name__ == "__main__":
    main()

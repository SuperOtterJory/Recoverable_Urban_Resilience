"""Run a small parameter sweep for the recovery LP."""

from __future__ import annotations

import argparse

import pandas as pd

from recoverable_resilience.calibration import calibrate_city, load_yaml
from recoverable_resilience.paths import find_repo_root
from recoverable_resilience.recovery_lp import solve_with_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / config["project"]["output_dir"]
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    tuning = config["tuning"]
    cities = args.cities or tuning["cities"]
    solver = config.get("solver", {})
    rows = []
    for city in cities:
        for budget in tuning["budget_intensity_grid"]:
            for delay_name, delays in tuning["response_delay_grid"].items():
                scenario = {
                    "name": f"budget_{budget:g}_{delay_name}",
                    "budget_intensity": budget,
                    "delays": delays,
                }
                print(f"Tuning {city}: budget={budget}, delay={delay_name}")
                params = calibrate_city(city, config, scenario_override=scenario, root=root)
                baseline, optimized = solve_with_baseline(
                    params,
                    output_flag=bool(solver.get("output_flag", False)),
                    method=int(solver.get("method", -1)),
                    time_limit_seconds=float(solver.get("time_limit_seconds", 120)),
                )
                rows.append(
                    {
                        "city": city,
                        "budget_intensity": budget,
                        "delay_name": delay_name,
                        "delay_R": delays["R"],
                        "delay_C": delays["C"],
                        "delay_S": delays["S"],
                        "baseline_objective": baseline.objective,
                        "optimized_objective": optimized.objective,
                        "recoverable_fraction": optimized.recoverable_fraction,
                        "total_budget": params.total_budget,
                        "weighted_b0": float((params.p * params.b0).sum()),
                    }
                )
    pd.DataFrame(rows).to_csv(table_dir / "parameter_tuning_summary.csv", index=False)
    print(f"Wrote tuning summary to {table_dir / 'parameter_tuning_summary.csv'}")


if __name__ == "__main__":
    main()

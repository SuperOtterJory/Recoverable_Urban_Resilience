"""Calibrate city-level parameters for the recovery optimization LP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recoverable_resilience.calibration import (
    calibrate_city,
    calibration_summary,
    load_yaml,
    save_params,
)
from recoverable_resilience.paths import find_repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    cities = args.cities or config["calibration"]["cities"]
    output_dir = root / config["project"]["output_dir"]
    scenario_dir = output_dir / "scenarios"
    table_dir = output_dir / "tables"
    save_calibrated_params = bool(config.get("outputs", {}).get("save_calibrated_params", True))
    rows = []
    for city in cities:
        params = calibrate_city(city, config, root=root)
        if save_calibrated_params:
            save_params(params, scenario_dir / f"{city.replace(' ', '_')}_base.json")
        rows.append(calibration_summary(params))
        print(f"Calibrated {city}: {params.n_units} units, horizon={params.horizon}")
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_dir / "calibration_summary.csv", index=False)
    print(f"Wrote calibration outputs to {output_dir}")


if __name__ == "__main__":
    main()

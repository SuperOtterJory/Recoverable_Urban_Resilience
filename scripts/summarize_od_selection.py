"""Summarize how top OD units compress raw demand into LP units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recoverable_resilience.calibration import select_top_units
from recoverable_resilience.paths import find_repo_root


UNIT_COUNT = 35


def main() -> None:
    root = find_repo_root()
    demand_base = root / "data" / "raw_data" / "demand"
    rows = []
    for demand_dir in sorted(demand_base.glob("* city")):
        demand_path = demand_dir / "demand.csv"
        if not demand_path.exists():
            continue
        city = parse_city(demand_dir)
        demand = pd.read_csv(demand_path)
        rows.append(summarize_city(city, demand))

    out = pd.DataFrame(rows).sort_values("city")
    path = root / "results" / "optimization" / "tables" / "od_selection_summary.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


def parse_city(path: Path) -> str:
    name = path.name
    return name.split("_", 1)[1].removesuffix(" city") if "_" in name else name.removesuffix(" city")


def summarize_city(city: str, demand: pd.DataFrame) -> dict[str, Any]:
    demand = demand.copy()
    demand["o_zone_id"] = demand["o_zone_id"].astype(str)
    demand["d_zone_id"] = demand["d_zone_id"].astype(str)
    demand["volume"] = pd.to_numeric(demand["volume"], errors="coerce").fillna(0.0)
    demand = demand[demand["volume"] > 0].copy()

    selected = set(select_top_units(demand, UNIT_COUNT))
    total_volume = float(demand["volume"].sum())
    selected_origin = demand[demand["o_zone_id"].isin(selected)]
    selected_dest = demand[demand["d_zone_id"].isin(selected)]
    selected_internal = demand[demand["o_zone_id"].isin(selected) & demand["d_zone_id"].isin(selected)]
    selected_touch = demand[demand["o_zone_id"].isin(selected) | demand["d_zone_id"].isin(selected)]
    all_zones = set(demand["o_zone_id"]).union(set(demand["d_zone_id"]))
    return {
        "city": city,
        "raw_od_rows": int(len(demand)),
        "raw_origin_zones": int(demand["o_zone_id"].nunique()),
        "raw_destination_zones": int(demand["d_zone_id"].nunique()),
        "raw_unique_zones_union": int(len(all_zones)),
        "raw_total_od_volume": total_volume,
        "selected_unit_count": UNIT_COUNT,
        "lp_q_matrix_cells": UNIT_COUNT * UNIT_COUNT,
        "selected_internal_od_rows": int(len(selected_internal)),
        "selected_internal_od_volume": float(selected_internal["volume"].sum()),
        "selected_internal_volume_share": safe_div(selected_internal["volume"].sum(), total_volume),
        "selected_origin_volume_share": safe_div(selected_origin["volume"].sum(), total_volume),
        "selected_destination_volume_share": safe_div(selected_dest["volume"].sum(), total_volume),
        "selected_touch_volume_share": safe_div(selected_touch["volume"].sum(), total_volume),
        "selected_unit_share_of_raw_union_zones": safe_div(UNIT_COUNT, len(all_zones)),
        "compression_ratio_raw_zones_to_units": safe_div(len(all_zones), UNIT_COUNT),
        "compression_ratio_raw_od_rows_to_q_cells": safe_div(len(demand), UNIT_COUNT * UNIT_COUNT),
    }


def safe_div(num: Any, den: Any) -> float:
    num = float(num)
    den = float(den)
    return float(num / den) if np.isfinite(den) and den > 0 else float("nan")


if __name__ == "__main__":
    main()

"""Build an event-zone speed footprint proxy from raw TMC speed records.

This analysis addresses the identifiability boundary exposed by the previous
footprint audit.  The current observed-event LP calibration projects a
city-level abnormal speed signal over zones through an OD vulnerability
template.  Here we build an independent proxy:

1. map each speed TMC segment midpoint to its nearest OD zone centroid;
2. compute TMC-hour speed deficits against a matched non-rain temporal baseline;
3. aggregate positive abnormal TMC-hour deficits into event-zone footprints;
4. compare those footprints with the OD vulnerability template.

The output is intentionally diagnostic.  It does not replace the LP calibration
yet; it tells us whether there is enough spatial signal to justify doing so.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from recoverable_resilience.calibration import (
    city_dir_name,
    load_yaml,
    normalize_zone_ids,
    select_units,
)
from recoverable_resilience.event_calibration import destination_vulnerability_for_event
from recoverable_resilience.paths import find_repo_root


INTERVENTION_CITIES = [
    "New York",
    "Chicago",
    "Houston",
    "Philadelphia",
    "San Antonio",
    "Dallas",
    "Austin",
]
WINDOW_HOURS = 12
PRE_EVENT_HOURS = 6
RAIN_INFLUENCE_HOURS = 12
DEFAULT_CHUNK_SIZE = 500_000
EPS = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/optimization.yml")
    parser.add_argument("--events", default="results/data_mining/tables/rainfall_event_impact_details.csv")
    parser.add_argument("--output-dir", default="results/event_spatial_footprint_proxy")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    root = find_repo_root()
    config = load_yaml(root / args.config)
    output_dir = root / args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    cities = args.cities or config["calibration"].get("cities", INTERVENTION_CITIES)
    events = load_event_details(root / args.events, cities)
    if events.empty:
        raise FileNotFoundError("No positive speed-impact rainfall events found for the requested cities.")

    footprint_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    mapping_frames: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] = []
    for city in cities:
        city_events = events[events["city"] == city].copy()
        if city_events.empty:
            continue
        print(f"Building spatial footprint proxy for {city} ({len(city_events)} events)", flush=True)
        city_result = process_city(root, config, city, city_events, chunk_size=args.chunk_size)
        if not city_result["event_zone"].empty:
            footprint_frames.append(city_result["event_zone"])
        if not city_result["event_summary"].empty:
            summary_frames.append(city_result["event_summary"])
        if not city_result["tmc_zone_mapping"].empty:
            mapping_frames.append(city_result["tmc_zone_mapping"])
        baseline_rows.append(city_result["baseline_summary"])

    event_zone = pd.concat(footprint_frames, ignore_index=True) if footprint_frames else pd.DataFrame()
    event_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    tmc_zone_mapping = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    baseline_summary = pd.DataFrame(baseline_rows)
    pairwise, pairwise_summary = build_pairwise_footprint_stability(event_zone)
    city_summary = build_city_summary(event_summary, pairwise_summary, baseline_summary)
    metrics = build_metrics(event_summary, city_summary, pairwise_summary, baseline_summary)

    write_table(event_zone, table_dir / "event_zone_speed_footprint.csv.gz")
    write_table(event_summary, table_dir / "event_spatial_footprint_summary.csv")
    write_table(tmc_zone_mapping, table_dir / "tmc_to_zone_mapping.csv")
    write_table(baseline_summary, table_dir / "event_spatial_footprint_baseline_summary.csv")
    write_table(pairwise, table_dir / "event_spatial_footprint_pairwise_jaccard.csv")
    write_table(pairwise_summary, table_dir / "event_spatial_footprint_pairwise_summary.csv")
    write_table(city_summary, table_dir / "event_spatial_footprint_city_summary.csv")
    (table_dir / "event_spatial_footprint_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_figures(event_summary, city_summary, pairwise_summary, figure_dir)
    write_report(
        report_dir / "event_spatial_footprint_proxy_report_zh.md",
        metrics,
        city_summary,
        event_summary,
        baseline_summary,
    )
    print(f"Wrote event spatial footprint proxy analysis to {output_dir}")


def load_event_details(path: Path, cities: list[str]) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_start", "event_end"])
    events = events[events["city"].isin(cities)].copy()
    impact_available = events["impact_available"].astype(str).str.lower().isin({"true", "1", "yes"})
    events["peak_positive_abnormal_deficit"] = pd.to_numeric(
        events["peak_positive_abnormal_deficit"],
        errors="coerce",
    ).fillna(0.0)
    events = events[impact_available & (events["peak_positive_abnormal_deficit"] > 0)].copy()
    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce").astype(int)
    return events.sort_values(["city", "event_start", "event_id"]).reset_index(drop=True)


def process_city(
    root: Path,
    config: dict[str, Any],
    city: str,
    city_events: pd.DataFrame,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    raw_dir = root / config["project"]["raw_data_dir"]
    speed_dir_name = city_dir_name(city, raw_dir / "speed")
    demand_dir_name = city_dir_name(city, raw_dir / "demand")
    if speed_dir_name is None or demand_dir_name is None:
        raise FileNotFoundError(f"Missing raw speed or demand directory for {city}.")
    speed_dir = raw_dir / "speed" / speed_dir_name
    demand_dir = raw_dir / "demand" / demand_dir_name
    speed_path = find_speed_csv(speed_dir)
    if speed_path is None:
        raise FileNotFoundError(f"Missing raw speed CSV for {city}.")

    all_rain_events = pd.read_csv(root / "results" / "data_mining" / "tables" / "rainfall_event_impact_details.csv", parse_dates=["event_start", "event_end"])
    all_rain_events = all_rain_events[all_rain_events["city"] == city].copy()
    influence_hours = build_influence_hours(all_rain_events)
    event_hours = build_event_hours(city_events)
    event_hour_frame = build_event_hour_frame(city_events)

    tmc_zone = map_tmc_to_zones(city, speed_dir / "TMC_Identification.csv", demand_dir / "node.csv")
    od_template = build_od_template(demand_dir / "demand.csv", config)
    city_speed = aggregate_city_speed(
        city,
        speed_path,
        influence_hours,
        set(event_hours),
        chunk_size=chunk_size,
    )
    event_zone, event_summary = build_event_zone_footprints(
        city,
        city_events,
        city_speed,
        tmc_zone,
        od_template,
        event_hour_frame,
    )
    baseline_summary = {
        "city": city,
        "speed_file": speed_path.name,
        "positive_event_count": int(len(city_events)),
        "event_hour_count": int(len(event_hours)),
        "rain_influence_hour_count": int(len(influence_hours)),
        **city_speed["baseline_summary"],
        **mapping_summary(tmc_zone),
    }
    return {
        "event_zone": event_zone,
        "event_summary": event_summary,
        "tmc_zone_mapping": tmc_zone,
        "baseline_summary": baseline_summary,
    }


def find_speed_csv(speed_dir: Path) -> Path | None:
    excluded = {"rainfall.csv", "resilience_index.csv", "TMC_Identification.csv"}
    candidates = [
        path
        for path in speed_dir.glob("*.csv")
        if path.name not in excluded and not path.name.lower().startswith("contents")
    ]
    candidates = [path for path in candidates if path.stat().st_size > 1024]
    return max(candidates, key=lambda path: path.stat().st_size) if candidates else None


def build_influence_hours(events: pd.DataFrame) -> set[pd.Timestamp]:
    hours: set[pd.Timestamp] = set()
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start) - pd.Timedelta(hours=PRE_EVENT_HOURS)
        end = pd.Timestamp(event.event_end) + pd.Timedelta(hours=RAIN_INFLUENCE_HOURS)
        for hour in pd.date_range(start.floor("h"), end.floor("h"), freq="h"):
            hours.add(pd.Timestamp(hour))
    return hours


def build_event_hours(events: pd.DataFrame) -> set[pd.Timestamp]:
    hours: set[pd.Timestamp] = set()
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start).floor("h")
        end = pd.Timestamp(event.event_end).floor("h") + pd.Timedelta(hours=WINDOW_HOURS)
        for hour in pd.date_range(start, end, freq="h"):
            hours.add(pd.Timestamp(hour))
    return hours


def build_event_hour_frame(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        start = pd.Timestamp(event.event_start).floor("h")
        end = pd.Timestamp(event.event_end).floor("h") + pd.Timedelta(hours=WINDOW_HOURS)
        for hour in pd.date_range(start, end, freq="h"):
            rows.append(
                {
                    "city": event.city,
                    "event_id": int(event.event_id),
                    "hour": pd.Timestamp(hour),
                    "event_start": pd.Timestamp(event.event_start),
                    "event_end": pd.Timestamp(event.event_end),
                    "event_total_precip": float(event.total_precip),
                    "event_peak_precip": float(event.peak_precip),
                    "event_peak_positive_abnormal_deficit": float(event.peak_positive_abnormal_deficit),
                }
            )
    return pd.DataFrame(rows)


def map_tmc_to_zones(city: str, tmc_path: Path, node_path: Path) -> pd.DataFrame:
    tmc = pd.read_csv(
        tmc_path,
        usecols=lambda col: col
        in {
            "tmc",
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "miles",
        },
    )
    tmc["tmc"] = tmc["tmc"].astype(str)
    for col in ["start_latitude", "start_longitude", "end_latitude", "end_longitude", "miles"]:
        tmc[col] = pd.to_numeric(tmc[col], errors="coerce")
    tmc["tmc_mid_lat"] = (tmc["start_latitude"] + tmc["end_latitude"]) / 2.0
    tmc["tmc_mid_lon"] = (tmc["start_longitude"] + tmc["end_longitude"]) / 2.0
    tmc["tmc_miles"] = tmc["miles"].fillna(0.0).clip(lower=0.001)
    tmc = tmc.dropna(subset=["tmc_mid_lat", "tmc_mid_lon"]).copy()

    nodes = pd.read_csv(node_path, usecols=lambda col: col in {"zone_id", "x_coord", "y_coord"})
    nodes = nodes.dropna(subset=["zone_id", "x_coord", "y_coord"]).copy()
    nodes["zone_id"] = normalize_zone_ids(nodes["zone_id"]).astype(str)
    nodes["x_coord"] = pd.to_numeric(nodes["x_coord"], errors="coerce")
    nodes["y_coord"] = pd.to_numeric(nodes["y_coord"], errors="coerce")
    nodes = nodes.dropna(subset=["x_coord", "y_coord"])
    centroids = (
        nodes.groupby("zone_id", as_index=False)
        .agg(zone_lon=("x_coord", "mean"), zone_lat=("y_coord", "mean"), zone_node_count=("zone_id", "size"))
        .sort_values("zone_id")
    )
    if centroids.empty:
        raise ValueError(f"No OD zone centroids could be built for {city}.")

    tree = cKDTree(centroids[["zone_lat", "zone_lon"]].to_numpy(dtype=float))
    dist_degrees, indices = tree.query(tmc[["tmc_mid_lat", "tmc_mid_lon"]].to_numpy(dtype=float), k=1)
    mapped = tmc.reset_index(drop=True).copy()
    nearest = centroids.iloc[indices].reset_index(drop=True)
    mapped["city"] = city
    mapped["zone_id"] = nearest["zone_id"].astype(str)
    mapped["zone_lat"] = nearest["zone_lat"].to_numpy(dtype=float)
    mapped["zone_lon"] = nearest["zone_lon"].to_numpy(dtype=float)
    mapped["zone_node_count"] = nearest["zone_node_count"].to_numpy(dtype=int)
    mapped["nearest_zone_distance_km"] = haversine_km(
        mapped["tmc_mid_lat"].to_numpy(dtype=float),
        mapped["tmc_mid_lon"].to_numpy(dtype=float),
        mapped["zone_lat"].to_numpy(dtype=float),
        mapped["zone_lon"].to_numpy(dtype=float),
    )
    return mapped[
        [
            "city",
            "tmc",
            "zone_id",
            "tmc_mid_lat",
            "tmc_mid_lon",
            "tmc_miles",
            "zone_lat",
            "zone_lon",
            "zone_node_count",
            "nearest_zone_distance_km",
        ]
    ]


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_od_template(demand_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    demand = pd.read_csv(demand_path, usecols=lambda col: col in {"o_zone_id", "d_zone_id", "volume"})
    demand["volume"] = pd.to_numeric(demand["volume"], errors="coerce").fillna(0.0)
    demand = demand[demand["volume"] > 0].copy()
    demand["o_zone_id"] = normalize_zone_ids(demand["o_zone_id"]).astype(str)
    demand["d_zone_id"] = normalize_zone_ids(demand["d_zone_id"]).astype(str)
    selected_units = select_units(demand, config["calibration"])
    vulnerability = destination_vulnerability_for_event(demand, selected_units)
    blend = float(config["calibration"]["vulnerability_blend"])
    relative = (1.0 - blend) + blend * vulnerability
    relative_sum = max(float(np.sum(relative)), EPS)
    dest = demand[demand["d_zone_id"].isin(selected_units)].groupby("d_zone_id")["volume"].sum()
    origin = demand[demand["o_zone_id"].isin(selected_units)].groupby("o_zone_id")["volume"].sum()
    template = pd.DataFrame(
        {
            "zone_id": pd.Index(selected_units, dtype=str),
            "od_template_weight": relative / relative_sum,
            "destination_volume": dest.reindex(selected_units, fill_value=0.0).to_numpy(dtype=float),
            "origin_volume": origin.reindex(selected_units, fill_value=0.0).to_numpy(dtype=float),
        }
    )
    template["od_template_rank_pct"] = rank_pct(template["od_template_weight"].to_numpy(dtype=float))
    return template


def aggregate_city_speed(
    city: str,
    speed_path: Path,
    influence_hours: set[pd.Timestamp],
    event_hours: set[pd.Timestamp],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    header = read_header(speed_path)
    denom = "historical_average_speed" if "historical_average_speed" in header else "reference_speed"
    usecols = {"tmc_code", "measurement_tstamp", "speed", denom}

    baseline_how_sum: pd.Series | None = None
    baseline_how_count: pd.Series | None = None
    baseline_hod_sum: pd.Series | None = None
    baseline_hod_count: pd.Series | None = None
    city_how_sum: pd.Series | None = None
    city_how_count: pd.Series | None = None
    city_hod_sum: pd.Series | None = None
    city_hod_count: pd.Series | None = None
    event_sum: pd.Series | None = None
    event_count: pd.Series | None = None
    baseline_total_sum = 0.0
    baseline_total_count = 0
    valid_rows = 0
    event_candidate_rows = 0

    influence_index = pd.DatetimeIndex(sorted(influence_hours)) if influence_hours else pd.DatetimeIndex([])
    event_index = pd.DatetimeIndex(sorted(event_hours)) if event_hours else pd.DatetimeIndex([])

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            speed_path,
            usecols=lambda col: col in usecols,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines="skip",
        ),
        start=1,
    ):
        ts = pd.to_datetime(chunk.get("measurement_tstamp"), errors="coerce").dt.floor("h")
        speed = pd.to_numeric(chunk.get("speed"), errors="coerce")
        baseline = pd.to_numeric(chunk.get(denom), errors="coerce")
        tmc = chunk.get("tmc_code").astype(str)
        valid = ts.notna() & speed.notna() & baseline.notna() & (baseline > 0) & (tmc.str.lower() != "nan")
        if not valid.any():
            continue
        ratio = (speed.loc[valid] / baseline.loc[valid]).clip(lower=0.0, upper=3.0)
        deficit = (1.0 - ratio).clip(lower=0.0, upper=1.0)
        frame = pd.DataFrame(
            {
                "tmc": tmc.loc[valid].to_numpy(dtype=str),
                "hour": ts.loc[valid].to_numpy(),
                "deficit": deficit.to_numpy(dtype=float),
            }
        )
        frame["hour"] = pd.to_datetime(frame["hour"])
        frame["hour_of_week"] = frame["hour"].dt.dayofweek * 24 + frame["hour"].dt.hour
        frame["hour_of_day"] = frame["hour"].dt.hour
        valid_rows += int(len(frame))

        baseline_mask = ~frame["hour"].isin(influence_index)
        baseline_frame = frame.loc[baseline_mask]
        if not baseline_frame.empty:
            baseline_total_sum += float(baseline_frame["deficit"].sum())
            baseline_total_count += int(len(baseline_frame))
            grouped = baseline_frame.groupby(["tmc", "hour_of_week"])["deficit"].agg(["sum", "count"])
            baseline_how_sum = add_series(baseline_how_sum, grouped["sum"])
            baseline_how_count = add_series(baseline_how_count, grouped["count"])
            grouped = baseline_frame.groupby(["tmc", "hour_of_day"])["deficit"].agg(["sum", "count"])
            baseline_hod_sum = add_series(baseline_hod_sum, grouped["sum"])
            baseline_hod_count = add_series(baseline_hod_count, grouped["count"])
            grouped = baseline_frame.groupby("hour_of_week")["deficit"].agg(["sum", "count"])
            city_how_sum = add_series(city_how_sum, grouped["sum"])
            city_how_count = add_series(city_how_count, grouped["count"])
            grouped = baseline_frame.groupby("hour_of_day")["deficit"].agg(["sum", "count"])
            city_hod_sum = add_series(city_hod_sum, grouped["sum"])
            city_hod_count = add_series(city_hod_count, grouped["count"])

        event_mask = frame["hour"].isin(event_index)
        event_frame = frame.loc[event_mask]
        if not event_frame.empty:
            event_candidate_rows += int(len(event_frame))
            grouped = event_frame.groupby(["tmc", "hour", "hour_of_week", "hour_of_day"])["deficit"].agg(["sum", "count"])
            event_sum = add_series(event_sum, grouped["sum"])
            event_count = add_series(event_count, grouped["count"])

        if chunk_idx % 25 == 0:
            print(
                f"  {city}: processed chunk {chunk_idx}, valid rows={valid_rows:,}, event rows={event_candidate_rows:,}",
                flush=True,
            )

    event_hourly = series_pair_to_frame(
        event_sum,
        event_count,
        ["tmc", "hour", "hour_of_week", "hour_of_day"],
        "deficit_sum",
        "observation_count",
    )
    baseline_how = series_pair_to_frame(
        baseline_how_sum,
        baseline_how_count,
        ["tmc", "hour_of_week"],
        "baseline_how_sum",
        "baseline_how_count",
    )
    baseline_hod = series_pair_to_frame(
        baseline_hod_sum,
        baseline_hod_count,
        ["tmc", "hour_of_day"],
        "baseline_hod_sum",
        "baseline_hod_count",
    )
    city_how = series_pair_to_frame(
        city_how_sum,
        city_how_count,
        ["hour_of_week"],
        "city_how_sum",
        "city_how_count",
    )
    city_hod = series_pair_to_frame(
        city_hod_sum,
        city_hod_count,
        ["hour_of_day"],
        "city_hod_sum",
        "city_hod_count",
    )
    global_baseline = baseline_total_sum / max(float(baseline_total_count), EPS)
    return {
        "event_hourly": event_hourly,
        "baseline_how": baseline_how,
        "baseline_hod": baseline_hod,
        "city_how": city_how,
        "city_hod": city_hod,
        "global_baseline": float(global_baseline),
        "baseline_summary": {
            "valid_speed_rows": int(valid_rows),
            "event_candidate_speed_rows": int(event_candidate_rows),
            "strict_baseline_rows": int(baseline_total_count),
            "strict_baseline_mean_deficit": float(global_baseline),
            "event_tmc_hour_rows": int(len(event_hourly)),
            "baseline_tmc_hour_of_week_rows": int(len(baseline_how)),
            "baseline_tmc_hour_of_day_rows": int(len(baseline_hod)),
        },
    }


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def add_series(current: pd.Series | None, new_values: pd.Series) -> pd.Series:
    if current is None:
        return new_values.copy()
    return current.add(new_values, fill_value=0.0)


def series_pair_to_frame(
    sum_series: pd.Series | None,
    count_series: pd.Series | None,
    names: list[str],
    sum_name: str,
    count_name: str,
) -> pd.DataFrame:
    if sum_series is None or count_series is None:
        return pd.DataFrame(columns=[*names, sum_name, count_name])
    frame = pd.DataFrame({sum_name: sum_series, count_name: count_series}).reset_index()
    frame.columns = [*names, sum_name, count_name]
    return frame


def build_event_zone_footprints(
    city: str,
    city_events: pd.DataFrame,
    city_speed: dict[str, Any],
    tmc_zone: pd.DataFrame,
    od_template: pd.DataFrame,
    event_hour_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_hourly = city_speed["event_hourly"].copy()
    if event_hourly.empty:
        return pd.DataFrame(), pd.DataFrame()
    event_hourly["mean_deficit"] = event_hourly["deficit_sum"] / event_hourly["observation_count"].replace(0, np.nan)
    baseline_how = city_speed["baseline_how"].copy()
    baseline_hod = city_speed["baseline_hod"].copy()
    city_how = city_speed["city_how"].copy()
    city_hod = city_speed["city_hod"].copy()
    for frame, sum_col, count_col, out_col in [
        (baseline_how, "baseline_how_sum", "baseline_how_count", "expected_tmc_how"),
        (baseline_hod, "baseline_hod_sum", "baseline_hod_count", "expected_tmc_hod"),
        (city_how, "city_how_sum", "city_how_count", "expected_city_how"),
        (city_hod, "city_hod_sum", "city_hod_count", "expected_city_hod"),
    ]:
        if not frame.empty:
            frame[out_col] = frame[sum_col] / frame[count_col].replace(0, np.nan)

    event_hourly = event_hourly.merge(
        baseline_how[["tmc", "hour_of_week", "expected_tmc_how"]],
        on=["tmc", "hour_of_week"],
        how="left",
    )
    event_hourly = event_hourly.merge(
        baseline_hod[["tmc", "hour_of_day", "expected_tmc_hod"]],
        on=["tmc", "hour_of_day"],
        how="left",
    )
    event_hourly = event_hourly.merge(
        city_how[["hour_of_week", "expected_city_how"]],
        on="hour_of_week",
        how="left",
    )
    event_hourly = event_hourly.merge(
        city_hod[["hour_of_day", "expected_city_hod"]],
        on="hour_of_day",
        how="left",
    )
    event_hourly["expected_deficit"] = (
        event_hourly["expected_tmc_how"]
        .fillna(event_hourly["expected_tmc_hod"])
        .fillna(event_hourly["expected_city_how"])
        .fillna(event_hourly["expected_city_hod"])
        .fillna(float(city_speed["global_baseline"]))
    )
    event_hourly["positive_abnormal_deficit"] = (
        event_hourly["mean_deficit"] - event_hourly["expected_deficit"]
    ).clip(lower=0.0)
    event_hourly = event_hourly[event_hourly["positive_abnormal_deficit"] > 0].copy()
    if event_hourly.empty:
        return pd.DataFrame(), empty_event_summary(city_events)

    event_hourly = event_hourly.merge(
        tmc_zone[["tmc", "zone_id", "tmc_miles", "nearest_zone_distance_km"]],
        on="tmc",
        how="inner",
    )
    event_hourly["sample_completeness"] = (event_hourly["observation_count"] / 6.0).clip(lower=0.0, upper=1.0)
    event_hourly["tmc_signal"] = (
        event_hourly["positive_abnormal_deficit"]
        * event_hourly["tmc_miles"]
        * event_hourly["sample_completeness"]
    )
    event_hourly = event_hourly[event_hourly["tmc_signal"] > 0].copy()
    event_hourly = event_hourly.merge(event_hour_frame, on="hour", how="inner", suffixes=("", "_event"))
    event_hourly = event_hourly[event_hourly["city_event"] == city] if "city_event" in event_hourly.columns else event_hourly
    if "city_y" in event_hourly.columns:
        event_hourly = event_hourly.rename(columns={"city_y": "city"})
    if "city_x" in event_hourly.columns:
        event_hourly = event_hourly.drop(columns=["city_x"])

    event_zone = (
        event_hourly.groupby(["city", "event_id", "zone_id"], as_index=False)
        .agg(
            zone_signal=("tmc_signal", "sum"),
            peak_tmc_positive_abnormal_deficit=("positive_abnormal_deficit", "max"),
            mean_tmc_positive_abnormal_deficit=("positive_abnormal_deficit", "mean"),
            contributing_tmc_count=("tmc", "nunique"),
            contributing_tmc_hour_count=("tmc", "size"),
            mean_mapping_distance_km=("nearest_zone_distance_km", "mean"),
        )
        .sort_values(["city", "event_id", "zone_signal"], ascending=[True, True, False])
    )
    totals = event_zone.groupby(["city", "event_id"])["zone_signal"].transform("sum")
    event_zone["zone_weight"] = event_zone["zone_signal"] / totals.replace(0.0, np.nan)
    event_zone = event_zone.merge(od_template, on="zone_id", how="left")
    event_zone["od_template_weight"] = event_zone["od_template_weight"].fillna(0.0)
    event_zone["od_template_rank_pct"] = event_zone["od_template_rank_pct"].fillna(0.0)

    event_summary = summarize_event_footprints(city_events, event_zone, od_template)
    return event_zone, event_summary


def empty_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        rows.append(
            {
                "city": event.city,
                "event_id": int(event.event_id),
                "event_start": str(event.event_start),
                "event_total_precip": float(event.total_precip),
                "event_peak_precip": float(event.peak_precip),
                "event_peak_positive_abnormal_deficit": float(event.peak_positive_abnormal_deficit),
                "footprint_available": False,
            }
        )
    return pd.DataFrame(rows)


def summarize_event_footprints(
    events: pd.DataFrame,
    event_zone: pd.DataFrame,
    od_template: pd.DataFrame,
) -> pd.DataFrame:
    template_all = od_template[["zone_id", "od_template_weight", "od_template_rank_pct"]].copy()
    template_all["zone_id"] = template_all["zone_id"].astype(str)
    n_units = int(len(template_all))
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        group = event_zone[(event_zone["city"] == event.city) & (event_zone["event_id"] == int(event.event_id))].copy()
        if group.empty:
            rows.append(
                {
                    "city": event.city,
                    "event_id": int(event.event_id),
                    "event_start": str(event.event_start),
                    "event_total_precip": float(event.total_precip),
                    "event_peak_precip": float(event.peak_precip),
                    "event_peak_positive_abnormal_deficit": float(event.peak_positive_abnormal_deficit),
                    "footprint_available": False,
                }
            )
            continue
        footprint = template_all.merge(group[["zone_id", "zone_weight", "zone_signal"]], on="zone_id", how="left")
        footprint["zone_weight"] = footprint["zone_weight"].fillna(0.0)
        footprint["zone_signal"] = footprint["zone_signal"].fillna(0.0)
        footprint_weight = footprint["zone_weight"].to_numpy(dtype=float)
        template_weight = footprint["od_template_weight"].to_numpy(dtype=float)
        top10_foot = top_set(footprint, "zone_weight", 10)
        top20_foot = top_set(footprint, "zone_weight", 20)
        top10_template = top_set(footprint, "od_template_weight", 10)
        top20_template = top_set(footprint, "od_template_weight", 20)
        top5pct_n = max(1, int(math.ceil(0.05 * n_units)))
        top5pct_foot = top_set(footprint, "zone_weight", top5pct_n)
        top5pct_template = top_set(footprint, "od_template_weight", top5pct_n)
        rows.append(
            {
                "city": event.city,
                "event_id": int(event.event_id),
                "event_start": str(event.event_start),
                "event_total_precip": float(event.total_precip),
                "event_peak_precip": float(event.peak_precip),
                "event_peak_positive_abnormal_deficit": float(event.peak_positive_abnormal_deficit),
                "footprint_available": True,
                "n_units": n_units,
                "n_footprint_zones": int((footprint_weight > 0).sum()),
                "footprint_signal_total": float(group["zone_signal"].sum()),
                "footprint_zone_hhi": float(np.sum(footprint_weight**2)),
                "footprint_effective_zone_count": float(1.0 / max(np.sum(footprint_weight**2), EPS)),
                "footprint_gini": gini(footprint_weight),
                "footprint_top_1pct_zone_share": top_share(footprint_weight, 0.01),
                "footprint_top_5pct_zone_share": top_share(footprint_weight, 0.05),
                "footprint_top_10pct_zone_share": top_share(footprint_weight, 0.10),
                "template_top_5pct_captures_footprint_share": float(footprint.loc[footprint["zone_id"].isin(top5pct_template), "zone_weight"].sum()),
                "footprint_top_5pct_template_mass": float(footprint.loc[footprint["zone_id"].isin(top5pct_foot), "od_template_weight"].sum()),
                "footprint_template_cosine": cosine_similarity(footprint_weight, template_weight),
                "footprint_template_spearman": safe_corr(footprint["zone_weight"], footprint["od_template_weight"], method="spearman"),
                "footprint_template_pearson": safe_corr(footprint["zone_weight"], footprint["od_template_weight"], method="pearson"),
                "top10_footprint_template_jaccard": jaccard(top10_foot, top10_template),
                "top20_footprint_template_jaccard": jaccard(top20_foot, top20_template),
                "top5pct_footprint_template_jaccard": jaccard(top5pct_foot, top5pct_template),
                "mean_mapping_distance_km": float(group["mean_mapping_distance_km"].mean()),
                "max_mapping_distance_km": float(group["mean_mapping_distance_km"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["city", "event_start", "event_id"])


def top_set(frame: pd.DataFrame, column: str, n: int) -> set[str]:
    return set(frame.sort_values(column, ascending=False).head(max(1, int(n)))["zone_id"].astype(str))


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else np.nan


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > EPS else np.nan


def safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return np.nan
    return float(pair["a"].corr(pair["b"], method=method))


def build_pairwise_footprint_stability(
    event_zone: pd.DataFrame,
    top_k_values: tuple[int, ...] = (10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if event_zone.empty:
        return pd.DataFrame(), pd.DataFrame()
    sets: dict[tuple[str, int, int], set[str]] = {}
    for (city, event_id), group in event_zone.groupby(["city", "event_id"]):
        ordered = group.sort_values("zone_weight", ascending=False)
        for top_k in top_k_values:
            sets[(str(city), int(event_id), top_k)] = set(ordered.head(top_k)["zone_id"].astype(str))
    rows = []
    for city, city_events in event_zone[["city", "event_id"]].drop_duplicates().groupby("city"):
        event_ids = sorted(int(value) for value in city_events["event_id"].unique())
        for top_k in top_k_values:
            for event_a, event_b in combinations(event_ids, 2):
                set_a = sets.get((str(city), event_a, top_k), set())
                set_b = sets.get((str(city), event_b, top_k), set())
                union = set_a | set_b
                inter = set_a & set_b
                rows.append(
                    {
                        "city": city,
                        "top_k_zones": top_k,
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
        pairwise.groupby(["city", "top_k_zones"], as_index=False)
        .agg(
            n_event_pairs=("jaccard", "count"),
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            min_jaccard=("jaccard", "min"),
            max_jaccard=("jaccard", "max"),
        )
        .sort_values(["top_k_zones", "city"])
    )
    return pairwise, summary


def build_city_summary(
    event_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    available = event_summary[event_summary["footprint_available"] == True].copy()
    rows = []
    for city, group in available.groupby("city", sort=True):
        row = {
            "city": city,
            "n_events": int(group["event_id"].nunique()),
            "mean_n_footprint_zones": float(group["n_footprint_zones"].mean()),
            "mean_footprint_effective_zone_count": float(group["footprint_effective_zone_count"].mean()),
            "mean_footprint_top_5pct_zone_share": float(group["footprint_top_5pct_zone_share"].mean()),
            "mean_template_top_5pct_captures_footprint_share": float(group["template_top_5pct_captures_footprint_share"].mean()),
            "mean_footprint_template_cosine": float(group["footprint_template_cosine"].mean()),
            "mean_top20_footprint_template_jaccard": float(group["top20_footprint_template_jaccard"].mean()),
            "std_footprint_template_cosine": float(group["footprint_template_cosine"].std(ddof=0)),
        }
        for top_k in [10, 20]:
            match = pairwise_summary[
                (pairwise_summary["city"] == city) & (pairwise_summary["top_k_zones"] == top_k)
            ]
            if not match.empty:
                row[f"top{top_k}_within_city_pairwise_mean_jaccard"] = float(match.iloc[0]["mean_jaccard"])
                row[f"top{top_k}_within_city_pairwise_min_jaccard"] = float(match.iloc[0]["min_jaccard"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if not baseline_summary.empty:
        keep = [
            col
            for col in [
                "city",
                "tmc_count",
                "mapped_within_5km_share",
                "mapped_within_10km_share",
                "median_mapping_distance_km",
                "p95_mapping_distance_km",
                "strict_baseline_rows",
                "event_tmc_hour_rows",
            ]
            if col in baseline_summary.columns
        ]
        out = out.merge(baseline_summary[keep], on="city", how="left")
    return out.sort_values("mean_footprint_template_cosine")


def build_metrics(
    event_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> dict[str, Any]:
    available = event_summary[event_summary.get("footprint_available", False) == True].copy() if not event_summary.empty else pd.DataFrame()
    top20_pairs = pairwise_summary[pairwise_summary.get("top_k_zones", pd.Series(dtype=int)) == 20].copy() if not pairwise_summary.empty else pd.DataFrame()
    return {
        "n_events_with_footprint": int(len(available)),
        "n_cities_with_footprint": int(available["city"].nunique()) if not available.empty else 0,
        "mean_footprint_template_cosine": float(available["footprint_template_cosine"].mean()) if not available.empty else np.nan,
        "median_footprint_template_cosine": float(available["footprint_template_cosine"].median()) if not available.empty else np.nan,
        "mean_top20_footprint_template_jaccard": float(available["top20_footprint_template_jaccard"].mean()) if not available.empty else np.nan,
        "mean_template_top_5pct_captures_footprint_share": float(available["template_top_5pct_captures_footprint_share"].mean()) if not available.empty else np.nan,
        "mean_footprint_top_5pct_zone_share": float(available["footprint_top_5pct_zone_share"].mean()) if not available.empty else np.nan,
        "mean_footprint_effective_zone_count": float(available["footprint_effective_zone_count"].mean()) if not available.empty else np.nan,
        "mean_top20_within_city_footprint_jaccard": float(top20_pairs["mean_jaccard"].mean()) if not top20_pairs.empty else np.nan,
        "min_city_top20_within_city_footprint_jaccard": float(top20_pairs["mean_jaccard"].min()) if not top20_pairs.empty else np.nan,
        "max_city_top20_within_city_footprint_jaccard": float(top20_pairs["mean_jaccard"].max()) if not top20_pairs.empty else np.nan,
        "mean_mapped_within_10km_share": float(baseline_summary["mapped_within_10km_share"].mean()) if "mapped_within_10km_share" in baseline_summary else np.nan,
        "lowest_template_alignment_city": str(city_summary.iloc[0]["city"]) if not city_summary.empty else "",
    }


def mapping_summary(tmc_zone: pd.DataFrame) -> dict[str, Any]:
    dist = tmc_zone["nearest_zone_distance_km"].to_numpy(dtype=float)
    miles = tmc_zone["tmc_miles"].to_numpy(dtype=float)
    return {
        "tmc_count": int(len(tmc_zone)),
        "mapped_within_2km_share": float(np.mean(dist <= 2.0)) if len(dist) else np.nan,
        "mapped_within_5km_share": float(np.mean(dist <= 5.0)) if len(dist) else np.nan,
        "mapped_within_10km_share": float(np.mean(dist <= 10.0)) if len(dist) else np.nan,
        "median_mapping_distance_km": float(np.median(dist)) if len(dist) else np.nan,
        "p95_mapping_distance_km": float(np.quantile(dist, 0.95)) if len(dist) else np.nan,
        "total_tmc_miles": float(np.sum(miles)) if len(miles) else np.nan,
    }


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_figures(
    event_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    if event_summary.empty or city_summary.empty:
        return
    available = event_summary[event_summary["footprint_available"] == True].copy()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    order = city_summary["city"].tolist()
    data = [available.loc[available["city"] == city, "footprint_template_cosine"].dropna().to_numpy() for city in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Observed TMC footprint vs OD vulnerability template")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "footprint_template_similarity_by_city.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.scatter(
        available["template_top_5pct_captures_footprint_share"],
        available["footprint_top_5pct_zone_share"],
        c=available["event_peak_positive_abnormal_deficit"],
        cmap="viridis",
        s=45,
        alpha=0.85,
    )
    ax.set_xlabel("Footprint mass captured by top 5% OD-template zones")
    ax.set_ylabel("Footprint top 5% zone share")
    ax.set_title("Footprint concentration is not the same as OD-template capture")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "footprint_concentration_vs_template_capture.png", dpi=180)
    plt.close(fig)

    if not pairwise_summary.empty:
        top20 = pairwise_summary[pairwise_summary["top_k_zones"] == 20].copy()
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.bar(top20["city"], top20["mean_jaccard"], color="#5b8def")
        ax.set_ylabel("Mean pairwise Jaccard")
        ax.set_ylim(0, 1)
        ax.set_title("Within-city stability of top-20 observed footprint zones")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "footprint_within_city_top20_jaccard.png", dpi=180)
        plt.close(fig)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    city_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Event Spatial Footprint Proxy V33",
        "",
        "本版目标是检查 V32 暴露的边界：当前 LP calibration 把城市级事件冲击按 OD vulnerability 投影到 zones，因此 event-level top-tail law 可能只是城市模板信号。这里额外从 raw TMC speed records 构造一个独立的 event-zone speed footprint proxy。",
        "",
        "## 方法",
        "",
        "1. 将每条 TMC segment 的中点映射到最近 OD zone centroid。",
        "2. 对 raw speed records 计算 `deficit=max(0,1-speed/baseline_speed)`。",
        "3. 使用非降雨、非事件影响窗口中的 same TMC + same hour-of-week mean 作为 expected deficit；缺失时退回 same TMC + hour-of-day、city hour-of-week、city hour-of-day、city global baseline。",
        "4. 在每个 rainfall event 的 12 小时窗口内计算 positive abnormal TMC deficit，并按 TMC miles 聚合到 OD zone。",
        "5. 比较 observed footprint distribution 与当前 calibration 使用的 OD vulnerability template。",
        "",
        "## 关键指标",
        "",
        f"- 有 footprint 的事件数：{metrics.get('n_events_with_footprint', 0)}，城市数：{metrics.get('n_cities_with_footprint', 0)}。",
        f"- footprint-template 平均 cosine similarity：{fmt(metrics.get('mean_footprint_template_cosine'))}。",
        f"- top-20 footprint zones 与 top-20 OD-template zones 的平均 Jaccard：{fmt(metrics.get('mean_top20_footprint_template_jaccard'))}。",
        f"- OD-template top 5% zones 平均捕获 observed footprint mass：{fmt(metrics.get('mean_template_top_5pct_captures_footprint_share'))}。",
        f"- observed footprint top 5% zones 平均集中度：{fmt(metrics.get('mean_footprint_top_5pct_zone_share'))}。",
        f"- 城市内 top-20 footprint zones 平均事件间 Jaccard：{fmt(metrics.get('mean_top20_within_city_footprint_jaccard'))}。",
        f"- TMC 到 OD zone 映射中，10km 内平均比例：{fmt(metrics.get('mean_mapped_within_10km_share'))}。",
        "",
        "## 城市摘要",
        "",
        "| city | events | cosine | top20 template Jaccard | template top5 captures footprint | within-city top20 Jaccard | mapped <=10km |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not city_summary.empty:
        for row in city_summary.sort_values("mean_footprint_template_cosine").itertuples(index=False):
            lines.append(
                "| {city} | {events} | {cosine} | {jaccard} | {capture} | {within} | {mapped} |".format(
                    city=row.city,
                    events=int(row.n_events),
                    cosine=fmt(row.mean_footprint_template_cosine),
                    jaccard=fmt(row.mean_top20_footprint_template_jaccard),
                    capture=fmt(row.mean_template_top_5pct_captures_footprint_share),
                    within=fmt(getattr(row, "top20_within_city_pairwise_mean_jaccard", np.nan)),
                    mapped=fmt(getattr(row, "mapped_within_10km_share", np.nan)),
                )
            )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "如果 observed footprint 与 OD vulnerability template 的重合度较低，说明当前 calibration 的空间投影确实过强，下一步应该把 `b0_i` 和 `h_i,t` 升级为 `OD vulnerability + observed TMC footprint` 的混合场。若重合度很高，则说明当前 OD template 已经近似捕捉了事件影响区域，V32 的限制反而没有那么严重。",
            "",
            "这个 proxy 仍然不是最终因果识别：TMC 到 OD zone 是最近邻映射，TMC coverage 与 OD zones 不是完全一致，且 TMC speed abnormal 仍受非降雨因素影响。但它已经比纯城市级投影多了一层事件内空间观测信号。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= EPS:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - (n + 1.0) / n)


def top_share(values: np.ndarray, pct: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= EPS or len(values) == 0:
        return np.nan
    n = max(1, int(math.ceil(len(values) * pct)))
    return float(np.sort(values)[-n:].sum() / total)


def rank_pct(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=float)


if __name__ == "__main__":
    main()

"""Data-mining pipeline for the recoverable urban resilience project."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import find_repo_root


try:
    import yaml
except Exception:  # pragma: no cover - optional fallback
    yaml = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots are skipped if matplotlib is absent
    plt = None


CITY_RE = re.compile(r"^\d+_(?P<name>.+?) city$")


@dataclass(frozen=True)
class CityPaths:
    city: str
    city_dir_name: str
    speed_dir: Path | None
    demand_dir: Path | None


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "raw_data_dir": "data/raw_data",
        "output_dir": "results/data_mining",
    },
    "analysis": {
        "rainfall_threshold_quantile": 0.90,
        "full_speed_temporal_scan": True,
        "max_speed_distribution_rows_per_city": 500_000,
        "chunk_size": 500_000,
        "minimum_observations_per_tmc": 12,
        "event_window_hours": 12,
    },
    "figures": {"dpi": 180},
}


def run_pipeline(config_path: Path, root: Path | None = None) -> None:
    """Run all analysis stages and write compact outputs."""
    repo_root = find_repo_root(root)
    config = load_config(repo_root / config_path if not config_path.is_absolute() else config_path)

    raw_dir = repo_root / config["project"]["raw_data_dir"]
    output_dir = repo_root / config["project"]["output_dir"]
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    for directory in [table_dir, figure_dir, report_dir, repo_root / "data" / "interim"]:
        directory.mkdir(parents=True, exist_ok=True)

    city_paths = discover_city_paths(raw_dir)
    print(f"Discovered {len(city_paths)} city directories.")

    inventory = build_inventory(city_paths)
    write_csv(inventory, table_dir / "city_data_inventory.csv")
    write_inventory_markdown(inventory, repo_root / "docs" / "data_inventory.md")

    rainfall = analyze_rainfall(city_paths, config)
    write_csv(rainfall, table_dir / "rainfall_event_summary.csv")

    speed, hourly, tmc_concentration, tmc_hotspots = analyze_speed(city_paths, config)
    write_csv(speed, table_dir / "speed_deficit_summary.csv")
    write_csv(hourly, table_dir / "speed_hourly_deficit_sample.csv")
    write_csv(tmc_concentration, table_dir / "speed_tmc_deficit_concentration.csv")
    write_csv(tmc_hotspots, table_dir / "speed_tmc_hotspots.csv")

    rain_speed = analyze_rainfall_speed_alignment(hourly, rainfall, city_paths, config)
    write_csv(rain_speed, table_dir / "rainfall_speed_alignment.csv")

    demand = analyze_demand_network(city_paths, config)
    write_csv(demand, table_dir / "demand_network_summary.csv")

    resilience = analyze_existing_resilience(city_paths)
    write_csv(resilience, table_dir / "existing_resilience_index_summary.csv")

    exposure_alignment = analyze_resilience_exposure_alignment(city_paths, config)
    write_csv(exposure_alignment, table_dir / "resilience_exposure_alignment.csv")

    fit = score_idea_fit(inventory, rainfall, speed, rain_speed, demand, resilience)
    write_csv(fit, table_dir / "idea_data_fit_scores.csv")

    make_figures(inventory, rain_speed, demand, fit, figure_dir, config, tmc_concentration, exposure_alignment)
    write_report(
        report_dir / "data_mining_report_zh.md",
        inventory,
        rainfall,
        speed,
        rain_speed,
        demand,
        resilience,
        fit,
        tmc_concentration,
        exposure_alignment,
    )

    print(f"Data-mining outputs written to {output_dir}")


def load_config(path: Path) -> dict[str, Any]:
    config = deep_copy(DEFAULT_CONFIG)
    if path.exists() and yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merge_dict(config, loaded)
    return config


def deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_copy(v) for v in value]
    return value


def merge_dict(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value


def discover_city_paths(raw_dir: Path) -> list[CityPaths]:
    speed_base = raw_dir / "speed"
    demand_base = raw_dir / "demand"
    by_key: dict[str, dict[str, Path | str]] = {}
    for kind, base in [("speed_dir", speed_base), ("demand_dir", demand_base)]:
        if not base.exists():
            continue
        for directory in base.iterdir():
            if not directory.is_dir():
                continue
            city = parse_city_name(directory.name)
            by_key.setdefault(city, {"city_dir_name": directory.name, "speed_dir": None, "demand_dir": None})
            by_key[city][kind] = directory
            by_key[city]["city_dir_name"] = directory.name
    return [
        CityPaths(
            city=city,
            city_dir_name=str(values["city_dir_name"]),
            speed_dir=values.get("speed_dir") if isinstance(values.get("speed_dir"), Path) else None,
            demand_dir=values.get("demand_dir") if isinstance(values.get("demand_dir"), Path) else None,
        )
        for city, values in sorted(by_key.items(), key=lambda item: city_sort_key(item[1]["city_dir_name"]))
    ]


def city_sort_key(name: Any) -> tuple[int, str]:
    match = re.match(r"^(?P<idx>\d+)_", str(name))
    return (int(match.group("idx")) if match else 999, str(name))


def parse_city_name(name: str) -> str:
    match = CITY_RE.match(name)
    return match.group("name") if match else name


def build_inventory(city_paths: list[CityPaths]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        speed_csv = find_speed_csv(cp.speed_dir)
        rainfall_csv = cp.speed_dir / "rainfall.csv" if cp.speed_dir else None
        tmc_csv = cp.speed_dir / "TMC_Identification.csv" if cp.speed_dir else None
        resilience_csv = cp.speed_dir / "resilience_index.csv" if cp.speed_dir else None
        demand_csv = cp.demand_dir / "demand.csv" if cp.demand_dir else None
        link_perf = cp.demand_dir / "link_performance.csv" if cp.demand_dir else None
        link_summary = cp.demand_dir / "link_performance_summary.csv" if cp.demand_dir else None
        node_csv = cp.demand_dir / "node.csv" if cp.demand_dir else None
        link_csv = cp.demand_dir / "link.csv" if cp.demand_dir else None
        speed_csv_present = exists(speed_csv)
        speed_csv_usable = speed_csv_present and file_size(speed_csv) > 1024
        rows.append(
            {
                "city": cp.city,
                "speed_dir": exists(cp.speed_dir),
                "demand_dir": exists(cp.demand_dir),
                "speed_csv": speed_csv_usable,
                "speed_csv_present": speed_csv_present,
                "speed_csv_name": speed_csv.name if speed_csv else "",
                "speed_csv_size_gb": round(file_size(speed_csv) / 1024**3, 3) if speed_csv else 0.0,
                "rainfall_csv": exists(rainfall_csv),
                "tmc_identification_csv": exists(tmc_csv),
                "resilience_index_csv": exists(resilience_csv),
                "demand_csv": exists(demand_csv),
                "link_performance_csv": exists(link_perf),
                "link_performance_summary_csv": exists(link_summary),
                "node_csv": exists(node_csv),
                "link_csv": exists(link_csv),
                "demand_dir_size_mb": round(directory_size(cp.demand_dir) / 1024**2, 1) if cp.demand_dir else 0.0,
            }
        )
    return pd.DataFrame(rows)


def exists(path: Path | None) -> bool:
    return bool(path and path.exists())


def file_size(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def directory_size(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def find_speed_csv(speed_dir: Path | None) -> Path | None:
    if not speed_dir or not speed_dir.exists():
        return None
    excluded = {"rainfall.csv", "resilience_index.csv", "TMC_Identification.csv"}
    candidates = [
        path
        for path in speed_dir.glob("*.csv")
        if path.name not in excluded and not path.name.lower().startswith("contents")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def analyze_rainfall(city_paths: list[CityPaths], config: dict[str, Any]) -> pd.DataFrame:
    quantile = float(config["analysis"]["rainfall_threshold_quantile"])
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        path = cp.speed_dir / "rainfall.csv" if cp.speed_dir else None
        if not path or not path.exists():
            rows.append({"city": cp.city, "rainfall_available": False})
            continue
        try:
            rain = pd.read_csv(path)
            timestamp_col = "Timestamp" if "Timestamp" in rain.columns else find_first_datetime_col(rain.columns)
            rain[timestamp_col] = pd.to_datetime(rain[timestamp_col], errors="coerce")
            precip_col = "precipitation" if "precipitation" in rain.columns else find_precip_col(rain.columns)
            precip = pd.to_numeric(rain[precip_col], errors="coerce").fillna(0)
            threshold = float(precip.quantile(quantile)) if len(precip) else np.nan
            selected = precip >= threshold if np.isfinite(threshold) else pd.Series(False, index=rain.index)
            positive = precip > 0
            rows.append(
                {
                    "city": cp.city,
                    "rainfall_available": True,
                    "rainfall_rows": int(len(rain)),
                    "rain_start": safe_date(rain[timestamp_col].min()),
                    "rain_end": safe_date(rain[timestamp_col].max()),
                    "precip_mean": float(precip.mean()),
                    "precip_p90": float(precip.quantile(0.90)),
                    "precip_p95": float(precip.quantile(0.95)),
                    "precip_max": float(precip.max()),
                    "rainy_hour_share": float(positive.mean()),
                    "event_threshold": threshold,
                    "event_hour_count": int(selected.sum()),
                    "top_event_timestamp": safe_date(rain.loc[precip.idxmax(), timestamp_col]) if len(rain) else "",
                }
            )
        except Exception as exc:
            rows.append({"city": cp.city, "rainfall_available": True, "rainfall_error": str(exc)})
    return pd.DataFrame(rows)


def find_first_datetime_col(columns: pd.Index) -> str:
    for column in columns:
        if "time" in column.lower() or "date" in column.lower():
            return column
    return str(columns[0])


def find_precip_col(columns: pd.Index) -> str:
    for column in columns:
        if "precip" in column.lower():
            return column
    return str(columns[1])


def safe_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def analyze_speed(
    city_paths: list[CityPaths], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_sample_rows = int(config["analysis"].get("max_speed_distribution_rows_per_city", 500_000))
    full_scan = bool(config["analysis"].get("full_speed_temporal_scan", True))
    chunk_size = int(config["analysis"]["chunk_size"])
    seed = int(config["analysis"].get("random_seed", 20260526))
    rows: list[dict[str, Any]] = []
    hourly_rows: list[pd.DataFrame] = []
    concentration_rows: list[dict[str, Any]] = []
    hotspot_rows: list[pd.DataFrame] = []
    for cp in city_paths:
        path = find_speed_csv(cp.speed_dir)
        if not path:
            rows.append({"city": cp.city, "speed_available": False})
            concentration_rows.append({"city": cp.city, "tmc_concentration_available": False})
            continue
        print(f"Analyzing speed sample for {cp.city}: {path.name}")
        try:
            summary, hourly, concentration, hotspots = analyze_speed_file(
                cp.city,
                path,
                max_sample_rows=max_sample_rows,
                chunk_size=chunk_size,
                full_scan=full_scan,
                seed=seed + sum(ord(char) for char in cp.city),
            )
            rows.append(summary)
            concentration_rows.append(concentration)
            if not hourly.empty:
                hourly_rows.append(hourly)
            if not hotspots.empty:
                hotspot_rows.append(hotspots)
        except Exception as exc:
            rows.append({"city": cp.city, "speed_available": True, "speed_error": str(exc)})
            concentration_rows.append({"city": cp.city, "tmc_concentration_available": False, "tmc_error": str(exc)})
    hourly_all = pd.concat(hourly_rows, ignore_index=True) if hourly_rows else pd.DataFrame()
    hotspots_all = pd.concat(hotspot_rows, ignore_index=True) if hotspot_rows else pd.DataFrame()
    return pd.DataFrame(rows), hourly_all, pd.DataFrame(concentration_rows), hotspots_all


def analyze_speed_file(
    city: str,
    path: Path,
    max_sample_rows: int,
    chunk_size: int,
    full_scan: bool,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    header = read_header(path)
    denom = "historical_average_speed" if "historical_average_speed" in header else "reference_speed"
    usecols = {"tmc_code", "measurement_tstamp", "speed", denom, "reference_speed", "confidence_score"}

    rng = np.random.default_rng(seed)
    processed_raw = 0
    processed_valid = 0
    ratio_sum = 0.0
    deficit_sum = 0.0
    severe_20_count = 0
    severe_40_count = 0
    tmc_values: set[str] = set()
    deficit_sample = np.array([], dtype=float)
    ratio_sample = np.array([], dtype=float)
    hourly_parts: list[pd.DataFrame] = []
    tmc_deficit: pd.Series | None = None
    tmc_obs: pd.Series | None = None
    tmc_severe: pd.Series | None = None
    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None

    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in usecols,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if not full_scan and max_sample_rows and processed_raw + len(chunk) > max_sample_rows:
            chunk = chunk.iloc[: max_sample_rows - processed_raw].copy()
        if chunk.empty:
            break

        speed = pd.to_numeric(chunk.get("speed"), errors="coerce")
        baseline = pd.to_numeric(chunk.get(denom), errors="coerce")
        valid = speed.notna() & baseline.notna() & (baseline > 0)
        ratio = (speed[valid] / baseline[valid]).clip(lower=0, upper=3)
        deficit = (1 - ratio).clip(lower=0, upper=1)
        ratio_values = ratio.to_numpy(dtype=float)
        deficit_values = deficit.to_numpy(dtype=float)
        processed_valid += int(len(deficit_values))
        ratio_sum += float(np.nansum(ratio_values))
        deficit_sum += float(np.nansum(deficit_values))
        severe_20_count += int(np.sum(deficit_values >= 0.20))
        severe_40_count += int(np.sum(deficit_values >= 0.40))
        ratio_sample = merge_random_sample(ratio_sample, ratio_values, max_sample_rows, rng)
        deficit_sample = merge_random_sample(deficit_sample, deficit_values, max_sample_rows, rng)

        if "tmc_code" in chunk.columns:
            tmc_codes = chunk.loc[valid, "tmc_code"].dropna().astype(str)
            tmc_values.update(tmc_codes.unique().tolist())
            tmc_frame = pd.DataFrame(
                {
                    "tmc_code": chunk.loc[valid, "tmc_code"].astype(str).to_numpy(),
                    "deficit": deficit_values,
                    "severe": deficit_values >= 0.20,
                }
            )
            tmc_frame = tmc_frame[tmc_frame["tmc_code"].str.lower() != "nan"]
            tmc_deficit = add_series(tmc_deficit, tmc_frame.groupby("tmc_code")["deficit"].sum())
            tmc_obs = add_series(tmc_obs, tmc_frame.groupby("tmc_code")["deficit"].count())
            tmc_severe = add_series(tmc_severe, tmc_frame.groupby("tmc_code")["severe"].sum())

        if "measurement_tstamp" in chunk.columns:
            ts = pd.to_datetime(chunk.loc[valid, "measurement_tstamp"], errors="coerce")
            if ts.notna().any():
                ts_min = ts.min()
                ts_max = ts.max()
                start_ts = ts_min if start_ts is None else min(start_ts, ts_min)
                end_ts = ts_max if end_ts is None else max(end_ts, ts_max)
                hourly = pd.DataFrame({"timestamp": ts, "deficit": deficit.to_numpy(dtype=float)})
                hourly = hourly.dropna(subset=["timestamp", "deficit"])
                if not hourly.empty:
                    hourly["hour"] = hourly["timestamp"].dt.floor("h")
                    grouped = hourly.groupby("hour", as_index=False).agg(
                        deficit_sum=("deficit", "sum"),
                        observation_count=("deficit", "count"),
                        mean_deficit=("deficit", "mean"),
                        p90_deficit=("deficit", lambda x: float(np.quantile(x, 0.90))),
                    )
                    hourly_parts.append(grouped)

        processed_raw += len(chunk)
        if not full_scan and max_sample_rows and processed_raw >= max_sample_rows:
            break

    hourly_city = combine_hourly(city, hourly_parts)
    tmc_summary, hotspots = summarize_tmc_deficit(city, tmc_deficit, tmc_obs, tmc_severe)
    row_count = int(processed_valid)
    sample_count = int(len(deficit_sample))
    severe_20 = float(severe_20_count / row_count) if row_count else np.nan
    severe_40 = float(severe_40_count / row_count) if row_count else np.nan

    return (
        {
            "city": city,
            "speed_available": True,
            "speed_file": path.name,
            "speed_file_size_gb": round(path.stat().st_size / 1024**3, 3),
            "scanned_valid_rows": row_count,
            "sampled_distribution_rows": sample_count,
            "sampled_unique_tmc": len(tmc_values),
            "sample_start": safe_date(start_ts),
            "sample_end": safe_date(end_ts),
            "mean_speed_ratio": safe_float(ratio_sum / row_count) if row_count else np.nan,
            "median_speed_ratio": safe_float(np.nanmedian(ratio_sample)) if sample_count else np.nan,
            "mean_deficit": safe_float(deficit_sum / row_count) if row_count else np.nan,
            "median_deficit": safe_float(np.nanmedian(deficit_sample)) if sample_count else np.nan,
            "p90_deficit": safe_float(np.nanquantile(deficit_sample, 0.90)) if sample_count else np.nan,
            "p95_deficit": safe_float(np.nanquantile(deficit_sample, 0.95)) if sample_count else np.nan,
            "severe_deficit_share_20pct": severe_20,
            "severe_deficit_share_40pct": severe_40,
            "hourly_observations": int(len(hourly_city)),
        },
        hourly_city,
        tmc_summary,
        hotspots,
    )


def merge_random_sample(
    current: np.ndarray,
    new_values: np.ndarray,
    max_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    new_values = new_values[np.isfinite(new_values)]
    if max_size <= 0 or len(new_values) == 0:
        return current
    if len(current) == 0 and len(new_values) <= max_size:
        return new_values.copy()
    combined = np.concatenate([current, new_values])
    if len(combined) <= max_size:
        return combined
    indices = rng.choice(len(combined), size=max_size, replace=False)
    return combined[indices]


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def concat_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    non_empty = [array for array in arrays if len(array)]
    if not non_empty:
        return np.array([], dtype=float)
    return np.concatenate(non_empty)


def combine_hourly(city: str, parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    hourly = pd.concat(parts, ignore_index=True)
    grouped = hourly.groupby("hour", as_index=False).agg(
        deficit_sum=("deficit_sum", "sum"),
        observation_count=("observation_count", "sum"),
        p90_deficit=("p90_deficit", "mean"),
    )
    grouped["mean_deficit"] = grouped["deficit_sum"] / grouped["observation_count"].replace(0, np.nan)
    grouped.insert(0, "city", city)
    return grouped[["city", "hour", "mean_deficit", "p90_deficit", "observation_count"]]


def summarize_tmc_deficit(
    city: str,
    tmc_deficit: pd.Series | None,
    tmc_obs: pd.Series | None,
    tmc_severe: pd.Series | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if tmc_deficit is None or tmc_obs is None or len(tmc_deficit) == 0:
        return {"city": city, "tmc_concentration_available": False}, pd.DataFrame()
    summary = pd.DataFrame(
        {
            "total_deficit": tmc_deficit,
            "observation_count": tmc_obs,
            "severe_observation_count": tmc_severe if tmc_severe is not None else 0,
        }
    ).fillna(0)
    summary["mean_deficit"] = summary["total_deficit"] / summary["observation_count"].replace(0, np.nan)
    summary["severe_observation_share"] = summary["severe_observation_count"] / summary["observation_count"].replace(0, np.nan)
    summary = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_deficit"])
    burden = summary["total_deficit"].to_numpy(dtype=float)
    n_tmc = len(summary)
    result = {
        "city": city,
        "tmc_concentration_available": True,
        "tmc_count": n_tmc,
        "total_speed_deficit_burden": safe_float(summary["total_deficit"].sum()),
        "tmc_deficit_gini": gini(burden),
        "top_1pct_tmc_deficit_share": top_percent_share(burden, 0.01),
        "top_5pct_tmc_deficit_share": top_percent_share(burden, 0.05),
        "top_10pct_tmc_deficit_share": top_percent_share(burden, 0.10),
        "tmc_mean_deficit_p90": safe_float(summary["mean_deficit"].quantile(0.90)),
        "high_deficit_tmc_share_mean_gt_0_2": safe_float((summary["mean_deficit"] > 0.20).mean()),
    }
    hotspots = (
        summary.sort_values(["total_deficit", "mean_deficit"], ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "tmc_code"})
    )
    hotspots.insert(0, "city", city)
    return result, hotspots


def top_percent_share(values: np.ndarray, pct: float) -> float:
    values = values[np.isfinite(values)]
    total = values.sum()
    if total <= 0 or len(values) == 0:
        return float("nan")
    n = max(1, int(math.ceil(len(values) * pct)))
    return float(np.sort(values)[-n:].sum() / total)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def analyze_rainfall_speed_alignment(
    hourly: pd.DataFrame,
    rainfall: pd.DataFrame,
    city_paths: list[CityPaths],
    config: dict[str, Any],
) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    window = int(config["analysis"]["event_window_hours"])
    quantile = float(config["analysis"]["rainfall_threshold_quantile"])
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        if cp.city not in set(hourly["city"]):
            rows.append({"city": cp.city, "alignment_available": False})
            continue
        rain_path = cp.speed_dir / "rainfall.csv" if cp.speed_dir else None
        if not rain_path or not rain_path.exists():
            rows.append({"city": cp.city, "alignment_available": False})
            continue
        rain = load_rainfall_hourly(rain_path)
        city_hourly = hourly[hourly["city"] == cp.city].copy()
        city_hourly["hour"] = pd.to_datetime(city_hourly["hour"], errors="coerce")
        merged = city_hourly.merge(rain, on="hour", how="inner")
        if merged.empty:
            rows.append({"city": cp.city, "alignment_available": False, "alignment_reason": "no temporal overlap"})
            continue
        corrs: dict[str, float] = {}
        for lag in [0, 1, 2, 3, 6, 12]:
            lagged = merged["precipitation"].shift(lag)
            corr = merged["mean_deficit"].corr(lagged)
            corrs[f"lag_{lag}h_corr"] = safe_float(corr)
        corr_values = [v for v in corrs.values() if np.isfinite(v)]
        threshold = merged["precipitation"].quantile(quantile)
        event_metrics = summarize_event_response(merged, threshold=threshold, window=window)
        rows.append(
            {
                "city": cp.city,
                "alignment_available": True,
                "overlap_hours": int(len(merged)),
                "overlap_start": safe_date(merged["hour"].min()),
                "overlap_end": safe_date(merged["hour"].max()),
                "rainfall_threshold_overlap": safe_float(threshold),
                "max_lag_corr": safe_float(max(corr_values) if corr_values else np.nan),
                **corrs,
                **event_metrics,
            }
        )
    return pd.DataFrame(rows)


def load_rainfall_hourly(path: Path) -> pd.DataFrame:
    rain = pd.read_csv(path)
    timestamp_col = "Timestamp" if "Timestamp" in rain.columns else find_first_datetime_col(rain.columns)
    precip_col = "precipitation" if "precipitation" in rain.columns else find_precip_col(rain.columns)
    rain["hour"] = pd.to_datetime(rain[timestamp_col], errors="coerce").dt.floor("h")
    rain["precipitation"] = pd.to_numeric(rain[precip_col], errors="coerce").fillna(0)
    return rain.dropna(subset=["hour"]).groupby("hour", as_index=False)["precipitation"].sum()


def summarize_event_response(merged: pd.DataFrame, threshold: float, window: int) -> dict[str, Any]:
    if not np.isfinite(threshold):
        return {"event_count": 0}
    candidates = merged[merged["precipitation"] >= threshold].sort_values("hour")
    if candidates.empty:
        return {"event_count": 0}

    selected: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    for hour in candidates["hour"]:
        if last is None or (hour - last).total_seconds() / 3600 >= window:
            selected.append(hour)
            last = hour
        if len(selected) >= 6:
            break

    impacts: list[float] = []
    recovery_hours: list[float] = []
    for event_hour in selected:
        before = merged[(merged["hour"] >= event_hour - pd.Timedelta(hours=6)) & (merged["hour"] < event_hour)]
        after = merged[(merged["hour"] >= event_hour) & (merged["hour"] <= event_hour + pd.Timedelta(hours=window))]
        if before.empty or after.empty:
            continue
        baseline = before["mean_deficit"].mean()
        peak_idx = after["mean_deficit"].idxmax()
        peak_hour = merged.loc[peak_idx, "hour"]
        peak = merged.loc[peak_idx, "mean_deficit"]
        impact = peak - baseline
        impacts.append(float(impact))
        if impact <= 0:
            continue
        target = baseline + 0.2 * impact
        post_peak = merged[(merged["hour"] > peak_hour) & (merged["hour"] <= event_hour + pd.Timedelta(hours=window))]
        recovered = post_peak[post_peak["mean_deficit"] <= target]
        if not recovered.empty:
            recovery_hours.append(float((recovered["hour"].iloc[0] - peak_hour).total_seconds() / 3600))
    return {
        "event_count": len(selected),
        "mean_event_deficit_impact": safe_float(np.nanmean(impacts)) if impacts else np.nan,
        "median_event_recovery_hours": safe_float(np.nanmedian(recovery_hours)) if recovery_hours else np.nan,
        "events_with_recovery_observed": len(recovery_hours),
    }


def analyze_demand_network(city_paths: list[CityPaths], config: dict[str, Any]) -> pd.DataFrame:
    chunk_size = int(config["analysis"]["chunk_size"])
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        if not cp.demand_dir:
            rows.append({"city": cp.city, "demand_available": False})
            continue
        print(f"Analyzing demand/network structure for {cp.city}")
        demand_summary = analyze_demand_file(cp.demand_dir / "demand.csv", chunk_size=chunk_size)
        link_summary = analyze_link_performance(cp.demand_dir / "link_performance.csv", chunk_size=chunk_size)
        rows.append({"city": cp.city, "demand_available": True, **demand_summary, **link_summary})
    return pd.DataFrame(rows)


def analyze_demand_file(path: Path, chunk_size: int) -> dict[str, Any]:
    if not path.exists():
        return {"od_available": False}
    origin_vol: pd.Series | None = None
    dest_vol: pd.Series | None = None
    rows = 0
    total_volume = 0.0
    within_volume = 0.0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in {"o_zone_id", "d_zone_id", "volume"},
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if not {"o_zone_id", "d_zone_id", "volume"}.issubset(chunk.columns):
            continue
        chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(0)
        rows += len(chunk)
        total_volume += float(chunk["volume"].sum())
        within_volume += float(chunk.loc[chunk["o_zone_id"] == chunk["d_zone_id"], "volume"].sum())
        origin_vol = add_series(origin_vol, chunk.groupby("o_zone_id")["volume"].sum())
        dest_vol = add_series(dest_vol, chunk.groupby("d_zone_id")["volume"].sum())

    origin_count = int(len(origin_vol)) if origin_vol is not None else 0
    dest_count = int(len(dest_vol)) if dest_vol is not None else 0
    dest_values = dest_vol.to_numpy(dtype=float) if dest_vol is not None else np.array([])
    origin_values = origin_vol.to_numpy(dtype=float) if origin_vol is not None else np.array([])
    return {
        "od_available": True,
        "od_rows": rows,
        "od_total_volume": total_volume,
        "origin_zone_count": origin_count,
        "destination_zone_count": dest_count,
        "od_density_observed": safe_float(rows / (origin_count * dest_count)) if origin_count and dest_count else np.nan,
        "within_zone_volume_share": safe_float(within_volume / total_volume) if total_volume else np.nan,
        "top10_destination_volume_share": top_share(dest_values, 10),
        "destination_volume_hhi": hhi(dest_values),
        "origin_volume_hhi": hhi(origin_values),
        "destination_volume_gini": gini(dest_values),
    }


def analyze_link_performance(path: Path, chunk_size: int) -> dict[str, Any]:
    if not path.exists():
        return {"link_performance_available": False}
    rows = 0
    total_volume = 0.0
    total_vmt = 0.0
    total_vht = 0.0
    volume_weighted_speed_sum = 0.0
    congested_volume = 0.0
    doc_over_1_volume = 0.0
    link_volume: pd.Series | None = None
    usecols = {
        "link_id",
        "vehicle_volume",
        "person_volume",
        "speed_kmph",
        "speed_ratio",
        "DOC",
        "VMT",
        "VHT",
        "distance_km",
    }
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in usecols,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if "vehicle_volume" in chunk.columns:
            volume = pd.to_numeric(chunk["vehicle_volume"], errors="coerce").fillna(0)
        elif "person_volume" in chunk.columns:
            volume = pd.to_numeric(chunk["person_volume"], errors="coerce").fillna(0)
        else:
            volume = pd.Series(np.ones(len(chunk)), index=chunk.index)
        rows += len(chunk)
        total_volume += float(volume.sum())
        if "speed_kmph" in chunk.columns:
            speed = pd.to_numeric(chunk["speed_kmph"], errors="coerce")
            valid = speed.notna()
            volume_weighted_speed_sum += float((speed[valid] * volume[valid]).sum())
        if "speed_ratio" in chunk.columns:
            speed_ratio = pd.to_numeric(chunk["speed_ratio"], errors="coerce")
            congested_volume += float(volume[(speed_ratio < 0.8).fillna(False)].sum())
        if "DOC" in chunk.columns:
            doc = pd.to_numeric(chunk["DOC"], errors="coerce")
            doc_over_1_volume += float(volume[(doc > 1.0).fillna(False)].sum())
        if "VMT" in chunk.columns:
            total_vmt += float(pd.to_numeric(chunk["VMT"], errors="coerce").fillna(0).sum())
        if "VHT" in chunk.columns:
            total_vht += float(pd.to_numeric(chunk["VHT"], errors="coerce").fillna(0).sum())
        if "link_id" in chunk.columns:
            link_volume = add_series(link_volume, pd.DataFrame({"link_id": chunk["link_id"], "volume": volume}).groupby("link_id")["volume"].sum())

    link_values = link_volume.to_numpy(dtype=float) if link_volume is not None else np.array([])
    return {
        "link_performance_available": True,
        "link_performance_rows": rows,
        "link_count_observed": int(len(link_volume)) if link_volume is not None else np.nan,
        "link_total_vehicle_volume": total_volume,
        "volume_weighted_speed_kmph": safe_float(volume_weighted_speed_sum / total_volume) if total_volume else np.nan,
        "congested_volume_share_speed_ratio_lt_0_8": safe_float(congested_volume / total_volume) if total_volume else np.nan,
        "doc_over_1_volume_share": safe_float(doc_over_1_volume / total_volume) if total_volume else np.nan,
        "top10_link_volume_share": top_share(link_values, 10),
        "link_volume_hhi": hhi(link_values),
        "total_vmt": total_vmt,
        "total_vht": total_vht,
    }


def add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.copy()
    return left.add(right, fill_value=0)


def top_share(values: np.ndarray, n: int) -> float:
    values = values[np.isfinite(values)]
    total = values.sum()
    if total <= 0 or len(values) == 0:
        return float("nan")
    return float(np.sort(values)[-n:].sum() / total)


def hhi(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    total = values.sum()
    if total <= 0 or len(values) == 0:
        return float("nan")
    share = values / total
    return float(np.sum(share**2))


def gini(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if len(values) == 0:
        return float("nan")
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / (n * total)) - ((n + 1) / n))


def analyze_existing_resilience(city_paths: list[CityPaths]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        path = cp.speed_dir / "resilience_index.csv" if cp.speed_dir else None
        if not path or not path.exists():
            rows.append({"city": cp.city, "resilience_index_available": False})
            continue
        try:
            df = pd.read_csv(path)
            rows.append(
                {
                    "city": cp.city,
                    "resilience_index_available": True,
                    "resilience_rows": int(len(df)),
                    "resilience_unique_tmc": int(df["tmc_code"].nunique()) if "tmc_code" in df else np.nan,
                    "speed_auc_mean": safe_float(pd.to_numeric(df.get("speed_auc"), errors="coerce").mean()) if "speed_auc" in df else np.nan,
                    "speed_auc_p10": safe_float(pd.to_numeric(df.get("speed_auc"), errors="coerce").quantile(0.10)) if "speed_auc" in df else np.nan,
                    "speed_pct_min_mean": safe_float(pd.to_numeric(df.get("speed_pct_min"), errors="coerce").mean()) if "speed_pct_min" in df else np.nan,
                    "speed_recovery_median": safe_float(pd.to_numeric(df.get("speed_recovery"), errors="coerce").median()) if "speed_recovery" in df else np.nan,
                }
            )
        except Exception as exc:
            rows.append({"city": cp.city, "resilience_index_available": True, "resilience_error": str(exc)})
    return pd.DataFrame(rows)


def analyze_resilience_exposure_alignment(city_paths: list[CityPaths], config: dict[str, Any]) -> pd.DataFrame:
    """Test whether observed recovery deficits align with high-exposure links."""
    chunk_size = int(config["analysis"]["chunk_size"])
    rows: list[dict[str, Any]] = []
    for cp in city_paths:
        resilience_path = cp.speed_dir / "resilience_index.csv" if cp.speed_dir else None
        link_perf_path = cp.demand_dir / "link_performance.csv" if cp.demand_dir else None
        if not resilience_path or not resilience_path.exists():
            rows.append({"city": cp.city, "alignment_available": False, "reason": "no resilience index"})
            continue
        if not link_perf_path or not link_perf_path.exists():
            rows.append({"city": cp.city, "alignment_available": False, "reason": "no link performance"})
            continue
        print(f"Analyzing resilience-exposure alignment for {cp.city}")
        try:
            resilience = pd.read_csv(resilience_path)
            if "link_id" not in resilience.columns:
                rows.append({"city": cp.city, "alignment_available": False, "reason": "resilience index has no link_id"})
                continue
            link_perf = aggregate_link_exposure(link_perf_path, chunk_size)
            resilience = resilience.copy()
            resilience["link_id"] = resilience["link_id"].astype(str)
            resilience["speed_auc"] = pd.to_numeric(resilience.get("speed_auc"), errors="coerce")
            resilience["speed_pct_min"] = pd.to_numeric(resilience.get("speed_pct_min"), errors="coerce")
            resilience["speed_recovery"] = pd.to_numeric(resilience.get("speed_recovery"), errors="coerce")
            resilience["loss_auc_positive"] = (-resilience["speed_auc"]).clip(lower=0)
            resilience["max_pct_loss_positive"] = (-resilience["speed_pct_min"]).clip(lower=0)
            merged = resilience.merge(link_perf, on="link_id", how="inner")
            if merged.empty:
                rows.append({"city": cp.city, "alignment_available": False, "reason": "no link_id matches"})
                continue
            volume = pd.to_numeric(merged["vehicle_volume"], errors="coerce").fillna(0)
            loss = pd.to_numeric(merged["loss_auc_positive"], errors="coerce").fillna(0)
            doc = pd.to_numeric(merged.get("DOC"), errors="coerce")
            high_volume_threshold = volume.quantile(0.75)
            low_volume_threshold = volume.quantile(0.25)
            high_volume = merged[volume >= high_volume_threshold]
            low_volume = merged[volume <= low_volume_threshold]
            rows.append(
                {
                    "city": cp.city,
                    "alignment_available": True,
                    "resilience_rows": int(len(resilience)),
                    "matched_link_rows": int(len(merged)),
                    "match_rate": safe_float(len(merged) / len(resilience)) if len(resilience) else np.nan,
                    "loss_auc_vs_volume_spearman": safe_float(loss.corr(volume, method="spearman")),
                    "max_pct_loss_vs_volume_spearman": safe_float(
                        pd.to_numeric(merged["max_pct_loss_positive"], errors="coerce").corr(volume, method="spearman")
                    ),
                    "recovery_time_vs_volume_spearman": safe_float(
                        pd.to_numeric(merged["speed_recovery"], errors="coerce").corr(volume, method="spearman")
                    ),
                    "loss_auc_vs_DOC_spearman": safe_float(loss.corr(doc, method="spearman")),
                    "high_volume_mean_loss_auc": safe_float(high_volume["loss_auc_positive"].mean()),
                    "low_volume_mean_loss_auc": safe_float(low_volume["loss_auc_positive"].mean()),
                    "high_vs_low_volume_loss_ratio": safe_float(
                        high_volume["loss_auc_positive"].mean() / low_volume["loss_auc_positive"].mean()
                    )
                    if low_volume["loss_auc_positive"].mean() not in [0, np.nan]
                    else np.nan,
                    "top10pct_volume_share_of_loss_auc": weighted_loss_share_top_volume(merged, top_pct=0.10),
                }
            )
        except Exception as exc:
            rows.append({"city": cp.city, "alignment_available": False, "reason": str(exc)})
    return pd.DataFrame(rows)


def aggregate_link_exposure(path: Path, chunk_size: int) -> pd.DataFrame:
    volume_by_link: pd.Series | None = None
    doc_weighted_sum: pd.Series | None = None
    speed_ratio_weighted_sum: pd.Series | None = None
    usecols = {"link_id", "vehicle_volume", "person_volume", "DOC", "speed_ratio"}
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in usecols,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if "link_id" not in chunk.columns:
            continue
        link_id = chunk["link_id"].astype(str)
        if "vehicle_volume" in chunk.columns:
            volume = pd.to_numeric(chunk["vehicle_volume"], errors="coerce").fillna(0)
        elif "person_volume" in chunk.columns:
            volume = pd.to_numeric(chunk["person_volume"], errors="coerce").fillna(0)
        else:
            volume = pd.Series(np.ones(len(chunk)), index=chunk.index)
        frame = pd.DataFrame({"link_id": link_id, "vehicle_volume": volume})
        volume_by_link = add_series(volume_by_link, frame.groupby("link_id")["vehicle_volume"].sum())
        if "DOC" in chunk.columns:
            doc = pd.to_numeric(chunk["DOC"], errors="coerce").fillna(0)
            frame["doc_weighted"] = doc * volume
            doc_weighted_sum = add_series(doc_weighted_sum, frame.groupby("link_id")["doc_weighted"].sum())
        if "speed_ratio" in chunk.columns:
            speed_ratio = pd.to_numeric(chunk["speed_ratio"], errors="coerce").fillna(0)
            frame["speed_ratio_weighted"] = speed_ratio * volume
            speed_ratio_weighted_sum = add_series(
                speed_ratio_weighted_sum,
                frame.groupby("link_id")["speed_ratio_weighted"].sum(),
            )
    if volume_by_link is None:
        return pd.DataFrame(columns=["link_id", "vehicle_volume", "DOC", "speed_ratio"])
    result = volume_by_link.rename("vehicle_volume").reset_index()
    result["link_id"] = result["link_id"].astype(str)
    result = result.set_index("link_id")
    if doc_weighted_sum is not None:
        result["DOC"] = doc_weighted_sum / result["vehicle_volume"].replace(0, np.nan)
    if speed_ratio_weighted_sum is not None:
        result["speed_ratio"] = speed_ratio_weighted_sum / result["vehicle_volume"].replace(0, np.nan)
    return result.reset_index()


def weighted_loss_share_top_volume(df: pd.DataFrame, top_pct: float) -> float:
    if df.empty:
        return float("nan")
    ranked = df.sort_values("vehicle_volume", ascending=False)
    n = max(1, int(math.ceil(len(ranked) * top_pct)))
    total_loss = pd.to_numeric(ranked["loss_auc_positive"], errors="coerce").fillna(0).sum()
    if total_loss <= 0:
        return float("nan")
    return float(pd.to_numeric(ranked.head(n)["loss_auc_positive"], errors="coerce").fillna(0).sum() / total_loss)


def score_idea_fit(
    inventory: pd.DataFrame,
    rainfall: pd.DataFrame,
    speed: pd.DataFrame,
    rain_speed: pd.DataFrame,
    demand: pd.DataFrame,
    resilience: pd.DataFrame,
) -> pd.DataFrame:
    merged = inventory[["city"]].copy()
    for df in [inventory, rainfall, speed, rain_speed, demand, resilience]:
        if "city" in df.columns:
            merged = merged.merge(df, on="city", how="left", suffixes=("", "_dup"))
            merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        speed_csv = bool(row.get("speed_csv", False)) and numeric(row.get("scanned_valid_rows")) > 0
        demand_csv = bool(row.get("demand_csv", False))
        link_perf = bool(row.get("link_performance_csv", False))
        rainfall_csv = bool(row.get("rainfall_csv", False))
        resilience_idx = bool(row.get("resilience_index_csv", False))

        coverage = 0.0
        coverage += 1.25 if speed_csv else 0
        coverage += 1.0 if rainfall_csv else 0
        coverage += 1.25 if demand_csv else 0
        coverage += 1.0 if link_perf else 0
        coverage += 0.5 if bool(row.get("tmc_identification_csv", False)) else 0

        p90_deficit = row.get("p90_deficit")
        severe_share = row.get("severe_deficit_share_20pct")
        deficit_signal = bounded_score(
            2.5 * numeric(p90_deficit) / 0.25 + 2.5 * numeric(severe_share) / 0.25
        )
        if not speed_csv:
            deficit_signal = 0.0

        max_corr = numeric(row.get("max_lag_corr"))
        event_impact = numeric(row.get("mean_event_deficit_impact"))
        rainfall_alignment = bounded_score(2.5 * max(max_corr, 0) / 0.25 + 2.5 * max(event_impact, 0) / 0.05)
        if not rainfall_csv or not speed_csv:
            rainfall_alignment = 0.0

        recovery = 0.0
        if resilience_idx:
            recovery += 2.5
        if is_finite_number(row.get("median_event_recovery_hours")):
            recovery += 1.5
        if speed_csv and numeric(row.get("hourly_observations")) > 24:
            recovery += 1.0
        recovery = bounded_score(recovery)

        od_rows = numeric(row.get("od_rows"))
        link_rows = numeric(row.get("link_performance_rows"))
        zone_count = numeric(row.get("origin_zone_count")) + numeric(row.get("destination_zone_count"))
        functional_dependence = bounded_score(
            1.8 * math.log10(max(od_rows, 1)) / 6
            + 1.6 * math.log10(max(link_rows, 1)) / 6
            + 1.6 * math.log10(max(zone_count, 1)) / 4
        )
        if not demand_csv:
            functional_dependence = 0.0

        counterfactual_gap = 5.0
        if resilience_idx and is_finite_number(row.get("max_lag_corr")):
            counterfactual_gap -= 0.75
        if not speed_csv or not demand_csv:
            counterfactual_gap += 0.5
        counterfactual_gap = bounded_score(counterfactual_gap)

        data_support = np.nanmean([coverage, deficit_signal, rainfall_alignment, recovery, functional_dependence])
        rows.append(
            {
                "city": row["city"],
                "coverage_score_0_5": round(coverage, 2),
                "deficit_signal_score_0_5": round(deficit_signal, 2),
                "rainfall_alignment_score_0_5": round(rainfall_alignment, 2),
                "recovery_dynamics_score_0_5": round(recovery, 2),
                "functional_dependence_score_0_5": round(functional_dependence, 2),
                "overall_data_support_score_0_5": round(float(data_support), 2),
                "counterfactual_model_need_0_5": round(counterfactual_gap, 2),
                "interpretation": interpret_fit(data_support, counterfactual_gap),
            }
        )
    return pd.DataFrame(rows)


def numeric(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else 0.0
    except Exception:
        return 0.0


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def bounded_score(value: float) -> float:
    return float(min(5.0, max(0.0, value)))


def interpret_fit(data_support: float, model_need: float) -> str:
    if data_support >= 4 and model_need >= 4:
        return "strong empirical basis, counterfactual model still essential"
    if data_support >= 3:
        return "moderate empirical basis, model needed for recoverability"
    if data_support >= 2:
        return "partial empirical basis, limited by missing speed/rainfall/recovery signal"
    return "weak direct basis for this city without additional data"


def make_figures(
    inventory: pd.DataFrame,
    rain_speed: pd.DataFrame,
    demand: pd.DataFrame,
    fit: pd.DataFrame,
    figure_dir: Path,
    config: dict[str, Any],
    tmc_concentration: pd.DataFrame,
    exposure_alignment: pd.DataFrame,
) -> None:
    if plt is None:
        print("matplotlib is unavailable; skipping figures.")
        return
    dpi = int(config.get("figures", {}).get("dpi", 180))
    make_coverage_figure(inventory, figure_dir / "city_data_coverage.png", dpi)
    make_rain_speed_figure(rain_speed, figure_dir / "rainfall_speed_signal.png", dpi)
    make_tmc_concentration_figure(tmc_concentration, figure_dir / "speed_deficit_concentration.png", dpi)
    make_demand_figure(demand, figure_dir / "demand_network_structure.png", dpi)
    make_exposure_alignment_figure(exposure_alignment, figure_dir / "resilience_exposure_alignment.png", dpi)
    make_fit_radar(fit, figure_dir / "idea_data_fit_radar.png", dpi)


def make_coverage_figure(inventory: pd.DataFrame, path: Path, dpi: int) -> None:
    cols = ["speed_csv", "rainfall_csv", "tmc_identification_csv", "resilience_index_csv", "demand_csv", "link_performance_csv"]
    labels = ["Speed", "Rain", "TMC", "RI", "OD", "Link perf"]
    data = inventory[cols].fillna(False).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(inventory)), labels=inventory["city"])
    ax.set_title("City data coverage")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "Y" if data[i, j] else "", ha="center", va="center", color="#0F172A", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_rain_speed_figure(rain_speed: pd.DataFrame, path: Path, dpi: int) -> None:
    if rain_speed.empty or "max_lag_corr" not in rain_speed:
        return
    df = rain_speed.copy()
    df["max_lag_corr"] = pd.to_numeric(df["max_lag_corr"], errors="coerce")
    df["mean_event_deficit_impact"] = pd.to_numeric(df.get("mean_event_deficit_impact"), errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    axes[0].barh(df["city"], df["max_lag_corr"].fillna(0), color="#2563EB")
    axes[0].axvline(0, color="#334155", linewidth=0.8)
    axes[0].set_title("Max lag corr: rain vs deficit")
    axes[0].set_xlabel("correlation")
    axes[1].barh(df["city"], df["mean_event_deficit_impact"].fillna(0), color="#DC2626")
    axes[1].axvline(0, color="#334155", linewidth=0.8)
    axes[1].set_title("Mean event deficit impact")
    axes[1].set_xlabel("deficit increase")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_demand_figure(demand: pd.DataFrame, path: Path, dpi: int) -> None:
    if demand.empty:
        return
    df = demand.copy()
    x = pd.to_numeric(df.get("destination_volume_hhi"), errors="coerce")
    y = pd.to_numeric(df.get("congested_volume_share_speed_ratio_lt_0_8"), errors="coerce")
    size = pd.to_numeric(df.get("od_total_volume"), errors="coerce")
    size = 80 + 500 * (size / size.max()).fillna(0)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(x, y, s=size, color="#059669", alpha=0.72, edgecolor="#064E3B", linewidth=0.6)
    for _, row in df.iterrows():
        ax.annotate(row["city"], (row.get("destination_volume_hhi", np.nan), row.get("congested_volume_share_speed_ratio_lt_0_8", np.nan)), fontsize=8)
    ax.set_xlabel("Destination demand HHI")
    ax.set_ylabel("Congested volume share")
    ax.set_title("Demand concentration and congestion exposure")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_tmc_concentration_figure(tmc_concentration: pd.DataFrame, path: Path, dpi: int) -> None:
    if tmc_concentration.empty or "top_10pct_tmc_deficit_share" not in tmc_concentration:
        return
    df = tmc_concentration.copy()
    df = df[pd.to_numeric(df.get("tmc_count"), errors="coerce").fillna(0) > 0]
    if df.empty:
        return
    df["top_10pct_tmc_deficit_share"] = pd.to_numeric(df["top_10pct_tmc_deficit_share"], errors="coerce")
    df["tmc_deficit_gini"] = pd.to_numeric(df["tmc_deficit_gini"], errors="coerce")
    df = df.sort_values("top_10pct_tmc_deficit_share", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    axes[0].barh(df["city"], df["top_10pct_tmc_deficit_share"], color="#EA580C")
    axes[0].set_title("Top 10% TMC share of deficit")
    axes[0].set_xlabel("share")
    axes[1].barh(df["city"], df["tmc_deficit_gini"], color="#0891B2")
    axes[1].set_title("TMC deficit Gini")
    axes[1].set_xlabel("Gini")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_exposure_alignment_figure(exposure_alignment: pd.DataFrame, path: Path, dpi: int) -> None:
    if (
        exposure_alignment.empty
        or "loss_auc_vs_volume_spearman" not in exposure_alignment
        or "alignment_available" not in exposure_alignment
    ):
        return
    df = exposure_alignment.copy()
    df = df[df.get("alignment_available").astype(str).str.lower().isin(["true", "1"])]
    if df.empty:
        return
    df["loss_auc_vs_volume_spearman"] = pd.to_numeric(df["loss_auc_vs_volume_spearman"], errors="coerce")
    df["top10pct_volume_share_of_loss_auc"] = pd.to_numeric(df["top10pct_volume_share_of_loss_auc"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].bar(df["city"], df["loss_auc_vs_volume_spearman"].fillna(0), color="#4F46E5")
    axes[0].axhline(0, color="#334155", linewidth=0.8)
    axes[0].set_title("Loss vs volume Spearman")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(df["city"], df["top10pct_volume_share_of_loss_auc"].fillna(0), color="#16A34A")
    axes[1].set_title("Top 10% volume links' loss share")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_fit_radar(fit: pd.DataFrame, path: Path, dpi: int) -> None:
    if fit.empty:
        return
    categories = [
        "coverage_score_0_5",
        "deficit_signal_score_0_5",
        "rainfall_alignment_score_0_5",
        "recovery_dynamics_score_0_5",
        "functional_dependence_score_0_5",
    ]
    labels = ["Coverage", "Deficit", "Rain align", "Recovery", "Dependence"]
    values = [pd.to_numeric(fit[col], errors="coerce").mean() for col in categories]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="#7C3AED", linewidth=2)
    ax.fill(angles, values, color="#7C3AED", alpha=0.18)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 5)
    ax.set_title("Average idea-data fit")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    rainfall: pd.DataFrame,
    speed: pd.DataFrame,
    rain_speed: pd.DataFrame,
    demand: pd.DataFrame,
    resilience: pd.DataFrame,
    fit: pd.DataFrame,
    tmc_concentration: pd.DataFrame,
    exposure_alignment: pd.DataFrame,
) -> None:
    cities = len(inventory)
    speed_cities = int(inventory.get("speed_csv", pd.Series(dtype=bool)).fillna(False).sum())
    if "scanned_valid_rows" in speed.columns:
        speed_cities = int((pd.to_numeric(speed["scanned_valid_rows"], errors="coerce").fillna(0) > 0).sum())
    demand_cities = int(inventory.get("demand_csv", pd.Series(dtype=bool)).fillna(False).sum())
    resilience_cities = int(inventory.get("resilience_index_csv", pd.Series(dtype=bool)).fillna(False).sum())
    avg_support = pd.to_numeric(fit.get("overall_data_support_score_0_5"), errors="coerce").mean()
    avg_model_need = pd.to_numeric(fit.get("counterfactual_model_need_0_5"), errors="coerce").mean()

    top_fit = fit.sort_values("overall_data_support_score_0_5", ascending=False).head(5)
    max_corr = rain_speed.sort_values("max_lag_corr", ascending=False).head(5) if "max_lag_corr" in rain_speed else pd.DataFrame()
    strongest_deficit = speed.sort_values("p90_deficit", ascending=False).head(5) if "p90_deficit" in speed else pd.DataFrame()
    most_concentrated = (
        tmc_concentration.sort_values("top_10pct_tmc_deficit_share", ascending=False).head(5)
        if "top_10pct_tmc_deficit_share" in tmc_concentration
        else pd.DataFrame()
    )
    exposure_ready = (
        exposure_alignment[exposure_alignment.get("alignment_available").astype(str).str.lower().isin(["true", "1"])]
        if "alignment_available" in exposure_alignment
        else pd.DataFrame()
    )

    lines = [
        "# Data Mining Report: Recoverable Urban Resilience",
        "",
        "## 核心结论",
        "",
        f"当前数据覆盖 {cities} 个美国城市，其中 {demand_cities} 个城市有需求/网络数据，{speed_cities} 个城市有大规模速度观测，{resilience_cities} 个城市已有初步 resilience index。",
        f"从数据本身看，平均 idea-data support score 为 {avg_support:.2f}/5，counterfactual model need 为 {avg_model_need:.2f}/5。",
        "",
        "这说明：数据足以支撑论文的经验基础，也足以证明城市之间存在扰动强度、恢复轨迹、网络依赖和暴露结构差异；但“有多少损失可通过管理干预恢复”这一核心结论不能仅凭描述性 data mining 得到，必须进入优化/反事实模型。",
        "",
        "## 最匹配的城市样本",
        "",
        dataframe_to_markdown(top_fit[["city", "overall_data_support_score_0_5", "counterfactual_model_need_0_5", "interpretation"]]),
        "",
        "## 降雨-速度扰动信号",
        "",
        dataframe_to_markdown(max_corr[[c for c in ["city", "overlap_hours", "max_lag_corr", "mean_event_deficit_impact", "median_event_recovery_hours"] if c in max_corr.columns]]),
        "",
        "## 速度功能损失信号",
        "",
        dataframe_to_markdown(strongest_deficit[[c for c in ["city", "scanned_valid_rows", "sampled_distribution_rows", "sampled_unique_tmc", "p90_deficit", "severe_deficit_share_20pct", "hourly_observations"] if c in strongest_deficit.columns]]),
        "",
        "## 空间集中度与潜在决策杠杆",
        "",
        dataframe_to_markdown(most_concentrated[[c for c in ["city", "tmc_count", "top_1pct_tmc_deficit_share", "top_5pct_tmc_deficit_share", "top_10pct_tmc_deficit_share", "tmc_deficit_gini", "high_deficit_tmc_share_mean_gt_0_2"] if c in most_concentrated.columns]]),
        "",
        "这组指标用于回答一个更接近 recoverability 的前置问题：loss 是否集中到少量可定位的 links/TMCs 上。如果前 10% TMC 承担了远高于 10% 的 deficit burden，则 targeted intervention 至少在空间结构上有潜在杠杆；如果 loss 极度分散，则优化模型可能也只能得到较低 recoverable fraction。",
        "",
        "## 需求和网络依赖结构",
        "",
        dataframe_to_markdown(demand[[c for c in ["city", "od_rows", "origin_zone_count", "destination_zone_count", "top10_destination_volume_share", "destination_volume_hhi", "congested_volume_share_speed_ratio_lt_0_8"] if c in demand.columns]].head(20)),
        "",
        "## 已有 resilience index 与网络暴露的对齐",
        "",
        dataframe_to_markdown(exposure_ready[[c for c in ["city", "matched_link_rows", "match_rate", "loss_auc_vs_volume_spearman", "loss_auc_vs_DOC_spearman", "top10pct_volume_share_of_loss_auc", "high_vs_low_volume_loss_ratio"] if c in exposure_ready.columns]]),
        "",
        "这一步检查已有速度恢复指标是否落在高流量、高拥堵暴露的 links 上。它不能证明干预有效，但可以判断 observed disruption 是否具有功能暴露意义，而不是只发生在低重要性的边缘路段。",
        "",
        "## 对论文 idea 的含义",
        "",
        "1. 数据支持 `b_t`：速度相对历史速度的 deficit 可以作为 mobility-functional deficit 的直接 proxy。",
        "2. 数据部分支持 `A_t`：速度数据已经按全月时间维度聚合，可以估计内生恢复趋势；分布分位数使用固定随机样本以控制内存。",
        "3. 数据部分支持 `Q_t`：OD demand、route/link performance 和 network links 可以构造 baseline functional dependence，但服务重要性 `S_j` 仍需要 POI、就业、医疗、零售或人口暴露数据增强。",
        "4. 数据支持 `h_t` 的一个版本：rainfall 可以作为外部扰动，但如果论文想泛化到 flood/storm/infrastructure failure，需要补充事件或 hazard 数据。",
        "5. 数据不直接支持 `eta^k`、预算、response delay 和 intervention effectiveness：这些必须通过反事实情景、敏感性分析或真实应急响应记录建模。",
        "",
        "## 当前边界判断",
        "",
        "继续做描述性 data mining 的边际收益已经开始下降。剩余关键问题不是“数据里有没有恢复现象”，而是“同一个扰动下，如果资源被不同地分配，损失能少多少”。这个问题必须由 recoverable-resilience optimization model 来回答。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    formatted = formatted.fillna("").astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def write_inventory_markdown(inventory: pd.DataFrame, path: Path) -> None:
    cols = [
        "city",
        "speed_csv",
        "speed_csv_present",
        "speed_csv_size_gb",
        "rainfall_csv",
        "tmc_identification_csv",
        "resilience_index_csv",
        "demand_csv",
        "link_performance_csv",
        "demand_dir_size_mb",
    ]
    lines = [
        "# Data Inventory",
        "",
        "Raw data are stored locally under `data/raw_data/` and are not tracked by git because several files exceed normal GitHub size limits.",
        "",
        "## City Coverage",
        "",
        dataframe_to_markdown(inventory[cols]),
        "",
        "## Notes",
        "",
        "- `speed_csv_size_gb` reports the size of the largest speed CSV selected for each city.",
        "- Demand directories include DTALite-style OD, link, node, route, and performance outputs.",
        "- The data-mining pipeline writes compact tracked outputs to `results/data_mining/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

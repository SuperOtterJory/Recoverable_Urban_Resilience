"""Build event-level rainfall impact tables with matched temporal baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


WINDOW_HOURS = 12
PRE_EVENT_HOURS = 6
RAIN_INFLUENCE_HOURS = 12


@dataclass(frozen=True)
class RainEvent:
    city: str
    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_hours: int
    total_precip: float
    peak_precip: float


def main() -> None:
    root = find_repo_root()
    table_dir = root / "results" / "data_mining" / "tables"
    report_dir = root / "results" / "data_mining" / "reports"
    hourly_speed = load_hourly_speed(table_dir / "speed_hourly_deficit_sample.csv")
    speed_dirs = sorted((root / "data" / "raw_data" / "speed").glob("* city"))

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    abnormal_frames: list[pd.DataFrame] = []
    for speed_dir in speed_dirs:
        city = parse_city(speed_dir)
        rain_path = speed_dir / "rainfall.csv"
        if not rain_path.exists():
            summary_rows.append({"city": city, "rainfall_available": False})
            continue
        rain = load_rainfall(rain_path)
        events = detect_positive_rainfall_events(city, rain)
        city_speed = hourly_speed[hourly_speed["city"] == city].copy()
        abnormal = build_abnormal_speed_series(city_speed, rain, events)
        if not abnormal.empty:
            abnormal_frames.append(abnormal)
        city_details = [event_to_row(event, abnormal) for event in events]
        detail_rows.extend(city_details)
        summary_rows.append(summarize_city(city, rain, events, city_details, abnormal))

    summary = pd.DataFrame(summary_rows).sort_values("city")
    detail = pd.DataFrame(detail_rows).sort_values(["city", "event_start"])
    abnormal_all = pd.concat(abnormal_frames, ignore_index=True) if abnormal_frames else pd.DataFrame()
    summary_path = table_dir / "rainfall_event_impact_summary.csv"
    detail_path = table_dir / "rainfall_event_impact_details.csv"
    abnormal_path = table_dir / "speed_hourly_abnormal_deficit.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    abnormal_all.to_csv(abnormal_path, index=False)
    write_report(report_dir / "rainfall_event_impact_report_zh.md", summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {abnormal_path}")


def parse_city(speed_dir: Path) -> str:
    name = speed_dir.name
    return name.split("_", 1)[1].removesuffix(" city") if "_" in name else name.removesuffix(" city")


def load_hourly_speed(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["city", "hour", "mean_deficit", "p90_deficit", "observation_count"])
    hourly = pd.read_csv(path)
    hourly["hour"] = pd.to_datetime(hourly["hour"], errors="coerce")
    for column in ["mean_deficit", "p90_deficit", "observation_count"]:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
    return hourly.dropna(subset=["city", "hour"])


def load_rainfall(path: Path) -> pd.DataFrame:
    rain = pd.read_csv(path, usecols=lambda column: column in {"Timestamp", "precipitation"})
    rain["hour"] = pd.to_datetime(rain["Timestamp"], errors="coerce").dt.floor("h")
    rain["precipitation"] = pd.to_numeric(rain["precipitation"], errors="coerce").fillna(0.0)
    rain = rain.dropna(subset=["hour"])
    return rain.groupby("hour", as_index=False)["precipitation"].sum().sort_values("hour")


def detect_positive_rainfall_events(city: str, rain: pd.DataFrame) -> list[RainEvent]:
    events: list[RainEvent] = []
    current: dict[str, Any] | None = None
    previous_hour: pd.Timestamp | None = None
    event_id = 0
    for row in rain.itertuples(index=False):
        hour = pd.Timestamp(row.hour)
        precip = float(row.precipitation)
        contiguous = previous_hour is not None and (hour - previous_hour).total_seconds() / 3600 == 1
        if precip > 0:
            if current is None or not contiguous:
                if current is not None:
                    event_id += 1
                    events.append(build_event(city, event_id, current))
                current = {
                    "start": hour,
                    "end": hour,
                    "duration_hours": 1,
                    "total_precip": precip,
                    "peak_precip": precip,
                }
            else:
                current["end"] = hour
                current["duration_hours"] += 1
                current["total_precip"] += precip
                current["peak_precip"] = max(float(current["peak_precip"]), precip)
            previous_hour = hour
        else:
            if current is not None:
                event_id += 1
                events.append(build_event(city, event_id, current))
            current = None
            previous_hour = None
    if current is not None:
        event_id += 1
        events.append(build_event(city, event_id, current))
    return events


def build_event(city: str, event_id: int, data: dict[str, Any]) -> RainEvent:
    return RainEvent(
        city=city,
        event_id=event_id,
        start=pd.Timestamp(data["start"]),
        end=pd.Timestamp(data["end"]),
        duration_hours=int(data["duration_hours"]),
        total_precip=float(data["total_precip"]),
        peak_precip=float(data["peak_precip"]),
    )


def build_abnormal_speed_series(
    city_speed: pd.DataFrame,
    rain: pd.DataFrame,
    events: list[RainEvent],
) -> pd.DataFrame:
    if city_speed.empty:
        return pd.DataFrame()
    speed = city_speed.copy()
    speed["hour"] = pd.to_datetime(speed["hour"], errors="coerce")
    speed = speed.dropna(subset=["hour"]).sort_values("hour")
    rain = rain.copy()
    speed = speed.merge(rain, on="hour", how="left")
    speed["precipitation"] = pd.to_numeric(speed["precipitation"], errors="coerce").fillna(0.0)
    speed["hour_of_week"] = speed["hour"].dt.dayofweek * 24 + speed["hour"].dt.hour
    speed["hour_of_day"] = speed["hour"].dt.hour
    speed["rain_hour"] = speed["precipitation"] > 0
    speed["rain_influence_hour"] = False

    speed_start = speed["hour"].min()
    speed_end = speed["hour"].max()
    for event in events:
        if event.end < speed_start or event.start > speed_end:
            continue
        mask = (
            (speed["hour"] >= event.start - pd.Timedelta(hours=PRE_EVENT_HOURS))
            & (speed["hour"] <= event.end + pd.Timedelta(hours=RAIN_INFLUENCE_HOURS))
        )
        speed.loc[mask, "rain_influence_hour"] = True

    baseline_pool = speed[(~speed["rain_hour"]) & (~speed["rain_influence_hour"])].copy()
    if baseline_pool.empty:
        baseline_pool = speed[~speed["rain_hour"]].copy()
    if baseline_pool.empty:
        baseline_pool = speed.copy()
    how_baseline = baseline_pool.groupby("hour_of_week")["mean_deficit"].median()
    hod_baseline = baseline_pool.groupby("hour_of_day")["mean_deficit"].median()
    global_baseline = float(baseline_pool["mean_deficit"].median())

    speed["expected_deficit_how"] = speed["hour_of_week"].map(how_baseline)
    speed["expected_deficit_hod"] = speed["hour_of_day"].map(hod_baseline)
    speed["expected_deficit"] = speed["expected_deficit_how"].fillna(speed["expected_deficit_hod"]).fillna(global_baseline)
    speed["abnormal_deficit"] = speed["mean_deficit"] - speed["expected_deficit"]
    speed["positive_abnormal_deficit"] = speed["abnormal_deficit"].clip(lower=0.0)
    return speed[
        [
            "city",
            "hour",
            "mean_deficit",
            "p90_deficit",
            "observation_count",
            "precipitation",
            "hour_of_week",
            "hour_of_day",
            "expected_deficit",
            "abnormal_deficit",
            "positive_abnormal_deficit",
            "rain_hour",
            "rain_influence_hour",
        ]
    ]


def event_to_row(event: RainEvent, abnormal: pd.DataFrame) -> dict[str, Any]:
    base: dict[str, Any] = {
        "city": event.city,
        "event_id": event.event_id,
        "event_start": event.start,
        "event_end": event.end,
        "duration_hours": event.duration_hours,
        "total_precip": event.total_precip,
        "peak_precip": event.peak_precip,
        "speed_overlap": False,
    }
    if abnormal.empty:
        return base

    speed_start = abnormal["hour"].min()
    speed_end = abnormal["hour"].max()
    if event.end < speed_start or event.start > speed_end:
        return base

    before = abnormal[
        (abnormal["hour"] >= event.start - pd.Timedelta(hours=PRE_EVENT_HOURS))
        & (abnormal["hour"] < event.start)
    ]
    after = abnormal[
        (abnormal["hour"] >= event.start)
        & (abnormal["hour"] <= event.end + pd.Timedelta(hours=WINDOW_HOURS))
    ]
    if after.empty:
        return {**base, "speed_overlap": True, "impact_available": False}

    peak_idx = after["positive_abnormal_deficit"].idxmax()
    peak_hour = pd.Timestamp(abnormal.loc[peak_idx, "hour"])
    peak_positive = float(abnormal.loc[peak_idx, "positive_abnormal_deficit"])
    peak_raw = float(after["mean_deficit"].max())
    start_row = after.sort_values("hour").head(1)
    start_positive = float(start_row["positive_abnormal_deficit"].iloc[0])
    pre_positive = float(before["positive_abnormal_deficit"].mean()) if not before.empty else np.nan
    legacy_baseline = float(before["mean_deficit"].mean()) if not before.empty else np.nan
    legacy_peak_extra = peak_raw - legacy_baseline if np.isfinite(legacy_baseline) else np.nan
    mean_positive = float(after["positive_abnormal_deficit"].mean())
    target = 0.2 * peak_positive if peak_positive > 0 else np.nan
    recovery_hours = np.nan
    if peak_positive > 0:
        post_peak = after[after["hour"] > peak_hour]
        recovered = post_peak[post_peak["positive_abnormal_deficit"] <= target]
        if not recovered.empty:
            recovery_hours = float((recovered["hour"].iloc[0] - peak_hour).total_seconds() / 3600)
    affected_hours = int((after["positive_abnormal_deficit"] > target).sum()) if peak_positive > 0 else 0
    return {
        **base,
        "speed_overlap": True,
        "impact_available": True,
        "baseline_method": "hour_of_week_non_rain_then_hour_of_day",
        "pre_event_mean_positive_abnormal_deficit": pre_positive,
        "start_positive_abnormal_deficit": start_positive,
        "peak_positive_abnormal_deficit": peak_positive,
        "mean_positive_abnormal_deficit_window": mean_positive,
        "legacy_baseline_mean_deficit_prev_6h": legacy_baseline,
        "legacy_peak_extra_deficit_prev_6h": legacy_peak_extra,
        "peak_mean_deficit_window": peak_raw,
        "peak_p90_deficit_window": float(after["p90_deficit"].max()),
        "peak_hour": peak_hour,
        "recovery_hours_after_peak": recovery_hours,
        "affected_hours_in_window": affected_hours,
        "speed_observations_in_window": int(after["observation_count"].sum()),
    }


def summarize_city(
    city: str,
    rain: pd.DataFrame,
    events: list[RainEvent],
    details: list[dict[str, Any]],
    abnormal: pd.DataFrame,
) -> dict[str, Any]:
    detail = pd.DataFrame(details)
    overlap = detail[detail.get("speed_overlap", False) == True] if not detail.empty else pd.DataFrame()
    impacts = (
        overlap[overlap["impact_available"] == True]
        if not overlap.empty and "impact_available" in overlap.columns
        else pd.DataFrame()
    )
    positive_impacts = (
        impacts[pd.to_numeric(impacts["peak_positive_abnormal_deficit"], errors="coerce") > 0]
        if not impacts.empty and "peak_positive_abnormal_deficit" in impacts.columns
        else pd.DataFrame()
    )
    worst = (
        positive_impacts.sort_values("peak_positive_abnormal_deficit", ascending=False).head(1)
        if "peak_positive_abnormal_deficit" in positive_impacts.columns
        else pd.DataFrame()
    )
    positive_rain = rain.loc[rain["precipitation"] > 0, "precipitation"]
    return {
        "city": city,
        "rainfall_available": True,
        "rainfall_hours": int(len(rain)),
        "rainy_hour_count": int((rain["precipitation"] > 0).sum()),
        "rainy_hour_share": safe_float((rain["precipitation"] > 0).mean()),
        "positive_rain_event_count": int(len(events)),
        "positive_precip_p75": safe_float(positive_rain.quantile(0.75)) if not positive_rain.empty else np.nan,
        "positive_precip_p90": safe_float(positive_rain.quantile(0.90)) if not positive_rain.empty else np.nan,
        "max_hourly_precip": safe_float(rain["precipitation"].max()),
        "speed_overlap_start": safe_date(abnormal["hour"].min()) if not abnormal.empty else "",
        "speed_overlap_end": safe_date(abnormal["hour"].max()) if not abnormal.empty else "",
        "rain_events_in_speed_overlap": int(len(overlap)),
        "events_with_impact_window": int(len(impacts)),
        "events_with_positive_speed_impact": int(len(positive_impacts)),
        "mean_peak_positive_abnormal_deficit": safe_float(positive_impacts["peak_positive_abnormal_deficit"].mean()) if not positive_impacts.empty else np.nan,
        "median_peak_positive_abnormal_deficit": safe_float(positive_impacts["peak_positive_abnormal_deficit"].median()) if not positive_impacts.empty else np.nan,
        "max_peak_positive_abnormal_deficit": safe_float(positive_impacts["peak_positive_abnormal_deficit"].max()) if not positive_impacts.empty else np.nan,
        "mean_affected_hours": safe_float(positive_impacts["affected_hours_in_window"].mean()) if not positive_impacts.empty else np.nan,
        "median_recovery_hours_after_peak": safe_float(positive_impacts["recovery_hours_after_peak"].median()) if not positive_impacts.empty else np.nan,
        "worst_event_start": safe_date(worst["event_start"].iloc[0]) if not worst.empty else "",
        "worst_event_total_precip": safe_float(worst["total_precip"].iloc[0]) if not worst.empty else np.nan,
        "worst_event_peak_positive_abnormal_deficit": safe_float(worst["peak_positive_abnormal_deficit"].iloc[0]) if not worst.empty else np.nan,
    }


def write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Rainfall Event Impact With Matched Baselines",
        "",
        "事件定义：连续正降雨小时段为一次 rainfall event。速度影响只在 speed 数据覆盖月份内估计。",
        "",
        "新的 impact 不再使用事件前 6 小时作为主 baseline，而是先用非降雨、非事件影响窗口中的 same-hour-of-week median speed deficit 作为 expected deficit；若样本不足，则回退到 same-hour-of-day median，再回退到全局非雨 median。",
        "",
        "| city | full rain events | overlap events | positive abnormal impact events | mean peak abnormal | max peak abnormal | median recovery h | worst event |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = summary.sort_values(["events_with_positive_speed_impact", "max_peak_positive_abnormal_deficit"], ascending=False)
    for row in ordered.itertuples():
        lines.append(
            "| {city} | {events} | {overlap} | {positive} | {mean_peak} | {max_peak} | {recovery} | {worst} |".format(
                city=row.city,
                events=int(row.positive_rain_event_count),
                overlap=int(row.rain_events_in_speed_overlap),
                positive=int(row.events_with_positive_speed_impact),
                mean_peak=format_float(row.mean_peak_positive_abnormal_deficit),
                max_peak=format_float(row.max_peak_positive_abnormal_deficit),
                recovery=format_float(row.median_recovery_hours_after_peak),
                worst=row.worst_event_start if isinstance(row.worst_event_start, str) else "",
            )
        )
    lines.extend(
        [
            "",
            "解释：positive abnormal impact event 表示事件窗口内出现了高于 matched temporal baseline 的正速度损失异常。它比前 6 小时比较更能降低早晚高峰和平峰切换造成的偏误，但仍不是严格因果识别。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def format_float(value: Any) -> str:
    number = safe_float(value)
    return "" if np.isnan(number) else f"{number:.4f}"


if __name__ == "__main__":
    main()

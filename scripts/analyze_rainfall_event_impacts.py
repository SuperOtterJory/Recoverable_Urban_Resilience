"""Build event-level rainfall impact tables from hourly rainfall and speed loss."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recoverable_resilience.paths import find_repo_root


WINDOW_HOURS = 12
BASELINE_HOURS = 6


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
    for speed_dir in speed_dirs:
        city = parse_city(speed_dir)
        rain_path = speed_dir / "rainfall.csv"
        if not rain_path.exists():
            summary_rows.append({"city": city, "rainfall_available": False})
            continue
        rain = load_rainfall(rain_path)
        events = detect_positive_rainfall_events(city, rain)
        city_speed = hourly_speed[hourly_speed["city"] == city].copy()
        city_details = [event_to_row(event, city_speed) for event in events]
        detail_rows.extend(city_details)
        summary_rows.append(summarize_city(city, rain, events, city_details, city_speed))

    summary = pd.DataFrame(summary_rows).sort_values("city")
    detail = pd.DataFrame(detail_rows).sort_values(["city", "event_start"])
    summary_path = table_dir / "rainfall_event_impact_summary.csv"
    detail_path = table_dir / "rainfall_event_impact_details.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    write_report(report_dir / "rainfall_event_impact_report_zh.md", summary, detail)
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")


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


def event_to_row(event: RainEvent, city_speed: pd.DataFrame) -> dict[str, Any]:
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
    if city_speed.empty:
        return base

    speed_start = city_speed["hour"].min()
    speed_end = city_speed["hour"].max()
    if event.end < speed_start or event.start > speed_end:
        return base

    before = city_speed[
        (city_speed["hour"] >= event.start - pd.Timedelta(hours=BASELINE_HOURS))
        & (city_speed["hour"] < event.start)
    ]
    after = city_speed[
        (city_speed["hour"] >= event.start)
        & (city_speed["hour"] <= event.end + pd.Timedelta(hours=WINDOW_HOURS))
    ]
    if before.empty or after.empty:
        return {**base, "speed_overlap": True, "impact_available": False}

    baseline = float(before["mean_deficit"].mean())
    peak_idx = after["mean_deficit"].idxmax()
    peak_hour = pd.Timestamp(city_speed.loc[peak_idx, "hour"])
    peak_deficit = float(city_speed.loc[peak_idx, "mean_deficit"])
    peak_p90_deficit = float(after["p90_deficit"].max())
    peak_extra = peak_deficit - baseline
    mean_extra = float(after["mean_deficit"].mean() - baseline)
    target = baseline + 0.2 * peak_extra if peak_extra > 0 else np.nan
    recovery_hours = np.nan
    if peak_extra > 0:
        post_peak = after[after["hour"] > peak_hour]
        recovered = post_peak[post_peak["mean_deficit"] <= target]
        if not recovered.empty:
            recovery_hours = float((recovered["hour"].iloc[0] - peak_hour).total_seconds() / 3600)
    affected_hours = int((after["mean_deficit"] > baseline + max(peak_extra, 0.0) * 0.2).sum())
    return {
        **base,
        "speed_overlap": True,
        "impact_available": True,
        "baseline_mean_deficit_prev_6h": baseline,
        "peak_mean_deficit_window": peak_deficit,
        "peak_p90_deficit_window": peak_p90_deficit,
        "peak_extra_deficit": peak_extra,
        "mean_extra_deficit_window": mean_extra,
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
    city_speed: pd.DataFrame,
) -> dict[str, Any]:
    detail = pd.DataFrame(details)
    overlap = detail[detail.get("speed_overlap", False) == True] if not detail.empty else pd.DataFrame()
    impacts = (
        overlap[overlap["impact_available"] == True]
        if not overlap.empty and "impact_available" in overlap.columns
        else pd.DataFrame()
    )
    positive_impacts = (
        impacts[pd.to_numeric(impacts["peak_extra_deficit"], errors="coerce") > 0]
        if not impacts.empty and "peak_extra_deficit" in impacts.columns
        else pd.DataFrame()
    )
    worst = (
        positive_impacts.sort_values("peak_extra_deficit", ascending=False).head(1)
        if "peak_extra_deficit" in positive_impacts.columns
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
        "speed_overlap_start": safe_date(city_speed["hour"].min()) if not city_speed.empty else "",
        "speed_overlap_end": safe_date(city_speed["hour"].max()) if not city_speed.empty else "",
        "rain_events_in_speed_overlap": int(len(overlap)),
        "events_with_impact_window": int(len(impacts)),
        "events_with_positive_speed_impact": int(len(positive_impacts)),
        "mean_peak_extra_deficit": safe_float(positive_impacts["peak_extra_deficit"].mean()) if not positive_impacts.empty else np.nan,
        "median_peak_extra_deficit": safe_float(positive_impacts["peak_extra_deficit"].median()) if not positive_impacts.empty else np.nan,
        "max_peak_extra_deficit": safe_float(positive_impacts["peak_extra_deficit"].max()) if not positive_impacts.empty else np.nan,
        "mean_affected_hours": safe_float(positive_impacts["affected_hours_in_window"].mean()) if not positive_impacts.empty else np.nan,
        "median_recovery_hours_after_peak": safe_float(positive_impacts["recovery_hours_after_peak"].median()) if not positive_impacts.empty else np.nan,
        "worst_event_start": safe_date(worst["event_start"].iloc[0]) if not worst.empty else "",
        "worst_event_total_precip": safe_float(worst["total_precip"].iloc[0]) if not worst.empty else np.nan,
        "worst_event_peak_extra_deficit": safe_float(worst["peak_extra_deficit"].iloc[0]) if not worst.empty else np.nan,
    }


def write_report(path: Path, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    available = summary.copy()
    lines = [
        "# 降雨事件与速度损失影响分析",
        "",
        "事件定义：连续正降雨小时段被视作一次 rainfall event。速度影响只在该城市 speed 数据覆盖的月份内计算；无 speed overlap 的城市只统计降雨事件数量，不估计速度影响。",
        "",
        "## 城市汇总",
        "",
        "| city | full rain events | overlap events | positive impact events | mean peak extra deficit | max peak extra deficit | median recovery h | worst event |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in available.sort_values(["events_with_positive_speed_impact", "max_peak_extra_deficit"], ascending=False).itertuples():
        lines.append(
            "| {city} | {events} | {overlap} | {positive} | {mean_peak} | {max_peak} | {recovery} | {worst} |".format(
                city=row.city,
                events=int(row.positive_rain_event_count),
                overlap=int(row.rain_events_in_speed_overlap),
                positive=int(row.events_with_positive_speed_impact),
                mean_peak=format_float(row.mean_peak_extra_deficit),
                max_peak=format_float(row.max_peak_extra_deficit),
                recovery=format_float(row.median_recovery_hours_after_peak),
                worst=row.worst_event_start if isinstance(row.worst_event_start, str) else "",
            )
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `peak_extra_deficit` 是事件窗口内最大平均速度损失减去事件前 6 小时平均速度损失；它刻画该事件相对事前状态的额外损失。",
            "- `affected_hours_in_window` 是事件开始到事件结束后 12 小时内，高于事前基准加 20% 峰值冲击的小时数；它是时间范围，不是空间覆盖范围。",
            "- `speed_observations_in_window` 是用于计算事件窗口速度损失的原始速度观测量聚合计数，可作为统计支撑强弱的 proxy。",
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

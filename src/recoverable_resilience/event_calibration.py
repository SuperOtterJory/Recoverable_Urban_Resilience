"""Observed-event calibration for recovery LP scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import (
    calibrate_dependence_matrix,
    calibrate_exposure_weights,
    calibrate_intervention_parameters,
    city_dir_name,
    merge_nested,
    normalize_zone_ids,
    numeric,
    select_units,
)
from .paths import find_repo_root
from .recovery_lp import RecoveryLPParameters


def calibrate_observed_event_city(
    city: str,
    config: dict[str, Any],
    event_row: pd.Series | dict[str, Any],
    dynamic_row: pd.Series | dict[str, Any],
    *,
    scenario_override: dict[str, Any] | None = None,
    abnormal_hourly: pd.DataFrame | None = None,
    root: Path | None = None,
) -> RecoveryLPParameters:
    """Calibrate a city LP around one observed rainfall event."""

    repo_root = find_repo_root(root)
    project = config["project"]
    calibration = config["calibration"]
    interventions = merge_nested(config["interventions"], scenario_override or {})
    raw_dir = repo_root / project["raw_data_dir"]
    mining_dir = repo_root / project["data_mining_tables_dir"]

    demand_base = raw_dir / "demand"
    demand_dir_name = city_dir_name(city, demand_base)
    if demand_dir_name is None:
        raise FileNotFoundError(f"No demand raw-data directory found for {city}.")
    demand_path = demand_base / demand_dir_name / "demand.csv"
    demand = pd.read_csv(demand_path)
    demand["volume"] = pd.to_numeric(demand["volume"], errors="coerce").fillna(0.0)
    demand = demand[demand["volume"] > 0].copy()
    demand["o_zone_id"] = normalize_zone_ids(demand["o_zone_id"])
    demand["d_zone_id"] = normalize_zone_ids(demand["d_zone_id"])

    selected_units = select_units(demand, calibration)
    q = calibrate_dependence_matrix(demand, selected_units, float(calibration["dependence_self_loop_floor"]))
    p = calibrate_exposure_weights(demand, selected_units)
    vulnerability = destination_vulnerability_for_event(demand, selected_units)

    event = dict(event_row)
    dynamics = dict(dynamic_row)
    horizon = int(calibration["horizon_steps"])
    event_start = pd.to_datetime(event["event_start"])
    a_city = numeric(dynamics.get("a_retention"), 0.90)
    b0_signal, h_signal, signal_metadata = event_city_signals(
        city,
        event_start,
        horizon,
        a_city,
        dynamics,
        event,
        mining_dir,
        raw_dir,
        abnormal_hourly=abnormal_hourly,
    )

    blend = float(calibration["vulnerability_blend"])
    relative = (1.0 - blend) + blend * vulnerability
    relative = relative / max(float(np.sum(p * relative)), 1e-9)
    b0 = np.clip(b0_signal * relative, 0.0, 0.90)
    h = np.zeros((len(selected_units), horizon + 1), dtype=float)
    for t in range(1, horizon + 1):
        h[:, t] = np.clip(h_signal[t] * relative, 0.0, 0.30)

    structural_drag = 0.035 * (vulnerability - vulnerability.min()) / max(vulnerability.max() - vulnerability.min(), 1e-9)
    a = np.clip(a_city + structural_drag, 0.50, 0.995)
    eta, cost, u_cap, u_segment_cap, segment_effectiveness = calibrate_intervention_parameters(
        interventions,
        vulnerability,
        b0,
        h,
    )
    period_budget, total_budget = calibrate_event_budgets(interventions, b0, h)

    metadata = {
        "city": city,
        "scenario_type": "observed_rainfall_event",
        "event_id": int(event.get("event_id", -1)),
        "event_start": str(event_start),
        "event_end": str(event.get("event_end", "")),
        "event_total_precip": numeric(event.get("total_precip"), 0.0),
        "event_peak_precip": numeric(event.get("peak_precip"), 0.0),
        "event_peak_positive_abnormal_deficit": numeric(event.get("peak_positive_abnormal_deficit"), 0.0),
        "city_signal_b0": float(b0_signal),
        "city_signal_h_total": float(np.sum(h_signal)),
        "city_signal_h_peak": float(np.max(h_signal)) if len(h_signal) else 0.0,
        "dynamic_a_retention": float(a_city),
        "dynamic_rain_kernel_sum": numeric(dynamics.get("rain_kernel_sum"), 0.0),
        "unit_count": len(selected_units),
        "source_demand_path": str(demand_path),
        "calibration_notes": (
            "Observed-event scenario. b0 is the matched-baseline abnormal deficit at event start; "
            "h[t] is the positive innovation left after applying the estimated natural retention a."
        ),
        **signal_metadata,
        "scenario": scenario_override or {"name": "base"},
    }
    return RecoveryLPParameters(
        city=city,
        units=selected_units,
        p=p,
        q=q,
        b0=b0,
        a=a,
        h=h,
        eta=eta,
        cost=cost,
        u_cap=u_cap,
        u_segment_cap=u_segment_cap,
        segment_effectiveness=segment_effectiveness,
        period_budget=period_budget,
        total_budget=total_budget,
        delays={k: int(v) for k, v in interventions["delays"].items()},
        delta_c=float(interventions["delta_C"]),
        delta_s=float(interventions["delta_S"]),
        delta_t=float(calibration["delta_t_hours"]),
        metadata=metadata,
    )


def event_city_signals(
    city: str,
    event_start: pd.Timestamp,
    horizon: int,
    a_city: float,
    dynamics: dict[str, Any],
    event: dict[str, Any],
    mining_dir: Path,
    raw_dir: Path,
    *,
    abnormal_hourly: pd.DataFrame | None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if abnormal_hourly is None:
        abnormal_path = mining_dir / "speed_hourly_abnormal_deficit.csv"
        abnormal_hourly = pd.read_csv(abnormal_path, parse_dates=["hour"])
    city_hourly = abnormal_hourly[abnormal_hourly["city"] == city].copy()
    city_hourly["hour"] = pd.to_datetime(city_hourly["hour"])
    city_hourly = city_hourly.set_index("hour").sort_index()
    hours = pd.date_range(event_start, periods=horizon + 1, freq="h")
    observed = city_hourly["positive_abnormal_deficit"].reindex(hours).astype(float).to_numpy()

    fallback_h = rainfall_kernel_signal(city, event_start, horizon, dynamics, raw_dir)
    h_signal = np.zeros(horizon + 1, dtype=float)
    observed_steps = 0
    fallback_steps = 0
    for t in range(1, horizon + 1):
        if np.isfinite(observed[t]) and np.isfinite(observed[t - 1]):
            h_signal[t] = max(float(observed[t] - a_city * observed[t - 1]), 0.0)
            observed_steps += 1
        else:
            h_signal[t] = fallback_h[t]
            fallback_steps += 1

    observed_start = observed[0] if len(observed) and np.isfinite(observed[0]) else np.nan
    b0_signal = max(
        numeric(observed_start, 0.0),
        numeric(event.get("start_positive_abnormal_deficit"), 0.0),
        numeric(event.get("pre_event_mean_positive_abnormal_deficit"), 0.0),
    )
    metadata = {
        "h_observed_innovation_steps": observed_steps,
        "h_rain_kernel_fallback_steps": fallback_steps,
        "h_signal_source": "observed_innovation" if observed_steps else "rain_kernel_fallback",
    }
    return float(max(b0_signal, 0.0)), np.clip(h_signal, 0.0, 0.30), metadata


def rainfall_kernel_signal(
    city: str,
    event_start: pd.Timestamp,
    horizon: int,
    dynamics: dict[str, Any],
    raw_dir: Path,
) -> np.ndarray:
    lag_hours = int(numeric(dynamics.get("rain_lag_hours"), 6))
    betas = np.array([numeric(dynamics.get(f"rain_beta_lag_{lag}h"), 0.0) for lag in range(lag_hours + 1)], dtype=float)
    h_signal = np.zeros(horizon + 1, dtype=float)
    if np.all(betas <= 0):
        return h_signal
    speed_base = raw_dir / "speed"
    speed_dir_name = city_dir_name(city, speed_base)
    if speed_dir_name is None:
        return h_signal
    rain_path = speed_base / speed_dir_name / "rainfall.csv"
    if not rain_path.exists():
        return h_signal
    rain = pd.read_csv(rain_path, parse_dates=["Timestamp"])
    rain = rain.set_index("Timestamp").sort_index()
    rain["precipitation"] = pd.to_numeric(rain["precipitation"], errors="coerce").fillna(0.0)
    for t in range(1, horizon + 1):
        current_hour = event_start + pd.Timedelta(hours=t - 1)
        signal = 0.0
        for lag, beta in enumerate(betas):
            signal += float(beta) * float(rain["precipitation"].get(current_hour - pd.Timedelta(hours=lag), 0.0))
        h_signal[t] = signal
    return np.clip(h_signal, 0.0, 0.30)


def destination_vulnerability_for_event(demand: pd.DataFrame, selected_units: list[str]) -> np.ndarray:
    units = pd.Index(selected_units, dtype=str)
    dest = demand[demand["d_zone_id"].isin(units)].groupby("d_zone_id")["volume"].sum()
    values = dest.reindex(units, fill_value=0.0).to_numpy(dtype=float)
    if values.sum() <= 0:
        return np.ones(len(units), dtype=float)
    normalized = values / values.max()
    return 0.35 + 0.65 * normalized


def calibrate_event_budgets(interventions: dict[str, Any], b0: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    horizon = h.shape[1] - 1
    intensity = float(interventions["budget_intensity"])
    period_share = float(interventions["period_budget_share"])
    total_multiplier = float(interventions["total_budget_multiplier"])
    event_burden = max(float(np.sum(b0) + np.sum(h)), 1e-6)
    total_budget = intensity * total_multiplier * event_burden
    period_budget = np.full(horizon, max(period_share * total_budget, total_budget / horizon), dtype=float)
    return period_budget, total_budget

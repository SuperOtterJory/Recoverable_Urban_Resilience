"""Calibration utilities for the recovery optimization model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import find_repo_root
from .recovery_lp import INTERVENTIONS, RecoveryLPParameters


try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for optimization configuration.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def city_dir_name(city: str, raw_group_dir: Path) -> str | None:
    if not raw_group_dir.exists():
        return None
    suffix = f"_{city} city"
    for directory in raw_group_dir.iterdir():
        if directory.is_dir() and directory.name.endswith(suffix):
            return directory.name
    for directory in raw_group_dir.iterdir():
        if directory.is_dir() and city.lower() in directory.name.lower():
            return directory.name
    return None


def calibrate_city(
    city: str,
    config: dict[str, Any],
    *,
    scenario_override: dict[str, Any] | None = None,
    root: Path | None = None,
) -> RecoveryLPParameters:
    """Calibrate one city into a tractable LP scenario."""

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
    if not demand_path.exists():
        raise FileNotFoundError(f"Missing demand.csv for {city}: {demand_path}")

    unit_count = int(calibration["unit_count"])
    demand = pd.read_csv(demand_path)
    demand["volume"] = pd.to_numeric(demand["volume"], errors="coerce").fillna(0.0)
    demand = demand[demand["volume"] > 0].copy()

    selected_units = select_top_units(demand, unit_count)
    q = calibrate_dependence_matrix(demand, selected_units, float(calibration["dependence_self_loop_floor"]))
    p = calibrate_exposure_weights(demand, selected_units)

    data_mining = load_data_mining_rows(city, mining_dir)
    vulnerability = destination_vulnerability(demand, selected_units)
    b0 = calibrate_initial_deficit(data_mining, vulnerability, calibration)
    a = calibrate_recovery_operator(data_mining, vulnerability, calibration)
    h = calibrate_disturbance(data_mining, vulnerability, calibration)
    eta, cost, u_cap = calibrate_intervention_parameters(interventions, vulnerability, b0, h)
    period_budget, total_budget = calibrate_budgets(interventions, b0, h.shape[1] - 1)

    metadata = {
        "city": city,
        "unit_count": unit_count,
        "source_demand_path": str(demand_path),
        "calibration_notes": "Units are top OD zones by origin+destination exposure. Deficit proxies combine city-level speed/rainfall mining outputs with destination exposure concentration.",
        "speed_deficit_source": data_mining.get("speed", {}),
        "rainfall_alignment_source": data_mining.get("rain_speed", {}),
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
        period_budget=period_budget,
        total_budget=total_budget,
        delays={k: int(v) for k, v in interventions["delays"].items()},
        delta_c=float(interventions["delta_C"]),
        delta_s=float(interventions["delta_S"]),
        delta_t=float(calibration["delta_t_hours"]),
        metadata=metadata,
    )


def select_top_units(demand: pd.DataFrame, unit_count: int) -> list[str]:
    origin = demand.groupby("o_zone_id")["volume"].sum()
    dest = demand.groupby("d_zone_id")["volume"].sum()
    exposure = origin.add(dest, fill_value=0.0).sort_values(ascending=False)
    return [str(int(x)) if float(x).is_integer() else str(x) for x in exposure.head(unit_count).index.astype(float)]


def calibrate_dependence_matrix(demand: pd.DataFrame, selected_units: list[str], self_loop_floor: float) -> np.ndarray:
    units = pd.Index(selected_units, dtype=str)
    selected = demand.copy()
    selected["o_zone_id"] = selected["o_zone_id"].astype(str)
    selected["d_zone_id"] = selected["d_zone_id"].astype(str)
    selected = selected[selected["o_zone_id"].isin(units) & selected["d_zone_id"].isin(units)]
    matrix = pd.pivot_table(
        selected,
        index="o_zone_id",
        columns="d_zone_id",
        values="volume",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=units, columns=units, fill_value=0.0)

    dest_fallback = demand.copy()
    dest_fallback["d_zone_id"] = dest_fallback["d_zone_id"].astype(str)
    dest_weights = dest_fallback[dest_fallback["d_zone_id"].isin(units)].groupby("d_zone_id")["volume"].sum()
    dest_weights = dest_weights.reindex(units, fill_value=0.0).to_numpy(dtype=float)
    if dest_weights.sum() <= 0:
        dest_weights = np.ones(len(units), dtype=float)
    dest_weights = dest_weights / dest_weights.sum()

    q = matrix.to_numpy(dtype=float)
    for i in range(q.shape[0]):
        row_sum = q[i].sum()
        if row_sum <= 0:
            q[i] = dest_weights
        else:
            q[i] = q[i] / row_sum
        q[i, i] = max(q[i, i], self_loop_floor)
        q[i] = q[i] / q[i].sum()
    return q


def calibrate_exposure_weights(demand: pd.DataFrame, selected_units: list[str]) -> np.ndarray:
    units = pd.Index(selected_units, dtype=str)
    demand = demand.copy()
    demand["o_zone_id"] = demand["o_zone_id"].astype(str)
    origin = demand[demand["o_zone_id"].isin(units)].groupby("o_zone_id")["volume"].sum()
    values = origin.reindex(units, fill_value=0.0).to_numpy(dtype=float)
    if values.sum() <= 0:
        values = np.ones(len(units), dtype=float)
    return values / values.sum()


def load_data_mining_rows(city: str, mining_dir: Path) -> dict[str, dict[str, Any]]:
    tables = {
        "speed": "speed_deficit_summary.csv",
        "rain_speed": "rainfall_speed_alignment.csv",
        "demand": "demand_network_summary.csv",
        "fit": "idea_data_fit_scores.csv",
        "concentration": "speed_tmc_deficit_concentration.csv",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for key, filename in tables.items():
        path = mining_dir / filename
        if not path.exists():
            loaded[key] = {}
            continue
        df = pd.read_csv(path)
        match = df[df["city"] == city]
        loaded[key] = match.iloc[0].to_dict() if not match.empty else {}
    return loaded


def destination_vulnerability(demand: pd.DataFrame, selected_units: list[str]) -> np.ndarray:
    units = pd.Index(selected_units, dtype=str)
    demand = demand.copy()
    demand["d_zone_id"] = demand["d_zone_id"].astype(str)
    dest = demand[demand["d_zone_id"].isin(units)].groupby("d_zone_id")["volume"].sum()
    values = dest.reindex(units, fill_value=0.0).to_numpy(dtype=float)
    if values.sum() <= 0:
        return np.ones(len(units), dtype=float)
    normalized = values / values.max()
    return 0.35 + 0.65 * normalized


def calibrate_initial_deficit(
    data_mining: dict[str, dict[str, Any]],
    vulnerability: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    speed = data_mining.get("speed", {})
    fallback = float(calibration["fallback_speed_deficit"])
    mean_deficit = numeric(speed.get("mean_deficit"), fallback)
    p90_deficit = numeric(speed.get("p90_deficit"), fallback * 2)
    severe_share = numeric(speed.get("severe_deficit_share_20pct"), 0.0)
    city_signal = np.clip(0.45 * mean_deficit + 0.45 * p90_deficit + 0.10 * severe_share, 0.02, 0.60)
    blend = float(calibration["vulnerability_blend"])
    relative = (1.0 - blend) + blend * vulnerability
    return np.clip(city_signal * relative, 0.0, 0.90)


def calibrate_recovery_operator(
    data_mining: dict[str, dict[str, Any]],
    vulnerability: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    rain_speed = data_mining.get("rain_speed", {})
    recovery_hours = numeric(rain_speed.get("median_event_recovery_hours"), float("nan"))
    if not np.isfinite(recovery_hours) or recovery_hours <= 0:
        recovery_hours = float(calibration["default_recovery_tau_hours"])
    tau = np.clip(
        recovery_hours + float(calibration["min_recovery_tau_hours"]),
        float(calibration["min_recovery_tau_hours"]),
        float(calibration["max_recovery_tau_hours"]),
    )
    base_retention = math.exp(-float(calibration["delta_t_hours"]) / tau)
    structural_drag = 0.04 * (vulnerability - vulnerability.min()) / max(vulnerability.max() - vulnerability.min(), 1e-9)
    return np.clip(base_retention + structural_drag, 0.70, 0.985)


def calibrate_disturbance(
    data_mining: dict[str, dict[str, Any]],
    vulnerability: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    horizon = int(calibration["horizon_steps"])
    rain_speed = data_mining.get("rain_speed", {})
    event_impact = max(numeric(rain_speed.get("mean_event_deficit_impact"), 0.0), 0.0)
    scale = float(calibration["rainfall_shock_scale"])
    h = np.zeros((len(vulnerability), horizon + 1), dtype=float)
    if event_impact <= 0:
        return h
    profile = np.array([0.45, 0.30, 0.17, 0.08], dtype=float)
    profile = profile / profile.sum()
    for offset, weight in enumerate(profile, start=1):
        if offset <= horizon:
            h[:, offset] = scale * event_impact * weight * vulnerability
    return np.clip(h, 0.0, 0.12)


def calibrate_intervention_parameters(
    interventions: dict[str, Any],
    vulnerability: np.ndarray,
    b0: np.ndarray,
    h: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    horizon = h.shape[1] - 1
    eta: dict[str, np.ndarray] = {}
    cost: dict[str, np.ndarray] = {}
    u_cap: dict[str, np.ndarray] = {}
    difficulty = 0.8 + 0.4 * vulnerability
    local_need = np.clip(b0 + h.sum(axis=1), 0.02, 1.0)
    for key in INTERVENTIONS:
        base_eta = float(interventions["eta"][key])
        base_cost = float(interventions["cost"][key])
        eta[key] = np.tile(base_eta * (0.85 + 0.30 * vulnerability)[:, None], (1, horizon))
        cost[key] = np.tile(base_cost * difficulty[:, None], (1, horizon))
        effective_cap = float(interventions["max_effective_deployment_fraction"][key]) * local_need
        u_cap[key] = np.tile((effective_cap / np.maximum(eta[key][:, 0], 1e-9))[:, None], (1, horizon))
    return eta, cost, u_cap


def calibrate_budgets(
    interventions: dict[str, Any],
    b0: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, float]:
    intensity = float(interventions["budget_intensity"])
    period_share = float(interventions["period_budget_share"])
    total_multiplier = float(interventions["total_budget_multiplier"])
    initial_burden = max(float(np.sum(b0)), 1e-6)
    total_budget = intensity * total_multiplier * initial_burden
    period_budget = np.full(horizon, max(period_share * total_budget, total_budget / horizon), dtype=float)
    return period_budget, total_budget


def params_to_jsonable(params: RecoveryLPParameters) -> dict[str, Any]:
    return {
        "city": params.city,
        "units": params.units,
        "p": params.p.tolist(),
        "q": params.q.tolist(),
        "b0": params.b0.tolist(),
        "a": params.a.tolist(),
        "h": params.h.tolist(),
        "eta": {k: v.tolist() for k, v in params.eta.items()},
        "cost": {k: v.tolist() for k, v in params.cost.items()},
        "u_cap": {k: v.tolist() for k, v in (params.u_cap or {}).items()},
        "period_budget": params.period_budget.tolist(),
        "total_budget": params.total_budget,
        "delays": params.delays,
        "delta_c": params.delta_c,
        "delta_s": params.delta_s,
        "delta_t": params.delta_t,
        "metadata": params.metadata,
    }


def params_from_jsonable(data: dict[str, Any]) -> RecoveryLPParameters:
    return RecoveryLPParameters(
        city=data["city"],
        units=list(data["units"]),
        p=np.asarray(data["p"], dtype=float),
        q=np.asarray(data["q"], dtype=float),
        b0=np.asarray(data["b0"], dtype=float),
        a=np.asarray(data["a"], dtype=float),
        h=np.asarray(data["h"], dtype=float),
        eta={k: np.asarray(v, dtype=float) for k, v in data["eta"].items()},
        cost={k: np.asarray(v, dtype=float) for k, v in data["cost"].items()},
        u_cap={k: np.asarray(v, dtype=float) for k, v in data.get("u_cap", {}).items()} or None,
        period_budget=np.asarray(data["period_budget"], dtype=float),
        total_budget=float(data["total_budget"]),
        delays={k: int(v) for k, v in data["delays"].items()},
        delta_c=float(data["delta_c"]),
        delta_s=float(data["delta_s"]),
        delta_t=float(data["delta_t"]),
        metadata=data.get("metadata", {}),
    )


def save_params(params: RecoveryLPParameters, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params_to_jsonable(params), indent=2), encoding="utf-8")


def load_params(path: Path) -> RecoveryLPParameters:
    return params_from_jsonable(json.loads(path.read_text(encoding="utf-8")))


def calibration_summary(params: RecoveryLPParameters) -> dict[str, Any]:
    return {
        "city": params.city,
        "n_units": params.n_units,
        "horizon": params.horizon,
        "mean_b0": float(np.mean(params.b0)),
        "weighted_b0": float(np.sum(params.p * params.b0)),
        "max_b0": float(np.max(params.b0)),
        "mean_a_retention": float(np.mean(params.a)),
        "total_disturbance": float(params.h.sum()),
        "total_budget": float(params.total_budget),
        "mean_period_budget": float(params.period_budget.mean()),
        "delay_R": int(params.delays.get("R", 0)),
        "delay_C": int(params.delays.get("C", 0)),
        "delay_S": int(params.delays.get("S", 0)),
        "q_row_sum_min": float(params.q.sum(axis=1).min()),
        "q_row_sum_max": float(params.q.sum(axis=1).max()),
    }


def merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_nested(result[key], value)
        else:
            result[key] = value
    return result


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default

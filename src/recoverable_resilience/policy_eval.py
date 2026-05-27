"""Evaluate simple non-optimized intervention policies under calibrated dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .recovery_lp import INTERVENTIONS, RecoveryLPParameters


@dataclass(frozen=True)
class HeuristicPolicy:
    name: str
    primitive_split: dict[str, float]
    weight_kind: str


DEFAULT_POLICIES = (
    HeuristicPolicy("damage_based", {"R": 0.50, "C": 0.30, "S": 0.20}, "damage"),
    HeuristicPolicy("exposure_based", {"R": 0.25, "C": 0.25, "S": 0.50}, "exposure"),
    HeuristicPolicy("access_based", {"R": 0.40, "C": 0.20, "S": 0.40}, "access"),
)


def evaluate_default_policies(
    params: RecoveryLPParameters,
    *,
    baseline_objective: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate several deterministic policy baselines."""

    summary_rows: list[dict[str, Any]] = []
    trajectory_frames: list[pd.DataFrame] = []
    for policy in DEFAULT_POLICIES:
        allocations = allocate_policy(params, policy)
        objective, trajectory = simulate_policy(params, allocations)
        summary_rows.append(
            {
                "city": params.city,
                "policy": policy.name,
                "objective": objective,
                "baseline_objective": baseline_objective,
                "recoverable_fraction": 1.0 - objective / baseline_objective if baseline_objective > 1e-10 else np.nan,
                "total_intervention_cost": allocation_cost(params, allocations),
                "total_budget": params.total_budget,
                **primitive_costs(params, allocations),
            }
        )
        trajectory["policy"] = policy.name
        trajectory_frames.append(trajectory)
    return pd.DataFrame(summary_rows), pd.concat(trajectory_frames, ignore_index=True)


def allocate_policy(params: RecoveryLPParameters, policy: HeuristicPolicy) -> dict[str, np.ndarray]:
    """Create fixed intervention allocations for a simple heuristic policy."""

    horizon = params.horizon
    n = params.n_units
    allocations = {key: np.zeros((n, horizon), dtype=float) for key in INTERVENTIONS}
    total_remaining = float(params.total_budget)
    weights = policy_weights(params, policy.weight_kind)

    for t in range(horizon):
        if total_remaining <= 1e-10:
            break
        period_budget = min(float(params.period_budget[t]), total_remaining)
        used_this_period = 0.0
        for key in INTERVENTIONS:
            if t < int(params.delays.get(key, 0)):
                continue
            split = float(policy.primitive_split.get(key, 0.0))
            if split <= 0:
                continue
            budget_for_key = period_budget * split
            u = allocate_budget_to_units(
                budget_for_key,
                weights[key],
                params.cost[key][:, t],
                params.u_cap[key][:, t],
            )
            allocations[key][:, t] = u
            used_this_period += float(np.sum(params.cost[key][:, t] * u))
        total_remaining -= used_this_period
    return allocations


def policy_weights(params: RecoveryLPParameters, kind: str) -> dict[str, np.ndarray]:
    damage = positive_normalize(params.b0)
    exposure = positive_normalize(params.p)
    destination_importance = positive_normalize(params.q.T @ params.p)
    access_damage = positive_normalize(params.b0 * (0.5 * params.p + 0.5 * destination_importance))
    if kind == "damage":
        base = damage
        return {key: base for key in INTERVENTIONS}
    if kind == "exposure":
        return {"R": exposure, "C": exposure, "S": exposure}
    if kind == "access":
        return {"R": access_damage, "C": destination_importance, "S": exposure}
    raise ValueError(f"Unknown policy weight kind: {kind}")


def allocate_budget_to_units(
    budget: float,
    weights: np.ndarray,
    costs: np.ndarray,
    caps: np.ndarray,
) -> np.ndarray:
    """Allocate budget proportionally while respecting continuous caps."""

    u = np.zeros_like(weights, dtype=float)
    remaining_budget = float(budget)
    active = caps > 1e-12
    safe_costs = np.maximum(costs, 1e-12)
    weights = positive_normalize(np.where(active, weights, 0.0))
    for _ in range(len(weights) + 2):
        if remaining_budget <= 1e-10 or not active.any() or weights.sum() <= 0:
            break
        desired_cost = remaining_budget * weights / weights.sum()
        increment = desired_cost / safe_costs
        available = np.maximum(caps - u, 0.0)
        actual_increment = np.minimum(increment, available)
        u += actual_increment
        spent = float(np.sum(actual_increment * costs))
        remaining_budget -= spent
        newly_capped = available - actual_increment <= 1e-10
        active = active & (~newly_capped)
        weights = np.where(active, weights, 0.0)
        if spent <= 1e-12:
            break
    return u


def simulate_policy(
    params: RecoveryLPParameters,
    allocations: dict[str, np.ndarray],
) -> tuple[float, pd.DataFrame]:
    """Simulate calibrated linear recovery dynamics under fixed allocations."""

    n = params.n_units
    horizon = params.horizon
    b = np.zeros((n, horizon + 1), dtype=float)
    r_c = np.zeros((n, horizon + 1), dtype=float)
    d = np.zeros((n, horizon + 1), dtype=float)
    r_s = np.zeros((n, horizon + 1), dtype=float)
    ell = np.zeros((n, horizon + 1), dtype=float)
    b[:, 0] = params.b0

    for t in range(horizon + 1):
        d[:, t] = np.clip(b[:, t] - r_c[:, t], 0.0, 1.0)
        ell[:, t] = np.clip(params.q @ d[:, t] - r_s[:, t], 0.0, 1.0)
        if t == horizon:
            break
        e_r = effective_output(params, "R", t, allocations["R"][:, t])
        e_c = effective_output(params, "C", t, allocations["C"][:, t])
        e_s = effective_output(params, "S", t, allocations["S"][:, t])
        b[:, t + 1] = np.clip(params.a * b[:, t] + params.h[:, t + 1] - e_r, 0.0, 1.0)
        r_c[:, t + 1] = np.clip((1.0 - params.delta_c) * r_c[:, t] + e_c, 0.0, 1.0)
        r_s[:, t + 1] = np.clip((1.0 - params.delta_s) * r_s[:, t] + e_s, 0.0, 1.0)

    objective = float(params.delta_t * np.sum(ell * params.p[:, None]))
    rows = []
    for t in range(horizon + 1):
        for i, unit in enumerate(params.units):
            rows.append(
                {
                    "city": params.city,
                    "unit": unit,
                    "t": t,
                    "p": params.p[i],
                    "b": b[i, t],
                    "rC": r_c[i, t],
                    "d": d[i, t],
                    "rS": r_s[i, t],
                    "ell": ell[i, t],
                    "weighted_loss": params.p[i] * ell[i, t],
                }
            )
    return objective, pd.DataFrame(rows)


def effective_output(params: RecoveryLPParameters, key: str, t: int, u: np.ndarray) -> np.ndarray:
    if params.u_segment_cap is None or params.segment_effectiveness is None:
        return np.clip(params.eta[key][:, t] * u, 0.0, 1.0)
    remaining = np.maximum(u.copy(), 0.0)
    effective_u = np.zeros_like(remaining)
    for s, multiplier in enumerate(params.segment_effectiveness[key]):
        segment_take = np.minimum(remaining, params.u_segment_cap[key][:, t, s])
        effective_u += float(multiplier) * segment_take
        remaining -= segment_take
    return np.clip(params.eta[key][:, t] * effective_u, 0.0, 1.0)


def allocation_cost(params: RecoveryLPParameters, allocations: dict[str, np.ndarray]) -> float:
    return float(sum(np.sum(params.cost[key] * allocations[key]) for key in INTERVENTIONS))


def primitive_costs(params: RecoveryLPParameters, allocations: dict[str, np.ndarray]) -> dict[str, float]:
    return {f"total_cost_{key}": float(np.sum(params.cost[key] * allocations[key])) for key in INTERVENTIONS}


def positive_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    if values.sum() <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return values / values.sum()

"""Gurobi implementation of the recoverable urban functional recovery LP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


INTERVENTIONS = ("R", "C", "S")


@dataclass
class RecoveryLPParameters:
    """Container for one calibrated recovery LP scenario."""

    city: str
    units: list[str]
    p: np.ndarray
    q: np.ndarray
    b0: np.ndarray
    a: np.ndarray
    h: np.ndarray
    eta: dict[str, np.ndarray]
    cost: dict[str, np.ndarray]
    u_cap: dict[str, np.ndarray] | None
    period_budget: np.ndarray
    total_budget: float
    delays: dict[str, int]
    delta_c: float
    delta_s: float
    delta_t: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        self.b0 = np.asarray(self.b0, dtype=float)
        self.a = np.asarray(self.a, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        self.period_budget = np.asarray(self.period_budget, dtype=float)
        self.metadata = self.metadata or {}
        if self.u_cap is None:
            self.u_cap = {
                key: np.full((self.n_units, self.horizon), np.inf, dtype=float)
                for key in INTERVENTIONS
            }
        self.validate()

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def horizon(self) -> int:
        return int(self.h.shape[1] - 1)

    def validate(self) -> None:
        n = self.n_units
        if n == 0:
            raise ValueError("RecoveryLPParameters requires at least one unit.")
        if self.p.shape != (n,):
            raise ValueError(f"p shape {self.p.shape} does not match units {n}.")
        if self.q.shape != (n, n):
            raise ValueError(f"q shape {self.q.shape} does not match ({n}, {n}).")
        if self.b0.shape != (n,):
            raise ValueError(f"b0 shape {self.b0.shape} does not match units {n}.")
        if self.a.shape != (n,):
            raise ValueError(f"a shape {self.a.shape} does not match units {n}.")
        if self.h.shape[0] != n or self.h.shape[1] < 2:
            raise ValueError("h must have shape (n_units, horizon + 1).")
        if self.period_budget.shape != (self.horizon,):
            raise ValueError("period_budget must have shape (horizon,).")
        for key in INTERVENTIONS:
            if key not in self.eta or key not in self.cost:
                raise ValueError(f"Missing eta/cost for intervention {key}.")
            if key not in self.u_cap:
                raise ValueError(f"Missing u_cap for intervention {key}.")
            if np.asarray(self.eta[key]).shape != (n, self.horizon):
                raise ValueError(f"eta[{key}] must have shape (n_units, horizon).")
            if np.asarray(self.cost[key]).shape != (n, self.horizon):
                raise ValueError(f"cost[{key}] must have shape (n_units, horizon).")
            if np.asarray(self.u_cap[key]).shape != (n, self.horizon):
                raise ValueError(f"u_cap[{key}] must have shape (n_units, horizon).")
        row_sums = self.q.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError("Each row of q must sum to 1.")
        if np.any(self.p < 0) or not np.isclose(self.p.sum(), 1.0, atol=1e-5):
            raise ValueError("p must be nonnegative and normalized to sum to 1.")
        if np.any(self.b0 < 0) or np.any(self.b0 > 1):
            raise ValueError("b0 must be bounded in [0, 1].")

    def copy_with_budget(self, budget_scale: float, delays: dict[str, int] | None = None) -> "RecoveryLPParameters":
        return RecoveryLPParameters(
            city=self.city,
            units=list(self.units),
            p=self.p.copy(),
            q=self.q.copy(),
            b0=self.b0.copy(),
            a=self.a.copy(),
            h=self.h.copy(),
            eta={k: v.copy() for k, v in self.eta.items()},
            cost={k: v.copy() for k, v in self.cost.items()},
            u_cap={k: v.copy() for k, v in (self.u_cap or {}).items()},
            period_budget=self.period_budget * budget_scale,
            total_budget=float(self.total_budget * budget_scale),
            delays=dict(delays or self.delays),
            delta_c=self.delta_c,
            delta_s=self.delta_s,
            delta_t=self.delta_t,
            metadata=dict(self.metadata),
        )


@dataclass
class RecoveryLPSolution:
    """Compact solution output from the recovery LP."""

    status: str
    objective: float
    baseline_objective: float | None
    recoverable_fraction: float | None
    runtime_seconds: float
    trajectory: pd.DataFrame
    interventions: pd.DataFrame
    summary: dict[str, Any]


def solve_recovery_lp(
    params: RecoveryLPParameters,
    *,
    no_intervention: bool = False,
    output_flag: bool = False,
    method: int = -1,
    time_limit_seconds: float | None = None,
) -> RecoveryLPSolution:
    """Solve the continuous recovery LP with Gurobi."""

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("gurobipy is required to solve the recovery LP.") from exc

    n = params.n_units
    horizon = params.horizon
    model = gp.Model(f"recoverable_resilience_{params.city}")
    model.Params.OutputFlag = 1 if output_flag else 0
    if method is not None:
        model.Params.Method = method
    if time_limit_seconds:
        model.Params.TimeLimit = float(time_limit_seconds)

    units = range(n)
    state_times = range(horizon + 1)
    action_times = range(horizon)

    b = model.addVars(n, horizon + 1, lb=0.0, ub=1.0, name="b")
    r_c = model.addVars(n, horizon + 1, lb=0.0, ub=1.0, name="rC")
    d = model.addVars(n, horizon + 1, lb=0.0, ub=1.0, name="d")
    r_s = model.addVars(n, horizon + 1, lb=0.0, ub=1.0, name="rS")
    ell = model.addVars(n, horizon + 1, lb=0.0, ub=1.0, name="ell")
    u = {
        key: model.addVars(n, horizon, lb=0.0, name=f"u_{key}")
        for key in INTERVENTIONS
    }
    e = {
        key: model.addVars(n, horizon, lb=0.0, ub=1.0, name=f"e_{key}")
        for key in INTERVENTIONS
    }

    model.setObjective(
        params.delta_t * gp.quicksum(float(params.p[i]) * ell[i, t] for i in units for t in state_times),
        GRB.MINIMIZE,
    )

    for i in units:
        model.addConstr(b[i, 0] == float(params.b0[i]), name=f"initial_b[{i}]")
        model.addConstr(r_c[i, 0] == 0.0, name=f"initial_rC[{i}]")
        model.addConstr(r_s[i, 0] == 0.0, name=f"initial_rS[{i}]")

    for t in action_times:
        for i in units:
            model.addConstr(
                b[i, t + 1]
                == float(params.a[i]) * b[i, t]
                + float(params.h[i, t + 1])
                - e["R"][i, t],
                name=f"b_transition[{i},{t}]",
            )
            model.addConstr(
                r_c[i, t + 1] == (1.0 - params.delta_c) * r_c[i, t] + e["C"][i, t],
                name=f"rC_transition[{i},{t}]",
            )
            model.addConstr(
                r_s[i, t + 1] == (1.0 - params.delta_s) * r_s[i, t] + e["S"][i, t],
                name=f"rS_transition[{i},{t}]",
            )
            for key in INTERVENTIONS:
                model.addConstr(
                    e[key][i, t] <= float(params.eta[key][i, t]) * u[key][i, t],
                    name=f"effectiveness[{key},{i},{t}]",
                )
                model.addConstr(
                    u[key][i, t] <= float(params.u_cap[key][i, t]),
                    name=f"deployment_cap[{key},{i},{t}]",
                )
                if no_intervention or t < int(params.delays.get(key, 0)):
                    model.addConstr(u[key][i, t] == 0.0, name=f"delay_or_zero[{key},{i},{t}]")

    for t in state_times:
        for i in units:
            model.addConstr(d[i, t] >= b[i, t] - r_c[i, t], name=f"local_deficit[{i},{t}]")
            model.addConstr(
                ell[i, t]
                >= gp.quicksum(float(params.q[i, j]) * d[j, t] for j in units) - r_s[i, t],
                name=f"access_loss[{i},{t}]",
            )

    if no_intervention:
        for t in action_times:
            model.addConstr(
                gp.quicksum(u[key][i, t] for key in INTERVENTIONS for i in units) <= 0.0,
                name=f"zero_period_budget[{t}]",
            )
    else:
        for t in action_times:
            model.addConstr(
                gp.quicksum(
                    float(params.cost[key][i, t]) * u[key][i, t]
                    for key in INTERVENTIONS
                    for i in units
                )
                <= float(params.period_budget[t]),
                name=f"period_budget[{t}]",
            )
        model.addConstr(
            gp.quicksum(
                float(params.cost[key][i, t]) * u[key][i, t]
                for key in INTERVENTIONS
                for i in units
                for t in action_times
            )
            <= float(params.total_budget),
            name="total_budget",
        )

    model.optimize()
    status = status_name(model.Status)
    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or model.SolCount == 0:
        raise RuntimeError(f"Gurobi did not return a feasible solution. Status: {status}")

    objective = float(model.ObjVal)
    trajectory_rows: list[dict[str, Any]] = []
    for t in state_times:
        for i in units:
            trajectory_rows.append(
                {
                    "city": params.city,
                    "unit": params.units[i],
                    "t": t,
                    "p": float(params.p[i]),
                    "b": b[i, t].X,
                    "rC": r_c[i, t].X,
                    "d": d[i, t].X,
                    "rS": r_s[i, t].X,
                    "ell": ell[i, t].X,
                    "weighted_loss": float(params.p[i]) * ell[i, t].X,
                }
            )

    intervention_rows: list[dict[str, Any]] = []
    for t in action_times:
        for i in units:
            for key in INTERVENTIONS:
                intervention_rows.append(
                    {
                        "city": params.city,
                        "unit": params.units[i],
                        "t": t,
                        "intervention": key,
                        "u": u[key][i, t].X,
                        "e": e[key][i, t].X,
                        "cost": float(params.cost[key][i, t]),
                        "effective_cost": float(params.cost[key][i, t]) * u[key][i, t].X,
                    }
                )

    trajectory = pd.DataFrame(trajectory_rows)
    interventions = pd.DataFrame(intervention_rows)
    summary = summarize_solution(params, objective, trajectory, interventions, status, model.Runtime)
    return RecoveryLPSolution(
        status=status,
        objective=objective,
        baseline_objective=None,
        recoverable_fraction=None,
        runtime_seconds=float(model.Runtime),
        trajectory=trajectory,
        interventions=interventions,
        summary=summary,
    )


def solve_with_baseline(
    params: RecoveryLPParameters,
    *,
    output_flag: bool = False,
    method: int = -1,
    time_limit_seconds: float | None = None,
) -> tuple[RecoveryLPSolution, RecoveryLPSolution]:
    """Solve no-intervention baseline and optimized scenario."""

    baseline = solve_recovery_lp(
        params,
        no_intervention=True,
        output_flag=output_flag,
        method=method,
        time_limit_seconds=time_limit_seconds,
    )
    optimized = solve_recovery_lp(
        params,
        no_intervention=False,
        output_flag=output_flag,
        method=method,
        time_limit_seconds=time_limit_seconds,
    )
    optimized.baseline_objective = baseline.objective
    optimized.recoverable_fraction = (
        1.0 - optimized.objective / baseline.objective if baseline.objective > 1e-10 else np.nan
    )
    optimized.summary["baseline_objective"] = baseline.objective
    optimized.summary["recoverable_fraction"] = optimized.recoverable_fraction
    return baseline, optimized


def summarize_solution(
    params: RecoveryLPParameters,
    objective: float,
    trajectory: pd.DataFrame,
    interventions: pd.DataFrame,
    status: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    final = trajectory[trajectory["t"] == params.horizon]
    by_k = interventions.groupby("intervention", as_index=False).agg(
        total_u=("u", "sum"),
        total_e=("e", "sum"),
        total_cost=("effective_cost", "sum"),
    )
    summary: dict[str, Any] = {
        "city": params.city,
        "status": status,
        "objective": objective,
        "runtime_seconds": runtime_seconds,
        "n_units": params.n_units,
        "horizon": params.horizon,
        "initial_mean_b": float(np.average(params.b0, weights=params.p)),
        "final_weighted_b": float(np.sum(final["p"] * final["b"])),
        "final_weighted_ell": float(np.sum(final["p"] * final["ell"])),
        "total_intervention_cost": float(interventions["effective_cost"].sum()),
        "total_budget": float(params.total_budget),
        "mean_period_budget": float(np.mean(params.period_budget)),
    }
    for _, row in by_k.iterrows():
        key = row["intervention"]
        summary[f"total_u_{key}"] = float(row["total_u"])
        summary[f"total_e_{key}"] = float(row["total_e"])
        summary[f"total_cost_{key}"] = float(row["total_cost"])
    return summary


def status_name(status_code: int) -> str:
    try:
        import gurobipy as gp

        mapping = {
            gp.GRB.OPTIMAL: "OPTIMAL",
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        return mapping.get(status_code, str(status_code))
    except Exception:  # pragma: no cover
        return str(status_code)

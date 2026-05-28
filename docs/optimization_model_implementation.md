# Optimization Model Implementation

This repository implements the continuous recovery LP from the draft in `src/recoverable_resilience/recovery_lp.py`.

## Implemented LP

For each calibrated city scenario, the solver minimizes cumulative access-weighted functional loss:

```text
min sum_t sum_i p_i ell_i,t
```

subject to:

- underlying deficit transition `b_{t+1} = A b_t + h_{t+1} - e^R_t`;
- temporary-capacity relief stock transition;
- substitution/control relief stock transition;
- local experienced deficit `d_t >= b_t - r^C_t`;
- access-weighted loss `ell_t >= Q d_t - r^S_t`;
- linear effectiveness `e^k <= eta^k u^k`, or the corresponding segment-weighted expression when diminishing returns are enabled;
- response delays;
- continuous primitive-specific deployment caps;
- concave piecewise-linear diminishing returns represented through continuous segment variables;
- period and total budgets;
- bounded state variables in `[0, 1]`.

The current implementation uses all available OD zones for cities with usable speed data. The functional-dependence matrix is stored in sparse CSR form and the LP only creates coefficients for observed nonzero OD dependence. Identity spatial effect matrices are used for `M^R`, `M^C`, and `M^S`; this keeps the empirical version directly interpretable. Continuous deployment caps and concave piecewise-linear diminishing returns remain linear and prevent unrealistic unlimited use of a single intervention type.

## Calibration

`src/recoverable_resilience/calibration.py` maps data into LP parameters:

- `Q`: row-normalized sparse OD demand matrix among all selected OD zones. The current canonical configuration uses all zones.
- `p`: normalized origin demand exposure.
- `b0`: city speed-deficit signal scaled by destination exposure vulnerability.
- `A`: endogenous recovery retention calibrated from observed event recovery hours.
- `h`: short rainfall-disturbance profile calibrated from event deficit impact.
- `eta`, costs, budgets, and delays: fixed recovery-regime parameters with exposure-based cost/effectiveness adjustments.

The calibration is a first empirical bridge, not a final causal estimate of intervention effects.

## Commands

```powershell
$env:PYTHONPATH='src'
python scripts\calibrate_optimization.py --config configs\optimization.yml
python scripts\run_optimization.py --config configs\optimization.yml
python scripts\analyze_city_structure.py
```

Optimization outputs are written to `results/optimization/`. City-structure outputs are written to `results/city_structure/`.

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
- linear effectiveness `e^k <= eta^k u^k`;
- response delays;
- period and total budgets;
- bounded state variables in `[0, 1]`.

The current implementation uses identity spatial effect matrices for `M^R`, `M^C`, and `M^S`. This keeps the first empirical version tractable and directly interpretable. The parameter object is structured so non-identity effect matrices can be added later.

## Calibration

`src/recoverable_resilience/calibration.py` maps data into LP parameters:

- `Q`: row-normalized OD demand matrix among the top OD-exposure zones.
- `p`: normalized origin demand exposure.
- `b0`: city speed-deficit signal scaled by destination exposure vulnerability.
- `A`: endogenous recovery retention calibrated from observed event recovery hours.
- `h`: short rainfall-disturbance profile calibrated from event deficit impact.
- `eta`, costs, budgets, and delays: scenario/tuning parameters with exposure-based cost/effectiveness adjustments.

The calibration is a first empirical bridge, not a final causal estimate of intervention effects.

## Commands

```powershell
$env:PYTHONPATH='src'
python scripts\calibrate_optimization.py --config configs\optimization.yml
python scripts\run_optimization.py --config configs\optimization.yml
python scripts\tune_recovery_parameters.py --config configs\optimization.yml
```

Outputs are written to `results/optimization/`.

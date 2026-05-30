# Broader Explicit Multi-Objective Footprint LP Validation V42

本版本在 V41 的 6 个代表性事件之外，选择更多 footprint-sensitive city-events，直接求解显式 recovery--footprint 二目标 LP。默认排除 New York，因为 1940-zone hybrid LP 在前序版本中已经形成计算边界。

## 关键结论

- selected events: 12; successful events per lambda: 12; lambdas: 3; excluded cities: New York.
- lambda=0 gain/LP = 1.0000; top-5% footprint mass = 0.0583.
- <=0.005 gain-loss best lambda = 0.0200; footprint delta = 0.0171; gain delta = -0.0008.
- max-footprint lambda = 0.0500; footprint mass = 0.1085; gain/LP = 0.9887.
- loss <= 0.001: lambda=0.0200, footprint delta=0.0171, gain delta=-0.0008
- loss <= 0.0025: lambda=0.0200, footprint delta=0.0171, gain delta=-0.0008
- loss <= 0.005: lambda=0.0200, footprint delta=0.0171, gain delta=-0.0008
- loss <= 0.01: lambda=0.0200, footprint delta=0.0171, gain delta=-0.0008
- loss <= 0.02: lambda=0.0500, footprint delta=0.0503, gain delta=-0.0113
- loss <= 0.05: lambda=0.0500, footprint delta=0.0503, gain delta=-0.0113

## Selected Events

| city | event_id | event_start | footprint_blend | finite_action_value_spearman | finite_top5pct_action_jaccard | delta_finite_top5pct_units_footprint_mass | base_finite_top5pct_units_footprint_mass | hybrid_finite_top5pct_units_footprint_mass | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 478 | 2024-07-06 17:00:00 | 0.5 | 0.9771 | 0.6217 | 0.2779 | 0.02607 | 0.304 | 0.2468 |
| Austin | 490 | 2024-07-25 21:00:00 | 0.5 | 0.9489 | 0.4402 | 0.2709 | 0.021 | 0.2919 | 0.2489 |
| Chicago | 158 | 2019-07-29 13:00:00 | 0.5 | 0.9759 | 0.577 | 0.2524 | 0.09872 | 0.3511 | 0.2235 |
| Chicago | 159 | 2019-07-29 17:00:00 | 0.5 | 0.9682 | 0.5592 | 0.2305 | 0.1033 | 0.3338 | 0.2026 |
| Dallas | 58 | 2019-04-18 22:00:00 | 0.5 | 0.932 | 0.3546 | 0.3557 | 0.02391 | 0.3796 | 0.338 |
| Dallas | 62 | 2019-04-28 01:00:00 | 0.5 | 0.9367 | 0.3932 | 0.363 | 0.02378 | 0.3868 | 0.3434 |
| Houston | 60 | 2019-04-18 05:00:00 | 0.5 | 0.966 | 0.6146 | 0.1203 | 0.04103 | 0.1614 | 0.08959 |
| Houston | 62 | 2019-04-30 03:00:00 | 0.5 | 0.9565 | 0.5578 | 0.1098 | 0.05215 | 0.1619 | 0.08191 |
| Philadelphia | 644 | 2023-07-09 11:00:00 | 0.5 | 0.9412 | 0.3458 | 0.4606 | 0.02234 | 0.4829 | 0.4433 |
| Philadelphia | 658 | 2023-07-29 09:00:00 | 0.5 | 0.9174 | 0.328 | 0.4494 | 0.02236 | 0.4718 | 0.433 |
| San Antonio | 494 | 2024-07-12 13:00:00 | 0.5 | 0.9798 | 0.6531 | 0.1785 | 0.06467 | 0.2432 | 0.1458 |
| San Antonio | 497 | 2024-07-19 12:00:00 | 0.5 | 0.9652 | 0.6165 | 0.172 | 0.06692 | 0.2389 | 0.1412 |

## Lambda Summary

| lambda_footprint | n_events | mean_fraction_of_reference_lp_gain | mean_delta_fraction_vs_lambda0 | mean_recoverable_fraction | mean_top5_allocated_unit_footprint_mass | mean_delta_top5_allocated_mass_vs_lambda0 | mean_selected_unit_footprint_mass | mean_cost_weighted_footprint_mass | mean_cost_weighted_footprint_reward_score | mean_footprint_reward_budget_share | mean_runtime_seconds | n_optimal | n_time_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 1 | 0 | 0.1256 | 0.05827 | 0 | 0.1878 | 0.002939 | 0.05844 | 0.05844 | 19.23 | 12 | 0 |
| 0.02 | 12 | 0.9992 | -0.0008478 | 0.1255 | 0.07536 | 0.01708 | 0.2257 | 0.00342 | 0.06823 | 0.06823 | 19.07 | 12 | 0 |
| 0.05 | 12 | 0.9887 | -0.01133 | 0.1242 | 0.1085 | 0.05026 | 0.2795 | 0.005185 | 0.1028 | 0.1028 | 16.96 | 12 | 0 |

## Pareto Frontier

| lambda_footprint | n_events | mean_fraction_of_reference_lp_gain | mean_delta_fraction_vs_lambda0 | mean_recoverable_fraction | mean_top5_allocated_unit_footprint_mass | mean_delta_top5_allocated_mass_vs_lambda0 | mean_selected_unit_footprint_mass | mean_cost_weighted_footprint_mass | mean_cost_weighted_footprint_reward_score | mean_footprint_reward_budget_share | mean_runtime_seconds | n_optimal | n_time_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 1 | 0 | 0.1256 | 0.05827 | 0 | 0.1878 | 0.002939 | 0.05844 | 0.05844 | 19.23 | 12 | 0 |
| 0.02 | 12 | 0.9992 | -0.0008478 | 0.1255 | 0.07536 | 0.01708 | 0.2257 | 0.00342 | 0.06823 | 0.06823 | 19.07 | 12 | 0 |
| 0.05 | 12 | 0.9887 | -0.01133 | 0.1242 | 0.1085 | 0.05026 | 0.2795 | 0.005185 | 0.1028 | 0.1028 | 16.96 | 12 | 0 |

## Event Best Within Loss Thresholds

| city | event_id | loss_threshold | lambda_footprint | fraction_of_reference_lp_gain | delta_fraction_vs_lambda0 | top5_allocated_unit_footprint_mass | delta_top5_allocated_mass_vs_lambda0 | lambda0_fraction_of_reference_lp_gain | lambda0_top5_allocated_unit_footprint_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 478 | 0.001 | 0 | 1 | 0 | 0.03938 | 0 | 1 | 0.03938 |
| Austin | 478 | 0.0025 | 0.02 | 0.9989 | -0.00112 | 0.09277 | 0.05339 | 1 | 0.03938 |
| Austin | 478 | 0.005 | 0.02 | 0.9989 | -0.00112 | 0.09277 | 0.05339 | 1 | 0.03938 |
| Austin | 478 | 0.01 | 0.02 | 0.9989 | -0.00112 | 0.09277 | 0.05339 | 1 | 0.03938 |
| Austin | 478 | 0.02 | 0.02 | 0.9989 | -0.00112 | 0.09277 | 0.05339 | 1 | 0.03938 |
| Austin | 478 | 0.05 | 0.05 | 0.9544 | -0.04558 | 0.1535 | 0.1141 | 1 | 0.03938 |
| Austin | 490 | 0.001 | 0.02 | 0.999 | -0.0009904 | 0.0791 | 0.03964 | 1 | 0.03945 |
| Austin | 490 | 0.0025 | 0.02 | 0.999 | -0.0009904 | 0.0791 | 0.03964 | 1 | 0.03945 |
| Austin | 490 | 0.005 | 0.02 | 0.999 | -0.0009904 | 0.0791 | 0.03964 | 1 | 0.03945 |
| Austin | 490 | 0.01 | 0.02 | 0.999 | -0.0009904 | 0.0791 | 0.03964 | 1 | 0.03945 |
| Austin | 490 | 0.02 | 0.05 | 0.99 | -0.01002 | 0.1274 | 0.08794 | 1 | 0.03945 |
| Austin | 490 | 0.05 | 0.05 | 0.99 | -0.01002 | 0.1274 | 0.08794 | 1 | 0.03945 |
| Chicago | 158 | 0.001 | 0 | 1 | 0 | 0.09712 | 0 | 1 | 0.09712 |
| Chicago | 158 | 0.0025 | 0.02 | 0.9988 | -0.001228 | 0.1472 | 0.05009 | 1 | 0.09712 |
| Chicago | 158 | 0.005 | 0.02 | 0.9988 | -0.001228 | 0.1472 | 0.05009 | 1 | 0.09712 |
| Chicago | 158 | 0.01 | 0.02 | 0.9988 | -0.001228 | 0.1472 | 0.05009 | 1 | 0.09712 |
| Chicago | 158 | 0.02 | 0.05 | 0.986 | -0.01396 | 0.1856 | 0.08853 | 1 | 0.09712 |
| Chicago | 158 | 0.05 | 0.05 | 0.986 | -0.01396 | 0.1856 | 0.08853 | 1 | 0.09712 |
| Chicago | 159 | 0.001 | 0 | 1 | 0 | 0.1052 | 0 | 1 | 0.1052 |
| Chicago | 159 | 0.0025 | 0.02 | 0.9987 | -0.001291 | 0.1525 | 0.04735 | 1 | 0.1052 |

## 解释

如果 V42 中 lambda=0.02 仍然以很小 recovery loss 换来 footprint gain，就说明 V41 的 near-free tie-breaker 不是只来自 6 个代表性事件。若 lambda=0.05 或更高出现明显 gain loss，则继续支持论文把 footprint 写成 secondary objective frontier，而不是 recovery-only law。

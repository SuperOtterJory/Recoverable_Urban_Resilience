# Hybrid Footprint Full-LP Validation V35

本版从 V34 的 footprint-sensitive events 中每个城市选择 finite footprint gain 最大的代表事件，并在 hybrid OD-template + TMC-footprint calibration 下重新求解完整 LP。

## 关键结论

- 代表性事件数：7；成功返回可行 LP 解：6；其中 OPTIMAL：6。
- base LP selected units 捕获的 observed footprint mass 平均为 0.1402，hybrid LP 为 0.1470，变化 0.0068。
- cost-weighted selected footprint score 从 0.0024 到 0.0027，变化 0.0003。
- base vs hybrid selected-action Jaccard 平均为 0.9228，selected-unit Jaccard 平均为 0.9371。
- hybrid/base no-intervention objective ratio 平均为 0.9644；recoverable fraction 平均变化为 0.0012。
- 对照 V34，代表事件的 finite top-5% footprint-mass gain 平均为 0.2931，但 full LP selected-unit gain 只有 0.0068。

## Selected Events

| city | event_id | event_start | footprint_blend | finite_action_value_spearman | finite_top5pct_action_jaccard | delta_finite_top5pct_units_footprint_mass | base_finite_top5pct_units_footprint_mass | hybrid_finite_top5pct_units_footprint_mass | hybrid_to_base_baseline_objective_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 484 | 2024-07-22 14:00:00 | 0.5000 | 0.9520 | 0.4485 | 0.2985 | 0.0192 | 0.3178 | 0.8828 |
| Chicago | 157 | 2019-07-29 10:00:00 | 0.5000 | 0.9729 | 0.5561 | 0.2585 | 0.0942 | 0.3526 | 1.0394 |
| Dallas | 64 | 2019-04-30 18:00:00 | 0.5000 | 0.9466 | 0.4218 | 0.3696 | 0.0209 | 0.3905 | 0.9601 |
| Houston | 52 | 2019-04-04 01:00:00 | 0.5000 | 0.9545 | 0.5492 | 0.1599 | 0.0375 | 0.1974 | 0.9329 |
| New York | 117 | 2019-06-20 07:00:00 | 0.5000 | 0.9584 | 0.5475 | 0.3691 | 0.0494 | 0.4185 | 0.9788 |
| Philadelphia | 641 | 2023-07-07 11:00:00 | 0.5000 | 0.9596 | 0.4080 | 0.4902 | 0.0207 | 0.5109 | 1.0158 |
| San Antonio | 499 | 2024-07-22 20:00:00 | 0.5000 | 0.9827 | 0.6810 | 0.1819 | 0.0614 | 0.2433 | 0.9554 |

## Event Metrics

| city | event_id | event_start | n_units | base_status | hybrid_status | runtime_seconds | base_baseline_objective | hybrid_baseline_objective | hybrid_to_base_baseline_objective_ratio | base_optimized_objective | hybrid_optimized_objective | base_recoverable_fraction | hybrid_recoverable_fraction | delta_recoverable_fraction | base_total_intervention_cost | hybrid_total_intervention_cost | base_selected_action_count | hybrid_selected_action_count | selected_action_jaccard | selected_unit_jaccard | base_selected_unit_footprint_mass | hybrid_selected_unit_footprint_mass | delta_selected_unit_footprint_mass | base_selected_cost_weighted_footprint_score | hybrid_selected_cost_weighted_footprint_score | delta_selected_cost_weighted_footprint_score | v34_finite_action_value_spearman | v34_finite_top5pct_action_jaccard | v34_delta_finite_top5pct_units_footprint_mass | v34_base_finite_top5pct_units_footprint_mass | v34_hybrid_finite_top5pct_units_footprint_mass | footprint_zone_count | error | footprint_relative_max | footprint_relative_gini | hybrid_relative_max | hybrid_relative_gini | od_footprint_relative_cosine | od_hybrid_relative_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 484 | 2024-07-22 14:00:00 | 207 | OPTIMAL | OPTIMAL | 1.6230 | 0.1084 | 0.0957 | 0.8828 | 0.0942 | 0.0820 | 0.1310 | 0.1439 | 0.0129 | 1.4296 | 1.4296 | 442.0000 | 497.0000 | 0.8818 | 0.8958 | 0.3018 | 0.3180 | 0.0161 | 0.0040 | 0.0048 | 0.0008 | 0.9520 | 0.4485 | 0.2985 | 0.0192 | 0.3178 | 198.0000 |  | 12.0980 | 0.6549 | 6.5750 | 0.3036 | 0.4564 | 0.7462 |
| Chicago | 157 | 2019-07-29 10:00:00 | 882 | OPTIMAL | OPTIMAL | 24.2710 | 0.2829 | 0.2940 | 1.0394 | 0.2448 | 0.2560 | 0.1347 | 0.1295 | -0.0052 | 12.0773 | 12.0773 | 1105.0000 | 1101.0000 | 0.9910 | 0.9899 | 0.1924 | 0.1917 | -0.0007 | 0.0023 | 0.0023 | 0.0001 | 0.9729 | 0.5561 | 0.2585 | 0.0942 | 0.3526 | 814.0000 |  | 13.5195 | 0.6084 | 7.3111 | 0.2936 | 0.4693 | 0.7462 |
| Dallas | 64 | 2019-04-30 18:00:00 | 366 | OPTIMAL | OPTIMAL | 3.1750 | 0.6217 | 0.5969 | 0.9601 | 0.5876 | 0.5628 | 0.0549 | 0.0571 | 0.0023 | 4.4114 | 4.4114 | 212.0000 | 213.0000 | 0.9953 | 1.0000 | 0.0484 | 0.0484 | 0.0000 | 0.0012 | 0.0012 | 0.0000 | 0.9466 | 0.4218 | 0.3696 | 0.0209 | 0.3905 | 363.0000 |  | 14.0411 | 0.6782 | 7.5321 | 0.3586 | 0.3908 | 0.6349 |
| Houston | 52 | 2019-04-04 01:00:00 | 393 | OPTIMAL | OPTIMAL | 9.7180 | 0.0896 | 0.0835 | 0.9329 | 0.0798 | 0.0740 | 0.1092 | 0.1147 | 0.0055 | 1.5050 | 1.5050 | 441.0000 | 474.0000 | 0.9023 | 0.9500 | 0.0904 | 0.0924 | 0.0020 | 0.0021 | 0.0022 | 0.0000 | 0.9545 | 0.5492 | 0.1599 | 0.0375 | 0.1974 | 386.0000 |  | 12.0645 | 0.6017 | 6.5343 | 0.3039 | 0.4952 | 0.7545 |
| New York | 117 | 2019-06-20 07:00:00 | 1940 | OPTIMAL | ERROR |  | 0.0286 | 0.0280 | 0.9788 | 0.0251 |  | 0.1233 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.9584 | 0.5475 | 0.3691 | 0.0494 | 0.4185 |  | Gurobi did not return a feasible solution. Status: TIME_LIMIT | 17.3211 | 0.7193 | 9.2299 | 0.3621 | 0.4041 | 0.6645 |
| Philadelphia | 641 | 2023-07-07 11:00:00 | 642 | OPTIMAL | OPTIMAL | 114.4300 | 0.0050 | 0.0051 | 1.0158 | 0.0040 | 0.0042 | 0.1860 | 0.1731 | -0.0128 | 0.2183 | 0.2183 | 549.0000 | 642.0000 | 0.8267 | 0.8553 | 0.0495 | 0.0591 | 0.0096 | 0.0007 | 0.0008 | 0.0001 | 0.9596 | 0.4080 | 0.4902 | 0.0207 | 0.5109 | 612.0000 |  | 14.3735 | 0.7044 | 7.7139 | 0.3821 | 0.3914 | 0.6211 |
| San Antonio | 499 | 2024-07-22 20:00:00 | 304 | OPTIMAL | OPTIMAL | 46.2430 | 0.0216 | 0.0206 | 0.9554 | 0.0185 | 0.0176 | 0.1418 | 0.1463 | 0.0045 | 0.4080 | 0.4080 | 358.0000 | 381.0000 | 0.9396 | 0.9318 | 0.1588 | 0.1726 | 0.0138 | 0.0040 | 0.0049 | 0.0008 | 0.9827 | 0.6810 | 0.1819 | 0.0614 | 0.2433 | 302.0000 |  | 12.1359 | 0.5922 | 6.5783 | 0.2905 | 0.5046 | 0.7725 |

## City Summary

| city | n_events | mean_base_selected_unit_footprint_mass | mean_hybrid_selected_unit_footprint_mass | mean_delta_selected_unit_footprint_mass | mean_base_selected_cost_weighted_footprint_score | mean_hybrid_selected_cost_weighted_footprint_score | mean_delta_selected_cost_weighted_footprint_score | mean_selected_action_jaccard | mean_selected_unit_jaccard | mean_delta_recoverable_fraction | mean_v34_delta_finite_top5pct_units_footprint_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 1 | 0.3018 | 0.3180 | 0.0161 | 0.0040 | 0.0048 | 0.0008 | 0.8818 | 0.8958 | 0.0129 | 0.2985 |
| San Antonio | 1 | 0.1588 | 0.1726 | 0.0138 | 0.0040 | 0.0049 | 0.0008 | 0.9396 | 0.9318 | 0.0045 | 0.1819 |
| Philadelphia | 1 | 0.0495 | 0.0591 | 0.0096 | 0.0007 | 0.0008 | 0.0001 | 0.8267 | 0.8553 | -0.0128 | 0.4902 |
| Houston | 1 | 0.0904 | 0.0924 | 0.0020 | 0.0021 | 0.0022 | 0.0000 | 0.9023 | 0.9500 | 0.0055 | 0.1599 |
| Dallas | 1 | 0.0484 | 0.0484 | 0.0000 | 0.0012 | 0.0012 | 0.0000 | 0.9953 | 1.0000 | 0.0023 | 0.3696 |
| Chicago | 1 | 0.1924 | 0.1917 | -0.0007 | 0.0023 | 0.0023 | 0.0001 | 0.9910 | 0.9899 | -0.0052 | 0.2585 |

## 解释

这版结果显示：V34 的 magnitude-aware first-order footprint shift 真实存在，但只有很小一部分转化成完整 LP 的 selected support。full LP 在代表事件中仍然高度接近 base OD-template support，说明预算约束、deployment caps、response delay、三类资源的替代关系和 diminishing returns 会重新吸收大部分 footprint signal。

因此目前可以写成一个重要边界：event-specific footprint 会改变 magnitude-aware recoverability field，但在当前管理参数和小段资源设定下，还不能直接推出最终优化投放会大幅转向 observed footprint。后续若要把 footprint-specific recovery law 作为主结论，需要进一步做全量 hybrid LP、残差 law closure，或让 b0/h 的 region-level footprint 与资源上限、成本、响应机制一起重新标定。

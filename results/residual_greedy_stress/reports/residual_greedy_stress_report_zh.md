# Residual Greedy Stress Test V8

## 这一版做了什么

V7 证明 residual finite greedy 在 base scenario 中几乎闭合 LP optimum。V8 进一步检验这个 finite-budget law 是否稳定：同一批 105 个 city-event，在 low/high budget、2/4 小时额外响应延迟、以及 scarce-and-late 条件下重新运行 residual replanning，并与 V5 的 static small-signal greedy replay 对比。

本版每次 replan 最多使用总预算的 10.00%，因此它不是一次性排序，而是在同一个 scenario 内反复更新 residual state。

## 关键结果

- base: static = 0.8228, residual = 0.9660, improvement = 0.1432
- low budget: static = 0.4959, residual = 0.5711, improvement = 0.0752
- high budget: static = 1.3293, residual = 1.5916, improvement = 0.2623
- delay 4h: static = 0.6804, residual = 0.7734, improvement = 0.0930

注意：非 base scenario 没有重新求解 Gurobi optimum，因此表中的 gain 都以 base LP gain 为归一化参照。它检验的是 law-guided policy 在参数扰动下相对 static greedy 是否稳定，而不是宣称这些新 scenario 下已经达到各自的 LP optimum。

## Scenario Summary

| policy_scenario   |   budget_scale |   delay_add_hours |   n_events |   mean_static_fraction_of_base_lp_gain |   mean_residual_fraction_of_base_lp_gain |   median_residual_fraction_of_base_lp_gain |   mean_residual_minus_static |   median_residual_minus_static |   positive_improvement_share |   mean_static_recoverable_fraction |   mean_residual_recoverable_fraction |   mean_residual_selected_actions |   mean_residual_allocated_cost |
|:------------------|---------------:|------------------:|-----------:|---------------------------------------:|-----------------------------------------:|-------------------------------------------:|-----------------------------:|-------------------------------:|-----------------------------:|-----------------------------------:|-------------------------------------:|---------------------------------:|-------------------------------:|
| low_budget        |            0.5 |                 0 |        105 |                                 0.4959 |                                   0.5711 |                                     0.5785 |                      0.07521 |                        0.05456 |                       0.9048 |                            0.06513 |                              0.07695 |                            377.1 |                          3.073 |
| base              |            1   |                 0 |        105 |                                 0.8228 |                                   0.966  |                                     0.9894 |                      0.1432  |                        0.111   |                       0.9143 |                            0.1073  |                              0.1294  |                            837.2 |                          6.146 |
| high_budget       |            2   |                 0 |        105 |                                 1.329  |                                   1.592  |                                     1.615  |                      0.2623  |                        0.2206  |                       0.9238 |                            0.1722  |                              0.2121  |                           1805   |                         12.29  |
| delay_2h          |            1   |                 2 |        105 |                                 0.756  |                                   0.878  |                                     0.8837 |                      0.122   |                        0.09587 |                       0.8857 |                            0.09959 |                              0.1185  |                            936.8 |                          6.146 |
| delay_4h          |            1   |                 4 |        105 |                                 0.6804 |                                   0.7734 |                                     0.7775 |                      0.09297 |                        0.07182 |                       0.8476 |                            0.09072 |                              0.1051  |                           1029   |                          6.146 |
| scarce_and_late   |            0.5 |                 2 |        105 |                                 0.4575 |                                   0.5222 |                                     0.5299 |                      0.06478 |                        0.04942 |                       0.8476 |                            0.06064 |                              0.07085 |                            427.3 |                          3.073 |

## City-Scenario Summary

| city         | policy_scenario   |   n_events |   mean_static_fraction_of_base_lp_gain |   mean_residual_fraction_of_base_lp_gain |   mean_residual_minus_static |   positive_improvement_share |
|:-------------|:------------------|-----------:|---------------------------------------:|-----------------------------------------:|-----------------------------:|-----------------------------:|
| Austin       | low_budget        |         15 |                                 0.5019 |                                   0.575  |                    0.0731    |                       1      |
| Chicago      | low_budget        |         13 |                                 0.4833 |                                   0.587  |                    0.1037    |                       1      |
| Dallas       | low_budget        |         10 |                                 0.5545 |                                   0.5546 |                    0.0001428 |                       0.2    |
| Houston      | low_budget        |          4 |                                 0.5576 |                                   0.5796 |                    0.02198   |                       1      |
| New York     | low_budget        |         24 |                                 0.5011 |                                   0.5668 |                    0.06565   |                       0.9583 |
| Philadelphia | low_budget        |         20 |                                 0.4744 |                                   0.571  |                    0.09656   |                       0.95   |
| San Antonio  | low_budget        |         19 |                                 0.4718 |                                   0.5695 |                    0.0977    |                       1      |
| Austin       | base              |         15 |                                 0.8696 |                                   0.9816 |                    0.112     |                       1      |
| Chicago      | base              |         13 |                                 0.7632 |                                   0.97   |                    0.2068    |                       1      |
| Dallas       | base              |         10 |                                 0.9979 |                                   0.9995 |                    0.001584  |                       0.3    |
| Houston      | base              |          4 |                                 0.9303 |                                   0.9926 |                    0.06222   |                       1      |
| New York     | base              |         24 |                                 0.8253 |                                   0.9486 |                    0.1233    |                       0.9583 |
| Philadelphia | base              |         20 |                                 0.7517 |                                   0.9489 |                    0.1971    |                       0.95   |
| San Antonio  | base              |         19 |                                 0.7834 |                                   0.9677 |                    0.1843    |                       1      |
| Austin       | high_budget       |         15 |                                 1.419  |                                   1.635  |                    0.2158    |                       1      |
| Chicago      | high_budget       |         13 |                                 1.184  |                                   1.547  |                    0.3626    |                       1      |
| Dallas       | high_budget       |         10 |                                 1.739  |                                   1.745  |                    0.005885  |                       0.3    |
| Houston      | high_budget       |          4 |                                 1.551  |                                   1.667  |                    0.1161    |                       1      |
| New York     | high_budget       |         24 |                                 1.319  |                                   1.538  |                    0.2189    |                       0.9583 |
| Philadelphia | high_budget       |         20 |                                 1.191  |                                   1.554  |                    0.363     |                       1      |
| San Antonio  | high_budget       |         19 |                                 1.254  |                                   1.599  |                    0.345     |                       1      |
| Austin       | delay_2h          |         15 |                                 0.8216 |                                   0.9179 |                    0.0963    |                       1      |
| Chicago      | delay_2h          |         13 |                                 0.6927 |                                   0.8757 |                    0.183     |                       1      |
| Dallas       | delay_2h          |         10 |                                 0.8241 |                                   0.8241 |                   -3.089e-11 |                       0      |
| Houston      | delay_2h          |          4 |                                 0.8876 |                                   0.9029 |                    0.01536   |                       1      |
| New York     | delay_2h          |         24 |                                 0.7828 |                                   0.8682 |                    0.08536   |                       0.9583 |
| Philadelphia | delay_2h          |         20 |                                 0.6983 |                                   0.8796 |                    0.1813    |                       0.95   |
| San Antonio  | delay_2h          |         19 |                                 0.711  |                                   0.8819 |                    0.1709    |                       1      |
| Austin       | delay_4h          |         15 |                                 0.7679 |                                   0.8352 |                    0.06733   |                       1      |
| Chicago      | delay_4h          |         13 |                                 0.6291 |                                   0.7712 |                    0.1421    |                       1      |
| Dallas       | delay_4h          |         10 |                                 0.6718 |                                   0.6718 |                    5.708e-12 |                       0      |
| Houston      | delay_4h          |          4 |                                 0.7834 |                                   0.7851 |                    0.001675  |                       0.5    |
| New York     | delay_4h          |         24 |                                 0.7247 |                                   0.7695 |                    0.04485   |                       0.875  |
| Philadelphia | delay_4h          |         20 |                                 0.6331 |                                   0.7904 |                    0.1572    |                       0.95   |
| San Antonio  | delay_4h          |         19 |                                 0.6234 |                                   0.7642 |                    0.1409    |                       1      |
| Austin       | scarce_and_late   |         15 |                                 0.4852 |                                   0.5399 |                    0.05469   |                       1      |
| Chicago      | scarce_and_late   |         13 |                                 0.4423 |                                   0.5365 |                    0.09421   |                       1      |
| Dallas       | scarce_and_late   |         10 |                                 0.4569 |                                   0.4569 |                    1.156e-11 |                       0      |
| Houston      | scarce_and_late   |          4 |                                 0.5205 |                                   0.5264 |                    0.00592   |                       1      |
| New York     | scarce_and_late   |         24 |                                 0.4755 |                                   0.5197 |                    0.04415   |                       0.7917 |
| Philadelphia | scarce_and_late   |         20 |                                 0.4364 |                                   0.5325 |                    0.09611   |                       0.95   |
| San Antonio  | scarce_and_late   |         19 |                                 0.4324 |                                   0.5246 |                    0.09215   |                       1      |

## Replan Summary

| city    |   event_id | policy_scenario   |   budget_scale |   delay_add_hours |   n_replans |   mean_pass_cost |   final_remaining_total |
|:--------|-----------:|:------------------|---------------:|------------------:|------------:|-----------------:|------------------------:|
| Austin  |        478 | base              |              1 |                 0 |          10 |          0.1459  |               5.149e-15 |
| Austin  |        479 | base              |              1 |                 0 |          10 |          0.2027  |               0         |
| Austin  |        480 | base              |              1 |                 0 |          10 |          0.1453  |               8.188e-16 |
| Austin  |        481 | base              |              1 |                 0 |          10 |          0.06244 |               6.939e-18 |
| Austin  |        482 | base              |              1 |                 0 |          10 |          0.2117  |               1.965e-15 |
| Austin  |        483 | base              |              1 |                 0 |          10 |          0.1029  |               0         |
| Austin  |        484 | base              |              1 |                 0 |          10 |          0.143   |               0         |
| Austin  |        485 | base              |              1 |                 0 |          10 |          0.2128  |               3.123e-17 |
| Austin  |        486 | base              |              1 |                 0 |          10 |          0.2883  |               0         |
| Austin  |        487 | base              |              1 |                 0 |          10 |          0.2479  |               7.633e-17 |
| Austin  |        488 | base              |              1 |                 0 |          10 |          0.09527 |               1.556e-15 |
| Austin  |        489 | base              |              1 |                 0 |          10 |          0.04441 |               4.648e-16 |
| Austin  |        490 | base              |              1 |                 0 |          10 |          0.1205  |               0         |
| Austin  |        491 | base              |              1 |                 0 |          10 |          0.02786 |               1.753e-16 |
| Austin  |        492 | base              |              1 |                 0 |          10 |          0.1517  |               1.951e-15 |
| Chicago |        145 | base              |              1 |                 0 |          10 |          0.4923  |               0         |
| Chicago |        146 | base              |              1 |                 0 |          10 |          0.1604  |               1.427e-15 |
| Chicago |        147 | base              |              1 |                 0 |          10 |          0.1521  |               6.64e-16  |
| Chicago |        149 | base              |              1 |                 0 |          10 |          0.06505 |               4.84e-16  |
| Chicago |        150 | base              |              1 |                 0 |          10 |          0.3463  |               0         |

## 科学解释

Residual law 在低预算、延迟和 scarce-and-late 场景下仍系统性优于 static small-signal greedy，说明 V7 发现不是 base 参数的偶然结果。它支持一个更稳定的 finite-budget law：恢复价值应当写成 `value(segment | residual state, remaining budget, remaining time)`，而不是只写成事件开始时的固定 action score。

高预算场景中 residual gain 可以超过 base LP gain，这是正常的，因为归一化分母仍然是 base LP optimized gain，而不是 high-budget 重新优化后的 optimum。下一步如果要做完全闭合的 robustness，需要对若干代表性 budget/delay scenario 重新求解 Gurobi LP，比较 residual policy 与各自 scenario optimum。

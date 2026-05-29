# Parameter-Ensemble LP Optimum Validation V24

## 本版回答的问题

V23 只在 action-token field 上重算 first-order small-signal target，检验的是局部 action-value law 的参数稳定性。V24 进一步对代表性 city-event 重新求解参数扰动后的完整 Gurobi LP optimum，并把 static small-signal greedy 与 residual finite greedy 放回同一参数场景中 replay。这样分母不再是 base LP gain，而是每个 eta/cost/delay/channel-favored 场景自己的 LP gain。

这仍然不是全 105 event、全 11 parameter scenarios 的最终闭合；它是一个代表性 full-LP closure，用来检验 V23 的 first-order 发现能否延伸到 finite-budget residual interaction。

## 求解覆盖

- selected events: 4
- parameter scenarios: cheap_all, slow_response_4h, R_favored, C_favored, S_favored
- successful LP scenario rows: 20
- LP status counts: {'OPTIMAL': 20}
- mean/max LP runtime seconds: 24.19 / 54.50
- replan budget share: 5.00%

## 关键结果

- static small-signal mean policy/LP gain = 0.7451
- residual finite greedy mean policy/LP gain = 0.9662
- residual-minus-static mean = 0.2211
- positive residual improvement share = 1.0000
- weakest residual scenario = R_favored at 0.9516

解释：如果 residual finite greedy 仍接近 1，说明 action-level law 经过 residual state 更新后不仅在 base regime 成立，也能在 eta/cost/delay/channel-favored 参数扰动下接近对应场景的真实 LP 上界。若某些场景 gap 变大，说明参数改变后 LP 的全局同时优化、period budget shadow price 或 R/C/S 替代关系变得更重要。

## Representative Events

| city         |   event_id | event_start         | selection_note                                   |   n_units |   runtime_seconds |   baseline_objective |   optimized_objective |   recoverable_fraction |   static_fraction_of_lp_gain |   residual_fraction_of_lp_gain |   residual_gain_improvement_over_static |   residual_gap_to_lp |   total_budget |   weighted_b0 |   weighted_h_total |   event_total_precip |   event_peak_precip |   event_peak_positive_abnormal_deficit |
|:-------------|-----------:|:--------------------|:-------------------------------------------------|----------:|------------------:|---------------------:|----------------------:|-----------------------:|-----------------------------:|-------------------------------:|----------------------------------------:|---------------------:|---------------:|--------------:|-------------------:|---------------------:|--------------------:|---------------------------------------:|
| San Antonio  |        500 | 2024-07-23 00:00:00 | highest_residual_improvement_under_runtime_guard |       304 |            15.67  |              0.1073  |               0.08958 |                 0.1653 |                       0.4677 |                         0.9379 |                                  0.4702 |             0.0621   |         3.817  |      0.00484  |           0.05206  |             0.01706  |            0.0105   |                               0.0397   |
| Austin       |        491 | 2024-07-27 07:00:00 | highest_residual_improvement_under_runtime_guard |       207 |             5.352 |              0.02247 |               0.01925 |                 0.1435 |                       0.7326 |                         0.9798 |                                  0.2472 |             0.02017  |         0.2786 |      0.001608 |           0.004461 |             0.005906 |            0.005906 |                               0.002076 |
| Chicago      |        158 | 2019-07-29 13:00:00 | highest_residual_improvement_under_runtime_guard |       882 |            15.16  |              0.2415  |               0.2112  |                 0.1252 |                       0.8774 |                         0.9948 |                                  0.1175 |             0.005187 |         9.754  |      0.03362  |           0.01681  |             0.4134   |            0.3819   |                               0.03362  |
| Philadelphia |        644 | 2023-07-09 11:00:00 | highest_residual_improvement_under_runtime_guard |       642 |            13.12  |              0.1111  |               0.0946  |                 0.1481 |                       0.8776 |                         0.9865 |                                  0.1089 |             0.01347  |         5.051  |      0.001213 |           0.03434  |             0.8091   |            0.4941   |                               0.02621  |

## Scenario Summary

| parameter_scenario   | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |   median_gap_to_scenario_lp_gain |   mean_recoverable_fraction |   mean_allocated_cost |   mean_selected_action_count |
|:---------------------|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|---------------------------------:|----------------------------:|----------------------:|-----------------------------:|
| C_favored            | residual_finite_greedy     |                   4 |                              0.9601 |                                0.9593 |                        0.03991 |                          0.04066 |                     0.1875  |                 4.725 |                        712.2 |
| C_favored            | static_small_signal_greedy |                   4 |                              0.7734 |                                0.8192 |                        0.2266  |                          0.1808  |                     0.149   |                 4.725 |                        629.5 |
| R_favored            | residual_finite_greedy     |                   4 |                              0.9516 |                                0.9466 |                        0.0484  |                          0.05336 |                     0.1379  |                 4.725 |                        466.5 |
| R_favored            | static_small_signal_greedy |                   4 |                              0.5551 |                                0.602  |                        0.4449  |                          0.398   |                     0.07997 |                 4.725 |                        283   |
| S_favored            | residual_finite_greedy     |                   4 |                              0.9716 |                                0.9759 |                        0.02837 |                          0.02413 |                     0.1543  |                 4.725 |                       1602   |
| S_favored            | static_small_signal_greedy |                   4 |                              0.8429 |                                0.9302 |                        0.1571  |                          0.06977 |                     0.1319  |                 4.725 |                       1501   |
| cheap_all            | residual_finite_greedy     |                   4 |                              0.9729 |                                0.9798 |                        0.02711 |                          0.02019 |                     0.174   |                 4.725 |                       1181   |
| cheap_all            | static_small_signal_greedy |                   4 |                              0.7425 |                                0.8022 |                        0.2575  |                          0.1978  |                     0.1311  |                 4.725 |                        875.5 |
| slow_response_4h     | residual_finite_greedy     |                   4 |                              0.9748 |                                0.9832 |                        0.02516 |                          0.01679 |                     0.1189  |                 4.725 |                        919   |
| slow_response_4h     | static_small_signal_greedy |                   4 |                              0.8117 |                                0.84   |                        0.1883  |                          0.16    |                     0.09846 |                 4.725 |                        770.8 |

## City Summary

| city         | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |
|:-------------|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|
| Chicago      | residual_finite_greedy     |                   5 |                              0.9928 |                                0.9945 |                       0.007169 |
| Philadelphia | residual_finite_greedy     |                   5 |                              0.9734 |                                0.9848 |                       0.02661  |
| Austin       | residual_finite_greedy     |                   5 |                              0.9592 |                                0.967  |                       0.04075  |
| San Antonio  | residual_finite_greedy     |                   5 |                              0.9394 |                                0.9374 |                       0.06062  |
| Chicago      | static_small_signal_greedy |                   5 |                              0.8812 |                                0.8811 |                       0.1188   |
| Philadelphia | static_small_signal_greedy |                   5 |                              0.8802 |                                0.9084 |                       0.1198   |
| Austin       | static_small_signal_greedy |                   5 |                              0.7394 |                                0.7464 |                       0.2606   |
| San Antonio  | static_small_signal_greedy |                   5 |                              0.4797 |                                0.5139 |                       0.5203   |

## Hardest Residual Cases

| city         |   event_id | parameter_scenario   |   residual_finite_greedy |   static_small_signal_greedy |   residual_minus_static |
|:-------------|-----------:|:---------------------|-------------------------:|-----------------------------:|------------------------:|
| Austin       |        491 | C_favored            |                   0.9263 |                       0.73   |                 0.1964  |
| Philadelphia |        644 | R_favored            |                   0.9301 |                       0.7018 |                 0.2283  |
| San Antonio  |        500 | slow_response_4h     |                   0.9347 |                       0.5919 |                 0.3429  |
| San Antonio  |        500 | S_favored            |                   0.9362 |                       0.5513 |                 0.3849  |
| San Antonio  |        500 | R_favored            |                   0.9374 |                       0.2504 |                 0.687   |
| San Antonio  |        500 | cheap_all            |                   0.9375 |                       0.491  |                 0.4465  |
| San Antonio  |        500 | C_favored            |                   0.9511 |                       0.5139 |                 0.4371  |
| Austin       |        491 | R_favored            |                   0.9559 |                       0.5022 |                 0.4537  |
| Austin       |        491 | S_favored            |                   0.967  |                       0.9194 |                 0.04758 |
| Philadelphia |        644 | C_favored            |                   0.9676 |                       0.9084 |                 0.05917 |
| Austin       |        491 | cheap_all            |                   0.9733 |                       0.7464 |                 0.2269  |
| Austin       |        491 | slow_response_4h     |                   0.9737 |                       0.7989 |                 0.1748  |

## 当前边界

V24 的参数场景改变 eta、cost 和 delay，但没有改变 OD demand、事件空间 footprint、自然恢复或预算制度本身。eta scaling 也沿用了既有 perturbation 逻辑：deployment cap 保持 base calibration，效率改变会同时改变单位资源效果和可达到的最大有效效果。因此它是管理参数敏感性检验，而不是真实干预因果识别。

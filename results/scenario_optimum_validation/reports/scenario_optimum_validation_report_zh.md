# Scenario-Specific LP Optimum Validation V9

## 这一版回答什么问题

V8 已经证明 residual finite greedy 在预算和延迟扰动下稳定优于 static small-signal greedy，但 V8 的非 base 场景没有重新求解 Gurobi optimum。V9 补上这个闭合检验：对一组代表性 city-event，在 low/high budget、delay 和 scarce-and-late 场景下重新求 scenario-specific LP optimum，然后比较 static 与 residual law policy 能获得各自 optimum 的多少。

为了控制求解成本，本版不是全量 105 事件闭合，而是每城优先选择 base residual-over-static improvement 最高、且 base LP runtime 不超过 180 秒的代表事件。这个设计的目的不是替代全量 robustness，而是先验证 V7/V8 的 finite-budget law 在真正非 base LP optimum 下是否仍然成立。

## 代表事件

| city         |   event_id |   interaction_rank_in_city |   runtime_seconds |   static_fraction_of_lp_gain |   residual_fraction_of_lp_gain |   residual_gain_improvement_over_static | selection_note                               |
|:-------------|-----------:|---------------------------:|------------------:|-----------------------------:|-------------------------------:|----------------------------------------:|:---------------------------------------------|
| Austin       |        491 |                          1 |             5.352 |                       0.7326 |                         0.9798 |                                 0.2472  | highest_interaction_gain_under_runtime_guard |
| Chicago      |        147 |                          4 |            57.12  |                       0.7105 |                         0.9731 |                                 0.2626  | highest_interaction_gain_under_runtime_guard |
| Dallas       |         58 |                          1 |             2.943 |                       0.9841 |                         0.9973 |                                 0.01318 | highest_interaction_gain_under_runtime_guard |
| Houston      |         61 |                          1 |             2.875 |                       0.8962 |                         0.9963 |                                 0.1001  | highest_interaction_gain_under_runtime_guard |
| New York     |        106 |                          7 |           168.1   |                       0.6867 |                         0.9418 |                                 0.2552  | highest_interaction_gain_under_runtime_guard |
| Philadelphia |        642 |                          1 |            94.25  |                       0.3831 |                         0.8762 |                                 0.4932  | highest_interaction_gain_under_runtime_guard |
| San Antonio  |        506 |                          1 |            30.02  |                       0.3975 |                         0.8905 |                                 0.493   | highest_interaction_gain_under_runtime_guard |

## 求解覆盖

- selected events: 7
- scenarios: low_budget, high_budget, delay_4h, scarce_and_late
- LP jobs with returned rows: 28
- LP status counts: {'OPTIMAL': 23, 'ERROR': 5}
- mean LP runtime seconds: 61.98
- max LP runtime seconds: 274.35

## Policy vs Scenario LP Optimum

| policy_scenario   |   budget_scale |   delay_add_hours | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |   median_gap_to_scenario_lp_gain |   mean_recoverable_fraction |   mean_allocated_cost |   mean_selected_action_count |
|:------------------|---------------:|------------------:|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|---------------------------------:|----------------------------:|----------------------:|-----------------------------:|
| delay_4h          |            1   |                 4 | residual_finite_greedy     |                   7 |                              0.9156 |                                0.9737 |                        0.08445 |                          0.02629 |                     0.1042  |                 3.209 |                        694.1 |
| delay_4h          |            1   |                 4 | static_small_signal_greedy |                   7 |                              0.7489 |                                0.7989 |                        0.2511  |                          0.2011  |                     0.08141 |                 3.209 |                        552.4 |
| high_budget       |            2   |                 0 | residual_finite_greedy     |                   4 |                              0.9549 |                                0.9815 |                        0.04514 |                          0.01852 |                     0.2067  |                 6.062 |                        519.2 |
| high_budget       |            2   |                 0 | static_small_signal_greedy |                   4 |                              0.7353 |                                0.8144 |                        0.2647  |                          0.1856  |                     0.1523  |                 6.062 |                        399.8 |
| low_budget        |            0.5 |                 0 | residual_finite_greedy     |                   6 |                              0.9602 |                                0.9845 |                        0.03984 |                          0.01545 |                     0.07709 |                 1.155 |                        130.3 |
| low_budget        |            0.5 |                 0 | static_small_signal_greedy |                   6 |                              0.6948 |                                0.7393 |                        0.3052  |                          0.2607  |                     0.05214 |                 1.155 |                        102.7 |
| scarce_and_late   |            0.5 |                 2 | residual_finite_greedy     |                   6 |                              0.9427 |                                0.9756 |                        0.05729 |                          0.0244  |                     0.07065 |                 1.745 |                        303.2 |
| scarce_and_late   |            0.5 |                 2 | static_small_signal_greedy |                   6 |                              0.7122 |                                0.7868 |                        0.2878  |                          0.2132  |                     0.05006 |                 1.745 |                        213.7 |

## 关键闭合结论

- mean static / scenario LP gain: 0.7228
- mean residual / scenario LP gain: 0.9411
- mean residual-minus-static: 0.2183
- positive residual improvement share: 0.9130

解释：这里的分母已经不再是 base LP gain，而是每个 budget/delay 场景重新求解得到的 scenario-specific LP gain。因此它比 V8 更直接地回答 residual finite-budget law 是否接近对应场景的真实优化上界。

## City Summary

| city         | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |
|:-------------|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|
| Dallas       | residual_finite_greedy     |                   4 |                              0.9985 |                                0.9999 |                       0.00145  |
| Houston      | residual_finite_greedy     |                   4 |                              0.998  |                                0.9984 |                       0.002042 |
| New York     | residual_finite_greedy     |                   2 |                              0.9871 |                                0.9871 |                       0.0129   |
| Austin       | residual_finite_greedy     |                   4 |                              0.9766 |                                0.974  |                       0.0234   |
| Chicago      | residual_finite_greedy     |                   2 |                              0.9713 |                                0.9713 |                       0.02867  |
| Philadelphia | residual_finite_greedy     |                   3 |                              0.904  |                                0.9115 |                       0.09599  |
| San Antonio  | residual_finite_greedy     |                   4 |                              0.7811 |                                0.8279 |                       0.2189   |
| Dallas       | static_small_signal_greedy |                   4 |                              0.9901 |                                0.9987 |                       0.009897 |
| Houston      | static_small_signal_greedy |                   4 |                              0.9446 |                                0.9495 |                       0.0554   |
| New York     | static_small_signal_greedy |                   2 |                              0.9007 |                                0.9007 |                       0.09931  |
| Austin       | static_small_signal_greedy |                   4 |                              0.747  |                                0.7451 |                       0.253    |
| Chicago      | static_small_signal_greedy |                   2 |                              0.7439 |                                0.7439 |                       0.2561   |
| Philadelphia | static_small_signal_greedy |                   3 |                              0.4436 |                                0.4418 |                       0.5564   |
| San Antonio  | static_small_signal_greedy |                   4 |                              0.3197 |                                0.3218 |                       0.6803   |

## 下一步

如果本版结果显示 residual law 在代表性非 base 场景中仍接近 scenario optimum，下一步就可以扩大到更多事件，或者转向提取更明确的 event-level decision-criticality law。若某些场景 residual gap 明显，则需要分析 gap 是否来自 period budget shadow price、R/C/S 互补关系，还是 LP 全局同时优化带来的剩余优势。

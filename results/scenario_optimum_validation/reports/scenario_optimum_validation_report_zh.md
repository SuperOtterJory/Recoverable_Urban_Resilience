# Scenario-Specific LP Optimum Validation V31

## 这一版回答什么问题

V31 updates the earlier V9 scenario-specific closure by rerunning hard representative non-base LPs with longer solve budgets. The closure set now has 26 optimal rows out of 28; the two unresolved rows are New York event 106 under low_budget and high_budget.

为了控制求解成本，本版仍不是全量 105 事件闭合，而是每城优先选择 base residual-over-static improvement 最高、且 base LP runtime 不超过 180 秒的代表事件。这个设计的目的不是替代全量 robustness，而是先验证 V7/V8 的 finite-budget law 在真正非 base LP optimum 下是否仍然成立。

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
- scenarios: delay_4h, high_budget, low_budget, scarce_and_late
- LP jobs with returned rows: 28
- LP status counts: {'OPTIMAL': 26, 'ERROR': 2}
- mean LP runtime seconds: 106.45
- max LP runtime seconds: 717.35
- unresolved cases: New York event 106 high_budget; New York event 106 low_budget

## Policy vs Scenario LP Optimum

| policy_scenario   |   budget_scale |   delay_add_hours | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |   median_gap_to_scenario_lp_gain |   mean_recoverable_fraction |   mean_allocated_cost |   mean_selected_action_count |
|:------------------|---------------:|------------------:|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|---------------------------------:|----------------------------:|----------------------:|-----------------------------:|
| delay_4h          |            1   |                 4 | residual_finite_greedy     |                   7 |                              0.9156 |                                0.9737 |                        0.08445 |                          0.02629 |                     0.1042  |                 3.209 |                        694.1 |
| delay_4h          |            1   |                 4 | static_small_signal_greedy |                   7 |                              0.7489 |                                0.7989 |                        0.2511  |                          0.2011  |                     0.08141 |                 3.209 |                        552.4 |
| high_budget       |            2   |                 0 | residual_finite_greedy     |                   6 |                              0.9414 |                                0.9682 |                        0.05857 |                          0.03176 |                     0.2198  |                 4.619 |                        679.2 |
| high_budget       |            2   |                 0 | static_small_signal_greedy |                   6 |                              0.6637 |                                0.7121 |                        0.3363  |                          0.2879  |                     0.1458  |                 4.619 |                        441.5 |
| low_budget        |            0.5 |                 0 | residual_finite_greedy     |                   6 |                              0.9602 |                                0.9845 |                        0.03984 |                          0.01545 |                     0.07709 |                 1.155 |                        130.3 |
| low_budget        |            0.5 |                 0 | static_small_signal_greedy |                   6 |                              0.6948 |                                0.7393 |                        0.3052  |                          0.2607  |                     0.05214 |                 1.155 |                        102.7 |
| scarce_and_late   |            0.5 |                 2 | residual_finite_greedy     |                   7 |                              0.9479 |                                0.9769 |                        0.05212 |                          0.02311 |                     0.07062 |                 1.605 |                        309.3 |
| scarce_and_late   |            0.5 |                 2 | static_small_signal_greedy |                   7 |                              0.7182 |                                0.7538 |                        0.2818  |                          0.2462  |                     0.05066 |                 1.605 |                        216.3 |

## 关键闭合结论

- mean static / scenario LP gain: 0.7085
- mean residual / scenario LP gain: 0.9405
- mean residual-minus-static: 0.2321
- positive residual improvement share: 0.9231

解释：这里的分母已经不再是 base LP gain，而是每个 budget/delay 场景重新求解得到的 scenario-specific LP gain。V31 说明 residual finite-budget law 在 26 个已闭合非 base 场景中仍然接近对应场景的真实优化上界；剩余 New York budget-only 场景是当前计算边界。

## City Summary

| city         | policy                     |   n_event_scenarios |   mean_fraction_of_scenario_lp_gain |   median_fraction_of_scenario_lp_gain |   mean_gap_to_scenario_lp_gain |
|:-------------|:---------------------------|--------------------:|------------------------------------:|--------------------------------------:|-------------------------------:|
| Dallas       | residual_finite_greedy     |                   4 |                              0.9985 |                                0.9999 |                       0.00145  |
| Houston      | residual_finite_greedy     |                   4 |                              0.998  |                                0.9984 |                       0.002042 |
| New York     | residual_finite_greedy     |                   2 |                              0.9871 |                                0.9871 |                       0.0129   |
| Austin       | residual_finite_greedy     |                   4 |                              0.9766 |                                0.974  |                       0.0234   |
| Chicago      | residual_finite_greedy     |                   4 |                              0.9724 |                                0.9734 |                       0.02762  |
| Philadelphia | residual_finite_greedy     |                   4 |                              0.8933 |                                0.8992 |                       0.1067   |
| San Antonio  | residual_finite_greedy     |                   4 |                              0.7811 |                                0.8279 |                       0.2189   |
| Dallas       | static_small_signal_greedy |                   4 |                              0.9901 |                                0.9987 |                       0.009897 |
| Houston      | static_small_signal_greedy |                   4 |                              0.9446 |                                0.9495 |                       0.0554   |
| New York     | static_small_signal_greedy |                   2 |                              0.9007 |                                0.9007 |                       0.09931  |
| Austin       | static_small_signal_greedy |                   4 |                              0.747  |                                0.7451 |                       0.253    |
| Chicago      | static_small_signal_greedy |                   4 |                              0.7299 |                                0.7309 |                       0.2701   |
| Philadelphia | static_small_signal_greedy |                   4 |                              0.4235 |                                0.4087 |                       0.5765   |
| San Antonio  | static_small_signal_greedy |                   4 |                              0.3197 |                                0.3218 |                       0.6803   |

## 下一步

下一步如果需要把 scenario-specific closure 作为中心证据，可以优先为 New York budget-only hard cases 做 decomposition、warm-start 或 solver tuning；否则可将这两个未闭合行明确报告为计算边界，并继续把重点放在结构性 action law 与 event-level decision-criticality 上。

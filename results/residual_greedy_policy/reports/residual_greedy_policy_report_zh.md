# Residual Greedy Policy V7

## 这一版做了什么

V6 说明 finite-budget gap 主要来自一阶 small-signal 排序在完整预算下的局部饱和和动作交互。V7 因此实现了 residual finite greedy：不是一次性按 passive trajectory 的一阶值排序，而是把预算分成若干 replan pass；每一轮先 replay 当前已分配资源，得到 residual `b/rC/rS/ell`，再用 `min(candidate_effect, remaining_loss)` 估计下一段资源的有限段平均边际值。

这个 policy 仍然是解释性 law，不是重新求解 LP。它的核心思想是：

```text
finite_budget_value(segment | current_state)
  ~= sum_future exposure_active
      * min(segment_effect_decay, residual_loss)
      / segment_cost
```

## 主要结果

- static small-signal greedy 平均获得 LP gain 的 0.8228
- residual finite greedy 平均获得 LP gain 的 0.9736，中位数 0.9921
- residual 相比 static 的平均提升为 LP gain 的 0.1508
- residual 有正提升的事件比例为 0.9143
- residual 剩余 gap 为 LP gain 的 0.0264
- 平均 replan 次数为 20.00

## 城市层面

| city         |   n_events |   mean_static_fraction_of_lp_gain |   mean_residual_fraction_of_lp_gain |   mean_residual_gain_improvement |   median_residual_gain_improvement |   mean_residual_gap_to_lp |   mean_static_selected_actions |   mean_residual_selected_actions |
|:-------------|-----------:|----------------------------------:|------------------------------------:|---------------------------------:|-----------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|
| Chicago      |         13 |                            0.7632 |                              0.9758 |                         0.2126   |                           0.1698   |                 0.02417   |                          761.1 |                           1187   |
| Philadelphia |         20 |                            0.7517 |                              0.9616 |                         0.2098   |                           0.2045   |                 0.03842   |                          662.4 |                            939.4 |
| San Antonio  |         19 |                            0.7834 |                              0.9767 |                         0.1932   |                           0.1632   |                 0.02334   |                          427   |                            529.7 |
| New York     |         24 |                            0.8253 |                              0.9564 |                         0.1311   |                           0.07205  |                 0.04358   |                         1121   |                           1429   |
| Austin       |         15 |                            0.8696 |                              0.9889 |                         0.1193   |                           0.1054   |                 0.01108   |                          350.1 |                            416.7 |
| Houston      |          4 |                            0.9303 |                              0.9936 |                         0.06327  |                           0.0587   |                 0.006399  |                          323.5 |                            416.2 |
| Dallas       |         10 |                            0.9979 |                              0.9995 |                         0.001609 |                           4.69e-11 |                 0.0004596 |                          212   |                            214.5 |

## 提升最大的事件

| city         |   event_id |   static_fraction_of_lp_gain |   residual_fraction_of_lp_gain |   residual_gain_improvement_over_static |   residual_gap_to_lp |   static_selected_action_count |   residual_selected_action_count |
|:-------------|-----------:|-----------------------------:|-------------------------------:|----------------------------------------:|---------------------:|-------------------------------:|---------------------------------:|
| Chicago      |        152 |                       0.3966 |                         0.9082 |                                  0.5116 |              0.09176 |                            869 |                             2340 |
| Philadelphia |        642 |                       0.3831 |                         0.8762 |                                  0.4932 |              0.1238  |                             80 |                              144 |
| San Antonio  |        506 |                       0.3975 |                         0.8905 |                                  0.493  |              0.1095  |                             49 |                               79 |
| San Antonio  |        500 |                       0.4677 |                         0.9379 |                                  0.4702 |              0.0621  |                            482 |                              688 |
| Philadelphia |        641 |                       0.4637 |                         0.9192 |                                  0.4554 |              0.08083 |                             83 |                              129 |
| Philadelphia |        640 |                       0.4464 |                         0.8755 |                                  0.4291 |              0.1245  |                            153 |                              225 |
| San Antonio  |        501 |                       0.5878 |                         0.9985 |                                  0.4107 |              0.00147 |                            482 |                              561 |
| New York     |         98 |                       0.581  |                         0.9706 |                                  0.3896 |              0.02941 |                            838 |                             2022 |
| Chicago      |        154 |                       0.5419 |                         0.9307 |                                  0.3889 |              0.06927 |                            516 |                             1138 |
| New York     |        100 |                       0.5284 |                         0.9094 |                                  0.3811 |              0.09057 |                           1316 |                             2830 |
| Chicago      |        149 |                       0.6065 |                         0.9434 |                                  0.3369 |              0.05656 |                            203 |                              322 |
| San Antonio  |        499 |                       0.6096 |                         0.9438 |                                  0.3342 |              0.05617 |                            164 |                              219 |
| New York     |        117 |                       0.6343 |                         0.9579 |                                  0.3236 |              0.04212 |                            314 |                              454 |
| New York     |        115 |                       0.6209 |                         0.9089 |                                  0.288  |              0.09112 |                            303 |                              509 |
| Philadelphia |        647 |                       0.6665 |                         0.9517 |                                  0.2853 |              0.04827 |                            186 |                              282 |

## 解释

如果 residual greedy 明显高于 static greedy，说明 V6 的判断成立：完整预算 law 需要显式考虑 residual state 和有限段截断，而不能只依赖 first-order action score。如果提升有限，则说明 LP 的剩余优势更多来自全局同时优化、period budget shadow price 或 R/C/S 联合互补，而不是简单 residual re-ranking。

下一步可以把 residual score 中的 shadow-price 信息进一步显式化：估计每个小时 period budget 的机会成本，或用 LP dual/shadow price 直接学习 finite-budget allocation law。

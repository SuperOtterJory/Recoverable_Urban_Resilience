# Finite-Budget Allocation Gap V6

## 这一版回答什么问题

V5 已经说明：对 single-action 第一小段资源，small-signal activated law 与 LP 一阶边际值基本一致。但完整预算下，按一阶值贪心分配的 fixed-policy replay 仍低于 LP optimum。因此 V6 专门分析这个剩余差距，目标是从 action-level marginal law 进入 finite-budget allocation law。

核心结论是：一阶边际排序解决的是“第一单位资源投向哪里最值”；LP optimum 解决的是“在总预算、单期预算、部署上限、diminishing returns 和 R/C/S 互相替代下，整组资源如何组合”。二者不是同一个 law。

## 总体结果

- small-signal greedy 平均获得 LP gain 的 0.8228，中位数为 0.8774
- 平均剩余 gap 为 LP gain 的 0.1772
- LP 与 greedy 的 action-cost Jaccard overlap 平均为 0.6127
- greedy first-order proxy 平均比真实 replay recoverable fraction 高 0.0421
- LP 使用的 action 数约为 greedy 的 1.6738 倍

## 机制解释

1. **small-signal proxy 会高估有限预算收益。** 它假设每个 segment 的边际效果都在 passive trajectory 上独立发挥作用；但当多个资源作用到相同或相邻的 loss channel 时，后投放的资源会被已经降低的 `b`、`d`、`ell` 截断。

2. **LP 比一阶 greedy 更会分散组合。** Greedy 会持续吃掉当前最高一阶值 segment；LP 会在容量和互相替代作用下，把资源扩散到更多 action，以避免局部饱和。

3. **剩余 law 不是新的局部一阶 law，而是 allocation interaction law。** 它需要刻画同一 unit/time 上 R、C、S 的替代关系、同一未来 loss channel 被多次处理后的边际下降、以及 period budget 导致的跨时间挤出。

## Gap 相关性最高的机制变量

| target                         | feature                              |   spearman |   abs_spearman |
|:-------------------------------|:-------------------------------------|-----------:|---------------:|
| replay_gap_fraction_of_lp_gain | greedy_proxy_over_realized           |     0.987  |         0.987  |
| replay_gap_fraction_of_lp_gain | action_cost_jaccard                  |    -0.9849 |         0.9849 |
| replay_gap_fraction_of_lp_gain | lp_only_cost_share                   |     0.9849 |         0.9849 |
| replay_gap_fraction_of_lp_gain | greedy_only_cost_share               |     0.9849 |         0.9849 |
| replay_gap_fraction_of_lp_gain | greedy_minus_lp_proxy                |     0.9703 |         0.9703 |
| replay_gap_fraction_of_lp_gain | lp_to_greedy_action_count_ratio      |     0.9688 |         0.9688 |
| replay_gap_fraction_of_lp_gain | lp_to_greedy_unit_count_ratio        |     0.9477 |         0.9477 |
| replay_gap_fraction_of_lp_gain | time_share_l1                        |     0.912  |         0.912  |
| replay_gap_fraction_of_lp_gain | intervention_share_l1                |     0.8147 |         0.8147 |
| replay_gap_fraction_of_lp_gain | event_peak_positive_abnormal_deficit |    -0.5981 |         0.5981 |

## 城市层面汇总

| city         |   n_events |   mean_lp_replay_recoverable_fraction |   mean_greedy_replay_recoverable_fraction |   mean_greedy_fraction_of_lp_gain |   mean_replay_gap_fraction_of_lp_gain |   mean_action_cost_jaccard |   mean_intervention_share_l1 |   mean_time_share_l1 |   mean_segment_share_l1 |   mean_greedy_proxy_over_realized |   mean_lp_to_greedy_action_count_ratio |   mean_lp_cost_gini |   mean_greedy_cost_gini |
|:-------------|-----------:|--------------------------------------:|------------------------------------------:|----------------------------------:|--------------------------------------:|---------------------------:|-----------------------------:|---------------------:|------------------------:|----------------------------------:|---------------------------------------:|--------------------:|------------------------:|
| Philadelphia |         20 |                               0.1597  |                                    0.1182 |                            0.7517 |                              0.2483   |                     0.4744 |                      0.1467  |             0.1594   |                0.05469  |                          0.06328  |                                  2.303 |              0.4329 |                  0.2517 |
| Chicago      |         13 |                               0.1497  |                                    0.1087 |                            0.7632 |                              0.2368   |                     0.5219 |                      0.1784  |             0.1231   |                0.08347  |                          0.06584  |                                  1.974 |              0.4172 |                  0.3615 |
| San Antonio  |         19 |                               0.1366  |                                    0.1063 |                            0.7834 |                              0.2166   |                     0.5463 |                      0.09818 |             0.1756   |                0.02638  |                          0.04505  |                                  1.846 |              0.4212 |                  0.2758 |
| New York     |         24 |                               0.1369  |                                    0.1102 |                            0.8253 |                              0.1747   |                     0.6197 |                      0.09866 |             0.1053   |                0.0826   |                          0.04145  |                                  1.443 |              0.4827 |                  0.4018 |
| Austin       |         15 |                               0.1311  |                                    0.1131 |                            0.8696 |                              0.1304   |                     0.6669 |                      0.1051  |             0.08286  |                0.05909  |                          0.02642  |                                  1.275 |              0.3561 |                  0.3435 |
| Houston      |          4 |                               0.11    |                                    0.1021 |                            0.9303 |                              0.06967  |                     0.7559 |                      0.05849 |             0.06973  |                0.07517  |                          0.01133  |                                  1.288 |              0.5133 |                  0.4057 |
| Dallas       |         10 |                               0.07209 |                                    0.0719 |                            0.9979 |                              0.002069 |                     0.9778 |                      0.00183 |             0.001865 |                0.006197 |                          0.000265 |                                  1.006 |              0.4292 |                  0.3316 |

## Gap 最大事件

| city         |   event_id |   greedy_fraction_of_lp_gain |   replay_gap_fraction_of_lp_gain |   action_cost_jaccard |   intervention_share_l1 |   time_share_l1 |   segment_share_l1 |   greedy_proxy_over_realized |   lp_to_greedy_action_count_ratio |
|:-------------|-----------:|-----------------------------:|---------------------------------:|----------------------:|------------------------:|----------------:|-------------------:|-----------------------------:|----------------------------------:|
| Philadelphia |        642 |                       0.3831 |                           0.6169 |                0.118  |                 0.2387  |         0.3631  |          1.258e-06 |                      0.2014  |                             6.3   |
| Chicago      |        152 |                       0.3966 |                           0.6034 |                0.1061 |                 0.377   |         0.2733  |          0.1783    |                      0.3334  |                             5.904 |
| San Antonio  |        506 |                       0.3975 |                           0.6025 |                0.1712 |                 0.08368 |         0.357   |          0         |                      0.1379  |                             5.102 |
| Philadelphia |        640 |                       0.4464 |                           0.5536 |                0.1933 |                 0.2497  |         0.3973  |          0.03231   |                      0.1411  |                             4.797 |
| Philadelphia |        641 |                       0.4637 |                           0.5363 |                0.1458 |                 0.2419  |         0.5039  |          0         |                      0.1664  |                             6.614 |
| San Antonio  |        500 |                       0.4677 |                           0.5323 |                0.1368 |                 0.1678  |         0.447   |          0.02733   |                      0.1722  |                             4.558 |
| New York     |        100 |                       0.5284 |                           0.4716 |                0.2179 |                 0.2178  |         0.2576  |          0.1487    |                      0.1491  |                             2.184 |
| Chicago      |        154 |                       0.5419 |                           0.4581 |                0.2203 |                 0.3654  |         0.2157  |          0.08282   |                      0.1539  |                             3.194 |
| New York     |         99 |                       0.5795 |                           0.4205 |                0.2583 |                 0.1669  |         0.2264  |          0.1155    |                      0.1077  |                             2.068 |
| New York     |         98 |                       0.581  |                           0.419  |                0.2248 |                 0.5381  |         0.3029  |          0.1514    |                      0.1851  |                             2.635 |
| San Antonio  |        501 |                       0.5878 |                           0.4122 |                0.2201 |                 0.1064  |         0.4419  |          0.01841   |                      0.08884 |                             2.654 |
| Chicago      |        149 |                       0.6065 |                           0.3935 |                0.308  |                 0.2979  |         0.09146 |          0.02533   |                      0.08327 |                             2.167 |

## 高差距事件中的 allocation 差异样例

| city    |   event_id |   unit |   t | intervention   |   lp_cost |   greedy_cost |   absolute_cost_difference | allocation_preference   |   marginal_resource_value |   finite_deficit_area_value |   active_weighted_horizon |   origin_exposure_rank |   destination_importance_rank |   local_remaining_rank |   access_remaining_rank |   eta_per_cost_rank |
|:--------|-----------:|-------:|----:|:---------------|----------:|--------------:|---------------------------:|:------------------------|--------------------------:|----------------------------:|--------------------------:|-----------------------:|------------------------------:|-----------------------:|------------------------:|--------------------:|
| Chicago |        152 |    849 |   2 | R              |   0       |       0.01975 |                    0.01975 | greedy_more             |                    0.2099 |                   0.001207  |                     4.956 |                 1      |                        1      |                 1      |                  0.9932 |            0.001134 |
| Chicago |        152 |    849 |   3 | R              |   0       |       0.01975 |                    0.01975 | greedy_more             |                    0.202  |                   0.001185  |                     4.771 |                 1      |                        1      |                 1      |                  0.9932 |            0.001134 |
| Chicago |        152 |    849 |   4 | R              |   0       |       0.01975 |                    0.01975 | greedy_more             |                    0.1926 |                   0.001168  |                     4.547 |                 1      |                        1      |                 1      |                  0.9932 |            0.001134 |
| Chicago |        152 |    849 |   5 | R              |   0       |       0.01975 |                    0.01975 | greedy_more             |                    0.1812 |                   0.001153  |                     4.277 |                 1      |                        1      |                 1      |                  0.9932 |            0.001134 |
| Chicago |        152 |    411 |   2 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.2056 |                   0.001176  |                     4.944 |                 0.9989 |                        0.9977 |                 0.9989 |                  0.9977 |            0.002268 |
| Chicago |        152 |    411 |   3 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1979 |                   0.001155  |                     4.76  |                 0.9989 |                        0.9977 |                 0.9989 |                  0.9977 |            0.002268 |
| Chicago |        152 |    411 |   4 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1887 |                   0.001138  |                     4.538 |                 0.9989 |                        0.9977 |                 0.9989 |                  0.9977 |            0.002268 |
| Chicago |        152 |    411 |   5 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1775 |                   0.001123  |                     4.269 |                 0.9989 |                        0.9977 |                 0.9989 |                  0.9977 |            0.002268 |
| Chicago |        152 |    719 |   2 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.2057 |                   0.001176  |                     4.944 |                 0.9898 |                        0.9989 |                 0.9977 |                  0.9524 |            0.003401 |
| Chicago |        152 |    719 |   3 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1981 |                   0.001156  |                     4.76  |                 0.9898 |                        0.9989 |                 0.9977 |                  0.9524 |            0.003401 |
| Chicago |        152 |    719 |   4 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1888 |                   0.001138  |                     4.537 |                 0.9898 |                        0.9989 |                 0.9977 |                  0.9524 |            0.003401 |
| Chicago |        152 |    719 |   5 | R              |   0       |       0.0196  |                    0.0196  | greedy_more             |                    0.1777 |                   0.001124  |                     4.269 |                 0.9898 |                        0.9989 |                 0.9977 |                  0.9524 |            0.003401 |
| Chicago |        152 |    463 |   2 | R              |   0       |       0.01906 |                    0.01906 | greedy_more             |                    0.1904 |                   0.001069  |                     4.901 |                 0.8866 |                        0.9966 |                 0.9966 |                  0.9966 |            0.004535 |
| Chicago |        152 |    463 |   3 | R              |   0       |       0.01906 |                    0.01906 | greedy_more             |                    0.1834 |                   0.00105   |                     4.722 |                 0.8866 |                        0.9966 |                 0.9966 |                  0.9966 |            0.004535 |
| Chicago |        152 |    463 |   4 | R              |   0       |       0.01906 |                    0.01906 | greedy_more             |                    0.175  |                   0.001034  |                     4.505 |                 0.8866 |                        0.9966 |                 0.9966 |                  0.9966 |            0.004535 |
| Chicago |        152 |    463 |   5 | R              |   0       |       0.01906 |                    0.01906 | greedy_more             |                    0.1648 |                   0.001022  |                     4.242 |                 0.8866 |                        0.9966 |                 0.9966 |                  0.9966 |            0.004535 |
| Chicago |        152 |    849 |   6 | R              |   0.00101 |       0.01975 |                    0.01874 | greedy_more             |                    0.1674 |                   0.001141  |                     3.952 |                 1      |                        1      |                 1      |                  0.9932 |            0.001134 |
| Chicago |        152 |    700 |   2 | R              |   0       |       0.01866 |                    0.01866 | greedy_more             |                    0.1791 |                   0.000991  |                     4.87  |                 0.8844 |                        0.9955 |                 0.9955 |                  0.9898 |            0.005669 |
| Chicago |        152 |    700 |   3 | R              |   0       |       0.01866 |                    0.01866 | greedy_more             |                    0.1726 |                   0.0009737 |                     4.694 |                 0.8844 |                        0.9955 |                 0.9955 |                  0.9898 |            0.005669 |
| Chicago |        152 |    700 |   4 | R              |   0       |       0.01866 |                    0.01866 | greedy_more             |                    0.1648 |                   0.0009594 |                     4.48  |                 0.8844 |                        0.9955 |                 0.9955 |                  0.9898 |            0.005669 |

## 下一步

下一版应把这组发现反馈到 policy construction：不再只按 first-order value 排序，而是实现 residual greedy 或 interaction-aware greedy。每选择一个 segment 后，重新用当前 replay state 更新剩余 `b/rC/rS/ell`，再计算下一轮边际值。若 residual greedy 能显著缩小 82% 到 100% 的差距，就能把 finite-budget law 写成“activated value under residual loss state”。

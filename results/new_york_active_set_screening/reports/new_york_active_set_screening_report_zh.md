# New York Active-Set Screening V44

## 结论

这版不是求解 New York footprint-aware LP 最优解，而是为后续 New York-scale decomposition 设计 active action set。做法是保留 1,940 个 New York OD zones、完整状态演化和完整 OD access-loss 约束，只限制哪些 unit-time-intervention action columns 可以产生 R/C/S deployment variables。

推荐的 `token_columns/value_footprint_structure_union` 在每类 token 信号取 top 10% 后取并集，平均 active units 为 1183.8，约占全城 61.0%，实际 active action tokens 约占全量 25.1%。在仍保留全城状态和 access 约束的情况下，估计总变量数降低 55.0%。

这个 active set 平均捕获 small-signal value 49.5%，finite value 60.7%，observed footprint mass 89.8%，OD destination importance 82.7%。最弱事件中的 small/finite/footprint 捕获分别为 48.4%/59.2%/87.6%。

如果要求最弱事件也同时达到 small-signal >=95%、finite >=95%、footprint >=80%，当前联合筛选需要 requested fraction 60%，实际 active action tokens 83.3%，active units 99.8%，总变量仍只能降低 12.3%。这说明 New York 的恢复价值高度分散，不能把它压成少数区域问题。

解释：这说明 New York 未闭合的问题不一定需要回退到小 OD。更自然的下一步是大规模全城 LP 的 decomposition/warm-start：全城结构仍在模型里，但候选投放动作先由 value、footprint 和 OD structure 共同筛选。unit-block 初版过粗，top 10% 联合筛选只能捕获 36.6% small-signal value 和 46.1% finite value；token-column 筛选更贴近 LP 的变量结构。

## Recommended Frontier

|   requested_screen_fraction |   active_unit_count_mean |   active_unit_fraction_mean |   active_action_token_fraction_mean |   restricted_total_variable_fraction_mean |   small_signal_value_capture_mean |   finite_value_capture_mean |   footprint_mass_capture_mean |   destination_importance_capture_mean |   small_signal_value_capture_min |   finite_value_capture_min |   footprint_mass_capture_min |
|----------------------------:|-------------------------:|----------------------------:|------------------------------------:|------------------------------------------:|----------------------------------:|----------------------------:|------------------------------:|--------------------------------------:|---------------------------------:|---------------------------:|-----------------------------:|
|                       0.01  |                    203.2 |                      0.1047 |                             0.03012 |                                    0.2874 |                            0.1071 |                      0.1717 |                        0.5705 |                                0.2945 |                           0.1017 |                     0.1627 |                       0.55   |
|                       0.025 |                    406.4 |                      0.2095 |                             0.07089 |                                    0.3174 |                            0.2066 |                      0.2949 |                        0.6741 |                                0.4604 |                           0.1995 |                     0.2829 |                       0.6477 |
|                       0.05  |                    714.6 |                      0.3684 |                             0.1342  |                                    0.3639 |                            0.3253 |                      0.4298 |                        0.7861 |                                0.6368 |                           0.3165 |                     0.4174 |                       0.7586 |
|                       0.1   |                   1184   |                      0.6102 |                             0.2512  |                                    0.4498 |                            0.4949 |                      0.6067 |                        0.8976 |                                0.827  |                           0.4838 |                     0.5919 |                       0.8758 |
|                       0.15  |                   1489   |                      0.7676 |                             0.3544  |                                    0.5257 |                            0.6165 |                      0.7241 |                        0.9487 |                                0.9147 |                           0.6073 |                     0.7117 |                       0.9353 |
|                       0.2   |                   1682   |                      0.8668 |                             0.4455  |                                    0.5926 |                            0.7086 |                      0.8074 |                        0.9745 |                                0.9593 |                           0.7009 |                     0.8    |                       0.965  |
|                       0.3   |                   1862   |                      0.96   |                             0.59    |                                    0.6988 |                            0.8305 |                      0.9047 |                        0.9941 |                                0.9911 |                           0.8235 |                     0.899  |                       0.9909 |
|                       0.4   |                   1915   |                      0.9873 |                             0.6961  |                                    0.7767 |                            0.904  |                      0.9535 |                        0.9988 |                                0.998  |                           0.897  |                     0.9493 |                       0.9979 |
|                       0.5   |                   1929   |                      0.9945 |                             0.7754  |                                    0.835  |                            0.9492 |                      0.9787 |                        0.9997 |                                0.9994 |                           0.944  |                     0.976  |                       0.9996 |
|                       0.6   |                   1936   |                      0.9977 |                             0.8333  |                                    0.8775 |                            0.9759 |                      0.9912 |                        1      |                                0.9999 |                           0.9734 |                     0.9902 |                       0.9999 |

## Unit-Block Baseline at 10%

|   requested_screen_fraction |   active_unit_count_mean |   active_unit_fraction_mean |   active_action_token_fraction_mean |   restricted_total_variable_fraction_mean |   small_signal_value_capture_mean |   finite_value_capture_mean |   footprint_mass_capture_mean |   destination_importance_capture_mean |   small_signal_value_capture_min |   finite_value_capture_min |   footprint_mass_capture_min |
|----------------------------:|-------------------------:|----------------------------:|------------------------------------:|------------------------------------------:|----------------------------------:|----------------------------:|------------------------------:|--------------------------------------:|---------------------------------:|---------------------------:|-----------------------------:|
|                         0.1 |                    396.2 |                      0.2042 |                              0.2042 |                                    0.4154 |                            0.3664 |                      0.4612 |                        0.7644 |                                0.4044 |                           0.3547 |                     0.4541 |                       0.7535 |

## Strategy Comparison at 10%

| strategy                        |   active_action_token_fraction_mean |   active_unit_fraction_mean |   small_signal_value_capture_mean |   finite_value_capture_mean |   footprint_mass_capture_mean |   destination_importance_capture_mean |
|:--------------------------------|------------------------------------:|----------------------------:|----------------------------------:|----------------------------:|------------------------------:|--------------------------------------:|
| value_footprint_structure_union |                             0.2512  |                      0.6102 |                            0.4949 |                      0.6067 |                        0.8976 |                                0.827  |
| value_footprint_union           |                             0.2035  |                      0.5253 |                            0.4222 |                      0.5803 |                        0.8778 |                                0.7322 |
| value_union                     |                             0.1387  |                      0.5032 |                            0.3755 |                      0.5192 |                        0.825  |                                0.7244 |
| finite_value_only               |                             0.09167 |                      0.4867 |                            0.2344 |                      0.4459 |                        0.8222 |                                0.6976 |
| footprint_only                  |                             0.09167 |                      0.1    |                            0.1012 |                      0.218  |                        0.7175 |                                0.1011 |
| small_signal_only               |                             0.09167 |                      0.2446 |                            0.3115 |                      0.3238 |                        0.2502 |                                0.513  |
| od_structure_only               |                             0.09167 |                      0.151  |                            0.2855 |                      0.293  |                        0.1252 |                                0.3914 |

## 写作含义

1. 这不是“逃避 New York 大规模”，而是把大规模问题拆成全城状态/约束 + 结构化 active action columns。
2. unit-block 筛选告诉我们 New York 的 recovery value 不是只集中在少数 zones；这反而支持 column generation，而不是简单删城市区域。
3. footprint-only 能覆盖 footprint，但会弱化 recovery value；small/finite value 只看恢复信号则可能漏掉 event footprint。联合筛选说明 city structure law 和 event footprint 是互补信号。
4. 后续真正需要模型部分完成的是：在这些 active columns 上重解 restricted LP，再做 column generation 或 active-set expansion，检验是否逼近 full New York optimum。

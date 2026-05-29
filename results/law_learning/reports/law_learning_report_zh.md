# Learning and Law Discovery V3

## 本版本做了什么

这一版把 event-level optimization outputs 转换成 action-token 学习问题。每个 token 表示 `city-event-unit-time-intervention`。目标不是直接学习 optimizer 是否选择该 token，而是构造一个可解释的 marginal recovery-value proxy：单位资源投到该 token 后，沿着无干预的被动恢复轨迹，估计它能减少多少未来加权功能损失。

V2 在 V1 的静态 action-value field 之上，新增了 budget-aware greedy oracle。V3 进一步把 greedy/law/simple baseline 生成的固定 allocation 放回复原始恢复动力学中 replay，直接计算 fixed policy 的 12 小时 objective 和 recoverable fraction。因此现在不仅能问“一个 law-guided policy 在 proxy 上接近 greedy oracle 吗”，也能问“它在实际 `b, rC, rS, ell` 演化里能拿到 LP optimum 的多少恢复收益”。

## 数据规模

- sampled action tokens: 186,536
- city-event scenarios: 105
- full candidate-action concentration rows: 105
- policy stress-test rows: 3,780
- fixed-policy replay rows: 3,885
- sampled-token greedy oracle selected share: 0.3971
- mean event top-5% value share: 0.2444
- mean event marginal-value gini: 0.5909

## Action Value Label

每个 action 的 label 不是 optimizer 的 0/1 选择，而是 marginal resource value。直观上，如果一个 action 作用于未来仍会持续存在的损失、又覆盖高 OD 暴露区域、且单位成本效率高，那么它的恢复价值就高。

对 `R` 和 `C` 类 action，价值主要来自某个 region 本地 deficit 被降低后，通过 OD dependence `Q` 减少其他 origins 的 accessibility loss；因此 exposure 使用 destination importance。对 `S` 类 action，价值直接作用在 origin 的 experienced loss 上；因此 exposure 使用 origin exposure。

```text
marginal_resource_value(i,t,k)
  = future_effect_value(i,t,k)
    * eta(i,t,k) / cost(i,t,k)
    / passive_event_loss
```

## Interpretable Law Score

这一版的可解释 law score 保留 action label 的核心结构，但不直接使用 optimizer 选择结果：

```text
activated_bottleneck_score(i,t,k)
  ≈ delay_feasible(i,t,k)
    × future_deficit_area(i,t,k)
    × OD_exposure_or_destination_importance(i,k)
    × intervention_efficiency_per_cost(i,t,k)
```

这里的 `future_deficit_area` 已经包含剩余时间窗口，因此不再额外乘一个简单的 time rank。早期草稿里我把 `1 - out_degree_rank` 当作 substitutability scarcity 强行乘进去，结果明显拉低排序表现；这一版先移除这个不稳定项，把 substitutability 留到后续用更可靠的替代路径或网络冗余指标刻画。

## Budget-Aware Greedy Oracle

V2 增加的 greedy oracle 不是重新求解 LP，而是一个可解释的 budget-aware policy simulator。它做三件事：

1. 把每个 `unit-time-intervention` 的部署上限按 PWL diminishing returns 拆成多个 segment。
2. 每个 segment 的真实评价仍用 oracle marginal value per cost，但乘上该 segment 的 diminishing multiplier。
3. 在 total budget、period budget 和 delay feasibility 下，按某个 policy score 贪心选择 segment。

因此 `greedy_oracle` 是这个解析标签体系下的上界 policy；`activated_bottleneck_law`、`exposure_only`、`deficit_only`、`structure_only`、`random_positive` 都在同一预算约束下与它比较。

## Fixed-Policy Replay

V3 的 replay validation 不再只看 action-value proxy。对每个 policy 生成的固定 allocation，脚本把分段资源量转换成 `R/C/S` 的实际 effect，然后按原始恢复动力学逐小时前推：

```text
b[t+1]  = clip(a * b[t] + h[t+1] - e_R[t], 0, 1)
rC[t+1] = clip((1 - delta_C) * rC[t] + e_C[t], 0, 1)
rS[t+1] = clip((1 - delta_S) * rS[t] + e_S[t], 0, 1)
ell[t]  = clip(Q * max(b[t] - rC[t], 0) - rS[t], 0, 1)
```

这里 `lp_optimizer_replay` 是一个 sanity check：把 Gurobi 输出的 optimized effects 放回同一个 replay engine，应该接近原 LP objective。这个检查用于确认 replay engine 和 LP 目标是一致的。

## 关键结果概览

- Leave-one-city-out mean Spearman: 0.8501
- Leave-one-city-out mean top-5% value capture: 0.8334
- Activated-bottleneck law top-5% value capture: 0.9636
- Activated-bottleneck law mean event Spearman: 0.9643
- Base scenario law policy / greedy oracle: 0.9814
- Base scenario best simple baseline / greedy oracle: 0.5767
- Base scenario law replay gain / LP optimized gain: 0.7904
- Base scenario best simple replay gain / LP optimized gain: 0.5591
- LP optimizer replay mean recoverable-fraction gap: 6.742e-09
- Base scenario law replay mean recoverable-fraction gap to LP: 0.0287

## Leave-One-City-Out Surrogate

| split          | heldout      |   n_tokens |   n_events |   pearson |   spearman |       mae |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |
|:---------------|:-------------|-----------:|-----------:|----------:|-----------:|----------:|-------------------------:|-------------------------:|--------------------------:|
| leave_city_out | Austin       |      19758 |         15 |    0.8706 |     0.9271 | 0.0005421 |                   0.8895 |                   0.917  |                    0.9306 |
| leave_city_out | Chicago      |      27896 |         13 |    0.7936 |     0.8246 | 0.0001507 |                   0.9311 |                   0.9213 |                    0.9202 |
| leave_city_out | Dallas       |      12825 |         10 |    0.7563 |     0.755  | 0.0003518 |                   0.5144 |                   0.6012 |                    0.6706 |
| leave_city_out | Houston      |       5158 |          4 |    0.726  |     0.7997 | 0.0003763 |                   0.5842 |                   0.669  |                    0.7623 |
| leave_city_out | New York     |      53346 |         24 |    0.7384 |     0.8474 | 0.0002048 |                   0.6371 |                   0.8413 |                    0.8626 |
| leave_city_out | Philadelphia |      37656 |         20 |    0.8396 |     0.8958 | 0.0002556 |                   0.9613 |                   0.9357 |                    0.933  |
| leave_city_out | San Antonio  |      29897 |         19 |    0.8552 |     0.9013 | 0.0003219 |                   0.9468 |                   0.948  |                    0.9433 |

## Leave-Regime-Out Surrogate

| split            | heldout   |   n_tokens |   n_events |   pearson |   spearman |       mae |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |
|:-----------------|:----------|-----------:|-----------:|----------:|-----------:|----------:|-------------------------:|-------------------------:|--------------------------:|
| leave_regime_out | low       |      83698 |         46 |    0.7619 |     0.6664 | 0.0003226 |                   0.8878 |                   0.8966 |                    0.9066 |
| leave_regime_out | medium    |      65929 |         37 |    0.9108 |     0.9328 | 0.0002039 |                   0.8955 |                   0.9136 |                    0.925  |
| leave_regime_out | high      |      36909 |         22 |    0.7045 |     0.8184 | 0.000354  |                   0.5833 |                   0.6846 |                    0.7289 |

## Law Score 与 Baselines

| policy_score             |   n_tokens |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |   mean_spearman_by_event |
|:-------------------------|-----------:|-------------------------:|-------------------------:|--------------------------:|-------------------------:|
| activated_bottleneck_law |     186536 |                   0.9606 |                   0.9636 |                    0.9634 |                  0.9643  |
| exposure_only            |     186536 |                   0.5449 |                   0.6811 |                    0.7304 |                  0.6264  |
| deficit_only             |     186536 |                   0.5508 |                   0.6764 |                    0.7133 |                  0.4638  |
| optimizer_selected       |     186536 |                   0.3548 |                   0.4596 |                    0.5283 |                  0.4336  |
| structure_only           |     186536 |                   0.2895 |                   0.3784 |                    0.4282 |                 -0.05895 |

解释：`optimizer_selected` 在 action-value 排序里不一定最高，因为 optimizer 选择受到总预算、单期预算、部署上限、分段边际收益和替代 action 的共同约束；而 law score 评价的是“单个 action 的边际价值排序”。因此这里更应该看 law score 是否能捕捉 value field 的 top tail，而不是是否复刻 optimizer 的最终稀疏解。

## Budget/Delay Policy Validation

| policy_scenario   |   budget_scale |   delay_add_hours | policy_score             |   n_events |   mean_value_proxy |   median_value_proxy |   mean_relative_to_greedy_oracle |   median_relative_to_greedy_oracle |   mean_allocated_cost |   mean_selected_action_count |   mean_value_vs_lp_recoverable_fraction |
|:------------------|---------------:|------------------:|:-------------------------|-----------:|-------------------:|---------------------:|---------------------------------:|-----------------------------------:|----------------------:|-----------------------------:|----------------------------------------:|
| base              |            1   |                 0 | greedy_oracle            |        105 |          0.00478   |            0.003526  |                          1       |                            1       |                 6.146 |                        959.7 |                                0.04054  |
| base              |            1   |                 0 | activated_bottleneck_law |        105 |          0.004638  |            0.00343   |                          0.9814  |                            0.9943  |                 6.146 |                       1138   |                                0.03926  |
| base              |            1   |                 0 | exposure_only            |        105 |          0.003029  |            0.00217   |                          0.5767  |                            0.5879  |                 6.146 |                        594.3 |                                0.02563  |
| base              |            1   |                 0 | deficit_only             |        105 |          0.002912  |            0.002074  |                          0.5515  |                            0.5588  |                 6.146 |                        593.2 |                                0.02469  |
| base              |            1   |                 0 | structure_only           |        105 |          0.001188  |            0.0007246 |                          0.2182  |                            0.2136  |                 6.146 |                        536   |                                0.01034  |
| base              |            1   |                 0 | random_positive          |        105 |          0.0004868 |            0.0003217 |                          0.09004 |                            0.08818 |                 6.146 |                        756.2 |                                0.004238 |
| delay_2h          |            1   |                 2 | greedy_oracle            |        105 |          0.003907  |            0.002781  |                          1       |                            1       |                 6.146 |                       1025   |                                0.03332  |
| delay_2h          |            1   |                 2 | activated_bottleneck_law |        105 |          0.003853  |            0.002781  |                          0.9925  |                            0.9996  |                 6.146 |                       1125   |                                0.03283  |
| delay_2h          |            1   |                 2 | exposure_only            |        105 |          0.002475  |            0.001719  |                          0.5793  |                            0.5884  |                 6.146 |                        608.9 |                                0.02093  |
| delay_2h          |            1   |                 2 | deficit_only             |        105 |          0.002374  |            0.00161   |                          0.5536  |                            0.558   |                 6.146 |                        607   |                                0.02014  |
| delay_2h          |            1   |                 2 | structure_only           |        105 |          0.001026  |            0.0006195 |                          0.2296  |                            0.2264  |                 6.146 |                        545.7 |                                0.009011 |
| delay_2h          |            1   |                 2 | random_positive          |        105 |          0.0004175 |            0.0002424 |                          0.09409 |                            0.09157 |                 6.146 |                        762.7 |                                0.003634 |
| delay_4h          |            1   |                 4 | greedy_oracle            |        105 |          0.00303   |            0.00212   |                          1       |                            1       |                 6.146 |                       1075   |                                0.02588  |
| delay_4h          |            1   |                 4 | activated_bottleneck_law |        105 |          0.003019  |            0.00212   |                          0.9985  |                            1       |                 6.146 |                       1109   |                                0.02579  |
| delay_4h          |            1   |                 4 | exposure_only            |        105 |          0.00192   |            0.001186  |                          0.5883  |                            0.5884  |                 6.146 |                        630.9 |                                0.01623  |
| delay_4h          |            1   |                 4 | deficit_only             |        105 |          0.001834  |            0.001131  |                          0.5595  |                            0.5596  |                 6.146 |                        628.3 |                                0.01555  |
| delay_4h          |            1   |                 4 | structure_only           |        105 |          0.000821  |            0.0004681 |                          0.237   |                            0.238   |                 6.146 |                        556.2 |                                0.007217 |
| delay_4h          |            1   |                 4 | random_positive          |        105 |          0.0003522 |            0.0002206 |                          0.103   |                            0.1036  |                 6.146 |                        771.6 |                                0.003045 |
| high_budget       |            2   |                 0 | greedy_oracle            |        105 |          0.007603  |            0.005745  |                          1       |                            1       |                12.29  |                       1887   |                                0.06471  |
| high_budget       |            2   |                 0 | activated_bottleneck_law |        105 |          0.007418  |            0.005393  |                          0.985   |                            0.9957  |                12.29  |                       2184   |                                0.06307  |
| high_budget       |            2   |                 0 | exposure_only            |        105 |          0.004943  |            0.003565  |                          0.5947  |                            0.6042  |                12.29  |                       1241   |                                0.04213  |
| high_budget       |            2   |                 0 | deficit_only             |        105 |          0.004756  |            0.003377  |                          0.5694  |                            0.5778  |                12.29  |                       1238   |                                0.04061  |
| high_budget       |            2   |                 0 | structure_only           |        105 |          0.002233  |            0.001334  |                          0.2531  |                            0.2414  |                12.29  |                       1074   |                                0.01964  |
| high_budget       |            2   |                 0 | random_positive          |        105 |          0.0009745 |            0.0006547 |                          0.1127  |                            0.1112  |                12.29  |                       1498   |                                0.008492 |
| low_budget        |            0.5 |                 0 | greedy_oracle            |        105 |          0.002932  |            0.002163  |                          1       |                            1       |                 3.073 |                        466.4 |                                0.02478  |
| low_budget        |            0.5 |                 0 | activated_bottleneck_law |        105 |          0.002835  |            0.002051  |                          0.979   |                            0.9927  |                 3.073 |                        569.7 |                                0.02391  |
| low_budget        |            0.5 |                 0 | exposure_only            |        105 |          0.0018    |            0.001319  |                          0.5556  |                            0.5745  |                 3.073 |                        285.5 |                                0.0151   |
| low_budget        |            0.5 |                 0 | deficit_only             |        105 |          0.001731  |            0.001225  |                          0.5334  |                            0.5436  |                 3.073 |                        284.8 |                                0.01456  |
| low_budget        |            0.5 |                 0 | structure_only           |        105 |          0.0006616 |            0.0004511 |                          0.1991  |                            0.1965  |                 3.073 |                        262.3 |                                0.00572  |
| low_budget        |            0.5 |                 0 | random_positive          |        105 |          0.0002439 |            0.0001683 |                          0.0739  |                            0.06674 |                 3.073 |                        381.6 |                                0.002119 |
| scarce_and_late   |            0.5 |                 2 | greedy_oracle            |        105 |          0.002419  |            0.001761  |                          1       |                            1       |                 3.073 |                        504.2 |                                0.02057  |
| scarce_and_late   |            0.5 |                 2 | activated_bottleneck_law |        105 |          0.00238   |            0.001724  |                          0.9911  |                            0.9994  |                 3.073 |                        564.2 |                                0.02022  |
| scarce_and_late   |            0.5 |                 2 | exposure_only            |        105 |          0.001487  |            0.001018  |                          0.5608  |                            0.5687  |                 3.073 |                        293.4 |                                0.01249  |
| scarce_and_late   |            0.5 |                 2 | deficit_only             |        105 |          0.001422  |            0.0009445 |                          0.5332  |                            0.5484  |                 3.073 |                        291.4 |                                0.01197  |
| scarce_and_late   |            0.5 |                 2 | structure_only           |        105 |          0.0005571 |            0.0003686 |                          0.2042  |                            0.2008  |                 3.073 |                        268   |                                0.004829 |
| scarce_and_late   |            0.5 |                 2 | random_positive          |        105 |          0.0002071 |            0.0001245 |                          0.07625 |                            0.07596 |                 3.073 |                        384.4 |                                0.001799 |

解释：这个表不是新的 Gurobi LP 解，而是用同一 action-value field 做的预算约束 stress test。它用于检验 law 在不同预算和延迟条件下是否仍能作为资源分配 policy 接近 greedy oracle。真正的最终版还需要对关键 budget/delay scenario 重新求解 LP 来闭合验证。

## Fixed-Policy Replay Validation

| policy_scenario   |   budget_scale |   delay_add_hours | policy_score             |   n_events |   mean_replay_recoverable_fraction |   median_replay_recoverable_fraction |   mean_fraction_of_base_lp_gain |   median_fraction_of_base_lp_gain |   mean_relative_to_greedy_replay_gain |   median_relative_to_greedy_replay_gain |   mean_gap_to_base_lp_recoverable |   mean_allocated_cost |   mean_selected_action_count |
|:------------------|---------------:|------------------:|:-------------------------|-----------:|-----------------------------------:|-------------------------------------:|--------------------------------:|----------------------------------:|--------------------------------------:|----------------------------------------:|----------------------------------:|----------------------:|-----------------------------:|
| base              |            1   |                 0 | lp_optimizer_replay      |        105 |                           0.1347   |                             0.1331   |                         1       |                           1       |                                1.268  |                                  1.168  |                         6.742e-09 |                 6.146 |                        971.7 |
| base              |            1   |                 0 | greedy_oracle            |        105 |                           0.1079   |                             0.1113   |                         0.8092  |                           0.8564  |                                1      |                                  1      |                         0.02681   |                 6.146 |                        959.7 |
| base              |            1   |                 0 | activated_bottleneck_law |        105 |                           0.1061   |                             0.1112   |                         0.7904  |                           0.8463  |                                0.9757 |                                  0.9914 |                         0.02868   |                 6.146 |                       1138   |
| base              |            1   |                 0 | exposure_only            |        105 |                           0.07179  |                             0.07012  |                         0.5591  |                           0.5476  |                                0.6904 |                                  0.6903 |                         0.06295   |                 6.146 |                        594.3 |
| base              |            1   |                 0 | deficit_only             |        105 |                           0.06754  |                             0.06547  |                         0.5284  |                           0.5164  |                                0.6517 |                                  0.651  |                         0.0672    |                 6.146 |                        593.2 |
| base              |            1   |                 0 | structure_only           |        105 |                           0.03027  |                             0.02978  |                         0.2436  |                           0.2311  |                                0.2999 |                                  0.2857 |                         0.1045    |                 6.146 |                        536   |
| base              |            1   |                 0 | random_positive          |        105 |                           0.02036  |                             0.02102  |                         0.1549  |                           0.1588  |                                0.191  |                                  0.1891 |                         0.1144    |                 6.146 |                        756.2 |
| delay_2h          |            1   |                 2 | greedy_oracle            |        105 |                           0.1013   |                             0.1037   |                         0.7552  |                           0.7779  |                                1      |                                  1      |                         0.03345   |                 6.146 |                       1025   |
| delay_2h          |            1   |                 2 | activated_bottleneck_law |        105 |                           0.1004   |                             0.1037   |                         0.7462  |                           0.7732  |                                0.9876 |                                  0.9975 |                         0.03437   |                 6.146 |                       1125   |
| delay_2h          |            1   |                 2 | exposure_only            |        105 |                           0.06569  |                             0.0624   |                         0.5061  |                           0.5007  |                                0.667  |                                  0.6843 |                         0.06904   |                 6.146 |                        608.9 |
| delay_2h          |            1   |                 2 | deficit_only             |        105 |                           0.06159  |                             0.05822  |                         0.4767  |                           0.471   |                                0.6279 |                                  0.6251 |                         0.07315   |                 6.146 |                        607   |
| delay_2h          |            1   |                 2 | structure_only           |        105 |                           0.02859  |                             0.02894  |                         0.2283  |                           0.2115  |                                0.3014 |                                  0.2831 |                         0.1061    |                 6.146 |                        545.7 |
| delay_2h          |            1   |                 2 | random_positive          |        105 |                           0.0191   |                             0.01891  |                         0.1444  |                           0.1475  |                                0.1906 |                                  0.1909 |                         0.1156    |                 6.146 |                        762.7 |
| delay_4h          |            1   |                 4 | greedy_oracle            |        105 |                           0.09192  |                             0.09101  |                         0.6812  |                           0.6809  |                                1      |                                  1      |                         0.04281   |                 6.146 |                       1075   |
| delay_4h          |            1   |                 4 | activated_bottleneck_law |        105 |                           0.09167  |                             0.09101  |                         0.6789  |                           0.6779  |                                0.9965 |                                  1      |                         0.04306   |                 6.146 |                       1109   |
| delay_4h          |            1   |                 4 | exposure_only            |        105 |                           0.0589   |                             0.05518  |                         0.4474  |                           0.4844  |                                0.649  |                                  0.6706 |                         0.07584   |                 6.146 |                        630.9 |
| delay_4h          |            1   |                 4 | deficit_only             |        105 |                           0.05504  |                             0.0512   |                         0.4199  |                           0.4511  |                                0.6088 |                                  0.6356 |                         0.0797    |                 6.146 |                        628.3 |
| delay_4h          |            1   |                 4 | structure_only           |        105 |                           0.02601  |                             0.02558  |                         0.2049  |                           0.1953  |                                0.2979 |                                  0.2678 |                         0.1087    |                 6.146 |                        556.2 |
| delay_4h          |            1   |                 4 | random_positive          |        105 |                           0.01779  |                             0.01705  |                         0.1331  |                           0.1376  |                                0.1954 |                                  0.196  |                         0.1169    |                 6.146 |                        771.6 |
| high_budget       |            2   |                 0 | greedy_oracle            |        105 |                           0.1769   |                             0.1817   |                         1.336   |                           1.401   |                                1      |                                  1      |                        -0.04212   |                12.29  |                       1887   |
| high_budget       |            2   |                 0 | activated_bottleneck_law |        105 |                           0.174    |                             0.1805   |                         1.307   |                           1.367   |                                0.9783 |                                  0.9929 |                        -0.03929   |                12.29  |                       2184   |
| high_budget       |            2   |                 0 | exposure_only            |        105 |                           0.1213   |                             0.117    |                         0.9482  |                           0.9465  |                                0.7051 |                                  0.7044 |                         0.01346   |                12.29  |                       1241   |
| high_budget       |            2   |                 0 | deficit_only             |        105 |                           0.1142   |                             0.1092   |                         0.8972  |                           0.8909  |                                0.6659 |                                  0.6623 |                         0.02056   |                12.29  |                       1238   |
| high_budget       |            2   |                 0 | structure_only           |        105 |                           0.05496  |                             0.05382  |                         0.4463  |                           0.4086  |                                0.3285 |                                  0.2968 |                         0.07978   |                12.29  |                       1074   |
| high_budget       |            2   |                 0 | random_positive          |        105 |                           0.04042  |                             0.04174  |                         0.308   |                           0.3171  |                                0.2295 |                                  0.2313 |                         0.09432   |                12.29  |                       1498   |
| low_budget        |            0.5 |                 0 | greedy_oracle            |        105 |                           0.06429  |                             0.06587  |                         0.4784  |                           0.5102  |                                1      |                                  1      |                         0.07045   |                 3.073 |                        466.4 |
| low_budget        |            0.5 |                 0 | activated_bottleneck_law |        105 |                           0.06333  |                             0.06551  |                         0.4694  |                           0.5044  |                                0.9802 |                                  0.9908 |                         0.07141   |                 3.073 |                        569.7 |
| low_budget        |            0.5 |                 0 | exposure_only            |        105 |                           0.04148  |                             0.03985  |                         0.3216  |                           0.3117  |                                0.6773 |                                  0.6687 |                         0.09325   |                 3.073 |                        285.5 |
| low_budget        |            0.5 |                 0 | deficit_only             |        105 |                           0.039    |                             0.03766  |                         0.3038  |                           0.2907  |                                0.6401 |                                  0.6439 |                         0.09573   |                 3.073 |                        284.8 |
| low_budget        |            0.5 |                 0 | structure_only           |        105 |                           0.01701  |                             0.01703  |                         0.1361  |                           0.1346  |                                0.2878 |                                  0.2713 |                         0.1177    |                 3.073 |                        262.3 |
| low_budget        |            0.5 |                 0 | random_positive          |        105 |                           0.01022  |                             0.01023  |                         0.07754 |                           0.07864 |                                0.1633 |                                  0.1557 |                         0.1245    |                 3.073 |                        381.6 |
| scarce_and_late   |            0.5 |                 2 | greedy_oracle            |        105 |                           0.06087  |                             0.06255  |                         0.4511  |                           0.4722  |                                1      |                                  1      |                         0.07387   |                 3.073 |                        504.2 |
| scarce_and_late   |            0.5 |                 2 | activated_bottleneck_law |        105 |                           0.06045  |                             0.0623   |                         0.4473  |                           0.4686  |                                0.991  |                                  0.9977 |                         0.07428   |                 3.073 |                        564.2 |
| scarce_and_late   |            0.5 |                 2 | exposure_only            |        105 |                           0.03814  |                             0.03599  |                         0.2929  |                           0.2945  |                                0.6508 |                                  0.6632 |                         0.0966    |                 3.073 |                        293.4 |
| scarce_and_late   |            0.5 |                 2 | deficit_only             |        105 |                           0.03574  |                             0.03346  |                         0.2755  |                           0.273   |                                0.6129 |                                  0.6149 |                         0.099     |                 3.073 |                        291.4 |
| scarce_and_late   |            0.5 |                 2 | structure_only           |        105 |                           0.01571  |                             0.01512  |                         0.1245  |                           0.1238  |                                0.2789 |                                  0.2713 |                         0.119     |                 3.073 |                        268   |
| scarce_and_late   |            0.5 |                 2 | random_positive          |        105 |                           0.009628 |                             0.009576 |                         0.07256 |                           0.07489 |                                0.1612 |                                  0.1556 |                         0.1251    |                 3.073 |                        384.4 |

解释：这个表的核心列是 `mean_fraction_of_base_lp_gain`，即 fixed policy replay 得到的恢复收益占 base LP optimized gain 的比例。对于 base scenario，这就是 law/simple baseline 与 LP optimum 的直接差距；对于 low/high budget 或 delay scenario，它仍以 base LP gain 作参照，因此用于观察趋势，而不是声明这些新场景下的最优性。

## Event-Level Top-Tail Law

| city         |   event_id | event_start         |   baseline_objective |   recoverable_fraction |   top_1pct_value_share |   top_5pct_value_share |   top_10pct_value_share |   marginal_value_gini |   optimizer_selected_value_share |   loss_magnitude_rank |   recoverable_rank |   top_tail_rank |   decision_criticality_score |   decision_criticality_rank |   event_peak_positive_abnormal_deficit |   event_total_precip |
|:-------------|-----------:|:--------------------|---------------------:|-----------------------:|-----------------------:|-----------------------:|------------------------:|----------------------:|---------------------------------:|----------------------:|-------------------:|----------------:|-----------------------------:|----------------------------:|---------------------------------------:|---------------------:|
| Philadelphia |        640 | 2023-07-04 01:00:00 |             0.009872 |                 0.165  |                0.1133  |                 0.3421 |                  0.5173 |                0.718  |                          0.1589  |               0.0381  |             0.8571 |          1      |                      0.04053 |                      1      |                               0.002906 |             0.005906 |
| San Antonio  |        506 | 2024-07-27 11:00:00 |             0.006044 |                 0.1548 |                0.1058  |                 0.3296 |                  0.503  |                0.7076 |                          0.1163  |               0.02857 |             0.8286 |          0.9762 |                      0.0361  |                      0.9905 |                               0.00252  |             0.01444  |
| San Antonio  |        499 | 2024-07-22 20:00:00 |             0.0216   |                 0.1418 |                0.1058  |                 0.3296 |                  0.503  |                0.7076 |                          0.171   |               0.1048  |             0.6381 |          0.9905 |                      0.03306 |                      0.981  |                               0.006083 |             0.1903   |
| San Antonio  |        495 | 2024-07-13 13:00:00 |             0.03348  |                 0.1346 |                0.1058  |                 0.3296 |                  0.503  |                0.7076 |                          0.2085  |               0.1905  |             0.5524 |          0.9762 |                      0.03138 |                      0.9714 |                               0.009429 |             0.1444   |
| Philadelphia |        657 | 2023-07-28 22:00:00 |             0.05621  |                 0.1357 |                0.1039  |                 0.3212 |                  0.4934 |                0.7028 |                          0.2793  |               0.2762  |             0.5714 |          0.9333 |                      0.03064 |                      0.9619 |                               0.01508  |             0.2618   |
| Philadelphia |        647 | 2023-07-14 08:00:00 |             0.01199  |                 0.1844 |                0.07684 |                 0.2576 |                  0.4119 |                0.636  |                          0.1084  |               0.04762 |             0.9429 |          0.619  |                      0.03021 |                      0.9524 |                               0.001728 |             0.005906 |
| Philadelphia |        641 | 2023-07-07 11:00:00 |             0.004974 |                 0.186  |                0.07245 |                 0.2504 |                  0.4047 |                0.6331 |                          0.1157  |               0.01905 |             0.9524 |          0.5714 |                      0.02948 |                      0.9429 |                               0.001488 |             0.009843 |
| Philadelphia |        652 | 2023-07-19 06:00:00 |             0.08686  |                 0.141  |                0.09266 |                 0.2992 |                  0.4683 |                0.6862 |                          0.2535  |               0.4667  |             0.619  |          0.8762 |                      0.02895 |                      0.9333 |                               0.01343  |             0.05118  |
| San Antonio  |        504 | 2024-07-24 11:00:00 |             0.06417  |                 0.1284 |                0.1001  |                 0.3212 |                  0.4946 |                0.7019 |                          0.2492  |               0.3333  |             0.419  |          0.9238 |                      0.02894 |                      0.9238 |                               0.01449  |             0.1483   |
| San Antonio  |        502 | 2024-07-23 11:00:00 |             0.1743   |                 0.1242 |                0.1054  |                 0.3283 |                  0.5016 |                0.7068 |                          0.2803  |               0.7143  |             0.3238 |          0.9524 |                      0.02881 |                      0.9143 |                               0.0397   |             0.4436   |
| New York     |         98 | 2019-06-02 18:00:00 |             0.01551  |                 0.2565 |                0.06699 |                 0.2145 |                  0.3381 |                0.5233 |                          0.1218  |               0.05714 |             0.9905 |          0.3048 |                      0.02879 |                      0.9048 |                               0.01284  |             0.126    |
| San Antonio  |        492 | 2024-07-09 16:00:00 |             0.09517  |                 0.1233 |                0.1058  |                 0.3294 |                  0.5028 |                0.7075 |                          0.2834  |               0.5238  |             0.3048 |          0.9619 |                      0.02873 |                      0.8952 |                               0.0268   |             0.1667   |
| San Antonio  |        491 | 2024-07-06 20:00:00 |             0.05846  |                 0.1251 |                0.1044  |                 0.3255 |                  0.4976 |                0.7031 |                          0.2667  |               0.2952  |             0.3429 |          0.9429 |                      0.02862 |                      0.8857 |                               0.01637  |             0.6549   |
| Chicago      |        149 | 2019-07-13 20:00:00 |             0.01616  |                 0.154  |                0.09167 |                 0.2813 |                  0.4339 |                0.6418 |                          0.09987 |               0.06667 |             0.8095 |          0.781  |                      0.0278  |                      0.8762 |                               0.002701 |             0.1417   |
| Philadelphia |        656 | 2023-07-25 15:00:00 |             0.1611   |                 0.1411 |                0.09555 |                 0.2963 |                  0.4577 |                0.665  |                          0.261   |               0.6762  |             0.6286 |          0.8571 |                      0.02779 |                      0.8667 |                               0.02882  |             0.5      |

## 当前可读出的初步 law

Local activated-bottleneck law 的第一版可写成：

```text
recovery_value(action)
  ≈ persistent_future_deficit_area
    × exposed_OD_importance_or_origin_exposure
    × intervention_efficiency_per_cost
    × response_feasibility
```

Event top-tail law 的第一版可写成：

```text
decision_criticality(event)
  ≈ recoverable_fraction
    × top_tail_concentration_of_action_values
    × inequality_of_recovery_value_field
```

## 需要继续改进的地方

1. 下一版应挑选代表性 city-event 做 single-action marginal LP 或 perturbed optimum stability，用真实 LP 边际值验证当前解析 action label。
2. 当前 budget/delay augmentation 已有 fixed-policy replay，但还不是重新求解 Gurobi 的多情景 optimum；后续应挑选代表性 scenario 重新优化验证。
3. 当前 surrogate 是 ridge baseline，不是最终神经模型；后续可升级为 factorized action-value scorer 或 graph surrogate。
4. 当前 substitutability 没有被可靠刻画。简单 out-degree scarcity 在本数据中会损害排序，后续应加入替代路径、网络冗余或 OD rerouting proxy。

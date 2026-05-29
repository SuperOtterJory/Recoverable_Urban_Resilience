# Learning and Law Discovery V1

## 本版本做了什么

这一版把 event-level optimization outputs 转换成 action-token 学习问题。每个 token 表示 `city-event-unit-time-intervention`。目标不是直接学习 optimizer 是否选择该 token，而是构造一个可解释的 marginal recovery-value proxy：单位资源投到该 token 后，沿着无干预的被动恢复轨迹，估计它能减少多少未来加权功能损失。

这仍然是 V1：action value 是解析近似标签，不是逐 token 重新求解 single-action LP，也不是 perturbed-optimum stability。它的作用是建立 learning/law pipeline 的可复现骨架，并检验一个可解释 law 是否能跨城市排序高价值 action。

## 数据规模

- sampled action tokens: 186,536
- city-event scenarios: 105
- full candidate-action concentration rows: 105
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

V1 的可解释 law score 保留 action label 的核心结构，但不直接使用 optimizer 选择结果：

```text
activated_bottleneck_score(i,t,k)
  ≈ delay_feasible(i,t,k)
    × future_deficit_area(i,t,k)
    × OD_exposure_or_destination_importance(i,k)
    × intervention_efficiency_per_cost(i,t,k)
```

这里的 `future_deficit_area` 已经包含剩余时间窗口，因此不再额外乘一个简单的 time rank。早期草稿里我把 `1 - out_degree_rank` 当作 substitutability scarcity 强行乘进去，结果明显拉低排序表现；这一版先移除这个不稳定项，把 substitutability 留到后续用更可靠的替代路径或网络冗余指标刻画。

## 关键结果概览

- Leave-one-city-out mean Spearman: 0.8486
- Leave-one-city-out mean top-5% value capture: 0.8331
- Activated-bottleneck law top-5% value capture: 0.9636
- Activated-bottleneck law mean event Spearman: 0.9639

## Leave-One-City-Out Surrogate

| split          | heldout      |   n_tokens |   n_events |   pearson |   spearman |       mae |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |
|:---------------|:-------------|-----------:|-----------:|----------:|-----------:|----------:|-------------------------:|-------------------------:|--------------------------:|
| leave_city_out | Austin       |      19758 |         15 |    0.8707 |     0.9278 | 0.0005428 |                   0.8895 |                   0.9171 |                    0.9309 |
| leave_city_out | Chicago      |      27896 |         13 |    0.7937 |     0.8242 | 0.0001508 |                   0.9311 |                   0.9219 |                    0.9203 |
| leave_city_out | Dallas       |      12825 |         10 |    0.7506 |     0.748  | 0.000355  |                   0.5122 |                   0.5991 |                    0.6689 |
| leave_city_out | Houston      |       5158 |          4 |    0.7244 |     0.7977 | 0.0003784 |                   0.5903 |                   0.6685 |                    0.7623 |
| leave_city_out | New York     |      53346 |         24 |    0.7344 |     0.8453 | 0.0002046 |                   0.6382 |                   0.8411 |                    0.8628 |
| leave_city_out | Philadelphia |      37656 |         20 |    0.839  |     0.8961 | 0.0002554 |                   0.9613 |                   0.9359 |                    0.9336 |
| leave_city_out | San Antonio  |      29897 |         19 |    0.8558 |     0.9015 | 0.0003221 |                   0.947  |                   0.9484 |                    0.9438 |

## Leave-Regime-Out Surrogate

| split            | heldout   |   n_tokens |   n_events |   pearson |   spearman |       mae |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |
|:-----------------|:----------|-----------:|-----------:|----------:|-----------:|----------:|-------------------------:|-------------------------:|--------------------------:|
| leave_regime_out | low       |      83698 |         46 |    0.7605 |     0.6653 | 0.000323  |                   0.8855 |                   0.8972 |                    0.9075 |
| leave_regime_out | medium    |      65929 |         37 |    0.9106 |     0.9319 | 0.0002041 |                   0.8962 |                   0.9138 |                    0.9247 |
| leave_regime_out | high      |      36909 |         22 |    0.7024 |     0.8161 | 0.0003548 |                   0.5803 |                   0.6813 |                    0.7259 |

## Law Score 与 Baselines

| policy_score             |   n_tokens |   top_1pct_value_capture |   top_5pct_value_capture |   top_10pct_value_capture |   mean_spearman_by_event |
|:-------------------------|-----------:|-------------------------:|-------------------------:|--------------------------:|-------------------------:|
| activated_bottleneck_law |     186536 |                   0.9606 |                   0.9636 |                    0.9635 |                  0.9639  |
| exposure_only            |     186536 |                   0.5428 |                   0.6816 |                    0.7304 |                  0.6273  |
| deficit_only             |     186536 |                   0.554  |                   0.6749 |                    0.7127 |                  0.4648  |
| optimizer_selected       |     186536 |                   0.3438 |                   0.4657 |                    0.5294 |                  0.4335  |
| structure_only           |     186536 |                   0.2885 |                   0.376  |                    0.4282 |                 -0.06228 |

解释：`optimizer_selected` 在 action-value 排序里不一定最高，因为 optimizer 选择受到总预算、单期预算、部署上限、分段边际收益和替代 action 的共同约束；而 law score 评价的是“单个 action 的边际价值排序”。因此这里更应该看 law score 是否能捕捉 value field 的 top tail，而不是是否复刻 optimizer 的最终稀疏解。

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

1. 下一版应生成更强的 action-level oracle label：single-action marginal LP、greedy residual marginal value 或 perturbed optimum stability。
2. 需要加入更多 scenario augmentation，尤其是 budget 和 delay 变化，否则 law 仍主要来自 base scenario。
3. 当前 surrogate 是 ridge baseline，不是最终神经模型；后续可升级为 factorized action-value scorer 或 graph surrogate。
4. 当前 substitutability 没有被可靠刻画。简单 out-degree scarcity 在本数据中会损害排序，后续应加入替代路径、网络冗余或 OD rerouting proxy。

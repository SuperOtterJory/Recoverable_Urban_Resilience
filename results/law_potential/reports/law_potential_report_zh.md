# Law-Potential Analysis for Recoverable Urban Resilience

## 核心判断

从当前 optimization outputs 看，已经存在适合后续 learning-to-optimize、XAI 与 symbolic regression 的结构信号，但这些信号的层级不同：

1. **强信号**：预算强度、响应延迟、PWL 边际收益递减共同形成稳定的 recoverability response surface。这个结构已经足够训练一个小型 surrogate 或 learning-to-optimize policy，并可用 XAI 检查预算/延迟/城市固定效应。
2. **中等信号**：optimized policy 相对 damage/exposure/access heuristic 有稳定 decision leverage，说明模型输出中确实包含“智能配置优于朴素规则”的可学习结构。
3. **弱到中等信号**：城市结构变量，例如 speed deficit、OD concentration、TMC deficit concentration，与 recoverability/decision leverage 存在方向性相关，但当前只有 5 个城市，不能把它们当作可靠 law。它们更像后续 law extraction 的候选变量。

因此，目前结果可以衔接后续模块，但最适合先学习的是 **scenario-conditioned managed recovery law**，而不是直接宣称跨城市通用 law 已经被识别。

## 最强相关结构

| dataset | feature | target | n | spearman | pearson |
| --- | --- | --- | --- | --- | --- |
| tuning | budget_squared | recoverable_fraction | 72 | 0.9815 | 0.9349 |
| tuning | log_budget | recoverable_fraction | 72 | 0.9815 | 0.9739 |
| tuning | budget_intensity | recoverable_fraction | 72 | 0.9815 | 0.973 |
| scenario | budget_intensity | recoverable_fraction | 30 | 0.9285 | 0.984 |
| scenario | budget_intensity | decision_leverage_fraction | 30 | 0.8159 | 0.8777 |
| scenario | cost_share_S | decision_leverage_fraction | 30 | -0.5008 | -0.4179 |
| scenario | destination_volume_hhi | decision_leverage_fraction | 30 | 0.3731 | 0.2979 |
| scenario | cost_share_S | recoverable_fraction | 30 | -0.3548 | -0.2715 |
| scenario | congested_volume_share_speed_ratio_lt_0_8 | decision_leverage_fraction | 30 | -0.3132 | -0.3359 |
| scenario | mean_a_retention | decision_leverage_fraction | 30 | -0.2859 | -0.1611 |
| scenario | cost_share_R | decision_leverage_fraction | 30 | 0.2783 | 0.2023 |
| scenario | delay_score | decision_leverage_fraction | 30 | -0.2668 | -0.1679 |

解释：预算相关变量通常是 recoverability 的最强驱动；delay_score 与 recoverability 负相关；primitive mix 和 decision leverage 也有强信号。城市结构变量的相关性应谨慎解释，因为城市样本数很小。

## 简单 law-like surrogate 拟合

| model | target | n | r2 | formula_skeleton |
| --- | --- | --- | --- | --- |
| scenario_budget_delay_city | recoverable_fraction | 30 | 0.9928 | y = 0.009053 +0.251*log_budget -0.003926*delay_score -0.006995*city_Chicago +0.003182*city_Houston -0.01254*city_New York -0.00619*city_San Antonio |
| scenario_decision_leverage | decision_leverage_fraction | 30 | 0.9874 | y = 0.007184 -0.08367*log_budget -0.0007982*delay_score +0.5712*recoverable_fraction -0.0002837*city_Chicago -0.002809*city_Houston -0.004317*city_New York -0.001848*city_San Antonio |
| tuning_city_quadratic_delay | recoverable_fraction | 72 | 0.9834 | y = 0.009164 +0.2097*budget_intensity -0.03629*budget_squared -0.007178*delay_score +0.01324*city_Houston -0.006589*city_New York |
| tuning_city_budget_delay | recoverable_fraction | 72 | 0.9826 | y = 0.004026 +0.2542*log_budget -0.007178*delay_score +0.01324*city_Houston -0.006589*city_New York |
| tuning_budget_delay | recoverable_fraction | 72 | 0.96 | y = 0.006242 +0.2542*log_budget -0.007178*delay_score |
| tuning_log_budget | recoverable_fraction | 72 | 0.9485 | y = -0.0009358 +0.2542*log_budget |

解释：如果加入 city fixed effects，预算和延迟的低维表达可以很好解释 tuning results。这说明后续 neural network 不一定一开始就需要很复杂；可从低维 response surface 开始，再逐步引入城市结构特征。

## 预算响应曲线与边际收益递减

| delay_name | monotonic_share | mean_slope_ratio | mean_rec_at_1 |
| --- | --- | --- | --- |
| base | 1 | 0.6969 | 0.1785 |
| fast | 1 | 0.692 | 0.1952 |
| slow | 1 | 0.6775 | 0.1604 |

解释：所有 tuning 曲线保持单调，且由于 PWL diminishing returns 与 caps，预算响应呈现更保守、更可信的增长。是否严格凹取决于城市和延迟情景，但总体上不再是无限线性收益。

## 响应延迟惩罚

| budget_intensity | mean_fast_minus_slow | mean_relative_delay_penalty |
| --- | --- | --- |
| 0.06 | 0.001979 | 0.1162 |
| 0.12 | 0.004432 | 0.1408 |
| 0.18 | 0.006752 | 0.1514 |
| 0.24 | 0.00914 | 0.1592 |
| 0.3 | 0.01158 | 0.1656 |
| 0.5 | 0.01906 | 0.1747 |
| 0.75 | 0.02707 | 0.1761 |
| 1 | 0.03484 | 0.1786 |

解释：fast response 相比 slow response 有稳定收益，且预算越大时 delay penalty 的绝对值通常更明显。这为论文中的 response delay 参数提供了可学习结构。

## Decision Leverage

| city | scenario | best_heuristic_policy | recoverable_fraction | best_heuristic_recoverable_fraction | decision_leverage_fraction | relative_gain_over_best_heuristic |
| --- | --- | --- | --- | --- | --- | --- |
| Houston | very_high_budget | damage_based | 0.1608 | 0.1134 | 0.0474 | 0.4181 |
| Austin | very_high_budget | damage_based | 0.1468 | 0.1038 | 0.04293 | 0.4133 |
| Chicago | very_high_budget | damage_based | 0.1358 | 0.09592 | 0.03987 | 0.4157 |
| San Antonio | very_high_budget | damage_based | 0.1395 | 0.1026 | 0.0369 | 0.3595 |
| Austin | high_budget | access_based | 0.07028 | 0.04443 | 0.02585 | 0.5817 |
| Houston | high_budget | access_based | 0.07437 | 0.04954 | 0.02483 | 0.5012 |
| Austin | fast_response | access_based | 0.0522 | 0.0282 | 0.024 | 0.8509 |
| New York | very_high_budget | damage_based | 0.1248 | 0.1015 | 0.02329 | 0.2295 |
| Chicago | high_budget | access_based | 0.06248 | 0.04172 | 0.02076 | 0.4975 |
| San Antonio | high_budget | access_based | 0.06373 | 0.04345 | 0.02028 | 0.4668 |

解释：optimized policy 不只是优于 no-intervention，也优于 best heuristic。这一点非常重要，因为它对应 high-level idea 里的 decision leverage：智能干预配置本身具有可识别价值。

## Primitive Mix 结构

| scenario | n | mean_cost_share_R | mean_cap_utilization_R | mean_cost_share_C | mean_cap_utilization_C | mean_cost_share_S | mean_cap_utilization_S | mean_recoverable_fraction | mean_decision_leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| very_high_budget | 5 | 0.534 | 0.02918 | 0.3127 | 0.05898 | 0.1533 | 0.08589 | 0.1415 | 0.03808 |
| high_budget | 5 | 0.5516 | 0.01183 | 0.2677 | 0.01975 | 0.1807 | 0.04027 | 0.06532 | 0.02093 |
| fast_response | 5 | 0.7389 | 0.009418 | 0.1134 | 0.004933 | 0.1477 | 0.01966 | 0.04614 | 0.01802 |
| base | 5 | 0.514 | 0.00655 | 0.2678 | 0.01171 | 0.2182 | 0.02918 | 0.04212 | 0.01495 |
| delayed_response | 5 | 0.3205 | 0.004067 | 0.3972 | 0.01762 | 0.2823 | 0.03778 | 0.03829 | 0.01193 |
| low_budget | 5 | 0.5372 | 0.003392 | 0.2149 | 0.004675 | 0.2479 | 0.01653 | 0.02295 | 0.009319 |

解释：随着预算上升，R/C/S 的组合发生变化。这个 primitive mix regime shift 是 learning-to-optimize 可学习的结构，也适合后续 XAI 分析。

## 自我审视：哪些已经分析，哪些还不能宣称

已经分析：预算响应、延迟惩罚、decision leverage、primitive mix、简单 surrogate 可拟合性、城市结构变量相关性、结果可信度检查。

仍不能宣称：当前只有 5 个优化城市，不能可靠提取跨城市 universal law；`eta/cost/cap` 仍是 scenario-calibrated 参数，不是观测因果参数；城市结构变量和 recoverability 的相关性只是候选规律。

最终判断：当前 optimization results **足以支持进入 learning-to-optimize + XAI/symbolic regression 的下一阶段**，但下一阶段最合理的学习对象应是“在给定城市状态、预算、延迟、primitive constraints 下的 managed recovery response surface 和 decision leverage”，而不是直接从 5 个城市中提取终极城市韧性定律。

# Optimization Report: Recoverable Urban Resilience

## 核心结论

我已经把 draft 中的 continuous LP 实现为 Gurobi 模型，并用 OD demand、speed-deficit summary、rainfall-speed alignment 和 demand/network summaries 对参数进行了第一版 calibration。

当前结果说明：模型可以稳定求解，recoverable fraction 对预算强度和响应延迟有单调且可解释的反应。low/base/high budget 的平均 recoverability 分别约为 2.3%/4.2%/6.5%；very-high-budget 情景下平均上升到 14.2%。在 tuning grid 中，当 budget intensity 到 1.0 时，Chicago/New York/Houston 的可恢复比例范围约为 14.6%-21.9%。

因此，初步结论不是“数据无法支持 recoverable resilience”，而是：在当前参数化下，recoverable resilience 是存在的，但其幅度高度依赖资源预算、响应延迟和 intervention primitive 的边际有效性。这个结果符合论文 idea：observed resilience 与 recoverable resilience 不是同一个量。

## 可信度检查

| check | passed | detail |
| --- | --- | --- |
| all_scenarios_optimal | True | Every city-scenario LP should solve to OPTIMAL. |
| optimized_not_worse_than_baseline | True | Managed counterfactual objective should not exceed the no-intervention baseline. |
| recoverable_fraction_bounds | True | Recoverable fraction should stay within [0, 1]. |
| budget_feasible | True | Total used intervention cost should not exceed total budget. |
| deployment_caps_respected | True | Total primitive deployment should not exceed available continuous caps. |
| scenario_budget_monotonicity | True | Austin: OK; Chicago: OK; Houston: OK; New York: OK; San Antonio: OK |
| response_delay_monotonicity | True | Austin: OK; Chicago: OK; Houston: OK; New York: OK; San Antonio: OK |
| tuning_budget_monotonicity | True | Chicago/base: OK; Chicago/fast: OK; Chicago/slow: OK; Houston/base: OK; Houston/fast: OK; Houston/slow: OK; New York/base: OK; New York/fast: OK; New York/slow: OK |
| optimized_beats_best_heuristic | True | Optimized LP should be at least as good as the best evaluated heuristic policy. |

这些检查用于确认 LP 结果没有违反基本可行性和单调性：优化情景不能差于 baseline，预算越高可恢复比例不应下降，响应越快可恢复比例不应下降，且所有资源使用都必须满足预算和连续部署上限。

## 最强 recoverability 情景

| city | scenario | recoverable_fraction | baseline_objective | optimized_objective | total_cost_R | total_cost_C | total_cost_S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Houston | very_high_budget | 0.1608 | 0.6797 | 0.5704 | 1.626 | 0.9994 | 0.4011 |
| Austin | very_high_budget | 0.1468 | 0.8514 | 0.7265 | 1.661 | 1.105 | 0.562 |
| San Antonio | very_high_budget | 0.1395 | 0.7895 | 0.6793 | 1.744 | 0.9075 | 0.4187 |
| Chicago | very_high_budget | 0.1358 | 3.106 | 2.684 | 6.227 | 3.044 | 1.828 |
| New York | very_high_budget | 0.1248 | 1.388 | 1.215 | 2.736 | 1.795 | 0.8873 |
| Houston | high_budget | 0.07437 | 0.6797 | 0.6291 | 0.5649 | 0.5119 | 0.1337 |
| Austin | high_budget | 0.07028 | 0.8514 | 0.7916 | 0.8156 | 0.3006 | 0.2149 |
| San Antonio | high_budget | 0.06373 | 0.7895 | 0.7392 | 0.6685 | 0.3901 | 0.1694 |
| Chicago | high_budget | 0.06248 | 3.106 | 2.912 | 2.754 | 0.8957 | 0.7901 |
| New York | high_budget | 0.05574 | 1.388 | 1.31 | 1.114 | 0.3695 | 0.6838 |

## 城市层面比较

| city | mean_recoverable | max_recoverable | baseline_objective |
| --- | --- | --- | --- |
| Houston | 0.06708 | 0.1608 | 0.6797 |
| Austin | 0.0639 | 0.1468 | 0.8514 |
| San Antonio | 0.05771 | 0.1395 | 0.7895 |
| Chicago | 0.0569 | 0.1358 | 3.106 |
| New York | 0.05136 | 0.1248 | 1.388 |

Houston 在当前 calibration 下 recoverable fraction 最高，Chicago、Austin、New York 紧随其后。Chicago 的 baseline loss 最大，但它的 recoverable fraction 不是最高，说明损失大并不自动等于可恢复比例高，这一点与论文的核心概念很一致。

## 情景层面比较

| scenario | mean_recoverable | max_recoverable | mean_R | mean_C | mean_S |
| --- | --- | --- | --- | --- | --- |
| very_high_budget | 0.1415 | 0.1608 | 2.799 | 1.57 | 0.8195 |
| high_budget | 0.06532 | 0.07437 | 1.183 | 0.4936 | 0.3984 |
| fast_response | 0.04614 | 0.0522 | 0.9578 | 0.1016 | 0.1858 |
| base | 0.04212 | 0.04746 | 0.6353 | 0.3259 | 0.284 |
| delayed_response | 0.03829 | 0.04409 | 0.4195 | 0.4497 | 0.3759 |
| low_budget | 0.02295 | 0.02628 | 0.3697 | 0.09138 | 0.1615 |

## 优化相对朴素策略的 decision leverage

| city | scenario | policy | recoverable_fraction_optimized | recoverable_fraction_best_heuristic | decision_leverage_fraction | relative_gain_over_best_heuristic |
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
| Houston | fast_response | access_based | 0.05043 | 0.03084 | 0.01959 | 0.6353 |
| Austin | base | access_based | 0.04678 | 0.02723 | 0.01956 | 0.7184 |

这一步是论文论证中很关键的部分：如果 optimized policy 只是比 no-intervention 好，但没有比 damage-based 或 exposure-based 朴素策略好，那么“智能管理”的贡献会比较弱。当前结果中，优化相对 best heuristic 仍有额外收益，说明 decision leverage 是可观测且可量化的。

low/base/high/very-high budget 的 recoverability 呈明显单调上升。fast response 相比 delayed response 有提升，但提升幅度小于预算变化，说明当前参数中资源总量比响应延迟更强地控制结果。

## Tuning 上界检查

| city | budget_intensity | mean_recoverable | max_recoverable | min_recoverable |
| --- | --- | --- | --- | --- |
| Chicago | 0.5 | 0.09679 | 0.1083 | 0.08567 |
| Chicago | 0.75 | 0.136 | 0.1519 | 0.1203 |
| Chicago | 1 | 0.1721 | 0.1922 | 0.1518 |
| Houston | 0.5 | 0.1144 | 0.1237 | 0.1044 |
| Houston | 0.75 | 0.16 | 0.1736 | 0.1454 |
| Houston | 1 | 0.2019 | 0.2193 | 0.1832 |
| New York | 0.5 | 0.08796 | 0.09574 | 0.08055 |
| New York | 0.75 | 0.125 | 0.1358 | 0.1143 |
| New York | 1 | 0.16 | 0.1741 | 0.146 |

扩展预算 sweep 后，recoverability 没有停在 10% 以下，而是随预算继续上升。这说明早期结果偏小主要来自预算尺度保守，而非模型结构错误。

## 结果诊断

1. 加入 primitive-specific continuous deployment caps 和 concave piecewise-linear diminishing returns 后，解不再无约束地偏向 `S`。`S` 仍然重要，因为它直接作用于 `ell`，但 `R` 和 `C` 在多数高预算情景中也会进入最优解。
2. 高预算时 `R` durable restoration 和 `C` temporary capacity 明显增加，尤其 Houston、Austin、San Antonio。这说明模型并非只能使用单一 primitive，而是在不同资源尺度下会切换干预组合。
3. recoverability 的数值范围对预算非常敏感，因此论文中必须把 `R_rec(B, Delta)` 表达为预算/延迟条件下的函数，而不是单个城市常数。
4. 当前 calibration 是 empirical proxy，不是 intervention causal estimate。`eta^k`、cost、delay、decay 参数仍需要通过 scenario ensemble 或真实响应记录进一步校准。

## 已完成的改进与仍需扩展

已完成：Gurobi LP、no-intervention baseline、optimized counterfactual、city calibration、multi-city scenario run、budget/delay tuning、中文报告和图表。

下一步最值得做的是加入非 identity 的空间效应矩阵和真实 demographic/equity exposure。当前 continuous deployment caps 与 PWL diminishing returns 已经解决了一部分单一 primitive 过度使用的问题，但空间 spillover 与公平约束仍需要更丰富数据来支撑。

# Optimization Report: Recoverable Urban Resilience

## 核心结论

我已经把 draft 中的 continuous LP 实现为 Gurobi 模型，并用 OD demand、speed-deficit summary、rainfall-speed alignment 和 demand/network summaries 对参数进行了第一版 calibration。

当前结果说明：模型可以稳定求解，recoverable fraction 对预算强度和响应延迟有单调且可解释的反应。low/base/high budget 的平均 recoverability 分别约为 2.6%/4.8%/7.5%；very-high-budget 情景下平均上升到 16.2%。在 tuning grid 中，当 budget intensity 到 1.0 时，Chicago/New York/Houston 的可恢复比例范围约为 16.8%-25.3%。

因此，初步结论不是“数据无法支持 recoverable resilience”，而是：在当前参数化下，recoverable resilience 是存在的，但其幅度高度依赖资源预算、响应延迟和 intervention primitive 的边际有效性。这个结果符合论文 idea：observed resilience 与 recoverable resilience 不是同一个量。

## 最强 recoverability 情景

| city | scenario | recoverable_fraction | baseline_objective | optimized_objective | total_cost_R | total_cost_C | total_cost_S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Houston | very_high_budget | 0.1856 | 0.6797 | 0.5535 | 1.239 | 1.517 | 0.2705 |
| Austin | very_high_budget | 0.1676 | 0.8514 | 0.7088 | 1.444 | 0.9042 | 0.9797 |
| San Antonio | very_high_budget | 0.1605 | 0.7895 | 0.6628 | 1.754 | 0.8197 | 0.4963 |
| Chicago | very_high_budget | 0.157 | 3.106 | 2.618 | 7.078 | 2.108 | 1.913 |
| New York | very_high_budget | 0.1412 | 1.388 | 1.192 | 2.867 | 0.7693 | 1.782 |
| Houston | high_budget | 0.08325 | 0.6797 | 0.6231 | 0.3892 | 0.6828 | 0.1385 |
| Austin | high_budget | 0.08253 | 0.8514 | 0.7812 | 0.6475 | 0.27 | 0.4135 |
| Chicago | high_budget | 0.07203 | 3.106 | 2.882 | 3.293 | 0.5734 | 0.5732 |
| San Antonio | high_budget | 0.07024 | 0.7895 | 0.734 | 0.7733 | 0.1983 | 0.2563 |
| New York | high_budget | 0.06465 | 1.388 | 1.298 | 0.7101 | 0.04277 | 1.414 |

## 城市层面比较

| city | mean_recoverable | max_recoverable | baseline_objective |
| --- | --- | --- | --- |
| Houston | 0.07633 | 0.1856 | 0.6797 |
| Austin | 0.07377 | 0.1676 | 0.8514 |
| Chicago | 0.06501 | 0.157 | 3.106 |
| San Antonio | 0.06416 | 0.1605 | 0.7895 |
| New York | 0.05936 | 0.1412 | 1.388 |

Houston 在当前 calibration 下 recoverable fraction 最高，Chicago、Austin、New York 紧随其后。Chicago 的 baseline loss 最大，但它的 recoverable fraction 不是最高，说明损失大并不自动等于可恢复比例高，这一点与论文的核心概念很一致。

## 情景层面比较

| scenario | mean_recoverable | max_recoverable | mean_R | mean_C | mean_S |
| --- | --- | --- | --- | --- | --- |
| very_high_budget | 0.1624 | 0.1856 | 2.876 | 1.224 | 1.088 |
| high_budget | 0.07454 | 0.08325 | 1.163 | 0.3534 | 0.5591 |
| fast_response | 0.05205 | 0.06153 | 0.8714 | 0.02604 | 0.3477 |
| base | 0.04764 | 0.05466 | 0.6855 | 0.1635 | 0.3962 |
| delayed_response | 0.04425 | 0.05082 | 0.2345 | 0.4491 | 0.5615 |
| low_budget | 0.02551 | 0.02938 | 0.2644 | 0.04819 | 0.31 |

low/base/high/very-high budget 的 recoverability 呈明显单调上升。fast response 相比 delayed response 有提升，但提升幅度小于预算变化，说明当前参数中资源总量比响应延迟更强地控制结果。

## Tuning 上界检查

| city | budget_intensity | mean_recoverable | max_recoverable | min_recoverable |
| --- | --- | --- | --- | --- |
| Chicago | 0.5 | 0.1124 | 0.1253 | 0.1001 |
| Chicago | 0.75 | 0.1578 | 0.1764 | 0.1401 |
| Chicago | 1 | 0.2012 | 0.225 | 0.1786 |
| Houston | 0.5 | 0.131 | 0.1402 | 0.1217 |
| Houston | 0.75 | 0.1849 | 0.1989 | 0.1701 |
| Houston | 1 | 0.2346 | 0.2529 | 0.2153 |
| New York | 0.5 | 0.09998 | 0.1073 | 0.09308 |
| New York | 0.75 | 0.1422 | 0.1542 | 0.1313 |
| New York | 1 | 0.1833 | 0.1998 | 0.1682 |

扩展预算 sweep 后，recoverability 没有停在 10% 以下，而是随预算继续上升。这说明早期结果偏小主要来自预算尺度保守，而非模型结构错误。

## 结果诊断

1. 加入 primitive-specific continuous deployment caps 后，解不再无约束地偏向 `S`。`S` 仍然重要，因为它直接作用于 `ell`，但 `R` 和 `C` 在多数高预算情景中也会进入最优解。
2. 高预算时 `R` durable restoration 和 `C` temporary capacity 明显增加，尤其 Houston、Austin、San Antonio。这说明模型并非只能使用单一 primitive，而是在不同资源尺度下会切换干预组合。
3. recoverability 的数值范围对预算非常敏感，因此论文中必须把 `R_rec(B, Delta)` 表达为预算/延迟条件下的函数，而不是单个城市常数。
4. 当前 calibration 是 empirical proxy，不是 intervention causal estimate。`eta^k`、cost、delay、decay 参数仍需要通过 scenario ensemble 或真实响应记录进一步校准。

## 已完成的改进与仍需扩展

已完成：Gurobi LP、no-intervention baseline、optimized counterfactual、city calibration、multi-city scenario run、budget/delay tuning、中文报告和图表。

下一步最值得做的是加入 concave piecewise-linear diminishing returns 和非 identity 的空间效应矩阵。当前 continuous deployment caps 已经解决了一部分单一 primitive 过度使用的问题，但 diminishing returns 更适合表达真实干预的边际收益递减。

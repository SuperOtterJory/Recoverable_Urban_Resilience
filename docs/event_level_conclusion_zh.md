# Event-Level Data Mining and Optimization Conclusion

## 1. 现在的实验对象

当前 canonical analysis 已经不是“一个城市一个集计代表场景”，而是：

> 一个真实 speed-overlap rainfall event 对应一个 12 小时 full-zone LP。

状态 0 是事件开始小时，状态 1-12 是事件开始后的 12 个小时。每个城市使用全部 OD zones：

- New York: 1,940 zones
- Chicago: 882 zones
- Philadelphia: 642 zones
- Houston: 393 zones
- Dallas: 366 zones
- San Antonio: 304 zones
- Austin: 207 zones

最终可优化事件数为 105：

| city | event-level LP count |
|---|---:|
| New York | 24 |
| Philadelphia | 20 |
| San Antonio | 19 |
| Austin | 15 |
| Chicago | 13 |
| Dallas | 10 |
| Houston | 4 |

## 2. 速度影响如何重新计算

旧方法使用事件前 6 小时作为 baseline，这会混入早晚高峰/平峰周期。新版改为 matched temporal baseline：

```text
expected_deficit(hour)
  = 非降雨、非事件影响窗口中的 same-hour-of-week median speed deficit
```

若 same-hour-of-week 样本不足，再回退到 same-hour-of-day median 和全局非降雨 median。

然后：

```text
abnormal_deficit = observed_mean_deficit - expected_deficit
positive_abnormal_deficit = max(abnormal_deficit, 0)
```

这使 “positive speed impact” 的含义更清楚：不是简单比事件前 6 小时更差，而是比该城市同星期同小时通常的非降雨状态更差。

## 3. Calibration 如何重做

城市级动态先拟合：

```text
y[t+1] = a * y[t] + sum_l beta_l * rainfall[t-l]
```

其中 `y[t]` 是 positive abnormal speed deficit。

然后对每个真实事件：

```text
b0_city = y[event_start]
h_city[t] = max(y[t] - a * y[t-1], 0)
```

这比旧方法更好，因为 `A` 和 `h` 不再分别由“恢复小时数”和“平均事件冲击”启发式指定，而是在同一个动态递推框架中分解。

## 4. 事件级优化结果

105 个事件全部求得 OPTIMAL。城市平均 recoverable fraction：

| city | mean recoverable fraction | median | min | max |
|---|---:|---:|---:|---:|
| Philadelphia | 0.1597 | 0.1571 | 0.1266 | 0.1964 |
| Chicago | 0.1497 | 0.1324 | 0.1182 | 0.2680 |
| New York | 0.1369 | 0.1299 | 0.0988 | 0.2565 |
| San Antonio | 0.1366 | 0.1340 | 0.1231 | 0.1653 |
| Austin | 0.1311 | 0.1273 | 0.1078 | 0.1832 |
| Houston | 0.1100 | 0.1135 | 0.0939 | 0.1189 |
| Dallas | 0.0721 | 0.0690 | 0.0549 | 0.0936 |

直观解释：

- Philadelphia 的平均可恢复比例最高，虽然事件冲击峰值不大，但网络结构和恢复资源匹配较好。
- Chicago 有最高单事件 recoverability，说明特定事件下存在很强的结构性可恢复空间。
- Dallas 的事件冲击峰值最高，但 recoverability 最低，提示“损失大”不等于“可恢复空间大”。
- New York 规模最大、OD 最稀疏，恢复比例中等偏高；其资源更多投向 R，说明事件形态和结构位置使直接恢复 deficit 更有价值。

## 5. 城市结构信号

城市层面相关性仍只有 7 个城市，不能当作最终 law，但可以形成研究假设：

- `q_density` / `od_density_observed` 与 recoverability 负相关，Spearman 约 -0.61。
- `od_sparsity` 与 recoverability 正相关，Spearman 约 0.61。
- `n_units`、`od_rows`、`q_nnz` 与 recoverability 正相关，Spearman 约 0.57。
- 控制降雨总量和峰值后，event-level partial correlation 仍显示：
  - `q_density` 为负；
  - `od_sparsity` 为正；
  - `log_n_units`、`log_od_rows` 为正。

这说明结果不只是 “雨越大/损失越大” 导致的，城市 OD 依赖结构本身也在影响 recoverable fraction。

## 6. 一个非直觉结果：不是所有资源都投向最活跃区域

跨城市平均看：

- `C` 有约 54.3% 的成本落在 top-10% origin exposure 之外，但 93.4% 落在 top-10% destination importance 之内。
- `R` 有约 43.5% 的成本落在 top-10% origin exposure 之外，但几乎全部落在 top-10% destination importance 之内。
- `S` 只有约 9.8% 的成本落在 top-10% origin exposure 之外，但约 50.1% 落在 top-10% destination importance 之外。

解释：

- `R` 和 `C` 更像 destination/local deficit repair，偏向高 destination importance 和高 event deficit 区域。
- `S` 更像 origin-side experienced-loss shielding，偏向高 origin exposure 区域。
- 因此最优投放不是简单“哪里最活跃就修哪里”，而是不同资源对应不同的结构位置。

## 7. 对 paper idea 的契合程度

当前数据与 idea 的契合程度是比较高的，但边界也清楚：

支持的部分：

- 数据中确实存在可观测的 rainfall-speed abnormal events。
- OD demand 可以构建大规模 full-zone functional-dependence LP。
- 城市之间在 recoverability、OD sparsity、resource allocation pattern 上存在结构差异。
- 控制降雨强度后，结构变量仍保留一定解释力。

仍需模型部分完成的部分：

- 当前是 observational calibration，不是严格因果识别。
- R/C/S 的真实干预效果、预算制度、delay 不是从原始数据直接观测到的。
- 最终 law 需要在模型部分进行 counterfactual、sensitivity、robustness 和 possibly out-of-sample validation。

因此，当前 data mining 可以支撑 paper 继续推进：它已经说明数据中存在足够的跨城市、跨事件结构信号；剩下的关键结论必须依赖正式模型与反事实实验来闭合。

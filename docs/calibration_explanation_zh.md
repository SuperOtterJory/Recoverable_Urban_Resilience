# Event-Level Calibration Explanation

## 当前优化场景是什么

新版 canonical analysis 不再使用“一个城市一个集计代表场景”。现在的优化问题是：

> 一个真实 rainfall event 对应一个 12 小时 full-zone LP。

更具体地说：

- 时间尺度：事件开始小时作为状态 0，之后 12 个 hourly steps 作为恢复窗口。
- 空间尺度：使用该城市全部 OD zones，不再抽取 top OD。
- `Q_ij`：由 `demand.csv` 中 origin `i` 到 destination `j` 的 OD volume 行归一化得到。
- `p_i`：origin `i` 的出行需求占比。
- `b0_i`：事件开始小时的城市级异常速度损失，按 destination vulnerability 分配到 zone。
- `A_i`：由动态标定得到的自然滞留率，再加入轻微 destination vulnerability drag。
- `h_i,t`：每个真实事件的观测异常序列在扣除自然滞留后仍然增加的正创新量。

因此，当前 LP 的含义是：

> 给定某次真实降雨事件在速度系统中形成的异常损失轨迹，在固定 R/C/S 资源制度下，城市 OD 功能结构能支持多大比例的可恢复功能损失。

## 速度损失如何得到

原始 speed 数据中先计算：

```text
speed_ratio = actual_speed / baseline_speed
speed_deficit = max(0, 1 - speed_ratio)
```

其中 `baseline_speed` 优先使用 `historical_average_speed`，否则使用 `reference_speed`。这一步是路段/时间对应的速度损失，不是和全城市平均速度比较。

事件影响不再用事件前 6 小时比较，而是：

```text
expected_deficit(hour)
  = 非降雨、非事件影响窗口中的 same-hour-of-week median speed deficit

abnormal_deficit = observed_mean_deficit - expected_deficit
positive_abnormal_deficit = max(abnormal_deficit, 0)
```

这样可以减少早晚高峰、夜间、周末等周期规律造成的伪冲击。

## `A` 如何标定

先在城市级 hourly series 上拟合一个动态模型：

```text
positive_abnormal_deficit[t+1]
  = a * positive_abnormal_deficit[t]
  + sum_l beta_l * precipitation[t-l]
```

这里：

- `a` 是自然滞留率：没有新的正冲击时，异常速度损失保留到下一小时的比例。
- `beta_l` 是降雨滞后核：用于诊断降雨冲击的时间滞后，也可在缺失速度观测时作为 fallback。

这个拟合不是严格因果识别，但比旧版本更合理，因为它把“当前损失的自然延续”和“降雨滞后输入”放在同一个递推方程中估计，而不是分别用峰值和恢复小时数启发式指定。

## `h` 如何标定

对每个真实事件，设城市级正异常速度损失为 `y[t]`。在给定自然滞留率 `a` 后，外部扰动创新定义为：

```text
h_city[t] = max(y[t] - a * y[t-1], 0)
```

含义是：

- 如果下一小时异常损失只是由上一小时损失自然保留下来，`h_city[t]` 接近 0。
- 如果下一小时异常损失高于自然滞留能解释的水平，多出来的正值被解释为外部扰动创新。
- 如果速度损失下降，`h_city[t] = 0`；下降部分由自然恢复解释，不把负扰动放进 LP。

这回答了 “`h` 和自然恢复在 ground truth 中混杂” 的问题：我们仍不能做严格识别，但现在至少使用同一个动态方程进行分解。`A` 先由跨小时递推拟合，`h` 再作为每个事件的正创新 residual 得到。

## 空间分配

当前没有 zone-level speed loss，所以城市级事件信号需要分配到 OD zones。分配权重使用 destination vulnerability：

```text
destination_vulnerability_i
  = 0.35 + 0.65 * normalized_destination_volume_i

relative_i
  = (1 - blend) + blend * destination_vulnerability_i
```

然后用 `p_i` 加权归一化，使城市级加权平均信号保持不变：

```text
sum_i p_i * relative_i = 1

b0_i = b0_city * relative_i
h_i,t = h_city[t] * relative_i
```

因此，`b0` 和 `h` 的城市平均水平直接来自 speed/rain event 数据；空间异质性来自 OD destination exposure。

## 预算如何给定

预算仍然是相对事件负担的比例，而不是固定美元量：

```text
event_burden = sum_i b0_i + sum_i,t h_i,t
total_budget = budget_intensity * total_budget_multiplier * event_burden
```

当前 base scenario 中：

```text
budget_intensity = 0.18
total_budget_multiplier = 1.25
R/C/S delays = 2/0/1 hours
```

这意味着不同事件的预算会随事件负担变化。后续如果要研究“同一绝对资源约束下的跨城市公平性”，可以再设置固定预算或按人口/OD volume 标准化预算。

## 当前结果能说明什么，不能说明什么

能说明：

- 在真实 rainfall event 的 12 小时窗口中，城市 OD 结构会影响 recoverable fraction。
- 如果某些结构变量在控制降雨强度后仍与 recoverability 相关，它们比“预算越大越好”更接近 paper 需要的 city-structure law。
- R/C/S 的投放位置可以检验“是否总是恢复最活跃区域”这一类非直觉问题。

不能说明：

- 不能严格证明降雨是所有异常速度损失的唯一原因。
- 不能直接把 7 个有 speed-overlap 的城市相关性宣称为 universal law。
- 不能替代后续模型部分；当前 data mining 的作用是判断数据是否有足够结构信号支撑研究方向。

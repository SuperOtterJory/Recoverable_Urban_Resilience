# Calibration Explanation

## 当前优化场景是什么

当前 full-zone LP 不是“一天一个模型”，也不是“把每一次真实降雨事件分别优化一次”。它是一个 **代表性恢复场景**：

- 时间尺度：12 个 hourly steps。
- 初始状态 `b0`：来自该城市 speed-deficit 数据的经验损失水平。
- 外部扰动 `h`：来自 rainfall-speed event analysis 的平均事件冲击。
- 自然恢复 `A`：来自事件窗口中观测到的恢复小时数。
- 管理动作：在同一恢复制度下分配 R/C/S 三类资源。

因此，当前 LP 的含义是：

> 给定一个由城市历史速度损失和降雨事件响应校准出来的代表性扰动-恢复过程，在固定资源制度下，不同城市结构能支持多少可恢复功能损失？

## 哪些量是直接来自数据

### OD demand

`OD` 是直接从 `demand.csv` 获得的。当前 canonical configuration 使用所有 OD zones，而不是抽样。

- `Q_ij`：从 origin `i` 到 destination `j` 的 OD volume 归一化得到。
- `p_i`：origin `i` 的出行需求占比。
- `destination_importance_j`：由 `Q^T p` 得到，表示 destination 被多少 origin exposure 依赖。

### 速度损失

速度损失来自 speed CSV：

```text
speed_ratio = actual_speed / baseline_speed
speed_deficit = max(0, 1 - speed_ratio)
```

其中 `baseline_speed` 优先使用 `historical_average_speed`，否则使用 `reference_speed`。所以它不是和全城市平均速度比，而是和该路段/时间对应的历史或参考速度比。

## `b0` 如何标定

城市级损失信号：

```text
city_signal =
  0.45 * mean_deficit
+ 0.45 * p90_deficit
+ 0.10 * severe_deficit_share_20pct
```

然后按 destination vulnerability 分配到 unit：

```text
destination_vulnerability_i = 0.35 + 0.65 * normalized_destination_volume_i
b0_i = city_signal * ((1 - blend) + blend * destination_vulnerability_i)
```

其中 `blend = 0.55`。

解释：`b0_i` 不是逐 zone 直接观测速度损失，而是用城市级 speed-deficit 信号和 OD destination exposure 做空间分配的 proxy。

## 自然恢复 `A` 如何标定

先从 rainfall-speed event windows 中得到 `median_event_recovery_hours`。如果没有可靠恢复时间，就用 default 12 小时。

```text
tau = clip(median_event_recovery_hours + min_recovery_tau_hours, 6, 24)
base_retention = exp(-1 / tau)
A_i = base_retention + structural_drag_i
```

`structural_drag_i` 随 destination vulnerability 增加，最大约 0.04，最后把 `A_i` 限制在 `[0.70, 0.985]`。

解释：`A_i` 越接近 1，损失自然消退越慢；越小，恢复越快。

## 外部扰动 `h` 如何标定

先从 rainfall-speed event windows 中得到平均事件冲击：

```text
event_impact = mean_event_deficit_impact
```

然后把它分配到 4 个小时的短扰动 profile：

```text
profile = [0.45, 0.30, 0.17, 0.08]
h_i,t = rainfall_shock_scale * event_impact * profile_t * destination_vulnerability_i
```

其中 `rainfall_shock_scale = 0.45`，并把 `h` 截断在 `[0, 0.12]`。

解释：`h` 表示降雨事件带来的额外短期扰动，而不是完整降雨序列。

## 关键限制：`A` 和 `h` 在 ground truth 中是混杂的

你的判断是正确的。真实观测中，我们看到的是速度损失曲线：

```text
observed loss trajectory = external shock + endogenous recovery + traffic dynamics + noise
```

仅靠当前数据，不能严格识别：

```text
哪些变化来自 h，哪些变化来自 A
```

当前做法是启发式分解：

- 用事件冲击峰值标定 `h` 的幅度；
- 用事件后恢复小时数标定 `A` 的恢复速度；
- 用 vulnerability 把城市级参数分配到各 zone。

因此它是 data-informed calibration，不是因果识别或统计拟合。后续更强版本应该对 event-level trajectories 做 joint fitting，例如：

```text
minimize || observed_deficit_t - simulated_deficit_t(A, h) ||
```

并使用 held-out rainfall events 验证。

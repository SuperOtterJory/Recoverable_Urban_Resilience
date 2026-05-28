# Rainfall Event Definitions

## 什么是 rainfall event

当前把连续的正降雨小时段定义为一次 rainfall event。

例如：

```text
10:00 rain > 0
11:00 rain > 0
12:00 rain = 0
13:00 rain > 0
```

这会被定义为两次事件：10:00-11:00 和 13:00。

## 什么是 speed overlap event

rainfall 数据覆盖多个年份，但 speed 数据通常只覆盖一个月。因此不是所有降雨事件都有同步速度观测。

以 New York 为例：

```text
speed coverage = 2019-06-01 00:00:00 到 2019-07-01 23:00:00
full rainfall events = 868
speed-overlap rainfall events = 26
positive abnormal impact events = 24
```

含义是：2019-2024 的降雨记录中有 868 次正降雨事件，但只有 26 次发生在当前拥有 speed observations 的月份内；这 26 次中有 24 次在事件窗口内出现了高于匹配时间基线的正异常速度损失。

## 为什么不再用事件前 6 小时作 baseline

事件前 6 小时很容易受到早晚高峰、平峰、夜间低流量等周期规律影响。例如一场雨从早上 8 点开始，如果只和凌晨 2-7 点比较，速度损失上升可能主要来自通勤高峰，而不是降雨。

因此新版 rainfall impact 使用 matched temporal baseline：

```text
expected_deficit(hour)
  = 非降雨、非事件影响窗口中的 same-hour-of-week median speed deficit
```

如果 same-hour-of-week 样本不足，则依次回退到：

```text
same-hour-of-day median
global non-rain median
```

然后定义：

```text
abnormal_deficit = observed_mean_deficit - expected_deficit
positive_abnormal_deficit = max(abnormal_deficit, 0)
```

这不是严格因果识别，但比“前 6 小时比较”更能排除日内周期和星期周期带来的机械波动。

## 什么是 positive abnormal impact event

对每个 speed-overlap rainfall event，取事件开始到事件结束后 12 小时的窗口，计算窗口内是否出现正异常速度损失峰值：

```text
peak_positive_abnormal_deficit > 0
```

若成立，就记为 positive abnormal impact event。

它的含义不是“降雨已被严格证明造成速度下降”，而是：

> 在降雨事件窗口内，观测速度损失高于该城市该星期小时通常会出现的非降雨速度损失。

这个定义已经尽量剥离了高峰/平峰周期，但仍可能混有事故、施工、其他天气、传感器覆盖差异等因素。因此它适合作为 data-informed event scenario 的输入，而不是最终因果结论。

## 输出表

主要输出位于：

```text
results/data_mining/tables/rainfall_event_impact_summary.csv
results/data_mining/tables/rainfall_event_impact_details.csv
results/data_mining/tables/speed_hourly_abnormal_deficit.csv
```

其中 `rainfall_event_impact_details.csv` 是事件级表，后续 event-level LP 只使用 speed-overlap 且 `peak_positive_abnormal_deficit > 0` 的事件。

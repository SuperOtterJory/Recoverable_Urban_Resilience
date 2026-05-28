# Rainfall Event Definitions

## 什么是 rainfall event

当前把 **连续正降雨小时段** 定义为一次 rainfall event。

例如：

```text
10:00 rain > 0
11:00 rain > 0
12:00 rain = 0
13:00 rain > 0
```

这会被定义为两次事件：10:00-11:00 和 13:00。

## 什么是 speed overlap event

rainfall 数据覆盖 2019-2024 的多个年份，但 speed 数据通常只覆盖一个月。例如 New York 的 speed 数据覆盖：

```text
2019-06-01 00:00:00 到 2019-07-01 23:00:00
```

因此：

- New York 全部 rainfall 数据中有 868 次正降雨事件；
- 其中只有 26 次落在 speed 数据覆盖月份内；
- 这 26 次才叫 speed overlap events；
- 其他 842 次虽然真实发生过降雨，但没有同步 speed 数据，不能估计速度影响。

所以 `speed overlap` 的含义不是“只有 26 次下雨”，而是：

> 只有 26 次降雨事件可以和当前可用 speed observations 对齐。

## 什么是 positive speed impact

对每个 speed-overlap rainfall event，计算：

```text
baseline = event_start 前 6 小时 mean speed deficit
peak = event_start 到 event_end 后 12 小时窗口内的最大 mean speed deficit
peak_extra_deficit = peak - baseline
```

如果：

```text
peak_extra_deficit > 0
```

就记为一次 positive speed impact event。

因此 New York 的结果：

```text
full rainfall events = 868
speed overlap events = 26
positive speed impact events = 23
```

应该解释为：

> 2019-2024 全部降雨记录中有 868 次正降雨事件；其中 26 次发生在有 speed 数据的月份；这 26 次里有 23 次在事件窗口内出现了相对事件前 6 小时更高的平均速度损失。

这不是严格因果声明。它说明 rainfall event 与 speed deficit increase 在时间窗口上对齐，但仍可能包含通勤高峰、事故、施工、其他天气因素等混杂。

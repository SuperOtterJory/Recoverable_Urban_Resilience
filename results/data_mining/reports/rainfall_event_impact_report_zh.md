# Rainfall Event Impact With Matched Baselines

事件定义：连续正降雨小时段为一次 rainfall event。速度影响只在 speed 数据覆盖月份内估计。

新的 impact 不再使用事件前 6 小时作为主 baseline，而是先用非降雨、非事件影响窗口中的 same-hour-of-week median speed deficit 作为 expected deficit；若样本不足，则回退到 same-hour-of-day median，再回退到全局非雨 median。

| city | full rain events | overlap events | positive abnormal impact events | mean peak abnormal | max peak abnormal | median recovery h | worst event |
|---|---:|---:|---:|---:|---:|---:|---|
| New York | 868 | 26 | 24 | 0.0245 | 0.0681 | 1.5000 | 2019-06-18 13:00:00 |
| Philadelphia | 946 | 20 | 20 | 0.0103 | 0.0288 | 1.5000 | 2023-07-25 15:00:00 |
| San Antonio | 553 | 19 | 19 | 0.0196 | 0.0397 | 2.0000 | 2024-07-23 00:00:00 |
| Austin | 534 | 15 | 15 | 0.0200 | 0.0430 | 2.0000 | 2024-07-18 08:00:00 |
| Chicago | 849 | 15 | 13 | 0.0171 | 0.0413 | 2.0000 | 2019-07-18 08:00:00 |
| Dallas | 555 | 17 | 10 | 0.0447 | 0.0622 | 6.0000 | 2019-04-23 19:00:00 |
| Houston | 462 | 11 | 4 | 0.0337 | 0.0528 | 1.5000 | 2019-04-18 05:00:00 |
| Los Angeles | 284 | 0 | 0 |  |  |  |  |
| Phoenix | 219 | 0 | 0 |  |  |  |  |
| San Diego | 458 | 0 | 0 |  |  |  |  |
| San Jose | 439 | 0 | 0 |  |  |  |  |

解释：positive abnormal impact event 表示事件窗口内出现了高于 matched temporal baseline 的正速度损失异常。它比前 6 小时比较更能降低早晚高峰和平峰切换造成的偏误，但仍不是严格因果识别。

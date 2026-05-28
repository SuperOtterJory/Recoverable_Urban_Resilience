# Event-Level Dynamic Calibration

本表用 matched temporal baseline 后的正异常速度损失来估计动态：

`positive_abnormal_deficit[t+1] = a * positive_abnormal_deficit[t] + sum_l beta_l * precipitation[t-l]`

`a` 表示没有新降雨冲击时异常损失保留到下一小时的比例；`beta_l` 是降雨滞后核，之后会被用于每个真实降雨事件的 LP disturbance `h[t]`。

| city | status | a retention | natural half-life h | rain kernel sum | peak lag h | test RMSE | test R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Austin | estimated | 0.7541 | 2.46 | 0.0004 | 6 | 0.0040 | 0.484 |
| Chicago | estimated | 0.7943 | 3.01 | 0.0071 | 6 | 0.0041 | 0.678 |
| Dallas | estimated | 0.9558 | 15.35 | 0.0054 | 5 | 0.0051 | 0.834 |
| Houston | estimated | 0.8648 | 4.77 | 0.0129 | 1 | 0.0057 | 0.754 |
| New York | estimated | 0.9021 | 6.73 | 0.0005 | 2 | 0.0052 | 0.676 |
| Philadelphia | estimated | 0.6900 | 1.87 | 0.0165 | 4 | 0.0025 | 0.626 |
| San Antonio | estimated | 0.7016 | 1.96 | 0.0008 | 6 | 0.0040 | 0.473 |

解释：这个标定仍不是严格因果识别，但它避免了“前 6 小时 baseline”把早晚高峰周期误当成降雨影响的问题，并且用时间递推关系把自然恢复项和外部降雨冲击项分离开。
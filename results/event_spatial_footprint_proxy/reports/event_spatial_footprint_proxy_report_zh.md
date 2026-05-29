# Event Spatial Footprint Proxy V33

本版目标是检查 V32 暴露的边界：当前 LP calibration 把城市级事件冲击按 OD vulnerability 投影到 zones，因此 event-level top-tail law 可能只是城市模板信号。这里额外从 raw TMC speed records 构造一个独立的 event-zone speed footprint proxy。

## 方法

1. 将每条 TMC segment 的中点映射到最近 OD zone centroid。
2. 对 raw speed records 计算 `deficit=max(0,1-speed/baseline_speed)`。
3. 使用非降雨、非事件影响窗口中的 same TMC + same hour-of-week mean 作为 expected deficit；缺失时退回 same TMC + hour-of-day、city hour-of-week、city hour-of-day、city global baseline。
4. 在每个 rainfall event 的 12 小时窗口内计算 positive abnormal TMC deficit，并按 TMC miles 聚合到 OD zone。
5. 比较 observed footprint distribution 与当前 calibration 使用的 OD vulnerability template。

## 关键指标

- 有 footprint 的事件数：105，城市数：7。
- footprint-template 平均 cosine similarity：0.3587。
- top-20 footprint zones 与 top-20 OD-template zones 的平均 Jaccard：0.0111。
- OD-template top 5% zones 平均捕获 observed footprint mass：0.0481。
- observed footprint top 5% zones 平均集中度：0.4952。
- 城市内 top-20 footprint zones 平均事件间 Jaccard：0.7793。
- TMC 到 OD zone 映射中，10km 内平均比例：0.8936。

## 城市摘要

| city | events | cosine | top20 template Jaccard | template top5 captures footprint | within-city top20 Jaccard | mapped <=10km |
|---|---:|---:|---:|---:|---:|---:|
| New York | 24 | 0.1934 | 0.0000 | 0.0548 | 0.8212 | 0.8934 |
| Dallas | 10 | 0.2772 | 0.0000 | 0.0244 | 0.8481 | 0.6731 |
| Philadelphia | 20 | 0.3085 | 0.0000 | 0.0291 | 0.8467 | 0.8366 |
| Chicago | 13 | 0.3694 | 0.0122 | 0.0851 | 0.7568 | 0.9751 |
| Austin | 15 | 0.5001 | 0.0017 | 0.0285 | 0.7161 | 0.9343 |
| San Antonio | 19 | 0.5106 | 0.0503 | 0.0616 | 0.6017 | 0.9598 |
| Houston | 4 | 0.5180 | 0.0064 | 0.0497 | 0.8642 | 0.9832 |

## 解释

如果 observed footprint 与 OD vulnerability template 的重合度较低，说明当前 calibration 的空间投影确实过强，下一步应该把 `b0_i` 和 `h_i,t` 升级为 `OD vulnerability + observed TMC footprint` 的混合场。若重合度很高，则说明当前 OD template 已经近似捕捉了事件影响区域，V32 的限制反而没有那么严重。

这个 proxy 仍然不是最终因果识别：TMC 到 OD zone 是最近邻映射，TMC coverage 与 OD zones 不是完全一致，且 TMC speed abnormal 仍受非降雨因素影响。但它已经比纯城市级投影多了一层事件内空间观测信号。

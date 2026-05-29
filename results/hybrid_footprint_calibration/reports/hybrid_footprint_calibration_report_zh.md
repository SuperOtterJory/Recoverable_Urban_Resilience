# Hybrid Footprint Calibration Sensitivity V34

本版把 V33 的 TMC-derived event-zone footprint 放入 calibration 敏感性测试：保持每个事件的城市级 `b0` 和 `h[t]` 总信号不变，只改变空间分配，从纯 OD vulnerability template 改为 `OD-template + observed TMC footprint` 的混合场。

这里同时报告两种 action-value field：

1. `small-signal`：当前 learning 主标签，只判断未来是否仍有正损失，因此对损失幅度不敏感。
2. `finite/magnitude-aware`：使用 `finite_deficit_area_value`，把未来损失幅度也放入 action value。

## 关键结论

- 主分析 blend = 0.50，覆盖 100 个事件、7 个城市。
- small-signal field 几乎完全不变：base vs hybrid Spearman = 1.0000，top-5% action Jaccard = 1.0000。
- magnitude-aware finite field 明显改变：base vs hybrid Spearman = 0.9579，top-5% action Jaccard = 0.5403。
- finite top-5% units 捕获的 observed footprint mass 从 0.0474 变为 0.3167，平均变化 0.2692。
- finite top-20 units 捕获的 observed footprint mass 从 0.0432 变为 0.2460。
- small-signal top-tail 零城市内变化的城市数为 5 -> 5；finite field 为 0 -> 0。
- hybrid/base no-intervention objective ratio 平均为 0.9803，说明这一步主要改变空间分布，而不是事件总强度。

## 城市摘要

| city | blend | small Spearman | finite Spearman | finite top5 Jaccard | base finite footprint mass | hybrid finite footprint mass | delta | finite base top5 range | finite hybrid top5 range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Philadelphia | 0.50 | 1.0000 | 0.9451 | 0.4111 | 0.0258 | 0.4468 | 0.4210 | 0.1744 | 0.1636 |
| Dallas | 0.50 | 1.0000 | 0.9417 | 0.4062 | 0.0238 | 0.3610 | 0.3372 | 0.0231 | 0.0207 |
| New York | 0.50 | 1.0000 | 0.9518 | 0.5713 | 0.0557 | 0.3344 | 0.2787 | 0.0673 | 0.0707 |
| Austin | 0.50 | 1.0000 | 0.9622 | 0.5474 | 0.0262 | 0.2692 | 0.2431 | 0.1333 | 0.1342 |
| Chicago | 0.50 | 1.0000 | 0.9717 | 0.5908 | 0.0945 | 0.3119 | 0.2173 | 0.1131 | 0.1203 |
| San Antonio | 0.50 | 1.0000 | 0.9747 | 0.6620 | 0.0575 | 0.2062 | 0.1487 | 0.1495 | 0.1618 |
| Houston | 0.50 | 1.0000 | 0.9569 | 0.5660 | 0.0447 | 0.1676 | 0.1229 | 0.0361 | 0.0349 |

## 解释

这个结果不是简单的“footprint 无效”。更准确地说：当前 small-signal law 的标签定义把所有仍有正损失的 zone 视为 active，因此只要 OD-template 与 hybrid 都让大多数 zone 保持正 deficit，事件 footprint 就不会改变该标签的排序。

但 magnitude-aware finite field 会随 hybrid footprint 改变，说明 V33 发现的空间信号确实能进入 recoverability learning target。下一步需要在 hybrid calibration 下重新求解 full LP，并检查 finite/residual law 的变化是否会转化为最终优化选择，而不只是 first-order proxy 的变化。

# Large-Scale City-Structure Analysis

## 核心变化

本轮分析不再使用 top-35 OD 单元作为主模型，而是对有可用 speed 数据的 7 个城市使用全 OD zone。New York 为 1,940 个单元、707,646 个非零 OD 依赖；Chicago 为 882 个单元、337,138 个非零 OD 依赖。LP 使用稀疏 CSR 形式，只对真实 OD 非零依赖建立 access-loss 约束。

为了避免把预算曲线误读成城市结构规律，large-scale 主结果只保留同一 base 情景：budget intensity = 0.18，R/C/S 延迟固定为 2/0/1 小时，干预效率、成本、部署上限和 diminishing returns 也保持一致。这里比较的是在同一制度设定下，不同城市结构对应的 recoverability 和 decision leverage。

## 城市结果排序

| city         |   recoverable_fraction |   decision_leverage_fraction | best_heuristic_policy   |   n_units |   q_nnz |   q_density |   od_density_observed |   p90_deficit |   mean_peak_extra_deficit |   max_peak_extra_deficit |   top_10pct_tmc_deficit_share |   congested_volume_share_speed_ratio_lt_0_8 |
|:-------------|-----------------------:|-----------------------------:|:------------------------|----------:|--------:|------------:|----------------------:|--------------:|--------------------------:|-------------------------:|------------------------------:|--------------------------------------------:|
| New York     |                0.09977 |                      0.05944 | access_based            |      1940 |  707646 |      0.188  |                0.1885 |        0.3014 |                   0.04299 |                  0.1013  |                        0.307  |                                      0.7678 |
| Chicago      |                0.08682 |                      0.05134 | access_based            |       882 |  337138 |      0.4334 |                0.4334 |        0.5333 |                   0.1209  |                  0.237   |                        0.2006 |                                      0.3826 |
| Houston      |                0.08319 |                      0.04489 | access_based            |       393 |   99541 |      0.6445 |                0.6445 |        0.1924 |                   0.02583 |                  0.08094 |                        0.2945 |                                      0.407  |
| Philadelphia |                0.076   |                      0.04393 | access_based            |       642 |  152419 |      0.3698 |                0.3704 |        0.2016 |                   0.03032 |                  0.06266 |                        0.3914 |                                      0.5076 |
| San Antonio  |                0.07043 |                      0.03762 | access_based            |       304 |   78928 |      0.8541 |                0.8541 |        0.1882 |                   0.02974 |                  0.06625 |                        0.4115 |                                      0.4359 |
| Austin       |                0.06846 |                      0.03517 | access_based            |       207 |   38249 |      0.8926 |                0.8926 |        0.2071 |                   0.0281  |                  0.06099 |                        0.399  |                                      0.4186 |
| Dallas       |                0.06395 |                      0.03123 | access_based            |       366 |   87499 |      0.6532 |                0.6532 |        0.18   |                   0.02763 |                  0.0564  |                        0.3101 |                                      0.3769 |

## 结构相关性

下表只使用城市结构变量，不使用 budget、delay、eta、cost 等情景参数。样本数为 7，因此只能作为结构假设生成，不能直接宣称 universal law。

| feature_family   | feature                | target               |   n |   spearman |   pearson |
|:-----------------|:-----------------------|:---------------------|----:|-----------:|----------:|
| rainfall         | max_peak_extra_deficit | recoverable_fraction |   7 |     0.9286 |    0.5131 |
| scale            | n_units                | recoverable_fraction |   7 |     0.8571 |    0.8811 |
| scale            | q_nnz                  | recoverable_fraction |   7 |     0.8571 |    0.8908 |
| scale            | od_rows                | recoverable_fraction |   7 |     0.8571 |    0.8908 |
| scale            | origin_zone_count      | recoverable_fraction |   7 |     0.8571 |    0.8813 |
| scale            | log_n_units            | recoverable_fraction |   7 |     0.8571 |    0.8778 |
| scale            | log_od_rows            | recoverable_fraction |   7 |     0.8571 |    0.8823 |
| dependence       | q_density              | recoverable_fraction |   7 |    -0.7857 |   -0.7849 |
| dependence       | od_density_observed    | recoverable_fraction |   7 |    -0.7857 |   -0.7851 |
| dependence       | od_sparsity            | recoverable_fraction |   7 |     0.7857 |    0.7849 |

## Decision Leverage 的结构相关性

| feature_family   | feature                | target                     |   n |   spearman |   pearson |
|:-----------------|:-----------------------|:---------------------------|----:|-----------:|----------:|
| rainfall         | max_peak_extra_deficit | decision_leverage_fraction |   7 |     0.9286 |    0.5581 |
| scale            | n_units                | decision_leverage_fraction |   7 |     0.8571 |    0.8858 |
| scale            | q_nnz                  | decision_leverage_fraction |   7 |     0.8571 |    0.8923 |
| scale            | od_rows                | decision_leverage_fraction |   7 |     0.8571 |    0.8923 |
| scale            | origin_zone_count      | decision_leverage_fraction |   7 |     0.8571 |    0.8861 |
| scale            | log_n_units            | decision_leverage_fraction |   7 |     0.8571 |    0.9069 |
| scale            | log_od_rows            | decision_leverage_fraction |   7 |     0.8571 |    0.9102 |
| dependence       | q_density              | decision_leverage_fraction |   7 |    -0.7857 |   -0.8256 |
| dependence       | od_density_observed    | decision_leverage_fraction |   7 |    -0.7857 |   -0.8257 |
| dependence       | od_sparsity            | decision_leverage_fraction |   7 |     0.7857 |    0.8256 |

## 初步结构解释

1. 在 full-zone LP 下，recoverability 最高的是 New York、Chicago、Houston。它们不是因为被选入 top-35，而是在全 OD zone 下仍然表现为更高可恢复比例。
2. recoverability 与城市规模变量呈正相关，例如 n_units、OD rows、Q nonzeros 的 Spearman 约为 0.86；与 OD density/Q density 呈负相关，Spearman 约为 -0.79。一个合理解释是：更大、更稀疏的功能依赖网络中，优化有更多空间寻找高杠杆恢复位置；高度稠密的小网络中，损失传播更均匀，边际定位价值较低。
3. 降雨事件冲击也很重要：max_peak_extra_deficit 与 recoverability 的 Spearman 约为 0.93。这说明可恢复性不是只由网络规模决定，灾害冲击在速度系统里是否形成清晰峰值，也会影响优化可挽回的损失空间。
4. speed-deficit severity 与 recoverability 正相关，p90_deficit 的 Spearman 约为 0.71。这与直觉一致：没有可观测损失，就没有太多 counterfactual recovery 空间。
5. TMC 损失集中度在当前样本里与 recoverability 呈负相关。这个方向需要谨慎解释：它可能意味着损失过度集中时，可恢复空间受部署上限限制；也可能只是小样本城市差异造成。

## 对论文的含义

更合适的 law 方向不应是“预算越大越好”，而应是：在相同恢复制度下，城市功能依赖结构、扰动峰值强度、拥堵暴露和损失集中度共同决定 recoverable fraction 与 optimization decision leverage。当前结果支持把后续学习任务改写为 city-structure law extraction，而不是 scenario-response curve fitting。

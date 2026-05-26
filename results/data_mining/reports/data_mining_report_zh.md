# Data Mining Report: Recoverable Urban Resilience

## 核心结论

当前数据覆盖 11 个美国城市，其中 11 个城市有需求/网络数据，7 个城市有大规模速度观测，3 个城市已有初步 resilience index。
从数据本身看，平均 idea-data support score 为 2.89/5，counterfactual model need 为 4.80/5。

这说明：数据足以支撑论文的经验基础，也足以证明城市之间存在扰动强度、恢复轨迹、网络依赖和暴露结构差异；但“有多少损失可通过管理干预恢复”这一核心结论不能仅凭描述性 data mining 得到，必须进入优化/反事实模型。

## 最匹配的城市样本

| city | overall_data_support_score_0_5 | counterfactual_model_need_0_5 | interpretation |
| --- | --- | --- | --- |
| Chicago | 4.8 | 4.25 | strong empirical basis, counterfactual model still essential |
| New York | 4.39 | 4.25 | strong empirical basis, counterfactual model still essential |
| Houston | 3.63 | 4.25 | moderate empirical basis, model needed for recoverability |
| San Antonio | 3.42 | 5 | moderate empirical basis, model needed for recoverability |
| Austin | 3.35 | 5 | moderate empirical basis, model needed for recoverability |

## 降雨-速度扰动信号

| city | overlap_hours | max_lag_corr | mean_event_deficit_impact | median_event_recovery_hours |
| --- | --- | --- | --- | --- |
| San Antonio | 744 | 0.2135 | 0.01602 | 3 |
| Austin | 744 | 0.149 | 0.01778 | 3 |
| New York | 744 | 0.1407 | 0.02715 | 2.5 |
| Chicago | 744 | 0.1042 | 0.07176 | 4 |
| Philadelphia | 744 | 0.09555 | 0.01295 | 5 |

## 速度功能损失信号

| city | scanned_valid_rows | sampled_distribution_rows | sampled_unique_tmc | p90_deficit | severe_deficit_share_20pct | hourly_observations |
| --- | --- | --- | --- | --- | --- | --- |
| Chicago | 5.631e+07 | 5e+05 | 1.27e+04 | 0.5333 | 0.4829 | 744 |
| New York | 8.565e+07 | 5e+05 | 1.92e+04 | 0.3014 | 0.1559 | 744 |
| Austin | 1.873e+07 | 5e+05 | 4200 | 0.2071 | 0.1022 | 744 |
| Philadelphia | 5.73e+07 | 5e+05 | 1.287e+04 | 0.2016 | 0.09765 | 744 |
| Houston | 4.489e+07 | 5e+05 | 1.048e+04 | 0.1924 | 0.06711 | 717 |

## 空间集中度与潜在决策杠杆

| city | tmc_count | top_1pct_tmc_deficit_share | top_5pct_tmc_deficit_share | top_10pct_tmc_deficit_share | tmc_deficit_gini | high_deficit_tmc_share_mean_gt_0_2 |
| --- | --- | --- | --- | --- | --- | --- |
| San Antonio | 7490 | 0.06996 | 0.2544 | 0.4115 | 0.5681 | 0.02897 |
| Austin | 4200 | 0.06481 | 0.2432 | 0.399 | 0.5613 | 0.05643 |
| Philadelphia | 1.287e+04 | 0.06541 | 0.2418 | 0.3914 | 0.5499 | 0.04756 |
| Dallas | 1.745e+04 | 0.04755 | 0.184 | 0.3101 | 0.4451 | 0.001318 |
| New York | 1.92e+04 | 0.05121 | 0.1869 | 0.307 | 0.4293 | 0.07072 |

这组指标用于回答一个更接近 recoverability 的前置问题：loss 是否集中到少量可定位的 links/TMCs 上。如果前 10% TMC 承担了远高于 10% 的 deficit burden，则 targeted intervention 至少在空间结构上有潜在杠杆；如果 loss 极度分散，则优化模型可能也只能得到较低 recoverable fraction。

## 需求和网络依赖结构

| city | od_rows | origin_zone_count | destination_zone_count | top10_destination_volume_share | destination_volume_hhi | congested_volume_share_speed_ratio_lt_0_8 |
| --- | --- | --- | --- | --- | --- | --- |
| New York | 707640 | 1935 | 1940 | 0.03371 | 0.0009493 | 0.7678 |
| Los Angeles | 135993 | 532 | 532 | 0.06639 | 0.00288 | 0.3314 |
| Chicago | 337138 | 882 | 882 | 0.05778 | 0.001885 | 0.3826 |
| Houston | 99541 | 393 | 393 | 0.09432 | 0.003894 | 0.407 |
| Phoenix | 86133 | 357 | 357 | 0.09349 | 0.003987 | 0.3617 |
| Philadelphia | 152418 | 641 | 642 | 0.06551 | 0.002328 | 0.5076 |
| San Antonio | 78928 | 304 | 304 | 0.1052 | 0.004795 | 0.4359 |
| San Diego | 57911 | 295 | 295 | 0.1462 | 0.006204 | 0.4282 |
| Dallas | 87499 | 366 | 366 | 0.08063 | 0.00378 | 0.3769 |
| San Jose | 62046 | 289 | 289 | 0.1117 | 0.005245 | 0.499 |
| Austin | 38249 | 207 | 207 | 0.1352 | 0.006882 | 0.4186 |

## 已有 resilience index 与网络暴露的对齐

| city | matched_link_rows | match_rate | loss_auc_vs_volume_spearman | loss_auc_vs_DOC_spearman | top10pct_volume_share_of_loss_auc | high_vs_low_volume_loss_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| New York | 3.15e+04 | 1 | 0.07997 | 0.0749 | 0.1495 | 1.3 |
| Chicago | 4.785e+04 | 1 | 0.2143 | 0.1422 | 0.1308 | 1.663 |
| Houston | 1.813e+04 | 1 | 0.009188 | 0.02283 | 0.08672 | 0.7873 |

这一步检查已有速度恢复指标是否落在高流量、高拥堵暴露的 links 上。它不能证明干预有效，但可以判断 observed disruption 是否具有功能暴露意义，而不是只发生在低重要性的边缘路段。

## 对论文 idea 的含义

1. 数据支持 `b_t`：速度相对历史速度的 deficit 可以作为 mobility-functional deficit 的直接 proxy。
2. 数据部分支持 `A_t`：速度数据已经按全月时间维度聚合，可以估计内生恢复趋势；分布分位数使用固定随机样本以控制内存。
3. 数据部分支持 `Q_t`：OD demand、route/link performance 和 network links 可以构造 baseline functional dependence，但服务重要性 `S_j` 仍需要 POI、就业、医疗、零售或人口暴露数据增强。
4. 数据支持 `h_t` 的一个版本：rainfall 可以作为外部扰动，但如果论文想泛化到 flood/storm/infrastructure failure，需要补充事件或 hazard 数据。
5. 数据不直接支持 `eta^k`、预算、response delay 和 intervention effectiveness：这些必须通过反事实情景、敏感性分析或真实应急响应记录建模。

## 当前边界判断

继续做描述性 data mining 的边际收益已经开始下降。剩余关键问题不是“数据里有没有恢复现象”，而是“同一个扰动下，如果资源被不同地分配，损失能少多少”。这个问题必须由 recoverable-resilience optimization model 来回答。

# Budget-Leverage Phase Analysis V13

## 这一版回答什么问题

High-level idea 里有一个预期：decision leverage 可能在中等预算最高。V13 用已有 low/base/high budget policy replay 与 action-value proxy 直接检验这个命题，并区分两类量：绝对决策收益（law 比 naive 多拿到多少 value）和单位预算/相对收益（每单位资源或相对 naive 的优势）。

## 主要结论

- law-vs-random absolute proxy leverage peak: high; interior peak supported = False
- law/random relative ratio peak: low; monotone decreasing = True
- residual-vs-static replay leverage peak: high; interior peak supported = False

当前三点预算扫描不支持“中等预算绝对 decision leverage 最高”。更准确的结论是：预算越高，law 相对 naive 的绝对额外收益继续增加；但相对优势和单位预算杠杆递减。这说明智能分配的收益存在 diminishing leverage，而不是简单的 interior nonmonotonic peak。

## Budget Summary

|   budget_scale | budget_label   |   n_events |   mean_law_value_proxy |   mean_law_value_per_cost |   mean_residual_minus_static_fraction_of_base_lp_gain |   mean_residual_minus_static_recoverable_fraction |   mean_residual_minus_static_per_cost |   mean_residual_relative_to_static_gain |   mean_proxy_leverage_vs_random_positive |   median_proxy_leverage_vs_random_positive |   mean_proxy_ratio_vs_random_positive |   mean_proxy_leverage_per_cost_vs_random_positive |   mean_proxy_leverage_vs_exposure_only |   median_proxy_leverage_vs_exposure_only |   mean_proxy_ratio_vs_exposure_only |   mean_proxy_leverage_per_cost_vs_exposure_only |   mean_proxy_leverage_vs_deficit_only |   median_proxy_leverage_vs_deficit_only |   mean_proxy_ratio_vs_deficit_only |   mean_proxy_leverage_per_cost_vs_deficit_only |   mean_proxy_leverage_vs_structure_only |   median_proxy_leverage_vs_structure_only |   mean_proxy_ratio_vs_structure_only |   mean_proxy_leverage_per_cost_vs_structure_only |
|---------------:|:---------------|-----------:|-----------------------:|--------------------------:|------------------------------------------------------:|--------------------------------------------------:|--------------------------------------:|----------------------------------------:|-----------------------------------------:|-------------------------------------------:|--------------------------------------:|--------------------------------------------------:|---------------------------------------:|-----------------------------------------:|------------------------------------:|------------------------------------------------:|--------------------------------------:|----------------------------------------:|-----------------------------------:|-----------------------------------------------:|----------------------------------------:|------------------------------------------:|-------------------------------------:|-------------------------------------------------:|
|            0.5 | low            |        105 |                0.08617 |                    0.134  |                                               0.07521 |                                           0.01182 |                               0.03109 |                                   1.196 |                                  0.07503 |                                     0.0714 |                                 7.873 |                                            0.118  |                                0.01924 |                                  0.01775 |                               1.293 |                                         0.03873 |                               0.02188 |                                 0.02025 |                              1.345 |                                        0.04137 |                                 0.05668 |                                   0.05099 |                                2.954 |                                          0.09216 |
|            1   | base           |        105 |                0.1494  |                    0.1186 |                                               0.1432  |                                           0.02211 |                               0.02679 |                                   1.22  |                                  0.1272  |                                     0.1198 |                                 6.811 |                                            0.1029 |                                0.03112 |                                  0.02923 |                               1.266 |                                         0.03072 |                               0.03597 |                                 0.03411 |                              1.319 |                                        0.03544 |                                 0.09561 |                                   0.08708 |                                2.776 |                                          0.07863 |
|            2   | high           |        105 |                0.2537  |                    0.1029 |                                               0.2623  |                                           0.03987 |                               0.02431 |                                   1.25  |                                  0.2094  |                                     0.1974 |                                 5.791 |                                            0.0872 |                                0.05029 |                                  0.04865 |                               1.247 |                                         0.02441 |                               0.05816 |                                 0.056   |                              1.296 |                                        0.02795 |                                 0.1539  |                                   0.1425  |                                2.528 |                                          0.06518 |

## Phase Tests

| metric                                              |     low |    base |    high | peak_budget   | interior_peak_supported   | monotone_increasing   | monotone_decreasing   |
|:----------------------------------------------------|--------:|--------:|--------:|:--------------|:--------------------------|:----------------------|:----------------------|
| mean_proxy_leverage_vs_random_positive              | 0.07503 | 0.1272  | 0.2094  | high          | False                     | True                  | False                 |
| mean_proxy_ratio_vs_random_positive                 | 7.873   | 6.811   | 5.791   | low           | False                     | False                 | True                  |
| mean_proxy_leverage_vs_exposure_only                | 0.01924 | 0.03112 | 0.05029 | high          | False                     | True                  | False                 |
| mean_proxy_ratio_vs_exposure_only                   | 1.293   | 1.266   | 1.247   | low           | False                     | False                 | True                  |
| mean_proxy_leverage_per_cost_vs_random_positive     | 0.118   | 0.1029  | 0.0872  | low           | False                     | False                 | True                  |
| mean_residual_minus_static_fraction_of_base_lp_gain | 0.07521 | 0.1432  | 0.2623  | high          | False                     | True                  | False                 |
| mean_residual_minus_static_recoverable_fraction     | 0.01182 | 0.02211 | 0.03987 | high          | False                     | True                  | False                 |
| mean_residual_minus_static_per_cost                 | 0.03109 | 0.02679 | 0.02431 | low           | False                     | False                 | True                  |

## Base-Budget City Ranking

| city         |   budget_scale | budget_label   |   n_events |   mean_proxy_leverage_vs_random |   mean_proxy_ratio_vs_random |   mean_proxy_leverage_vs_exposure |   mean_residual_minus_static_fraction_of_base_lp_gain |   mean_residual_minus_static_recoverable_fraction | absolute_proxy_random_peak_budget   | relative_proxy_random_peak_budget   | residual_static_peak_budget   |
|:-------------|---------------:|:---------------|-----------:|--------------------------------:|-----------------------------:|----------------------------------:|------------------------------------------------------:|--------------------------------------------------:|:------------------------------------|:------------------------------------|:------------------------------|
| Chicago      |              1 | base           |         13 |                         0.1509  |                        7.392 |                           0.02776 |                                              0.2068   |                                         0.03519   | high                                | low                                 | high                          |
| Philadelphia |              1 | base           |         20 |                         0.1564  |                        7.406 |                           0.04351 |                                              0.1971   |                                         0.03289   | high                                | low                                 | high                          |
| San Antonio  |              1 | base           |         19 |                         0.1267  |                        6.182 |                           0.03551 |                                              0.1843   |                                         0.02562   | high                                | low                                 | high                          |
| New York     |              1 | base           |         24 |                         0.1329  |                        8.161 |                           0.02679 |                                              0.1233   |                                         0.01924   | high                                | low                                 | high                          |
| Austin       |              1 | base           |         15 |                         0.1141  |                        5.503 |                           0.03262 |                                              0.112    |                                         0.0152    | high                                | low                                 | high                          |
| Houston      |              1 | base           |          4 |                         0.09412 |                        5.882 |                           0.01792 |                                              0.06222  |                                         0.006982  | high                                | low                                 | high                          |
| Dallas       |              1 | base           |         10 |                         0.05812 |                        5.158 |                           0.01576 |                                              0.001584 |                                         0.0001468 | high                                | low                                 | high                          |

## Correlations at Base Budget

|   budget_scale | target                                         | feature                              |   spearman |
|---------------:|:-----------------------------------------------|:-------------------------------------|-----------:|
|              1 | proxy_leverage_vs_exposure_only                | recoverable_fraction                 |    0.8058  |
|              1 | proxy_leverage_vs_exposure_only                | decision_criticality_score           |    0.4547  |
|              1 | proxy_leverage_vs_exposure_only                | marginal_value_gini                  |    0.1078  |
|              1 | proxy_leverage_vs_exposure_only                | top_5pct_value_share                 |   -0.04822 |
|              1 | proxy_leverage_vs_exposure_only                | event_total_precip                   |   -0.2142  |
|              1 | proxy_leverage_vs_exposure_only                | event_peak_positive_abnormal_deficit |   -0.5854  |
|              1 | proxy_leverage_vs_exposure_only                | baseline_objective                   |   -0.758   |
|              1 | proxy_leverage_vs_random_positive              | recoverable_fraction                 |    0.9535  |
|              1 | proxy_leverage_vs_random_positive              | decision_criticality_score           |    0.7795  |
|              1 | proxy_leverage_vs_random_positive              | marginal_value_gini                  |    0.4575  |
|              1 | proxy_leverage_vs_random_positive              | top_5pct_value_share                 |    0.321   |
|              1 | proxy_leverage_vs_random_positive              | event_total_precip                   |   -0.1767  |
|              1 | proxy_leverage_vs_random_positive              | event_peak_positive_abnormal_deficit |   -0.5754  |
|              1 | proxy_leverage_vs_random_positive              | baseline_objective                   |   -0.6809  |
|              1 | proxy_ratio_vs_random_positive                 | marginal_value_gini                  |    0.8318  |
|              1 | proxy_ratio_vs_random_positive                 | decision_criticality_score           |    0.7947  |
|              1 | proxy_ratio_vs_random_positive                 | top_5pct_value_share                 |    0.7516  |
|              1 | proxy_ratio_vs_random_positive                 | recoverable_fraction                 |    0.4233  |
|              1 | proxy_ratio_vs_random_positive                 | event_total_precip                   |   -0.1202  |
|              1 | proxy_ratio_vs_random_positive                 | baseline_objective                   |   -0.5055  |
|              1 | proxy_ratio_vs_random_positive                 | event_peak_positive_abnormal_deficit |   -0.509   |
|              1 | residual_minus_static_fraction_of_base_lp_gain | recoverable_fraction                 |    0.6452  |
|              1 | residual_minus_static_fraction_of_base_lp_gain | decision_criticality_score           |    0.4175  |
|              1 | residual_minus_static_fraction_of_base_lp_gain | marginal_value_gini                  |    0.2538  |
|              1 | residual_minus_static_fraction_of_base_lp_gain | top_5pct_value_share                 |    0.1412  |
|              1 | residual_minus_static_fraction_of_base_lp_gain | event_total_precip                   |   -0.04222 |
|              1 | residual_minus_static_fraction_of_base_lp_gain | event_peak_positive_abnormal_deficit |   -0.5444  |
|              1 | residual_minus_static_fraction_of_base_lp_gain | baseline_objective                   |   -0.6785  |
|              1 | residual_minus_static_recoverable_fraction     | recoverable_fraction                 |    0.7241  |
|              1 | residual_minus_static_recoverable_fraction     | decision_criticality_score           |    0.4787  |
|              1 | residual_minus_static_recoverable_fraction     | marginal_value_gini                  |    0.2715  |
|              1 | residual_minus_static_recoverable_fraction     | top_5pct_value_share                 |    0.1536  |
|              1 | residual_minus_static_recoverable_fraction     | event_total_precip                   |   -0.06139 |
|              1 | residual_minus_static_recoverable_fraction     | event_peak_positive_abnormal_deficit |   -0.5578  |
|              1 | residual_minus_static_recoverable_fraction     | baseline_objective                   |   -0.6897  |

## 科学含义

这个结果让论文叙事更精确：managed recoverability 的预算规律不是单纯“预算越多越好”，也不是当前样本中已经证明的“中等预算最高”。从现有数据看，绝对可恢复收益随预算增加，但每单位预算产生的额外决策杠杆下降；因此预算 law 应表述为 scale-dependent diminishing leverage。城市差异也很强，Dallas 的 residual-vs-static gap 接近零，而 Chicago、Philadelphia、San Antonio 的 gap 明显更大，说明预算杠杆仍由城市结构和 action-value top tail 决定。

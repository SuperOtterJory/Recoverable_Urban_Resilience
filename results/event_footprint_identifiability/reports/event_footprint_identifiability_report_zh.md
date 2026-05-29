# Event Spatial Footprint Identifiability V32

## Question

This audit asks whether the current event-level top-tail law is identifying event-specific spatial footprints, or whether it is mostly identifying a city-level OD vulnerability template.

## Main Findings

- events/cities audited: 105 / 7
- cities with zero within-city variation in top-5% value share: 5 (58.1% of events)
- cities with event-specific top-tail variation: New York; Philadelphia
- cities with zero within-city variation in marginal-value Gini: 5 (58.1% of events)
- mean within-city range of optimizer-selected value share: 0.1636
- top-10 greedy unit mean pairwise Jaccard across cities: 0.9685

Interpretation: the current top-tail concentration is strongly constrained by the spatial template used in calibration. It remains useful as a city-structure/top-tail law, but not yet as a fully resolved within-city event-footprint law.

## City Summary

| city         |   n_events |   top_5pct_value_share_unique |   top_5pct_value_share_range |   top_5pct_value_share_std | top_5pct_value_share_zero_variance   |   marginal_value_gini_unique |   marginal_value_gini_range |   marginal_value_gini_std | marginal_value_gini_zero_variance   |   optimizer_selected_value_share_unique |   optimizer_selected_value_share_range |   optimizer_selected_value_share_std | optimizer_selected_value_share_zero_variance   |   recoverable_fraction_unique |   recoverable_fraction_range |   recoverable_fraction_std | recoverable_fraction_zero_variance   |   baseline_objective_unique |   baseline_objective_range |   baseline_objective_std | baseline_objective_zero_variance   |
|:-------------|-----------:|------------------------------:|-----------------------------:|---------------------------:|:-------------------------------------|-----------------------------:|----------------------------:|--------------------------:|:------------------------------------|----------------------------------------:|---------------------------------------:|-------------------------------------:|:-----------------------------------------------|------------------------------:|-----------------------------:|---------------------------:|:-------------------------------------|----------------------------:|---------------------------:|-------------------------:|:-----------------------------------|
| Austin       |         15 |                             1 |                       0      |                  0         | True                                 |                            1 |                      0      |                 5.551e-17 | True                                |                                      15 |                              0.1462    |                            0.03565   | False                                          |                            15 |                      0.07536 |                    0.02184 | False                                |                          15 |                     0.2029 |                  0.05622 | False                              |
| Chicago      |         13 |                             1 |                       0      |                  0         | True                                 |                            1 |                      0      |                 5.551e-17 | True                                |                                      13 |                              0.3334    |                            0.07567   | False                                          |                            13 |                      0.1499  |                    0.04179 | False                                |                          13 |                     0.3438 |                  0.1056  | False                              |
| Dallas       |         10 |                             1 |                       0      |                  2.776e-17 | True                                 |                            1 |                      0      |                 1.11e-16  | True                                |                                       4 |                              0.0008201 |                            0.0003519 | False                                          |                            10 |                      0.03878 |                    0.01251 | False                                |                          10 |                     0.5238 |                  0.1593  | False                              |
| Houston      |          4 |                             1 |                       0      |                  0         | True                                 |                            1 |                      0      |                 0         | True                                |                                       4 |                              0.01884   |                            0.007214  | False                                          |                             4 |                      0.02498 |                    0.01002 | False                                |                           4 |                     0.3676 |                  0.142   | False                              |
| New York     |         24 |                             7 |                       0.113  |                  0.02416   | False                                |                            7 |                      0.2009 |                 0.04576   | False                               |                                      24 |                              0.1276    |                            0.0352    | False                                          |                            24 |                      0.1577  |                    0.03005 | False                                |                          24 |                     0.7191 |                  0.2136  | False                              |
| Philadelphia |         20 |                             2 |                       0.1232 |                  0.02685   | False                                |                            2 |                      0.225  |                 0.04903   | False                               |                                      20 |                              0.1571    |                            0.0481    | False                                          |                            20 |                      0.06982 |                    0.02021 | False                                |                          20 |                     0.1567 |                  0.0406  | False                              |
| San Antonio  |         19 |                             1 |                       0      |                  5.551e-17 | True                                 |                            1 |                      0      |                 1.11e-16  | True                                |                                      19 |                              0.3613    |                            0.0737    | False                                          |                            19 |                      0.04214 |                    0.01095 | False                                |                          19 |                     0.1683 |                  0.04832 | False                              |

## Key Metric Variation

| city         | metric                         |   n_events |   n_unique_rounded_1e10 |     min |     max |     range |    mean |       std |    cv_abs | zero_within_city_variance   |
|:-------------|:-------------------------------|-----------:|------------------------:|--------:|--------:|----------:|--------:|----------:|----------:|:----------------------------|
| Austin       | marginal_value_gini            |         15 |                       1 | 0.4289  | 0.4289  | 0         | 0.4289  | 5.551e-17 | 1.294e-16 | True                        |
| Chicago      | marginal_value_gini            |         13 |                       1 | 0.4585  | 0.4585  | 0         | 0.4585  | 5.551e-17 | 1.211e-16 | True                        |
| Dallas       | marginal_value_gini            |         10 |                       1 | 0.4124  | 0.4124  | 0         | 0.4124  | 1.11e-16  | 2.692e-16 | True                        |
| Houston      | marginal_value_gini            |          4 |                       1 | 0.4274  | 0.4274  | 0         | 0.4274  | 0         | 0         | True                        |
| New York     | marginal_value_gini            |         24 |                       7 | 0.4738  | 0.6747  | 0.2009    | 0.4927  | 0.04576   | 0.09288   | False                       |
| Philadelphia | marginal_value_gini            |         20 |                       2 | 0.4373  | 0.6623  | 0.225     | 0.4485  | 0.04903   | 0.1093    | False                       |
| San Antonio  | marginal_value_gini            |         19 |                       1 | 0.4295  | 0.4295  | 0         | 0.4295  | 1.11e-16  | 2.585e-16 | True                        |
| Austin       | optimizer_selected_value_share |         15 |                      15 | 0.09578 | 0.242   | 0.1462    | 0.1795  | 0.03565   | 0.1986    | False                       |
| Chicago      | optimizer_selected_value_share |         13 |                      13 | 0.0678  | 0.4012  | 0.3334    | 0.1612  | 0.07567   | 0.4695    | False                       |
| Dallas       | optimizer_selected_value_share |         10 |                       4 | 0.05999 | 0.06081 | 0.0008201 | 0.06022 | 0.0003519 | 0.005843  | False                       |
| Houston      | optimizer_selected_value_share |          4 |                       4 | 0.09951 | 0.1184  | 0.01884   | 0.1083  | 0.007214  | 0.06659   | False                       |
| New York     | optimizer_selected_value_share |         24 |                      24 | 0.04696 | 0.1745  | 0.1276    | 0.1105  | 0.0352    | 0.3186    | False                       |
| Philadelphia | optimizer_selected_value_share |         20 |                      20 | 0.08078 | 0.2378  | 0.1571    | 0.1618  | 0.0481    | 0.2972    | False                       |
| San Antonio  | optimizer_selected_value_share |         19 |                      19 | 0.08377 | 0.4451  | 0.3613    | 0.1881  | 0.0737    | 0.3918    | False                       |
| Austin       | top_5pct_value_share           |         15 |                       1 | 0.1494  | 0.1494  | 0         | 0.1494  | 0         | 0         | True                        |
| Chicago      | top_5pct_value_share           |         13 |                       1 | 0.1722  | 0.1722  | 0         | 0.1722  | 0         | 0         | True                        |
| Dallas       | top_5pct_value_share           |         10 |                       1 | 0.142   | 0.142   | 0         | 0.142   | 2.776e-17 | 1.955e-16 | True                        |
| Houston      | top_5pct_value_share           |          4 |                       1 | 0.1544  | 0.1544  | 0         | 0.1544  | 0         | 0         | True                        |
| New York     | top_5pct_value_share           |         24 |                       7 | 0.189   | 0.302   | 0.113     | 0.1981  | 0.02416   | 0.1219    | False                       |
| Philadelphia | top_5pct_value_share           |         20 |                       2 | 0.1535  | 0.2768  | 0.1232    | 0.1597  | 0.02685   | 0.1681    | False                       |
| San Antonio  | top_5pct_value_share           |         19 |                       1 | 0.148   | 0.148   | 0         | 0.148   | 5.551e-17 | 3.752e-16 | True                        |

## Top Unit Stability

| city         |   top_k_units |   n_event_pairs |   mean_jaccard |   median_jaccard |   min_jaccard |   max_jaccard |
|:-------------|--------------:|----------------:|---------------:|-----------------:|--------------:|--------------:|
| Austin       |             5 |             105 |         1      |           1      |        1      |             1 |
| Chicago      |             5 |              78 |         1      |           1      |        1      |             1 |
| Dallas       |             5 |              45 |         1      |           1      |        1      |             1 |
| Houston      |             5 |               6 |         1      |           1      |        1      |             1 |
| New York     |             5 |             276 |         1      |           1      |        1      |             1 |
| Philadelphia |             5 |             190 |         1      |           1      |        1      |             1 |
| San Antonio  |             5 |             171 |         0.9649 |           1      |        0.6667 |             1 |
| Austin       |            10 |             105 |         0.933  |           1      |        0.6667 |             1 |
| Chicago      |            10 |              78 |         0.9301 |           1      |        0.8182 |             1 |
| Dallas       |            10 |              45 |         1      |           1      |        1      |             1 |
| Houston      |            10 |               6 |         1      |           1      |        1      |             1 |
| New York     |            10 |             276 |         1      |           1      |        1      |             1 |
| Philadelphia |            10 |             190 |         0.9818 |           1      |        0.8182 |             1 |
| San Antonio  |            10 |             171 |         0.9344 |           1      |        0.7    |             1 |
| Austin       |            20 |             105 |         0.8362 |           1      |        0.4348 |             1 |
| Chicago      |            20 |              78 |         0.9524 |           0.9524 |        0.9048 |             1 |
| Dallas       |            20 |              45 |         1      |           1      |        1      |             1 |
| Houston      |            20 |               6 |         1      |           1      |        1      |             1 |
| New York     |            20 |             276 |         1      |           1      |        1      |             1 |
| Philadelphia |            20 |             190 |         0.8712 |           0.9048 |        0.55   |             1 |
| San Antonio  |            20 |             171 |         0.8958 |           1      |        0.35   |             1 |

## Writing Implication

The paper should keep the event-level law, but phrase it carefully: decision-criticality is a top-tail law under the current OD-vulnerability spatial calibration. A stronger future claim about rainfall-footprint-specific recovery laws requires zone-level speed/rainfall footprints or a calibrated spatial footprint augmentation.

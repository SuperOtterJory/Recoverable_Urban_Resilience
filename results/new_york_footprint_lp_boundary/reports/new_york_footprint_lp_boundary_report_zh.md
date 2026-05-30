# New York Footprint LP Boundary Audit

## 结论

V43 不声称已经得到 New York-scale footprint-aware LP 最优解。它证明的是：New York 的 event-footprint 信号存在，而且强度不低；当前缺口主要是全 LP 闭合的计算边界。在选出的 5 个 New York footprint-sensitive events 中，hybrid finite top-5% footprint mass 平均从 0.0527 升到 0.4042，平均增量 +0.3515。

最大 New York case 有 1,940 zones、约 475,300 个 LP 变量、550,973 个约束、9,199,398 个 access-loss 非零项。相对 V42 已闭合样本最大值，变量规模为 2.20x，access-loss 非零项为 2.10x。

此前 hybrid-footprint full LP 已尝试的 New York event 为 117；当前记录中 1/1 个已尝试 New York row 未闭合。因此论文中应写为 computational boundary，而不是 footprint law 失败。

## Selected New York Events

|   event_id | event_start         |   n_units |   delta_finite_top5pct_units_footprint_mass |   base_finite_top5pct_units_footprint_mass |   hybrid_finite_top5pct_units_footprint_mass | prior_hybrid_lp_status   | prior_hybrid_lp_error                                         |
|-----------:|:--------------------|----------:|--------------------------------------------:|-------------------------------------------:|---------------------------------------------:|:-------------------------|:--------------------------------------------------------------|
|        117 | 2019-06-20 07:00:00 |      1940 |                                      0.3691 |                                    0.04938 |                                       0.4185 | ERROR                    | Gurobi did not return a feasible solution. Status: TIME_LIMIT |
|        115 | 2019-06-20 01:00:00 |      1940 |                                      0.3601 |                                    0.05135 |                                       0.4115 | not_attempted            |                                                               |
|        116 | 2019-06-20 04:00:00 |      1940 |                                      0.3593 |                                    0.05157 |                                       0.4108 | not_attempted            |                                                               |
|        112 | 2019-06-19 07:00:00 |      1940 |                                      0.3399 |                                    0.05567 |                                       0.3955 | not_attempted            |                                                               |
|        113 | 2019-06-19 10:00:00 |      1940 |                                      0.329  |                                    0.05571 |                                       0.3847 | not_attempted            |                                                               |

## Size Comparison

| scope                 | city         |   event_id |   n_units |   q_nnz |   estimated_total_variables |   estimated_total_constraints |   estimated_access_nonzero_terms | max_runtime_seconds   |
|:----------------------|:-------------|-----------:|----------:|--------:|----------------------------:|------------------------------:|---------------------------------:|:----------------------|
| V42_broader_solved    | Austin       |        478 |       207 |   38249 |                       50715 |                         58801 |                           497237 | 5.838                 |
| V42_broader_solved    | Austin       |        490 |       207 |   38249 |                       50715 |                         58801 |                           497237 | 3.41                  |
| V42_broader_solved    | Chicago      |        158 |       882 |  337138 |                      216090 |                        250501 |                          4382794 | 28.95                 |
| V42_broader_solved    | Chicago      |        159 |       882 |  337138 |                      216090 |                        250501 |                          4382794 | 96.88                 |
| V42_broader_solved    | Dallas       |         58 |       366 |   87499 |                       89670 |                        103957 |                          1137487 | 3.963                 |
| V42_broader_solved    | Dallas       |         62 |       366 |   87499 |                       89670 |                        103957 |                          1137487 | 3.852                 |
| V42_broader_solved    | Houston      |         60 |       393 |   99541 |                       96285 |                        111625 |                          1294033 | 4.642                 |
| V42_broader_solved    | Houston      |         62 |       393 |   99541 |                       96285 |                        111625 |                          1294033 | 4.833                 |
| V42_broader_solved    | Philadelphia |        644 |       642 |  152419 |                      157290 |                        182341 |                          1981447 | 32.06                 |
| V42_broader_solved    | Philadelphia |        658 |       642 |  152419 |                      157290 |                        182341 |                          1981447 | 3.377                 |
| V42_broader_solved    | San Antonio  |        494 |       304 |   78928 |                       74480 |                         86349 |                          1026064 | 23.93                 |
| V42_broader_solved    | San Antonio  |        497 |       304 |   78928 |                       74480 |                         86349 |                          1026064 | 28.21                 |
| V43_New_York_boundary | New York     |        112 |      1940 |  707646 |                      475300 |                        550973 |                          9199398 |                       |
| V43_New_York_boundary | New York     |        113 |      1940 |  707646 |                      475300 |                        550973 |                          9199398 |                       |
| V43_New_York_boundary | New York     |        115 |      1940 |  707646 |                      475300 |                        550973 |                          9199398 |                       |
| V43_New_York_boundary | New York     |        116 |      1940 |  707646 |                      475300 |                        550973 |                          9199398 |                       |
| V43_New_York_boundary | New York     |        117 |      1940 |  707646 |                      475300 |                        550973 |                          9199398 |                       |

## 写作含义

1. V42 已经证明非 New York full-zone 城市中 recovery--footprint frontier 存在；V43 说明未闭合的是最大城市的直接 LP 计算边界。
2. New York 的 observed footprint signal 很强，因此不能把 New York 排除解释为“没有事件空间信号”。
3. 在模型部分可以继续声称大规模 all-zone base recovery LP 已覆盖 New York；在 footprint-aware direct LP 部分则应明确 New York-scale closure 仍需 solver tuning、decomposition、warm-start basis 或 city-specific preference experiments。

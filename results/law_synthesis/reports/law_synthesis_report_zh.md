# Recoverability Law Synthesis V13

## 这版做了什么

V13 把 V5-V12 的 learning/law 证据链和新的 budget-leverage phase analysis 合并成论文可用的 synthesis：从 action-level marginal value，到 finite-budget residual law，再到 event-level top-tail decision-criticality，并进一步给出公式复杂度、跨城市泛化、feature-group ablation 和预算相图证据。所有数字都从已有结果表重新读取生成。

## 三条当前可写入论文的 law

1. **Small-signal activated recovery law**：第一小段资源的边际价值主要由 active future horizon、OD exposure 或 destination importance、intervention efficiency per cost 共同决定，并按 passive event loss 归一化。它回答“第一单位资源投向哪里最值”。

2. **Residual finite-budget allocation law**：完整预算下，价值必须写成 `value(segment | residual state, remaining budget, remaining time)`。每轮投放后要重新计算剩余 `b/rC/rS/ell`，用 `min(segment_effect_decay, residual_loss)` 截断后续边际收益。它回答“整组资源如何避免局部饱和”。

3. **Top-tail decision-criticality law**：事件是否 decision-critical 不只取决于 observed loss 大小，而取决于 recoverable value 是否集中在少数高价值 action 上。高 recoverable fraction 与高 top-tail concentration 共同定义了管理决策的杠杆。

## 关键指标

- action tokens: 170,137
- city-event scenarios: 105
- single-action LP validation: small-signal Spearman = 0.8540, median LP/label ratio = 1.0000
- finite-area label Spearman = 0.1146
- leave-one-city-out mean Spearman = 0.8866, top-5% capture = 0.7640
- base static greedy / LP gain = 0.8228
- base residual greedy / LP gain = 0.9736
- representative non-base static / scenario LP gain = 0.7228
- representative non-base residual / scenario LP gain = 0.9411
- perturbed optimum residual frequency-mass capture = 0.6097 versus static = 0.4659
- symbolic activated law top-5% capture = 0.1121; minimal log-law top-5% capture = 0.1026
- largest symbolic ablation drop = od_exposure_structure (0.0276 top-5 capture)
- budget phase: absolute law-vs-random leverage peaks at high; interior peak supported = False
- budget phase: residual-vs-static per-cost leverage = 0.0311 / 0.0268 / 0.0243 for low/base/high
- event mean top-5% value share = 0.1645; marginal-value Gini = 0.4494

## Evidence Ladder

| version   | evidence_step                                        | main_question                                                                                 | key_metric                                |   value | interpretation                                                                                                                                                |
|:----------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------|:------------------------------------------|--------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| V4/V5     | Single-action LP validates label                     | What is the right marginal recovery-value label?                                              | small_signal_spearman                     |  0.854  | The first-segment derivative, not finite deficit area, matches direct single-action LP probes.                                                                |
| V5        | Cross-city action-value field                        | Can normalized structural features recover the value ranking?                                 | leave_city_mean_spearman                  |  0.8866 | A simple surrogate generalizes action-value ranking across held-out cities.                                                                                   |
| V6        | Finite-budget gap                                    | Does a static first-order ranking solve the full budget problem?                              | static_fraction_of_lp_gain                |  0.8228 | Static small-signal greedy captures substantial value but leaves a finite-budget interaction gap.                                                             |
| V7        | Residual finite-budget law                           | Does re-scoring on the residual state close the base LP gap?                                  | residual_fraction_of_lp_gain              |  0.9736 | Residual replanning nearly closes the base-scenario LP optimum on average.                                                                                    |
| V8        | Budget/delay stress test                             | Is the residual law stable under resource and response perturbations?                         | delay4_residual_fraction_of_base_lp_gain  |  0.7734 | Residual replanning remains systematically above static greedy under delayed response.                                                                        |
| V9        | Scenario-specific LP closure                         | Does the law still approach the true non-base scenario optimum?                               | residual_fraction_of_scenario_lp_gain     |  0.9411 | Representative non-base LP solves show residual finite greedy close to scenario-specific optima.                                                              |
| V11       | Perturbed-optimum stability                          | Are recovery-critical actions stable under small cost/effectiveness perturbations?            | residual_perturbed_frequency_mass_capture |  0.6097 | Residual finite greedy captures more perturbed LP selection-frequency mass than static greedy, but stable action lists remain parameter-sensitive.            |
| V12       | Symbolic formula extraction and structure decoupling | Can the action-value field be compressed into a low-complexity law and stable feature groups? | activated_symbolic_top5_capture           |  0.1121 | The compact activated law sits on the formula Pareto frontier; OD exposure/structure is the largest feature-group contributor in ablation.                    |
| V13       | Budget-leverage phase analysis                       | Is decision leverage highest at intermediate budget levels?                                   | interior_budget_peak_supported            |  0      | The current low/base/high scan does not support an interior absolute-leverage peak; absolute leverage rises with budget while per-budget leverage diminishes. |

## Policy Closure

| scope                                 | comparison                                 |   n |   mean_fraction_of_lp_gain |   median_fraction_of_lp_gain | mean_improvement_over_static   |
|:--------------------------------------|:-------------------------------------------|----:|---------------------------:|-----------------------------:|:-------------------------------|
| base_all_105                          | static_small_signal_vs_base_LP             | 105 |                     0.8228 |                       0.8774 | 0                              |
| base_all_105                          | residual_finite_greedy_vs_base_LP          | 105 |                     0.9736 |                       0.9921 | 0.1508                         |
| stress_low_budget_105                 | residual_finite_greedy_vs_static_reference | 105 |                     0.5711 |                       0.5785 | 0.07521                        |
| stress_base_105                       | residual_finite_greedy_vs_static_reference | 105 |                     0.966  |                       0.9894 | 0.1432                         |
| stress_high_budget_105                | residual_finite_greedy_vs_static_reference | 105 |                     1.592  |                       1.615  | 0.2623                         |
| stress_delay_2h_105                   | residual_finite_greedy_vs_static_reference | 105 |                     0.878  |                       0.8837 | 0.122                          |
| stress_delay_4h_105                   | residual_finite_greedy_vs_static_reference | 105 |                     0.7734 |                       0.7775 | 0.09297                        |
| stress_scarce_and_late_105            | residual_finite_greedy_vs_static_reference | 105 |                     0.5222 |                       0.5299 | 0.06478                        |
| representative_nonbase_scenario_LP_23 | residual_finite_greedy_vs_scenario_LP      |  23 |                     0.9411 |                       0.9769 |                                |
| representative_nonbase_scenario_LP_23 | static_small_signal_greedy_vs_scenario_LP  |  23 |                     0.7228 |                       0.7797 |                                |

## City Closure

| city         |   n_events |   mean_static_fraction_of_lp_gain |   mean_residual_fraction_of_lp_gain |   mean_residual_gain_improvement |   mean_residual_gap_to_lp |
|:-------------|-----------:|----------------------------------:|------------------------------------:|---------------------------------:|--------------------------:|
| Chicago      |         13 |                            0.7632 |                              0.9758 |                         0.2126   |                 0.02417   |
| Philadelphia |         20 |                            0.7517 |                              0.9616 |                         0.2098   |                 0.03842   |
| San Antonio  |         19 |                            0.7834 |                              0.9767 |                         0.1932   |                 0.02334   |
| New York     |         24 |                            0.8253 |                              0.9564 |                         0.1311   |                 0.04358   |
| Austin       |         15 |                            0.8696 |                              0.9889 |                         0.1193   |                 0.01108   |
| Houston      |          4 |                            0.9303 |                              0.9936 |                         0.06327  |                 0.006399  |
| Dallas       |         10 |                            0.9979 |                              0.9995 |                         0.001609 |                 0.0004596 |

## Top Decision-Critical Events

| city         |   event_id | event_start         |   baseline_objective |   recoverable_fraction |   top_5pct_value_share |   marginal_value_gini |   loss_magnitude_rank |   recoverable_rank |   top_tail_rank |   decision_criticality_score |   event_peak_positive_abnormal_deficit |   event_total_precip |
|:-------------|-----------:|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------------:|----------------------:|-------------------:|----------------:|-----------------------------:|---------------------------------------:|---------------------:|
| New York     |         98 | 2019-06-02 18:00:00 |             0.01551  |                 0.2565 |                 0.2367 |                0.5866 |              0.05714  |             0.9905 |          0.981  |                      0.03561 |                              0.01284   |             0.126    |
| New York     |        119 | 2019-06-20 21:00:00 |             0.01783  |                 0.1276 |                 0.302  |                0.6747 |              0.07619  |             0.4    |          1      |                      0.026   |                              0.009007  |             0.03806  |
| Philadelphia |        658 | 2023-07-29 09:00:00 |             0.01981  |                 0.1266 |                 0.2768 |                0.6623 |              0.09524  |             0.3714 |          0.9905 |                      0.02321 |                              0.008792  |             0.01575  |
| Chicago      |        152 | 2019-07-17 19:00:00 |             0.02217  |                 0.268  |                 0.1722 |                0.4585 |              0.1238   |             1      |          0.719  |                      0.02116 |                              0.01334   |             0.2795   |
| New York     |        100 | 2019-06-05 21:00:00 |             0.06729  |                 0.1696 |                 0.2044 |                0.511  |              0.3524   |             0.8857 |          0.9619 |                      0.01771 |                              0.01875   |             0.0105   |
| New York     |        115 | 2019-06-20 01:00:00 |             0.01889  |                 0.1517 |                 0.2132 |                0.534  |              0.08571  |             0.8    |          0.9714 |                      0.01727 |                              0.003382  |             0.007874 |
| Chicago      |        154 | 2019-07-20 05:00:00 |             0.02405  |                 0.2165 |                 0.1722 |                0.4585 |              0.1524   |             0.981  |          0.719  |                      0.01709 |                              0.005707  |             0.01181  |
| New York     |         99 | 2019-06-02 22:00:00 |             0.07897  |                 0.1477 |                 0.1973 |                0.492  |              0.4286   |             0.7429 |          0.9429 |                      0.01433 |                              0.01581   |             0.374    |
| New York     |        108 | 2019-06-16 19:00:00 |             0.2806   |                 0.1541 |                 0.189  |                0.4738 |              0.781    |             0.819  |          0.8619 |                      0.01379 |                              0.02596   |             0.03018  |
| New York     |        116 | 2019-06-20 04:00:00 |             0.02444  |                 0.1451 |                 0.1918 |                0.4792 |              0.1714   |             0.7238 |          0.9333 |                      0.01334 |                              0.003382  |             0.01181  |
| New York     |        104 | 2019-06-11 05:00:00 |             0.5142   |                 0.1488 |                 0.189  |                0.4738 |              0.9143   |             0.781  |          0.8619 |                      0.01332 |                              0.04682   |             0.3412   |
| New York     |        109 | 2019-06-18 07:00:00 |             0.7346   |                 0.1479 |                 0.189  |                0.4738 |              1        |             0.7524 |          0.8619 |                      0.01324 |                              0.06814   |             0.1037   |
| New York     |        106 | 2019-06-13 22:00:00 |             0.09319  |                 0.1475 |                 0.189  |                0.4738 |              0.5143   |             0.7333 |          0.8619 |                      0.01321 |                              0.01414   |             0.1995   |
| Philadelphia |        642 | 2023-07-07 17:00:00 |             0.004405 |                 0.1964 |                 0.1535 |                0.4373 |              0.009524 |             0.9714 |          0.4714 |                      0.01319 |                              0.0008257 |             0.01575  |
| Philadelphia |        654 | 2023-07-20 18:00:00 |             0.06759  |                 0.1947 |                 0.1535 |                0.4373 |              0.3714   |             0.9619 |          0.5571 |                      0.01307 |                              0.01437   |             0.1299   |

## Event-Level Correlations

| target                     | feature                              |   spearman |
|:---------------------------|:-------------------------------------|-----------:|
| decision_criticality_score | marginal_value_gini                  |    0.8412  |
| decision_criticality_score | top_1pct_value_share                 |    0.7759  |
| decision_criticality_score | top_5pct_value_share                 |    0.765   |
| decision_criticality_score | top_10pct_value_share                |    0.765   |
| decision_criticality_score | optimizer_selected_value_share       |    0.01857 |
| decision_criticality_score | event_total_precip                   |   -0.1242  |
| decision_criticality_score | event_peak_positive_abnormal_deficit |   -0.4333  |
| decision_criticality_score | baseline_objective                   |   -0.4585  |
| recoverable_fraction       | marginal_value_gini                  |    0.3709  |
| recoverable_fraction       | optimizer_selected_value_share       |    0.3     |
| recoverable_fraction       | top_1pct_value_share                 |    0.2735  |
| recoverable_fraction       | top_5pct_value_share                 |    0.2373  |
| recoverable_fraction       | top_10pct_value_share                |    0.2373  |
| recoverable_fraction       | event_total_precip                   |   -0.1508  |
| recoverable_fraction       | event_peak_positive_abnormal_deficit |   -0.5295  |
| recoverable_fraction       | baseline_objective                   |   -0.6152  |

## 当前边界与下一步

| item                                  | current_status                                                                                                              | implication                                                                                                                                               | next_step                                                                                                                   |
|:--------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
| scenario_optimum_coverage             | 23 successful LP closures; 5 time-limit/error rows                                                                          | V9 supports representative non-base closure, not full 105-event scenario-optimum closure.                                                                 | Expand scenario-specific LP validation with resume mode or longer time limits for hard New York/Chicago/Philadelphia cases. |
| intervention_parameter_identification | R/C/S effectiveness, cost, caps, delays, and diminishing returns are recovery-regime assumptions.                           | The law is conditional on the specified management regime.                                                                                                | Run parameter ensembles or incorporate observed intervention records if available.                                          |
| surrogate_architecture                | Current surrogate is normalized ridge/ranking evidence plus V12 symbolic formula extraction, not a full graph neural model. | The symbolic law is now explicit and reproducible, but the neural structure-extractor stage remains lightweight.                                          | Train a factorized graph/action-value surrogate if the paper needs a stronger AI-law-discovery component.                   |
| perturbed_optimum_stability           | Representative perturbation solves are available for 4 events with 3 cost/effectiveness perturbations each.                 | The perturbation evidence supports stable value principles, but not yet full-sample action-list stability.                                                | Increase perturbation count and city-event coverage if action stability becomes a central claim.                            |
| budget_phase_coverage                 | Budget-leverage phase analysis currently uses low/base/high budget scales from existing policy replay and proxy tables.     | The current evidence rejects an interior peak over these three scales, but a finer budget sweep would be needed to rule out a narrower nonmonotonic peak. | Run additional budget scales or scenario-specific LP closures if budget phase shape becomes a central contribution.         |

## 论文写作含义

现在可以把 learning/law 部分从“未来要做 law extraction”改成“已经得到一个可复现实证链条”：action-level 的 activated marginal law、finite-budget 的 residual allocation law、event-level 的 top-tail decision-criticality law，以及 V12 的 formula extractor/structure decoupler。V13 进一步修正了一个预期命题：当前 low/base/high 三点预算扫描不支持“中等预算绝对 decision leverage 最高”，而支持“绝对杠杆随预算增加、单位预算杠杆递减”。论文中需要谨慎表述的是：资源效率和 diminishing returns 仍是 recovery-regime 参数；V9/V11 是代表性验证，不是全量非 base 与全量 perturbation closure；完整 graph neural surrogate 仍可作为后续增强，而不是当前主结论的必要条件。

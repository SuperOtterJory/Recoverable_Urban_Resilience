# Event-Regime Generalization V18

## 这一版做了什么

V18 专门补 high-level idea 中的 leave-event-regime-out validation。做法是把 city-event 按 total rain、peak rain、speed impact、rain duration、baseline loss、recoverable fraction、time of day、weekday/weekend 分成 regime；每次整类 regime 留出，只在其他 regime 上训练 surrogate，再测试 held-out regime 的 action-value top-tail capture。

## 主要结论

- tested regime splits = 24。
- factorized low-dimensional law mean top-5% capture = 0.9280，worst split = time_of_day_regime / offpeak_night，capture = 0.8880。
- full additive surrogate mean top-5% capture = 0.9652，minimum = 0.9356。
- activated hand law mean top-5% capture = 1.0000，minimum = 1.0000。
- factorized minus full additive mean top-5% delta = -0.0372；activated law minus deficit-only mean delta = +0.1482。

解释：主要独立证据应看 trained factorized surrogate 与 full additive surrogate。`activated_bottleneck_score` 是 analytic score，和当前 small-signal label 有同源构造关系，因此适合作为公式一致性参照，不应单独当作独立预测胜利。若 factorized law 在 held-out regimes 上接近 full additive model，说明它不是只记住某一类雨型或时段；如果某些 regime 明显掉分，这些 regime 就是后续需要参数敏感性或更强动态表征的位置。

## Regime Split Summary

| split_family                | heldout_regime     |   n_test_events |   n_train_events |   n_test_tokens |   n_train_tokens |   n_test_cities |   mean_total_rain |   mean_peak_speed_impact |   mean_baseline_loss |   mean_recoverable_fraction |
|:----------------------------|:-------------------|----------------:|-----------------:|----------------:|-----------------:|----------------:|------------------:|-------------------------:|---------------------:|----------------------------:|
| baseline_loss_regime        | loss_high          |              35 |               70 |           54986 |           115151 |               7 |           0.3232  |                 0.03929  |              0.3871  |                      0.1108 |
| baseline_loss_regime        | loss_low           |              35 |               70 |           54040 |           116097 |               5 |           0.1112  |                 0.006298 |              0.03283 |                      0.1544 |
| baseline_loss_regime        | loss_medium        |              35 |               70 |           61111 |           109026 |               6 |           0.2548  |                 0.01929  |              0.09911 |                      0.139  |
| duration_regime             | duration_high      |              23 |               82 |           37911 |           132226 |               7 |           0.6145  |                 0.03454  |              0.2864  |                      0.1199 |
| duration_regime             | duration_low       |              37 |               68 |           58880 |           111257 |               7 |           0.03682 |                 0.01433  |              0.124   |                      0.1447 |
| duration_regime             | duration_medium    |              45 |               60 |           73346 |            96791 |               6 |           0.1917  |                 0.02103  |              0.1554  |                      0.1341 |
| peak_rain_regime            | peak_high          |              35 |               70 |           60456 |           109681 |               7 |           0.578   |                 0.02832  |              0.2002  |                      0.1328 |
| peak_rain_regime            | peak_low           |              35 |               70 |           52581 |           117556 |               7 |           0.01082 |                 0.01732  |              0.1541  |                      0.1393 |
| peak_rain_regime            | peak_medium        |              35 |               70 |           57100 |           113037 |               6 |           0.1004  |                 0.01924  |              0.1648  |                      0.1321 |
| recoverable_fraction_regime | recoverable_high   |              35 |               70 |           67415 |           102722 |               5 |           0.1953  |                 0.01474  |              0.1006  |                      0.168  |
| recoverable_fraction_regime | recoverable_low    |              35 |               70 |           49944 |           120193 |               6 |           0.3351  |                 0.03365  |              0.3137  |                      0.1028 |
| recoverable_fraction_regime | recoverable_medium |              35 |               70 |           52778 |           117359 |               5 |           0.1588  |                 0.01648  |              0.1048  |                      0.1334 |
| speed_impact_regime         | speed_high         |              35 |               70 |           54370 |           115767 |               7 |           0.3709  |                 0.0413   |              0.3652  |                      0.1151 |
| speed_impact_regime         | speed_low          |              35 |               70 |           50119 |           120018 |               5 |           0.1041  |                 0.005324 |              0.04452 |                      0.1492 |
| speed_impact_regime         | speed_medium       |              35 |               70 |           65648 |           104489 |               7 |           0.2142  |                 0.01826  |              0.1094  |                      0.1398 |
| time_of_day_regime          | evening_peak       |              21 |               84 |           37163 |           132974 |               6 |           0.3072  |                 0.01896  |              0.1423  |                      0.1476 |
| time_of_day_regime          | midday             |              31 |               74 |           45345 |           124792 |               6 |           0.2162  |                 0.02472  |              0.2057  |                      0.1242 |
| time_of_day_regime          | morning_peak       |              14 |               91 |           20608 |           149529 |               6 |           0.2629  |                 0.02414  |              0.2074  |                      0.1285 |
| time_of_day_regime          | offpeak_night      |              39 |               66 |           67021 |           103116 |               7 |           0.1869  |                 0.0197   |              0.1513  |                      0.1384 |
| total_rain_regime           | rain_high          |              35 |               70 |           63305 |           106832 |               7 |           0.5896  |                 0.02864  |              0.2051  |                      0.1315 |
| total_rain_regime           | rain_low           |              35 |               70 |           51717 |           118420 |               7 |           0.01052 |                 0.01663  |              0.1442  |                      0.139  |
| total_rain_regime           | rain_medium        |              35 |               70 |           55115 |           115022 |               6 |           0.08915 |                 0.0196   |              0.1697  |                      0.1338 |
| weekend_regime              | weekday            |              73 |               32 |          118291 |            51846 |               7 |           0.2245  |                 0.02437  |              0.1989  |                      0.132  |
| weekend_regime              | weekend            |              32 |               73 |           51846 |           118291 |               6 |           0.2417  |                 0.01536  |              0.114   |                      0.1409 |

## Model Summary

| model_id                  | family            | description                                             |   n_splits |   mean_top5_capture |   median_top5_capture |   min_top5_capture |   mean_top5_ndcg |   mean_spearman | hardest_split_family        | hardest_heldout_regime   |
|:--------------------------|:------------------|:--------------------------------------------------------|-----------:|--------------------:|----------------------:|-------------------:|-----------------:|----------------:|:----------------------------|:-------------------------|
| H4_activated_law          | direct_law        | hand-built activated bottleneck score                   |         24 |              1      |                1      |             1      |           1      |         0.7053  | baseline_loss_regime        | loss_high                |
| R2_full_additive          | trained_surrogate | full additive action-value surrogate                    |         24 |              0.9652 |                0.9648 |             0.9356 |           0.9615 |         0.93    | recoverable_fraction_regime | recoverable_medium       |
| R3_full_interaction       | trained_surrogate | full additive surrogate plus explicit interaction terms |         24 |              0.9507 |                0.9504 |             0.9119 |           0.9434 |         0.9314  | speed_impact_regime         | speed_high               |
| R4_factorized_interaction | trained_surrogate | factorized law plus compact interaction terms           |         24 |              0.9388 |                0.9387 |             0.9057 |           0.9314 |         0.7204  | time_of_day_regime          | offpeak_night            |
| R1_factorized_low_dim     | trained_surrogate | seven-feature factorized activated law                  |         24 |              0.928  |                0.9301 |             0.888  |           0.9222 |         0.7189  | time_of_day_regime          | offpeak_night            |
| H2_exposure_only          | heuristic         | OD exposure-only one-factor score                       |         24 |              0.8738 |                0.8748 |             0.8501 |           0.8638 |         0.3044  | speed_impact_regime         | speed_low                |
| H1_deficit_only           | heuristic         | deficit-only one-factor score                           |         24 |              0.8518 |                0.8508 |             0.8405 |           0.8441 |         0.2309  | baseline_loss_regime        | loss_medium              |
| H3_structure_only         | heuristic         | static structure-only one-factor score                  |         24 |              0.5044 |                0.5059 |             0.4634 |           0.488  |        -0.03763 | recoverable_fraction_regime | recoverable_high         |

## Gap Summary

| comparison                        |   n_splits |   mean_delta_top5_capture |   min_delta_top5_capture |   max_delta_top5_capture |   mean_delta_spearman |
|:----------------------------------|-----------:|--------------------------:|-------------------------:|-------------------------:|----------------------:|
| activated_law_vs_deficit_only     |         24 |                   0.1482  |                  0.132   |                 0.1595   |              0.4744   |
| activated_law_vs_structure_only   |         24 |                   0.4956  |                  0.4583  |                 0.5366   |              0.7429   |
| factorized_vs_activated_law       |         24 |                  -0.07198 |                 -0.112   |                -0.03751  |              0.01363  |
| factorized_vs_full_additive       |         24 |                  -0.03722 |                 -0.07963 |                 0.003856 |             -0.2111   |
| full_interaction_vs_full_additive |         24 |                  -0.01458 |                 -0.05446 |                 0.003793 |              0.001341 |

## 论文写作含义

这一版可以把 cross-regime generalization 写进 learning/law 的验证链条：当前 law 不只在 leave-city 下有效，也在不同雨强、速度冲击、持续时间、损失规模和时段留出时保持较高 top-tail capture。边界是：这仍基于当前 sampled action-token table；year-based temporal split 与城市强混杂，因此不应声称已经完成无混杂的 leave-time-period-out。

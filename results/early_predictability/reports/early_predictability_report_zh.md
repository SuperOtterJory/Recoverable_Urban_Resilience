# Early Predictability Analysis V14

## 这一版回答什么问题

主分析是 hindsight counterfactual recoverability：用完整 12 小时事实轨迹问“如果管理者知道后来发生了什么，理论上能恢复多少”。V14 检验一个更谨慎的问题：只用事件开始后前 1/2/3/6 小时的 rainfall 和 abnormal speed 信息，加上静态城市结构，能否在 leave-one-city-out 设置下预测最终 recoverability 与 decision-criticality。

## 主要结果

- best decision-criticality early model: window = 1h, group = speed_plus_static, Spearman = 0.8353, top-20 recall = 0.5714
- 2h all-early decision-criticality: Spearman = 0.7964, top-20 recall = 0.6190
- 1h decision-criticality controls: static-city Spearman = 0.7675, early-speed Spearman = 0.5272, early-rain Spearman = -0.7866
- best recoverable-fraction early model: window = 12h, group = all_early, Spearman = 0.7822, top-20 recall = 0.6190
- 2h all-early recoverable fraction: Spearman = 0.5990, top-20 recall = 0.5238

## Best Metrics By Target

|   window_hours | target                     | feature_group     |   n_features |   n_events |   pearson |   spearman |      mae |   top20_recall |   top20_precision |
|---------------:|:---------------------------|:------------------|-------------:|-----------:|----------:|-----------:|---------:|---------------:|------------------:|
|              1 | decision_criticality_score | speed_plus_static |           19 |        105 |    0.6573 |     0.8353 | 0.002299 |         0.5714 |            0.5714 |
|              1 | marginal_value_gini        | static_city       |            7 |        105 |    0.6198 |     0.9175 | 0.02127  |         0.8095 |            0.8095 |
|             12 | recoverable_fraction       | all_early         |           24 |        105 |    0.686  |     0.7822 | 0.01667  |         0.619  |            0.619  |
|              1 | top_5pct_value_share       | speed_plus_static |           19 |        105 |    0.6388 |     0.7194 | 0.01356  |         0.5714 |            0.5714 |

## All Metrics

|   window_hours | target                     | feature_group     |   n_features |   n_events |   pearson |   spearman |      mae |   top20_recall |   top20_precision |
|---------------:|:---------------------------|:------------------|-------------:|-----------:|----------:|-----------:|---------:|---------------:|------------------:|
|              1 | recoverable_fraction       | early_rain        |            5 |        105 | -0.2469   | -0.3167    | 0.02442  |        0.1905  |           0.1905  |
|              1 | recoverable_fraction       | early_speed       |            9 |        105 |  0.4116   |  0.3987    | 0.02742  |        0.1905  |           0.1905  |
|              1 | recoverable_fraction       | static_city       |            7 |        105 |  0.4772   |  0.5664    | 0.0281   |        0.5238  |           0.5238  |
|              1 | recoverable_fraction       | start_state       |            3 |        105 |  0.4733   |  0.4884    | 0.02205  |        0.3333  |           0.3333  |
|              1 | recoverable_fraction       | rain_plus_speed   |           14 |        105 |  0.4748   |  0.4375    | 0.02535  |        0.2381  |           0.2381  |
|              1 | recoverable_fraction       | speed_plus_static |           19 |        105 |  0.6896   |  0.6957    | 0.01783  |        0.5714  |           0.5714  |
|              1 | recoverable_fraction       | all_early         |           24 |        105 |  0.6895   |  0.6956    | 0.01784  |        0.5714  |           0.5714  |
|              1 | decision_criticality_score | early_rain        |            5 |        105 | -0.5879   | -0.7866    | 0.003152 |        0       |           0       |
|              1 | decision_criticality_score | early_speed       |            9 |        105 |  0.3553   |  0.5272    | 0.003057 |        0.2857  |           0.2857  |
|              1 | decision_criticality_score | static_city       |            7 |        105 |  0.5593   |  0.7675    | 0.003073 |        0.4762  |           0.4762  |
|              1 | decision_criticality_score | start_state       |            3 |        105 |  0.2168   |  0.2246    | 0.002983 |        0.1429  |           0.1429  |
|              1 | decision_criticality_score | rain_plus_speed   |           14 |        105 |  0.3556   |  0.516     | 0.003045 |        0.2857  |           0.2857  |
|              1 | decision_criticality_score | speed_plus_static |           19 |        105 |  0.6573   |  0.8353    | 0.002299 |        0.5714  |           0.5714  |
|              1 | decision_criticality_score | all_early         |           24 |        105 |  0.6571   |  0.835     | 0.0023   |        0.5714  |           0.5714  |
|              1 | top_5pct_value_share       | early_rain        |            5 |        105 | -0.6095   | -0.7569    | 0.02372  |        0       |           0       |
|              1 | top_5pct_value_share       | early_speed       |            9 |        105 |  0.02815  | -0.04285   | 0.0237   |        0       |           0       |
|              1 | top_5pct_value_share       | static_city       |            7 |        105 |  0.7073   |  0.7148    | 0.01637  |        0.9048  |           0.9048  |
|              1 | top_5pct_value_share       | start_state       |            3 |        105 | -0.4426   | -0.5561    | 0.02408  |        0.04762 |           0.04762 |
|              1 | top_5pct_value_share       | rain_plus_speed   |           14 |        105 |  0.04288  | -0.06831   | 0.02453  |        0       |           0       |
|              1 | top_5pct_value_share       | speed_plus_static |           19 |        105 |  0.6388   |  0.7194    | 0.01356  |        0.5714  |           0.5714  |
|              1 | top_5pct_value_share       | all_early         |           24 |        105 |  0.6386   |  0.7194    | 0.01356  |        0.5714  |           0.5714  |
|              1 | marginal_value_gini        | early_rain        |            5 |        105 | -0.5105   | -0.7115    | 0.0316   |        0       |           0       |
|              1 | marginal_value_gini        | early_speed       |            9 |        105 |  0.1016   |  0.2334    | 0.03078  |        0       |           0       |
|              1 | marginal_value_gini        | static_city       |            7 |        105 |  0.6198   |  0.9175    | 0.02127  |        0.8095  |           0.8095  |
|              1 | marginal_value_gini        | start_state       |            3 |        105 | -0.2165   | -0.3355    | 0.032    |        0.04762 |           0.04762 |
|              1 | marginal_value_gini        | rain_plus_speed   |           14 |        105 |  0.1012   |  0.1938    | 0.03181  |        0       |           0       |
|              1 | marginal_value_gini        | speed_plus_static |           19 |        105 |  0.5427   |  0.8194    | 0.01946  |        0.4286  |           0.4286  |
|              1 | marginal_value_gini        | all_early         |           24 |        105 |  0.5423   |  0.819     | 0.01947  |        0.4286  |           0.4286  |
|              2 | recoverable_fraction       | early_rain        |            5 |        105 |  0.001894 | -0.04613   | 0.02524  |        0.381   |           0.381   |
|              2 | recoverable_fraction       | early_speed       |            9 |        105 |  0.4407   |  0.3987    | 0.02613  |        0.1905  |           0.1905  |
|              2 | recoverable_fraction       | static_city       |            7 |        105 |  0.4772   |  0.5664    | 0.0281   |        0.5238  |           0.5238  |
|              2 | recoverable_fraction       | start_state       |            3 |        105 |  0.4414   |  0.4287    | 0.02274  |        0.2857  |           0.2857  |
|              2 | recoverable_fraction       | rain_plus_speed   |           14 |        105 |  0.4331   |  0.3947    | 0.02517  |        0.2381  |           0.2381  |
|              2 | recoverable_fraction       | speed_plus_static |           19 |        105 |  0.5156   |  0.6003    | 0.02217  |        0.5238  |           0.5238  |
|              2 | recoverable_fraction       | all_early         |           24 |        105 |  0.5255   |  0.599     | 0.02297  |        0.5238  |           0.5238  |
|              2 | decision_criticality_score | early_rain        |            5 |        105 | -0.5231   | -0.6485    | 0.003274 |        0       |           0       |
|              2 | decision_criticality_score | early_speed       |            9 |        105 |  0.5266   |  0.5717    | 0.002389 |        0.381   |           0.381   |
|              2 | decision_criticality_score | static_city       |            7 |        105 |  0.5593   |  0.7675    | 0.003073 |        0.4762  |           0.4762  |
|              2 | decision_criticality_score | start_state       |            3 |        105 |  0.2361   |  0.2507    | 0.002919 |        0.1905  |           0.1905  |
|              2 | decision_criticality_score | rain_plus_speed   |           14 |        105 |  0.449    |  0.4877    | 0.002493 |        0.381   |           0.381   |
|              2 | decision_criticality_score | speed_plus_static |           19 |        105 |  0.6669   |  0.8092    | 0.002154 |        0.5714  |           0.5714  |
|              2 | decision_criticality_score | all_early         |           24 |        105 |  0.6145   |  0.7964    | 0.002251 |        0.619   |           0.619   |
|              2 | top_5pct_value_share       | early_rain        |            5 |        105 | -0.608    | -0.7711    | 0.02437  |        0       |           0       |
|              2 | top_5pct_value_share       | early_speed       |            9 |        105 | -0.04688  | -0.0124    | 0.01997  |        0       |           0       |
|              2 | top_5pct_value_share       | static_city       |            7 |        105 |  0.7073   |  0.7148    | 0.01637  |        0.9048  |           0.9048  |
|              2 | top_5pct_value_share       | start_state       |            3 |        105 | -0.5048   | -0.5326    | 0.02408  |        0.04762 |           0.04762 |
|              2 | top_5pct_value_share       | rain_plus_speed   |           14 |        105 | -0.1193   | -0.09842   | 0.02041  |        0       |           0       |
|              2 | top_5pct_value_share       | speed_plus_static |           19 |        105 |  0.6152   |  0.7092    | 0.01357  |        0.4762  |           0.4762  |
|              2 | top_5pct_value_share       | all_early         |           24 |        105 |  0.5975   |  0.7052    | 0.01345  |        0.4762  |           0.4762  |
|              2 | marginal_value_gini        | early_rain        |            5 |        105 | -0.5281   | -0.7315    | 0.03253  |        0       |           0       |
|              2 | marginal_value_gini        | early_speed       |            9 |        105 |  0.07406  |  0.2094    | 0.02698  |        0       |           0       |
|              2 | marginal_value_gini        | static_city       |            7 |        105 |  0.6198   |  0.9175    | 0.02127  |        0.8095  |           0.8095  |
|              2 | marginal_value_gini        | start_state       |            3 |        105 | -0.2458   | -0.3368    | 0.03191  |        0.04762 |           0.04762 |
|              2 | marginal_value_gini        | rain_plus_speed   |           14 |        105 |  0.02979  |  0.1581    | 0.02693  |        0       |           0       |
|              2 | marginal_value_gini        | speed_plus_static |           19 |        105 |  0.4911   |  0.8181    | 0.01967  |        0.4286  |           0.4286  |
|              2 | marginal_value_gini        | all_early         |           24 |        105 |  0.4625   |  0.8083    | 0.0196   |        0.4286  |           0.4286  |
|              3 | recoverable_fraction       | early_rain        |            5 |        105 | -0.04916  | -0.04665   | 0.02621  |        0.09524 |           0.09524 |
|              3 | recoverable_fraction       | early_speed       |            9 |        105 |  0.4439   |  0.4224    | 0.02461  |        0.2857  |           0.2857  |
|              3 | recoverable_fraction       | static_city       |            7 |        105 |  0.4772   |  0.5664    | 0.0281   |        0.5238  |           0.5238  |
|              3 | recoverable_fraction       | start_state       |            3 |        105 |  0.4418   |  0.44      | 0.02282  |        0.2857  |           0.2857  |
|              3 | recoverable_fraction       | rain_plus_speed   |           14 |        105 |  0.3881   |  0.369     | 0.02593  |        0.2381  |           0.2381  |
|              3 | recoverable_fraction       | speed_plus_static |           19 |        105 |  0.464    |  0.5959    | 0.02266  |        0.619   |           0.619   |
|              3 | recoverable_fraction       | all_early         |           24 |        105 |  0.4509   |  0.5857    | 0.02279  |        0.619   |           0.619   |
|              3 | decision_criticality_score | early_rain        |            5 |        105 | -0.2866   | -0.286     | 0.003109 |        0.09524 |           0.09524 |
|              3 | decision_criticality_score | early_speed       |            9 |        105 |  0.5233   |  0.5467    | 0.002349 |        0.5238  |           0.5238  |
|              3 | decision_criticality_score | static_city       |            7 |        105 |  0.5593   |  0.7675    | 0.003073 |        0.4762  |           0.4762  |
|              3 | decision_criticality_score | start_state       |            3 |        105 |  0.2315   |  0.2508    | 0.002932 |        0.1905  |           0.1905  |
|              3 | decision_criticality_score | rain_plus_speed   |           14 |        105 |  0.4708   |  0.486     | 0.00247  |        0.5238  |           0.5238  |
|              3 | decision_criticality_score | speed_plus_static |           19 |        105 |  0.6037   |  0.704     | 0.002327 |        0.5714  |           0.5714  |
|              3 | decision_criticality_score | all_early         |           24 |        105 |  0.5759   |  0.6896    | 0.002349 |        0.5238  |           0.5238  |
|              3 | top_5pct_value_share       | early_rain        |            5 |        105 | -0.6646   | -0.8347    | 0.02431  |        0       |           0       |
|              3 | top_5pct_value_share       | early_speed       |            9 |        105 |  0.01438  |  0.01762   | 0.01997  |        0.04762 |           0.04762 |
|              3 | top_5pct_value_share       | static_city       |            7 |        105 |  0.7073   |  0.7148    | 0.01637  |        0.9048  |           0.9048  |
|              3 | top_5pct_value_share       | start_state       |            3 |        105 | -0.5014   | -0.5274    | 0.02413  |        0.04762 |           0.04762 |
|              3 | top_5pct_value_share       | rain_plus_speed   |           14 |        105 | -0.06273  | -0.04781   | 0.02065  |        0       |           0       |
|              3 | top_5pct_value_share       | speed_plus_static |           19 |        105 |  0.6174   |  0.716     | 0.01342  |        0.5238  |           0.5238  |
|              3 | top_5pct_value_share       | all_early         |           24 |        105 |  0.5854   |  0.7032    | 0.014    |        0.4762  |           0.4762  |
|              3 | marginal_value_gini        | early_rain        |            5 |        105 | -0.5338   | -0.7363    | 0.03212  |        0       |           0       |
|              3 | marginal_value_gini        | early_speed       |            9 |        105 |  0.1105   |  0.2279    | 0.02721  |        0.04762 |           0.04762 |
|              3 | marginal_value_gini        | static_city       |            7 |        105 |  0.6198   |  0.9175    | 0.02127  |        0.8095  |           0.8095  |
|              3 | marginal_value_gini        | start_state       |            3 |        105 | -0.2444   | -0.3364    | 0.03199  |        0.04762 |           0.04762 |
|              3 | marginal_value_gini        | rain_plus_speed   |           14 |        105 |  0.06429  |  0.1838    | 0.02763  |        0.04762 |           0.04762 |
|              3 | marginal_value_gini        | speed_plus_static |           19 |        105 |  0.5112   |  0.7813    | 0.01986  |        0.381   |           0.381   |
|              3 | marginal_value_gini        | all_early         |           24 |        105 |  0.4689   |  0.7571    | 0.0207   |        0.381   |           0.381   |
|              6 | recoverable_fraction       | early_rain        |            5 |        105 | -0.123    | -0.1901    | 0.02683  |        0.09524 |           0.09524 |
|              6 | recoverable_fraction       | early_speed       |            9 |        105 |  0.411    |  0.3551    | 0.02585  |        0.2381  |           0.2381  |
|              6 | recoverable_fraction       | static_city       |            7 |        105 |  0.4772   |  0.5664    | 0.0281   |        0.5238  |           0.5238  |
|              6 | recoverable_fraction       | start_state       |            3 |        105 |  0.4333   |  0.4333    | 0.0232   |        0.2857  |           0.2857  |
|              6 | recoverable_fraction       | rain_plus_speed   |           14 |        105 |  0.3588   |  0.3009    | 0.02789  |        0.2381  |           0.2381  |
|              6 | recoverable_fraction       | speed_plus_static |           19 |        105 |  0.4953   |  0.6207    | 0.022    |        0.5714  |           0.5714  |
|              6 | recoverable_fraction       | all_early         |           24 |        105 |  0.5443   |  0.6314    | 0.02023  |        0.5714  |           0.5714  |
|              6 | decision_criticality_score | early_rain        |            5 |        105 | -0.356    | -0.5222    | 0.003407 |        0       |           0       |
|              6 | decision_criticality_score | early_speed       |            9 |        105 |  0.3802   |  0.4328    | 0.002625 |        0.4762  |           0.4762  |
|              6 | decision_criticality_score | static_city       |            7 |        105 |  0.5593   |  0.7675    | 0.003073 |        0.4762  |           0.4762  |
|              6 | decision_criticality_score | start_state       |            3 |        105 |  0.2527   |  0.2621    | 0.002902 |        0.1905  |           0.1905  |
|              6 | decision_criticality_score | rain_plus_speed   |           14 |        105 |  0.3232   |  0.3681    | 0.002895 |        0.2857  |           0.2857  |
|              6 | decision_criticality_score | speed_plus_static |           19 |        105 |  0.3974   |  0.5295    | 0.002859 |        0.3333  |           0.3333  |
|              6 | decision_criticality_score | all_early         |           24 |        105 |  0.4682   |  0.5677    | 0.002587 |        0.381   |           0.381   |
|              6 | top_5pct_value_share       | early_rain        |            5 |        105 | -0.6219   | -0.8314    | 0.02458  |        0       |           0       |
|              6 | top_5pct_value_share       | early_speed       |            9 |        105 |  0.1033   |  0.07553   | 0.01974  |        0.1429  |           0.1429  |
|              6 | top_5pct_value_share       | static_city       |            7 |        105 |  0.7073   |  0.7148    | 0.01637  |        0.9048  |           0.9048  |
|              6 | top_5pct_value_share       | start_state       |            3 |        105 | -0.4926   | -0.5097    | 0.02389  |        0.04762 |           0.04762 |
|              6 | top_5pct_value_share       | rain_plus_speed   |           14 |        105 |  0.08118  |  0.07787   | 0.01994  |        0.1429  |           0.1429  |
|              6 | top_5pct_value_share       | speed_plus_static |           19 |        105 |  0.5398   |  0.5679    | 0.01534  |        0.5238  |           0.5238  |
|              6 | top_5pct_value_share       | all_early         |           24 |        105 |  0.5453   |  0.5721    | 0.01523  |        0.5714  |           0.5714  |
|              6 | marginal_value_gini        | early_rain        |            5 |        105 | -0.5079   | -0.8007    | 0.03286  |        0       |           0       |
|              6 | marginal_value_gini        | early_speed       |            9 |        105 |  0.1739   |  0.1745    | 0.02614  |        0.2381  |           0.2381  |
|              6 | marginal_value_gini        | static_city       |            7 |        105 |  0.6198   |  0.9175    | 0.02127  |        0.8095  |           0.8095  |
|              6 | marginal_value_gini        | start_state       |            3 |        105 | -0.2257   | -0.3117    | 0.03163  |        0.04762 |           0.04762 |
|              6 | marginal_value_gini        | rain_plus_speed   |           14 |        105 |  0.1544   |  0.1649    | 0.02635  |        0.1905  |           0.1905  |
|              6 | marginal_value_gini        | speed_plus_static |           19 |        105 |  0.4408   |  0.6179    | 0.02292  |        0.381   |           0.381   |
|              6 | marginal_value_gini        | all_early         |           24 |        105 |  0.4564   |  0.6571    | 0.02225  |        0.381   |           0.381   |
|             12 | recoverable_fraction       | early_rain        |            5 |        105 | -0.05446  | -0.1296    | 0.0269   |        0.09524 |           0.09524 |
|             12 | recoverable_fraction       | early_speed       |            9 |        105 |  0.4862   |  0.462     | 0.02373  |        0.3333  |           0.3333  |
|             12 | recoverable_fraction       | static_city       |            7 |        105 |  0.4772   |  0.5664    | 0.0281   |        0.5238  |           0.5238  |
|             12 | recoverable_fraction       | start_state       |            3 |        105 |  0.4175   |  0.423     | 0.02346  |        0.2857  |           0.2857  |
|             12 | recoverable_fraction       | rain_plus_speed   |           14 |        105 |  0.4661   |  0.4282    | 0.02463  |        0.2857  |           0.2857  |
|             12 | recoverable_fraction       | speed_plus_static |           19 |        105 |  0.6149   |  0.7313    | 0.01872  |        0.5238  |           0.5238  |
|             12 | recoverable_fraction       | all_early         |           24 |        105 |  0.686    |  0.7822    | 0.01667  |        0.619   |           0.619   |
|             12 | decision_criticality_score | early_rain        |            5 |        105 | -0.4368   | -0.7046    | 0.003423 |        0       |           0       |
|             12 | decision_criticality_score | early_speed       |            9 |        105 |  0.2836   |  0.3981    | 0.002988 |        0.1905  |           0.1905  |
|             12 | decision_criticality_score | static_city       |            7 |        105 |  0.5593   |  0.7675    | 0.003073 |        0.4762  |           0.4762  |
|             12 | decision_criticality_score | start_state       |            3 |        105 |  0.2548   |  0.2345    | 0.00289  |        0.1905  |           0.1905  |
|             12 | decision_criticality_score | rain_plus_speed   |           14 |        105 |  0.2608   |  0.3534    | 0.003046 |        0.1429  |           0.1429  |
|             12 | decision_criticality_score | speed_plus_static |           19 |        105 |  0.3746   |  0.5867    | 0.002784 |        0.4286  |           0.4286  |
|             12 | decision_criticality_score | all_early         |           24 |        105 |  0.3605   |  0.6089    | 0.002763 |        0.4286  |           0.4286  |
|             12 | top_5pct_value_share       | early_rain        |            5 |        105 | -0.4932   | -0.6928    | 0.02378  |        0.04762 |           0.04762 |
|             12 | top_5pct_value_share       | early_speed       |            9 |        105 |  0.07066  |  0.09578   | 0.02421  |        0       |           0       |
|             12 | top_5pct_value_share       | static_city       |            7 |        105 |  0.7073   |  0.7148    | 0.01637  |        0.9048  |           0.9048  |
|             12 | top_5pct_value_share       | start_state       |            3 |        105 | -0.4718   | -0.4941    | 0.02368  |        0.04762 |           0.04762 |
|             12 | top_5pct_value_share       | rain_plus_speed   |           14 |        105 |  0.0464   |  0.03749   | 0.02346  |        0.04762 |           0.04762 |
|             12 | top_5pct_value_share       | speed_plus_static |           19 |        105 |  0.4883   |  0.5559    | 0.01498  |        0.619   |           0.619   |
|             12 | top_5pct_value_share       | all_early         |           24 |        105 |  0.3941   |  0.4645    | 0.01659  |        0.619   |           0.619   |
|             12 | marginal_value_gini        | early_rain        |            5 |        105 | -0.3959   | -0.6521    | 0.03178  |        0.04762 |           0.04762 |
|             12 | marginal_value_gini        | early_speed       |            9 |        105 |  0.03689  | -0.03276   | 0.03301  |        0       |           0       |
|             12 | marginal_value_gini        | static_city       |            7 |        105 |  0.6198   |  0.9175    | 0.02127  |        0.8095  |           0.8095  |
|             12 | marginal_value_gini        | start_state       |            3 |        105 | -0.1993   | -0.2937    | 0.03135  |        0.04762 |           0.04762 |
|             12 | marginal_value_gini        | rain_plus_speed   |           14 |        105 |  0.007559 | -0.0006079 | 0.03178  |        0.04762 |           0.04762 |
|             12 | marginal_value_gini        | speed_plus_static |           19 |        105 |  0.4128   |  0.6422    | 0.02119  |        0.5714  |           0.5714  |
|             12 | marginal_value_gini        | all_early         |           24 |        105 |  0.2888   |  0.5618    | 0.02472  |        0.5714  |           0.5714  |

## 2h Feature Correlations With Decision-Criticality

|   window_hours | target                     | feature                           |   spearman |
|---------------:|:---------------------------|:----------------------------------|-----------:|
|              2 | decision_criticality_score | n_units                           |    0.6885  |
|              2 | decision_criticality_score | early_positive_deficit_slope      |    0.1646  |
|              2 | decision_criticality_score | early_p90_deficit_max             |    0.1613  |
|              2 | decision_criticality_score | early_mean_deficit_mean           |    0.1441  |
|              2 | decision_criticality_score | dynamic_rain_kernel_sum           |    0.08083 |
|              2 | decision_criticality_score | mean_a_retention                  |   -0.07382 |
|              2 | decision_criticality_score | early_precip_max                  |   -0.074   |
|              2 | decision_criticality_score | event_peak_precip_observed_so_far |   -0.074   |

## 科学解释

这版结果用于给论文加边界：decision-criticality 在很早窗口已有一定可识别性，但主要来自静态城市结构与早期速度异常的组合，而不是 rainfall-only signal。recoverable fraction 更依赖后续轨迹，2 小时窗口只是中等强度预测。因此主文仍应坚持 hindsight counterfactual framing；早期模型可以作为 supplementary evidence，说明哪些事件可能较早显露 decision-criticality，但不能等同于完整在线控制策略。

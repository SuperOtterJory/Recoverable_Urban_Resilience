# Explicit Multi-Objective Footprint LP Validation V41

本版本把 observed event footprint 直接放入 LP 目标函数，而不再只在 residual replay policy 里改变排序。每个 lambda 解一个完整 LP：

`min true_loss - lambda * baseline_loss * sum(cost_ikt * footprint_score_i * u_ikt) / total_budget`

其中 `true_loss` 仍然是论文主目标；第二项表示研究者愿意用多少主目标损失，换取资源投向 observed footprint 更强的区域。报告中的 gain fraction 始终用原始 true loss 重新计算，不使用带 reward 的修改目标。

## 关键结论

- 代表性 hybrid LP 事件数：6；lambda 数：7；Pareto frontier 点数：5。
- lambda=0 的平均 hybrid-LP gain fraction：1.0000；top-5% allocated-unit footprint mass：0.0482。
- 最高 recovery gain 出现在 lambda=0.0000，gain fraction=1.0000，top-5% footprint mass=0.0482。
- 最高 footprint coverage 出现在 lambda=0.2000，gain fraction=0.4170，top-5% footprint mass=0.2446。
- 在平均 LP-gain fraction 损失不超过 0.001 时，最佳 lambda=0.0200，top-5% footprint mass 从 lambda=0 增加 0.0252，gain fraction 变化 -0.0004。
- 在平均 LP-gain fraction 损失不超过 0.0025 时，最佳 lambda=0.0200，top-5% footprint mass 从 lambda=0 增加 0.0252，gain fraction 变化 -0.0004。
- 在平均 LP-gain fraction 损失不超过 0.005 时，最佳 lambda=0.0200，top-5% footprint mass 从 lambda=0 增加 0.0252，gain fraction 变化 -0.0004。
- 在平均 LP-gain fraction 损失不超过 0.01 时，最佳 lambda=0.0200，top-5% footprint mass 从 lambda=0 增加 0.0252，gain fraction 变化 -0.0004。
- 在平均 LP-gain fraction 损失不超过 0.02 时，最佳 lambda=0.0200，top-5% footprint mass 从 lambda=0 增加 0.0252，gain fraction 变化 -0.0004。
- 在平均 LP-gain fraction 损失不超过 0.05 时，最佳 lambda=0.0500，top-5% footprint mass 从 lambda=0 增加 0.0783，gain fraction 变化 -0.0389。

## Lambda Summary

| lambda_footprint | n_events | mean_fraction_of_reference_lp_gain | mean_delta_fraction_vs_lambda0 | mean_recoverable_fraction | mean_top5_allocated_unit_footprint_mass | mean_delta_top5_allocated_mass_vs_lambda0 | mean_selected_unit_footprint_mass | mean_cost_weighted_footprint_mass | mean_cost_weighted_footprint_reward_score | mean_footprint_reward_budget_share | mean_runtime_seconds | n_optimal | n_time_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6 | 1 | 0 | 0.1275 | 0.04819 | 0 | 0.147 | 0.002702 | 0.04849 | 0.04849 | 38.02 | 6 | 0 |
| 0.005 | 6 | 0.9999 | -5.302e-05 | 0.1275 | 0.04819 | 0 | 0.1534 | 0.002867 | 0.05115 | 0.05115 | 30.99 | 6 | 0 |
| 0.02 | 6 | 0.9996 | -0.0004184 | 0.1274 | 0.07342 | 0.02523 | 0.1609 | 0.003107 | 0.05503 | 0.05503 | 21.99 | 6 | 0 |
| 0.05 | 6 | 0.9611 | -0.03893 | 0.1243 | 0.1265 | 0.07828 | 0.256 | 0.009039 | 0.1279 | 0.1279 | 20.78 | 6 | 0 |
| 0.1 | 6 | 0.8262 | -0.1738 | 0.1097 | 0.213 | 0.1648 | 0.3101 | 0.0213 | 0.3175 | 0.3175 | 21.19 | 6 | 0 |
| 0.2 | 6 | 0.417 | -0.583 | 0.05609 | 0.2446 | 0.1964 | 0.2516 | 0.04284 | 0.6836 | 0.6836 | 18.7 | 6 | 0 |
| 0.5 | 6 | 0.2127 | -0.7873 | 0.02635 | 0.2025 | 0.1544 | 0.2025 | 0.049 | 0.7955 | 0.7955 | 17.71 | 6 | 0 |

## Pareto Frontier

| lambda_footprint | n_events | mean_fraction_of_reference_lp_gain | mean_delta_fraction_vs_lambda0 | mean_recoverable_fraction | mean_top5_allocated_unit_footprint_mass | mean_delta_top5_allocated_mass_vs_lambda0 | mean_selected_unit_footprint_mass | mean_cost_weighted_footprint_mass | mean_cost_weighted_footprint_reward_score | mean_footprint_reward_budget_share | mean_runtime_seconds | n_optimal | n_time_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6 | 1 | 0 | 0.1275 | 0.04819 | 0 | 0.147 | 0.002702 | 0.04849 | 0.04849 | 38.02 | 6 | 0 |
| 0.02 | 6 | 0.9996 | -0.0004184 | 0.1274 | 0.07342 | 0.02523 | 0.1609 | 0.003107 | 0.05503 | 0.05503 | 21.99 | 6 | 0 |
| 0.05 | 6 | 0.9611 | -0.03893 | 0.1243 | 0.1265 | 0.07828 | 0.256 | 0.009039 | 0.1279 | 0.1279 | 20.78 | 6 | 0 |
| 0.1 | 6 | 0.8262 | -0.1738 | 0.1097 | 0.213 | 0.1648 | 0.3101 | 0.0213 | 0.3175 | 0.3175 | 21.19 | 6 | 0 |
| 0.2 | 6 | 0.417 | -0.583 | 0.05609 | 0.2446 | 0.1964 | 0.2516 | 0.04284 | 0.6836 | 0.6836 | 18.7 | 6 | 0 |

## Event Best Within Loss Thresholds

| city | event_id | loss_threshold | lambda_footprint | fraction_of_reference_lp_gain | delta_fraction_vs_lambda0 | top5_allocated_unit_footprint_mass | delta_top5_allocated_mass_vs_lambda0 | lambda0_fraction_of_reference_lp_gain | lambda0_top5_allocated_unit_footprint_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Austin | 484 | 0.001 | 0 | 1 | 0 | 0.02537 | 0 | 1 | 0.02537 |
| Austin | 484 | 0.0025 | 0.02 | 0.9989 | -0.001072 | 0.09941 | 0.07404 | 1 | 0.02537 |
| Austin | 484 | 0.005 | 0.02 | 0.9989 | -0.001072 | 0.09941 | 0.07404 | 1 | 0.02537 |
| Austin | 484 | 0.01 | 0.02 | 0.9989 | -0.001072 | 0.09941 | 0.07404 | 1 | 0.02537 |
| Austin | 484 | 0.02 | 0.02 | 0.9989 | -0.001072 | 0.09941 | 0.07404 | 1 | 0.02537 |
| Austin | 484 | 0.05 | 0.05 | 0.9798 | -0.02023 | 0.108 | 0.08262 | 1 | 0.02537 |
| Chicago | 157 | 0.001 | 0.02 | 0.9994 | -0.0006365 | 0.144 | 0.06482 | 1 | 0.07918 |
| Chicago | 157 | 0.0025 | 0.02 | 0.9994 | -0.0006365 | 0.144 | 0.06482 | 1 | 0.07918 |
| Chicago | 157 | 0.005 | 0.02 | 0.9994 | -0.0006365 | 0.144 | 0.06482 | 1 | 0.07918 |
| Chicago | 157 | 0.01 | 0.02 | 0.9994 | -0.0006365 | 0.144 | 0.06482 | 1 | 0.07918 |
| Chicago | 157 | 0.02 | 0.05 | 0.989 | -0.01099 | 0.1805 | 0.1013 | 1 | 0.07918 |
| Chicago | 157 | 0.05 | 0.05 | 0.989 | -0.01099 | 0.1805 | 0.1013 | 1 | 0.07918 |
| Dallas | 64 | 0.001 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Dallas | 64 | 0.0025 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Dallas | 64 | 0.005 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Dallas | 64 | 0.01 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Dallas | 64 | 0.02 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Dallas | 64 | 0.05 | 0 | 1 | 0 | 0.02186 | 0 | 1 | 0.02186 |
| Houston | 52 | 0.001 | 0.02 | 0.9996 | -0.0004399 | 0.04557 | 0.002301 | 1 | 0.04327 |
| Houston | 52 | 0.0025 | 0.02 | 0.9996 | -0.0004399 | 0.04557 | 0.002301 | 1 | 0.04327 |

## 解释

这个检验比 V40 更接近论文里可以写成模型扩展的结论：如果 footprint 是一个社会偏好或公平/可见影响目标，它应该作为显式 secondary objective 进入 LP，而不是被包装成纯 recovery law。

如果小 lambda 就能提高 footprint 且几乎不损失 true recovery gain，说明 footprint 可以作为 tie-breaker；如果更大的 footprint improvement 必须牺牲 recovery gain，则论文应把它写成 recovery-vs-footprint frontier，而不是声称 observed footprint 自然就是最高恢复价值区域。

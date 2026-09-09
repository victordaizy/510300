# 510300 多尺度非线性趋势延续与反转 V1

- 最终状态：`REJECTED_FROZEN_NONLINEAR_TREND_MECHANISM_GATE_FAILED_NO_RESCUE`
- 机制门通过：`false`
- 组合评价执行：`false`
- 收益评价：`NOT_ALLOWED`
- 净夏普率：`NOT_COMPUTED`
- 实盘授权：`false`

## 冻结机制结果

- 有效日：2822
- 线性系数：0.05060035；Newey-West t=0.5271
- 三次系数：-0.11901445；Newey-West t=-1.0633
- 线性系数区块Bootstrap 90%区间：[-0.12438218730465786, 0.22442121347069705]
- 三次系数区块Bootstrap 90%区间：[-0.32010763674510595, 0.07320393905786543]
- 固定论文分数斜率：-1.781781；Newey-West t=-0.3270
- 固定论文分数MSE相对零预测改善：0.00014615

## 硬门

- `minimum_valid_rows`：`true`
- `full_sample_linear_coefficient_positive`：`true`
- `full_sample_cubic_coefficient_negative`：`true`
- `full_sample_linear_newey_west_t_minimum`：`false`
- `full_sample_cubic_newey_west_t_maximum`：`false`
- `linear_bootstrap_90pct_lower_positive`：`false`
- `cubic_bootstrap_90pct_upper_negative`：`false`
- `fixed_paper_score_slope_positive`：`false`
- `fixed_paper_score_newey_west_t_minimum`：`false`
- `fixed_paper_score_bootstrap_90pct_lower_positive`：`false`
- `fixed_paper_score_mse_improvement_vs_zero_positive`：`true`
- `both_structural_periods_preserve_linear_positive_cubic_negative`：`true`
- `both_structural_periods_fixed_score_slope_positive`：`false`

## 裁决

机制硬门未全部通过，冻结协议禁止组合收益评价。不得修改尺度、权重、系数、方向、仓位映射或与既有否决候选组合救援。

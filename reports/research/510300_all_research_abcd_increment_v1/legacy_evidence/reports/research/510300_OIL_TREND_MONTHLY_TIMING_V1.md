# 510300 原油趋势月度择时 V1 正式裁决

- 项目标识：`510300_OIL_TREND_MONTHLY_TIMING_V1`
- 状态：`REJECTED_FROZEN_OIL_TREND_PREDICTIVE_GATE_FAILED_NO_RESCUE`
- 评价区间：2015-01-05 至 2026-08-12
- 可执行资产：仅 `510300.SH` 或人民币现金
- 实盘授权：否

## 冻结前数据决定

WTI 因 `2020-04-20现货价格为负导致预注册对数公式无定义；禁止删点、截尾或平移` 被排除；改用 EIA Brent `RBRTE`。这一决定发生在任何 510300 候选收益读取之前。

## 机制门

- 是否通过：`False`
- 完整月度目标：139
- 风险规避组 / 持有组：67 / 72
- 风险规避组减持有组平均收益：0.01215467877089336
- 风险规避组减持有组中位收益：0.004357230393111311
- Newey-West 斜率 / t 值：-0.026167870807557808 / -0.9104640559650015
- 90% 区块自助置信区间：[-0.0032710976644583314, 0.028836828425067677]

### 逐门槛结果

- `minimum_complete_monthly_targets`：`True`
- `minimum_observations_each_group`：`True`
- `full_sample_mean_difference_negative`：`False`
- `full_sample_median_difference_negative`：`False`
- `bootstrap_90pct_upper_negative`：`False`
- `newey_west_slope_negative`：`True`
- `newey_west_one_sided_t`：`False`
- `every_structural_period_mean_difference_negative`：`False`
- `every_leave_one_calendar_year_out_mean_difference_negative`：`False`

## 组合与夏普率

机制门失败，因此组合收益读取被协议禁止。

- `RETURN_EVALUATION=NOT_ALLOWED`
- `NET_SHARPE=NOT_COMPUTED`

## 最终边界

历史目标是否达到：`False`。前瞻目标是否达到：`False`。本研究不生成订单、不连接券商、不改变任何持仓。

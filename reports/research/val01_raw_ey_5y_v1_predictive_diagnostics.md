# VAL01_RAW_EY_5Y_V1 未来收益预测诊断

> 状态：`REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST`。全部结果均为受污染历史诊断，不是严格样本外Alpha。

## 标签与成熟度审计

- 标签行数：183；有效信号：61。
- 60日：成熟58，右删失3。
- 120日：成熟55，右删失6。
- 242日：成熟49，右删失12。
- 共同日历截至冻结截止日：2427日；H00300独有日期：['2018-06-18']。
- 合成H00300收盘最大误差：9.095e-13。
- 标签表不包含EY、分位、仓位或订单字段。

## IC与分档诊断

|期限|目标|样本|Spearman IC|HAC t|单侧p|均值单调|最高档-最低档均值|
|---:|---|---:|---:|---:|---:|---|---:|
|60|etf_total_return|58|0.3998|2.5527|0.005345|否|5.62%|
|60|h00300_total_return|58|0.4059|2.6118|0.004503|否|5.82%|
|120|etf_total_return|55|0.5373|4.4020|5.363e-06|否|11.95%|
|120|h00300_total_return|55|0.5375|4.4343|4.618e-06|否|12.21%|
|242|etf_total_return|49|0.4915|2.6222|0.004368|是|29.98%|
|242|h00300_total_return|49|0.5017|2.7087|0.003378|是|30.25%|

## 242日冻结分档明细

|分档|样本|510300均值|510300中位数|正收益率|H00300均值|
|---|---:|---:|---:|---:|---:|
|B1_EXPENSIVE|1|-20.40%|-20.40%|0.00%|-20.12%|
|B2|11|-4.01%|-10.47%|36.36%|-3.41%|
|B3|12|0.83%|-3.40%|33.33%|1.22%|
|B4_CHEAP|25|9.58%|12.49%|68.00%|10.13%|

- 510300前/后半段IC：0.5046/0.2032。
- 242日510300与H00300标签Pearson相关：0.9996。


## 242日主要硬门槛

- `minimum_matured_observations`：`PASS`。
- `minimum_observations_per_bucket`：`FAIL`。
- `etf_spearman_ic_strictly_positive`：`PASS`。
- `etf_hac_one_sided_p_value_at_most_0_10`：`PASS`。
- `both_chronological_half_ics_strictly_positive`：`PASS`。
- `etf_bucket_mean_returns_nondecreasing`：`PASS`。
- `etf_cheapest_minus_expensive_mean_positive`：`PASS`。
- `etf_cheapest_minus_expensive_median_positive`：`PASS`。
- `h00300_spearman_confirmation_positive`：`PASS`。
- `h00300_cheapest_minus_expensive_mean_positive`：`PASS`。

主要屏幕结论：`REJECT_PREDICTIVE_SCREEN_STOP_NO_STRATEGY_BACKTEST`。
停止原因：`PRIMARY_BUCKET_SUPPORT_INSUFFICIENT`。

## 治理边界

- 没有生成历史或当前仓位、目标份额、订单或券商动作。
- 没有运行策略收益、成本压力、稳定性或组合回测。
- `alpha_pass=false`；若主要屏幕失败，必须停止该候选的策略回测。

# 沪深300小账户 Alpha V2 2015—2021外部时段结果

- 总状态：`EXTERNAL_PERIOD_REJECTED_FROZEN`
- 父公式修改：`false`
- 正式验证期每日成员：300
- 成员价格缺口：0

|情景|策略年化|基准年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|基础5bp|-14.27%|7.19%|-21.46%|-71.71%|99|
|压力15bp|-6.83%|7.19%|-14.02%|-54.50%|187|

- Bootstrap年化超额95%区间：[-0.3870718580756482, -0.00413277870785377]
- 最小成交：5001.00元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- FAIL `positive_predefined_periods`
- PASS `all_trades_at_least_5000`

结果后禁止在该区间修改父公式补救；无论是否通过，均不自动生成仓位或订单。

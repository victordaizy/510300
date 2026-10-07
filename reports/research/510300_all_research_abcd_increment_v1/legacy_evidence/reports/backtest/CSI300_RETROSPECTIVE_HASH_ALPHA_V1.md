# 沪深300历史成员回溯性代码哈希留出 V1

- 状态：`SECONDARY_HASH_HOLDOUT_REJECTED_FROZEN`
- 证据等级：`RETROSPECTIVE_HASH_HOLDOUT_SECONDARY_EVIDENCE`
- 永久盲测声明：`禁止`

|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|5bp|9.74%|16.55%|-6.81%|-32.12%|115|
|15bp|9.33%|16.55%|-7.22%|-30.26%|115|

- Bootstrap年化超额95%区间：[-0.35665881695383234, 0.2856186652063716]
- 最小成交额：5000.00元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- PASS `positive_predefined_periods`
- PASS `all_trades_at_least_5000`
- PASS `t_plus_one_and_lot_engine`

即使全部门槛通过，本分支也只提供补充证据；必须等待全A永久盲测或真实前瞻期确认。
本结果不生成仓位、订单或券商连接。

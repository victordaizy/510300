# 沪深300十因子时间迁移 Alpha V1 近期评估

- 状态：`TEMPORAL_TRANSFER_REJECTED_FROZEN`
- 模型训练：仅2015—2021
- 近期评估：2021—2026，模型不更新
- 因子：10个

|成本|策略年化|基准年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|5bp|1.71%|1.23%|0.48%|-51.71%|167|
|15bp|-1.31%|1.23%|-2.54%|-49.63%|158|

- Bootstrap 95%区间：[-0.2744258500367887, 0.2827031756247682]
- 最小成交：5055.00元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- FAIL `positive_predefined_periods`
- PASS `all_trades_at_least_5000`

评估后禁止修改模型、周期、持股数或保留排名补救；不生成仓位或订单。

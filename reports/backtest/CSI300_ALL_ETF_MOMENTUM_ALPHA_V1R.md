# 全ETF九因子轮动 V1R 复权修复评估

- 状态：`ALL_ETF_V1R_REJECTED_FROZEN`
- 公式：与首次冻结V1完全一致
- 总收益链：Tushare fund_daily 除权昨收

|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|5bp|7.90%|4.50%|3.40%|-53.50%|129|
|15bp|5.78%|4.50%|1.28%|-51.59%|125|

- Bootstrap 95%区间：[-0.12923068001296228, 0.2063471447767183]
- 最小成交：10829.20元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- PASS `positive_predefined_periods`
- PASS `all_trades_at_least_5000`

仅为冻结历史研究，不生成仓位、订单或实盘连接。

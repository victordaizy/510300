# CSI300 点时全ETF九因子轮动 V1

- 状态：`ALL_ETF_REJECTED_FROZEN`
- 母表资产：1295只
- 因子：9个

|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|5bp|-12.41%|4.50%|-16.91%|-93.31%|113|
|15bp|-12.74%|4.50%|-17.24%|-94.18%|115|

- Bootstrap 95%区间：[-0.39634833649783563, 0.06305527442250977]
- 最小成交：6867.20元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- PASS `positive_predefined_periods`
- PASS `all_trades_at_least_5000`

仅为冻结历史研究，不生成仓位、订单或实盘连接。

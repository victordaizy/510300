# CSI300 ETF 跨资产轮动 Alpha V1

- 状态：`ETF_ROTATION_REJECTED_FROZEN`
- 因子：8个

|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|
|---|---:|---:|---:|---:|---:|
|5bp|7.39%|4.50%|2.90%|-40.70%|67|
|15bp|6.81%|4.50%|2.31%|-41.02%|67|

- Bootstrap 95%区间：[-0.10663472137296942, 0.15287168524315115]
- 最小成交：10830.00元

## 门槛

- FAIL `base_annualized_excess_20pct`
- FAIL `stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- PASS `positive_predefined_periods`
- PASS `all_trades_at_least_5000`

历史研究结果不生成纸面仓位、订单或实盘连接。

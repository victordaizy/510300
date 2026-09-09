# 双袖带信号向2万元三仓投影 V1

- 状态：`DUAL_SLEEVE_PROJECTION_REJECTED_FROZEN`

|时段|策略年化|基准年化|基础超额|压力超额|Bootstrap下界|
|---|---:|---:|---:|---:|---:|
|2015—2021|-16.41%|7.19%|-23.60%|-15.14%|-40.79%|
|2021—2026|14.95%|1.23%|13.72%|12.15%|-2.92%|

## 门槛

- FAIL `recent_base_excess_20pct`
- FAIL `recent_stress_excess_20pct`
- FAIL `recent_bootstrap_lower_positive`
- FAIL `external_base_excess_20pct`
- FAIL `external_stress_excess_20pct`
- FAIL `external_bootstrap_lower_positive`
- PASS `all_trades_at_least_5000`

历史已受研究污染；不生成仓位、订单或实盘连接。

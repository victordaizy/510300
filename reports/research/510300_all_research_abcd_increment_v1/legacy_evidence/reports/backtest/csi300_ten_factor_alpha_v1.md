# 沪深300十因子非线性小账户 Alpha V1 历史结果

- 总状态：`HISTORICAL_FORMULA_REJECTED_FROZEN`
- 证据标签：`FORMULA_PSEUDO_OOS_PROJECT_HISTORY_CONTAMINATED`
- 严格未来样本外：`NOT_AVAILABLE`
- 因子数量：10 / 10

|区间/成本|策略年化|基准年化|年化超额|最大回撤|
|---|---:|---:|---:|---:|
|全伪样本外/5bp|3.86%|7.42%|-3.56%|-33.02%|
|全伪样本外/15bp|3.23%|7.42%|-4.19%|-34.66%|
|2025后/5bp|-7.81%|14.18%|-22.00%|-33.02%|
|2025后/15bp|-7.61%|14.18%|-21.79%|-33.90%|

- 20日块Bootstrap年化超额95%区间：[-0.2214972243720559, 0.25336341644056143]
- 日度Spearman IC均值：0.0319
- Top3未来10日平均对数超额：-0.1401%
- 最小实际成交：5000.00元

## 门槛

- FAIL `full_base_annualized_excess_20pct`
- FAIL `full_stress_annualized_excess_20pct`
- FAIL `subperiod_base_annualized_excess_20pct`
- FAIL `subperiod_stress_annualized_excess_20pct`
- FAIL `bootstrap_lower_bound_positive`
- PASS `all_trades_at_least_5000`

## 边界

模型失败后不得在同一历史上修改十因子、模型参数、持股数或周期补救。即使全部历史门槛通过，也不能替代冻结日之后的真正前向证据，且不生成仓位或订单。

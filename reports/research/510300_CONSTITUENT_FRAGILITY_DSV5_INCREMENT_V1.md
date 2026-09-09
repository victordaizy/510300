# 510300 成分脆弱性 DSV5 增量检验 V1

- 最终状态：`REJECTED_FROZEN_CONSTITUENT_FRAGILITY_INCREMENT`
- G0：`PASS`
- G1：`PASS_NONOVERLAPPING_ORIGIN_IDENTIFIABILITY`
- G2（B1 对 B0）：`PASS`
- G3（B2 对 B1）：`FAIL`
- 证据类别：`RESEARCH_OBSERVED_PREQUENTIAL`，不是严格未观察样本外。
- 仓位、净值、组合收益、Sharpe、Paper/Shadow、订单和实盘：均未生成。

## 无标签原点门

- 主偏移合格原点：`258`
- 首次训练/严格前序评价：`80` / `178`
- 三个时代：`ERA_1=60` / `ERA_2=59` / `ERA_3=59`

## G2：B1 对 B0

- 总体 QLIKE 相对改善：`2.083190%`
- 13 原点块 Bootstrap 单侧 90% 下界：`0.0119752778`
- 时代通过数：`3/3`
- 最新时代改善：`true`

## G3：B2 对 B1

- 总体 QLIKE 相对改善：`0.401508%`（门槛 2%）
- log(DSV5+1e-8) MSE：B1=`11.75720498`，B2=`11.66003535`
- Bootstrap 单侧 90% 下界：`-0.0082902504`
- 改善偏移：`5/5`
- 最高/最低预测风险五分位实际 DSV5 比：`1.394380`
- 平均预测/平均实际：`1.419283`

### G3 原子门

- `overall_qlike_improvement_at_least_2pct`：`FAIL`
- `log_dsv5_mse_improves`：`PASS`
- `bootstrap_90pct_lower_positive`：`FAIL`
- `minimum_positive_eras`：`PASS`
- `latest_era_improves`：`PASS`
- `minimum_positive_registered_offsets`：`PASS`
- `highest_to_lowest_actual_dsv5_ratio`：`FAIL`
- `mean_prediction_to_actual_ratio`：`FAIL`

## 裁决

B1通过，但成分脆弱性B2未满足全部增量门；该成分增量家族冻结拒绝，不得加入宏观、NBS或其他特征救援，组合评价不允许。

`POSITION_IMPACT=0`；本结果不得转换为当前仓位或订单。

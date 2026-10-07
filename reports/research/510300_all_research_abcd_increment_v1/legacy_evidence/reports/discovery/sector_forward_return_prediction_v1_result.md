# 510300 板块未来贡献预测 V1 历史结果

- 总状态：`HISTORICAL_REJECTED_FROZEN`
- 数据截止：`2026-08-14`
- 历史标签：`HISTORICALLY_CONTAMINATED`
- 预测资格：`NOT_AUTHORIZED`
- 仓位、订单、券商：`DISABLED`

## 固定预测问题

信号日官方权重与中信一级行业固定，目标从下一交易日开盘至第60个交易日收盘。缺失端点不重加权。S1 只使用板块权重、盈利收益率、账面收益率、ROE、TTM盈利和收入同比。

## 样本

- 成熟目标快照：77
- NO_VIEW 目标快照：0
- 板块目标行：2122
- 伪样本外预测：38
- 已校准分布预测：24
- 预测日期：2023-03-31 至 2026-04-30

## 唯一候选 S1

- Spearman：-0.15877010613852718
- 移动块 Bootstrap 95%区间：[-0.5389809673126909, 0.2914454734912994]
- S1 MAE：0.0922993449262584
- B0 聚合估值 MAE：0.11344545944346436
- 扩展历史均值 MAE：0.05747516895284123
- 相对 B0 改善：0.1863989499530755
- 相对历史均值改善：-0.6058994972592537
- 方向准确率：0.39473684210526316
- 前半段 Spearman：0.08771929824561404
- 后半段 Spearman：-0.22982456140350876
- 与 ETF_X60 Spearman：-0.17934128460444249
- Brier Skill：-0.3776022488728277
- 80%区间覆盖率：0.75

## 冻结门槛

- PASS `minimum_mature_oos_predictions`
- FAIL `minimum_spearman`
- FAIL `bootstrap_spearman_lower_bound`
- PASS `mae_improvement_vs_aggregate`
- FAIL `mae_improvement_vs_expanding_mean`
- FAIL `direction_accuracy`
- PASS `first_half_spearman`
- FAIL `second_half_spearman`
- FAIL `etf_x60_spearman`
- PASS `minimum_calibrated_predictions`
- FAIL `brier_skill`
- PASS `interval_80_coverage`

## 解释边界

历史通过也只允许进入真正前向准备；历史失败则冻结失败。任何结果都不生成仓位、订单或券商动作，也不允许改期限、alpha、特征或门槛补救。

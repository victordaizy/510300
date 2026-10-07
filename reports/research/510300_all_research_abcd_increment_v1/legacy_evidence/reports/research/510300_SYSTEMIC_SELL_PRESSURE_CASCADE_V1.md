# 510300 全A股系统性卖压扩散次日退出 V1

- 状态：`VISIBLE_REJECTED_FROZEN_REPLICATION_UNREAD`
- 目标达成：`false`
- 仓位：仅满仓510300或空仓现金；信号后只退出一个交易日。

## 2019—2022可见验证

- 风险分数校准样本：974日；95%阈值：0.935795。
- 空仓信号：24日，占验证信号日2.47%。
- 基础/压力年化净超额：-5.71% / -6.75%。
- 基础/压力242日滚动超额中位数：-6.51% / -7.14%。
- 压力策略年化：1.96%；H00300年化：8.71%。
- 压力最大回撤：-44.35%；成交腿数：48。
- 坏日召回：1.68%；好日误退出：3.22%；坏日机会损失捕获：2.67%。

## 硬门

- `base_annualized_excess_at_least_20pct`：FAIL
- `stress_annualized_excess_at_least_20pct`：FAIL
- `base_rolling_excess_median_at_least_20pct`：FAIL
- `stress_rolling_excess_median_at_least_20pct`：FAIL
- `minimum_signal_days`：PASS
- `all_visible_gates_pass`：FAIL

## 条件复验

可见门未全部通过，2023—2026独立哈希股票面板未载入。

## 决策

失败后不改变特征、窗口、阈值、空仓期限或验证区间救回；本报告未生成当前状态、Paper/Shadow、订单或实盘授权。

# 广泛流动ETF流动性冲击反转 V1：visible

- 状态：`REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN`
- 目标已达成：否。历史结果不能验证40个百分点目标。
- 区间：2016-01-04 至 2021-12-31。
- 账户：500,000元；用户交易费率为每条买卖腿0.0001。

## 成本后结果

- H00300年化：6.99%
- 基础净年化：0.84%；净超额：-6.15%；净夏普：-0.047；最大回撤：-13.55%
- 压力净年化：-1.21%；净超额：-8.20%；净夏普：-0.295；最大回撤：-13.95%
- 未通过门槛：base_annualized_excess_at_least_40pct, stress_annualized_excess_at_least_40pct, base_strategy_sharpe_at_least_1_5, stress_strategy_sharpe_at_least_1_5, base_rolling_excess_median_at_least_40pct, stress_rolling_excess_median_at_least_40pct, base_rolling_sharpe_median_at_least_1_5, stress_rolling_sharpe_median_at_least_1_5, base_bootstrap_excess_lower_bound_positive, stress_bootstrap_excess_lower_bound_positive, base_bootstrap_sharpe_lower_bound_at_least_1, stress_bootstrap_sharpe_lower_bound_at_least_1, minimum_target_qualified_year_blocks

## 覆盖与执行

- 可执行信号：821条；信号日：266个；涉及ETF：65只。
- 买入成交：261笔；卖出成交：261笔；受阻退出尝试：0次。
- 最大成交额占过去20日中位成交额：0.0972%。
- 最大持仓数：5；最大毛敞口：1.000。

## 证据限制

- ETF行情和基金主表是当前归档，缺少逐日不可变供应商版本链。
- 历史数据没有逐笔买卖价差；基础与压力滑点、冲击为冻结模型值。
- 历史结果只用于淘汰或进入下一阶段，不是收益承诺、仓位、订单或实盘授权。

## 决策

冻结拒绝本候选，不打开封存期，不调参救回。

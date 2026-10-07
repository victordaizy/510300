# 510300 驱动周期历史发现 V1 结果

- 总状态：`PARTIAL_OR_REJECTED_NO_FORECAST_AUTHORIZATION`
- 数据截止：`2026-08-14`
- 历史标签：`HISTORICALLY_CONTAMINATED`
- 预测资格：`NOT_AUTHORIZED`
- 仓位、订单、券商：`DISABLED`

## 周期结构

- 归因有效日：1489
- NO_VIEW 日：122
- 驱动周期：62
- M1 成熟事件：36
- T1 成熟事件：56

## M1 鱼中候选

- 状态：`HISTORICAL_REJECTED_FROZEN`
- MIDDLE20 Lift：0.9489664082687338
- Bootstrap 95% Lift 区间：[0.5582155342757258, 1.3955388356893144]
- X20 中位数：-0.0036642888121188877
- TAIL20 发生率：0.1388888888888889；基准：0.10211027910142954
- PASS `minimum_mature_events`
- FAIL `minimum_middle20_lift`
- FAIL `bootstrap_lift_lower_bound`
- FAIL `median_x20_positive`
- FAIL `positive_x20_rate`
- FAIL `tail20_not_worse_than_baseline`
- FAIL `delay_retention`
- FAIL `first_half_direction`
- PASS `second_half_direction`

## T1 鱼尾候选

- 状态：`HISTORICAL_REJECTED_FROZEN`
- TAIL20 Lift：1.0492857142857142
- Bootstrap 95% Lift 区间：[0.0, 2.0985714285714283]
- PASS `minimum_mature_events`
- PASS `minimum_tail20_events`
- FAIL `minimum_tail20_lift`
- FAIL `bootstrap_lift_lower_bound`
- FAIL `delay_retention`
- PASS `first_half_direction`
- FAIL `second_half_direction`

## 解释边界

历史通过只允许进入真正前向准备；历史失败则冻结失败。任何结果都不生成仓位、订单或券商动作。

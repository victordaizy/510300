# 510300宏观压力规避：2015起点滚动敏感性结果

- 最终状态：`REJECTED_2015_START_SENSITIVITY_HISTORICAL_OR_ROBUSTNESS_GATE_FAILED_NO_RESCUE`
- 证据等级：`POST_REJECTION_WINDOW_SENSITIVITY_ONLY`
- 收益评估：`COMPLETED_ONCE`
- 组合回测：`COMPLETED_ONCE_AFTER_EVENT_GATE_PASS`
- 数据范围：2015-01-01至2026-08-25
- 资产边界：仅510300.SH与人民币现金
- 2021起点原拒绝是否被覆盖：`false`
- 实盘授权：`false`

## 严格滚动窗口可用性

- 3年模型首个有效信号日：2020-08-24
- 5年模型首个有效信号日：2022-08-22
- 当期观察不进入90%分位；未使用扩展窗口或缩短窗口

## 5年主模型事件门

- 有效交易日：972
- 原始事件起点：16
- 间隔20交易日后的独立事件：10
- 具有完整20日结果的独立事件：9
- 事件门状态：`PASS_EVENT_GATE_PORTFOLIO_RUN_ALLOWED_ONCE`

| 事件硬门 | 通过 |
|---|---:|
| minimum_eight_independent_complete_events | 是 |
| event_mean_below_nonpressure | 是 |
| event_median_below_nonpressure | 是 |
| minimum_sixty_percent_negative | 是 |
| delete_worst_direction_unchanged | 是 |
| events_span_at_least_two_years | 是 |

## 组合层

- 5年净夏普率：0.151844
- 5年相对H00300年化超额：-3.671231%
- 5年相对510300含分红买入持有年化超额：-3.204631%
- 3年净夏普率：-0.051628
- 5年双倍成本净夏普率：0.107711
- 组合机制判定：`REJECTED_FROZEN_HISTORICAL_OR_ROBUSTNESS_GATE_FAILED_NO_RESCUE`

## 证据与治理边界

- 本次窗口变化是在看到2021起点事件不足后提出，只属于事后窗口敏感性。
- 即使历史硬门通过，也不能把它重标为预注册确认，不能覆盖原拒绝结论。
- 参数营救、真实账户仓位映射、订单生成、券商连接与实盘授权均关闭。
- `goal_achieved=false`：需要独立冻结前向证据，而非再改变历史窗口。

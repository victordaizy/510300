# 510300宏观压力规避V1冻结研究结果

- 最终状态：`REJECTED_FROZEN_INSUFFICIENT_EVENT_SUPPORT_NO_RESCUE`
- 收益评估：`NOT_ALLOWED`
- 组合回测：`SKIPPED_EVENT_GATE_FAILED`
- 数据范围：2021-01-01至2026-08-25，未使用2016—2020数据
- 资产边界：仅510300.SH与人民币现金
- 实盘授权：`false`

## 严格滚动窗口可用性

- 3年模型首个有效信号日：2024-04-01
- 5年模型首个有效信号日：2026-03-31
- 当期观察不进入90%分位；未使用扩展窗口或缩短窗口

## 5年主模型事件门

- 有效交易日：101
- 原始事件起点：1
- 间隔20交易日后的独立事件：1
- 具有完整20日结果的独立事件：0
- 事件门状态：`REJECTED_FROZEN_INSUFFICIENT_EVENT_SUPPORT_NO_RESCUE`

| 事件硬门 | 通过 |
|---|---:|
| minimum_eight_independent_complete_events | 否 |
| event_mean_below_nonpressure | 否 |
| event_median_below_nonpressure | 否 |
| minimum_sixty_percent_negative | 否 |
| delete_worst_direction_unchanged | 否 |
| events_span_at_least_two_years | 否 |

## 组合层

事件门未通过，因此未运行5年、3年或双倍成本组合回测；
没有可报告的净夏普率、净超额收益或最大回撤。该状态不能解释为零收益或亏损。

## 治理边界

- 参数营救：禁止
- 仓位映射到真实账户：禁用
- 订单生成：禁用
- 券商连接：禁用
- 即使历史通过，也必须新增252个交易日影子前向后再评估

# 510300 大盘压力传导、卖盘吸收与耗竭机制研究 V1

- 最终状态：`EXPLANATORY_ONLY_SYNCHRONOUS_OR_LAGGING_NO_STRATEGY_PROMOTION_NO_RESCUE`
- 数据合同：`PASS`
- 证据等级：回顾性、历史污染的事件机制发现；不是策略回测。
- 旧路由器：继续冻结拒绝，未改名、未改阈值、未改仓位。

## 四道冻结门

| 门 | 结果 |
|---|---:|
| 领先门 | FAIL |
| 事件门 | FAIL |
| 机制门 | PASS |
| 外部复制门 | PASS |

## 两个预测问题

| 问题 | 事件行 | 主系数 | 90%自助区间 | 留一时期AUC中位数 | 通过 |
|---|---:|---:|---:|---:|---:|
| A 未来10日重大左尾 | 61 | NA | `[None, None]` | NA | 否 |
| B 未来20日持续修复 | 26 | -4.6523 | `[-33.36940547185539, 15.756794689872102]` | 0.1200 | 否 |

## 领先、同步与滞后

| 通道 | 有效事件 | 中位有符号时差 | 领先/同步/滞后 | 冻结分类 | 通过 |
|---|---:|---:|---:|---|---:|
| PRESSURE_SOURCE | 9 | 12.0 | 8/0/1 | `INSUFFICIENT_NONCENSORED_TIMING_EVENTS` | 否 |
| TRANSMISSION | 36 | 7.0 | 27/4/5 | `LEADING` | 是 |
| CROSS_INDEX_ETF | 42 | 16.0 | 41/1/0 | `LEADING` | 是 |
| SAME_INDEX_ETF | 41 | 12.0 | 39/2/0 | `LEADING` | 是 |
| CSI300_COMPONENT_BREADTH | 35 | 9.0 | 23/7/5 | `LEADING` | 是 |
| IF_RAW_BASIS | 41 | 13.0 | 41/0/0 | `LEADING` | 是 |
| H3_DOWNSIDE | 39 | 12.0 | 35/4/0 | `LEADING` | 是 |
| EXHAUSTION_TO_REPAIR | 40 | 0.0 | 16/9/15 | `SYNCHRONOUS` | 否 |

## 事件与来源

- 完整上游压力事件：61，每个事件等权。
- 独立价格损害事件：47。
- 点时耗竭候选事件：26。
- 来源分类：`{'DISCOUNT_RATE': 26, 'LEVERAGE_LIQUIDITY': 22, 'MULTI_SOURCE': 11, 'MIXED_BALANCED': 2}`。
- 回顾性路径形态：`{'MIXED_OR_UNRESOLVED_RETROSPECTIVE': 34, 'FAST_SHOCK_THEN_REPAIR_RETROSPECTIVE': 22, 'PERSISTENT_TRANSMISSION_RETROSPECTIVE': 5}`。
- 完整点时基本面预期来源：`NO_VIEW_INCOMPLETE_CONTRACT`；未以新闻或年份补标签。
- `FAST_SHOCK`与`PERSISTENT_TRANSMISSION`只描述事后路径，不等于已证明的外生冲击或去杠杆因果类型。

## 边界

没有组合净值、净夏普、净超额、最大回撤、仓位、当前信号、Paper、Shadow、订单、券商连接或实盘授权。

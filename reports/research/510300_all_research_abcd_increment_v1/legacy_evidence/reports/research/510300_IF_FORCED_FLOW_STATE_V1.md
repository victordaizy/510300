# 510300 IF强制资金流状态 V1

最终状态：`REJECTED_FROZEN_INSUFFICIENT_INDEPENDENT_EVENT_SUPPORT_NO_RESCUE`

研究阶段：`DISCOVERY_ONLY`；交易资产边界仅为510300或人民币现金。

## 数据门

- 状态：`PASS_WITH_CONSERVATIVE_ONE_TRADING_DAY_AVAILABILITY_LAG`
- IF逐合约：15860行，199个合约，2010-04-16至2026-08-12。
- 未使用连续合约；收盘价与结算价未混用；因同日官方发布时间不能证明，全部IF日数据延迟一个交易日进入信号。

## 机制门

状态：`REJECTED_FROZEN_INSUFFICIENT_INDEPENDENT_EVENT_SUPPORT_NO_RESCUE`

| 事件 | 独立候选 | 完整匹配 | 覆盖年份 | 10日匹配效应 | Holm校正p值 | 通过 |
|---|---:|---:|---:|---:|---:|---:|
| 压力开始 | 10 | 0 | 5 | 无 | 无 | 否 |
| 压力耗竭 | 10 | 0 | 5 | 无 | 无 | 否 |

## 组合层

机制门未全部通过，冻结组合回测未运行；`NOT_ALLOWED`不是零收益，也不是亏损。

任何失败均冻结为不救援；不允许改方向、分位数、期限桶、退出条件，或加入宏观、估值、趋势、期权、北向、PCF/IOPV和广度。

`POSITION_MAPPING_ENABLED=false`，`ORDER_GENERATION_ENABLED=false`，`LIVE_TRADING_AUTHORIZED=false`。

# 510300研究方向切换 V1 最终报告

最终状态：`REJECTED_FROZEN_IF_FORCED_FLOW_MECHANISM_GATE_FAILED_NO_RESCUE`

本轮已按冻结顺序完成宏观分支封存与事后归因、收益解剖、受约束Oracle、预测能力前沿，以及IF基差残差×持仓冲击×耗竭机制门。

## 方向切换结果

- 宏观分支：`POST_MORTEM_ONLY_NO_MODEL_CHANGE`；模型保持拒绝，不允许营救。
- 收益解剖：`COMPLETED_POST_MORTEM_DIAGNOSTIC_ONLY`。
- Oracle有限搜索族最高净夏普率：3.8021。
- 预测能力前沿：562/1728个冻结网格格子通过全部目标门。
- IF机制：`REJECTED_FROZEN_INSUFFICIENT_INDEPENDENT_EVENT_SUPPORT_NO_RESCUE`。

## 授权边界

- 夏普率1.2目标是否已实现：`false`
- 历史收益是否允许解释：`NOT_ALLOWED_FOR_PORTFOLIO`
- Paper/Shadow信号：关闭
- 仓位映射：关闭
- 订单生成：关闭
- 券商连接：关闭
- 实盘交易：未授权

负结果不会通过改窗口、阈值、方向或追加宏观/估值/趋势/期权/广度因子营救。

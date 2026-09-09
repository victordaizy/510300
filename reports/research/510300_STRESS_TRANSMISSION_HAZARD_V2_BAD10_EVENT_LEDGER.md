# 510300 压力传导危险率 V2 BAD10 事件账本

本文件只报告标签事件结构与样本权重，不包含特征、概率、AUC、收益、
夏普、净值、仓位或订单。

## 固定构造

- 正样本按 `[下一开盘, 十日路径末日]` 闭区间的传递重叠合并；
- 每个独立事件内所有正样本训练权重之和为 1；
- 每个非事件风险日训练权重为 1；
- 每个事件保存唯一 split group、事件 Bootstrap block 与日历年 block；
- 当前 split 一律为 `UNASSIGNED_PRE_MODEL_FREEZE`。

## 计数前提

- 独立 BAD10 事件：58
- BAD10 正样本原点：463
- 非事件风险日：2350
- 机制发现事件数前提（至少 30）：PASS
- 完整三系数模型/组合事件数前提（至少 40）：PASS
- 非事件风险日前提（至少 750）：PASS

这些计数只裁决 G1 的事件数量前提。历史成分收益四态尚未完成来源重建，
因此 G0 和完整 G1 均未通过，G2—G7 未运行。

```text
RESEARCH_STATE = DISCOVERY_ONLY
RETURN_EVALUATION = NOT_ALLOWED
MODEL_TRAINING = NOT_ALLOWED_AT_CURRENT_STAGE
PORTFOLIO_EVALUATION = NOT_ALLOWED
POSITION_IMPACT = 0
```

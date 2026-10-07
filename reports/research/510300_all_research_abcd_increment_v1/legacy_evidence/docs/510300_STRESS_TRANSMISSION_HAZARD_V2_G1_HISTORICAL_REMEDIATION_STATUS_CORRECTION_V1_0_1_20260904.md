# 510300 压力传导危险率 V2：G1 历史修复状态纠正 V1.0.1

## 原因

V1 的内存预演发现一个只影响最终状态映射的缺陷：状态生成器仅以“完整 B3 模型门是否通过”选择 `next_allowed_step`。因此在“B2 机制门通过、B3 完整模型门未通过”时，V1 会正确保留 `G1_DATA_AND_EVENTS=PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE`，却把下一步错误写成总失败路径。

## 纠正范围

本纠正不修改 V1 的官方来源、四态账本、M/F/T、样本、事件、门槛、计数、报告、构建收据、重放收据或原状态文件。它只在 V1 独立新进程重放通过后，从重放收据中的原始 G1 门结果生成一个追加式状态文件。

若机制门通过而完整模型门未通过，必须同时记录：

- B2 机制分支：`PASS_MECHANISM_DISCOVERY_PREREQUISITE`；
- B3 完整模型分支：`NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY`；
- 总状态：`PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE`；
- 下一允许步骤沿用冻结 G1 门结果，但本执行不授权该步骤；
- G2、模型、概率、组合、仓位和订单均未运行或生成。

## 停止边界

本纠正不能把 31/40 写成 B3 通过，不能把 B2 通过抹成总失败，也不能借状态纠正运行 G2。任何下一阶段都需要新的明确授权。

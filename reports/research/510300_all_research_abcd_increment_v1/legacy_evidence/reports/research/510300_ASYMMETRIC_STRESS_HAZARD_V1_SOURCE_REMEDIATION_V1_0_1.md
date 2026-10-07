# 510300_ASYMMETRIC_STRESS_HAZARD_V1 来源修复 V1.0.1

## 裁决

- 状态：`NO_VIEW_SOURCE_REMEDIATION_PARTIAL_DR007_BLOCKED`。
- 当前整源阻断通道：`M2`。
- 父级 V1.0.0 结果保持不可变；没有重跑 BAD10，没有修改标签、窗口或阈值。
- 未构造 M/F/T 特征，未读取未来收益或组合指标，未训练模型，未生成仓位、Paper/Shadow、订单或券商动作。

## 已补齐并准入

- 申万官方 PIT 行业：846900 个 member-day，其中 743098 个在当日收盘前可证明可得，103802 个保持 `NO_VIEW`；完整 300 只行业映射的交易日为 54 个。
- 央行 7 天逆回购：归档 2850 篇必要候选公告，正式账本含 2773 条（含一个观察期前锚点）；实际披露利率日 2104 个，实际变化点 25 个。
- 零操作公告未披露利率的数量为 6；均未猜值、未插值、未倒推变更日。
- `F3`、`T2`、`T3` 的整源阻断解除；受行业可得性或成分回报缺口影响的日期仍逐日 `NO_VIEW`。

## 唯一剩余缺口

- DR007：`BLOCKED_LICENSED_DR007_HISTORY_REQUIRED`。
- ChinaMoney 公共端点本次证明的起始日为 2026-08-19，不能覆盖冻结区间 2015-01-05 至 2026-08-14。
- 项目内只有 FDR007；它与 R007、FR007、交易所 R-007 一样不得替代 DR007。
- 需要一份获授权的 DR007 日度加权平均利率导出，以及对应来源证明 JSON。导入器只接受 `DR007`、百分数口径，并统一采用“利率日后的下一交易日开盘可得”这一保守时钟。

## 允许的后续动作

`IMPORT_LICENSED_DR007_WEIGHTED_AVERAGE_DAILY_EXPORT_WITH_PROVENANCE`

在 DR007 通过前，来源仍未全部准入，禁止特征构造、G2、收益评估和任何交易动作。

## 权限边界

- `RESEARCH_STATE=DISCOVERY_ONLY`
- `MODEL_POSITION_TARGET=UNSET`
- `ORDER_AUTHORIZATION=NOT_AUTHORIZED`
- `POSITION_IMPACT=0`
- `RETURN_EVALUATION=NOT_ALLOWED`

# 510300_ASYMMETRIC_STRESS_HAZARD_V1 点时来源准入 V1

## 裁决

- 状态：`NO_VIEW_SOURCE_ADMISSION_FAILED`。
- 研究处置：`NO_VIEW`。
- 必需来源没有全部通过，因此没有构造 M、F、T 特征值，也没有进入 G2。
- 未训练模型，未选择阈值，未读取收益、夏普或回撤，未生成仓位、Paper/Shadow、订单或券商动作。
- 来源冻结 manifest：`39998b05bbc1416a157a8bf50ffd9d1f2b53efbd18d529f7c9f62ebc8a0935aa`。

## 通道准入

- `M1`：`PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW`
- `M2`：`BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED`
- `M3`：`PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW`
- `F1`：`PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW`
- `F2`：`PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW`
- `F3`：`BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED`
- `T1`：`PASS_SOURCE_ADMISSION_WITH_DATE_LEVEL_NO_VIEW`
- `T2`：`BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED`
- `T3`：`BLOCKED_REQUIRED_SOURCE_ADMISSION_FAILED`

## 整源阻断

- DR007：`BLOCKED_NO_FROZEN_DR007_DAILY_SERIES`。本地没有冻结历史序列，FDR007、R007、FR007 或交易所 R-007 均不得替代。
- 央行 7 天逆回购政策利率：`BLOCKED_NO_FROZEN_PBOC_7D_REVERSE_REPO_POLICY_RATE_SCHEDULE`。不得插值或倒推出变更日。
- 申万一级行业区间：`BLOCKED_NO_INDEPENDENT_VERSION_PROVEN_PIT_INDUSTRY_PROVENANCE`。现有代理 payload 缺少独立可验证的版本来源。

## 已准入来源与逐日 NO_VIEW 边界

- 中证 300 官方 PE：3752 行，2011-06-28 至 2026-08-12；原始 payload 逐行一致。末端 2 个交易日缺值，且因未归档精确发布时间只可从下一交易日开盘起使用。
- 中国国债 10 年收益率：2509 行，2016-08-12 至 2026-08-25；冻结观察窗前段 394 个交易日只能 `NO_VIEW`，原始 payload 未归档且采用下一交易日开盘可得时钟。
- 社融存量同比首发 vintage：127 个连续月份，2016-01 至 2026-07；未使用修订值。构造三个月变化所需的前置月份只能产生领先期 `NO_VIEW`。
- 官方 PIT 成分：2823 个交易日、每天严格 300 只；本框架不需要且没有使用未准入的历史权重。
- 成分股总回报面板：在 1611 个重叠交易日中，完整 300 只的日期为 1561 个；50 个日期缺少至少一只，合计缺少 263 个 member-day，最少仅 272 只。这些日期均为 `NO_VIEW`，不插值、不用当前成分回填。
- 行业区间结构覆盖：1611 个交易日中完整覆盖 974 个，缺口日期 637 个，合计缺少 1607 个 member-day；即使逐日缺口可按 `NO_VIEW` 处理，独立版本来源仍未通过，故 F3/T2/T3 整体阻断。

## 允许的后续动作

`REMEDIATE_M2_DR007_AND_7D_REVERSE_REPO_POLICY_RATE_AND_F3_PIT_INDUSTRY_PROVENANCE_VIA_VERSIONED_SOURCE_CONTRACT`

后续只能用新的版本化来源契约补齐上述来源证据；不得修改 BAD10 标签、窗口、阈值，不得用替代利率、当前成分或当前行业分类救援。

## 权限边界

- `RESEARCH_STATE=DISCOVERY_ONLY`
- `MODEL_POSITION_TARGET=UNSET`
- `ORDER_AUTHORIZATION=NOT_AUTHORIZED`
- `ACTUAL_HOLDINGS_STATE=UNKNOWN_OUT_OF_SCOPE`
- `POSITION_IMPACT=0`
- `NO_VIEW` 是研究终态，不是现金仓位建议，也不描述用户实际持仓。

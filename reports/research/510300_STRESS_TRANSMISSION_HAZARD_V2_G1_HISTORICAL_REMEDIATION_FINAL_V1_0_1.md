# 510300 压力传导危险率 V2：G1 历史证据修复最终裁决 V1.0.1

## 一句话裁决

`PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE`

全量、结果盲的交易所官方停复牌证据修复已经完成正式构建和独立新进程重放。B2 机制发现的 G1 数据与事件前置门槛真实通过；B3 完整三系数模型仍因可识别事件不足而保持 `NO_VIEW`。G2 未运行。

## 修复前后

| 指标 | 修复前 | 修复后 | 冻结门槛 | 裁决 |
| --- | ---: | ---: | ---: | --- |
| B2 可识别独立事件 | 23 | 31 | 30 | `PASS` |
| B2 非事件风险日 | 767 | 1,102 | 750 | `PASS` |
| B3 可识别独立事件 | 23 | 31 | 40 | `NO_VIEW` |
| B3 非事件风险日 | 767 | 1,102 | 750 | `PASS` |

冻结的压力事件总数始终为 58，没有增加、删除或重新挑选事件。

## 全量、结果盲证据修复

- 修复目标：父版本全部 `source_observed=false` 点时成员股日，共 15,340 行、324 个证券。
- 官方区间记录：14,205 条，其中上交所 6,186 条、深交所 8,019 条。
- 命中官方停复牌区间：14,943 个目标成员股日。
- 严格提升为 `OFFICIAL_SUSPENSION`：14,771 行。
- 公司行动候选阻断：58 行。
- 缺少前收盘价阻断：116 行。
- 未命中官方整日区间：397 行。
- 提升集合外父级行变化：0。
- 采集和逐行修复未读取事件 ID 或标签；BAD10 最小必要列只在 M/F/T 全量重建后读取。

其中 58 与 116 是可重叠的诊断标志，不能与 397 直接相加。按互斥的最终逐行决策统计为：14,771 行可提升、58 行公司行动阻断、114 行缺少前收盘阻断、397 行未命中，合计 15,340 行。

官方来源：

- 上交所停复牌信息：<https://www.sse.com.cn/disclosure/dealinstruc/suspension/stock/>
- 深交所市场统计月报：<https://www.szse.cn/market/periodical/month/index.html>

## 年代分布

| 冻结年代 | 全部事件 | B2 可识别 | B3 可识别 |
| --- | ---: | ---: | ---: |
| 2015-2017 | 10 | 0 | 0 |
| 2018-2020 | 15 | 5 | 5 |
| 2021-2023 | 22 | 19 | 19 |
| 2024-2026 | 11 | 7 | 7 |

2015-2017 仍为 0/10，说明本次没有为了年代覆盖或门槛结果进行定向补数。

## 独立重放与测试

- 正式构建：`PASS_G1_REMEDIATION_BUILD_PENDING_FRESH_PROCESS_REPLAY`。
- 独立新进程重放：`PASS_FRESH_PROCESS_FULL_SCOPE_REMEDIATION_AND_G1_REPLAY`。
- 9 张构建产出表全部通过持久化文件身份检查和精确内容比较；比较允许 Parquet 读回产生的 dtype 表示差异，但不允许单元格值漂移。
- V1.0.1 状态纠正：`PASS_APPEND_ONLY_STATUS_CORRECTION_WITH_NO_DATA_OR_GATE_CHANGE`。
- 组合回归测试：41 项全部通过。

V1 原始状态文件保留。V1.0.1 只纠正“B2-only PASS 的下一允许步骤”映射，不改变数据、特征、样本、事件、门槛或裁决。

## 最终权限边界

- `B2_MECHANISM_G1=PASS_MECHANISM_DISCOVERY_PREREQUISITE`
- `B3_FULL_MODEL_G1=NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY`
- `G2=NOT_RUN_REQUIRES_NEW_EXPLICIT_AUTHORIZATION`
- `model_trained=false`
- `current_market_probability=NOT_GENERATED`
- `position_impact=0`
- `next_allowed_step=FREEZE_B2_VS_B1_EVENT_LEVEL_PREDICTION_EXECUTION`
- `next_step_authorized_in_this_execution=false`

本裁决只证明 B2 机制发现所需的数据与事件可识别性门槛已经满足，不证明模型有效、策略有效、存在可交易信号或允许任何 Paper/Shadow、券商、仓位、订单和实盘动作。

## 权威证据

- 机器可读最终状态：`reports/research/510300_stress_transmission_hazard_v2_g1_historical_remediation_status_v1_0_1.json`
- 独立重放收据：`reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_replay_v1.json`
- 状态纠正收据：`reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_status_correction_v1_0_1.json`
- 正式构建收据：`reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_build_v1.json`
- 官方采集收据：`data/raw/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/collection_receipt.json`
- 全量逐成员股日证据：`data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/all_missing_member_day_official_evidence.parquet`

> 2026-09-06 固定截距静态底仓与择时归因完成：`510300_CSP_V1_STATIC_VS_TIMING_ATTRIBUTION_V1`，终态 `COMPLETED_DIAGNOSTIC_ONLY_NO_PROMOTION`。新增4个账户、4条事后理想路径、一次20日×2000次联合区块统计，0拟合/网格/下载。C0/C1两档成本路径完全相同，机械增量0；FULL相对C1基础净年化差+0.4538个百分点，95%区间[-0.2162,1.1690]，同波动理想差+0.2088个百分点，区间亦跨零。本次未确认稳定状态择时优势；原V1拒绝与1.2目标不变，不晋升对照、不自动下一轮调参。见[归因报告](reports/research/510300_csp_v1_static_vs_timing_attribution_v1/研究报告.md)及[封卷结论](reports/research/510300_csp_v1_static_vs_timing_attribution_v1/封卷结论与审阅重点.md)。仓位影响0。

> 2026-09-05 连续条件评分直接持仓独立研究完成：`510300_CONDITIONAL_SCORE_POLICY_V1`。54次内层拟合、2次2015—2019重估，模型冻结后完成2020-01-02至2026-08-14的1,604日连续账户。完整模型基础/压力净夏普0.353/0.351，复合年化1.054%/1.045%，均未达到1.2且收益落后同口径买入持有；状态 `REJECTED_FROZEN_CONDITIONAL_SCORE_POLICY_V1`。平均模拟暴露14.49%，仅7个成交日，修复项净增量很小。以[本轮报告](reports/research/510300_conditional_score_policy_v1/研究报告.md)、[三类交付导航](reports/research/510300_conditional_score_policy_v1/交付导航.md)及[完成回执](reports/research/510300_conditional_score_policy_v1/execution_receipt.json)为准。此项来自本轮完整方案的独立授权，既有失败分支保持原裁决；不拟合未来模型、不晋升对照，仓位影响0。

> 2026-09-05 日内—隔夜条件增量一次性历史研究完成：242个训练原点、320个成熟评价原点；2020-01-02至2026-08-14开盘连续账户。M1基础成本净累计-4.32%、净夏普0.018；压力成本净累计+9.29%、净夏普0.167；基础相对M0无净增量，D20训练/评价条件方向反转。裁决 `STOP_REPRESENTATION_NO_PARAMETER_RESCUE`，未拟合最终全历史模型。以 [完整结果](reports/research/510300_intraday_overnight_increment_v1/REPORT.md)、[比较表](reports/research/510300_intraday_overnight_increment_v1/comparison.csv) 及 [交付说明](reports/research/510300_intraday_overnight_increment_v1/DELIVERY_NOTE.md) 为准。本轮 [V5有限历史研究授权](config/510300_research_authority_v5.json) 的一次机会已用完，旧裁决保留；模型ABSTAIN、目标UNSET、真实持仓未知、仓位影响0。

> 2026-09-05 最新结果：NBS 已按用户明确批准的固定五分钟用途合同通过 G0/G1，并完成一次性真实 G2。G2 失败：NBS B1 的严格前序 MSE 比 B0 高 7.82%，三个时代均未改善，两个安慰剂优越性门也未通过。NBS 家族冻结，G3/G4 未运行，研究状态为 `STRICT_FORWARD_ONLY`。以 [V4 当前研究权限](config/510300_research_authority_v4.json) 和 [本次完整报告](docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md) 为准；下文旧 G0 阻断和 V3 状态仅为历史。DSV5 B1 只读观察器继续，原政策保持暂停。仓位影响为 0。

> 2026-09-05 NBS 数据续行：77/77 异常月份重取完成，373,791 根重取记录及完整候选数据与原版本一致；G0 仍未通过。G2 实现已冻结并由入口阻断，真实事件标签为 0。详见 [本轮报告](<C:/Users/戴周阳/Documents/New project 8/docs/510300_NBS_V2_REFETCH_AND_G2_GATE_RESULT_20260905.md>)。

# 510300 研究权威状态

2026-09-05 最新研究权限：以 [V3 权威状态](config/510300_research_authority_v3.json) 和 [NBS V2 最终交接](reports/research/510300_nbs_1000_negative_information_drift_v2/RESEARCH_SUMMARY.md) 为准。510300 已切换 `STRICT_FORWARD_ONLY`；DSV5 政策归档、B1 仓位映射关闭；NBS V2 因分钟 VWAP 硬门失败而在收益读取前永久关闭。B1 仅保留独立预测观察。仓位影响为 0，实盘未授权。下文保留此前状态历史。

截至：2026-08-31

机器可读权威索引：`reports/research/510300_authoritative_research_status_v1.json`

## 当前仓位与动作语义

```text
RESEARCH_STATUS = DISCOVERY_ONLY
MODEL_POSITION_TARGET = UNSET
MODEL_ACTION = ABSTAIN
NEW_ORDER_ALLOWED = false
EXISTING_HOLDINGS_OVERRIDE_ALLOWED = false
PAPER_SIGNAL_ALLOWED = false
SHADOW_SIGNAL_ALLOWED = false
LIVE_TRADING_AUTHORIZED = false
```

研究系统当前不产生任何仓位目标或订单。`ABSTAIN` 不等于目标仓位为 100% 现金；它既不要求卖出已有 510300，也不要求继续持有或新买入。用户实际持仓保持未知，不受研究结果覆盖。

## 研究对象状态

| 研究对象 | 权威状态 | 交易含义 |
|---|---|---|
| 宏观压力规避 | `REJECTED_FROZEN_NO_RESCUE` | 不生成仓位 |
| 510300 收益结构解剖 | `DIAGNOSTIC_COMPLETE_NON_TRADABLE` | 仅归因 |
| 受约束 Oracle | `DIAGNOSTIC_COMPLETE_HINDSIGHT_ONLY` | 仅可达性诊断 |
| 夏普目标能力前沿 | `DIAGNOSTIC_COMPLETE_SIMULATION_ONLY` | 仅能力需求映射 |
| IF 强制资金流 | `REJECTED_FROZEN_INSUFFICIENT_PREVALENCE` | 未检验有效性，不可救援 |
| 聚合成交额可见性 | `REJECTED_FROZEN_AGGREGATE_TURNOVER_VISIBILITY_PREDICTIVE_GATE_FAILED_NO_RESCUE` | 预测门失败，不生成仓位 |
| 沪深300横截面收益离散度 | `REJECTED_FROZEN_CSI300_RETURN_DISPERSION_VOLATILITY_GATE_FAILED_NO_RESCUE` | 波动预测门失败，不生成仓位 |
| 沪深300 PCA 吸收率择时 | `REJECTED_PRE_FREEZE_DATA_COVERAGE_GATE_FAILED_NO_RESCUE` | 上市历史覆盖门失败，禁止收益评价 |
| 多尺度非线性趋势—反转 | `REJECTED_FROZEN_NONLINEAR_TREND_MECHANISM_GATE_FAILED_NO_RESCUE` | 论文机制迁移门失败，禁止组合收益评价 |
| 技术因子准入裁决 | `NO_ADMISSIBLE_UNUSED_RETROSPECTIVE_TECHNICAL_CANDIDATE_FORWARD_EVIDENCE_ONLY` | 当前冻结候选集不再追加历史技术回测，只等待独立前向证据 |
| 当前模型仓位输出 | `ABSTAIN_POSITION_UNSET` | 不覆盖现有持仓 |
| 510300 期权左尾候选 | `BLOCKED_NO_COMPLETE_POINT_IN_TIME_OPTION_CHAIN` | 数据门失败，不得建立预测模型 |
| PIT 盈利信息扩散备线 | `BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS` | 点时成分已修复；权重版本、事实A3、人工复核和事实数量仍阻断正式建模 |
| PIT 盈利信息扩散低置信度代理诊断 | `DIAGNOSTIC_COMPLETED_UNRELIABLE_INPUTS_NOT_ADMISSIBLE` | 用户授权强行继续后的隔离诊断；所有预测硬门失败，不是仓位或收益结果 |

## 当前证据边界

- `510300_DIRECTION_SWITCH_V1` 的最终状态是 `REJECTED_FROZEN_IF_FORCED_FLOW_MECHANISM_GATE_FAILED_NO_RESCUE`。
- IF 候选只因预注册事件数量不足而停止：`EFFICACY_TESTED=false`。
- IF 组合评价未运行；`NOT_ALLOWED` 不是零收益，也不是亏损。
- 不允许降低事件门槛、建立连续 IF 评分或增加过滤器来重开该候选。
- 聚合成交额可见性候选形成139个完整预测观测；冻结预测门失败，组合评价未运行，`NET_SHARPE=NOT_COMPUTED`。
- 沪深300横截面收益离散度候选虽有正向波动关系，但冻结的相对预测改进门未通过；组合评价未运行，`NET_SHARPE=NOT_COMPUTED`。
- PCA 吸收率候选已完成714只相关点时成员价格处理，但2015-12-30至2022-08-10有986个交易日不满足冻结的285只且95%完整500日历史覆盖门；最低277/300。该失败来自新成员上市历史不足，并非下载缺失；未创建冻结清单、未读取市场结果、未运行组合评价。
- 多尺度非线性趋势—反转候选按论文固定的10个二进制日频尺度、指数线性趋势权重、正负2.5截断及论文系数方向一次性迁移；2015-01-05至2026-08-14有2,822个有效观测。全样本线性项 Newey-West t=0.527、三次项 t=-1.063，固定论文评分斜率为-1.782且 t=-0.327，三个90%区间均跨零。机制硬门失败，组合评价未运行，`RETURN_EVALUATION=NOT_ALLOWED`、`NET_SHARPE=NOT_COMPUTED`，禁止更换尺度、权重、系数比例或方向救援。
- 技术因子准入裁决已完成：当前冻结候选集中，未使用且可直接进入历史组合评价的技术候选为0。高成交量趋势已在Round2明确拒绝，成交量冲击仅可做前向观察；继续更换窗口、阈值、方向或拼接失败因子均属于结果后救援。唯一可产生新增独立证据的路径为冻结的PCF/IOPV前向采集；截至本次裁决为2/20个完整质量日，下一合法窗口为2026-09-01 09:25（Asia/Shanghai）。
- `510300_OPTION_CHAIN_FEASIBILITY_V1` 已在 `FUTURE_DATA_READS=0` 下完成；有效曲面日为 271/1611（16.82%），并缺少完整调整公告账本及历史日行情点时/授权证据。
- 期权左尾候选不得读取未来 10 日收益、训练预测模型或运行组合回测。
- 唯一备线 `510300_PIT_EARNINGS_INFORMATION_DIFFUSION_10D_RISK_V1` 已在期权主线正式阻断后顺序启动，并保持 `MARKET_PRICE_READS=0`、`FUTURE_DATA_READS=0`。
- 官方公告元数据与 PDF 门通过：88,677 条目标公告、88,667 份可用官方 PDF，覆盖率 99.9887%。
- 原冻结预检的成员失败证据继续保留：候选相关的 1,045 个信息可用日中，旧重建口径有 258 日不是 300（210 日为 299、48 日为 301）。后续来源修复没有改写或救援该预检，而是建立了独立的官方公告链与锚点协议。
- 点时成分来源修复已通过：2015-01-05 至 2026-08-14 共 2,823 个开市日、846,900 行，每日恰好 300 只；2015 年 5 轮官方调样、3 个官方历史锚点、2016H1 交接以及 2016-08 至 2026-07 的 120 个月末快照均为零集合差异。
- 历史权重仍未准入：现有 120 期月度数值结构合格，但公开留存只覆盖少量官方历史版本锚点，兼容代理凭据已过期，无法为每一期证明版本或 as-of 来源。因此候选权威状态保持 `BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS`，当前剩余数据障碍为 `BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS`。
- V1.0.3 结果盲数据补齐已经完成：官方逐日成分 2,823 日；月度权重 120 期结构合格但版本不可证明；官方首次公开完整核心利润事实全市场 852 条、历史沪深300成员 62 条、严格前序权重有效 60 条，2021 年以来仅 46 条且其中严格负利润区间 10 条。自动事实 A3 为 900/1,624（55.42%），低于 90% 硬门；双人独立复核为 0 对；没有任何一年达到至少 20 条有效事实。
- 用户明确要求不论下一轮数据可靠性都继续后，另行冻结并运行了 `DIAGNOSTIC_ONLY` 代理预测。2021-01-04 至 2026-07-30 共 1,350 个样本外观测、23 次季度重估；23 次校准均无法同时满足 75% 召回和 20% 误报。合并观察为召回 42.70%、误报 39.78%、精确率 55.02%、相对价格基线 Brier Skill -9.04%、损失加权召回 37.63%、关键上涨遗漏 38.20%、单年最大往返 14 次；2021—2026 每一年 Brier 增量都为负。
- 该代理诊断使用了预先标注为不可靠的权重与稀疏事实，只回答“强行继续会看到什么”。它不改变正式候选状态，不授权组合回测、仓位映射、Paper/Shadow、订单或实盘。正式备线仍不得用代理结果救援。
- 旧 V1.0.2 清单冻结后，上游全量事实任务原地更新了同一路径 PDF 清单；当前字节哈希为 `632b7f...`，旧清单要求 `49465b...`，且旧字节副本未留存。因此旧清单重放测试保留为显式失败，未静默更新旧哈希；当前 V1.0.3 清单及成分/代理测试为 14 项通过。
- 独立公告事实管线当前为 `PAUSED_BY_USER`；已有检查点保留，监督器和定期元数据采集进程均已停止。本次成分来源修复没有停止、重启或改写该管线。

仓位语义的正式补充说明见：`reports/research/510300_DIRECTION_SWITCH_V1_POSITION_SEMANTICS_ADDENDUM.md`。

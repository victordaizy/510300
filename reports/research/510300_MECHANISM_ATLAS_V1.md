# 510300 机制图谱 V1

- 状态：`MECHANISM_ATLAS_DISCOVERY_COMPLETE_NO_STRATEGY_AUTHORIZATION`
- 来源分支：`REJECTED_FROZEN_MACD_BREADTH_DOWNSIDE_PREFLIGHT_V1_0_1_NO_RESCUE`（继续冻结拒绝）
- 共同窗口：`2021-08-12—2026-08-12`
- 状态面板：`1211`个交易日、`120`个字段
- 交易域：仅`510300.SH`与人民币现金；本报告不生成策略净值、仓位或订单。

## 六病例与失败对照

- 病例：`6`；不放回失败对照：`18`。
- 最大匹配距离：`2.936`；最大绝对标准化均值差：`0.678`。
- 标准五环传导顺序完整出现的病例：`5/6`。
- 失败对照按冻结十日结果不为正选择；下表差异只用于尸检，不能解释为因果或样本外收益。

| 期限 | 病例均值 | 对照均值 | 差异 |
|---:|---:|---:|---:|
| 5日 | 0.08% | -1.21% | 1.28% |
| 10日 | 1.03% | -2.35% | 3.37% |
| 20日 | -0.16% | -1.64% | 1.48% |

## 连续条件响应分类

| 因子/交互 | 主要期限 | 全样本系数 | HAC p值 | Holm p值 | 早/晚方向一致 | 当前状态 |
|---|---:|---:|---:|---:|---|---|
| MACD_CONVERSION | 10 | 0.000773 | 0.6954 | NA | `false` | `DESCRIPTIVE_ONLY` |
| BREADTH_DIFFUSION | 20 | 0.002428 | 0.4466 | NA | `false` | `DESCRIPTIVE_ONLY` |
| DOWNSIDE_PRESSURE | 10 | -0.004394 | 0.0183 | NA | `true` | `STRUCTURAL` |
| MACD_X_BREADTH | 10 | 0.002238 | 0.0934 | 0.5605 | `true` | `DESCRIPTIVE_ONLY` |
| MACD_X_DOWNSIDE | 10 | -0.000149 | 0.9377 | 1.0000 | `true` | `DESCRIPTIVE_ONLY` |
| IF_RAW_ANNUALIZED_BASIS | 10 | NA | NA | NA | `false` | `DESCRIPTIVE_ONLY` |
| MACRO_SLOW_PRIOR | 60 | NA | NA | NA | `false` | `DESCRIPTIVE_ONLY` |
| INTRADAY_ETF_IF_IOPV_PRESSURE | 0 | NA | NA | NA | `false` | `DATA_BLOCKED` |

## 固定数据缺口

- `IF_FAIR_BASIS`：缺少完整点时carry，现有字段只能称原始年化基差。
- `OPTION_SKEW_TERM_STRUCTURE`：V1未准入，不能用短覆盖或失败分支补洞。
- `PCF_IOPV_HISTORY`：质量日不足，不能覆盖六病例和长期对照。
- `CSI300_EARNINGS_BREADTH`：点时数据契约缺失。
- `H4_INTRADAY_TEMPORARY_PRESSURE`：`NO_VIEW_DATA_CONTRACT_FAILED`。

## 裁决

- 组合收益评估：`NOT_ALLOWED`
- 组合回测：`NOT_RUN`
- 净夏普率：`NOT_COMPUTED`
- 累计净超额：`NOT_COMPUTED`
- 夏普1.2与累计净超额20%：均未在本版本评估，不能声称达到。
- 若未来有机制通过，必须建立新的冻结策略ID，并完成严格样本外与至少252个新Shadow交易日。
- Paper、Shadow仓位映射、订单、券商连接和实盘：全部关闭。

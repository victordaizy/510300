# 510300 技术因子准入裁决 V1

- 截至：2026-08-31 13:12（Asia/Shanghai）
- 状态：`NO_ADMISSIBLE_UNUSED_RETROSPECTIVE_TECHNICAL_CANDIDATE_FORWARD_EVIDENCE_ONLY`
- 交易域：仅 `510300.SH` 与人民币现金，多头、无杠杆、无做空、无衍生品。
- 目标：净夏普率 1.2；当前未达到，也没有被允许计算但尚未披露的净夏普。

## 裁决

在当前冻结候选集、点时数据合同和“不救援”规则下，尚未使用且可直接进入历史组合评价的技术候选数为 **0**。

这不等于断言技术分析永远无效；它表示继续在同一历史样本上更换窗口、阈值、方向或把失败因子重新组合，已经不能提供独立证据，只会扩大结果后选择偏差。

## 已裁决的主要技术路径

| 路径 | 权威结果 | 夏普含义 |
|---|---|---|
| MACD 收敛 × 宽度 × 下行风险 | 冻结预检拒绝，禁止救援 | `NOT_COMPUTED` |
| 多尺度非线性趋势—反转 | 2,822 个观测；线性 t=0.527、三次项 t=-1.063、固定论文评分斜率为负 | 机制门失败，`NOT_COMPUTED` |
| 下行风险预算 | 预测门失败，禁止救援 | `NOT_COMPUTED` |
| 日内二元趋势家族 | 13 个固定候选，硬通过 0，正压力 Shadow 0 | 家族拒绝 |
| Donchian 55/20 | 历史筛选失败 | 小账户基准成本夏普 0.435，压力成本 0.412 |
| 月末流动性漂移 | 历史硬门失败 | 基准成本夏普 0.270，压力成本 0.174 |
| 隔夜吸收 | 在 Bootstrap/HAC/组合回测前停止 | `NOT_COMPUTED` |
| 第二轮价量确认 | `ETF_HIGH_VOLUME_TREND_20D` 已拒绝；`ETF_VOLUME_SHOCK_20D` 仅前向观察 | 无历史策略候选 |
| 第三轮点时宽度 | 全研究 40 个日线假设；本轮 12 个，合格因子 0 | 无策略候选 |
| 聚合成交额、横截面离散度、PCA 吸收率 | 分别为预测门失败、预测改进门失败、数据覆盖门失败 | 均未获准组合评价 |

## 唯一合规的新增证据路径

`PRIMARY_MARKET_PCF_IOPV` 仍处于前向采集阶段：当前 4 个观察交易日、2 个完整质量日，首道质量门为 20 日。

- 2026-08-28 直接采集：`EXTERNAL_FREE_SOURCE_FAILED`，退出码 3，不计质量日，也不允许同日重试。
- 严格 TLS 路由修复的只读探针已通过三个官方提供方；探针不落盘、不计质量日。
- 2026-08-31 12:39 编排：`MISSED_START_WINDOW`，退出码 2，按冻结规则不补跑。
- Windows 任务当前显示 `Ready`，下一合法触发为 **2026-09-01 09:25（Asia/Shanghai）**。调度状态不是研究成功证据，届时只认新的同日原子收据。

## 当前目标状态

```text
HISTORICAL_NET_SHARPE_1_2 = NOT_ACHIEVED
VERIFIED_FORWARD_TARGET = NOT_ACHIEVED
GOAL_ACHIEVED = false
PAPER_SIGNAL_ALLOWED = false
SHADOW_SIGNAL_ALLOWED = false
ORDER_GENERATION_ENABLED = false
LIVE_TRADING_ENABLED = false
```

机器可读裁决：`reports/research/510300_technical_factor_eligibility_decision_v1.json`

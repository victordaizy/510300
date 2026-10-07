# 510300 情景型 Alpha 库：第二批资源与状态裁决

## 权威结论

第一批发现阶段完成，但没有形成可交易策略。一级市场分支继续只建设前向数据；开盘折价分支只保留被动日志；盘后离岸信息传导成为当前最高优先级的纯前向观察；聚合 OI V2 的唯一研究次数已经消耗，并以特征构造失败关闭整个公开日频衍生品压力代理族。

当前仍为：`CURRENT_VALIDATED_HIGH_SHARPE_STRATEGY=NONE`、`CURRENT_HOLDING_ROUTE=CASH_CNY`、`POSITION_IMPACT=0`、`LIVE_TRADING_AUTHORIZED=false`。

## 衍生品压力 V2：接受失败

- 科学状态：`NO_VIEW_FEATURE_CONSTRUCTION_FAILED`
- 资源状态：`CLOSED_PUBLIC_DAILY_DERIVATIVE_PRESSURE_PROXY_FAMILY_AFTER_ONE_SHOT_V2`
- 收益评价：`NOT_ALLOWED_PRESSURE_PERCENTILE_NOT_CONSTRUCTED`
- 正式运行：`1/1`，已永久消耗
- 特征行：2,427
- 聚合 OI 非空行：2,427；聚合 OI 变化非空行：2,426
- Pressure 非空行：38；prior-252 Pressure 95% 门非空行：0
- 候选/成熟事件：0/0；这里的零不构成经济结论

聚合 OI 已修复 V1 的近月移仓缺口，但期权偏度与 VRP 的共同历史仍只产生 38 个完整 Pressure 值，无法满足冻结的 252 个有效历史值要求。程序因此没有构造候选，也没有读取入场或退出收益。不得降低窗口、改权重、改代理，或建立 V3 及以后版本。

## 盘后离岸信息传导 V1

- 冻结状态：`FROZEN_FORWARD_ONLY_ZERO_POSITION`
- 当前科学状态：`NO_VIEW_INSUFFICIENT_MATURE_OBSERVATIONS`
- 权威前向起点规则：`2026-09-01_OR_FIRST_LATER_COMPLIANT_SSE_TRADING_DAY`
- 当前权威观测/成熟观测：0/0；`beta_LCB=null`
- 信号/零仓位意向/实际成交：0/0/0
- 当前订单授权数量：0；实盘成交假设：禁止

四套账已经空账初始化。只有先积累 60 个合规成熟日并证明预先冻结的 `beta_LCB>0`，且当日 `beta_LCB*r_post>=42bp`，才可能写入零仓位研究意向。真实 100 份测试仍需单独授权。

## 一级市场数据成熟度账

- 父策略：`NO_VIEW_DATA_CONTRACT_FAILED`，冻结协议未改
- 资源状态：`FORWARD_DATA_BUILD_ONLY_NO_RETURN_VIEW`
- 7 月 6 日至 8 月 31 日格式准入诊断：4 行
- 权威前向行：0
- 严格完整日：0/120
- 收益读取：`FORBIDDEN`

历史候选输入没有合规的实际公开时刻、原始文档/修订/拆分证据和一档买卖报价覆盖，因此不能计为严格完整日。达到 120 日也只获得另行预冻结收益研究的申请资格。

## 其他分支

- `510300_OPENING_DISCOUNT_RECOVERY_V1`：科学状态保持 `RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS`；资源状态为 `PASSIVE_LOG_ONLY_NO_ACTIVE_ENGINEERING`。已建立空 CSV 账头，但没有新增引擎或调度器，规则不变。
- `510300_OPTION_ORDER_IMBALANCE_FULL_V1`：`BLOCKED_FREE_DATA_CANNOT_IDENTIFY_AGGRESSOR_SIDE`，资源为 0。
- 20 日状态、Atlas、HMM 与趋势路由：保持关闭，资源为 0。

## 资源账

V2 运行前计划为离岸 50%、衍生品 V2 30%、一级市场数据 20%。V2 已关闭后，当前资源记录为离岸 50%、一级市场数据 20%、未分配 30%；不得自动把释放资源转给其他分支。

## 时点边界

2026-07-06 至 2026-08-31 只用于格式与时间戳准入，没有读取新盘后分支的策略收益。2026-09-01 已到，但当前时点尚未形成当日合规收盘后窗口或一级市场完整回执，因此权威前向起点仍为 `null`，不得提前写入，也不得日后回填遗漏日。

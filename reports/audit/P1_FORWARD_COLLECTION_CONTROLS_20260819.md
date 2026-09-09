# P1 三条前瞻链执行审计（2026-08-19）

## PCF/IOPV

- 当前状态：`COLLECTING_NOT_ELIGIBLE`。
- 完整质量日：`2/20`；观察日4个，逐日去重快照数为1、1、154、234。
- 统计聚类单位固定为交易日，不把日内快照当独立统计样本。
- 完整日同时要求：至少120个去重快照、PCF日期唯一对应、交易所时间戳与交易日一致、检索时间带明确时区、IOPV陈旧度在[-5,180]秒内、IOPV为正。
- 20日只允许质量审计；40日才允许冻结发现期特征；80日才允许第一次未见段研究评价；120日才允许复制段评价。
- 当前特征冻结、收益评价、仓位映射、订单和实盘全部禁止。

权威状态文件：`reports/data_quality/510300_primary_market_readiness.json`。

## T_ONLY

- 当前状态：`COLLECTING_FORWARD_NOT_STARTED`，0个前瞻交易日、0个闭合周期。
- 新权威运行状态只展示账本连续性、哈希链、信号/有效时点、下一交易日开盘影子执行、成本完整性、闭合周期计数和成熟度计数。
- 每日事件记录采用前序哈希链；既有事件不得改写，单次运行不得事后补写多个交易日。
- 252日/3周期只定义运行成熟度，不是收益通过门。
- 正式功效计划固定为最多1,500个交易日、至少20个闭合周期、5个百分点MDE、20日块Bootstrap，并且只评价一次。
- 正式绩效字段保持 `BLINDED_UNTIL_FIXED_FORMAL_EVALUATION`；当前报告中为 `null`。

权威状态文件：`reports/forward/t_only_forward_integrity_status.json`。不可变事件账本：`paper/t_only_forward_integrity_v1/daily_event_ledger.jsonl`。

## 行业预期差

- 当前状态：`COLLECTING_FORWARD`。
- `origin_cluster` 固定等于预测日期；当前1个原点簇、0个成熟簇、0个非重叠60日块。
- 同一日期的行业行仅用于横截面描述，不增加独立时间样本数。
- 20个成熟原点簇且至少4个非重叠60日块才允许校准；40个成熟原点簇且至少8个非重叠60日块才允许模型比较。
- 原指数观点保持 `NO_VIEW`，任何结果都不得事后升级。
- 仓位映射、订单、券商连接和实盘全部关闭。

权威状态文件：`reports/forward/industry_expectation_gap_v1_evaluation/origin_cluster_maturity_status.json`。

## 验证

定向及相关回归测试共38项，全部通过。没有启动新历史候选，没有修改全局研究注册表、V3、Round5联合新鲜度或桶2永久冻结材料。


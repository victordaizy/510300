# 510300 第一阶段基础特征研究（旧探索产物，未获批准）

> 治理状态：`EXPLORATORY_NOT_APPROVED`。本报告生成于正式Target与因子讨论之前，不代表项目选择；当前正式结果以`registered_factor_research.md`为准。

## 研究约束

- 三类特征并列研究：估值、趋势、波动率；未构造综合分数。
- 所有信号以当日收盘为信息截止点；只能在下一交易日开盘或更晚入场。
- 1日标签只用于信息研究，因股票ETF T+1，不作为新买仓位可执行策略收益。
- 开发集确定分位边界，验证集只复用开发集边界。
- 未来20日标签存在重叠，因此同时报告错位非重叠IC与近似独立样本数。
- JSON报告另含20交易日移动区块Bootstrap的IC区间；多特征多周期检验仍存在数据窥探风险。
- 2025-08-01至2026-08-11曾被探索性检查触及，已降级为受污染回顾测试集；本报告仍不展示其特征效果。

## 样本切分

- 开发集：2021-08-12 至 2024-06-28
- 验证集：2024-08-01 至 2025-06-30
- 回顾测试集：2025-08-01 至 2026-08-11（受污染，不用于选参）
- 真正前向验证起点：等待正式候选冻结后的下一个交易日

## 稳定性摘要

|特征组|特征|周期|开发IC|验证IC|IC同号|分位差同号|
|---|---|---:|---:|---:|---|---|
|valuation|signal_pe_ttm|1|-0.0623|-0.1326|True|True|
|valuation|signal_pe_ttm|3|-0.1291|-0.2381|True|True|
|valuation|signal_pe_ttm|5|-0.1569|-0.2531|True|True|
|valuation|signal_pe_ttm|10|-0.2176|-0.3326|True|True|
|valuation|signal_pe_ttm|20|-0.3217|-0.4788|True|True|
|valuation|signal_pb|1|-0.0597|-0.1380|True|False|
|valuation|signal_pb|3|-0.1074|-0.2453|True|False|
|valuation|signal_pb|5|-0.1286|-0.2387|True|False|
|valuation|signal_pb|10|-0.1825|-0.2752|True|False|
|valuation|signal_pb|20|-0.2853|-0.4329|True|False|
|valuation|signal_earnings_yield|1|0.0623|0.1326|True|True|
|valuation|signal_earnings_yield|3|0.1291|0.2381|True|True|
|valuation|signal_earnings_yield|5|0.1569|0.2531|True|True|
|valuation|signal_earnings_yield|10|0.2176|0.3326|True|True|
|valuation|signal_earnings_yield|20|0.3217|0.4788|True|True|
|valuation|signal_pe_ttm_percentile_5y|1|-0.0683|-0.1216|True|False|
|valuation|signal_pe_ttm_percentile_5y|3|-0.1179|-0.2265|True|True|
|valuation|signal_pe_ttm_percentile_5y|5|-0.1562|-0.2272|True|True|
|valuation|signal_pe_ttm_percentile_5y|10|-0.2025|-0.3149|True|True|
|valuation|signal_pe_ttm_percentile_5y|20|-0.3001|-0.4303|True|True|
|valuation|signal_pb_percentile_5y|1|-0.0642|-0.1419|True|True|
|valuation|signal_pb_percentile_5y|3|-0.1113|-0.2462|True|True|
|valuation|signal_pb_percentile_5y|5|-0.1309|-0.2446|True|True|
|valuation|signal_pb_percentile_5y|10|-0.1844|-0.3163|True|True|
|valuation|signal_pb_percentile_5y|20|-0.2841|-0.4788|True|True|
|trend|signal_trend_ma20_over_ma60|1|-0.0277|-0.0150|True|False|
|trend|signal_trend_ma20_over_ma60|3|-0.0240|-0.0881|True|False|
|trend|signal_trend_ma20_over_ma60|5|-0.0312|-0.0820|True|False|
|trend|signal_trend_ma20_over_ma60|10|-0.0409|-0.1714|True|False|
|trend|signal_trend_ma20_over_ma60|20|-0.0600|-0.4424|True|False|
|trend|signal_trend_close_over_ma120|1|-0.0734|-0.0811|True|True|
|trend|signal_trend_close_over_ma120|3|-0.0897|-0.0952|True|True|
|trend|signal_trend_close_over_ma120|5|-0.1232|-0.1503|True|True|
|trend|signal_trend_close_over_ma120|10|-0.1741|-0.1481|True|True|
|trend|signal_trend_close_over_ma120|20|-0.1714|-0.3697|True|True|
|volatility|signal_rv_5|1|0.0165|-0.0327|False|True|
|volatility|signal_rv_5|3|-0.0110|-0.0294|True|True|
|volatility|signal_rv_5|5|-0.0012|-0.1098|True|False|
|volatility|signal_rv_5|10|0.0502|-0.0109|False|False|
|volatility|signal_rv_5|20|0.1418|-0.0545|False|False|
|volatility|signal_rv_20|1|0.0501|0.0309|True|True|
|volatility|signal_rv_20|3|0.0509|0.0941|True|True|
|volatility|signal_rv_20|5|0.1019|0.1471|True|False|
|volatility|signal_rv_20|10|0.1321|0.2368|True|False|
|volatility|signal_rv_20|20|0.2053|0.0424|True|False|
|volatility|signal_rv_60|1|0.0806|0.0443|True|False|
|volatility|signal_rv_60|3|0.1436|0.0751|True|False|
|volatility|signal_rv_60|5|0.1782|0.0497|True|False|
|volatility|signal_rv_60|10|0.2248|0.0370|True|False|
|volatility|signal_rv_60|20|0.2678|-0.1273|False|False|
|volatility|signal_rv_120|1|0.0288|-0.0360|False|False|
|volatility|signal_rv_120|3|0.0379|-0.0322|False|False|
|volatility|signal_rv_120|5|0.0547|-0.0447|False|False|
|volatility|signal_rv_120|10|0.0384|-0.1175|False|False|
|volatility|signal_rv_120|20|0.0686|-0.3576|False|False|
|volatility|signal_downside_rv_20|1|0.0391|0.0427|True|True|
|volatility|signal_downside_rv_20|3|0.0346|0.1381|True|True|
|volatility|signal_downside_rv_20|5|0.0785|0.2007|True|True|
|volatility|signal_downside_rv_20|10|0.1226|0.3175|True|True|
|volatility|signal_downside_rv_20|20|0.1143|0.1636|True|False|
|volatility|signal_parkinson_rv_20|1|0.0710|0.0405|True|False|
|volatility|signal_parkinson_rv_20|3|0.0924|0.0730|True|True|
|volatility|signal_parkinson_rv_20|5|0.1263|0.1151|True|False|
|volatility|signal_parkinson_rv_20|10|0.1484|0.1803|True|False|
|volatility|signal_parkinson_rv_20|20|0.1986|0.0061|True|False|
|volatility|signal_vol_term_20_120|1|0.0362|0.0260|True|True|
|volatility|signal_vol_term_20_120|3|0.0267|0.1018|True|True|
|volatility|signal_vol_term_20_120|5|0.0631|0.1349|True|True|
|volatility|signal_vol_term_20_120|10|0.1336|0.2961|True|True|
|volatility|signal_vol_term_20_120|20|0.2124|0.3212|True|True|

详细分位条件收益、命中率、MAE、MFE保存在同目录JSON报告中。

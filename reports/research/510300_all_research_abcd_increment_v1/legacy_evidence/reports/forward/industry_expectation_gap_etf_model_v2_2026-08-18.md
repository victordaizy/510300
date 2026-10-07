# 行业预期差—ETF模型 V2：板块重构与数据补齐

- 截止日：`2026-08-18`
- 状态：`CURRENT_SECTOR_CONTEXT_READY_ORIGINAL_NO_VIEW_UNCHANGED`
- 原V1方向状态：`NO_VIEW`（未改变）
- 边界：`RESEARCH_ONLY / SHADOW_ONLY / NO_POSITION_CHANGE`

## 一句话结论

官方CICS信息技术加通信服务占沪深300的33.70%；中信25行业研究口径下，核心科技加数字内容与通信覆盖30.58%。当前价格、订单分类资金流和ETF份额已补到最近可得日，但这些事实不自动把V1的NO_VIEW升级成方向预测。

## 中证官方宽板块

| CICS一级行业 | 权重 |
|---|---:|
| 信息技术 (Information Technology) | 23.099% |
| 金融 (Financials) | 18.744% |
| 工业 (Industrials) | 16.671% |
| 通信服务 (Communication Services) | 10.604% |
| 原材料 (Materials) | 9.284% |
| 主要消费 (Consumer Staples) | 6.217% |
| 可选消费 (Consumer Discretionary) | 5.318% |
| 医药卫生 (Health Care) | 4.490% |
| 公用事业 (Utilities) | 2.696% |
| 能源 (Energy) | 2.556% |
| 房地产 (Real Estate) | 0.319% |

官方信息技术为 `23.099%`，通信服务为 `10.604%`，两者合计 `33.704%`；合计只用于科技宽口径展示。

## 互斥经济引擎

| 经济引擎 | 权重 | 5日 | 20日 | 60日 | 60日历史分位 | 预期差净贡献 | 预期差覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 核心科技 | 21.25% | 3.72% | -3.06% | -8.69% | 18.74% | 0.00% | 21.25% |
| 金融 | 20.07% | -0.79% | -1.21% | 3.49% | 68.77% | 0.00% | 8.74% |
| 先进制造 | 15.31% | 0.78% | 4.38% | -11.28% | 16.03% | -3.14% | 7.18% |
| 数字内容与通信 | 9.34% | 8.59% | -10.01% | -8.44% | 17.58% | 0.00% | 9.34% |
| 能源资源 | 8.55% | 0.37% | 8.15% | -5.59% | 23.63% | 5.92% | 7.46% |
| 必选消费 | 6.70% | -2.66% | -1.19% | 0.29% | 53.64% | 未观察 | 0.00% |
| 公用运输 | 5.79% | -0.37% | 0.33% | -0.61% | 40.70% | 0.00% | 5.79% |
| 医疗健康 | 4.35% | -1.12% | 5.76% | 16.66% | 90.15% | 未观察 | 0.00% |
| 中游材料 | 3.20% | -0.53% | 3.07% | -7.97% | 26.53% | 1.25% | 3.20% |
| 可选消费 | 3.15% | -1.59% | -0.47% | 2.01% | 54.28% | 0.00% | 3.02% |
| 地产基建 | 1.66% | -2.11% | -1.67% | -7.28% | 25.82% | -1.66% | 1.66% |

## 未来研究排序（定性先行）

| 经济引擎 | 研究优先级 | 预期差状态 | 价格阶段 | 预期差覆盖率 |
|---|---|---|---|---:|
| 能源资源 | PRIORITY_FISH_MIDDLE_CANDIDATE | NET_POSITIVE_GAP | EARLY_RECOVERY_FROM_WEAK_BASE | 87.3% |
| 中游材料 | WATCH_FOR_TURN_NOT_ENTER | NET_POSITIVE_GAP | WEAK_BASE_NOT_TURNED | 100.0% |
| 核心科技 | WATCH_FOR_EXPECTATION_UPGRADE | BALANCED_OR_NO_GAP | EARLY_RECOVERY_FROM_WEAK_BASE | 100.0% |
| 数字内容与通信 | WATCH_FOR_EXPECTATION_UPGRADE | BALANCED_OR_NO_GAP | EARLY_RECOVERY_FROM_WEAK_BASE | 100.0% |
| 地产基建 | DETERIORATION_RISK_OR_COUNTERTREND | NET_NEGATIVE_GAP | WEAK_BASE_NOT_TURNED | 100.0% |
| 金融 | NO_VIEW_EXPECTATION_DATA_GAP | INSUFFICIENT_EXPECTATION_COVERAGE | MIXED_OR_SOFT | 43.5% |
| 先进制造 | NO_VIEW_EXPECTATION_DATA_GAP | INSUFFICIENT_EXPECTATION_COVERAGE | EARLY_RECOVERY_FROM_WEAK_BASE | 46.9% |
| 必选消费 | NO_VIEW_EXPECTATION_DATA_GAP | INSUFFICIENT_EXPECTATION_COVERAGE | MIXED_OR_SOFT | 0.0% |
| 医疗健康 | NO_VIEW_EXPECTATION_DATA_GAP | INSUFFICIENT_EXPECTATION_COVERAGE | LATE_OR_EXTENDED | 0.0% |
| 公用运输 | OBSERVE_NO_EDGE | BALANCED_OR_NO_GAP | MIDDLE_TREND | 100.0% |
| 可选消费 | OBSERVE_NO_EDGE | BALANCED_OR_NO_GAP | MIXED_OR_SOFT | 95.7% |

该排序只回答下一步先研究谁：`PRIORITY_FISH_MIDDLE_CANDIDATE`仍不是买入信号；`WATCH_FOR_TURN_NOT_ENTER`明确表示基本面/预期差可能有利，但价格尚未确认。

## 重叠主题链

| 主题 | 核心覆盖 | 邻接覆盖 | 核心20日价格 | 核心60日价格 | 1日订单流强度 | 5日订单流强度 |
|---|---:|---:|---:|---:|---:|---:|
| 科技大类 | 30.58% | 0.00% | -4.96% | -8.73% | -5.59% | -0.12% |
| AI与数字基础设施 | 29.80% | 8.13% | -5.05% | -8.68% | -5.66% | 0.00% |
| 新质生产力 | 45.11% | 8.15% | -1.86% | -9.50% | -5.45% | -0.10% |
| 先进制造主题 | 15.31% | 19.71% | 4.35% | -11.09% | -4.18% | -0.64% |
| 绿色转型 | 14.16% | 8.15% | 3.24% | -8.41% | -3.46% | -1.13% |
| 内需 | 14.20% | 3.14% | 1.10% | 5.70% | -0.52% | -4.09% |
| 地产链 | 2.18% | 3.47% | -1.18% | -6.33% | -8.34% | -4.01% |
| 顺周期 | 31.29% | 7.00% | 1.90% | 0.10% | -3.59% | -2.82% |
| 红利防御 | 19.75% | 3.02% | 1.67% | 3.37% | 6.75% | -0.21% |
| 出口制造 | 34.54% | 2.22% | -0.25% | -8.06% | -3.29% | -0.12% |

主题之间存在重叠，禁止把核心覆盖或邻接覆盖跨主题相加。

## ETF与市场流动性

- 510300份额截止：`2026-08-18`；1日变化 `-1.519%`，5日变化 `-8.582%`。
- 融资融券截止：`2026-08-17`；融资净买入 `-132,672,328` 元。
- ETF行情截止：`2026-08-18`；成交额 `3,143,413,605` 元。
- 份额变化是申赎活动代理，不是精确现金净流入；个股订单分类资金流不是ETF申赎。

## 独立交叉核对

沪深300成分行业收益与全市场中信一级行业指数在 `25` 个可比行业中，同向比例为 `88.0%`。

## 仍未观察

- `CURRENT_NATIONAL_TEAM_HOLDINGS`
- `CURRENT_NATIONAL_TEAM_ACTIVITY`
- `POINT_IN_TIME_CONSENSUS_VINTAGE`
- `THEME_LEVEL_REVENUE_EXPOSURE`
- `MARGIN_DETAIL_2026_08_18_PROVIDER_NOT_YET_AVAILABLE`

V2解决了板块分割和当前数据缺口，但没有凭空制造未来信息。定性判断仍来自冻结的行业证据台账；价格和资金只用于判断是否已被计价及当前拥挤程度。国家队身份、点时一致预期和主题收入暴露仍缺可靠证据，因此保持研究观察，不做仓位映射。

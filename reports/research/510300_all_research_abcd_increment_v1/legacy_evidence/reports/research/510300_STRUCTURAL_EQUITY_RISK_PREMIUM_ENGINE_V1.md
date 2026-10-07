# 510300 结构性权益风险溢价引擎 V1：第一阶段报告

## 结论

第一阶段已经完成三张状态面板、解释性收益来源分解和机械重大周期图谱。
本轮没有运行组合净值、夏普率、仓位映射、阈值搜索、Paper/Shadow 或实盘。
结果状态为 `STAGE_1_COMPLETE_WITH_EXPLICIT_DATA_LIMITATIONS`；它不是可交易策略，
也没有让任何旧的盈利、估值、IF、期权、宏观或技术协议重新获得历史验证资格。

## 冻结边界

- 项目：`510300_STRUCTURAL_EQUITY_RISK_PREMIUM_ENGINE_V1`。
- 协议版本：`1.0.5`；修正：`OUTCOME_BLIND_DATETIME_UNIT_NORMALIZATION_CORRECTION_5`。
- 本次修正发生在收益读取前，只补充已登记点时总股本回退并让市值覆盖门
  正确传播；原 1.0.0 清单、已采财务收据及其哈希均被保留并绑定。
- 阶段：`MECHANISM_IDENTIFICATION`。
- 范围：2015-01-01 至 2026-08-14。
- 标的边界：`510300.SH_OR_CASH`；但本阶段 `MODEL_POSITION_TARGET=UNSET`。
- 期限固定为 `20D / 60D / 120D`。
- 机械重大回撤：完整水下周期峰谷回撤至少 15%；没有按年份或表现挑事件。
- 所有状态产物先写入并哈希冻结，随后才允许读取沪深300总收益。

## 三张状态面板

| 产品 | 覆盖/状态 | 关键说明 |
|---|---:|---|
| `CSI300_PIT_CASHFLOW_STATE_PANEL_V1` | 140 个月度 origin | CF 状态计数：`{"PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT": 125, "PARTIAL_PIT_CF_STATE_COVERAGE_BELOW_ONE_OR_MORE_GATES": 15}` |
| `CSI300_CROSS_SECTIONAL_PRESENT_VALUE_PANEL_V1` | 16 个固定组合 | PCA 状态：`PASS_EXPLANATORY_FULL_SAMPLE_PCA_WITHOUT_RETURN_LABELS`；没有使用收益标签 |
| `CSI300_RISK_BEARING_CAPACITY_PANEL_V1` | 2823 个日度观测 | RC 状态计数：`{"PASS_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR": 2429, "PARTIAL_RC_STATE_PCA_COVERAGE_NO_VIEW": 315, "NO_VIEW_RC_STATE_WARMUP_OR_COVERAGE": 79}` |

CF 六个核心量为收入同比、营业利润同比、经营现金流同比、ROE 变化、
改善权重广度和恶化集中度。DR 使用行业中性 E/P、B/P、S/P、经营现金流/P
及 16 个冻结组合提取无收益标签的共同因子。RC 同时保留资金利率、期限结构、
信用、汇率、价格冲击、融资余额、ETF 份额、成分相关性、行业相关性和内部背离。

## 数据准入与明确缺口

- 成分集合使用已准入的官方点时重放，每个 origin 固定 300 只。
- 财务值来自带 `NOTICE_DATE/UPDATE_DATE` 的东方财富 HSF10 二级聚合源；
  `NOTICE_DATE` 决定信息可得日，`UPDATE_DATE` 仅诊断修订风险。当前接口值可能
  已含后续修订，因此它不是首发值版本库，也不是完整官方原始 PDF 档案。
- 收入、归母利润、权益和股本与独立本地提供方做了只读核对；经营现金流没有
  第二个本地提供方，因此保留 `NO_VIEW_NO_INDEPENDENT_LOCAL_PROVIDER_CROSSCHECK`。
- 历史指数权重没有逐期版本凭证。主序列使用点时总市值代理权重；未版本化的
  历史指数权重只作诊断，状态仍是 `BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS`。
- 月度市场截面缺总市值时，使用已登记独立提供方在 origin 前可得的总股本
  与 origin 当日原始收盘价回退；市值可见成分数低于 270 时 CF 必须为 `NO_VIEW`。
- 成分股点时股息率没有合格档案，保留 `NO_VIEW`；现金流久期只使用代理量。
- 信用利差和部分 ETF 份额历史的发布时间没有完整版本证明，只能是
  `STATE_MEASUREMENT_ONLY / DISCOVERY_EVIDENCE_ONLY`。

## 解释性收益来源分解

- 完整 OLS 月份：109。
- 解释性 R²：0.4785。
- 系数：`{"intercept":0.005623267518493701,"cf_news":0.02313250647845799,"discount_rate_easing_news":0.06641532036048793,"risk_capacity_news":0.012460668328170013}`。
- 可观察前向期限数：`{"20": 138, "60": 136, "120": 133}`。
- 该分解是状态创新与月度收益的全样本解释性 OLS，不是现金流恒等式、
  不是预测通过，也没有根据拟合优度选择变量或惩罚参数。

## 重大周期因果图谱

- 机械事件类型计数：`{"INDEX_DOWN_EARNINGS_IMPROVING": 11, "INDEX_UP_INTERNAL_DETERIORATION": 8, "MAJOR_DRAWDOWN": 2, "MAJOR_RECOVERY": 1, "LONG_SIDEWAYS": 1, "MAJOR_RECOVERY_RIGHT_CENSORED": 1}`。
- 固定符号映射后的机制计数：`{"MIXED_OR_NO_VIEW": 16, "FUNDAMENTAL_EXPANSION": 5, "TRUE_CONTRACTION": 2, "LIQUIDITY_EXPANSION": 1}`。
- 事件窗口总行数：168。
- 每个事件统一保留前 120/60/20 日、事件起点、事件终点、后 20/60 日状态；
  分类只使用冻结的 CF/折现率宽松/RC 变化符号，不先写新闻叙事。

## 510300 与底层指数交叉检查

- 重叠日：2822。
- 510300 含分红日收益与沪深300全收益日收益相关系数：0.979679。
- 日收益差绝对值中位数：0.09%。
- 该检查只确认解释对象与可交易 ETF 的一致性，不是组合回测。

## 当前决定

1. 第一阶段机制识别完成，但数据状态不是无条件 `PASS`；所有缺口已进入收据。
2. 现在仍不得生成 0/25/50/75/100% 仓位，不得计算净值或夏普率。
3. 下一阶段若继续，应只在已冻结的 20/60/120 日条件地图上检验四种
   `CF × DR` 结构，并把 RC 作为放大/削弱项；不得从最好周期反选规则。
4. 夏普率 1.2 仍是最终项目目标，不是本阶段结果；本轮没有声称目标已经实现。

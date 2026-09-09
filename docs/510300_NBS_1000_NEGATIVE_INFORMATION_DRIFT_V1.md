# 510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1

## 1. 当前裁决与新研究身份

```text
MODEL_ID = 510300_NBS_1000_NEGATIVE_INFORMATION_DRIFT_V1
VERSION = 1.0.0
中文名称 = 510300国家统计局10时宏观负向信息漂移策略
PROTOCOL_STATE = FROZEN_BEFORE_EVENT_RETURN_READ
RESEARCH_STAGE = SOURCE_AND_EVENT_ADMISSION_ONLY
TRADABLE_UNIVERSE = 510300.SH + CASH_CNY
DEFAULT_POSITION = 100% 510300.SH
OPTIONS = NOT_USED
FUTURES = NOT_USED
CONSTITUENTS = NOT_USED
VALIDATED_ALPHA = FALSE
LIVE_TRADING_AUTHORIZED = FALSE
POSITION_IMPACT = 0
```

已经归档的`510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1`继续保持
`ARCHIVED_BLOCKED_G0_AND_FAILED_G1_OBSERVATION`。本研究不得使用其347个曲面日，不得修改期限规则、外推规则或以分钟数据营救该模型。

本研究的唯一机制假设是：国家统计局按事前日程在10:00集中发布“国民经济运行情况”时，如果510300在发布后最初5分钟出现相对同一时钟正常波动显著的负向反应，宏观增长、盈利和政策含义的机构重估可能继续延伸至下午。研究只判断是否值得暂时回避剩余日内风险，不预测宏观数据值，也不预测长期方向。

## 2. 时间范围和数据边界

```text
START_DATE = 2017-01-01
END_DATE = 2026-09-04
TIMEZONE = Asia/Shanghai
MINUTE_SOURCE_MODEL = 510300_STK_MINS_SOURCE_ADMISSION_V1
```

强制输入只有：

1. 已通过来源准入的510300未复权1分钟数据；
2. 同源`trade_cal`和`fund_daily`，只用于交易日和分钟日级核对；
3. 国家统计局官网的年度统计信息发布日程及可证明的事前调整；
4. 仅在G0至G3全部通过后，使用已经固定哈希的510300官方现金分红账本构造P0总财富基准。

不使用宏观实际值、一致预期、新闻文本、PMI、CPI/PPI、LPR、社融、工业企业利润、房价、期权、期货、成份股、成交量特征、技术指标或外部情绪。

## 3. 唯一事件家族

只接受国家统计局年度发布日程中标题规范化后属于以下集合的事件：

```text
国民经济运行情况
国民经济运行情况新闻发布会
```

事件必须同时满足：

```text
official_schedule_source = 国家统计局官网
scheduled_time = 10:00
scheduled_date = 上交所开放交易日
schedule_page_publication_time < scheduled_at
官方日程原始文件及SHA-256存在
若有调整，revision_known_at < revised_scheduled_at
510300当日有标准分钟交易
四个关键窗口完整
```

年度日程写为15:00、9:30或没有明确时刻的记录一律排除；不改变为盘后或次日事件。多个同一时点发布的工业、消费、投资、房地产数据合并为一个事件：

```text
event_id = NBS_NATIONAL_ECONOMY_YYYYMMDD_1000
```

年度日程属于初步计划。无法证明调整在事件前已公开时，事件状态为`EXCLUDED_UNVERIFIABLE_SCHEDULE_REVISION`；不得用新闻实际发布时间事后替换日程。

事件账本字段至少包括：

```text
event_id
scheduled_date
scheduled_at
schedule_publication_date
schedule_source_url
schedule_source_url_sha256
schedule_document_sha256
event_title_raw
event_title_normalized
original_scheduled_at
revised_scheduled_at
revision_known_at
sse_open_day
minute_data_state
final_event_eligibility
exclusion_reason
```

阶段D以前事件账本只包含日程、资格和分钟覆盖，不包含窗口价格、收益或标签。

## 4. 分钟时间语义和固定窗口

分钟来源必须通过`510300_STK_MINS_SOURCE_ADMISSION_V1`。时间戳固定为`BAR_END`；普通分钟柱标签`t`表示`(t-1分钟,t]`。

| 用途 | 固定BAR_END标签 | 根数 |
| --- | --- | ---: |
| 发布前 | 09:56—10:00 | 5 |
| 初始反应 | 10:01—10:05 | 5 |
| 模拟卖出 | 10:07—10:11 | 5 |
| 模拟买回 | 14:51—14:55 | 5 |

10:05之后至10:06之后保留至少一个完整分钟作为确认与模拟下单缓冲。任何窗口少于5根、成交量合计不为正或成交额合计不为正时，该事件为`NO_VIEW_INCOMPLETE_MINUTE_WINDOW`。

窗口价格统一为：

\[
VWAP=\frac{\sum amount}{\sum volume}
\]

不得改用开盘价、收盘价、典型价、未来最优价或理论中间价。

## 5. 唯一特征

对事件日`t`：

\[
g_t=\ln\left(\frac{P_t^{reaction}}{P_t^{pre}}\right)
\]

基准样本固定为事件日之前最近60个具备发布前和反应窗口完整数据的非事件开放交易日；60日必须全部早于事件日。不得使用事件日自身、未来日或其他宏观事件日。

\[
m_t=\operatorname{median}(g_d)
\]

\[
s_t=1.4826\operatorname{MAD}(g_d)
\]

若少于60日或`s_t<=0`，事件为`NO_VIEW_INSUFFICIENT_CAUSAL_BASELINE`。

\[
z_t=\frac{g_t-m_t}{s_t},\qquad X_t=\max(-z_t,0)
\]

`X_t`是V1唯一预测特征。不得加入成交量、路径效率、均线、RSI、波动率过滤或宏观数据值。

## 6. 唯一预测目标

标签在事件日14:55之后成熟：

\[
Y_t=-\ln\left(\frac{P_t^{exit}}{P_t^{entry}}\right)
\]

`Y_t>0`表示从模拟卖出窗口到模拟买回窗口价格继续下降，暂时持有现金有价值。不得改成10:30、午间、收盘、次日、未来最低价或结果最优窗口。

## 7. B0、B1和严格前序预测

```text
MIN_TRAINING_EVENTS = 36
REESTIMATION = 每个新事件发生前
TRAINING_WINDOW = EXPANDING_MATURED_EVENTS_ONLY
```

B0为此前所有成熟合格事件的`Y`均值。B1为带截距的一元普通最小二乘：

\[
Y_t=\alpha+\beta X_t+\epsilon_t
\]

使用HC3稳健协方差。预测均值单侧90%下界：

\[
LCB_t=\widehat\mu_t-1.645\,SE(\widehat\mu_t)
\]

`beta`不截断；如果`beta<=0`，不能反向改成买入反弹。禁止网格搜索、变量选择、模型切换和根据Sharpe选择模型。

训练起点为前36个合格成熟事件。之后按事件顺序固定评价时代：

```text
ERA_1 = 第37—56个事件
ERA_2 = 第57—76个事件
ERA_3 = 第77个及以后事件
```

ERA_3至少15个事件；否则G1失败。评价事件不足40个时为`NO_VIEW_INSUFFICIENT_EVENT_COUNT`。

## 8. 成本与交易规则

```text
INITIAL_CAPITAL_CNY = 200000
COMMISSION_RATE = 0.0002
MINIMUM_COMMISSION_CNY = 5 / leg
LOT_SIZE = 100 shares
BASE_SLIPPAGE = 5bp / leg
STRESS_SLIPPAGE = 10bp / leg
ROUND_TRIP_BASE_REFERENCE_COST = 14bp
SIGNAL_LCB_THRESHOLD = 42bp
LEVERAGE = FORBIDDEN
SHORTING = FORBIDDEN
INTRADAY_T = FORBIDDEN
```

默认100%持有510300旧库存。只有同时满足：

```text
X_t > 0
beta_hat_t > 0
LCB_t >= 0.0042
```

才在10:07—10:11标签对应窗口按VWAP并扣除卖出滑点、佣金卖出全部可用的100份整数倍旧库存；14:51—14:55标签对应窗口按VWAP并加入买入滑点、佣金用可用现金买回100份整数倍。当天买回份额当日不得再次卖出，余额保留现金。

账户状态必须显式保存：

```text
old_available_shares
same_day_bought_shares
cash_available
sell_proceeds
commission
rounding_cash
```

## 9. 固定统计规则

所有Bootstrap固定为：

```text
RESAMPLES = 10000
RANDOM_SEED = 5103001000
UNIT = EVENT
LOWER_CONFIDENCE_BOUND = 10th percentile
```

Spearman相关使用双边定义但只检查点估计是否严格大于0。平方误差改善定义为每个严格前序评价事件上`B0平方误差-B1平方误差`的均值。

两个固定安慰剂：

1. 同一事件日的09:30—09:35反应、09:36—09:41模拟入场，其余计算形式相同；只做统计安慰剂，不产生候选策略。
2. 每个事件匹配此前最近一个具备完整窗口且不属于任何本协议事件的开放交易日，使用相同10:00窗口；一个非事件日最多匹配一次，不足则该事件安慰剂为`NO_VIEW`。

宏观事件的`X→Y`关系必须强于两个安慰剂，否则`REJECTED_MECHANISM_NOT_EVENT_SPECIFIC`。安慰剂结果不得反向成为新策略。

## 10. 分层停止门

### G0：数据来源

必须全部通过：

```text
分钟来源一般准入通过
2017年以来开放日分钟数据覆盖率 >= 98%
标准241根日覆盖率 >= 98%
事件日四窗口覆盖率 = 100%
BAR_END语义有文档与交易时段结构证据
日级OHLC对账通过率 >= 99.9%
国家统计局年度日程原始文件和SHA-256完整
```

### G1：事件可识别性

```text
合格10:00事件 >= 85
具备四窗口的事件 >= 80
首次训练事件 >= 36
ERA_1事件 >= 15
ERA_2事件 >= 15
ERA_3事件 >= 15
评价事件 >= 40
```

G0或G1失败时不构造`Y`，不训练模型。

### G2：机制门

B1相对B0必须全部满足：

```text
严格前序平均平方误差改善 > 0
全样本beta点估计 > 0
事件Bootstrap beta单侧90%下界 > 0
三个评价时代至少两个beta > 0
三个评价时代至少两个B1平方误差 < B0
Spearman(X,Y) > 0
宏观事件X→Y关系强于两个固定安慰剂
```

失败状态：`REJECTED_FROZEN_NO_SIGNAL_REVERSAL`或`REJECTED_MECHANISM_NOT_EVENT_SPECIFIC`。

### G3：可交易信号门

```text
LCB >= 42bp的独立信号 >= 12
至少两个评价时代各有 >= 3个信号
基础成本后平均避免损失 > 0
压力成本后平均避免损失 > 0
压力成本后事件Bootstrap均值单侧90%下界 > 0
压力成本后净命中率 >= 60%
任一事件贡献 <= 总净收益30%
```

信号少于12时为`NO_VIEW_INSUFFICIENT_HIGH_MARGIN_SIGNAL_DENSITY`，不是可以降低阈值的失败。

### G4：一次性账户回测

仅G0—G3全部通过后运行一次。P0为510300含官方现金分红的买入持有；P1为与策略全样本平均日内现金暴露相同、但在每个开放日机械均匀减持的暴露基准；P2为对每个真实信号使用固定种子，从同一自然年、相同星期且四窗口完整的非事件日中无放回抽取同数量现金窗口，不足时按同年任意星期扩展，仍不足则`NO_VIEW_CONTROL_MATCH_FAILED`。

正式门：

```text
基础成本净Sharpe >= 1.20
净Sharpe - P0净Sharpe >= 0.25
净Sharpe - P1净Sharpe > 0
净CAGR >= P0 CAGR - 1个百分点
最大回撤相对P0降低 >= 15%
压力滑点下净超额 > 0
三个评价时代至少两个净超额 > 0
年均完整往返 <= 4
单一事件对总超额贡献 <= 30%
```

年化固定为252日；无风险收益固定为0。Deflated Sharpe的试验次数取运行前`config/research_registry.jsonl`中`research_scope=ETF_510300`的唯一模型/实现尝试数加本模型1；不得事后减少。

## 11. 永久禁止事项

- 不恢复期权V1；
- 不加入9:30的PMI、CPI或其他事件凑数量；
- 不读取宏观实际值或一致预期；
- 不改变四个分钟窗口、60日标准化期、36事件训练门、42bp阈值或成本；
- 不把负向漂移失败后反向改成买入反弹；
- 不删除反弹、无成交或亏损事件；
- 不只汇报发生交易的盈利日；
- 不忽略100份整数、最低佣金、现金余额和T+1；
- 不在任何门失败后同历史调参营救。

## 12. 当前授权

协议冻结只授权数据来源、事件资格和研究评价。Paper、Shadow、券商连接、仓位目标、订单生成和实盘交易均未授权。任何历史通过也不会自动改变：

```text
LIVE_TRADING_AUTHORIZED = FALSE
POSITION_IMPACT = 0
```

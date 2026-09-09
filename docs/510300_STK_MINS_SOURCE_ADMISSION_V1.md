# 510300_STK_MINS_SOURCE_ADMISSION_V1

## 1. 身份与边界

```text
DATA_MODEL_ID = 510300_STK_MINS_SOURCE_ADMISSION_V1
VERSION = 1.0.0
STATE = SOURCE_PROTOCOL_FROZEN_BEFORE_BULK_ACQUISITION
PURPOSE = 510300公共分钟数据底座
SYMBOL = 510300.SH
FREQUENCY = 1min
START_DATE = 2017-01-01
END_DATE = 2026-09-04
POSITION_IMPACT = 0
```

本协议只裁决分钟数据是否可作为研究输入，不生成信号、收益、仓位、订单或交易授权。它不修改、不重启，也不营救已经归档的
`510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1`。

## 2. 固定来源契约

```text
PROVIDER = TUSHARE_COMPATIBLE_PROXY
ENDPOINT = https://tt.xiaodefa.cn
API_NAME = stk_mins
AUTHENTICATION = x-api-key请求头中的临时56位凭据
REQUEST_METHOD = HTTPS POST JSON
ACCEPT_ENCODING = gzip
```

固定请求参数：

```text
ts_code = 510300.SH
freq = 1min
start_date = YYYY-MM-DD 00:00:00
end_date = YYYY-MM-DD 23:59:59
```

按自然月请求。单次响应达到或超过8,000行时，视为可能截断并停止，不接受静默分页猜测。请求启动间隔不得低于0.65秒；只有连接错误、超时、HTTP 429、500、502、503、504可以按固定退避重试。权限、认证、参数错误和成功空响应不重试。

当前Tushare网页将ETF历史分钟接口写作`etf_mins`，而Tushare接口索引和当前代理行为仍允许`stk_mins`返回ETF数据。本协议不声称二者天然等价，只承认以下精确事实：指定代理、指定接口名、指定代码、指定时间范围所返回并通过本协议核验的数据。

不允许改用AkShare、东方财富、新浪、腾讯、券商接口或其他相似数据填补失败月份。任何新来源都需要新的版本化来源协议。

## 3. 凭据规则

凭据只允许来自隐藏交互输入或`TUSHARE_PROXY_TOKEN`环境变量。凭据不得出现在：

- 请求正文；
- 文件名和目录名；
- 原始响应、Parquet、CSV；
- 日志、异常、报告、清单、Git提交。

## 4. 原始数据与收据

每个请求必须以不可变检索目录保存：

```text
data/raw/tushare/510300_stk_mins_source_admission_v1/
  trade_cal/range=20170101_20260904/retrieved_at=.../
  fund_daily/510300.SH/range=20170101_20260904/retrieved_at=.../
  stk_mins/510300.SH/1min/month=YYYYMM/retrieved_at=.../
```

每个成功或失败响应保存原始响应字节与收据；成功响应另外保存规范化Parquet。收据至少包含：

```text
source
api
request_parameters
request_started_at
retrieved_at
http_status
business_code
success
row_count
columns
response_bytes
response_sha256
raw_relative_path
normalized_relative_path
normalized_sha256
endpoint_origin
credential_persisted=false
```

断点续传只能复用请求参数完全一致、原始响应哈希正确、规范化文件哈希正确、行数和字段一致的成功收据。检查点只是进度提示，不是数据存在证明。

## 5. 字段、单位和主键

必需字段：

```text
ts_code
trade_time
open
high
low
close
vol
amount
```

固定解释：

```text
price = 未复权人民币价格
vol = 份
amount = 元
PRIMARY_KEY = ts_code + trade_time
SORT_ORDER = trade_time升序
```

与`fund_daily`对账时执行固定单位变换：

```text
minute_vol_in_daily_unit = sum(vol) / 100
minute_amount_in_daily_unit = sum(amount) / 1000
```

不得根据对账结果寻找其他缩放倍数。

## 6. 时间戳语义

```text
BAR_TIMESTAMP_SEMANTICS = BAR_END
```

除09:30首柱外，标签为`HH:MM`的记录表示区间`(HH:MM-1分钟, HH:MM]`；09:30柱包含开盘集合竞价和首个记录区间。标准完整交易日的合法标签固定为：

```text
09:30, 09:31, ..., 11:30
13:01, 13:02, ..., 15:00
```

共241根，不应存在13:00柱。本数据不包含15:00之后的盘后固定价格成交，不能用于研究该时段。

策略真实窗口转换为以下固定标签，不再根据结果调整：

| 真实研究窗口 | BAR_END标签 | 根数 |
| --- | --- | ---: |
| 发布前`09:55—10:00` | `09:56—10:00` | 5 |
| 初始反应`10:00—10:05` | `10:01—10:05` | 5 |
| 模拟卖出`10:06—10:11` | `10:07—10:11` | 5 |
| 模拟买回`14:50—14:55` | `14:51—14:55` | 5 |

这里的区间实际覆盖为`(start, end]`。该选择由来源的BAR_END聚合口径决定，不允许在看到事件收益后改成相邻标签。

## 7. 质量检查

逐行必须满足：

```text
ts_code == 510300.SH
trade_time可解析且位于Asia/Shanghai交易时段
open, high, low, close为有限正数
high >= max(open, close)
low <= min(open, close)
high >= low
vol >= 0
amount >= 0
```

逐日必须记录：

- 行数和标准241标签是否完整；
- 重复主键和跨月重复；
- 上午、下午区间是否完整；
- 四个关键窗口是否各有5根；
- 日内首开、最高、最低、末收；
- 分钟成交量和成交额合计；
- 与同日`fund_daily`的OHLC和固定单位换算后总量差异。

2026-07-06以后日线可能包含15:00后的ETF盘后固定价格成交，而`stk_mins`不包含。对该日期后的成交量、成交额差异单独记录为市场制度分段，不允许把日线新增量伪造成分钟成交。由于本策略四个窗口均在15:00之前，只有当OHLC或关键窗口受影响时才阻断策略分钟数据；总量对账门在该制度分段上使用15:00前可比口径，如无法取得可比口径则标记`NO_VIEW_POST_CLOSE_TOTAL_NOT_COMPARABLE`，不伪报通过。

## 8. 来源准入门

必须全部满足：

```text
G0_PERMISSION_PROBE = PASS
ALL_EXPECTED_MONTH_PARTITIONS = PRESENT_AND_HASH_VALID
DUPLICATE_PRIMARY_KEYS = 0
UNEXPECTED_SYMBOL_ROWS = 0
INVALID_OHLC_ROWS = 0
NEGATIVE_VOLUME_OR_AMOUNT_ROWS = 0
OPEN_TRADING_DAY_ANY_DATA_COVERAGE >= 98%
STANDARD_241_BAR_DAY_COVERAGE >= 98%
NBS_ELIGIBLE_EVENT_FOUR_WINDOW_COVERAGE = 100%
DAILY_OHLC_RECONCILIATION_RATE >= 99.9%
PRE_2026_07_06_VOLUME_RELATIVE_ERROR_MAX <= 0.1%
PRE_2026_07_06_AMOUNT_RELATIVE_ERROR_MAX <= 0.1%
BAR_TIMESTAMP_SEMANTICS = BAR_END_WITH_DOCUMENT_AND_SESSION_EVIDENCE
UNEXPLAINED_TIMESTAMP_DRIFT = 0
```

如果事件账本尚未完成，只能先形成`PASS_GENERAL_MINUTE_SOURCE_PENDING_EVENT_WINDOW_GATE`，不能形成策略G0最终通过。

任一必需门失败：

```text
SOURCE_STATE = BLOCKED
STRATEGY_RESEARCH = NOT_ALLOWED
RETURN_EVALUATION = NOT_ALLOWED
POSITION_IMPACT = 0
```

## 9. 输出

```text
data/curated/510300_stk_mins_source_admission_v1/510300_1min.parquet
data/curated/510300_stk_mins_source_admission_v1/daily_quality_ledger.parquet
reports/data_quality/510300_stk_mins_source_admission_v1/permission_probe.json
reports/data_quality/510300_stk_mins_source_admission_v1/acquisition_checkpoint.json
reports/data_quality/510300_stk_mins_source_admission_v1/acquisition_summary.json
reports/data_quality/510300_stk_mins_source_admission_v1/source_adjudication.json
reports/data_quality/510300_stk_mins_source_admission_v1/SOURCE_ADMISSION_SUMMARY.md
```

所有输出继续保持研究用途；通过来源门不代表存在Alpha或获得任何交易权限。


# 数据字典

## 1. 日线行情

| 字段 | 类型 | 含义 |
|---|---|---|
| symbol | string | 证券代码，例如510300.SH |
| date | date | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量，必须记录供应商单位 |
| amount | float | 成交额，单位必须记录 |
| pct_change | float | 数据源提供的涨跌幅，仅作校验 |
| source | string | 数据源 |
| retrieved_at | datetime | 下载时间 |

## 2. 分钟K线

| 字段 | 类型 | 含义 |
|---|---|---|
| symbol | string | 证券或合约代码 |
| bar_start | datetime | 名义区间起点；特殊集合竞价柱仍需结合阶段标签解释 |
| bar_end | datetime | K线结束时点，已冻结为时间标签语义 |
| trade_date | date | 交易日期 |
| open | float | 分钟开盘价 |
| high | float | 分钟最高价 |
| low | float | 分钟最低价 |
| close | float | 分钟收盘价 |
| volume | float | 分钟成交量及其单位 |
| amount | float | 分钟成交额及其单位 |
| vendor_avg_price | float | 数据源均价，不能替代自行计算的VWAP |
| source | string | 数据源 |
| retrieved_at | datetime | 下载时间 |
| bar_position | string | first、afternoon_first、last或regular |
| session_phase | string | 开盘、连续竞价、收盘集合竞价混合等阶段 |
| contains_opening_auction | bool | 是否混入开盘集合竞价语义 |
| contains_closing_auction | bool | 是否包含收盘集合竞价成交 |
| post_close_included | bool | 是否包含15:00后的盘后固定价格成交 |
| post_close_separately_identifiable | bool | 盘后成交能否从普通时段独立识别 |

腾讯15分钟源的`amount`为缺失值，因为返回末字段尚无可靠成交额定义。不得用不明字段填补。

## 3. 当日累计成交快照

| 字段 | 类型 | 含义 |
|---|---|---|
| timestamp | datetime | 累计快照时点 |
| price | float | 时点价格 |
| cumulative_volume | float | 截至时点累计成交量，标准化为份 |
| cumulative_amount | float | 截至时点累计成交额，元 |
| incremental_volume | float | 相邻快照间新增成交量，份 |
| incremental_amount | float | 相邻快照间新增成交额，元 |
| session_phase | string | 包括`post_close_fixed_price` |

该接口只返回当前交易日；15:00与15:30累计值之差才是当日盘后成交量/额的直接观测。

## 4. 公司行动

| 字段 | 类型 | 含义 |
|---|---|---|
| symbol | string | 基金代码 |
| record_date | date | 权益登记日 |
| ex_date | date | 除息日 |
| payment_date | date | 分红发放日 |
| cash_dividend_per_share | float | 每份现金分红 |
| split_ratio | float | 拆分或折算比例 |
| source | string | 数据源 |
| retrieved_at | datetime | 下载时间 |

## 5. 质量状态

每个数据分区都应记录：

- `quality_status`：PASS、WARN或FAIL；
- `quality_errors`：错误列表；
- `row_count`：行数；
- `min_timestamp`：最早时间；
- `max_timestamp`：最晚时间；
- `content_hash`：文件哈希。

## 6. IF研究日线

第一阶段保存新浪提供的IF主力连续研究序列，字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| symbol | string | `IF0`，仅用于研究，不代表可直接交易的单一合约 |
| date | date | 交易日期 |
| open | float | 主力连续开盘价 |
| high | float | 主力连续最高价 |
| low | float | 主力连续最低价 |
| close | float | 主力连续收盘价 |
| volume | float | 成交量 |
| open_interest | float | 持仓量 |
| settle | float | 数据源结算价 |
| contract_type | string | `main_continuous` |
| source | string | 数据源 |
| retrieved_at | datetime | 下载时间 |

`IF0`存在换月拼接和价格连续性问题，不能直接作为实盘下单合约。若后续策略依赖IF，将另行保存实际合约并定义换月规则。

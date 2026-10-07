# 期权曲面数据可行性审计

审计时间：`2026-08-18T20:11:07.338617+08:00`  
状态：`BLOCKED_MISSING_LOCAL_OPTION_HISTORY`  
收益回测许可：`DISABLED`

## 结论

O1—O4 当前不得运行收益检验。

## 本地输入

| 输入 | 存在 | 字节数 | 文件 |
|---|---:|---:|---|
| quote | false | 0 | `data/raw/return_tail/options/510300_eod_chain.parquet` |
| contract_master | true | 50057 | `data/raw/return_tail/options/510300_contract_master.parquet` |
| cffex_cross_validation | true | 11287335 | `data/raw/return_tail/options/IO_eod_chain.parquet` |
| underlying | true | 102194 | `data/raw/market/510300_daily_raw.parquet` |
| dividend | true | 2021 | `data/reference/510300_dividends.csv` |
| risk_free | true | 58130 | `data/raw/macro/china_government_bond_yields_daily.parquet` |

## 已执行检查

```json
{}
```

## 阻断项

- 本地已有合约主表、逐合约日线、风险指标和最后五日分钟成交，但没有2019-12-23至2026-08-14逐日收盘bid1、ask1及市场时间戳，无法验证历史曲面。
- 不得用网页当前行情、仅成交价日线、最后一笔残留盘口或估计价差替代逐日历史盘口。

## 官方数据边界

交易所规则证明实时行情含买卖报价、成交量和持仓量；这不等于本地已经拥有可回放的历史收盘快照。510300期权历史盘口需要上交所授权行情产品或具备相应许可的数据商交付；中金所快照只用于IO交叉验证。

- https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250611_10781547.shtml
- https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781448.shtml
- https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml
- https://www.cffex.com.cn/lssjfw/

只有行情链、合约主表、标准/调整标志、合约单位、收盘买卖报价、成交量、持仓量、市场时间戳和曲面覆盖全部通过后，才可把本报告状态改为 PASS。

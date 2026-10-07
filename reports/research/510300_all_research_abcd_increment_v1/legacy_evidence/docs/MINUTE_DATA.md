# 510300分钟行情接口与审计口径

## 数据源与用途

| 数据 | 来源 | 当前范围 | 用途 |
|---|---|---:|---|
| 15分钟主数据 | 新浪 `CN_MarketDataService.getKLineData` | 1970根，约123个完整交易日 | 原型研究与主特征时钟 |
| 15分钟第二来源 | 腾讯 `ifzq.mkline` | 最近320根 | 独立供应商价格、成交量交叉 |
| 1分钟验证数据 | 新浪同一K线接口 | 最近1970根，8个完整日 | 验证15分钟时间戳和聚合边界 |
| 当日累计快照 | 腾讯 `minute.query` | 仅当前交易日 | 识别并从接入日起积累盘后固定价格成交 |

全部价格为未复权原始价格。主数据采用滚动窗口合并保存，但公开接口无法自动回补窗口之前的历史。

## 文件

- `data/raw/market/510300_15m_raw.parquet`
- `data/raw/market/510300_15m_tencent_raw.parquet`
- `data/raw/market/510300_1m_validation_raw.parquet`
- `data/raw/market/510300_intraday_tencent_snapshots.parquet`
- 各数据文件旁的 `.metadata.json`

## 刷新与审计

```powershell
.\scripts\run_minute_data_pipeline.ps1
```

该命令依次执行以下独立模块：

```powershell
.\.venv\Scripts\python.exe -m scripts.download_510300_15m
.\.venv\Scripts\python.exe -m scripts.download_510300_minute_auxiliary
.\.venv\Scripts\python.exe -m scripts.quality_check_510300_15m
.\.venv\Scripts\python.exe -m scripts.cross_check_510300_minute_sources
```

审计报告：

- `reports/data_quality/510300_15m_quality.json`
- `reports/data_quality/510300_minute_cross_source.json`

建议每个交易日15:30以后运行辅助下载。腾讯当日累计接口不能回补既往日期，因此只有持续运行才能形成真实盘后历史。

## 时间戳与交易阶段

`bar_end`表示K线结束时点。1分钟反聚合已验证：

- 10:00柱由09:46—10:00组成；
- 新制下15:00柱由14:46—14:57记录及15:00收盘集合竞价记录组成；
- 新浪1分钟源不单列14:58、14:59，新制完整日为238条；
- 普通15分钟K线没有15:05—15:30盘后数据。

15分钟数据保存：

- `bar_position`：`first`、`afternoon_first`、`last`或`regular`；
- `session_phase`：开盘混合、上午连续、下午连续、旧制末柱或新制收盘集合竞价混合；
- `contains_opening_auction`、`contains_closing_auction`；
- `post_close_included`、`post_close_separately_identifiable`。

第一根、午后第一根和最后一根不得默认与普通中间K线同质。2026-07-06以后最后一根尤其混合连续竞价与收盘集合竞价。

## 当前质量结论

- 日线OHLC边界交叉：`PASS`；
- 成交量/成交额逐日完全一致：`WARN`；
- 盘后历史完整性：`WARN`；
- 历史长度：`WARN`，仅能用于策略机制和数据管线原型。

日线总量减15分钟总量在2026-07-06以后出现正缺口，可作为缺失盘后量的代理上界，但不能直接等同于真实盘后量，因为旧制度样本也有供应商差异。真实盘后量优先使用腾讯当日累计快照在15:00与15:30之间的差。

腾讯15分钟数据的成交量已由手乘100转换为份。其末字段不是已确认的成交额，标准化字段`amount`明确留空；不得用不明字段伪造成交额。腾讯与新浪单根K线不能混源拼接。

## 研究边界

当前15分钟OHLCV可用于：

- K线结构与FVG定义验证；
- VWAP、相对成交量、波动、开盘区间等特征计算验证；
- T+1状态机、信号时间和MAE/MFE管线验证。

当前数据不能支持：

- 把约123个交易日误当作1970个独立样本；
- 大规模参数搜索后宣称策略有效；
- 把K线high、low、open或VWAP默认视为任意规模可成交价；
- PV/LC、主动买卖、bid/ask、盘口深度、撤单、队列位置等订单流研究。

因此，`15m OHLCV接通`与`PV/LC订单流接通`是两个独立里程碑。

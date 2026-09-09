# 510300一级—二级市场前向数据与2万元纸面执行

## 当前状态

- 数据采集器：`510300_PRIMARY_SECONDARY_FORWARD_V1`；
- 小账户纸面策略：`510300_SMALL_ACCOUNT_PAPER_V1`；
- 盘中IOPV和PCF仅从实际接入日起积累，不伪造历史回填；
- PCF每个采集进程只请求一次，IOPV仅在`09:30–11:30`、`13:00–15:00`采样；
- 连续模式使用Windows文件锁防止重复进程；
- 申赎/IOPV覆盖层尚未授权仓位映射，不连接券商，不自动下单。

## 数据源

### 每日PCF

通过上海证券交易所ETF申购赎回清单接口获取：

- 单位净值与最小申购赎回单位净值；
- 最小申购赎回单位；
- 预估现金部分与现金替代比例上限；
- 申购、赎回是否开放及各类上限；
- 是否公布IOPV。

### 盘中快照

通过上交所官方行情快照接口获取：

- 510300最新价、成交量额与交易阶段；
- IOPV；
- `(最新价 / IOPV - 1) × 10,000`得到的盘中折溢价基点；
- 盘后固定价格成交量额。

每次采集同时保存标准化Parquet、原始JSON载荷、获取时间和SHA-256指纹。

## Windows运行方式

单次抓取并刷新纸面目标：

```powershell
.\scripts\run_510300_primary_market_forward.ps1
```

交易时间内每60秒采集，到15:00停止：

```powershell
.\scripts\run_510300_primary_market_forward.ps1 -Watch
```

隐藏窗口后台启动：

```powershell
.\scripts\start_510300_primary_market_forward_background.ps1
```

安装工作日09:25自动采集任务：

```powershell
.\scripts\install_510300_daily_collection_task.ps1
```

移除自动采集任务：

```powershell
.\scripts\uninstall_510300_daily_collection_task.ps1
```

计划任务按当前Windows用户运行，要求电脑已开机且用户已登录。若错过09:25，会在下一次可用时尝试启动；采集器自身仍严格限制交易时段，不会把收盘后快照计作完整盘中样本。

只采集10次，用于运行验收：

```powershell
.\scripts\run_510300_primary_market_forward.ps1 -Watch -MaxSamples 10
```

已知完整账户状态时，例如持有1500份，其中旧仓1200份、当日新仓300份，可用现金12000元：

```powershell
.\scripts\run_510300_primary_market_forward.ps1 -CurrentShares 1500 -AvailableCash 12000 -SellableShares 1200 -TodayBoughtShares 300
```

四项账户输入缺一时只报告目标，不生成可执行差额。可卖旧仓与当日新仓之和必须等于当前总份额；卖出量不得超过可卖旧仓。所有结果只用于纸面研究，不发送任何委托。

连续采集正常结束后，主入口会先按R5冻结参数刷新当日基础信号，再生成外层目标。基础信号、IOPV交易日和执行许可必须同日；非复核日不允许把目标差额误报成交易。历史日期只能使用`refresh_r5_daily_signal.py --audit-only`审计，禁止覆盖最新纸面信号。

## 2万元执行约束

- 仓位仅使用`0% / 25% / 50% / 75% / 100%`五档；
- 买入按100份整手；
- 普通调仓差额不少1000份；
- 小于1000份的差额标记为`累积等待`；
- 按单边3bp佣金、最低5元佣金和单边5bp滑点做保守估算；
- 不杠杆、不卖空、不参与90万份的一级市场申购赎回。

## 输出

- `data/raw/primary_market/510300_pcf_daily.parquet`；
- `data/raw/primary_market/510300_iopv_snapshots.parquet`；
- `data/raw/primary_market/raw/`下的原始载荷；
- `reports/data_quality/510300_primary_market_forward_status.json`；
- `reports/data_quality/510300_primary_market_readiness.json`；
- `paper/510300_small_account_latest_signal.json`。
- `paper/round5_daily_refresh_status.json`（R5刷新步骤、日期与耗时）。

## 样本成熟度门槛

- 每个交易日至少120个去重IOPV快照，才计为一个完整覆盖日；
- 满20个完整覆盖日，只允许开始研究评估；
- 满40个完整覆盖日，进入建议研究样本状态；
- 达到天数门槛不等于策略有效，也不会自动打开仓位映射；
- 仍须验证时序稳定性、样本外增益与计入费用后的收益。

## 治理边界

历史日终折溢价不能冒充盘中IOPV。新采集数据只用于预先冻结的纸面研究。在足够的前向事件、时序稳定性、成本后收益和校准门槛全部通过前，`primary_market_overlay.position_delta`固定为0。

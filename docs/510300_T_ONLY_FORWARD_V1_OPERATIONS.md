# 510300 T_ONLY_FORWARD_V1 每日运行手册

## 1. 自动运行

- Windows 计划任务：`Codex-510300-T-Only-Forward-V1`
- 运行时间：北京时间每周一至周五 16:30
- 执行身份：当前用户已登录、最低权限
- 补跑：错过计划时间后，在用户登录且网络可用时由 `StartWhenAvailable` 启动
- 并发：`IgnoreNew`，同一时间只允许一个实例
- 重试：失败后每 15 分钟重试一次，最多 3 次
- 单次上限：30 分钟；内部刷新和前瞻阶段各自最多 180 秒

16:30 用于避开尚未收盘的日线。刷新脚本若在 16:00 前被手工执行，会主动排除本地当天，防止把盘中数据当作完整日线。

## 2. 每日链路

1. 校验自动化清单、父研究清单、协议、实现和冻结输入的 SHA-256。
2. 从新浪 ETF 日线入口刷新独立行情副本，禁止覆盖父研究输入。
3. 对冻结区间的 OHLC、成交量和成交额逐值复核。
4. 用官方分红事件表和新浪累计分红事件做日期、金额复核。
5. 使用上交所年度休市日历识别完整周；周内只有最后交易日可使用本周周线状态。
6. 仅从 2026-08-19 收盘开始重建V1.1前瞻影子账本并生成每日操作卡。
7. 核对数据闸门日期、V1.1前瞻报告日期、操作卡日期、账本行数和日期唯一性。
8. 原子更新每日运行状态；每次运行保留独立 stdout/stderr 日志及哈希。

任何步骤失败都输出 `NO_VIEW_OPERATIONAL_FAILURE`，并停止后续阶段，不能沿用上一日结果冒充本次成功。

## 3. 必查产物

- `paper/t_only_forward_v1/daily_run_status.json`：本次运行是否真正成功。
- `reports/data_quality/t_only_forward_v1_data_gate.json`：行情、重叠区间及分红核验。
- `reports/forward/t_only_forward_v1_1_status.json`：V1.1前瞻成熟度、影子指标和当前影子信号。
- `reports/forward/t_only_forward_v1_1_daily_guide.md`：每天直接阅读的影子操作卡。
- `paper/t_only_forward_v1_1/daily_ledger.parquet`：逐日影子权益；前瞻尚未开始时可以不存在。
- `reports/forward/t_only_forward_v1_task_audit.json`：计划任务配置、最近调度结果和最近产物运行的联合审计。
- `output/t_only_forward_v1_logs/<run_id>/`：每个阶段的标准输出、错误输出及对应哈希。

不得仅依据 Windows 计划任务的 `LastTaskResult = 0` 判定成功。还必须同时满足：最新 `run_id` 已变化、每日状态退出码为 0、数据闸门为 `PASS`、报告与行情截止日一致、账本无重复日期。

## 4. 状态解释

- `SUCCESS_COLLECTING_NOT_STARTED`：运行成功，但尚未到前瞻信号起点。
- `SUCCESS_COLLECTING`：运行成功，前瞻样本仍未成熟。
- `SUCCESS_FORWARD_GATES_PASS`：运行成功且成熟后全部冻结门槛通过；仍只是影子证据。
- `SUCCESS_FORWARD_GATES_FAIL_EVIDENCE`：运行成功，但成熟后至少一个研究门槛失败；这是有效否证，不是系统故障。
- `NO_VIEW_OPERATIONAL_FAILURE`：数据、网络、指纹、超时或产物一致性失败；当日无可用视图。

## 5. 手工运行与验收

在项目根目录使用 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts\run_t_only_forward_v1_daily.py
& .\scripts\audit_t_only_forward_v1_task.ps1
```

安装和移除计划任务：

```powershell
& .\scripts\install_t_only_forward_v1_task.ps1
& .\scripts\uninstall_t_only_forward_v1_task.ps1
```

移除计划任务不会删除历史影子账本、报告或日志。

## 6. 研究纪律

至少达到 252 个新交易日和 3 个闭合周期前，一律不评价策略通过或失败，不根据中途收益修改参数。模型目标仓位仅是影子状态，不读取或映射真实持仓，不生成订单，不连接券商。

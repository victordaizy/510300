# 510300期权前向盘口 2026-08-19 NO_VIEW 审计

- 最终状态：`NO_VIEW_RUNNER_PARSE_FAILURE`
- 预期采集窗口：2026-08-19 14:58:30 至 15:05:30（Asia/Shanghai）
- 实际合格快照：0
- 最近原始盘口日期：2026-08-18
- 2026-08-19 历史补采：禁止

## 根因证据

计划任务于 2026-08-19 05:52 启动 Windows PowerShell 5.1。启动脚本是无 BOM 的
UTF-8 文件，脚本内中文日志被本地代码页错误解码，导致第 43 行字符串终止符解析失败；
等待到 14:59 的进程从未建立。因此当日没有执行采集器，也没有原始盘口可供审计。

关键证据文件：

- `reports/forward/510300_option_orderbook_v1/runner_logs/20260819_process_stderr.log`
- `reports/forward/510300_option_orderbook_v1/runner_logs/20260819_runner.log`
- `reports/forward/510300_option_orderbook_v1/latest_collection_status.json`

## 修复与验证

2026-08-19 15:39 后将启动脚本中的非 ASCII 日志文字改为 ASCII，避免 Windows
PowerShell 5.1 对无 BOM UTF-8 的误解码。使用系统 `powershell.exe` 执行过窗保护测试，
脚本成功解析并按设计返回退出码 11；脚本非 ASCII 字节数为 0。

回归测试：`tests/test_510300_option_forward_orderbook_v1.py`，12项全部通过，其中新增
测试锁定启动脚本必须保持纯 ASCII。

## 治理结论

2026-08-19 只能记为程序故障导致的 `NO_VIEW`，不得使用历史收盘价、盘后查询或次日数据
冒充 14:59 的真实盘口。修复只影响未来采集，不改变已冻结信号公式、样本起点或数据门槛。

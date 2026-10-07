# R6事后审计可复现性检查

- 执行日期：2026-08-16（Asia/Shanghai）
- 命令：`.venv\Scripts\python.exe scripts\run_r6_postmortem.py`
- 检查方式：相同冻结输入与随机种子连续执行两次，逐字节比较核心机器报告和完整事件表。

| 文件 | 第一次SHA-256 | 第二次SHA-256 | 结果 |
|---|---|---|---|
| `reports/backtest/r6_postmortem.json` | `5B02906687EE32DABB731BE3A0D1081F99734DFEB13E54C4A30790FD3360FE1C` | `5B02906687EE32DABB731BE3A0D1081F99734DFEB13E54C4A30790FD3360FE1C` | 一致 |
| `reports/backtest/r6_signal_event_audit.csv` | `A28ED93464C767937A491CA7D8B9E7CD11B75183AF0359F16F065E14002EA05C` | `A28ED93464C767937A491CA7D8B9E7CD11B75183AF0359F16F065E14002EA05C` | 一致 |

结论：R6事后审计在冻结输入、协议和随机种子下可复现。

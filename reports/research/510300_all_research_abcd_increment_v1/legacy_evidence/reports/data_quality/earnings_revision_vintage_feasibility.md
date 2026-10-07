# 盈利预测历史快照可行性审计

审计时间：`2026-08-18T20:11:07.338617+08:00`  
状态：`TERMINATED_NO_POINT_IN_TIME_CONSENSUS_HISTORY`  
收益回测许可：`DISABLED`

## 结论

E1 按预注册停止条件终止。

公司公告财报、当前一致预期或事后整理的 FY1/FY2 数值都不能替代真实历史 vintage。

## 本地输入

| 输入 | 存在 | 文件 |
|---|---:|---|
| consensus_vintages | false | `data/raw/return_tail/earnings/csi300_consensus_vintages.parquet` |
| membership | true | `data/raw/constituents/000300_historical_membership_intervals.parquet` |
| weights | true | `data/raw/constituents/000300_historical_weights.parquet` |
| actual_company_financials_non_substitute | true | `data/raw/fundamentals/csi300_financials_point_in_time.parquet` |

## 已执行检查

```json
{}
```

## 停止原因

- 没有分析师一致预期历史快照文件。
- 现有点时财务表记录公司已公告财报，不含FTM EPS一致预期、分析师数、分歧或vendor vintage。
- 协议禁止用当前数据库回填过去一致预期，因此E1在首轮直接终止。

若未来采购到包含 as-of 时间、预测财年、EPS、分析师数、分歧和不可变 vintage ID 的历史快照，必须建立下一份独立数据接入协议；不得在当前首轮静默恢复 E1。

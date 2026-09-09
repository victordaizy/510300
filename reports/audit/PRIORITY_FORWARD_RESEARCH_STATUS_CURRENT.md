# 优先前瞻研究状态

- 总状态：`COLLECTING_FORWARD_WITH_BLOCKERS`
- 生成时间：`2026-08-28T10:39:49.928003+08:00`
- 治理状态：`BLOCKED_GOVERNANCE_UNVERSIONED`

## 1. PCF/IOPV执行信息

- 成熟度：2/20 个最低质量日；首次未见评价要求80日，复制要求120日。
- 最新观测交易日：`2026-08-18`
- 最近任务：`FAILED` / `EXTERNAL_FREE_SOURCE_FAILED`
- 动作：只继续严格前瞻采集，不生成仓位或订单。

## 2. 行业预期差

- 独立原点：1；成熟原点：0；非重叠60日块：0。
- 新原点闸门：`WAITING_FOR_NEXT_ORIGIN_EVIDENCE`
- 结果输入闸门：`COLLECTING_OUTCOME_INPUT_STALE`
- 最近任务收据：`reports/forward/industry_expectation_gap_v1_evaluation/task_runs/20260825T0905018732105Z_14416.json`
- 动作：等待新的官方证据包和更新后的月度点时面板；不复制旧预测。

## 3. 正交低波复制

- 状态：`NOT_STARTED_GOVERNANCE_GATE_BLOCKED`
- 动作：治理通过前不启动。

## 4. Windows计划任务

- 状态：`CODEX_HEARTBEATS_ACTIVE_WINDOWS_FALLBACK_BLOCKED`
- 已核验安装：`False`
- 手动运行器已验证：`True`
- Windows PowerShell 5.1解析：`True`
- 当前会话守护进程：`RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION`
- 心跳新鲜：`False`
- 登录自启动已验证：`False`
- Codex心跳自动化：`True`
- 证据：`reports/audit/priority_forward_scheduler_installation_20260819.json`
- 守护进程证据：`reports/audit/priority_forward_supervisor_status.json`
- Codex自动化证据：`reports/audit/priority_forward_codex_automation_v1_4_20260819.json`

## 告警

- `WARNING` `PRIMARY_MARKET_PCF_IOPV` `EXTERNAL_FREE_SOURCE_FAILED`：RuntimeException: 免费外部来源访问或解析失败，运行器退出码为3
- `WARNING` `INDUSTRY_EXPECTATION_GAP` `COLLECTING_OUTCOME_INPUT_STALE`：结果输入尚未达到严格前瞻评价日期闸门。
- `BLOCK` `ORTHOGONAL_LOW_VOL_REPLICATION` `BLOCKED_GOVERNANCE_UNVERSIONED`：治理未通过，禁止启动正交低波复制。
- `WARNING` `WINDOWS_TASK_SCHEDULER` `CODEX_HEARTBEATS_ACTIVE_WINDOWS_FALLBACK_BLOCKED`：早间严格采集与收盘后验收的Codex心跳均已启用；Windows任务调度器和登录启动项仍无权限，只缺本机持久备用通道。

## 门槛事件

- 首次跨越事件总数：0
- 本次新增事件：0
- 追加式账本：`reports/audit/priority_forward_threshold_events.jsonl`
- 门槛事件只开放对应研究步骤，不授权仓位、订单或实盘。

仓位映射、订单生成、券商连接与实盘均为关闭状态。

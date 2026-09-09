# 优先前瞻研究运行 V1.5：零付费来源与原子占位补充协议

## 1. 版本关系

V1.5 取代 V1.4 的活动任务部署，但不改写 `PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_4_MANIFEST` 及其仓库冻结文件。既有样本、成熟度门槛、任务时钟和否定结果保持原样；V1.4 清单中的外部任务哈希继续作为当时快照，不再代表当前活动入口。

当前总状态仍为 `PAUSE_AND_FIX_FOUNDATION`，研究证据状态仍为 `NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET`。

## 2. 本版唯一变更

1. 两个 PCF/IOPV PowerShell 入口使用带 BOM 的 UTF-8，可由 Windows PowerShell 5.1 正确解析；旧入口保留，不再作为自动任务入口。
2. 编排器在启动子任务前，以 `任务ID + 日期` 创建 `O_EXCL` 不可覆盖占位。并发心跳只有一个进程取得执行权，其余记录 `ALREADY_CLAIMED_ATOMIC`。
3. 每次 PCF 或 IOPV 请求保存不可覆盖的原始响应，并写入 `FREE_SOURCE_ACQUISITION_RECEIPT_V1` 来源回执；缺失、超时或解析失败不得写成零值。
4. 免费外部来源失败固定为退出码 `3` 和 `EXTERNAL_FREE_SOURCE_FAILED`；程序故障固定为退出码 `1` 和 `PROGRAM_FAILED`。
5. 编排收据增加随机唯一后缀，避免同一微秒内的文件名碰撞。

## 3. 入口与时钟

- 配置：`config/priority_forward_supervisor_v1_1.yaml`
- 单一编排入口：`scripts/run_priority_forward_codex_automation_v1_5.py`
- 早间阶段：仅 `PRIMARY_MARKET_PCF_IOPV`，原计划窗口 `09:25—09:35` 不变。
- 收盘阶段：仅行业预期差刷新和统一状态输出，禁止盘后补跑 PCF。
- 已存在当日成功或失败证据、或已存在原子占位时，不重复启动。

## 4. 证据门槛

工程修复只表示入口可运行，不增加完整质量日。PCF/IOPV 仍按 20/40/80/120 个完整质量日推进；行业预期差和 T-only 仍按原冻结门槛等待。

每次运行至少核对：编排收据、原子占位、任务收据、免费来源回执、原始响应哈希、当天交易日与成熟度。没有当天不可变证据时，不得把退出码 0 或旧状态当成完成。

## 5. 验收状态

V1.5 只有在以下自动测试均通过且两个既有心跳均切换到新入口后，才记为 `FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW`：

- Windows PowerShell 5.1 解析与 UTF-8 BOM；
- 跨进程同日原子占位；
- 免费来源成功与失败回执；
- 外部来源退出码 3 的端到端分类；
- 既有 PCF/IOPV 与 V1.4 冻结清单回归测试。

本版不改变研究、Shadow、仓位、订单或实盘状态。

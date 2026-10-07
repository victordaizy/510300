# 优先前瞻研究运行层 V1.1 兼容性补充

## 变更原因

V1 冻结后，真实安装演练发现 `install_510300_daily_collection_task.ps1` 在 Windows PowerShell 5.1 中因无 BOM 的非 ASCII 描述文本产生解析错误。V1 原清单保留不改，V1.1 仅把该任务描述改成 ASCII；采集逻辑、研究门槛和安全边界均未变化。

## 调度器部署状态

- 三个手动运行器已经真实执行或通过失败路径测试，并能写入不可变运行收据。
- PCF 安装器修复后，六个安装/运行脚本均通过 Windows PowerShell 5.1 解析。
- 当前受限会话调用 `Get-ScheduledTask` 和 `New-ScheduledTaskAction` 均收到本机权限拒绝；`schtasks.exe /Query` 也未能核验三个任务。
- 因此计划任务部署状态必须保持 `BLOCKED_LOCAL_TASK_SCHEDULER_PERMISSION`，不能声称已经安装。
- 具备任务调度器权限的交互式 Windows 会话应分别运行三个安装器，再核验下一次运行时间、实际日志与不可变收据。

## 不变边界

- PCF/IOPV 仍只做严格前瞻采集，20/40/80/120 日门槛不变。
- 行业预期差仍按独立预测原点及非重叠 60 日块成熟，不复制旧原点。
- 正交低波复制仍受治理 `PASS` 和独立预注册协议双重约束。
- 仓位映射、订单生成、券商连接和实盘交易始终关闭。


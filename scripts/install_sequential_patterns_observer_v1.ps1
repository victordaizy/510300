$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskName = 'Codex-510300-Sequential-Patterns-Observe-V1'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$observerPath = Join-Path $projectRoot 'scripts\observe_sequential_patterns_regime_v1.py'
$outputPath = Join-Path $projectRoot 'reports\research\510300_sequential_patterns_regime_v1\forward'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw '项目 Python 无窗口入口缺失。' }
if (-not (Test-Path -LiteralPath $observerPath -PathType Leaf)) { throw '观察脚本缺失。' }
if ((Get-TimeZone).Id -ne 'China Standard Time') { throw '主机时区不是北京时间，不能按16:10直接安装。' }
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) { throw '同名任务已经存在，不能覆盖其已有配置。' }
$taskAction = New-ScheduledTaskAction -Execute $pythonPath -Argument ('"' + $observerPath + '"') -WorkingDirectory $projectRoot
$taskTrigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '16:10'
$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$registered = Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Settings $taskSettings -Principal $taskPrincipal -Description '510300三类连续形态V1收盘研究观察；仅读取本地完整资料，不采集、不下单；无变化不重复提示。'
$taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
$registered | Export-ScheduledTask | Set-Content -LiteralPath (Join-Path $outputPath 'windows_task.xml') -Encoding utf8
$receipt = [ordered]@{
    recorded_at = (Get-Date).ToString('o')
    scheduler = 'WINDOWS_TASK_SCHEDULER'
    codex_native_automation_created = $false
    task_name = $taskName
    task_state = [string]$registered.State
    enabled = [bool]$registered.Settings.Enabled
    next_run_time = $taskInfo.NextRunTime.ToString('o')
    schedule = '北京时间，周一至周五16:10；脚本按冻结交易日历判定节假日。'
    collection_enabled = $false
    notifications = '只在本地attention_changes.jsonl记录有意义变化，无聊天推送。'
    runtime_requirement = 'Windows主机开机且当前用户登录；本地输入清单与当天行情已人工更新。'
    orders_authorized = $false
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputPath 'scheduler_receipt.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 5

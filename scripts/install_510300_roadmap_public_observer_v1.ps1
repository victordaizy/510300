param()
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$TaskName = 'Codex_510300_Roadmap_PublicVintages_V1'
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\pythonw.exe'
$RunnerPath = Join-Path $ProjectRoot 'scripts\run_510300_roadmap_public_observer_scheduled_v1.py'
$ReceiptPath = Join-Path $ProjectRoot 'reports\research\510300_roadmap_execution_v1\scheduled_task_receipt.json'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw '项目无pythonw.exe，不能建立后台观察任务。' }
if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) { throw '公开源观察入口缺失。' }
$TaskArguments = '-X utf8 "' + $RunnerPath + '"'
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    if ($Existing.Actions.Execute -ne $PythonPath -or $Existing.Actions.Arguments -ne $TaskArguments) {
        throw '存在同名但不同执行入口的任务，未覆盖。'
    }
    $Status = 'EXISTING_MATCHING_TASK'
} else {
    $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $TaskArguments -WorkingDirectory $ProjectRoot
    $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '09:15'
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 3) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $UserIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Principal = New-ScheduledTaskPrincipal -UserId $UserIdentity -LogonType Interactive -RunLevel Limited
    $Definition = New-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description '510300研究：免费官方融资和ETF份额的原始版本观察。每次最多3请求，250MiB上限；不生成策略、不下单。'
    Register-ScheduledTask -TaskName $TaskName -InputObject $Definition | Out-Null
    $Status = 'INSTALLED_PUBLIC_SOURCE_OBSERVER'
}
$Saved = Get-ScheduledTask -TaskName $TaskName
$Info = Get-ScheduledTaskInfo -TaskName $TaskName
$Receipt = [ordered]@{
    status = $Status
    recorded_at = (Get-Date).ToString('o')
    task_name = $TaskName
    task_state = [string]$Saved.State
    execute = $Saved.Actions.Execute
    arguments = $Saved.Actions.Arguments
    frequency = '周一至周五09:15，入口另核官方交易日历；非交易日只写跳过回执'
    next_scheduler_run = $Info.NextRunTime.ToString('o')
    user_must_be_logged_on = $true
    device_must_be_available = $true
    max_requests_per_run = 3
    data_fee_budget = 0
    strategy_forward_validation = $false
    actual_orders = 0
    storage_limit_mib = 250
    calendar_end = '2026-12-31'
    other_tasks_changed = $false
}
$Receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
$Receipt | ConvertTo-Json -Depth 6

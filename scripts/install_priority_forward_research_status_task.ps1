param(
    [string]$TaskName = "Codex-Priority-Forward-Research-Daily-Status",
    [string]$RunAt = "17:15"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_priority_forward_research_status_task.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "未找到优先研究状态任务运行器：$launcher"
}

$escapedLauncher = $launcher.Replace('"', '""')
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$escapedLauncher`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $RunAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "汇总PCF/IOPV、行业预期差与治理阻断。只报告研究状态，不生成仓位、订单或券商连接。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Output "计划任务已安装：$($registered.TaskName)"
Write-Output "当前状态：$($registered.State)"
Write-Output "下次运行：$($info.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "安全边界：只报告研究状态；仓位、订单、券商和实盘全部关闭"

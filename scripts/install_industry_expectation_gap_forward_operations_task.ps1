param(
    [string]$TaskName = "Codex-Industry-Expectation-Gap-Forward-Operations-V1",
    [string]$RunAt = "17:05"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_industry_expectation_gap_forward_operations_task.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "未找到行业预期差任务运行器：$launcher"
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
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "行业预期差严格前瞻运行。只评价冻结原点并监控月度新证据闸门；不自动生成预测，不生成仓位、订单或券商连接。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Output "计划任务已安装：$($registered.TaskName)"
Write-Output "当前状态：$($registered.State)"
Write-Output "下次运行：$($info.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "安全边界：不自动生成新原点；仓位、订单、券商和实盘全部关闭"

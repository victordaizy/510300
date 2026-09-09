param(
    [string]$TaskName = "Codex-510300-Primary-Market-Collector"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_510300_primary_market_collection_task.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Collector launcher not found: $launcher"
}

$escapedLauncher = $launcher.Replace('"', '""')
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$escapedLauncher`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "09:25"
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Weekday 510300 PCF and IOPV research collection only. Propagates the real exit code and never creates positions, orders, or broker connections."

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Output "Scheduled task installed: $($registered.TaskName)"
Write-Output "Current state: $($registered.State)"
Write-Output "Next run: $($info.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"

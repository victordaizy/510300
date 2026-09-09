param(
    [string]$TaskName = "Codex-510300-Primary-Market-Collector",
    [string]$RunAt = "09:25",
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$orchestrator = Join-Path $projectRoot "scripts\run_priority_forward_codex_automation_v1_5.py"
$arguments = "`"$orchestrator`" --phase morning"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到项目 Python：$python"
}
if (-not (Test-Path -LiteralPath $orchestrator)) {
    throw "未找到 V1.5 编排器：$orchestrator"
}

function Get-TaskAudit {
    param(
        [Microsoft.Management.Infrastructure.CimInstance]$Task
    )

    $info = Get-ScheduledTaskInfo -TaskName $Task.TaskName -TaskPath $Task.TaskPath
    $action = @($Task.Actions)[0]
    $trigger = @($Task.Triggers)[0]
    $executeMatches = [System.StringComparer]::OrdinalIgnoreCase.Equals(
        [System.IO.Path]::GetFullPath([string]$action.Execute),
        [System.IO.Path]::GetFullPath($python)
    )
    $argumentsMatch = ([string]$action.Arguments) -eq $arguments
    $workingDirectoryMatches = [System.StringComparer]::OrdinalIgnoreCase.Equals(
        [System.IO.Path]::GetFullPath([string]$action.WorkingDirectory),
        [System.IO.Path]::GetFullPath($projectRoot)
    )
    $restartCount = [int]$Task.Settings.RestartCount
    $exactMatch = $executeMatches -and $argumentsMatch -and $workingDirectoryMatches -and ($restartCount -eq 0)
    return [ordered]@{
        schema_version = "1.0.0"
        task_name = $Task.TaskName
        task_path = $Task.TaskPath
        state = [string]$Task.State
        next_run_time = $info.NextRunTime.ToString("o")
        last_run_time = $info.LastRunTime.ToString("o")
        last_task_result = $info.LastTaskResult
        action_execute = [string]$action.Execute
        action_arguments = [string]$action.Arguments
        working_directory = [string]$action.WorkingDirectory
        trigger_start_boundary = [string]$trigger.StartBoundary
        multiple_instances = [string]$Task.Settings.MultipleInstances
        restart_count = $restartCount
        atomic_claim_entrypoint = $true
        exact_match = $exactMatch
    }
}

if ($AuditOnly) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $audit = Get-TaskAudit -Task $existing
    $audit | ConvertTo-Json -Depth 5
    if (-not $audit.exact_match) {
        exit 2
    }
    exit 0
}

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument $arguments `
    -WorkingDirectory $projectRoot
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
    -Description "工作日运行 V1.5 研究编排入口；同日原子占位负责跨调度器去重。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$audit = Get-TaskAudit -Task $registered
if (-not $audit.exact_match) {
    throw "计划任务注册后与 V1.5 入口不一致"
}
$audit | ConvertTo-Json -Depth 5

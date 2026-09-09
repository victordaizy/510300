param(
    [string]$TaskName = "Codex-510300-T-Only-Forward-V1",
    [string]$RunAt = "16:30",
    [ValidateSet("Interactive", "S4U")]
    [string]$LogonType = "Interactive",
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$frozenTaskName = "Codex-510300-T-Only-Forward-V1"
$frozenRunAt = "16:30"
if ($TaskName -ne $frozenTaskName) {
    throw "任务名已冻结为：$frozenTaskName"
}
if ($RunAt -ne $frozenRunAt) {
    throw "运行时间已冻结为：$frozenRunAt"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\run_t_only_forward_v1_daily_v1_2.py"
$manifest = Join-Path $projectRoot "config\t_only_forward_v1_1_maturity_only_v1_2_manifest.json"
$expectedUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    $stream = [System.IO.File]::OpenRead($LiteralPath)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $algorithm.ComputeHash($stream)
        return (($hashBytes | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

foreach ($requiredFile in @($python, $runner, $manifest)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "计划任务依赖文件缺失：$requiredFile"
    }
}
$manifestSha256 = Get-FileSha256 -LiteralPath $manifest
$arguments = "`"$runner`" --manifest `"$manifest`" --expected-manifest-sha256 $manifestSha256"

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
    $startWhenAvailable = [bool]$Task.Settings.StartWhenAvailable
    $actualLogonType = [string]$Task.Principal.LogonType
    $principalMatches = $actualLogonType -eq $LogonType
    $exactMatch = (
        $executeMatches -and
        $argumentsMatch -and
        $workingDirectoryMatches -and
        ($restartCount -eq 0) -and
        (-not $startWhenAvailable) -and
        $principalMatches
    )
    return [ordered]@{
        schema_version = "1.2.0"
        task_name = $Task.TaskName
        task_path = $Task.TaskPath
        state = [string]$Task.State
        next_run_time = $info.NextRunTime.ToString("o")
        last_run_time = $info.LastRunTime.ToString("o")
        last_task_result = $info.LastTaskResult
        action_execute = [string]$action.Execute
        action_arguments = [string]$action.Arguments
        manifest_path = $manifest
        manifest_sha256 = $manifestSha256
        working_directory = [string]$action.WorkingDirectory
        restart_count = $restartCount
        start_when_available = $startWhenAvailable
        multiple_instances = [string]$Task.Settings.MultipleInstances
        principal_user_id = [string]$Task.Principal.UserId
        principal_logon_type = $actualLogonType
        principal_run_level = [string]$Task.Principal.RunLevel
        logout_reboot_verified = $false
        frozen_manifest_entrypoint = $true
        maturity_only_public_output = $true
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
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$settings.StartWhenAvailable = $false
$principal = New-ScheduledTaskPrincipal `
    -UserId $expectedUser `
    -LogonType $LogonType `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "510300 T-only V1.1 日更；V1.2 入口仅发布成熟度与完整性。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$audit = Get-TaskAudit -Task $registered
if (-not $audit.exact_match) {
    throw "计划任务注册后与 V1.2 冻结入口不一致"
}
$audit | ConvertTo-Json -Depth 5

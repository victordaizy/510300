param(
    [string]$TaskName = "Codex-510300-Primary-Market-Collector",
    [string]$RunAt = "09:25",
    [ValidateSet("Interactive", "S4U")]
    [string]$LogonType = "Interactive",
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$orchestrator = Join-Path $projectRoot "scripts\run_priority_forward_codex_automation_v1_9.py"
$supervisorConfig = Join-Path $projectRoot "config\priority_forward_supervisor_v1_5.yaml"
$manifest = Join-Path $projectRoot "config\priority_forward_research_operations_v1_9_manifest.json"
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

foreach ($requiredFile in @($python, $orchestrator, $supervisorConfig, $manifest)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "计划任务依赖文件缺失：$requiredFile"
    }
}
$manifestSha256 = Get-FileSha256 -LiteralPath $manifest
$arguments = "`"$orchestrator`" --phase morning --config `"$supervisorConfig`" --manifest `"$manifest`" --expected-manifest-sha256 $manifestSha256"

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
    $exactMatch = $executeMatches -and $argumentsMatch -and $workingDirectoryMatches -and ($restartCount -eq 0) -and (-not $startWhenAvailable) -and $principalMatches
    return [ordered]@{
        schema_version = "1.9.0"
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
        trigger_start_boundary = [string]$trigger.StartBoundary
        multiple_instances = [string]$Task.Settings.MultipleInstances
        restart_count = $restartCount
        start_when_available = $startWhenAvailable
        principal_user_id = [string]$Task.Principal.UserId
        principal_logon_type = $actualLogonType
        principal_run_level = [string]$Task.Principal.RunLevel
        persistence_capability = if ($actualLogonType -eq "S4U") { "LOGON_INDEPENDENT_CONFIGURED_NOT_REBOOT_VERIFIED" } else { "LOGGED_IN_SESSION_ONLY" }
        logout_reboot_verified = $false
        atomic_claim_entrypoint = $true
        immutable_receipt_deduplication = $true
        frozen_manifest_entrypoint = $true
        runtime_patch_id = "PCF_IOPV_STRICT_TLS_ROUTE_V1_3"
        transport_contract = "SSE_STRICT_TRANSPORT_CONTRACT_V1"
        tls_verification_required = $true
        insecure_tls_allowed = $false
        late_backfill_enabled = $false
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
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)
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
    -Description "工作日运行零付费 V1.9 严格TLS冻结入口；仅以不可变任务回执按日原子去重，错过窗口不补跑。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$audit = Get-TaskAudit -Task $registered
if (-not $audit.exact_match) {
    throw "计划任务注册后与 V1.9 入口不一致"
}
$audit | ConvertTo-Json -Depth 5

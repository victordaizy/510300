param(
    [string]$TaskName = "Codex-510300-T-Only-Forward-V1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\run_t_only_forward_v1_daily.py"
$runStatusPath = Join-Path $projectRoot "paper\t_only_forward_v1\daily_run_status.json"
$reportPath = Join-Path $projectRoot "reports\forward\t_only_forward_v1_task_audit.json"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    throw "计划任务不存在：$TaskName"
}
$info = Get-ScheduledTaskInfo -TaskName $TaskName
$action = @($task.Actions)[0]
$trigger = @($task.Triggers)[0]
$start = [datetimeoffset]::Parse($trigger.StartBoundary)

$checks = [ordered]@{
    task_exists = $true
    enabled = [bool]$task.Settings.Enabled
    execute_matches = [string]::Equals(
        [System.IO.Path]::GetFullPath($action.Execute),
        [System.IO.Path]::GetFullPath($python),
        [System.StringComparison]::OrdinalIgnoreCase
    )
    arguments_include_runner = $action.Arguments.Contains($runner)
    working_directory_matches = [string]::Equals(
        [System.IO.Path]::GetFullPath($action.WorkingDirectory),
        [System.IO.Path]::GetFullPath($projectRoot),
        [System.StringComparison]::OrdinalIgnoreCase
    )
    weekday_mask_matches = ([int]$trigger.DaysOfWeek -eq 62)
    trigger_time_matches = ($start.Hour -eq 16 -and $start.Minute -eq 30)
    start_when_available = [bool]$task.Settings.StartWhenAvailable
    ignore_new_instances = ([int]$task.Settings.MultipleInstances -eq 2)
    network_required = [bool]$task.Settings.RunOnlyIfNetworkAvailable
    execution_limit_30_minutes = ($task.Settings.ExecutionTimeLimit -eq "PT30M")
    restart_count_three = ([int]$task.Settings.RestartCount -eq 3)
    restart_interval_15_minutes = ($task.Settings.RestartInterval -eq "PT15M")
}
$configurationPass = -not ($checks.Values -contains $false)

$latestRun = $null
if (Test-Path -LiteralPath $runStatusPath) {
    $status = Get-Content -LiteralPath $runStatusPath -Raw | ConvertFrom-Json
    $latestRun = [ordered]@{
        run_id = $status.run_id
        started_at = $status.started_at
        finished_at = $status.finished_at
        overall_status = $status.overall_status
        exit_code = $status.exit_code
        as_of_market_date = $status.output_audit.as_of_market_date
        new_trading_days = $status.output_audit.new_trading_days
        closed_cycles = $status.output_audit.closed_cycles
        daily_decision = $status.output_audit.daily_decision
        daily_headline = $status.output_audit.daily_headline
    }
}

$report = [ordered]@{
    task_name = $TaskName
    audited_at = [datetimeoffset]::Now.ToString("o")
    configuration_status = $(if ($configurationPass) { "PASS" } else { "FAIL" })
    checks = $checks
    scheduler = [ordered]@{
        state = [string]$task.State
        last_run_time = $info.LastRunTime.ToString("o")
        last_task_result = [int]$info.LastTaskResult
        next_run_time = $info.NextRunTime.ToString("o")
        missed_runs = [int]$info.NumberOfMissedRuns
    }
    latest_artifact_run = $latestRun
    interpretation = "调度器结果0不单独代表成功；必须同时检查本次状态文件、行情截止日、账本行数和输出审计。"
    safety = [ordered]@{
        shadow_ledger_only = $true
        live_position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
    }
}

$directory = Split-Path -Parent $reportPath
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$temporary = "$reportPath.tmp"
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
Move-Item -LiteralPath $temporary -Destination $reportPath -Force
$report | ConvertTo-Json -Depth 8

if (-not $configurationPass) {
    exit 1
}

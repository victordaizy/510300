param(
    [string]$TaskName = "Codex-510300-Primary-Market-Collector",
    [string]$RunnerPath = "",
    [string]$RuntimeRoot = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$sourceProjectRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $sourceProjectRoot
}
else {
    [System.IO.Path]::GetFullPath($RuntimeRoot)
}
$runner = if ([string]::IsNullOrWhiteSpace($RunnerPath)) {
    Join-Path $sourceProjectRoot "scripts\run_510300_primary_market_forward_v1_2_1.ps1"
}
else {
    [System.IO.Path]::GetFullPath($RunnerPath)
}
$powershellExecutable = Join-Path $PSHOME "powershell.exe"
$readinessFile = Join-Path $projectRoot "reports\data_quality\510300_primary_market_readiness_v1_2.json"
$taskStatusFile = Join-Path $projectRoot "reports\data_quality\510300_primary_market_task_status_v1_2_1.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\data_quality\primary_market_task_runs_v1_2_1"
$tradingCalendarFile = Join-Path $projectRoot "data\reference\sse_trade_calendar_2026.csv"
$logDirectory = Join-Path $projectRoot "output\primary_market_forward_logs_v1_2_1"
$dateStamp = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $logDirectory "$dateStamp.task.log"
$startedAt = Get-Date
$receiptId = "{0}_{1}" -f $startedAt.ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ"), $PID
$taskReceiptFile = Join-Path $taskReceiptDirectory "$receiptId.json"
$runnerFile = if ($runner.StartsWith($sourceProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    $runner.Substring($sourceProjectRoot.Length + 1).Replace("\", "/")
}
else {
    $runner
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $taskReceiptDirectory -Force | Out-Null

$runStatus = "FAILED"
$collectionStatus = "PROGRAM_FAILED"
$taskExitCode = 1
$errorText = $null

try {
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
        throw "研究采集运行器缺失：$runner"
    }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $powershellExecutable -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $runner -Watch *>> $logFile
        $taskExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($taskExitCode -eq 3) {
        $collectionStatus = "EXTERNAL_FREE_SOURCE_FAILED"
        throw "免费外部来源访问或解析失败，运行器退出码为3"
    }
    if ($taskExitCode -ne 0) {
        throw "研究采集运行器退出码为$taskExitCode"
    }
    $runStatus = "SUCCESS"
    $collectionStatus = "COMPLETED_OR_MARKET_CLOSED"
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    [System.IO.File]::AppendAllText(
        $logFile,
        $errorText + [Environment]::NewLine,
        $utf8NoBom
    )
    if ($taskExitCode -ne 3) {
        $taskExitCode = 1
        $collectionStatus = "PROGRAM_FAILED"
    }
}
finally {
    $readiness = $null
    if (Test-Path -LiteralPath $readinessFile -PathType Leaf) {
        try {
            $readiness = Get-Content -LiteralPath $readinessFile -Raw -Encoding utf8 | ConvertFrom-Json
        }
        catch {
            $runStatus = "FAILED"
            $collectionStatus = "PROGRAM_FAILED_INVALID_READINESS"
            $taskExitCode = 1
            $errorText = "成熟度文件无法解析：$($_.Exception.Message)"
        }
    }

    $today = (Get-Date).ToString("yyyy-MM-dd")
    if ($runStatus -eq "SUCCESS") {
        if ($null -eq $readiness -or $readiness.last_observed_trade_date -ne $today) {
            if (-not (Test-Path -LiteralPath $tradingCalendarFile -PathType Leaf)) {
                $runStatus = "FAILED"
                $collectionStatus = "PROGRAM_FAILED_MISSING_TRADING_CALENDAR"
                $taskExitCode = 1
                $errorText = "交易日历缺失：$tradingCalendarFile"
            }
            else {
                $tradingDates = @(Import-Csv -LiteralPath $tradingCalendarFile | ForEach-Object { $_.trade_date })
                if ($tradingDates -contains $today) {
                    $runStatus = "FAILED"
                    $collectionStatus = "FAILED_NO_CURRENT_TRADING_DAY_DATA"
                    $taskExitCode = 1
                    $errorText = "交易日$today没有当天 PCF/IOPV 证据；不得按休市跳过。"
                }
                else {
                    $collectionStatus = "SKIPPED_NON_TRADING_DAY"
                }
            }
        }
        else {
            $todayQuality = @($readiness.daily_quality | Where-Object { $_.trade_date -eq $today })
            if ($todayQuality.Count -eq 1 -and $todayQuality[0].complete_quality_day) {
                $collectionStatus = "COMPLETE_QUALITY_DAY"
            }
            else {
                $collectionStatus = "COLLECTED_INCOMPLETE_QUALITY_DAY"
            }
        }
    }

    $status = [ordered]@{
        schema_version = "1.2.1"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        runtime_patch_id = "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1"
        python_module_entrypoint = $true
        runner_file = $runnerFile
        run_status = $runStatus
        collection_status = $collectionStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        latest_observed_trade_date = if ($null -ne $readiness) { $readiness.last_observed_trade_date } else { $null }
        full_coverage_days = if ($null -ne $readiness) { $readiness.full_coverage_days } else { $null }
        readiness_status = if ($null -ne $readiness) { $readiness.status } else { $null }
        second_official_endpoint_required_from = "2026-08-26"
        position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
        live_trading_enabled = $false
        receipt_file = $taskReceiptFile.Substring($projectRoot.Length + 1).Replace("\", "/")
        log_file = $logFile.Substring($projectRoot.Length + 1).Replace("\", "/")
    }
    $json = $status | ConvertTo-Json -Depth 8
    if (Test-Path -LiteralPath $taskReceiptFile) {
        throw "任务运行收据已存在，禁止覆盖：$taskReceiptFile"
    }
    $receiptTemporary = "$taskReceiptFile.$PID.tmp"
    [System.IO.File]::WriteAllText($receiptTemporary, $json, $utf8NoBom)
    Move-Item -LiteralPath $receiptTemporary -Destination $taskReceiptFile
    $statusTemporary = "$taskStatusFile.$PID.tmp"
    [System.IO.File]::WriteAllText($statusTemporary, $json, $utf8NoBom)
    Move-Item -LiteralPath $statusTemporary -Destination $taskStatusFile -Force
}

exit $taskExitCode

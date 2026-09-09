param(
    [string]$TaskName = "Codex-Priority-Forward-Research-Daily-Status"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\render_priority_forward_research_status.py"
$reportFile = Join-Path $projectRoot "reports\audit\priority_forward_research_status_current.json"
$reportMarkdownFile = Join-Path $projectRoot "reports\audit\PRIORITY_FORWARD_RESEARCH_STATUS_CURRENT.md"
$reportSnapshotDirectory = Join-Path $projectRoot "reports\audit\priority_forward_status_runs"
$taskStatusFile = Join-Path $projectRoot "reports\audit\priority_forward_research_status_task.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\audit\priority_forward_status_task_runs"
$logDirectory = Join-Path $projectRoot "output\priority_forward_research_status_logs"
$dateStamp = Get-Date -Format "yyyyMMdd"
$logFile = Join-Path $logDirectory "$dateStamp.task.log"
$startedAt = Get-Date
$receiptId = "{0}_{1}" -f $startedAt.ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ"), $PID
$taskReceiptFile = Join-Path $taskReceiptDirectory "$receiptId.json"
$reportSnapshotFile = Join-Path $reportSnapshotDirectory "$receiptId.json"
$reportMarkdownSnapshotFile = Join-Path $reportSnapshotDirectory "$receiptId.md"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $taskReceiptDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $reportSnapshotDirectory -Force | Out-Null
$runStatus = "FAILED"
$taskExitCode = 1
$errorText = $null

try {
    & $python $runner *>> $logFile
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -ne 0) {
        throw "优先研究状态渲染器退出码为$taskExitCode"
    }
    if (-not (Test-Path -LiteralPath $reportFile) -or -not (Test-Path -LiteralPath $reportMarkdownFile)) {
        throw "优先研究状态渲染成功但报告文件缺失"
    }
    if ((Test-Path -LiteralPath $reportSnapshotFile) -or (Test-Path -LiteralPath $reportMarkdownSnapshotFile)) {
        throw "优先研究报告快照已存在，禁止覆盖：$receiptId"
    }
    [System.IO.File]::WriteAllBytes($reportSnapshotFile, [System.IO.File]::ReadAllBytes($reportFile))
    [System.IO.File]::WriteAllBytes($reportMarkdownSnapshotFile, [System.IO.File]::ReadAllBytes($reportMarkdownFile))
    $runStatus = "SUCCESS"
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    $errorText | Out-File -LiteralPath $logFile -Append -Encoding utf8
    $taskExitCode = 1
}
finally {
    $report = $null
    if (Test-Path -LiteralPath $reportFile) {
        $report = Get-Content -LiteralPath $reportFile -Raw -Encoding utf8 | ConvertFrom-Json
    }
    $status = [ordered]@{
        schema_version = "1.0.0"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        run_status = $runStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        report_status = if ($null -ne $report) { $report.overall_status } else { $null }
        alert_count = if ($null -ne $report) { $report.alerts.Count } else { $null }
        report_snapshot_file = if (Test-Path -LiteralPath $reportSnapshotFile) { $reportSnapshotFile.Substring($projectRoot.Length + 1).Replace("\", "/") } else { $null }
        report_markdown_snapshot_file = if (Test-Path -LiteralPath $reportMarkdownSnapshotFile) { $reportMarkdownSnapshotFile.Substring($projectRoot.Length + 1).Replace("\", "/") } else { $null }
        position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
        live_trading_enabled = $false
        receipt_file = $taskReceiptFile.Substring($projectRoot.Length + 1).Replace("\", "/")
        log_file = $logFile.Substring($projectRoot.Length + 1).Replace("\", "/")
    }
    $json = $status | ConvertTo-Json -Depth 5
    if (Test-Path -LiteralPath $taskReceiptFile) {
        throw "任务运行收据已存在，禁止覆盖：$taskReceiptFile"
    }
    $receiptTemporary = "$taskReceiptFile.tmp"
    [System.IO.File]::WriteAllText($receiptTemporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $receiptTemporary -Destination $taskReceiptFile
    $temporary = "$taskStatusFile.tmp"
    [System.IO.File]::WriteAllText($temporary, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $taskStatusFile -Force
}

exit $taskExitCode

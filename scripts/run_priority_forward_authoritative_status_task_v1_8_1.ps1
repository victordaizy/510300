param(
    [string]$TaskName = "Codex-Priority-Forward-Authoritative-Status-V1.8.1"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runnerModule = "scripts.render_priority_forward_authoritative_status_v1_8_1"
$configFile = Join-Path $projectRoot "config\priority_forward_authoritative_status_v1_8_1.yaml"
$reportFile = Join-Path $projectRoot "reports\audit\priority_forward_authoritative_status_v1_8_1.json"
$reportMarkdownFile = Join-Path $projectRoot "reports\audit\PRIORITY_FORWARD_AUTHORITATIVE_STATUS_V1_8_1.md"
$reportSnapshotDirectory = Join-Path $projectRoot "reports\audit\priority_forward_authoritative_status_runs_v1_8_1"
$taskStatusFile = Join-Path $projectRoot "reports\audit\priority_forward_authoritative_status_task_v1_8_1.json"
$taskReceiptDirectory = Join-Path $projectRoot "reports\audit\priority_forward_authoritative_status_task_runs_v1_8_1"
$logDirectory = Join-Path $projectRoot "output\priority_forward_authoritative_status_logs_v1_8_1"
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
    if (-not (Test-Path -LiteralPath $configFile -PathType Leaf)) {
        throw "权威状态配置缺失：$configFile"
    }
    & $python -m $runnerModule --config $configFile *>> $logFile
    $taskExitCode = $LASTEXITCODE
    if ($taskExitCode -ne 0) {
        throw "权威状态生成器退出码为$taskExitCode"
    }
    if (-not (Test-Path -LiteralPath $reportFile -PathType Leaf) -or -not (Test-Path -LiteralPath $reportMarkdownFile -PathType Leaf)) {
        throw "权威状态生成成功但报告文件缺失"
    }
    if ((Test-Path -LiteralPath $reportSnapshotFile) -or (Test-Path -LiteralPath $reportMarkdownSnapshotFile)) {
        throw "权威状态快照已存在，禁止覆盖：$receiptId"
    }
    [System.IO.File]::WriteAllBytes($reportSnapshotFile, [System.IO.File]::ReadAllBytes($reportFile))
    [System.IO.File]::WriteAllBytes($reportMarkdownSnapshotFile, [System.IO.File]::ReadAllBytes($reportMarkdownFile))
    $runStatus = "SUCCESS"
}
catch {
    $errorText = "{0}: {1}" -f $_.Exception.GetType().Name, $_.Exception.Message
    [System.IO.File]::AppendAllText(
        $logFile,
        $errorText + [Environment]::NewLine,
        $utf8NoBom
    )
    $taskExitCode = 1
}
finally {
    $report = $null
    if (Test-Path -LiteralPath $reportFile -PathType Leaf) {
        try {
            $report = Get-Content -LiteralPath $reportFile -Raw -Encoding utf8 | ConvertFrom-Json
        }
        catch {
            $runStatus = "FAILED"
            $taskExitCode = 1
            $errorText = "权威状态报告无法解析：$($_.Exception.Message)"
        }
    }
    $status = [ordered]@{
        schema_version = "1.8.1"
        receipt_id = $receiptId
        immutable_receipt = $true
        task_name = $TaskName
        runtime_patch_id = "AUTHORITY_STATUS_PYTHON_MODULE_ENTRYPOINT_V1_8_1"
        python_module_entrypoint = $true
        run_status = $runStatus
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        task_exit_code = $taskExitCode
        error = $errorText
        overall_research_status = if ($null -ne $report) { $report.overall_research_status } else { $null }
        decision = if ($null -ne $report) { $report.decision } else { $null }
        primary_market_deployment_status = if ($null -ne $report) { $report.active_research_streams[1].deployment_status } else { $null }
        primary_market_collection_status = if ($null -ne $report) { $report.active_research_streams[1].latest_task_authority.collection_status } else { $null }
        report_snapshot_file = if (Test-Path -LiteralPath $reportSnapshotFile) { $reportSnapshotFile.Substring($projectRoot.Length + 1).Replace("\", "/") } else { $null }
        report_markdown_snapshot_file = if (Test-Path -LiteralPath $reportMarkdownSnapshotFile) { $reportMarkdownSnapshotFile.Substring($projectRoot.Length + 1).Replace("\", "/") } else { $null }
        research_only = $true
        shadow_enabled = $false
        position_mapping_enabled = $false
        order_generation_enabled = $false
        broker_connection_enabled = $false
        live_trading_enabled = $false
        receipt_file = $taskReceiptFile.Substring($projectRoot.Length + 1).Replace("\", "/")
        log_file = $logFile.Substring($projectRoot.Length + 1).Replace("\", "/")
    }
    $json = $status | ConvertTo-Json -Depth 8
    if (Test-Path -LiteralPath $taskReceiptFile) {
        throw "权威状态任务收据已存在，禁止覆盖：$taskReceiptFile"
    }
    $receiptTemporary = "$taskReceiptFile.$PID.tmp"
    [System.IO.File]::WriteAllText($receiptTemporary, $json, $utf8NoBom)
    Move-Item -LiteralPath $receiptTemporary -Destination $taskReceiptFile
    $statusTemporary = "$taskStatusFile.$PID.tmp"
    [System.IO.File]::WriteAllText($statusTemporary, $json, $utf8NoBom)
    Move-Item -LiteralPath $statusTemporary -Destination $taskStatusFile -Force
}

exit $taskExitCode

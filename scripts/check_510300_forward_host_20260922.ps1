param([switch]$ScheduledSmoke)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportRoot = Join-Path $projectRoot 'reports\research\510300_forward_host_diagnosis_20260922'
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Save-ResearchJson([string]$Name, $Value) {
    $destination = Join-Path $reportRoot $Name
    [System.IO.File]::WriteAllText($destination, ($Value | ConvertTo-Json -Depth 12), $utf8)
}

$taskName = 'Codex-510300-Primary-Market-Collector'
$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName -TaskPath $task.TaskPath
$operatingSystem = Get-CimInstance Win32_OperatingSystem
$xml = Export-ScheduledTask -TaskName $taskName -TaskPath $task.TaskPath
[System.IO.File]::WriteAllText((Join-Path $reportRoot 'existing_task.xml'), $xml, $utf8)
$snapshot = [ordered]@{
    captured_at=(Get-Date).ToString('o')
    task_name=$taskName
    task_state=[string]$task.State
    last_run_time=$info.LastRunTime.ToString('o')
    last_task_result=[long]$info.LastTaskResult
    last_task_result_hex=('0x{0:X8}' -f [long]$info.LastTaskResult)
    next_run_time=$info.NextRunTime.ToString('o')
    boot_time=$operatingSystem.LastBootUpTime.ToString('o')
    actions=@($task.Actions | Select-Object Execute,Arguments,WorkingDirectory)
    wake_to_run=[bool]$task.Settings.WakeToRun
    start_when_available=[bool]$task.Settings.StartWhenAvailable
    restart_count=[int]$task.Settings.RestartCount
    principal_logon_type=[string]$task.Principal.LogonType
    original_task_changed=$false
    market_collection_triggered=$false
}
Save-ResearchJson 'host_and_task.json' $snapshot
$events = @(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-General';Id=@(12,13);StartTime=(Get-Date).Date.AddDays(-3)} -MaxEvents 15 | ForEach-Object {
    [ordered]@{time_created=$_.TimeCreated.ToString('o');event_id=$_.Id;record_id=$_.RecordId;message=$_.Message}
})
Save-ResearchJson 'boot_shutdown_events.json' $events
$log = Get-WinEvent -ListLog 'Microsoft-Windows-TaskScheduler/Operational'
Save-ResearchJson 'scheduler_log_state.json' ([ordered]@{enabled=[bool]$log.IsEnabled;record_count=$log.RecordCount;history_retroactively_created=$false})
$receiptDirectories = @('reports\audit\priority_forward_codex_runs_v1_11','reports\audit\priority_forward_task_claims_v1_10','reports\data_quality\primary_market_task_runs_v1_3')
$today = (Get-Date).ToString('yyyy-MM-dd')
$todayCompact = (Get-Date).ToString('yyyyMMdd')
$presence = @($receiptDirectories | ForEach-Object {
    $relative = $_
    $directory = Join-Path $projectRoot $relative
    $files = @(Get-ChildItem -LiteralPath $directory -File -ErrorAction SilentlyContinue | Where-Object {$_.Name.Contains($today) -or $_.Name.Contains($todayCompact)} | Select-Object -ExpandProperty Name)
    [ordered]@{directory=$relative;same_day_files=$files}
})
Save-ResearchJson 'same_day_receipt_presence.json' $presence
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location -LiteralPath $projectRoot
try {
    $statusText = & $python -X utf8 -m scripts.run_priority_forward_codex_automation_v1_11_1 --status-only --expected-manifest-sha256 2ffef04f1edcba3171a7f5f83e84c99ea0bdbbafaedbc194e6683ce954b40a52
    if ($LASTEXITCODE -ne 0) { throw '原入口只读核对未通过。' }
    Save-ResearchJson 'current_source_status.json' ($statusText | ConvertFrom-Json)
} finally { Pop-Location }

if (-not $ScheduledSmoke) {
    $snapshot | ConvertTo-Json -Depth 8
    exit 0
}

$smokeName = 'Codex-510300-Readonly-Preflight-20260922'
if (Get-ScheduledTask -TaskName $smokeName -ErrorAction SilentlyContinue) {
    throw '专用检查任务已存在，不覆盖。'
}
$smokeScript = Join-Path $reportRoot 'scheduled_readonly_smoke.py'
$smokeResult = Join-Path $reportRoot 'scheduled_readonly_smoke_result.json'
if (Test-Path -LiteralPath $smokeResult) { throw '启动检查回执已存在，不覆盖。' }
$smokePython = @'
"""仅由临时计划任务调用原入口的校验模式；不采集行情。"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[3]
interpreter = root / '.venv/Scripts/python.exe'
command = [str(interpreter), '-X', 'utf8', '-m', 'scripts.run_priority_forward_codex_automation_v1_11_1', '--verify-only', '--expected-manifest-sha256', '2ffef04f1edcba3171a7f5f83e84c99ea0bdbbafaedbc194e6683ce954b40a52']
try:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    payload = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'collection_triggered': False, 'quality_day_added': False}
except Exception as exc:
    payload = {'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'status': 'SMOKE_FAILED', 'error': str(exc), 'collection_triggered': False, 'quality_day_added': False}
Path(__file__).with_name('scheduled_readonly_smoke_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
'@
[System.IO.File]::WriteAllText($smokeScript, $smokePython, $utf8)
$pythonWindowless = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonWindowless)) { throw '未找到无窗口Python入口，原任务保持。' }
$action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument ('"' + $smokeScript + '"') -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId $task.Principal.UserId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 2) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$registered = $false
try {
    Register-ScheduledTask -TaskName $smokeName -Action $action -Principal $principal -Settings $settings -Description '临时只读启动检查，不采集、不交易。' | Out-Null
    $registered = $true
    Start-ScheduledTask -TaskName $smokeName
    $deadline = (Get-Date).AddSeconds(50)
    do {
        Start-Sleep -Milliseconds 500
        $checkTask = Get-ScheduledTask -TaskName $smokeName
    } while (((-not (Test-Path -LiteralPath $smokeResult)) -or [string]$checkTask.State -eq 'Running') -and (Get-Date) -lt $deadline)
    $checkInfo = Get-ScheduledTaskInfo -TaskName $smokeName
    $evidence = [ordered]@{task_name=$smokeName;state=[string]$checkTask.State;last_result=[long]$checkInfo.LastTaskResult;result_file_exists=(Test-Path -LiteralPath $smokeResult);original_task_changed=$false;collection_triggered=$false;quality_day_added=$false}
    Save-ResearchJson 'scheduled_smoke_receipt.json' $evidence
    if ([string]$checkTask.State -eq 'Running') { throw '只读任务仍在运行；保留任务及回执供继续查询。' }
    if (-not (Test-Path -LiteralPath $smokeResult)) { throw '只读任务没有产生结果，详见计划任务回执。' }
    $result = Get-Content -LiteralPath $smokeResult -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($result.exit_code -ne 0) { throw '只读入口核对返回失败。' }
    $result | ConvertTo-Json -Depth 6
} catch {
    Save-ResearchJson 'scheduled_smoke_error.json' ([ordered]@{message=$_.Exception.Message;error_id=$_.FullyQualifiedErrorId;collection_triggered=$false;original_task_changed=$false})
    throw
} finally {
    if ($registered) {
        $remaining = Get-ScheduledTask -TaskName $smokeName -ErrorAction SilentlyContinue
        if ($remaining -and [string]$remaining.State -ne 'Running') {
            Unregister-ScheduledTask -TaskName $smokeName -Confirm:$false
            Save-ResearchJson 'temporary_task_cleanup.json' ([ordered]@{task_name=$smokeName;removed=$true;original_task_changed=$false})
        }
    }
}

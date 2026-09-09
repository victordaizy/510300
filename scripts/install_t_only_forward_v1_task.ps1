param(
    [string]$TaskName = "Codex-510300-T-Only-Forward-V1",
    [string]$RunAt = "16:30"
)

$ErrorActionPreference = "Stop"
$frozenTaskName = "Codex-510300-T-Only-Forward-V1"
$frozenRunAt = "16:30"
if ($TaskName -ne $frozenTaskName) {
    throw "任务名已冻结为：$frozenTaskName"
}
if ($RunAt -ne $frozenRunAt) {
    throw "运行时间已冻结为：$frozenRunAt"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "config\t_only_forward_v1_automation_manifest.json"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\run_t_only_forward_v1_daily.py"

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "自动运行冻结清单缺失：$manifestPath"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "项目Python环境缺失：$python"
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw "日更脚本缺失：$runner"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($sectionName in @("automation_files", "research_freeze_files")) {
    $section = $manifest.$sectionName
    foreach ($property in $section.PSObject.Properties) {
        $path = Join-Path $projectRoot ($property.Name.Replace("/", "\"))
        if (-not (Test-Path -LiteralPath $path)) {
            throw "冻结文件缺失：$($property.Name)"
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $property.Value) {
            throw "冻结文件指纹变化：$($property.Name)"
        }
    }
}

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$runner`"" `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $RunAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RunOnlyIfNetworkAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "510300 T_ONLY冻结前瞻影子验证。只更新研究数据和影子账本；不读取真实持仓，不生成订单，不连接券商。"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
$registered = Get-ScheduledTask -TaskName $TaskName
$info = Get-ScheduledTaskInfo -TaskName $TaskName

Write-Output "计划任务已安装：$($registered.TaskName)"
Write-Output "当前状态：$($registered.State)"
Write-Output "下次运行：$($info.NextRunTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "运行身份：仅用户登录时，最低权限"
Write-Output "安全边界：仅影子账本；真实仓位映射、订单生成、券商连接均关闭"

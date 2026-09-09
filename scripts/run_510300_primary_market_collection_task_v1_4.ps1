param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$task = Join-Path $projectRoot "scripts\run_510300_primary_market_collection_task_v1_3.ps1"
$runner = Join-Path $projectRoot "scripts\run_510300_primary_market_forward_v1_4.ps1"
# 沿用原直接回执目录和任务身份，仅替换修复数据保存路径的运行器。
& $task -RunnerPath $runner
exit $LASTEXITCODE

$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $PSScriptRoot
$researchPython = Join-Path $researchRoot '.venv\Scripts\python.exe'
$researchEntry = Join-Path $researchRoot 'research\selected_mix_reappraisal_v1.py'
$researchReport = Join-Path $researchRoot 'research\selected_mix_reappraisal_delivery_v1.py'
$researchOutput = Join-Path $researchRoot 'reports\research\510300_selected_mix_reappraisal_v1'

if (-not (Test-Path -LiteralPath $researchPython -PathType Leaf)) {
    throw '项目 Python 环境不存在，请先恢复项目原有 .venv 环境。'
}

Push-Location -LiteralPath $researchRoot
try {
    $researchStages = @(
        @{ Command = 'freeze'; Receipt = 'freeze.json' },
        @{ Command = 'reproduce'; Receipt = 'reproduction_receipt.json' },
        @{ Command = 'adjudicate'; Receipt = 'result.json' }
    )
    foreach ($researchStage in $researchStages) {
        if (-not (Test-Path -LiteralPath (Join-Path $researchOutput $researchStage.Receipt))) {
            if ($researchStage.Command -eq 'reproduce' -and
                (Test-Path -LiteralPath (Join-Path $researchOutput 'REPRODUCTION_STARTED.json'))) {
                throw '原版复现已有启动记录而没有完成回执，请先检查原运行；本入口不会覆盖中断记录。'
            }
            & $researchPython -X utf8 $researchEntry $researchStage.Command
            if ($LASTEXITCODE -ne 0) {
                throw ('研究步骤失败：' + $researchStage.Command)
            }
        }
    }
    & $researchPython -X utf8 $researchReport
    if ($LASTEXITCODE -ne 0) { throw '固定结果展示失败。' }
    & $researchPython -X utf8 $researchEntry status
    if ($LASTEXITCODE -ne 0) { throw '研究状态读取失败。' }
    Write-Output '原版固定研究已完成。重复运行复用已有结果；当前资料不足以形成当日信号。'
}
finally {
    Pop-Location
}

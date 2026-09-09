[CmdletBinding()]
param(
    [switch]$KeepTemporaryWorktree
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GitLongPathArgs = @("-c", "core.longpaths=true")
$Bad10ClaimPath = Join-Path $ProjectRoot (
    "reports\audit\510300_asymmetric_stress_hazard_v1_bad10_census_claim.json"
)
$Bad10ReceiptPath = Join-Path $ProjectRoot (
    "reports\audit\510300_asymmetric_stress_hazard_v1_bad10_census_receipt.json"
)
$Bad10FailureReceiptPath = Join-Path $ProjectRoot (
    "reports\audit\510300_asymmetric_stress_hazard_v1_bad10_census_" +
    "failure_receipt.json"
)
$Bad10StatusPath = Join-Path $ProjectRoot (
    "reports\research\510300_asymmetric_stress_hazard_v1_status.json"
)
$FocusedTestFiles = @(
    "tests/test_project_evidence_contract_v1.py",
    "tests/test_510300_asymmetric_stress_hazard_v1.py"
)
$TemporaryBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$TemporaryWorktree = Join-Path (
    $TemporaryBase
) ("codex-510300-g0-replay-" + [guid]::NewGuid().ToString("N"))
$WorktreeCreated = $false
$TestExitCode = $null

function Assert-TemporaryPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $Resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $Resolved.StartsWith(
            $TemporaryBase,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        throw "临时路径越界，拒绝继续：$Resolved"
    }
}

Assert-TemporaryPath -Path $TemporaryWorktree
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "项目虚拟环境Python不存在：$PythonExecutable"
}
if (Test-Path -LiteralPath $TemporaryWorktree) {
    throw "临时worktree目标已存在：$TemporaryWorktree"
}

Push-Location $ProjectRoot
try {
    $InitialClaimDetected = Test-Path `
        -LiteralPath $Bad10ClaimPath `
        -PathType Leaf
    $InitialReceiptDetected = Test-Path `
        -LiteralPath $Bad10ReceiptPath `
        -PathType Leaf
    $InitialFailureReceiptDetected = Test-Path `
        -LiteralPath $Bad10FailureReceiptPath `
        -PathType Leaf

    if ($InitialFailureReceiptDetected) {
        Write-Host "[G0] 检测到census失败回执；保留失败且禁止重跑。"
        $ProtocolG0ExecutionStatus = "NOT_RUN_CENSUS_FAILURE_RECEIPT_PRESENT"
        $ProtocolLifecycleVerification = "FAILURE_RECEIPT_PRESENT"
    }
    elseif ($InitialReceiptDetected) {
        if (-not $InitialClaimDetected) {
            throw "BAD10完成回执存在但一次性claim缺失"
        }
        Write-Host "[G0] census已消费；跳过pre-census G0并只读验证结果回执。"
        & $PythonExecutable `
            -m scripts.run_510300_asymmetric_stress_hazard_v1_bad10_census `
            --phase verify-results
        if ($LASTEXITCODE -ne 0) {
            throw "BAD10已完成回执验证失败，退出码：$LASTEXITCODE"
        }
        $ProtocolG0ExecutionStatus = "NOT_RUN_CENSUS_ALREADY_CONSUMED"
        $ProtocolLifecycleVerification = (
            "PASS_BAD10_CENSUS_RESULTS_AND_RECEIPT_VERIFIED"
        )
    }
    elseif ($InitialClaimDetected) {
        throw "BAD10一次性claim存在但没有终态回执；禁止G0或census重跑"
    }
    else {
        Write-Host "[G0] 运行冻结协议的只读pre-census工程准入。"
        & $PythonExecutable `
            -m scripts.run_510300_asymmetric_stress_hazard_v1_bad10_census `
            --phase g0
        if ($LASTEXITCODE -ne 0) {
            throw "冻结协议G0失败，退出码：$LASTEXITCODE"
        }
        $ProtocolG0ExecutionStatus = "PASS_G0_READY_FOR_ONE_SHOT_BAD10_CENSUS"
        $ProtocolLifecycleVerification = "PASS_PRE_CENSUS_G0"
    }

    $Head = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Head) {
        throw "无法读取当前Git HEAD"
    }

    Write-Host "[G0] 创建不挂载真实数据根的临时worktree：$Head"
    & git @GitLongPathArgs worktree add --detach -- $TemporaryWorktree $Head
    if ($LASTEXITCODE -ne 0) {
        throw "创建临时worktree失败，退出码：$LASTEXITCODE"
    }
    $WorktreeCreated = $true

    Write-Host "[G0] 在隔离worktree运行项目证据契约与本分支核心测试。"
    Push-Location $TemporaryWorktree
    try {
        $env:PYTHONDONTWRITEBYTECODE = "1"
        & $PythonExecutable -m pytest -q @FocusedTestFiles
        $TestExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($TestExitCode -ne 0) {
        throw "隔离worktree聚焦测试失败，退出码：$TestExitCode"
    }

    $ExistingClaimDetected = Test-Path `
        -LiteralPath $Bad10ClaimPath `
        -PathType Leaf
    $ExistingReceiptDetected = Test-Path `
        -LiteralPath $Bad10ReceiptPath `
        -PathType Leaf
    $ExistingFailureReceiptDetected = Test-Path `
        -LiteralPath $Bad10FailureReceiptPath `
        -PathType Leaf

    if (
        $ExistingClaimDetected -ne $InitialClaimDetected -or
        $ExistingReceiptDetected -ne $InitialReceiptDetected -or
        $ExistingFailureReceiptDetected -ne $InitialFailureReceiptDetected
    ) {
        throw "隔离G0复验期间BAD10一次性生命周期发生变化"
    }

    if ($ExistingFailureReceiptDetected) {
        $CensusLifecycle = "ONE_SHOT_FAILED_DO_NOT_RERUN"
        $NextAllowedAction = "STOP_AND_PRESERVE_FAILURE_RECEIPT"
        $ReceiptVerification = "FAILURE_RECEIPT_PRESENT"
    }
    elseif ($ExistingReceiptDetected) {
        if (-not $ExistingClaimDetected) {
            throw "BAD10完成回执存在但一次性claim缺失"
        }
        if (-not (Test-Path -LiteralPath $Bad10StatusPath -PathType Leaf)) {
            throw "BAD10完成回执存在但权威状态文件缺失"
        }
        $Bad10Status = Get-Content -LiteralPath $Bad10StatusPath -Raw |
            ConvertFrom-Json
        $CensusLifecycle = "ONE_SHOT_CONSUMED_VERIFIED_DO_NOT_RERUN"
        $NextAllowedAction = [string]$Bad10Status.next_allowed_action
        $ReceiptVerification = "PASS_BAD10_CENSUS_RESULTS_AND_RECEIPT_VERIFIED"
    }
    elseif ($ExistingClaimDetected) {
        $CensusLifecycle = "CLAIM_EXISTS_WITHOUT_TERMINAL_RECEIPT_INVESTIGATE"
        $NextAllowedAction = "INVESTIGATE_CLAIM_WITHOUT_RERUN"
        $ReceiptVerification = "NOT_TERMINAL"
    }
    else {
        $CensusLifecycle = "NOT_CLAIMED"
        $NextAllowedAction = "ONE_SHOT_BAD10_INDEPENDENT_EVENT_CENSUS_ONLY"
        $ReceiptVerification = "NOT_APPLICABLE"
    }

    [ordered]@{
        status = "PASS_G0_PROJECT_ENGINEERING_REPLAY"
        git_head = $Head
        protocol_g0_execution_status = $ProtocolG0ExecutionStatus
        protocol_lifecycle_verification = $ProtocolLifecycleVerification
        focused_pytest_exit_code = $TestExitCode
        focused_test_files = $FocusedTestFiles
        focused_test_file_count = @($FocusedTestFiles).Count
        external_data_copy_performed = $false
        writable_data_junction_created = $false
        g0_execution_label_values_read = $false
        existing_bad10_one_shot_claim_detected = $ExistingClaimDetected
        existing_bad10_census_receipt_detected = $ExistingReceiptDetected
        existing_bad10_failure_receipt_detected = $ExistingFailureReceiptDetected
        bad10_census_lifecycle = $CensusLifecycle
        census_receipt_verification = $ReceiptVerification
        next_allowed_action = $NextAllowedAction
        position_impact = 0
        trading_authorized = $false
    } | ConvertTo-Json -Depth 5
}
finally {
    Pop-Location
    if ($WorktreeCreated -and -not $KeepTemporaryWorktree) {
        Write-Host "[G0] 清理隔离worktree。"
        Assert-TemporaryPath -Path $TemporaryWorktree
        & git -C $ProjectRoot @GitLongPathArgs worktree remove --force -- $TemporaryWorktree
        $GitRemoveExitCode = $LASTEXITCODE
        if (Test-Path -LiteralPath $TemporaryWorktree) {
            $ExtendedTemporaryPath = "\\?\" + (
                [System.IO.Path]::GetFullPath($TemporaryWorktree)
            )
            [System.IO.Directory]::Delete($ExtendedTemporaryPath, $true)
        }
        & git -C $ProjectRoot @GitLongPathArgs worktree prune
        if (Test-Path -LiteralPath $TemporaryWorktree) {
            throw "临时worktree仍有残留：$TemporaryWorktree"
        }
        if ($GitRemoveExitCode -ne 0) {
            Write-Warning (
                "Git受Windows长路径影响未完整清理，已用已验证的扩展临时路径完成清理；" +
                "git_exit_code=$GitRemoveExitCode"
            )
        }
    }
}

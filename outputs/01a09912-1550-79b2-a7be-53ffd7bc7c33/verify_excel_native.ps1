$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $taskRoot)
$workbookPath = Join-Path $taskRoot '510300_D60每日手算.xlsx'
$inputPath = Join-Path $workspaceRoot 'reports\research\510300_daily_manual_signal_v1\2026-09-11\工作簿输入.json'
$seed = Get-Content -LiteralPath $inputPath -Raw -Encoding UTF8 | ConvertFrom-Json
$originalHash = (Get-FileHash -LiteralPath $workbookPath -Algorithm SHA256).Hash
$excel = $null
$book = $null
$dataSheet = $null
$summarySheet = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $book = $excel.Workbooks.Open($workbookPath, 0, $true)
    $dataSheet = $book.Worksheets.Item('每日数据')
    $summarySheet = $book.Worksheets.Item('日常操作')
    $excel.CalculateFullRebuild()
    $rowEnd = $seed.rows.Count + 1
    $actual = $dataSheet.Range("K61:M$rowEnd").Value2
    $largestError = 0.0
    for ($i = 59; $i -lt $seed.rows.Count; $i++) {
        $localRow = $i - 58
        $expected = [double]$seed.rows[$i].score60
        if ($actual[$localRow,1] -isnot [double]) {
            foreach ($debugCell in @('A2','A61','N61','F61','G61','H61','I61','J61','K61')) {
                Write-Output "$debugCell 值=$($dataSheet.Range($debugCell).Value2) 公式=$($dataSheet.Range($debugCell).Formula)"
            }
            throw "第 $($i+2) 行 Excel 因子不是数值"
        }
        $delta = [Math]::Abs([double]$actual[$localRow,1] - $expected)
        $largestError = [Math]::Max($largestError, $delta)
        if ($delta -gt 1e-10) { throw "第 $($i+2) 行 Excel 与源码分数不一致" }
        if ($i -gt 59) {
            if ([int]$actual[$localRow,2] -ne [int]$seed.rows[$i].entry_condition_2days) { throw '入场条件不一致' }
            if ([int]$actual[$localRow,3] -ne [int]$seed.rows[$i].difference_exit_condition_2days) { throw '差值退出条件不一致' }
        }
    }
    $lastScore = [double]$dataSheet.Range("K$rowEnd").Value2
    $summarySheet.Range('B15').Value2 = $lastScore
    $excel.CalculateFullRebuild()
    if ([int]$dataSheet.Range("L$rowEnd").Value2 -ne 0) { throw '严格大于阈值的等号边界失败' }
    $summarySheet.Range('B15').Value2 = [double]1
    $newRow = $rowEnd + 1
    $previousClose = [double]$seed.rows[-1].close
    $dataSheet.Range("A$newRow").Value2 = [datetime]::ParseExact('2026-09-14','yyyy-MM-dd',$null).ToOADate()
    $dataSheet.Range("C$newRow").Value2 = $previousClose
    $dataSheet.Range("D$newRow").Value2 = $previousClose + 0.01
    $excel.CalculateFullRebuild()
    if ($dataSheet.Range("K$newRow").Value2 -ne '') { throw '缺失分红被误当成0' }
    $dataSheet.Range("E$newRow").Value2 = [double]0
    $excel.CalculateFullRebuild()
    if ($dataSheet.Range("K$newRow").Value2 -isnot [double]) { throw '续填新日期未完成重算' }
    $futureScore = [double]$dataSheet.Range("K$newRow").Value2
    $dataSheet.Range("E$newRow").Value2 = [double]0.123
    $excel.CalculateFullRebuild()
    $expectedNight = [Math]::Log(($previousClose + 0.123) / $previousClose)
    $expectedIntraday = [Math]::Log(($previousClose + 0.01 + 0.123) / ($previousClose + 0.123))
    if ([Math]::Abs([double]$dataSheet.Range("H$newRow").Value2 - ($expectedIntraday-$expectedNight)) -gt 1e-12) { throw '除息现金变更未按现版财富分解重算' }
    $zeroStart = $newRow - 59
    $dataSheet.Range("B${zeroStart}:D$newRow").Value2 = [double]4
    $dataSheet.Range("E${zeroStart}:E$newRow").Value2 = [double]0
    $excel.CalculateFullRebuild()
    if ($dataSheet.Range("K$newRow").Value2 -ne '') { throw '标准差为0仍产生因子' }
    # 可复制文本另外在新建的内存工作表中检查，关闭时不保存。
    $formulaPath = Join-Path $taskRoot 'Excel公式_可复制.txt'
    $textFormulaChecked = $false
    if (Test-Path -LiteralPath $formulaPath) {
        $copySheet = $book.Worksheets.Add()
        $copySheet.Name = '临时文本公式验证'
        for ($j = 0; $j -lt $seed.rows.Count; $j++) {
            $r = $j + 2
            $copySheet.Range("A$r").Value2 = [datetime]::ParseExact($seed.rows[$j].date,'yyyy-MM-dd',$null).ToOADate()
            $copySheet.Range("B$r").Value2 = [double]$seed.rows[$j].previous_close
            $copySheet.Range("C$r").Value2 = [double]$seed.rows[$j].open
            $copySheet.Range("D$r").Value2 = [double]$seed.rows[$j].close
            $copySheet.Range("E$r").Value2 = [double]$seed.rows[$j].cash_dividend_per_share
        }
        $formulaLines = @(Get-Content -LiteralPath $formulaPath -Encoding UTF8 | Where-Object { $_.StartsWith('=') })
        $addresses = @('B3','F2','G2','H2','I61','J61','K61','K61','L62','M62')
        if ($formulaLines.Count -ne $addresses.Count) { throw '可复制公式数量与定位不一致' }
        for ($j = 0; $j -lt $addresses.Count; $j++) {
            $addr = $addresses[$j]
            $copySheet.Range($addr).Formula = $formulaLines[$j]
            $col = $addr -replace '[0-9]',''
            $copySheet.Range("${addr}:$col$rowEnd").FillDown()
            $excel.CalculateFullRebuild()
            if ($j -in @(6,7)) {
                if ([Math]::Abs([double]$copySheet.Range("K$rowEnd").Value2-$lastScore) -gt 1e-10) { throw '可复制的D60公式与交付分数不一致' }
            }
        }
        if ([int]$copySheet.Range("L$rowEnd").Value2 -ne 1 -or [int]$copySheet.Range("M$rowEnd").Value2 -ne 0) { throw '可复制公式的两日条件有误' }
        $textFormulaChecked = $true
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($copySheet)
    }
    $receipt = [ordered]@{
        status = 'PASS_NATIVE_MICROSOFT_EXCEL_RECALCULATION'
        excel_version = $excel.Version
        checked_factor_rows = $actual.GetLength(0)
        max_absolute_error = $largestError
        latest_score = $lastScore
        next_day_input_test_score = $futureScore
        checks = @('历史因子及两日门槛与源码一致','严格大于阈值边界','缺失分红不当成0','下一交易日实时重算','除息现金重算','零标准差不产生因子')
        workbook_opened_read_only = $true
        workbook_saved = $false
        copyable_text_formulas_pasted_and_recalculated = $textFormulaChecked
    }
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRoot 'Excel原生重算核验.json') -Encoding UTF8
    Write-Output 'Microsoft Excel 原生重算与续填检查通过，交付文件保持原样。'
} finally {
    if ($null -ne $book) { $book.Close($false) }
    if ($null -ne $excel) { $excel.Quit() }
    foreach ($comObject in @($summarySheet,$dataSheet,$book,$excel)) {
        if ($null -ne $comObject) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($comObject) }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
if ((Get-FileHash -LiteralPath $workbookPath -Algorithm SHA256).Hash -ne $originalHash) { throw '原生核验意外改变交付文件' }

param(
    [string[]]$ImagePath = @(),
    [string]$ImageListPath
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if ($ImageListPath) {
    $listedPaths = Get-Content -LiteralPath $ImageListPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $ImagePath = @($listedPaths)
}
if (@($ImagePath).Count -eq 0) {
    throw "At least one image path is required"
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

$asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
        $_.Name -eq "AsTask" -and
        $_.IsGenericMethod -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } |
    Select-Object -First 1

if ($null -eq $asTaskGeneric) {
    throw "Windows Runtime generic AsTask bridge was not found"
}

function Wait-WindowsRuntimeOperation {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Operation,
        [Parameter(Mandatory = $true)]
        [type]$ResultType
    )

    $task = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

$language = [Windows.Globalization.Language]::new("zh-Hans-CN")
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
    throw "Windows Simplified Chinese OCR engine is unavailable"
}

foreach ($candidatePath in $ImagePath) {
    $resolvedPath = (Resolve-Path -LiteralPath $candidatePath).Path
    $file = Wait-WindowsRuntimeOperation `
        -Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPath)) `
        -ResultType ([Windows.Storage.StorageFile])
    $stream = Wait-WindowsRuntimeOperation `
        -Operation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) `
        -ResultType ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Wait-WindowsRuntimeOperation `
            -Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) `
            -ResultType ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Wait-WindowsRuntimeOperation `
            -Operation ($decoder.GetSoftwareBitmapAsync()) `
            -ResultType ([Windows.Graphics.Imaging.SoftwareBitmap])
        try {
            if (
                $bitmap.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
                $bitmap.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension
            ) {
                throw "Rendered page exceeds Windows OCR maximum image dimension"
            }
            $result = Wait-WindowsRuntimeOperation `
                -Operation ($engine.RecognizeAsync($bitmap)) `
                -ResultType ([Windows.Media.Ocr.OcrResult])
            $lineRows = @(
                foreach ($line in $result.Lines) {
                    $wordRows = @(
                        foreach ($word in $line.Words) {
                            [ordered]@{
                                text = $word.Text
                                x = [double]$word.BoundingRect.X
                                y = [double]$word.BoundingRect.Y
                                width = [double]$word.BoundingRect.Width
                                height = [double]$word.BoundingRect.Height
                            }
                        }
                    )
                    [ordered]@{
                        text = $line.Text
                        words = $wordRows
                    }
                }
            )
            [ordered]@{
                image_path = $resolvedPath
                language_tag = $language.LanguageTag
                ocr_engine = "Windows.Media.Ocr.OcrEngine"
                max_image_dimension = [int][Windows.Media.Ocr.OcrEngine]::MaxImageDimension
                pixel_width = [int]$bitmap.PixelWidth
                pixel_height = [int]$bitmap.PixelHeight
                text_angle = if ($null -eq $result.TextAngle) { $null } else { [double]$result.TextAngle }
                text = $result.Text
                lines = $lineRows
            } | ConvertTo-Json -Depth 8 -Compress
        }
        finally {
            if ($bitmap -is [System.IDisposable]) {
                $bitmap.Dispose()
            }
        }
    }
    finally {
        if ($stream -is [System.IDisposable]) {
            $stream.Dispose()
        }
    }
}

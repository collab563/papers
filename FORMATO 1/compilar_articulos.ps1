param(
    [switch]$Watch
)

$ErrorActionPreference = "Stop"
$formatDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$compiler = Get-Command lualatex -ErrorAction SilentlyContinue
if (-not $compiler) {
    throw "No se encontró lualatex. Instala MiKTeX/TeX Live o agrega lualatex al PATH."
}

$buildRoot = Join-Path $env:TEMP "crown-order-paper-build"
$documents = @(
    @{ Source = "aticulo.tex"; Pdf = "aticulo.pdf"; Build = "english" },
    @{ Source = "articulo_es.tex"; Pdf = "articulo_es.pdf"; Build = "spanish" }
)

function Build-Papers {
    foreach ($document in $documents) {
        $sourcePath = Join-Path $formatDirectory $document.Source
        $outputDirectory = Join-Path $buildRoot $document.Build
        New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
        $logPath = Join-Path $outputDirectory "build.log"

        for ($pass = 1; $pass -le 2; $pass++) {
            & $compiler.Source `
                "-interaction=nonstopmode" `
                "-halt-on-error" `
                "-file-line-error" `
                "-output-directory=$outputDirectory" `
                $sourcePath *> $logPath

            if ($LASTEXITCODE -ne 0) {
                Get-Content $logPath -Tail 80
                throw "Falló la compilación de $($document.Source), pasada $pass."
            }
        }

        $builtPdf = Join-Path $outputDirectory $document.Pdf
        $targetPdf = Join-Path $formatDirectory $document.Pdf
        if (-not (Test-Path $builtPdf)) {
            throw "La compilación terminó sin generar $builtPdf."
        }
        Copy-Item -LiteralPath $builtPdf -Destination $targetPdf -Force

        $warnings = Select-String -Path $logPath -Pattern `
            "Overfull|LaTeX Warning|undefined references"
        if ($warnings) {
            Write-Warning "Revisa las advertencias de $($document.Source):"
            $warnings | ForEach-Object { Write-Warning $_.Line }
        }
        else {
            Write-Host "OK: $($document.Pdf)"
        }
    }
}

Build-Papers

if ($Watch) {
    Write-Host "Vigilando aticulo.tex y articulo_es.tex. Ctrl+C para terminar."
    $sources = $documents | ForEach-Object {
        Join-Path $formatDirectory $_.Source
    }
    $lastWrite = @{}
    foreach ($source in $sources) {
        $lastWrite[$source] = (Get-Item -LiteralPath $source).LastWriteTimeUtc
    }

    while ($true) {
        Start-Sleep -Seconds 2
        $changed = $false
        foreach ($source in $sources) {
            $currentWrite = (Get-Item -LiteralPath $source).LastWriteTimeUtc
            if ($currentWrite -ne $lastWrite[$source]) {
                $lastWrite[$source] = $currentWrite
                $changed = $true
            }
        }
        if ($changed) {
            Start-Sleep -Milliseconds 800
            Build-Papers
            foreach ($source in $sources) {
                $lastWrite[$source] = (Get-Item -LiteralPath $source).LastWriteTimeUtc
            }
        }
    }
}

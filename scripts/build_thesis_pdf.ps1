[CmdletBinding()]
param(
    [string]$TexDir = "",
    [string]$TexFile = "main.tex",
    [switch]$Clean,
    [switch]$Open
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RequiredCommand {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Nie znaleziono polecenia '$Name' w PATH. Zainstaluj MiKTeX/TeX Live albo dodaj katalog binarny TeX do PATH."
    }

    return $command.Source
}

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host ("> " + $Executable + " " + ($Arguments -join " "))
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Polecenie zakonczylo sie bledem: $Executable $($Arguments -join ' ')"
    }
}

if ([string]::IsNullOrWhiteSpace($TexDir)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $TexDir = Join-Path $scriptRoot "..\overleaf"
}

$texRoot = (Resolve-Path -LiteralPath $TexDir).Path
$texPath = Join-Path $texRoot $TexFile
if (-not (Test-Path -LiteralPath $texPath -PathType Leaf)) {
    throw "Nie znaleziono pliku TeX: $texPath"
}

$baseName = [System.IO.Path]::GetFileNameWithoutExtension($TexFile)
$pdflatex = Resolve-RequiredCommand "pdflatex"

if ($Clean) {
    $extensions = @(
        "aux",
        "bbl",
        "bcf",
        "blg",
        "fdb_latexmk",
        "fls",
        "lof",
        "log",
        "lot",
        "out",
        "run.xml",
        "synctex.gz",
        "toc"
    )

    foreach ($extension in $extensions) {
        $artifact = Join-Path $texRoot "$baseName.$extension"
        if (Test-Path -LiteralPath $artifact) {
            Remove-Item -LiteralPath $artifact -Force
        }
    }

    Get-ChildItem -LiteralPath $texRoot -Recurse -File -Filter "*-eps-converted-to.pdf" |
        Remove-Item -Force
}

Push-Location $texRoot
try {
    Invoke-Checked $pdflatex @("-interaction=nonstopmode", "-halt-on-error", $TexFile)

    $auxPath = Join-Path $texRoot "$baseName.aux"
    $hasCitations = (Test-Path -LiteralPath $auxPath) -and
        [bool](Select-String -LiteralPath $auxPath -Pattern "\\citation\{" -Quiet)

    if ($hasCitations) {
        $bibtex = Resolve-RequiredCommand "bibtex"
        Invoke-Checked $bibtex @($baseName)
    }
    else {
        Write-Host "Brak cytowan w $baseName.aux, pomijam BibTeX."
    }

    Invoke-Checked $pdflatex @("-interaction=nonstopmode", "-halt-on-error", $TexFile)
    Invoke-Checked $pdflatex @("-interaction=nonstopmode", "-halt-on-error", $TexFile)
}
finally {
    Pop-Location
}

$pdfPath = Join-Path $texRoot "$baseName.pdf"
if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
    throw "Kompilacja zakonczyla sie bez utworzenia pliku PDF: $pdfPath"
}

Write-Host "PDF zapisany: $pdfPath"

if ($Open) {
    Start-Process -FilePath $pdfPath
}

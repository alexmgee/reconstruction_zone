param(
    [Parameter(Mandatory = $true)]
    [string]$DistDir,

    [string]$PythonExe = "C:\Python314\python.exe"
)

$ErrorActionPreference = "Stop"

$dist = (Resolve-Path -LiteralPath $DistDir).Path

$cfgSource = & $PythonExe -c "import pathlib, ultralytics; print(pathlib.Path(ultralytics.__file__).resolve().parent / 'cfg')"
if (-not $cfgSource) {
    throw "Could not resolve ultralytics cfg directory"
}

$cfgSource = (Resolve-Path -LiteralPath $cfgSource).Path
$ultralyticsDest = Join-Path $dist "ultralytics"
$cfgDest = Join-Path $ultralyticsDest "cfg"

New-Item -ItemType Directory -Force -Path $ultralyticsDest | Out-Null

if (Test-Path -LiteralPath $cfgDest) {
    Remove-Item -LiteralPath $cfgDest -Recurse -Force
}

Copy-Item -LiteralPath $cfgSource -Destination $ultralyticsDest -Recurse -Force

$defaultCfg = Join-Path $cfgDest "default.yaml"
if (-not (Test-Path -LiteralPath $defaultCfg)) {
    throw "Ultralytics default.yaml was not copied to $defaultCfg"
}

$files = Get-ChildItem -LiteralPath $cfgDest -Recurse -File
$bytes = ($files | Measure-Object -Property Length -Sum).Sum

Write-Host "Copied Ultralytics cfg runtime data to $cfgDest"
Write-Host "Files: $($files.Count)"
Write-Host "Bytes: $bytes"

param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [Parameter(Mandatory = $true)]
    [string]$CaseName,

    [int]$DurationSeconds = 20,

    [switch]$WithFixtureModels,

    [switch]$SetupTestMode
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$exe = Resolve-Path -LiteralPath $ExePath
$caseRoot = Join-Path $root "dist_test\new-install-sandbox\$CaseName"
$appHome = Join-Path $caseRoot "app-home"
$modelDir = Join-Path $appHome "models"
$hfHome = Join-Path $caseRoot "hf-home"
$logDir = Join-Path $caseRoot "launch-logs"
$probeDir = Join-Path $caseRoot "probe-results"

New-Item -ItemType Directory -Force -Path $modelDir, $hfHome, $logDir, $probeDir | Out-Null

if ($WithFixtureModels) {
    $fixtures = @(
        @{ Source = Join-Path $root "rf-detr-seg-small.pt"; Dest = Join-Path $modelDir "rf-detr-seg-small.pt" },
        @{ Source = Join-Path $root "yolo26n-seg.pt"; Dest = Join-Path $modelDir "yolo26n-seg.pt" }
    )
    foreach ($fixture in $fixtures) {
        if (Test-Path -LiteralPath $fixture.Source) {
            Copy-Item -LiteralPath $fixture.Source -Destination $fixture.Dest -Force
        }
    }
}

$out = Join-Path $logDir "stdout.txt"
$err = Join-Path $logDir "stderr.txt"
$summary = Join-Path $probeDir "summary.txt"
$envFile = Join-Path $probeDir "environment.txt"
$modelListing = Join-Path $probeDir "models.txt"
$crashTail = Join-Path $probeDir "crash-log-tail.txt"

$oldAppHome = $env:RECONSTRUCTION_ZONE_APP_HOME
$oldModelDir = $env:RECONSTRUCTION_ZONE_MODEL_DIR
$oldHfHome = $env:HF_HOME
$oldTransformersCache = $env:TRANSFORMERS_CACHE
$oldSetupTestMode = $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE
$oldStrictModelDirs = $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS

try {
    $env:RECONSTRUCTION_ZONE_APP_HOME = $appHome
    $env:RECONSTRUCTION_ZONE_MODEL_DIR = $modelDir
    $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = "1"
    $env:HF_HOME = $hfHome
    $env:TRANSFORMERS_CACHE = Join-Path $hfHome "transformers"
    if ($SetupTestMode) {
        $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE = "1"
    } else {
        Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue
    }

    @(
        "CASE=$CaseName"
        "EXE=$exe"
        "APP_HOME=$env:RECONSTRUCTION_ZONE_APP_HOME"
        "MODEL_DIR=$env:RECONSTRUCTION_ZONE_MODEL_DIR"
        "STRICT_MODEL_DIRS=$env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"
        "HF_HOME=$env:HF_HOME"
        "TRANSFORMERS_CACHE=$env:TRANSFORMERS_CACHE"
        "SETUP_TEST_MODE=$env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE"
    ) | Set-Content -LiteralPath $envFile -Encoding utf8

    $workingDirectory = Split-Path -Parent $exe
    $p = Start-Process -FilePath $exe -WorkingDirectory $workingDirectory -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Start-Sleep -Seconds $DurationSeconds
    $p.Refresh()

    if ($p.HasExited) {
        "PROCESS_EXITED`t$($p.Id)`t$($p.ExitCode)" | Set-Content -LiteralPath $summary -Encoding utf8
    } else {
        "PROCESS_STILL_RUNNING`t$($p.Id)" | Set-Content -LiteralPath $summary -Encoding utf8
        Stop-Process -Id $p.Id -Force
        "PROCESS_STOPPED`t$($p.Id)" | Add-Content -LiteralPath $summary -Encoding utf8
    }

    Get-ChildItem -LiteralPath $modelDir -Force -ErrorAction SilentlyContinue |
        Select-Object Name,Length,LastWriteTime |
        Format-Table -AutoSize |
        Out-String |
        Set-Content -LiteralPath $modelListing -Encoding utf8

    $crashLog = Join-Path $appHome "logs\crash.log"
    if (Test-Path -LiteralPath $crashLog) {
        Get-Content -LiteralPath $crashLog -Tail 120 | Set-Content -LiteralPath $crashTail -Encoding utf8
    } else {
        "NO_CRASH_LOG" | Set-Content -LiteralPath $crashTail -Encoding utf8
    }

    Get-Content -LiteralPath $summary
    Write-Output "CASE_ROOT`t$caseRoot"
    Write-Output "STDOUT`t$out"
    Write-Output "STDERR`t$err"
    Write-Output "SUMMARY`t$summary"
    Write-Output "CRASH_TAIL`t$crashTail"
} finally {
    if ($null -eq $oldAppHome) { Remove-Item Env:\RECONSTRUCTION_ZONE_APP_HOME -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_APP_HOME = $oldAppHome }
    if ($null -eq $oldModelDir) { Remove-Item Env:\RECONSTRUCTION_ZONE_MODEL_DIR -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_MODEL_DIR = $oldModelDir }
    if ($null -eq $oldHfHome) { Remove-Item Env:\HF_HOME -ErrorAction SilentlyContinue } else { $env:HF_HOME = $oldHfHome }
    if ($null -eq $oldTransformersCache) { Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue } else { $env:TRANSFORMERS_CACHE = $oldTransformersCache }
    if ($null -eq $oldSetupTestMode) { Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE = $oldSetupTestMode }
    if ($null -eq $oldStrictModelDirs) { Remove-Item Env:\RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = $oldStrictModelDirs }
}

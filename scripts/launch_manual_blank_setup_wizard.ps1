param(
    [string]$ExePath = "",
    [string]$CaseName = "",
    [switch]$Reset,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not $ExePath) {
    $ExePath = Join-Path $root "dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe"
}

$exe = (Resolve-Path -LiteralPath $ExePath).Path

if (-not $CaseName) {
    $CaseName = "manual-blank-setup-wizard-{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")
}

$sandboxBase = [System.IO.Path]::GetFullPath((Join-Path $root "dist_test\new-install-sandbox"))
$caseRoot = [System.IO.Path]::GetFullPath((Join-Path $sandboxBase $CaseName))

if (-not $caseRoot.StartsWith($sandboxBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use case root outside sandbox base: $caseRoot"
}

if ($Reset -and (Test-Path -LiteralPath $caseRoot)) {
    Remove-Item -LiteralPath $caseRoot -Recurse -Force
}

$appHome = Join-Path $caseRoot "app-home"
$modelDir = Join-Path $appHome "models"
$hfHome = Join-Path $caseRoot "hf-home"
$logDir = Join-Path $caseRoot "launch-logs"
$notesDir = Join-Path $caseRoot "manual-test-notes"

New-Item -ItemType Directory -Force -Path $modelDir, $hfHome, $logDir, $notesDir | Out-Null

$stdout = Join-Path $logDir "stdout.txt"
$stderr = Join-Path $logDir "stderr.txt"
$envFile = Join-Path $caseRoot "manual-test-environment.txt"
$readme = Join-Path $caseRoot "README-manual-test.txt"
$exitFile = Join-Path $caseRoot "manual-test-exit.txt"

$oldAppHome = $env:RECONSTRUCTION_ZONE_APP_HOME
$oldModelDir = $env:RECONSTRUCTION_ZONE_MODEL_DIR
$oldStrictModelDirs = $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS
$oldHfHome = $env:HF_HOME
$oldTransformersCache = $env:TRANSFORMERS_CACHE
$oldSetupTestMode = $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE

try {
    $env:RECONSTRUCTION_ZONE_APP_HOME = $appHome
    $env:RECONSTRUCTION_ZONE_MODEL_DIR = $modelDir
    $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = "1"
    $env:HF_HOME = $hfHome
    $env:TRANSFORMERS_CACHE = Join-Path $hfHome "transformers"
    Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue

    @(
        "CASE=$CaseName"
        "CASE_ROOT=$caseRoot"
        "EXE=$exe"
        "APP_HOME=$env:RECONSTRUCTION_ZONE_APP_HOME"
        "MODEL_DIR=$env:RECONSTRUCTION_ZONE_MODEL_DIR"
        "STRICT_MODEL_DIRS=$env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"
        "HF_HOME=$env:HF_HOME"
        "TRANSFORMERS_CACHE=$env:TRANSFORMERS_CACHE"
        "STDOUT=$stdout"
        "STDERR=$stderr"
        "STARTED=$(Get-Date -Format o)"
    ) | Set-Content -LiteralPath $envFile -Encoding utf8

    @"
Manual blank setup-wizard test
==============================

This folder is an isolated new-install sandbox for Reconstruction Zone.

Use the app window that just opened and test the first-launch setup wizard as a
new user would.

Suggested checks:
1. Confirm the setup wizard appears.
2. Confirm it shows SAM3, RF-DETR-Seg, and YOLO26-n-seg as missing.
3. Confirm the Projects tab behind it shows 0 projects and an app-home tracker path.
4. Click Begin Setup.
5. Confirm the SAM3 Access screen appears.
6. Try Verify with an empty token and confirm it reports that a token is required.
7. If you want to test the real gated flow, paste your HuggingFace token and verify.
8. If verification succeeds, continue downloads and note whether RF-DETR/YOLO/SAM3 complete.

Sandbox:
$caseRoot

The app is launched with:
RECONSTRUCTION_ZONE_APP_HOME=$appHome
RECONSTRUCTION_ZONE_MODEL_DIR=$modelDir
RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS=1
HF_HOME=$hfHome
TRANSFORMERS_CACHE=$env:TRANSFORMERS_CACHE

Logs:
$stdout
$stderr

Write any observations into this folder:
$notesDir
"@ | Set-Content -LiteralPath $readme -Encoding utf8

    $workingDirectory = Split-Path -Parent $exe
    $process = Start-Process -FilePath $exe `
        -WorkingDirectory $workingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    "PROCESS_ID=$($process.Id)" | Add-Content -LiteralPath $envFile -Encoding utf8

    Write-Host "Launched Reconstruction Zone manual blank setup-wizard test."
    Write-Host "Process ID: $($process.Id)"
    Write-Host "Sandbox: $caseRoot"
    Write-Host "README: $readme"
    Write-Host "Logs: $logDir"
    Write-Host ""
    Write-Host "Close the app when you are done testing."

    if ($NoWait) {
        Write-Host "NoWait was specified; leaving app running and returning now."
    } else {
        Wait-Process -Id $process.Id
        $process.Refresh()
        @(
            "PROCESS_ID=$($process.Id)"
            "EXIT_CODE=$($process.ExitCode)"
            "FINISHED=$(Get-Date -Format o)"
        ) | Set-Content -LiteralPath $exitFile -Encoding utf8
        Write-Host "Manual test process exited with code $($process.ExitCode)."
        Write-Host "Exit record: $exitFile"
    }
} finally {
    if ($null -eq $oldAppHome) { Remove-Item Env:\RECONSTRUCTION_ZONE_APP_HOME -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_APP_HOME = $oldAppHome }
    if ($null -eq $oldModelDir) { Remove-Item Env:\RECONSTRUCTION_ZONE_MODEL_DIR -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_MODEL_DIR = $oldModelDir }
    if ($null -eq $oldStrictModelDirs) { Remove-Item Env:\RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = $oldStrictModelDirs }
    if ($null -eq $oldHfHome) { Remove-Item Env:\HF_HOME -ErrorAction SilentlyContinue } else { $env:HF_HOME = $oldHfHome }
    if ($null -eq $oldTransformersCache) { Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue } else { $env:TRANSFORMERS_CACHE = $oldTransformersCache }
    if ($null -eq $oldSetupTestMode) { Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE = $oldSetupTestMode }
}

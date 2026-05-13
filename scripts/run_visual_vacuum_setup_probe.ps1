param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [string[]]$Arguments = @(),

    [string]$WorkingDirectory = "",

    [Parameter(Mandatory = $true)]
    [string]$CaseName,

    [int]$WarmupSeconds = 14,

    [switch]$Reset
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$exe = (Resolve-Path -LiteralPath $ExePath).Path
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
$probeDir = Join-Path $caseRoot "probe-results"
$screenshotDir = Join-Path $caseRoot "screenshots"

New-Item -ItemType Directory -Force -Path $modelDir, $hfHome, $logDir, $probeDir, $screenshotDir | Out-Null

$out = Join-Path $logDir "stdout.txt"
$err = Join-Path $logDir "stderr.txt"
$summary = Join-Path $probeDir "summary.txt"
$envFile = Join-Path $probeDir "environment.txt"
$modelListing = Join-Path $probeDir "models.txt"
$crashTail = Join-Path $probeDir "crash-log-tail.txt"
$screenPng = Join-Path $screenshotDir "desktop-after-warmup.png"
$windowPng = Join-Path $screenshotDir "main-window-after-warmup.png"

function Save-PrimaryScreen {
    param([string]$Path)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Save-Window {
    param(
        [IntPtr]$Handle,
        [string]$Path
    )

    Add-Type -AssemblyName System.Drawing

    $rect = New-Object RzVisualProbe.Win32+RECT
    if (-not [RzVisualProbe.Win32]::GetWindowRect($Handle, [ref]$rect)) {
        throw "GetWindowRect failed for handle $Handle"
    }

    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -le 0 -or $height -le 0) {
        throw "Window bounds were empty: $($rect.Left),$($rect.Top),$($rect.Right),$($rect.Bottom)"
    }

    $bitmap = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, (New-Object System.Drawing.Size $width, $height))
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

namespace RzVisualProbe
{
    public static class Win32
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
    }
}
"@

$oldAppHome = $env:RECONSTRUCTION_ZONE_APP_HOME
$oldModelDir = $env:RECONSTRUCTION_ZONE_MODEL_DIR
$oldHfHome = $env:HF_HOME
$oldTransformersCache = $env:TRANSFORMERS_CACHE
$oldSetupTestMode = $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE
$oldStrictModelDirs = $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS

$process = $null

try {
    $env:RECONSTRUCTION_ZONE_APP_HOME = $appHome
    $env:RECONSTRUCTION_ZONE_MODEL_DIR = $modelDir
    $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = "1"
    $env:HF_HOME = $hfHome
    $env:TRANSFORMERS_CACHE = Join-Path $hfHome "transformers"
    Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue

    @(
        "CASE=$CaseName"
        "EXE=$exe"
        "ARGUMENTS=$($Arguments -join ' ')"
        "WORKING_DIRECTORY=$WorkingDirectory"
        "APP_HOME=$env:RECONSTRUCTION_ZONE_APP_HOME"
        "MODEL_DIR=$env:RECONSTRUCTION_ZONE_MODEL_DIR"
        "STRICT_MODEL_DIRS=$env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"
        "HF_HOME=$env:HF_HOME"
        "TRANSFORMERS_CACHE=$env:TRANSFORMERS_CACHE"
        "SETUP_TEST_MODE=$env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE"
        "WARMUP_SECONDS=$WarmupSeconds"
    ) | Set-Content -LiteralPath $envFile -Encoding utf8

    if ($WorkingDirectory) {
        $resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
    } else {
        $resolvedWorkingDirectory = Split-Path -Parent $exe
    }

    if ($Arguments.Count -gt 0) {
        $process = Start-Process -FilePath $exe -ArgumentList $Arguments -WorkingDirectory $resolvedWorkingDirectory -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    } else {
        $process = Start-Process -FilePath $exe -WorkingDirectory $resolvedWorkingDirectory -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    }

    Start-Sleep -Seconds $WarmupSeconds
    $process.Refresh()

    $lines = New-Object System.Collections.Generic.List[string]
    if ($process.HasExited) {
        $lines.Add("PROCESS_EXITED`t$($process.Id)`t$($process.ExitCode)")
    } else {
        $liveProcess = Get-Process -Id $process.Id -ErrorAction Stop
        $handle = $liveProcess.MainWindowHandle
        $lines.Add("PROCESS_STILL_RUNNING`t$($process.Id)")
        $lines.Add("MAIN_WINDOW_HANDLE`t$handle")

        if ($handle -ne [IntPtr]::Zero) {
            [RzVisualProbe.Win32]::SetForegroundWindow($handle) | Out-Null
            Start-Sleep -Milliseconds 800
        }

        try {
            Save-PrimaryScreen -Path $screenPng
            $lines.Add("SCREENSHOT_DESKTOP`t$screenPng")
        } catch {
            $lines.Add("SCREENSHOT_DESKTOP_FAILED`t$($_.Exception.Message)")
        }

        if ($handle -ne [IntPtr]::Zero) {
            try {
                Save-Window -Handle $handle -Path $windowPng
                $lines.Add("SCREENSHOT_MAIN_WINDOW`t$windowPng")
            } catch {
                $lines.Add("SCREENSHOT_MAIN_WINDOW_FAILED`t$($_.Exception.Message)")
            }
        } else {
            $lines.Add("SCREENSHOT_MAIN_WINDOW_SKIPPED`tNO_MAIN_WINDOW_HANDLE")
        }

        Stop-Process -Id $process.Id -Force
        $lines.Add("PROCESS_STOPPED`t$($process.Id)")
    }

    $lines | Set-Content -LiteralPath $summary -Encoding utf8

    Get-ChildItem -LiteralPath $modelDir -Force -ErrorAction SilentlyContinue |
        Select-Object Name,Length,LastWriteTime |
        Format-Table -AutoSize |
        Out-String |
        Set-Content -LiteralPath $modelListing -Encoding utf8

    $crashLog = Join-Path $appHome "logs\crash.log"
    if (Test-Path -LiteralPath $crashLog) {
        Get-Content -LiteralPath $crashLog -Tail 160 | Set-Content -LiteralPath $crashTail -Encoding utf8
    } else {
        "NO_CRASH_LOG" | Set-Content -LiteralPath $crashTail -Encoding utf8
    }

    Get-Content -LiteralPath $summary
    Write-Output "CASE_ROOT`t$caseRoot"
    Write-Output "STDOUT`t$out"
    Write-Output "STDERR`t$err"
    Write-Output "SUMMARY`t$summary"
    Write-Output "CRASH_TAIL`t$crashTail"
    Write-Output "MODELS`t$modelListing"
} finally {
    if ($null -eq $oldAppHome) { Remove-Item Env:\RECONSTRUCTION_ZONE_APP_HOME -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_APP_HOME = $oldAppHome }
    if ($null -eq $oldModelDir) { Remove-Item Env:\RECONSTRUCTION_ZONE_MODEL_DIR -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_MODEL_DIR = $oldModelDir }
    if ($null -eq $oldHfHome) { Remove-Item Env:\HF_HOME -ErrorAction SilentlyContinue } else { $env:HF_HOME = $oldHfHome }
    if ($null -eq $oldTransformersCache) { Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue } else { $env:TRANSFORMERS_CACHE = $oldTransformersCache }
    if ($null -eq $oldSetupTestMode) { Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE = $oldSetupTestMode }
    if ($null -eq $oldStrictModelDirs) { Remove-Item Env:\RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = $oldStrictModelDirs }
}

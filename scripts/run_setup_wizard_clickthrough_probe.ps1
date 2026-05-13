param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [Parameter(Mandatory = $true)]
    [string]$CaseName,

    [int]$WarmupSeconds = 14,

    [switch]$Reset,

    [switch]$SkipEmptyVerifyClick
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
$windowsFile = Join-Path $probeDir "windows.txt"
$crashTail = Join-Path $probeDir "crash-log-tail.txt"
$modelsFile = Join-Path $probeDir "models.txt"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

Add-Type @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace RzClickProbe
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

        public class WindowInfo
        {
            public IntPtr Handle;
            public string Title;
            public RECT Rect;
            public override string ToString()
            {
                return Handle + "\t" + Title + "\t" + Rect.Left + "," + Rect.Top + "," + Rect.Right + "," + Rect.Bottom;
            }
        }

        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);

        [DllImport("user32.dll")]
        public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);

        [DllImport("user32.dll")]
        public static extern int GetWindowTextLength(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);

        [DllImport("user32.dll")]
        public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        [DllImport("user32.dll")]
        public static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        public static extern bool SetCursorPos(int X, int Y);

        [DllImport("user32.dll")]
        public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

        public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        public const uint MOUSEEVENTF_LEFTUP = 0x0004;

        public static List<WindowInfo> GetWindowsForProcess(int pid)
        {
            var windows = new List<WindowInfo>();
            EnumWindows(delegate(IntPtr hWnd, IntPtr lParam)
            {
                if (!IsWindowVisible(hWnd))
                {
                    return true;
                }
                uint windowPid;
                GetWindowThreadProcessId(hWnd, out windowPid);
                if (windowPid != pid)
                {
                    return true;
                }
                int length = GetWindowTextLength(hWnd);
                var builder = new StringBuilder(length + 1);
                GetWindowText(hWnd, builder, builder.Capacity);
                RECT rect;
                GetWindowRect(hWnd, out rect);
                windows.Add(new WindowInfo { Handle = hWnd, Title = builder.ToString(), Rect = rect });
                return true;
            }, IntPtr.Zero);
            return windows;
        }

        public static void LeftClick(int x, int y)
        {
            SetCursorPos(x, y);
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
        }
    }
}
"@

function Save-PrimaryScreen {
    param([string]$Path)
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
        [RzClickProbe.Win32+WindowInfo]$Window,
        [string]$Path
    )
    $rect = $Window.Rect
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
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

function Get-WizardWindow {
    param([int]$Pid)
    $windows = [RzClickProbe.Win32]::GetWindowsForProcess($Pid)
    $windows | ForEach-Object { $_.ToString() } | Set-Content -LiteralPath $windowsFile -Encoding utf8
    $wizard = $windows | Where-Object { $_.Title -like "*First Launch Setup*" } | Select-Object -First 1
    if ($wizard) {
        return $wizard
    }
    return $windows | Where-Object { $_.Title -like "*Reconstruction Zone*" } | Select-Object -First 1
}

function Invoke-Click {
    param(
        [RzClickProbe.Win32+WindowInfo]$Window,
        [double]$RelativeX,
        [double]$RelativeY,
        [string]$Label
    )
    $rect = $Window.Rect
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    $x = [int]($rect.Left + ($width * $RelativeX))
    $y = [int]($rect.Top + ($height * $RelativeY))
    [RzClickProbe.Win32]::SetForegroundWindow($Window.Handle) | Out-Null
    Start-Sleep -Milliseconds 500
    [RzClickProbe.Win32]::LeftClick($x, $y)
    return "$Label`t$x`t$y"
}

$oldAppHome = $env:RECONSTRUCTION_ZONE_APP_HOME
$oldModelDir = $env:RECONSTRUCTION_ZONE_MODEL_DIR
$oldHfHome = $env:HF_HOME
$oldTransformersCache = $env:TRANSFORMERS_CACHE
$oldSetupTestMode = $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE
$oldStrictModelDirs = $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS

$process = $null
$lines = New-Object System.Collections.Generic.List[string]

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
        "APP_HOME=$env:RECONSTRUCTION_ZONE_APP_HOME"
        "MODEL_DIR=$env:RECONSTRUCTION_ZONE_MODEL_DIR"
        "STRICT_MODEL_DIRS=$env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"
        "HF_HOME=$env:HF_HOME"
        "TRANSFORMERS_CACHE=$env:TRANSFORMERS_CACHE"
        "WARMUP_SECONDS=$WarmupSeconds"
    ) | Set-Content -LiteralPath $envFile -Encoding utf8

    $workingDirectory = Split-Path -Parent $exe
    $process = Start-Process -FilePath $exe -WorkingDirectory $workingDirectory -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $lines.Add("PROCESS_STARTED`t$($process.Id)")

    Start-Sleep -Seconds $WarmupSeconds
    $process.Refresh()
    if ($process.HasExited) {
        $lines.Add("PROCESS_EXITED_BEFORE_CLICK`t$($process.Id)`t$($process.ExitCode)")
    } else {
        $wizard = Get-WizardWindow -Pid $process.Id
        if (-not $wizard) {
            $lines.Add("WIZARD_WINDOW_NOT_FOUND")
        } else {
            $lines.Add("WIZARD_WINDOW`t$($wizard.ToString())")
            [RzClickProbe.Win32]::SetForegroundWindow($wizard.Handle) | Out-Null
            Start-Sleep -Milliseconds 800
            Save-PrimaryScreen -Path (Join-Path $screenshotDir "01-welcome-desktop.png")
            Save-Window -Window $wizard -Path (Join-Path $screenshotDir "01-welcome-window.png")

            # Begin Setup button spans the lower width of the welcome page.
            $lines.Add((Invoke-Click -Window $wizard -RelativeX 0.50 -RelativeY 0.93 -Label "CLICK_BEGIN_SETUP"))
            Start-Sleep -Seconds 3

            $wizard = Get-WizardWindow -Pid $process.Id
            Save-PrimaryScreen -Path (Join-Path $screenshotDir "02-after-begin-desktop.png")
            Save-Window -Window $wizard -Path (Join-Path $screenshotDir "02-after-begin-window.png")

            if (-not $SkipEmptyVerifyClick) {
                # Verify button is at the right of the token row on the SAM3 Access page.
                $lines.Add((Invoke-Click -Window $wizard -RelativeX 0.90 -RelativeY 0.62 -Label "CLICK_VERIFY_EMPTY_TOKEN"))
                Start-Sleep -Seconds 2
                $wizard = Get-WizardWindow -Pid $process.Id
                Save-PrimaryScreen -Path (Join-Path $screenshotDir "03-after-empty-verify-desktop.png")
                Save-Window -Window $wizard -Path (Join-Path $screenshotDir "03-after-empty-verify-window.png")
            }

            $process.Refresh()
            if ($process.HasExited) {
                $lines.Add("PROCESS_EXITED_AFTER_CLICK`t$($process.Id)`t$($process.ExitCode)")
            } else {
                Stop-Process -Id $process.Id -Force
                $lines.Add("PROCESS_STOPPED`t$($process.Id)")
            }
        }
    }

    Get-ChildItem -LiteralPath $modelDir -Force -ErrorAction SilentlyContinue |
        Select-Object Name,Length,LastWriteTime |
        Format-Table -AutoSize |
        Out-String |
        Set-Content -LiteralPath $modelsFile -Encoding utf8

    $crashLog = Join-Path $appHome "logs\crash.log"
    if (Test-Path -LiteralPath $crashLog) {
        Get-Content -LiteralPath $crashLog -Tail 200 | Set-Content -LiteralPath $crashTail -Encoding utf8
    } else {
        "NO_CRASH_LOG" | Set-Content -LiteralPath $crashTail -Encoding utf8
    }

    $lines | Set-Content -LiteralPath $summary -Encoding utf8
    Get-Content -LiteralPath $summary
    Write-Output "CASE_ROOT`t$caseRoot"
    Write-Output "SCREENSHOTS`t$screenshotDir"
    Write-Output "STDOUT`t$out"
    Write-Output "STDERR`t$err"
    Write-Output "SUMMARY`t$summary"
    Write-Output "CRASH_TAIL`t$crashTail"
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -eq $oldAppHome) { Remove-Item Env:\RECONSTRUCTION_ZONE_APP_HOME -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_APP_HOME = $oldAppHome }
    if ($null -eq $oldModelDir) { Remove-Item Env:\RECONSTRUCTION_ZONE_MODEL_DIR -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_MODEL_DIR = $oldModelDir }
    if ($null -eq $oldHfHome) { Remove-Item Env:\HF_HOME -ErrorAction SilentlyContinue } else { $env:HF_HOME = $oldHfHome }
    if ($null -eq $oldTransformersCache) { Remove-Item Env:\TRANSFORMERS_CACHE -ErrorAction SilentlyContinue } else { $env:TRANSFORMERS_CACHE = $oldTransformersCache }
    if ($null -eq $oldSetupTestMode) { Remove-Item Env:\RECONSTRUCTION_ZONE_SETUP_TEST_MODE -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_SETUP_TEST_MODE = $oldSetupTestMode }
    if ($null -eq $oldStrictModelDirs) { Remove-Item Env:\RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS -ErrorAction SilentlyContinue } else { $env:RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS = $oldStrictModelDirs }
}

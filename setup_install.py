"""
Reconstruction Zone Lite — Installer

Runs inside the .venv created by install.bat. Handles:
  1. Python version check
  2. Disk space check
  3. GPU detection via nvidia-smi
  4. CUDA version → PyTorch index mapping
  5. PyTorch install with correct CUDA wheels
  6. CUDA verification
  7. pip install requirements-lite.txt
  8. ffmpeg check + download
  9. Desktop shortcut creation
  10. Summary
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR / "tools"
FFMPEG_DIR = TOOLS_DIR / "ffmpeg"
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements-lite.txt"
LAUNCH_BAT = SCRIPT_DIR / "launch.bat"
ICON_FILE = SCRIPT_DIR / "reconstruction-zone.ico"

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)
MIN_DISK_GB = 10

# CUDA version → PyTorch index URL mapping
# nvidia-smi reports the max CUDA version the driver supports.
# NOTE: Default PyPI torch is CPU-only. CUDA wheels always require --index-url.
CUDA_INDEX_MAP = [
    # (min_version, max_version, index_url)
    ((13, 0), (99, 99), "https://download.pytorch.org/whl/cu130"),
    ((12, 6), (12, 99), "https://download.pytorch.org/whl/cu128"),
    ((12, 1), (12, 5),  "https://download.pytorch.org/whl/cu124"),
    ((11, 8), (12, 0),  "https://download.pytorch.org/whl/cu118"),
]

FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def print_step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}")
    print("─" * 50)


def print_ok(msg: str):
    print(f"  OK: {msg}")


def print_warn(msg: str):
    print(f"  WARNING: {msg}")


def print_fail(msg: str):
    print(f"  FAILED: {msg}")


def abort(msg: str):
    print(f"\n  ABORT: {msg}")
    print("\nSetup cannot continue. Fix the issue above and re-run install.bat.")
    sys.exit(1)


def parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Parse '12.6' or '3.11.0' into (12, 6) or (3, 11, 0)."""
    parts = []
    for part in version_str.strip().split("."):
        digits = re.match(r"(\d+)", part)
        if digits:
            parts.append(int(digits.group(1)))
    return tuple(parts)


def run_pip(args: list, desc: str) -> bool:
    """Run pip with the given args. Returns True on success."""
    pip = str(Path(sys.executable).parent / "pip")
    cmd = [pip] + args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print_fail(f"{desc} failed (exit code {result.returncode})")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Step implementations
# ═══════════════════════════════════════════════════════════════════════════════

def step_check_python() -> None:
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    ver_tuple = (ver.major, ver.minor)

    if ver_tuple < MIN_PYTHON:
        abort(
            f"Python {ver_str} is too old. "
            f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer from python.org."
        )
    if ver_tuple > MAX_PYTHON:
        print_warn(
            f"Python {ver_str} is newer than tested ({MAX_PYTHON[0]}.{MAX_PYTHON[1]}). "
            f"PyTorch wheels may not be available. If install fails, try Python 3.12."
        )
    else:
        print_ok(f"Python {ver_str}")


def step_check_disk() -> None:
    try:
        usage = shutil.disk_usage(SCRIPT_DIR)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < MIN_DISK_GB:
            print_warn(
                f"Only {free_gb:.1f} GB free. Setup needs ~{MIN_DISK_GB} GB "
                f"(PyTorch ~2.5 GB, models ~3.5 GB, deps ~1 GB)."
            )
        else:
            print_ok(f"{free_gb:.0f} GB free")
    except Exception as e:
        print_warn(f"Could not check disk space: {e}")


def step_detect_gpu() -> Tuple[str, Tuple[int, ...]]:
    """Detect GPU and CUDA version. Returns (gpu_name, cuda_version_tuple)."""
    # Try nvidia-smi in common locations
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        alt_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ]
        for p in alt_paths:
            if Path(p).is_file():
                nvidia_smi = p
                break

    if not nvidia_smi:
        abort(
            "nvidia-smi not found. This means either:\n"
            "  - No NVIDIA GPU is installed\n"
            "  - NVIDIA drivers are not installed\n"
            "  - nvidia-smi is not on PATH\n\n"
            "Install the latest NVIDIA drivers from https://www.nvidia.com/drivers/"
        )

    try:
        result = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10
        )
    except Exception as e:
        abort(f"Failed to run nvidia-smi: {e}")

    if result.returncode != 0:
        abort(f"nvidia-smi returned an error:\n{result.stderr}")

    output = result.stdout
    print(f"  nvidia-smi output:\n")
    # Print just the header lines (first 10 lines)
    for i, line in enumerate(output.splitlines()):
        if i < 10:
            print(f"    {line}")
    print()

    # Parse GPU name
    gpu_name = "Unknown GPU"
    gpu_match = re.search(r"\|\s+\d+\s+(.+?)\s+\w+\s+\|", output)
    if gpu_match:
        gpu_name = gpu_match.group(1).strip()

    # Parse CUDA version
    cuda_match = re.search(r"CUDA Version:\s+([\d.]+)", output)
    if not cuda_match:
        abort(
            "Could not parse CUDA version from nvidia-smi output.\n"
            "This is unexpected. Please check your NVIDIA driver installation."
        )

    cuda_str = cuda_match.group(1)
    cuda_ver = parse_version_tuple(cuda_str)

    print_ok(f"GPU: {gpu_name}")
    print_ok(f"CUDA driver version: {cuda_str}")

    return gpu_name, cuda_ver


def step_select_pytorch_index(cuda_ver: Tuple[int, ...]) -> str:
    """Map CUDA version to PyTorch index URL."""
    cuda_major_minor = cuda_ver[:2]

    for min_ver, max_ver, index_url in CUDA_INDEX_MAP:
        if min_ver <= cuda_major_minor <= max_ver:
            print_ok(f"Using PyTorch index: {index_url}")
            return index_url

    abort(
        f"CUDA {'.'.join(str(v) for v in cuda_ver)} is too old for PyTorch.\n"
        f"Minimum supported: CUDA 11.8.\n"
        f"Update your NVIDIA drivers from https://www.nvidia.com/drivers/"
    )


def step_install_pytorch(index_url: str) -> None:
    packages = ["torch", "torchvision", "torchaudio"]
    args = ["install"] + packages
    args += ["--index-url", index_url]

    if not run_pip(args, "PyTorch install"):
        abort(
            "PyTorch installation failed. Common causes:\n"
            "  - Network issues (try again)\n"
            "  - Python version too new (try Python 3.12)\n"
            "  - Disk space full"
        )


def step_verify_cuda() -> None:
    """Verify PyTorch can see the GPU."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; "
             "print(f'PyTorch {torch.__version__}'); "
             "print(f'CUDA available: {torch.cuda.is_available()}'); "
             "print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None; "
             "print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; "
             "assert torch.cuda.is_available(), 'CUDA not available'"
             ],
            capture_output=True, text=True, timeout=30
        )
    except Exception as e:
        abort(f"Failed to verify CUDA: {e}")

    print(result.stdout.strip())

    if result.returncode != 0:
        # Try to diagnose
        diag_lines = []
        try:
            diag = subprocess.run(
                [sys.executable, "-c",
                 "import torch; "
                 "print(f'torch.version.cuda = {torch.version.cuda}'); "
                 "print(f'torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}'); "
                 "print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')"
                 ],
                capture_output=True, text=True, timeout=15
            )
            diag_lines = diag.stdout.strip().splitlines()
        except Exception:
            pass

        msg = "PyTorch installed but CUDA is not available.\n"
        if diag_lines:
            msg += "  Diagnostics:\n"
            for line in diag_lines:
                msg += f"    {line}\n"
        msg += (
            "\n  Common causes:\n"
            "  - NVIDIA driver too old for the installed CUDA toolkit\n"
            "  - GPU architecture not supported by this PyTorch build\n"
            "  - PyTorch installed CPU-only by mistake\n"
            "\n  Try updating your NVIDIA drivers and re-running install.bat."
        )
        abort(msg)

    print_ok("PyTorch CUDA verification passed")


def step_install_requirements() -> None:
    if not REQUIREMENTS_FILE.exists():
        abort(f"requirements-lite.txt not found at {REQUIREMENTS_FILE}")

    if not run_pip(
        ["install", "-r", str(REQUIREMENTS_FILE)],
        "dependency install"
    ):
        abort("Failed to install dependencies. Check the output above for errors.")

    print_ok("All dependencies installed")


def step_check_ffmpeg() -> None:
    """Check for ffmpeg on PATH or in tools/. Download if missing."""
    # Check local tools dir first
    local_ffmpeg = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    if local_ffmpeg.is_file():
        print_ok(f"ffmpeg found at {local_ffmpeg}")
        return

    # Check system PATH
    if shutil.which("ffmpeg"):
        print_ok(f"ffmpeg found on system PATH: {shutil.which('ffmpeg')}")
        return

    print("  ffmpeg not found. Downloading...")
    _download_ffmpeg()


def _download_ffmpeg() -> None:
    """Download ffmpeg essentials from gyan.dev and extract to tools/ffmpeg/."""
    import urllib.request

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLS_DIR / "ffmpeg-essentials.zip"

    try:
        print(f"  Downloading from {FFMPEG_DOWNLOAD_URL}...")
        print("  (This is ~100 MB, may take a minute)")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb:.0f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, str(zip_path), _progress)
        print()  # newline after progress
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        print_warn(
            f"Failed to download ffmpeg: {e}\n"
            "  You can install ffmpeg manually:\n"
            "  1. Download from https://www.gyan.dev/ffmpeg/builds/\n"
            "  2. Extract to tools/ffmpeg/ so that tools/ffmpeg/bin/ffmpeg.exe exists"
        )
        return

    # Extract
    try:
        print("  Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            # The zip contains a top-level folder like ffmpeg-7.1-essentials_build/
            # We need to find it and extract to tools/ffmpeg/
            top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            if len(top_dirs) == 1:
                top_dir = top_dirs.pop()
            else:
                top_dir = None

            FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
            for member in zf.namelist():
                if top_dir and member.startswith(top_dir + "/"):
                    # Strip the top-level directory
                    rel_path = member[len(top_dir) + 1:]
                    if not rel_path:
                        continue
                    target = FFMPEG_DIR / rel_path
                    if member.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)

        # Verify
        ffmpeg_exe = FFMPEG_DIR / "bin" / "ffmpeg.exe"
        if ffmpeg_exe.is_file():
            print_ok(f"ffmpeg installed to {ffmpeg_exe}")
        else:
            print_warn(
                "ffmpeg zip extracted but ffmpeg.exe not found at expected location.\n"
                f"  Check {FFMPEG_DIR} and ensure bin/ffmpeg.exe exists."
            )
    except Exception as e:
        print_warn(f"Failed to extract ffmpeg: {e}")
    finally:
        # Clean up zip
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass


def step_create_shortcut() -> None:
    """Create a desktop shortcut to launch.bat."""
    if not LAUNCH_BAT.exists():
        print_warn("launch.bat not found — skipping shortcut creation")
        return

    desktop = Path.home() / "Desktop"
    if not desktop.is_dir():
        print_warn("Desktop folder not found — skipping shortcut creation")
        return

    shortcut_path = desktop / "Reconstruction Zone.lnk"
    icon_arg = f", '{ICON_FILE}', 0" if ICON_FILE.is_file() else ""

    ps_script = (
        f"$ws = New-Object -ComObject WScript.Shell; "
        f"$sc = $ws.CreateShortcut('{shortcut_path}'); "
        f"$sc.TargetPath = '{LAUNCH_BAT}'; "
        f"$sc.WorkingDirectory = '{SCRIPT_DIR}'; "
        f"$sc.Description = 'Reconstruction Zone Lite'; "
    )
    if ICON_FILE.is_file():
        ps_script += f"$sc.IconLocation = '{ICON_FILE}'; "
    ps_script += "$sc.Save()"

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print_ok(f"Desktop shortcut created: {shortcut_path}")
        else:
            print_warn(f"Could not create shortcut: {result.stderr.strip()}")
    except Exception as e:
        print_warn(f"Could not create shortcut: {e}")


def step_summary(gpu_name: str, cuda_ver: Tuple[int, ...]) -> None:
    cuda_str = ".".join(str(v) for v in cuda_ver)
    ffmpeg_local = (FFMPEG_DIR / "bin" / "ffmpeg.exe").is_file()
    ffmpeg_system = shutil.which("ffmpeg") is not None

    print("\n" + "=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  GPU:     {gpu_name}")
    print(f"  CUDA:    {cuda_str}")
    print(f"  ffmpeg:  {'local (tools/)' if ffmpeg_local else 'system PATH' if ffmpeg_system else 'NOT FOUND'}")
    print(f"  venv:    {Path(sys.executable).parent.parent}")
    print()
    print("  To launch the app:")
    print(f"    Double-click launch.bat")
    print(f"    Or use the desktop shortcut")
    print()
    print("  On first launch, the setup wizard will download")
    print("  AI model weights (~140 MB for YOLO + RF-DETR,")
    print("  ~3.3 GB additional if you opt into SAM3).")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    total = 10
    print()

    print_step(1, total, "Checking Python version")
    step_check_python()

    print_step(2, total, "Checking disk space")
    step_check_disk()

    print_step(3, total, "Detecting GPU")
    gpu_name, cuda_ver = step_detect_gpu()

    print_step(4, total, "Selecting PyTorch CUDA index")
    index_url = step_select_pytorch_index(cuda_ver)

    print_step(5, total, "Installing PyTorch")
    step_install_pytorch(index_url)

    print_step(6, total, "Verifying CUDA")
    step_verify_cuda()

    print_step(7, total, "Installing dependencies")
    step_install_requirements()

    print_step(8, total, "Checking ffmpeg")
    step_check_ffmpeg()

    print_step(9, total, "Creating desktop shortcut")
    step_create_shortcut()

    print_step(10, total, "Summary")
    step_summary(gpu_name, cuda_ver)


if __name__ == "__main__":
    main()

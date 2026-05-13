"""
Setup Wizard (Lite) — Environment health check + model onboarding.

Extended version of setup_wizard.py for the Lite GitHub distribution.
Checks environment health (CUDA, ffmpeg, disk, network) in addition
to model readiness. Shows a confirmation screen on first launch even
if everything is healthy.

Flow:
  - First launch: always show wizard with health check results + model status.
    User clicks through to enter the app. Sets 'setup_wizard_lite_completed' pref.
  - Subsequent launches: if all checks pass and pref is set, wizard skips.
    If any check fails, wizard reappears regardless of pref.

Queue-based threading: background threads put status messages on a
queue, and the main thread polls the queue to update the UI. No
tkinter calls are ever made from background threads.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk

logger = logging.getLogger(__name__)

# Import model infrastructure from setup_wizard.py
try:
    from setup_wizard import (
        ModelInfo, ALL_MODELS, check_all_models, all_models_ready,
        run_diagnostics, _CHECK_FNS, _DOWNLOAD_FNS,
    )
    from sam3_setup import check_sam3_setup, verify_hf_token_detailed
except ImportError:
    from reconstruction_gui.setup_wizard import (
        ModelInfo, ALL_MODELS, check_all_models, all_models_ready,
        run_diagnostics, _CHECK_FNS, _DOWNLOAD_FNS,
    )
    from reconstruction_gui.sam3_setup import check_sam3_setup, verify_hf_token_detailed


# ═══════════════════════════════════════════════════════════════════════════════
# Design tokens (matching setup_wizard.py)
# ═══════════════════════════════════════════════════════════════════════════════

_BG = "#1a1a2e"
_SURFACE = "#1e293b"
_ACCENT = "#2563eb"
_TEXT_PRIMARY = "#ffffff"
_TEXT_SECONDARY = "#e2e8f0"
_TEXT_MUTED = "#94a3b8"
_TEXT_DIM = "#475569"
_SUCCESS = "#22c55e"
_ERROR = "#EF4444"
_WARNING = "#eab308"

_CARD_RADIUS = 4
_CARD_PAD = 14

_PREFS_KEY = "setup_wizard_lite_completed"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_TOOLS_FFMPEG = _PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
_FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


# ═══════════════════════════════════════════════════════════════════════════════
# Environment health checks
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthCheck:
    """Result of a single environment check."""
    name: str
    description: str
    status: str  # "pass", "fail", "warn"
    detail: str = ""
    fixable: bool = False  # True if the wizard can fix it (e.g. download ffmpeg)
    fix_key: str = ""  # key for fix action


def check_cuda() -> HealthCheck:
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            return HealthCheck(
                "CUDA", "GPU acceleration for AI models",
                "pass", f"{gpu} — CUDA {torch.version.cuda}"
            )
        else:
            return HealthCheck(
                "CUDA", "GPU acceleration for AI models",
                "warn",
                "PyTorch installed but CUDA not available. "
                "Processing will be very slow on CPU. "
                "Re-run install.bat to fix."
            )
    except ImportError:
        return HealthCheck(
            "CUDA", "GPU acceleration for AI models",
            "fail", "PyTorch not installed. Run install.bat first."
        )


def check_ffmpeg() -> HealthCheck:
    """Check if ffmpeg is available (system PATH or tools/)."""
    if _TOOLS_FFMPEG.is_file():
        return HealthCheck(
            "ffmpeg", "Video frame extraction",
            "pass", f"Found at {_TOOLS_FFMPEG}"
        )
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return HealthCheck(
            "ffmpeg", "Video frame extraction",
            "pass", f"Found on PATH: {system_ffmpeg}"
        )
    return HealthCheck(
        "ffmpeg", "Video frame extraction",
        "fail",
        "Not found. Video extraction will not work.",
        fixable=True, fix_key="ffmpeg"
    )


def check_ffprobe() -> HealthCheck:
    """Check if ffprobe is available."""
    local = _PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
    if local.is_file() or shutil.which("ffprobe"):
        return HealthCheck("ffprobe", "Video analysis", "pass")
    return HealthCheck(
        "ffprobe", "Video analysis",
        "fail", "Not found. Video analysis will not work.",
        fixable=True, fix_key="ffmpeg"  # same fix as ffmpeg
    )


def check_disk_space() -> HealthCheck:
    """Check free disk space."""
    try:
        free_gb = shutil.disk_usage(str(_PROJECT_ROOT)).free / (1024 ** 3)
        if free_gb > 5:
            return HealthCheck(
                "Disk Space", "Storage for model weights",
                "pass", f"{free_gb:.0f} GB free"
            )
        else:
            return HealthCheck(
                "Disk Space", "Storage for model weights",
                "warn", f"Only {free_gb:.1f} GB free. Models need ~3.5 GB."
            )
    except Exception:
        return HealthCheck("Disk Space", "Storage for model weights", "warn", "Could not check")


def check_network() -> HealthCheck:
    """Check basic internet connectivity."""
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return HealthCheck(
            "Network", "Internet access for model downloads",
            "pass", "Connected"
        )
    except Exception:
        return HealthCheck(
            "Network", "Internet access for model downloads",
            "warn", "Cannot reach huggingface.co. Model downloads may fail."
        )


def run_all_health_checks() -> List[HealthCheck]:
    """Run all environment health checks."""
    return [
        check_cuda(),
        check_ffmpeg(),
        check_ffprobe(),
        check_disk_space(),
        check_network(),
    ]


def all_checks_pass(checks: List[HealthCheck]) -> bool:
    """True if no checks failed (warnings are OK)."""
    return all(c.status != "fail" for c in checks)


# ═══════════════════════════════════════════════════════════════════════════════
# ffmpeg download (reused from setup_install.py pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def _download_ffmpeg_worker(q: queue.Queue) -> None:
    """Download ffmpeg essentials to tools/ffmpeg/. Posts progress to queue."""
    import zipfile

    tools_dir = _PROJECT_ROOT / "tools"
    ffmpeg_dir = tools_dir / "ffmpeg"
    zip_path = tools_dir / "ffmpeg-essentials.zip"

    tools_dir.mkdir(parents=True, exist_ok=True)

    try:
        q.put(("ffmpeg_status", "Downloading ffmpeg (~100 MB)..."))
        urllib.request.urlretrieve(_FFMPEG_DOWNLOAD_URL, str(zip_path))

        q.put(("ffmpeg_status", "Extracting..."))
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
            top_dir = top_dirs.pop() if len(top_dirs) == 1 else None

            ffmpeg_dir.mkdir(parents=True, exist_ok=True)
            for member in zf.namelist():
                if top_dir and member.startswith(top_dir + "/"):
                    rel_path = member[len(top_dir) + 1:]
                    if not rel_path:
                        continue
                    target = ffmpeg_dir / rel_path
                    if member.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)

        q.put(("ffmpeg_done", True))
    except Exception as e:
        q.put(("ffmpeg_done", False, str(e)))
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Prefs helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_prefs() -> dict:
    try:
        from app_paths import prefs_file
    except ImportError:
        from reconstruction_gui.app_paths import prefs_file
    path = prefs_file(create=False)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_pref(key: str, value) -> None:
    try:
        from app_paths import prefs_file
    except ImportError:
        from reconstruction_gui.app_paths import prefs_file
    path = prefs_file(create=True)
    prefs = {}
    if path.exists():
        try:
            prefs = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    prefs[key] = value
    path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Setup Wizard Lite — CTkToplevel
# ═══════════════════════════════════════════════════════════════════════════════

class SetupWizardLite(ctk.CTkToplevel):
    """First-launch setup wizard with environment health checks + model downloads."""

    def __init__(self, parent, health_checks: List[HealthCheck], models: List[ModelInfo]):
        super().__init__(parent)
        self._parent = parent
        self.title("Reconstruction Zone — Setup")
        self.geometry("640x720")
        self.resizable(False, False)
        self.configure(fg_color=_BG)

        self.health_checks = health_checks
        self.models = models
        self.missing_models = [m for m in models if not m.ready]
        self.success = False
        self._queue: queue.Queue = queue.Queue()
        self._current_step = "overview"
        self._sam3_gate_resolved = False
        self._failed_keys: List[str] = []
        self._failed_errors: Dict[str, str] = {}

        self.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        self._container = ctk.CTkFrame(self, fg_color=_BG)
        self._container.pack(fill="both", expand=True, padx=24, pady=24)

        self._show_overview()

    # ── Helpers ──

    def _clear(self):
        for w in self._container.winfo_children():
            w.destroy()

    def _header(self, subtitle: str, color: str = "#60a5fa"):
        hdr = ctk.CTkFrame(self._container, fg_color="transparent")
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="Reconstruction Zone", font=("", 18, "bold"),
                     text_color=_TEXT_PRIMARY).pack()
        ctk.CTkLabel(hdr, text=subtitle, font=("", 13),
                     text_color=color).pack()

    def _status_icon(self, status: str) -> Tuple[str, str]:
        if status == "pass":
            return "\u2713", _SUCCESS
        elif status == "warn":
            return "!", _WARNING
        else:
            return "\u2717", _ERROR

    # ── Step 1: Overview ──

    def _show_overview(self):
        self._clear()
        self._current_step = "overview"
        self._header("Environment Check")

        # Health checks
        env_frame = ctk.CTkFrame(self._container, fg_color="transparent")
        env_frame.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(env_frame, text="Environment", font=("", 13, "bold"),
                     text_color=_TEXT_SECONDARY).pack(anchor="w")

        for check in self.health_checks:
            row = ctk.CTkFrame(env_frame, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
            row.pack(fill="x", pady=(4, 0))
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)

            icon, color = self._status_icon(check.status)
            ctk.CTkLabel(inner, text=icon, font=("", 13), text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(inner, text=check.name, font=("", 12, "bold"),
                         text_color=_TEXT_SECONDARY, width=90, anchor="w").pack(side="left")

            detail_text = check.detail if check.detail else check.description
            ctk.CTkLabel(inner, text=detail_text, font=("", 11),
                         text_color=_TEXT_MUTED if check.status == "pass" else color,
                         anchor="w", wraplength=380).pack(side="left", fill="x", expand=True)

            if check.fixable and check.status == "fail":
                if check.fix_key == "ffmpeg":
                    ctk.CTkButton(inner, text="Download", width=76, height=24,
                                  font=("", 10, "bold"), corner_radius=3,
                                  fg_color=_ACCENT, hover_color="#1d4ed8",
                                  command=self._download_ffmpeg).pack(side="right")

        # Models
        model_frame = ctk.CTkFrame(self._container, fg_color="transparent")
        model_frame.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(model_frame, text="AI Models", font=("", 13, "bold"),
                     text_color=_TEXT_SECONDARY).pack(anchor="w")

        for m in self.models:
            row = ctk.CTkFrame(model_frame, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
            row.pack(fill="x", pady=(4, 0))
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)

            if m.ready:
                icon, color = "\u2713", _SUCCESS
                status_text = f"{m.size_label} — ready"
            else:
                icon, color = "\u2022", _TEXT_DIM
                status_text = f"{m.size_label} — not downloaded"
                if m.gated:
                    status_text += " (requires HuggingFace token)"

            ctk.CTkLabel(inner, text=icon, font=("", 13), text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(inner, text=m.name, font=("", 12, "bold"),
                         text_color=_TEXT_SECONDARY, width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(inner, text=status_text, font=("", 11),
                         text_color=_TEXT_MUTED if m.ready else _TEXT_DIM,
                         anchor="w").pack(side="left", fill="x", expand=True)

        # Summary + action
        footer = ctk.CTkFrame(self._container, fg_color="transparent")
        footer.pack(fill="x", pady=(16, 0))

        env_ok = all_checks_pass(self.health_checks)
        models_ok = all(m.ready for m in self.models)

        if env_ok and models_ok:
            ctk.CTkLabel(footer, text="Everything looks good!",
                         font=("", 12), text_color=_SUCCESS).pack()
            ctk.CTkButton(footer, text="Enter Reconstruction Zone", height=40,
                          font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                          fg_color=_ACCENT, hover_color="#1d4ed8",
                          command=self._finish).pack(fill="x", pady=(10, 0))
        elif env_ok and not models_ok:
            missing_names = [m.name for m in self.models if not m.ready]
            ctk.CTkLabel(footer,
                         text=f"Models need downloading: {', '.join(missing_names)}",
                         font=("", 11), text_color=_TEXT_MUTED).pack()
            ctk.CTkButton(footer, text="Download Models", height=40,
                          font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                          fg_color=_ACCENT, hover_color="#1d4ed8",
                          command=self._begin_model_setup).pack(fill="x", pady=(10, 0))
            ctk.CTkButton(footer, text="Skip — launch without models", height=30,
                          font=("", 11), corner_radius=_CARD_RADIUS,
                          fg_color="transparent", hover_color=_SURFACE,
                          text_color=_TEXT_DIM,
                          command=self._finish).pack(fill="x", pady=(6, 0))
        else:
            ctk.CTkLabel(footer,
                         text="Some environment checks failed. Fix the issues above, then continue.",
                         font=("", 11), text_color=_WARNING, wraplength=540).pack()
            if not models_ok:
                ctk.CTkButton(footer, text="Download Models Anyway", height=40,
                              font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                              fg_color=_ACCENT, hover_color="#1d4ed8",
                              command=self._begin_model_setup).pack(fill="x", pady=(10, 0))
            ctk.CTkButton(footer, text="Skip — launch anyway", height=30,
                          font=("", 11), corner_radius=_CARD_RADIUS,
                          fg_color="transparent", hover_color=_SURFACE,
                          text_color=_TEXT_DIM,
                          command=self._finish).pack(fill="x", pady=(6, 0))

    # ── ffmpeg download ──

    def _download_ffmpeg(self):
        threading.Thread(target=_download_ffmpeg_worker, args=(self._queue,), daemon=True).start()
        self._poll()

    # ── Model download flow ──

    def _begin_model_setup(self):
        gated_missing = [m for m in self.missing_models if m.gated]
        if gated_missing and not self._sam3_gate_resolved:
            self._show_sam3_gate()
        else:
            self._sam3_gate_resolved = True
            self._show_downloading()

    def _show_sam3_gate(self):
        """SAM3 HuggingFace token gate — same flow as setup_wizard.py."""
        self._clear()
        self._current_step = "sam3_gate"
        self._header("SAM3 Access")

        ctx = ctk.CTkFrame(self._container, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
        ctx.pack(fill="x", pady=(12, 0))
        ci = ctk.CTkFrame(ctx, fg_color="transparent")
        ci.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(ci,
                     text="SAM3 is a gated model on HuggingFace. You need a "
                          "free account, model access approval, and a token.",
                     font=("", 11), text_color=_TEXT_SECONDARY,
                     wraplength=540, justify="left", anchor="w").pack(fill="x")
        ctk.CTkLabel(ci,
                     text="If you don't have a HuggingFace account, you can skip SAM3. "
                          "YOLO26 and RF-DETR will still work for masking.",
                     font=("", 11), text_color=_TEXT_MUTED,
                     wraplength=540, justify="left", anchor="w").pack(fill="x", pady=(6, 0))

        # Step-by-step instructions
        steps_frame = ctk.CTkFrame(self._container, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
        steps_frame.pack(fill="x", pady=(8, 0))
        si = ctk.CTkFrame(steps_frame, fg_color="transparent")
        si.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(si, text="Steps:", font=("", 12, "bold"),
                     text_color=_TEXT_SECONDARY).pack(anchor="w")
        steps = [
            "1.  Go to huggingface.co and create a free account (or log in)",
            "2.  Go to huggingface.co/facebook/sam3",
            "3.  Click \"Expand to review and access\" then \"Agree and access\"",
            "4.  Wait for the approval email (usually instant, can take hours)",
            "5.  Go to huggingface.co/settings/tokens",
            "6.  Click \"Create new token\", give it any name, role = Read",
            "7.  Copy the token (starts with hf_...) and paste it below",
        ]
        for step in steps:
            ctk.CTkLabel(si, text=step, font=("Consolas", 10),
                         text_color=_TEXT_MUTED, anchor="w",
                         wraplength=520, justify="left").pack(fill="x", pady=1)

        # Token input
        tok_box = ctk.CTkFrame(self._container, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
        tok_box.pack(fill="x", pady=(12, 0))
        ti = ctk.CTkFrame(tok_box, fg_color="transparent")
        ti.pack(fill="x", padx=16, pady=12)
        tok_row = ctk.CTkFrame(ti, fg_color="transparent")
        tok_row.pack(fill="x")
        ctk.CTkLabel(tok_row, text="Token:", font=("", 12),
                     text_color="#cbd5e1").pack(side="left")
        self._token_entry = ctk.CTkEntry(tok_row, placeholder_text="hf_...",
                                         show="*", fg_color="#0f172a",
                                         corner_radius=3, height=30)
        self._token_entry.pack(side="left", fill="x", expand=True, padx=8)
        self._verify_btn = ctk.CTkButton(tok_row, text="Verify", width=76, height=30,
                                         font=("", 12, "bold"), corner_radius=3,
                                         fg_color=_ACCENT, hover_color="#1d4ed8",
                                         command=self._on_verify_token)
        self._verify_btn.pack(side="left")

        self._gate_status = ctk.CTkLabel(self._container, text="",
                                         font=("", 11), wraplength=540)
        self._gate_status.pack(pady=(8, 0))

        # Check if token already exists
        report = check_sam3_setup()
        if report.overall_stage in ("ready", "needs_weights"):
            self._gate_status.configure(text="SAM3 access verified!",
                                        text_color=_SUCCESS)
            self._sam3_gate_resolved = True
            self._parent.after(500, self._show_downloading)
        elif report.overall_stage == "needs_access":
            self._gate_status.configure(text=report.message, text_color=_WARNING)

        # Skip button
        ctk.CTkButton(self._container, text="Skip SAM3 — download other models only",
                      height=30, font=("", 11), corner_radius=_CARD_RADIUS,
                      fg_color="transparent", hover_color=_SURFACE,
                      text_color=_TEXT_DIM,
                      command=self._skip_sam3).pack(fill="x", pady=(12, 0))

    def _skip_sam3(self):
        self._sam3_gate_resolved = True
        self.missing_models = [m for m in self.missing_models if not m.gated]
        self._show_downloading()

    def _on_verify_token(self):
        token = self._token_entry.get().strip()
        if not token:
            self._gate_status.configure(text="Please paste a HuggingFace token.",
                                        text_color=_ERROR)
            return
        self._verify_btn.configure(state="disabled", text="Verifying...")
        self._gate_status.configure(text="Checking token...", text_color=_TEXT_MUTED)
        threading.Thread(target=self._verify_worker, args=(token,), daemon=True).start()
        self._poll()

    def _verify_worker(self, token: str):
        try:
            report = verify_hf_token_detailed(token)
            self._queue.put(("sam3_verify", report))
        except Exception as e:
            self._queue.put(("error", f"Verification failed: {e}"))

    # ── Downloading ──

    def _show_downloading(self):
        self._clear()
        self._current_step = "downloading"
        self._header("Downloading Models")

        self._dl_frame = ctk.CTkFrame(self._container, fg_color="transparent")
        self._dl_frame.pack(fill="x", pady=(10, 0))

        self._dl_status = ctk.CTkLabel(self._container, text="Starting downloads...",
                                       font=("", 11), text_color=_TEXT_MUTED)
        self._dl_status.pack(pady=(10, 0))

        # Show model list
        for m in self.models:
            row = ctk.CTkFrame(self._dl_frame, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
            row.pack(fill="x", pady=(4, 0))
            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=12, pady=8)

            if m.ready:
                icon, color = "\u2713", _SUCCESS
                text = "Ready"
            elif m in self.missing_models:
                icon, color = "\u25B6", "#60a5fa"
                text = "Queued"
            else:
                icon, color = "\u2014", _TEXT_DIM
                text = "Skipped"

            ctk.CTkLabel(inner, text=icon, font=("", 13), text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(inner, text=m.name, font=("", 12, "bold"),
                         text_color=_TEXT_SECONDARY, width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(inner, text=text, font=("", 11),
                         text_color=_TEXT_MUTED, anchor="w").pack(side="left")

        threading.Thread(target=self._download_worker, daemon=True).start()
        self._poll()

    def _download_worker(self):
        failures = []
        for m in self.missing_models:
            self._queue.put(("dl_progress", m.key))
            fn = _DOWNLOAD_FNS.get(m.key)
            if fn:
                try:
                    fn(self._queue)
                    self._queue.put(("model_done", m.key))
                except Exception as e:
                    logger.error("Download failed for %s: %s", m.name, e)
                    self._queue.put(("model_error", m.key, str(e)))
                    failures.append(m.key)

        if failures:
            diag = run_diagnostics(failures)
            self._queue.put(("diagnostics", diag))
            self._queue.put(("all_done_with_errors", failures))
        else:
            self._queue.put(("all_done",))

    # ── Done ──

    def _show_done(self):
        self._clear()
        self._current_step = "done"

        # Re-check everything for final display
        self.health_checks = run_all_health_checks()
        self.models = check_all_models()
        env_ok = all_checks_pass(self.health_checks)
        models_ok = all(m.ready for m in self.models)

        if env_ok and models_ok:
            self._header("Setup Complete", color=_SUCCESS)
        else:
            self._header("Setup Complete", color=_WARNING)

        # Final status
        items = ctk.CTkFrame(self._container, fg_color="transparent")
        items.pack(fill="x", pady=(12, 0))

        for check in self.health_checks:
            icon, color = self._status_icon(check.status)
            row = ctk.CTkFrame(items, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=icon, font=("", 12), text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(row, text=check.name, font=("", 11),
                         text_color=_TEXT_SECONDARY, width=90, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=check.detail or "OK", font=("", 11),
                         text_color=_TEXT_MUTED).pack(side="left")

        for m in self.models:
            icon = "\u2713" if m.ready else "\u2717"
            color = _SUCCESS if m.ready else _TEXT_DIM
            row = ctk.CTkFrame(items, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=icon, font=("", 12), text_color=color,
                         width=20).pack(side="left")
            ctk.CTkLabel(row, text=m.name, font=("", 11),
                         text_color=_TEXT_SECONDARY, width=90, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text="Ready" if m.ready else "Not installed", font=("", 11),
                         text_color=_TEXT_MUTED).pack(side="left")

        ctk.CTkButton(self._container, text="Enter Reconstruction Zone", height=40,
                      font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                      fg_color=_ACCENT, hover_color="#1d4ed8",
                      command=self._finish).pack(fill="x", pady=(20, 0))

        self.protocol("WM_DELETE_WINDOW", self._finish)

    # ── Queue polling ──

    def _poll(self):
        try:
            while not self._queue.empty():
                msg = self._queue.get_nowait()
                self._handle_message(msg)
        except Exception:
            logger.exception("Wizard queue handler failed")
        if self._current_step not in ("done", "overview"):
            self._parent.after(100, self._poll)

    def _handle_message(self, msg):
        msg_type = msg[0]

        if msg_type == "status":
            if hasattr(self, "_dl_status"):
                self._dl_status.configure(text=msg[1])

        elif msg_type == "dl_progress":
            if hasattr(self, "_dl_status"):
                key = msg[1]
                name = next((m.name for m in self.models if m.key == key), key)
                self._dl_status.configure(text=f"Downloading {name}...")

        elif msg_type == "model_done":
            key = msg[1]
            for m in self.models:
                if m.key == key:
                    m.ready = True

        elif msg_type == "model_error":
            key, error = msg[1], msg[2]
            if key not in self._failed_keys:
                self._failed_keys.append(key)
            self._failed_errors[key] = error

        elif msg_type == "all_done":
            self._show_done()

        elif msg_type == "diagnostics":
            pass  # Stored for error display if needed

        elif msg_type == "all_done_with_errors":
            self._show_done()

        elif msg_type == "sam3_verify":
            report = msg[1]
            self._verify_btn.configure(state="normal", text="Verify")
            if report.overall_stage in ("ready", "ready_to_install", "needs_weights"):
                self._gate_status.configure(text="Access verified! Proceeding...",
                                            text_color=_SUCCESS)
                self._sam3_gate_resolved = True
                self._parent.after(1000, self._show_downloading)
            elif report.overall_stage == "needs_access":
                self._gate_status.configure(
                    text=report.message + "\n" + report.next_action,
                    text_color=_WARNING)
            elif report.overall_stage == "needs_token":
                self._gate_status.configure(text=report.message, text_color=_ERROR)
            else:
                self._gate_status.configure(text=report.message, text_color=_ERROR)

        elif msg_type == "ffmpeg_status":
            pass  # Could update a label if we had one

        elif msg_type == "ffmpeg_done":
            success = msg[1]
            if success:
                # Refresh health checks and redraw overview
                self.health_checks = run_all_health_checks()
                self._show_overview()

        elif msg_type == "error":
            logger.error("Wizard error: %s", msg[1])

    # ── Lifecycle ──

    def _finish(self):
        self.success = True
        _save_pref(_PREFS_KEY, True)
        self._close()

    def _close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_close_attempt(self):
        if self._current_step in ("overview", "done"):
            self._finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_setup_wizard_if_needed(parent) -> bool:
    """Check environment and models. Show wizard if needed or first launch."""
    health_checks = run_all_health_checks()
    models = check_all_models()
    prefs = _load_prefs()

    env_ok = all_checks_pass(health_checks)
    models_ok = all(m.ready for m in models)
    first_launch = not prefs.get(_PREFS_KEY, False)

    if env_ok and models_ok and not first_launch:
        logger.info("All checks pass, wizard not needed")
        return True

    if first_launch:
        logger.info("First launch — showing setup wizard")
    else:
        failed = [c.name for c in health_checks if c.status == "fail"]
        missing = [m.name for m in models if not m.ready]
        logger.info("Showing wizard — failed: %s, missing models: %s", failed, missing)

    wizard = SetupWizardLite(parent, health_checks, models)
    try:
        wizard.transient(parent)
        wizard.grab_set()
        wizard.focus()
    except Exception:
        logger.exception("Could not make wizard modal")
    return False

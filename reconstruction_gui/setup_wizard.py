"""
First-Launch Setup Wizard — Model onboarding for new users.

Handles the complete model setup flow:
  1. Welcome — explain models, show readiness
  2. SAM3 gate — HuggingFace token + access approval
  3. Download — progress for each model
  4. Done — all models ready
  5. Error — diagnostics + per-model retry

If all models are cached, the wizard never appears.

Queue-based threading: background threads put status messages on a
queue, and the main thread polls the queue to update the UI. No
tkinter calls are ever made from background threads.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk

logger = logging.getLogger(__name__)

try:
    from model_paths import (
        resolve_sam3_weights, resolve_rfdetr_seg_weights, resolve_yolo26_weights,
        app_model_dir, RFDETR_SEG_URLS, SAM3_MODEL_ID,
    )
except ImportError:
    from reconstruction_gui.model_paths import (
        resolve_sam3_weights, resolve_rfdetr_seg_weights, resolve_yolo26_weights,
        app_model_dir, RFDETR_SEG_URLS, SAM3_MODEL_ID,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Design tokens (from planning.pen)
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

_CARD_HEIGHT = 101
_CARD_RADIUS = 4
_CARD_PAD = 14
_CARD_GAP = 6

_MODEL_DESCRIPTIONS = {
    "sam3": "Describe what to remove in plain language. The primary masking engine.",
    "rfdetr": "Transformer-based detector with DINOv2 backbone.",
    "yolo26": "Lightweight CNN detector. Fastest option for known object types.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """Describes a model that the app needs."""
    name: str
    key: str
    size_label: str
    gated: bool = False
    ready: bool = False


def _sam3_ready() -> bool:
    return resolve_sam3_weights() is not None

def _rfdetr_ready() -> bool:
    return resolve_rfdetr_seg_weights("small") is not None

def _yolo26_ready() -> bool:
    return resolve_yolo26_weights("n") is not None


def _download_sam3(q: queue.Queue) -> None:
    q.put(("status", "Downloading SAM3 weights (~3.3 GB)..."))
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(SAM3_MODEL_ID)
    except Exception as e:
        raise RuntimeError(f"SAM3 download failed: {e}") from e


def _download_rfdetr(q: queue.Queue) -> None:
    dest_dir = app_model_dir(create=True)
    seg_dest = dest_dir / "rf-detr-seg-small.pt"
    if seg_dest.exists():
        return
    seg_url = RFDETR_SEG_URLS["small"]
    q.put(("status", "Downloading RF-DETR-Seg-Small (129 MB)..."))
    tmp_dest = seg_dest.with_suffix(".pt.tmp")
    try:
        urllib.request.urlretrieve(seg_url, str(tmp_dest))
        tmp_dest.replace(seg_dest)
    except Exception as e:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise RuntimeError(f"RF-DETR download failed: {e}") from e


def _download_yolo26(q: queue.Queue) -> None:
    dest_dir = app_model_dir(create=True)
    yolo_dest = dest_dir / "yolo26n-seg.pt"
    if yolo_dest.exists():
        return
    q.put(("status", "Downloading YOLO26-n-seg (6.5 MB)..."))
    try:
        from huggingface_hub import hf_hub_download
        hf_path = hf_hub_download("openvision/yolo26-n-seg", "model.pt")
        shutil.copy2(hf_path, str(yolo_dest))
    except Exception as e:
        if yolo_dest.exists():
            yolo_dest.unlink()
        raise RuntimeError(f"YOLO26 download failed: {e}") from e


_CHECK_FNS: Dict[str, callable] = {"sam3": _sam3_ready, "rfdetr": _rfdetr_ready, "yolo26": _yolo26_ready}
_DOWNLOAD_FNS: Dict[str, callable] = {"sam3": _download_sam3, "rfdetr": _download_rfdetr, "yolo26": _download_yolo26}

ALL_MODELS = [
    ModelInfo("SAM3", "sam3", "3.3 GB", gated=True),
    ModelInfo("RF-DETR-Seg", "rfdetr", "129 MB"),
    ModelInfo("YOLO26-n-seg", "yolo26", "6.5 MB"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_ENDPOINTS: Dict[str, List[Tuple[str, str, str]]] = {
    "sam3": [("https://huggingface.co", "huggingface.co", "SAM3 weights")],
    "rfdetr": [("https://storage.googleapis.com", "storage.googleapis.com", "RF-DETR seg weights")],
    "yolo26": [("https://huggingface.co", "huggingface.co", "YOLO26 weights")],
}


def run_diagnostics(failed_model_keys: List[str]) -> List[Tuple[str, str]]:
    """Run connectivity and environment checks for failed downloads.

    Returns list of ("pass"|"fail", message) tuples.
    """
    results: List[Tuple[str, str]] = []

    # Internet connectivity
    try:
        urllib.request.urlopen("https://www.google.com", timeout=5)
        results.append(("pass", "Internet connection active"))
    except Exception:
        results.append(("fail", "No internet connection detected"))
        return results

    # Per-model endpoint checks
    checked: set = set()
    for key in failed_model_keys:
        for url, host, label in _MODEL_ENDPOINTS.get(key, []):
            if host in checked:
                continue
            checked.add(host)
            try:
                urllib.request.urlopen(url, timeout=5)
                results.append(("pass", f"{host} reachable ({label})"))
            except Exception:
                results.append(("fail", f"{host} unreachable ({label})"))

    # Disk space
    try:
        free_gb = shutil.disk_usage(".").free / (1024 ** 3)
        if free_gb > 5:
            results.append(("pass", f"Disk space: {free_gb:.0f} GB free"))
        else:
            results.append(("fail", f"Low disk space: {free_gb:.1f} GB free (need ~5 GB)"))
    except Exception:
        results.append(("fail", "Could not check disk space"))

    # SAM3-specific: HF token
    if "sam3" in failed_model_keys:
        try:
            from huggingface_hub import get_token
            token = get_token()
            if token and len(token) > 0:
                results.append(("pass", "HuggingFace token saved"))
            else:
                results.append(("fail", "No HuggingFace token found"))
        except Exception:
            results.append(("fail", "Could not check HuggingFace token"))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# State checking
# ═══════════════════════════════════════════════════════════════════════════════

def check_all_models() -> List[ModelInfo]:
    result = []
    for m in ALL_MODELS:
        info = ModelInfo(m.name, m.key, m.size_label, m.gated)
        check = _CHECK_FNS.get(m.key)
        if check:
            try:
                info.ready = check()
            except Exception:
                info.ready = False
        result.append(info)
    return result


def all_models_ready() -> bool:
    return all(m.ready for m in check_all_models())


# ═══════════════════════════════════════════════════════════════════════════════
# Setup Wizard — 5-step CTkToplevel
# ═══════════════════════════════════════════════════════════════════════════════

class SetupWizard(ctk.CTkToplevel):
    """First-launch setup wizard matching the planning.pen mockups."""

    def __init__(self, parent, models: List[ModelInfo]):
        super().__init__(parent)
        self._app_root = parent
        self.title("Reconstruction Zone — First Launch Setup")
        self.geometry("620x660")
        self.resizable(False, False)
        self.configure(fg_color=_BG)
        self.models = models
        self.missing = [m for m in models if not m.ready]
        self.success = False
        self._queue: queue.Queue = queue.Queue()
        self._sam3_gate_resolved = False
        self._current_step = "welcome"
        self._failed_keys: List[str] = []
        self._failed_errors: Dict[str, str] = {}

        self.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        # Container for all step content — swapped per step
        self._container = ctk.CTkFrame(self, fg_color=_BG)
        self._container.pack(fill="both", expand=True, padx=24, pady=24)

        self._show_step1_welcome()

    # ── Helpers ──

    def _clear_container(self):
        for w in self._container.winfo_children():
            w.destroy()

    def _add_header(self, subtitle: str, subtitle_color: str = "#60a5fa"):
        hdr = ctk.CTkFrame(self._container, fg_color="transparent")
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="Reconstruction Zone", font=("", 18, "bold"),
                     text_color=_TEXT_PRIMARY).pack()
        ctk.CTkLabel(hdr, text=subtitle, font=("", 13),
                     text_color=subtitle_color).pack()

    def _make_model_card(self, parent, model: ModelInfo, state: str = "neutral",
                         status_text: str = "", status_color: str = _TEXT_MUTED,
                         show_progress: bool = False, show_retry: bool = False,
                         retry_key: str = ""):
        """Create a 101px model card matching the design spec.

        States: neutral, completed, active, waiting, failed
        """
        opacity = 1.0
        if state == "completed":
            opacity = 0.6
        elif state == "waiting":
            opacity = 0.4

        card = ctk.CTkFrame(parent, fg_color=_SURFACE, corner_radius=_CARD_RADIUS,
                            height=_CARD_HEIGHT)
        card.pack(fill="x", pady=(0, 8))
        card.pack_propagate(False)
        if opacity < 1.0:
            card.configure(fg_color=_SURFACE)
            # Apply opacity via alpha on all children text colors

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=_CARD_PAD, pady=_CARD_PAD)
        inner.pack_propagate(False)

        # Header row
        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x")

        icon_text = ""
        icon_color = _TEXT_MUTED
        if state == "completed":
            icon_text = "\u2713"
            icon_color = _SUCCESS
        elif state == "active":
            icon_text = "\u25B6"
            icon_color = "#60a5fa"
        elif state == "failed":
            icon_text = "\u2717"
            icon_color = _ERROR
        elif state == "waiting":
            icon_text = "\u2022"
            icon_color = _TEXT_DIM
        else:
            icon_text = "\u2022"
            icon_color = _TEXT_MUTED

        # Dim text for faded states
        name_color = _TEXT_SECONDARY if opacity >= 0.6 else _TEXT_DIM
        desc_color = _TEXT_MUTED if opacity >= 0.6 else _TEXT_DIM

        ctk.CTkLabel(header, text=icon_text, font=("", 14),
                     text_color=icon_color, width=20).pack(side="left")
        name_label = ctk.CTkLabel(header, text=f"{model.name} \u2014 {_short_name(model.key)}",
                     font=("", 12, "bold"), text_color=name_color)
        name_label.pack(side="left", padx=(4, 0))

        if model.gated and state == "neutral":
            ctk.CTkLabel(header, text="gated", font=("", 9, "bold"),
                         text_color=_WARNING).pack(side="right")

        # Description
        desc = ctk.CTkLabel(inner, text=_MODEL_DESCRIPTIONS.get(model.key, ""),
                            font=("", 11), text_color=desc_color,
                            anchor="nw", justify="left",
                            wraplength=540)
        desc.pack(fill="both", expand=True, pady=(4, 0))

        # Status row (anchored to bottom)
        status_row = ctk.CTkFrame(inner, fg_color="transparent", height=20)
        status_row.pack(fill="x", side="bottom")
        status_row.pack_propagate(False)

        if show_progress:
            prog_frame = ctk.CTkFrame(status_row, fg_color="transparent")
            prog_frame.pack(fill="x")
            pb = ctk.CTkProgressBar(prog_frame, height=8, width=400,
                                    fg_color="#374151", progress_color=_ACCENT)
            pb.pack(fill="x")
            pb.set(0.35)
            ctk.CTkLabel(status_row, text=status_text, font=("", 11),
                         text_color="#60a5fa", anchor="w").pack(fill="x")
        elif show_retry:
            ctk.CTkLabel(status_row, text=status_text, font=("", 10),
                         text_color=_ERROR, anchor="w").pack(side="left")
            btn = ctk.CTkButton(status_row, text="Retry", width=66, height=20,
                                font=("", 10, "bold"), corner_radius=3,
                                fg_color=_ACCENT, hover_color="#1d4ed8",
                                command=lambda k=retry_key: self._retry_model(k))
            btn.pack(side="right")
        elif status_text:
            ctk.CTkLabel(status_row, text=status_text, font=("", 11),
                         text_color=status_color, anchor="w").pack(fill="x")

        return card

    # ── Step 1: Welcome ──

    def _show_step1_welcome(self):
        self._clear_container()
        self._current_step = "welcome"
        self._add_header("Setup Wizard")

        # Context box
        ctx = ctk.CTkFrame(self._container, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
        ctx.pack(fill="x", pady=(12, 0))
        ctx_inner = ctk.CTkFrame(ctx, fg_color="transparent")
        ctx_inner.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(ctx_inner,
                     text="This app uses AI models to automatically detect and mask "
                          "objects in your images for photogrammetry and 3D reconstruction.",
                     font=("", 11), text_color=_TEXT_SECONDARY,
                     wraplength=540, justify="left", anchor="w").pack(fill="x")
        ctk.CTkLabel(ctx_inner,
                     text="The following models need to be downloaded before you can "
                          "start. This is a one-time setup.",
                     font=("", 11), text_color=_TEXT_MUTED,
                     wraplength=540, justify="left", anchor="w").pack(fill="x", pady=(6, 0))

        # Model cards
        cards = ctk.CTkFrame(self._container, fg_color="transparent")
        cards.pack(fill="x", pady=(10, 0))
        for m in self.models:
            state = "completed" if m.ready else "neutral"
            status = f"{m.size_label} downloaded" if m.ready else ""
            color = _SUCCESS if m.ready else _TEXT_MUTED
            self._make_model_card(cards, m, state=state,
                                  status_text=status, status_color=color)

        # Footer
        footer = ctk.CTkFrame(self._container, fg_color="transparent")
        footer.pack(fill="x", pady=(14, 0))
        ready_count = sum(1 for m in self.models if m.ready)
        total_mb = sum(_size_mb(m) for m in self.models if not m.ready)
        total_str = f"~{total_mb / 1000:.1f} GB" if total_mb > 1000 else f"~{total_mb} MB"

        row = ctk.CTkFrame(footer, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=f"Total download: {total_str}",
                     font=("", 11), text_color="#64748b").pack(side="left")
        ctk.CTkLabel(row, text=f"{ready_count}/{len(self.models)} ready",
                     font=("", 11, "bold"),
                     text_color=_SUCCESS if ready_count == len(self.models) else _TEXT_MUTED
                     ).pack(side="right")

        ctk.CTkButton(footer, text="Begin Setup", height=40,
                      font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                      fg_color=_ACCENT, hover_color="#1d4ed8",
                      command=self._on_begin_setup).pack(fill="x", pady=(14, 0))

    def _on_begin_setup(self):
        gated_missing = [m for m in self.missing if m.gated]
        if gated_missing:
            self._show_step2_sam3_gate()
        else:
            self._sam3_gate_resolved = True
            self._show_step3_downloading()

    # ── Step 2: SAM3 Gate ──

    def _show_step2_sam3_gate(self):
        self._clear_container()
        self._current_step = "sam3_gate"
        self._add_header("SAM3 Access")

        # Context
        ctx = ctk.CTkFrame(self._container, fg_color=_SURFACE, corner_radius=_CARD_RADIUS)
        ctx.pack(fill="x", pady=(12, 0))
        ci = ctk.CTkFrame(ctx, fg_color="transparent")
        ci.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(ci,
                     text="SAM3 is the primary masking engine \u2014 it lets you describe "
                          "what to remove in plain language. It\u0027s developed by Meta and "
                          "hosted on HuggingFace as a gated model, which means you need "
                          "to request access before downloading.",
                     font=("", 11), text_color=_TEXT_SECONDARY,
                     wraplength=540, justify="left", anchor="w").pack(fill="x")
        ctk.CTkLabel(ci,
                     text="This is a one-time process. Approval is usually instant "
                          "but can take up to a few hours.",
                     font=("", 11), text_color=_WARNING,
                     wraplength=540, justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        # Steps
        steps_frame = ctk.CTkFrame(self._container, fg_color="transparent")
        steps_frame.pack(fill="x", pady=(10, 0))
        for text in [
            "Create a free account at huggingface.co",
            "Visit huggingface.co/facebook/sam3 and click Request Access",
            "Once approved, create a token at huggingface.co/settings/tokens",
            "Paste the token below and click Verify",
        ]:
            row = ctk.CTkFrame(steps_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkFrame(row, fg_color="#334155", corner_radius=10,
                         width=20, height=20).pack(side="left")
            ctk.CTkLabel(row, text=text, font=("", 11),
                         text_color="#cbd5e1", wraplength=520,
                         anchor="w", justify="left").pack(side="left", padx=(10, 0))

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
        ctk.CTkLabel(ti, text="Your token is saved locally and never shared.",
                     font=("", 10), text_color=_TEXT_DIM).pack(anchor="w", pady=(4, 0))

        # Status label
        self._gate_status = ctk.CTkLabel(self._container, text="",
                                         font=("", 11), wraplength=540)
        self._gate_status.pack(pady=(8, 0))

        # Check if already have token
        try:
            from sam3_setup import check_sam3_setup
        except ImportError:
            from reconstruction_gui.sam3_setup import check_sam3_setup
        report = check_sam3_setup()
        if report.overall_stage in ("ready", "needs_weights"):
            self._gate_status.configure(text="SAM3 access verified!",
                                        text_color=_SUCCESS)
            self._sam3_gate_resolved = True
            self._app_root.after(500, self._show_step3_downloading)
        elif report.overall_stage == "needs_access":
            self._gate_status.configure(text=report.message,
                                        text_color=_WARNING)

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
            try:
                from sam3_setup import verify_hf_token_detailed
            except ImportError:
                from reconstruction_gui.sam3_setup import verify_hf_token_detailed
            report = verify_hf_token_detailed(token)
            self._queue.put(("sam3_verify", report))
        except Exception as e:
            self._queue.put(("error", f"Verification failed: {e}"))

    # ── Step 3: Downloading ──

    def _show_step3_downloading(self):
        self._clear_container()
        self._current_step = "downloading"
        self._add_header("Downloading Models")

        self._dl_cards_frame = ctk.CTkFrame(self._container, fg_color="transparent")
        self._dl_cards_frame.pack(fill="x", pady=(10, 0))

        # Build initial card states
        to_download = []
        for m in self.models:
            if m.ready:
                self._make_model_card(self._dl_cards_frame, m, state="completed",
                                      status_text=f"{m.size_label} downloaded",
                                      status_color=_SUCCESS)
            elif m.key in [x.key for x in self.missing]:
                to_download.append(m)
                self._make_model_card(self._dl_cards_frame, m, state="waiting",
                                      status_text="Waiting...",
                                      status_color=_TEXT_DIM)
            else:
                self._make_model_card(self._dl_cards_frame, m, state="waiting",
                                      status_text="Waiting...",
                                      status_color=_TEXT_DIM)

        self._to_download = to_download
        threading.Thread(target=self._download_worker, daemon=True).start()
        self._poll()

    def _download_worker(self):
        total = len(self._to_download)
        failures: List[str] = []
        for i, m in enumerate(self._to_download):
            self._queue.put(("progress", i, total, m.key))
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
            # Run diagnostics before reporting
            diag = run_diagnostics(failures)
            self._queue.put(("diagnostics", diag))
            self._queue.put(("all_done_with_errors", failures))
        else:
            self._queue.put(("all_done",))

    # ── Step 4: Done ──

    def _show_step4_done(self):
        self._clear_container()
        self._current_step = "done"
        self._add_header("Setup Complete", subtitle_color=_SUCCESS)

        cards = ctk.CTkFrame(self._container, fg_color="transparent")
        cards.pack(fill="x", pady=(10, 0))
        for m in self.models:
            self._make_model_card(cards, m, state="completed",
                                  status_text=f"{m.size_label} downloaded",
                                  status_color=_SUCCESS)

        ctk.CTkLabel(self._container, text="All models are installed and ready.",
                     font=("", 12), text_color=_TEXT_MUTED).pack(pady=(14, 0))

        ctk.CTkButton(self._container, text="Enter Reconstruction Zone", height=40,
                      font=("", 13, "bold"), corner_radius=_CARD_RADIUS,
                      fg_color=_ACCENT, hover_color="#1d4ed8",
                      command=self._close).pack(fill="x", pady=(14, 0))

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.success = True

    # ── Step 5: Error ──

    def _show_step5_error(self):
        self._clear_container()
        self._current_step = "error"
        self._add_header("Download Issue", subtitle_color=_ERROR)

        cards = ctk.CTkFrame(self._container, fg_color="transparent")
        cards.pack(fill="x", pady=(10, 0))
        for m in self.models:
            if m.key in self._failed_keys:
                error_msg = self._failed_errors.get(m.key, "Download failed")
                self._make_model_card(cards, m, state="failed",
                                      status_text=error_msg, show_retry=True,
                                      retry_key=m.key)
            else:
                self._make_model_card(cards, m, state="completed",
                                      status_text=f"{m.size_label} downloaded",
                                      status_color=_SUCCESS)

        # Diagnostics
        if hasattr(self, '_diag_results') and self._diag_results:
            diag_frame = ctk.CTkFrame(self._container, fg_color="transparent")
            diag_frame.pack(fill="x", padx=_CARD_PAD, pady=(10, 0))
            ctk.CTkLabel(diag_frame, text="Diagnostics", font=("", 12, "bold"),
                         text_color=_TEXT_SECONDARY).pack(anchor="w")
            for status, message in self._diag_results:
                row = ctk.CTkFrame(diag_frame, fg_color="transparent")
                row.pack(fill="x", pady=1)
                icon = "\u2713" if status == "pass" else "\u2717"
                color = _SUCCESS if status == "pass" else _ERROR
                text_c = _TEXT_MUTED if status == "pass" else _ERROR
                ctk.CTkLabel(row, text=icon, font=("", 11),
                             text_color=color, width=20).pack(side="left")
                ctk.CTkLabel(row, text=message, font=("", 11),
                             text_color=text_c).pack(side="left")

        self.protocol("WM_DELETE_WINDOW", self._close)

    def _retry_model(self, key: str):
        """Retry a single failed model download."""
        self._failed_keys = [k for k in self._failed_keys if k != key]
        if key in self._failed_errors:
            del self._failed_errors[key]

        # Find the model
        model = next((m for m in self.models if m.key == key), None)
        if not model:
            return

        # Switch to downloading view for this single model
        self._current_step = "downloading"

        def _worker():
            fn = _DOWNLOAD_FNS.get(key)
            if fn:
                try:
                    self._queue.put(("status", f"Retrying {model.name}..."))
                    fn(self._queue)
                    self._queue.put(("model_done", key))
                    self._queue.put(("retry_complete", key, True))
                except Exception as e:
                    self._queue.put(("model_error", key, str(e)))
                    self._queue.put(("retry_complete", key, False))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll()

    # ── Queue polling ──

    def _poll(self):
        try:
            while not self._queue.empty():
                msg = self._queue.get_nowait()
                self._handle_message(msg)
        except Exception:
            logger.exception("Setup wizard queue handler failed")
        if self._current_step not in ("done",):
            self._app_root.after(100, self._poll)

    def _handle_message(self, msg):
        msg_type = msg[0]

        if msg_type == "progress":
            pass  # Cards are static in step 3; progress shown via model_done

        elif msg_type == "status":
            pass  # Status updates logged but not shown in card UI

        elif msg_type == "model_done":
            key = msg[1]
            for m in self.models:
                if m.key == key:
                    m.ready = True
                    break

        elif msg_type == "model_error":
            key, error = msg[1], msg[2]
            if key not in self._failed_keys:
                self._failed_keys.append(key)
            self._failed_errors[key] = error

        elif msg_type == "all_done":
            self._show_step4_done()

        elif msg_type == "diagnostics":
            self._diag_results = msg[1]

        elif msg_type == "all_done_with_errors":
            self._show_step5_error()

        elif msg_type == "retry_complete":
            key, success = msg[1], msg[2]
            if success:
                # Check if all failures are now resolved
                if not self._failed_keys:
                    self._show_step4_done()
                else:
                    self._show_step5_error()
            else:
                # Re-run diagnostics
                diag = run_diagnostics(self._failed_keys)
                self._diag_results = diag
                self._show_step5_error()

        elif msg_type == "sam3_verify":
            report = msg[1]
            self._verify_btn.configure(state="normal", text="Verify")
            if report.overall_stage in ("ready", "ready_to_install", "needs_weights"):
                self._gate_status.configure(text="Access verified! Proceeding to download...",
                                            text_color=_SUCCESS)
                self._sam3_gate_resolved = True
                self._app_root.after(1000, self._show_step3_downloading)
            elif report.overall_stage == "needs_access":
                self._gate_status.configure(
                    text=report.message + "\n" + report.next_action,
                    text_color=_WARNING)
            elif report.overall_stage == "needs_token":
                self._gate_status.configure(text=report.message, text_color=_ERROR)
            else:
                self._gate_status.configure(text=report.message, text_color=_ERROR)

        elif msg_type == "error":
            logger.error("Wizard error: %s", msg[1])

    # ── Lifecycle ──

    def _close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _show_done(self):
        self._show_step4_done()

    def _on_close_attempt(self):
        if self._current_step in ("done", "error", "welcome", "sam3_gate"):
            self._close()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SHORT_NAMES = {
    "sam3": "Text-Prompted Segmentation",
    "rfdetr": "Object Detection",
    "yolo26": "Fast Detection",
}

def _short_name(key: str) -> str:
    return _SHORT_NAMES.get(key, key)

def _size_mb(m: ModelInfo) -> float:
    s = m.size_label.lower()
    if "gb" in s:
        return float(s.replace("gb", "").replace("~", "").strip()) * 1000
    elif "mb" in s:
        return float(s.replace("mb", "").replace("~", "").strip())
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_setup_wizard_if_needed(parent) -> bool:
    """Check all models and show the setup wizard if any are missing."""
    models = check_all_models()
    if all(m.ready for m in models):
        logger.info("All models ready — setup wizard not needed")
        return True

    missing_names = [m.name for m in models if not m.ready]
    logger.info("Models need setup: %s", ", ".join(missing_names))

    wizard = SetupWizard(parent, models)
    try:
        wizard.transient(parent)
        wizard.grab_set()
        wizard.focus()
    except Exception:
        logger.exception("Could not make setup wizard modal")
    return False

"""
First-run model weight checker with CTk progress UI.

Checks whether SAM3, RF-DETR, and YOLO26 weights are cached locally.
If any are missing, shows a dialog that downloads them by exercising
each library's native download path (HuggingFace hub, torch hub, etc).
"""
import os
import threading
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import customtkinter as ctk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight detection — check if each model's weights are already cached
# ---------------------------------------------------------------------------

def _sam3_cached() -> bool:
    """True if SAM3 checkpoint is in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache("facebook/sam3", "sam3.pt")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


def _rfdetr_cached() -> bool:
    """True if RF-DETR-Seg-Small weights exist in CWD or package dir."""
    # RF-DETR downloads to CWD as rf-detr-seg-small.pt
    candidates = [
        Path("rf-detr-seg-small.pt"),
        Path(__file__).resolve().parent.parent / "rf-detr-seg-small.pt",
    ]
    return any(p.exists() for p in candidates)


def _yolo26_cached() -> bool:
    """True if YOLO26-n-seg weights are in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache("openvision/yolo26-n-seg", "model.pt")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# (name, check_fn, download_fn_name, approx_size_mb)
MODEL_CHECKS = [
    ("SAM3 (3.3 GB)",     _sam3_cached,   "_download_sam3",   3300),
    ("RF-DETR-Seg (100 MB)", _rfdetr_cached, "_download_rfdetr", 100),
    ("YOLO26-n-seg (6 MB)",  _yolo26_cached, "_download_yolo26", 6),
]


def check_missing_models() -> List[Tuple[str, str, int]]:
    """Return list of (name, download_method, size_mb) for models not cached."""
    missing = []
    for name, check_fn, dl_method, size_mb in MODEL_CHECKS:
        try:
            if not check_fn():
                missing.append((name, dl_method, size_mb))
        except Exception:
            missing.append((name, dl_method, size_mb))
    return missing


# ---------------------------------------------------------------------------
# Download functions — exercise each library's native loading path
# ---------------------------------------------------------------------------

def _download_sam3(status_callback=None):
    """Download SAM3 weights via HuggingFace hub."""
    if status_callback:
        status_callback("Downloading SAM3 (3.3 GB) — this may take a few minutes...")
    from huggingface_hub import hf_hub_download
    hf_hub_download("facebook/sam3", "config.json")
    hf_hub_download("facebook/sam3", "sam3.pt")


def _download_rfdetr(status_callback=None):
    """Download RF-DETR-Seg-Small weights."""
    if status_callback:
        status_callback("Downloading RF-DETR-Seg (100 MB)...")
    import rfdetr
    # Instantiation triggers weight download
    model = rfdetr.RFDETRSegSmall()
    del model


def _download_yolo26(status_callback=None):
    """Download YOLO26-n-seg weights via ultralytics/HF."""
    if status_callback:
        status_callback("Downloading YOLO26-n-seg (6 MB)...")
    from ultralytics import YOLO
    model = YOLO("yolo26n-seg")
    del model


_DOWNLOAD_FNS = {
    "_download_sam3": _download_sam3,
    "_download_rfdetr": _download_rfdetr,
    "_download_yolo26": _download_yolo26,
}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ModelDownloadDialog(ctk.CTkToplevel):
    """Dialog that downloads missing model weights on first launch."""

    def __init__(self, parent, missing_models):
        super().__init__(parent)
        self.title("Downloading Model Weights")
        self.geometry("500x250")
        self.resizable(False, False)
        self.missing = missing_models  # list of (name, dl_method, size_mb)
        self.success = False

        total_mb = sum(m[2] for m in missing_models)
        total_str = f"{total_mb / 1000:.1f} GB" if total_mb > 1000 else f"{total_mb} MB"

        ctk.CTkLabel(self, text="First Launch — Downloading Models",
                     font=("", 14, "bold")).pack(pady=(15, 5))

        ctk.CTkLabel(self, text=f"{len(missing_models)} model{'s' if len(missing_models) != 1 else ''}, ~{total_str} total",
                     font=("Consolas", 11)).pack(pady=5)

        self.current_label = ctk.CTkLabel(self, text="Starting...",
                                          font=("Consolas", 11))
        self.current_label.pack(pady=8)

        self.progress = ctk.CTkProgressBar(self, width=440)
        self.progress.pack(pady=8, padx=25)
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(self, text="",
                                         font=("Consolas", 10),
                                         text_color="#9ca3af")
        self.status_label.pack(pady=5)

        # Prevent closing during download
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        # Start download thread
        threading.Thread(target=self._download_all, daemon=True).start()

    def _update_status(self, text):
        """Thread-safe status update."""
        self.after(0, lambda: self.status_label.configure(text=text))

    def _download_all(self):
        n = len(self.missing)
        try:
            for i, (name, dl_method, size_mb) in enumerate(self.missing):
                self.after(0, lambda n=name, i=i: (
                    self.current_label.configure(text=f"[{i+1}/{n}] {n}"),
                    self.progress.set(i / max(n, 1)),
                ))
                # Fix: capture name properly for the lambda
                self.after(0, lambda nm=name, idx=i: self.current_label.configure(
                    text=f"[{idx+1}/{n}] {nm}"))

                fn = _DOWNLOAD_FNS.get(dl_method)
                if fn:
                    fn(status_callback=self._update_status)

            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self.current_label.configure(
                text="All models ready!"))
            self.after(0, lambda: self.status_label.configure(text=""))
            self.success = True
            self.after(2000, self.destroy)
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            self.after(0, lambda: self.current_label.configure(
                text=f"Download failed"))
            self.after(0, lambda: self.status_label.configure(
                text=str(e)[:80]))
            # Re-enable close button on error
            self.after(0, lambda: self.protocol(
                "WM_DELETE_WINDOW", self.destroy))

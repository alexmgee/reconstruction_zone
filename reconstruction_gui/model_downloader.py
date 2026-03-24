"""
First-run model weight downloader with CTk progress UI.
Downloads SAM3 and RF-DETR weights from HuggingFace on first launch.
"""
import os
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import customtkinter as ctk


# Model registry: (name, url, local_path, size_mb)
# Populated when HuggingFace repos are finalized
MODEL_REGISTRY = []


def get_model_cache_dir() -> Path:
    """Return the model cache directory."""
    cache = Path(os.environ.get("PREP360_MODEL_DIR",
                                Path.home() / ".prep360_models"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def check_missing_models() -> List[Tuple[str, str, Path, int]]:
    """Return list of models that need downloading."""
    cache = get_model_cache_dir()
    missing = []
    for name, url, rel_path, size_mb in MODEL_REGISTRY:
        if not (cache / rel_path).exists():
            missing.append((name, url, cache / rel_path, size_mb))
    return missing


def download_model(url: str, dest: Path,
                   progress_callback: Optional[Callable] = None) -> bool:
    """Download a model file with progress reporting."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')

    try:
        req = urllib.request.urlopen(url)
        total = int(req.headers.get('Content-Length', 0))
        downloaded = 0

        with open(tmp, 'wb') as f:
            while True:
                chunk = req.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    progress_callback(downloaded / total)

        tmp.rename(dest)
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise


class ModelDownloadDialog(ctk.CTkToplevel):
    """Dialog showing model download progress."""

    def __init__(self, parent, missing_models):
        super().__init__(parent)
        self.title("Downloading Models")
        self.geometry("450x300")
        self.resizable(False, False)
        self.missing = missing_models
        self.success = False

        ctk.CTkLabel(self, text="Downloading required model weights...",
                     font=("", 14, "bold")).pack(pady=10)

        total_mb = sum(m[3] for m in missing_models)
        ctk.CTkLabel(self, text=f"{len(missing_models)} models, ~{total_mb} MB total"
                     ).pack(pady=5)

        self.current_label = ctk.CTkLabel(self, text="")
        self.current_label.pack(pady=5)

        self.progress = ctk.CTkProgressBar(self, width=400)
        self.progress.pack(pady=10, padx=20)
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(self, text="Starting...")
        self.status_label.pack(pady=5)

        # Start download thread
        threading.Thread(target=self._download_all, daemon=True).start()

    def _download_all(self):
        try:
            for i, (name, url, dest, size_mb) in enumerate(self.missing):
                self.current_label.configure(
                    text=f"[{i+1}/{len(self.missing)}] {name}")
                self.status_label.configure(text=f"Downloading {size_mb} MB...")

                download_model(url, dest,
                             progress_callback=lambda p: self.progress.set(p))

            self.success = True
            self.status_label.configure(text="All models downloaded!")
            self.after(1500, self.destroy)
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")

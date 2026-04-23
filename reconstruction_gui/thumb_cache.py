"""Thumbnail generation and disk caching for the Projects tab viewer."""

import hashlib
import threading
from pathlib import Path
from typing import Optional, Callable

from PIL import Image

THUMB_SIZE = (150, 150)
THUMB_QUALITY = 80
CACHE_DIR = Path.home() / ".prep360_thumb_cache"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _cache_path(image_path: Path) -> Path:
    """Deterministic cache path: hash of absolute path + mtime."""
    stat = image_path.stat()
    key = f"{image_path.resolve()}:{stat.st_mtime_ns}"
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.jpg"


def get_thumbnail(image_path: Path) -> Optional[Image.Image]:
    """Get thumbnail, generating and caching if needed. Returns PIL Image or None."""
    cached = _cache_path(image_path)
    if cached.exists():
        try:
            return Image.open(cached)
        except Exception:
            cached.unlink(missing_ok=True)
    try:
        img = Image.open(image_path)
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img.save(cached, "JPEG", quality=THUMB_QUALITY)
        return img
    except Exception:
        return None


def load_thumbnails_async(image_paths, callback, done_callback=None):
    """Load thumbnails in a background thread, calling callback for each.

    callback(path, pil_image) is called from the worker thread — caller must
    schedule UI updates on the main thread via app.after().
    done_callback() is called when all thumbnails are loaded.
    """
    def _worker():
        for path in image_paths:
            thumb = get_thumbnail(path)
            if thumb:
                callback(path, thumb)
        if done_callback:
            done_callback()

    threading.Thread(target=_worker, daemon=True).start()

"""Static mask layer management.

There is one shared library directory inside the reconstruction-zone project
root (``static_mask_library/``).  All static mask layers live in a single
``static_masks.json`` index inside that folder.  The same library is visible
across every dataset — you paint a mask once and reuse it.

Two save modes the GUI exposes:

- **Save to library** (default): ``add_layer(name, mask)`` writes the PNG to
  ``<library>/<name>.png`` and stores a relative ``filename`` in the index.
- **Save elsewhere**: ``add_layer_at_path(name, mask, dest)`` writes the PNG
  to a user-picked path and stores the absolute path in the index.

Import: ``import_layer(name, source_path)`` adds an absolute reference to an
existing PNG anywhere on disk (no copy is made).
"""

import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Library lives at <reconstruction-zone>/static_mask_library/ — same level as
# reconstruction_gui/ and prep360/.  Resolved once at import time so behaviour
# doesn't depend on cwd.
LIBRARY_DIR = Path(__file__).resolve().parent.parent / "static_mask_library"


@dataclass
class StaticMaskLayer:
    """A single named mask layer.

    ``filename`` is either a bare filename (resolved against the library
    directory) or an absolute path on disk.  The two modes are distinguished
    by ``Path(filename).is_absolute()``.
    """
    name: str
    filename: str
    enabled: bool = True
    created: str = ""


class StaticMaskManager:
    """Owns the shared library index and the PNG files inside it.

    Construct once at app startup; the library path is fixed.  Constructor
    arg is optional and only used for tests that need an isolated dir.
    """

    INDEX_NAME = "static_masks.json"

    def __init__(self, library_dir: Optional[Path] = None):
        self.static_dir = Path(library_dir) if library_dir else LIBRARY_DIR
        self.index_path = self.static_dir / self.INDEX_NAME
        self.layers: List[StaticMaskLayer] = []
        self._load_index()

    # ------------------------------------------------------------------ paths
    def _resolve(self, layer: StaticMaskLayer) -> Path:
        """Map a layer's ``filename`` to an absolute path on disk."""
        p = Path(layer.filename)
        return p if p.is_absolute() else (self.static_dir / p)

    # ------------------------------------------------------------------ index
    def _load_index(self):
        """Load layer index from sidecar JSON, or start empty."""
        if not self.index_path.exists():
            self.layers = []
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load static mask index: {e}")
            self.layers = []
            return
        try:
            self.layers = [StaticMaskLayer(**entry) for entry in data.get("layers", [])]
        except TypeError as e:
            logger.error(f"Static mask index has unexpected fields: {e}")
            self.layers = []
            return
        # Prune entries whose PNGs no longer exist (handles moved/deleted
        # files for both relative-in-library and absolute-anywhere layers).
        before = len(self.layers)
        self.layers = [l for l in self.layers if self._resolve(l).exists()]  # noqa: E741
        if len(self.layers) < before:
            logger.warning(f"Pruned {before - len(self.layers)} missing static mask layers")
            self._save_index()

    def _save_index(self):
        """Persist layer index to sidecar JSON."""
        self.static_dir.mkdir(parents=True, exist_ok=True)
        data = {"layers": [asdict(l) for l in self.layers]}  # noqa: E741
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ helpers
    def _unique_library_filename(self, name: str) -> str:
        """Build a library-local filename derived from ``name``, collision-free."""
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        safe = safe.strip().replace(" ", "_").lower() or "static_mask"
        filename = f"{safe}.png"
        counter = 1
        while (self.static_dir / filename).exists():
            filename = f"{safe}_{counter}.png"
            counter += 1
        return filename

    # ------------------------------------------------------------------ CRUD
    def add_layer(self, name: str, mask: np.ndarray) -> StaticMaskLayer:
        """Save ``mask`` into the library and add it as a layer.

        Args:
            name: Human-readable layer name.
            mask: Binary mask array, 0/1 uint8.

        Returns:
            The created StaticMaskLayer (filename is library-relative).
        """
        filename = self._unique_library_filename(name)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        # Convention matches regular output masks: black (0) = masked.
        cv2.imwrite(str(self.static_dir / filename), (1 - mask) * 255)
        layer = StaticMaskLayer(
            name=name, filename=filename, enabled=True,
            created=datetime.now().isoformat(),
        )
        self.layers.append(layer)
        self._save_index()
        logger.info(f"Added static mask (library): {name} -> {filename}")
        return layer

    def add_layer_at_path(self, name: str, mask: np.ndarray, dest: Path) -> StaticMaskLayer:
        """Save ``mask`` at a user-specified absolute path, then index it.

        Args:
            name: Human-readable layer name.
            mask: Binary mask array, 0/1 uint8.
            dest: Absolute path where the PNG should be written.  Parent
                  directories are created if missing.

        Returns:
            The created StaticMaskLayer (filename is the absolute path).
        """
        dest = Path(dest).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), (1 - mask) * 255)
        layer = StaticMaskLayer(
            name=name, filename=str(dest), enabled=True,
            created=datetime.now().isoformat(),
        )
        self.layers.append(layer)
        self._save_index()
        logger.info(f"Added static mask (external): {name} -> {dest}")
        return layer

    def import_layer(self, name: str, source_path: Path) -> Optional[StaticMaskLayer]:
        """Reference an existing PNG anywhere on disk as a new layer.

        Does not copy the file; only stores its absolute path in the index.
        Returns None if the source file is unreadable.
        """
        source = Path(source_path).resolve()
        if not source.exists():
            logger.warning(f"Import skipped (missing): {source}")
            return None
        # Sanity check it's a readable image
        if cv2.imread(str(source), cv2.IMREAD_GRAYSCALE) is None:
            logger.warning(f"Import skipped (unreadable as mask): {source}")
            return None
        layer = StaticMaskLayer(
            name=name, filename=str(source), enabled=True,
            created=datetime.now().isoformat(),
        )
        self.layers.append(layer)
        self._save_index()
        logger.info(f"Imported static mask: {name} -> {source}")
        return layer

    def update_layer(self, index: int, mask: np.ndarray):
        """Overwrite an existing layer's mask PNG in place (wherever it lives)."""
        layer = self.layers[index]
        cv2.imwrite(str(self._resolve(layer)), (1 - mask) * 255)
        logger.info(f"Updated static mask layer: {layer.name}")

    def remove_layer(self, index: int):
        """Delete a layer's PNG and remove the index entry.

        Files that live inside the library are unlinked.  Externally-located
        files (absolute paths outside the library) are NOT deleted — we only
        drop the reference, so an imported library file used elsewhere is
        preserved.
        """
        layer = self.layers[index]
        png_path = self._resolve(layer)
        # Only delete if the file sits inside the library
        try:
            png_path.resolve().relative_to(self.static_dir.resolve())
            is_in_library = True
        except ValueError:
            is_in_library = False

        if is_in_library and png_path.exists():
            png_path.unlink()
        self.layers.pop(index)
        self._save_index()
        logger.info(
            f"Removed static mask layer: {layer.name} "
            f"({'deleted file' if is_in_library else 'index reference only'})"
        )

    def rename_layer(self, index: int, new_name: str):
        """Rename a layer (metadata only, filename unchanged)."""
        self.layers[index].name = new_name
        self._save_index()

    def set_enabled(self, index: int, enabled: bool):
        """Toggle a layer on/off."""
        self.layers[index].enabled = enabled
        self._save_index()

    # ------------------------------------------------------------------ compositing
    def get_composite(self, target_shape: tuple = None) -> Optional[np.ndarray]:
        """Composite all enabled layers into a single mask via np.maximum.

        Args:
            target_shape: (height, width) to resize layers to if they
                          don't match. None = use layers' native resolution.

        Returns:
            Composited 0/1 uint8 mask, or None if no enabled layers.
        """
        composite = None
        for layer in self.layers:
            if not layer.enabled:
                continue
            path = self._resolve(layer)
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                logger.warning(f"Could not load static mask: {path}")
                continue
            # Disk convention: black (0) = masked.  Internal: 1 = masked.
            binary = (raw < 128).astype(np.uint8)

            if target_shape and (binary.shape[0], binary.shape[1]) != target_shape:
                binary = cv2.resize(binary, (target_shape[1], target_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            if composite is None:
                composite = binary
            else:
                composite = np.maximum(composite, binary)

        return composite

    def get_enabled_paths(self) -> List[str]:
        """Return absolute paths to all enabled layer PNGs (for pipeline)."""
        paths = []
        for layer in self.layers:
            if layer.enabled:
                p = self._resolve(layer)
                if p.exists():
                    paths.append(str(p))
        return paths

    # ------------------------------------------------------------------ stamp
    def count_stampable_targets(self, target_dir: Path) -> int:
        """Count mask files in target_dir that would be modified by a stamp."""
        target_dir = Path(target_dir)
        if not target_dir.is_dir():
            return 0
        n = 0
        for entry in target_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in (".png", ".npy"):
                continue
            n += 1
        return n

    def stamp_onto_directory(self, target_dir: Path) -> dict:
        """OR the enabled-layers composite onto every mask file in target_dir.

        Modifies top-level *.png / *.npy files in place.  PNGs use the same
        on-disk convention as regular output masks (black=masked, white=keep);
        NPYs store the 0/1 uint8 array directly.

        Returns a dict with counts: modified, skipped_unreadable,
        skipped_resize_failed, total_examined, no_composite.
        """
        target_dir = Path(target_dir)
        stats = {
            "modified": 0,
            "skipped_unreadable": 0,
            "skipped_resize_failed": 0,
            "total_examined": 0,
            "no_composite": False,
        }
        composite = self.get_composite()
        if composite is None:
            stats["no_composite"] = True
            return stats

        for entry in target_dir.iterdir():
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix not in (".png", ".npy"):
                continue
            stats["total_examined"] += 1

            if suffix == ".npy":
                try:
                    existing_internal = np.load(entry).astype(np.uint8)
                except Exception as e:
                    logger.warning(f"Stamp skip (unreadable npy): {entry.name}: {e}")
                    stats["skipped_unreadable"] += 1
                    continue
            else:
                raw = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    logger.warning(f"Stamp skip (unreadable png): {entry.name}")
                    stats["skipped_unreadable"] += 1
                    continue
                existing_internal = (raw < 128).astype(np.uint8)

            stamp = composite
            if stamp.shape != existing_internal.shape:
                try:
                    stamp = cv2.resize(
                        composite,
                        (existing_internal.shape[1], existing_internal.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                except cv2.error as e:
                    logger.warning(f"Stamp skip (resize failed): {entry.name}: {e}")
                    stats["skipped_resize_failed"] += 1
                    continue

            merged = np.maximum(existing_internal, stamp).astype(np.uint8)

            if suffix == ".npy":
                np.save(entry, merged)
            else:
                cv2.imwrite(str(entry), (1 - merged) * 255)
            stats["modified"] += 1

        logger.info(
            f"Stamped {stats['modified']}/{stats['total_examined']} masks in {target_dir}"
        )
        return stats

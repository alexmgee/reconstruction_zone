"""Static mask layer management.

Each dataset can have a static_masks/ folder containing user-authored
mask overlays. Layers are composited via np.maximum and applied to
every frame during batch masking.
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


@dataclass
class StaticMaskLayer:
    """A single named mask layer."""
    name: str
    filename: str
    enabled: bool = True
    created: str = ""


class StaticMaskManager:
    """Manages static_masks/ folder and its sidecar JSON."""

    FOLDER_NAME = "static_masks"
    INDEX_NAME = "static_masks.json"

    def __init__(self, dataset_dir: Path):
        """
        Args:
            dataset_dir: Parent directory containing images/ and masks/.
                         static_masks/ will be created as a sibling.
        """
        self.dataset_dir = Path(dataset_dir)
        self.static_dir = self.dataset_dir / self.FOLDER_NAME
        self.index_path = self.static_dir / self.INDEX_NAME
        self.layers: List[StaticMaskLayer] = []
        self._load_index()

    def _load_index(self):
        """Load layer index from sidecar JSON, or start empty."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.layers = [
                    StaticMaskLayer(**entry) for entry in data.get("layers", [])
                ]
                # Prune entries whose PNGs no longer exist
                before = len(self.layers)
                self.layers = [
                    l for l in self.layers
                    if (self.static_dir / l.filename).exists()
                ]
                if len(self.layers) < before:
                    logger.warning(f"Pruned {before - len(self.layers)} missing static mask layers")
                    self._save_index()
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load static mask index: {e}")
                self.layers = []
        else:
            self.layers = []

    def _save_index(self):
        """Persist layer index to sidecar JSON."""
        self.static_dir.mkdir(parents=True, exist_ok=True)
        data = {"layers": [asdict(l) for l in self.layers]}
        self.index_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def add_layer(self, name: str, mask: np.ndarray) -> StaticMaskLayer:
        """Save a new static mask layer.

        Args:
            name: Human-readable layer name (e.g. "Nadir pole").
            mask: Binary mask array, 0/1 uint8, same resolution as input frames.

        Returns:
            The created StaticMaskLayer.
        """
        # Generate filename from name (sanitize)
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        safe = safe.strip().replace(" ", "_").lower()
        if not safe:
            safe = "static_mask"
        filename = f"{safe}.png"

        # Avoid collisions
        counter = 1
        while (self.static_dir / filename).exists():
            filename = f"{safe}_{counter}.png"
            counter += 1

        self.static_dir.mkdir(parents=True, exist_ok=True)
        # Save as white=mask (0/1 -> 0/255)
        cv2.imwrite(str(self.static_dir / filename), mask * 255)

        layer = StaticMaskLayer(
            name=name,
            filename=filename,
            enabled=True,
            created=datetime.now().isoformat(),
        )
        self.layers.append(layer)
        self._save_index()
        logger.info(f"Added static mask layer: {name} -> {filename}")
        return layer

    def update_layer(self, index: int, mask: np.ndarray):
        """Overwrite an existing layer's mask PNG."""
        layer = self.layers[index]
        cv2.imwrite(str(self.static_dir / layer.filename), mask * 255)
        logger.info(f"Updated static mask layer: {layer.name}")

    def remove_layer(self, index: int):
        """Delete a layer's PNG and remove from index."""
        layer = self.layers[index]
        png_path = self.static_dir / layer.filename
        if png_path.exists():
            png_path.unlink()
        self.layers.pop(index)
        self._save_index()
        logger.info(f"Removed static mask layer: {layer.name}")

    def rename_layer(self, index: int, new_name: str):
        """Rename a layer (metadata only, filename unchanged)."""
        self.layers[index].name = new_name
        self._save_index()

    def set_enabled(self, index: int, enabled: bool):
        """Toggle a layer on/off."""
        self.layers[index].enabled = enabled
        self._save_index()

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
            path = self.static_dir / layer.filename
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                logger.warning(f"Could not load static mask: {path}")
                continue
            # Convert 0-255 -> 0/1
            binary = (raw > 127).astype(np.uint8)

            if target_shape and (binary.shape[0], binary.shape[1]) != target_shape:
                binary = cv2.resize(binary, (target_shape[1], target_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            if composite is None:
                composite = binary
            else:
                composite = np.maximum(composite, binary)

        return composite

    def get_enabled_paths(self) -> List[str]:
        """Return absolute paths to all enabled layer PNGs."""
        paths = []
        for layer in self.layers:
            if layer.enabled:
                p = self.static_dir / layer.filename
                if p.exists():
                    paths.append(str(p))
        return paths

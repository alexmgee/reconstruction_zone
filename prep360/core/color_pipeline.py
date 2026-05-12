"""Shared image adjustment pipeline used by preview and export code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .adjustment_recipe import AdjustmentRecipe, OutputSettings
from .lut import LUTProcessor


@dataclass
class ImageLoadResult:
    path: Path
    image: np.ndarray
    source_dtype: str
    source_max_value: float
    alpha: Optional[np.ndarray] = None


def apply_adjustment_recipe(
    image_float_bgr: np.ndarray,
    recipe: AdjustmentRecipe,
    lut_processor: LUTProcessor | None = None,
) -> np.ndarray:
    """Apply an AdjustmentRecipe to a float32 BGR image in [0, 1]."""
    if image_float_bgr.ndim != 3 or image_float_bgr.shape[2] != 3:
        raise ValueError("apply_adjustment_recipe expects a 3-channel BGR image")

    result = np.clip(image_float_bgr.astype(np.float32), 0.0, 1.0)

    if recipe.input_lut.enabled:
        processor = lut_processor or LUTProcessor()
        lut_3d, _ = processor.load_cube(recipe.input_lut.path)
        result = processor.apply_float(result, lut_3d, recipe.input_lut.strength)

    state = _recipe_to_adjustment_state(recipe)
    result = _apply_adjustments(result, state)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def load_image_float(path: str | Path) -> ImageLoadResult:
    """Load an image as float32 BGR in [0, 1], preserving 8-bit/16-bit precision."""
    image_path = Path(path)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].copy()
        img = img[:, :, :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for {image_path}: {img.shape}")

    source_dtype = str(img.dtype)
    if img.dtype == np.uint8:
        max_value = 255.0
    elif img.dtype == np.uint16:
        max_value = 65535.0
    elif np.issubdtype(img.dtype, np.floating):
        max_value = 1.0 if float(np.nanmax(img)) <= 1.0 else 255.0
    else:
        max_value = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 255.0

    image_float = np.clip(img.astype(np.float32) / max_value, 0.0, 1.0)
    return ImageLoadResult(
        path=image_path,
        image=image_float,
        source_dtype=source_dtype,
        source_max_value=max_value,
        alpha=alpha,
    )


def write_image_float(path: str | Path, image: np.ndarray, output: OutputSettings) -> None:
    """Write a float32 BGR image according to OutputSettings."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = output.format.lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt == "tif":
        fmt = "tiff"

    bgr = np.clip(image, 0.0, 1.0)
    params: list[int] = []

    if fmt == "jpg":
        out_img = np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(output.quality)]
    elif fmt == "png":
        if output.bit_depth == "16-bit":
            out_img = np.clip(np.rint(bgr * 65535.0), 0, 65535).astype(np.uint16)
        else:
            out_img = np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)
        params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, (100 - int(output.quality)) // 10))]
    elif fmt == "tiff":
        if output.bit_depth == "16-bit":
            out_img = np.clip(np.rint(bgr * 65535.0), 0, 65535).astype(np.uint16)
        else:
            out_img = np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported output format: {output.format}")

    ok = cv2.imwrite(str(out_path), out_img, params)
    if not ok:
        raise ValueError(f"Could not write image: {out_path}")


def output_extension(output: OutputSettings) -> str:
    fmt = output.format.lower()
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt == "png":
        return ".png"
    if fmt in {"tiff", "tif"}:
        return ".tiff"
    raise ValueError(f"Unsupported output format: {output.format}")


def _recipe_to_adjustment_state(recipe: AdjustmentRecipe):
    AdjustmentState = _adjust_engine_symbol("AdjustmentState")
    state = AdjustmentState()
    state.exposure = recipe.tone.exposure
    state.contrast = recipe.tone.contrast
    state.highlights = recipe.tone.highlights
    state.shadows = recipe.tone.shadows
    state.whites = recipe.tone.whites
    state.blacks = recipe.tone.blacks
    state.temperature = recipe.white_balance.temperature
    state.tint = recipe.white_balance.tint
    state.saturation = recipe.color.saturation
    state.vibrance = recipe.color.vibrance
    state.sharpen_amount = recipe.detail.sharpen_amount
    state.sharpen_radius = recipe.detail.sharpen_radius
    state.sharpen_threshold = recipe.detail.sharpen_threshold
    state.denoise_strength = recipe.detail.denoise_strength
    state.denoise_method = recipe.detail.denoise_method
    state.clahe_clip = recipe.corrections.clahe_clip
    state.vignette_strength = recipe.corrections.vignette_strength
    return state


def _apply_adjustments(image: np.ndarray, state) -> np.ndarray:
    return _adjust_engine_symbol("apply_adjustments")(image, state)


def _adjust_engine_symbol(name: str):
    try:
        from reconstruction_gui import adjust_engine as engine
    except ImportError:
        import adjust_engine as engine
    return getattr(engine, name)


__all__ = [
    "ImageLoadResult",
    "apply_adjustment_recipe",
    "load_image_float",
    "write_image_float",
    "output_extension",
]

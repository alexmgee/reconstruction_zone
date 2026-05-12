"""Pure Adjust workflow helpers for dataset detection and export."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from prep360.core.adjustment_recipe import AdjustmentRecipe
from prep360.core.color_pipeline import (
    apply_adjustment_recipe,
    load_image_float,
    output_extension,
    write_image_float,
)
from prep360.core.lut import LUTProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".dng", ".arw", ".raf", ".raw"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".webm"}
ALL_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | RAW_EXTENSIONS


@dataclass
class AdjustDataset:
    kind: str
    root: Path
    images: list[Path] = field(default_factory=list)
    front_images: list[Path] = field(default_factory=list)
    back_images: list[Path] = field(default_factory=list)
    source_manifest: Optional[Path] = None
    video_path: Optional[Path] = None

    @property
    def is_paired(self) -> bool:
        return self.kind == "paired_extraction"

    @property
    def count(self) -> int:
        if self.is_paired:
            return min(len(self.front_images), len(self.back_images))
        return len(self.images)


@dataclass
class AdjustmentExportResult:
    success: bool
    cancelled: bool
    output_root: Path
    written_files: list[Path] = field(default_factory=list)
    count: int = 0
    manifest_path: Optional[Path] = None
    recipe_path: Optional[Path] = None
    error: str = ""


class AdjustmentExportCancelled(RuntimeError):
    pass


def detect_adjust_input(path: str | Path) -> AdjustDataset:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        return AdjustDataset(kind="video_sample", root=p.parent, video_path=p)
    if p.is_file() and p.suffix.lower() in ALL_IMAGE_EXTENSIONS:
        return AdjustDataset(kind="single_image", root=p.parent, images=[p])
    if not p.is_dir():
        raise ValueError(f"Adjust input not found or not supported: {p}")

    front_dir = p / "front" / "frames"
    back_dir = p / "back" / "frames"
    if front_dir.is_dir() and back_dir.is_dir():
        source_manifest = _first_existing(
            p / "paired_extraction_manifest.json",
            p / "extraction_manifest.json",
        )
        front_images, back_images = _paired_images_from_manifest(
            p,
            source_manifest,
            list_images(front_dir),
            list_images(back_dir),
        )
        return AdjustDataset(
            kind="paired_extraction",
            root=p,
            front_images=front_images,
            back_images=back_images,
            source_manifest=source_manifest,
        )

    frames_dir = p / "frames"
    if frames_dir.is_dir():
        return AdjustDataset(
            kind="single_extraction",
            root=p,
            images=list_images(frames_dir),
            source_manifest=_first_existing(p / "extraction_manifest.json"),
        )

    return AdjustDataset(kind="image_folder", root=p, images=list_images(p))


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    files: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in ALL_IMAGE_EXTENSIONS:
            files.append(child)
    return files


def _paired_images_from_manifest(
    root: Path,
    manifest_path: Optional[Path],
    fallback_front: list[Path],
    fallback_back: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Return paired front/back image lists in manifest order when possible."""
    if not manifest_path or not manifest_path.exists():
        return fallback_front, fallback_back

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback_front, fallback_back

    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        return fallback_front, fallback_back

    front_images: list[Path] = []
    back_images: list[Path] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            return fallback_front, fallback_back
        front_value = pair.get("front_image")
        back_value = pair.get("back_image")
        if not isinstance(front_value, str) or not isinstance(back_value, str):
            return fallback_front, fallback_back
        front_path = _manifest_image_path(root, front_value)
        back_path = _manifest_image_path(root, back_value)
        if not front_path.exists() or not back_path.exists():
            return fallback_front, fallback_back
        front_images.append(front_path)
        back_images.append(back_path)

    return front_images, back_images


def _manifest_image_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def validate_adjust_export(dataset: AdjustDataset, recipe: AdjustmentRecipe) -> list[str]:
    errors = recipe.validate()
    if dataset.kind == "paired_extraction":
        if len(dataset.front_images) != len(dataset.back_images):
            errors.append(
                f"Paired export blocked: front/back count mismatch "
                f"({len(dataset.front_images)} front, {len(dataset.back_images)} back)"
            )
        if not dataset.front_images or not dataset.back_images:
            errors.append("Paired export blocked: no front/back images found")
    elif dataset.kind in {"single_extraction", "image_folder", "single_image"}:
        if not dataset.images:
            errors.append("No images found to export")
    elif dataset.kind == "video_sample":
        errors.append("Video sample export is handled by the Adjust tab, not dataset export")
    else:
        errors.append(f"Unsupported Adjust input kind: {dataset.kind}")
    return errors


def ensure_output_ready(output_root: str | Path, refuse_non_empty: bool = True) -> None:
    root = Path(output_root)
    if root.exists() and refuse_non_empty and any(root.iterdir()):
        raise ValueError(f"Output folder is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)


def export_adjusted_dataset(
    dataset: AdjustDataset,
    output_root: str | Path,
    recipe: AdjustmentRecipe,
    *,
    refuse_non_empty: bool = True,
    keep_partials: bool = False,
    normalize_exposure: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    lut_processor: Optional[LUTProcessor] = None,
) -> AdjustmentExportResult:
    out_root = Path(output_root)
    written: list[Path] = []
    created_dirs: list[Path] = []
    processor = lut_processor or LUTProcessor()

    errors = validate_adjust_export(dataset, recipe)
    if errors:
        raise ValueError("; ".join(errors))

    ensure_output_ready(out_root, refuse_non_empty=refuse_non_empty)
    created_dirs.append(out_root)

    try:
        if _is_cancelled(cancel_check):
            raise AdjustmentExportCancelled()

        ext = output_extension(recipe.output)
        entries = _build_export_entries(dataset, out_root, ext)
        total = len(entries)

        exposure_scales: dict[Path, float] = {}
        if normalize_exposure and total > 1:
            exposure_scales = _compute_normalization_scales(entries, recipe, processor, cancel_check, progress_callback)

        count = 0
        for idx, (source, target) in enumerate(entries, start=1):
            if _is_cancelled(cancel_check):
                raise AdjustmentExportCancelled()
            _remember_dir(target.parent, created_dirs)
            loaded = load_image_float(source)
            adjusted = apply_adjustment_recipe(loaded.image, recipe, processor)
            scale = exposure_scales.get(source)
            if scale is not None:
                adjusted = np.clip(adjusted * scale, 0.0, 1.0)
            write_image_float(target, adjusted, recipe.output)
            written.append(target)
            count += 1
            if progress_callback:
                progress_callback(idx, total, source.name)
            if _is_cancelled(cancel_check):
                raise AdjustmentExportCancelled()

        copied_manifest = _copy_source_manifest(dataset, out_root, written)
        recipe_path = out_root / "adjustment_recipe.json"
        recipe.save(recipe_path)
        written.append(recipe_path)
        manifest_path = out_root / "adjustment_manifest.json"
        _write_adjustment_manifest(dataset, out_root, recipe, manifest_path, copied_manifest, count)
        written.append(manifest_path)

        return AdjustmentExportResult(
            success=True,
            cancelled=False,
            output_root=out_root,
            written_files=list(written),
            count=count,
            manifest_path=manifest_path,
            recipe_path=recipe_path,
        )

    except AdjustmentExportCancelled:
        if not keep_partials:
            cleanup_written_outputs(written, created_dirs)
        return AdjustmentExportResult(
            success=False,
            cancelled=True,
            output_root=out_root,
            written_files=[] if not keep_partials else list(written),
            count=0,
            error="cancelled",
        )
    except Exception as e:
        if not keep_partials:
            cleanup_written_outputs(written, created_dirs)
        raise e


def cleanup_written_outputs(written_files: Iterable[Path], created_dirs: Iterable[Path] = ()) -> None:
    for path in reversed(list(written_files)):
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass
    for path in sorted(set(created_dirs), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass


def _build_export_entries(dataset: AdjustDataset, output_root: Path, ext: str) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    if dataset.kind == "paired_extraction":
        for lens, images in (("front", dataset.front_images), ("back", dataset.back_images)):
            out_dir = output_root / lens / "frames"
            for src in images:
                entries.append((src, out_dir / f"{src.stem}{ext}"))
    elif dataset.kind == "single_extraction":
        out_dir = output_root / "frames"
        for src in dataset.images:
            entries.append((src, out_dir / f"{src.stem}{ext}"))
    else:
        for src in dataset.images:
            entries.append((src, output_root / f"{src.stem}{ext}"))
    return entries


def _compute_normalization_scales(
    entries: list[tuple[Path, Path]],
    recipe: AdjustmentRecipe,
    processor: LUTProcessor,
    cancel_check: Optional[Callable[[], bool]],
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> dict[Path, float]:
    means: list[tuple[Path, float]] = []
    total = len(entries)
    for idx, (source, _target) in enumerate(entries, start=1):
        if _is_cancelled(cancel_check):
            raise AdjustmentExportCancelled()
        loaded = load_image_float(source)
        adjusted = apply_adjustment_recipe(loaded.image, recipe, processor)
        means.append((source, _luma_mean(adjusted)))
        if progress_callback:
            progress_callback(idx, total * 2, f"analyze {source.name}")
    target = float(np.median([mean for _path, mean in means])) if means else 0.0
    if target <= 0:
        return {}
    return {path: max(0.25, min(4.0, target / max(mean, 1e-6))) for path, mean in means}


def _luma_mean(image_bgr: np.ndarray) -> float:
    b, g, r = image_bgr[:, :, 0], image_bgr[:, :, 1], image_bgr[:, :, 2]
    return float(np.mean(0.2126 * r + 0.7152 * g + 0.0722 * b))


def _copy_source_manifest(dataset: AdjustDataset, output_root: Path, written: list[Path]) -> Optional[Path]:
    if not dataset.source_manifest or not dataset.source_manifest.exists():
        return None
    target = output_root / dataset.source_manifest.name
    shutil.copy2(dataset.source_manifest, target)
    written.append(target)
    return target


def _write_adjustment_manifest(
    dataset: AdjustDataset,
    output_root: Path,
    recipe: AdjustmentRecipe,
    manifest_path: Path,
    copied_manifest: Optional[Path],
    count: int,
) -> None:
    data = {
        "version": 1,
        "operation": "adjust_export",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(dataset.root),
        "output_dataset": str(output_root),
        "input_kind": dataset.kind,
        "complete": True,
        "image_count": count,
        "front_count": len(dataset.front_images) if dataset.is_paired else 0,
        "back_count": len(dataset.back_images) if dataset.is_paired else 0,
        "recipe_file": "adjustment_recipe.json",
        "source_manifest": copied_manifest.name if copied_manifest else None,
        "output": asdict(recipe.output),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _remember_dir(path: Path, created_dirs: list[Path]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path not in created_dirs:
        created_dirs.append(path)


def _is_cancelled(cancel_check: Optional[Callable[[], bool]]) -> bool:
    return bool(cancel_check and cancel_check())


__all__ = [
    "AdjustDataset",
    "AdjustmentExportResult",
    "AdjustmentExportCancelled",
    "detect_adjust_input",
    "list_images",
    "validate_adjust_export",
    "ensure_output_ready",
    "export_adjusted_dataset",
    "cleanup_written_outputs",
]

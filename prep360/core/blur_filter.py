"""
Blur detection and filtering for image sequences.

Uses multiple sharpness metrics (Laplacian variance, Sobel gradient, Brenner gradient)
to score images and filter out blurry frames. Useful as a post-extraction quality step
before reframing.
"""

import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BlurFilterConfig:
    """Configuration for blur filtering."""
    threshold: Optional[float] = None   # Absolute score threshold (keep if score > threshold)
    percentile: float = 80.0            # Keep images above this percentile (0-100)
    keep_top_n: Optional[int] = None    # Keep only the top N sharpest images
    workers: int = 4                     # Parallel workers for scoring


@dataclass
class BlurScore:
    """Blur score for a single image."""
    image_name: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BlurFilterResult:
    """Result of a blur filtering operation."""
    success: bool
    total_images: int
    kept_count: int
    rejected_count: int
    kept: List[str]         # Filenames of kept images
    rejected: List[str]     # Filenames of rejected images
    score_stats: Dict[str, float] = field(default_factory=dict)


def _score_image(image_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Score a single image for sharpness. Worker function for ProcessPoolExecutor.

    Returns (path, primary_score, metrics_dict).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_path, 0.0, {}

    metrics = {}

    # Laplacian variance — primary blur metric
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    metrics["laplacian_var"] = float(laplacian.var())

    # Sobel gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    metrics["sobel_mag"] = float(np.sqrt(sobelx**2 + sobely**2).mean())

    # Brenner gradient
    shifted = np.roll(img, 2, axis=1)
    metrics["brenner"] = float(((img.astype(float) - shifted.astype(float)) ** 2).mean())

    return image_path, metrics["laplacian_var"], metrics


class BlurFilter:
    """Blur detection and filtering for image sequences."""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    def __init__(self, config: Optional[BlurFilterConfig] = None):
        self.config = config or BlurFilterConfig()

    def analyze_image(self, image_path: str) -> BlurScore:
        """Score a single image for sharpness."""
        path_str, score, metrics = _score_image(str(image_path))
        return BlurScore(
            image_name=Path(path_str).name,
            score=score,
            metrics=metrics,
        )

    def analyze_batch(
        self,
        input_dir: str,
        progress_callback: Optional[Callable] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> List[BlurScore]:
        """
        Score all images in a directory for sharpness.

        Args:
            input_dir: Directory containing images.
            progress_callback: Optional callback(current, total, image_name).
            log: Optional callable for diagnostic log messages.

        Returns:
            List of BlurScore sorted by score descending (sharpest first).
        """
        _log = log or (lambda _: None)

        input_path = Path(input_dir)
        images = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if not images:
            return []

        paths = [str(p) for p in images]
        scores = []

        import time
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            for i, (path_str, score, metrics) in enumerate(
                executor.map(_score_image, paths)
            ):
                bs = BlurScore(
                    image_name=Path(path_str).name,
                    score=score,
                    metrics=metrics,
                )
                scores.append(bs)
                if progress_callback:
                    progress_callback(i + 1, len(images), bs.image_name)

        scores.sort(key=lambda s: s.score, reverse=True)
        _log(f"  Scored {len(scores)} images ({time.perf_counter() - t0:.1f}s)")
        if scores:
            vals = [s.score for s in scores]
            _log(f"  Laplacian variance: min={min(vals):.1f} max={max(vals):.1f} "
                 f"mean={np.mean(vals):.1f} median={np.median(vals):.1f} std={np.std(vals):.1f}")
            _log(f"  Top 3: {', '.join(f'{s.image_name}={s.score:.1f}' for s in scores[:3])}")
            _log(f"  Bottom 3: {', '.join(f'{s.image_name}={s.score:.1f}' for s in scores[-3:])}")
        return scores

    def filter_images(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[BlurFilterConfig] = None,
        progress_callback: Optional[Callable] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> BlurFilterResult:
        """
        Score and filter images, copying sharp ones to output_dir.

        Args:
            input_dir: Directory containing source images.
            output_dir: Directory to copy kept images to.
            config: Override config for this run (uses self.config if None).
            progress_callback: Optional callback(current, total, message).
            log: Optional callable for diagnostic log messages.

        Returns:
            BlurFilterResult with kept/rejected lists and statistics.
        """
        _log = log or (lambda _: None)
        cfg = config or self.config
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Score all images
        scores = self.analyze_batch(input_dir, progress_callback, log=log)

        if not scores:
            return BlurFilterResult(
                success=True,
                total_images=0,
                kept_count=0,
                rejected_count=0,
                kept=[],
                rejected=[],
            )

        score_values = [s.score for s in scores]

        # Determine cutoff
        if cfg.keep_top_n is not None:
            idx = min(cfg.keep_top_n, len(scores)) - 1
            cutoff = scores[idx].score
            _log(f"  Cutoff: keep top {cfg.keep_top_n} (score >= {cutoff:.1f})")
        elif cfg.threshold is not None:
            cutoff = cfg.threshold
            _log(f"  Cutoff: absolute threshold = {cfg.threshold:.1f}")
        else:
            cutoff = float(np.percentile(score_values, 100 - cfg.percentile))
            _log(f"  Cutoff: {cfg.percentile}th percentile = {cutoff:.1f}")

        kept = [s for s in scores if s.score >= cutoff]
        rejected = [s for s in scores if s.score < cutoff]
        _log(f"  Kept: {len(kept)}, Rejected: {len(rejected)}")

        if cutoff > 0:
            near = [s for s in scores if abs(s.score - cutoff) / cutoff < 0.1]
            if near:
                _log(f"  Borderline ({len(near)} within 10% of cutoff):")
                for s in near[:5]:
                    status = "KEPT" if s.score >= cutoff else "CUT"
                    _log(f"    {s.image_name}: {s.score:.1f} ({status})")

        # Copy kept images
        output_path.mkdir(parents=True, exist_ok=True)
        for s in kept:
            src = input_path / s.image_name
            if src.exists():
                shutil.copy2(src, output_path / s.image_name)

        return BlurFilterResult(
            success=True,
            total_images=len(scores),
            kept_count=len(kept),
            rejected_count=len(rejected),
            kept=[s.image_name for s in kept],
            rejected=[s.image_name for s in rejected],
            score_stats={
                "min": float(min(score_values)),
                "max": float(max(score_values)),
                "mean": float(np.mean(score_values)),
                "median": float(np.median(score_values)),
                "cutoff": float(cutoff),
            },
        )

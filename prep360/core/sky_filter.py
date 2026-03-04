"""
Sky Filter Module

Detect and filter frames that are predominantly sky (useless for photogrammetry).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable

import cv2
import numpy as np


@dataclass
class SkyMetrics:
    """Metrics used for sky detection."""
    brightness: float       # Mean brightness (0-1)
    saturation: float       # Mean saturation (0-1)
    keypoint_count: int     # Number of ORB keypoints
    blue_ratio: float       # Blue channel ratio
    edge_density: float     # Edge pixel ratio
    is_sky: bool            # Final classification


@dataclass
class SkyFilterConfig:
    """Configuration for sky filtering."""
    brightness_threshold: float = 0.85    # Above this = likely sky
    saturation_threshold: float = 0.15    # Below this = likely sky
    keypoint_threshold: int = 50          # Below this = likely sky
    edge_threshold: float = 0.02          # Below this = likely sky
    pitch_threshold: float = 30.0         # Skip views with pitch > this


class SkyFilter:
    """Detect and filter sky-dominated images."""

    def __init__(self, config: Optional[SkyFilterConfig] = None):
        self.config = config or SkyFilterConfig()
        self._orb = None  # Lazy initialization

    @property
    def orb(self):
        """Lazy-initialize ORB detector."""
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=500)
        return self._orb

    def analyze(self, image: np.ndarray) -> SkyMetrics:
        """
        Analyze an image and return sky metrics.

        Args:
            image: BGR image array

        Returns:
            SkyMetrics with analysis results
        """
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Brightness (from grayscale or V channel)
        brightness = gray.mean() / 255.0

        # Saturation (S channel)
        saturation = hsv[:, :, 1].mean() / 255.0

        # Feature detection (sky has few features)
        keypoints = self.orb.detect(gray, None)
        kp_count = len(keypoints)

        # Blue ratio (sky tends to be blue)
        b, g, r = cv2.split(image)
        total = b.mean() + g.mean() + r.mean() + 1e-6
        blue_ratio = b.mean() / total

        # Edge density (sky has few edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Classify
        is_sky = self._classify(brightness, saturation, kp_count, edge_density)

        return SkyMetrics(
            brightness=brightness,
            saturation=saturation,
            keypoint_count=kp_count,
            blue_ratio=blue_ratio,
            edge_density=edge_density,
            is_sky=is_sky,
        )

    def _classify(
        self,
        brightness: float,
        saturation: float,
        keypoint_count: int,
        edge_density: float
    ) -> bool:
        """Classify if image is sky based on metrics."""
        # All conditions must be met for sky classification
        bright_check = brightness > self.config.brightness_threshold
        sat_check = saturation < self.config.saturation_threshold
        kp_check = keypoint_count < self.config.keypoint_threshold
        edge_check = edge_density < self.config.edge_threshold

        # Require at least 3 of 4 conditions
        score = sum([bright_check, sat_check, kp_check, edge_check])
        return score >= 3

    def is_sky(self, image: np.ndarray) -> bool:
        """Quick check if image is sky."""
        return self.analyze(image).is_sky

    def is_sky_by_pitch(self, pitch: float) -> bool:
        """Check if view should be skipped based on pitch angle."""
        return pitch > self.config.pitch_threshold

    def filter_images(
        self,
        input_dir: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        dry_run: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Filter sky images from a directory.

        Args:
            input_dir: Input directory with images
            output_dir: Output directory for non-sky images
            progress_callback: Called with (current, total, filename)
            dry_run: If True, don't copy files

        Returns:
            (kept_files, sky_files)
        """
        import shutil

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

        kept = []
        sky = []

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images), img_path.name)

            # Load and analyze
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            metrics = self.analyze(image)

            if metrics.is_sky:
                sky.append(img_path.name)
            else:
                kept.append(img_path.name)
                if not dry_run:
                    shutil.copy2(img_path, output_path / img_path.name)

        return kept, sky

    def analyze_batch(
        self,
        input_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, SkyMetrics]:
        """
        Analyze all images in a directory without filtering.

        Returns:
            Dict mapping filename to SkyMetrics
        """
        input_path = Path(input_dir)

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

        results = {}

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images), img_path.name)

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            results[img_path.name] = self.analyze(image)

        return results


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter sky images")
    parser.add_argument("input", help="Input directory with images")
    parser.add_argument("output", nargs="?", help="Output directory for non-sky images")
    parser.add_argument("--brightness", "-b", type=float, default=0.85,
                        help="Brightness threshold (default: 0.85)")
    parser.add_argument("--saturation", "-s", type=float, default=0.15,
                        help="Saturation threshold (default: 0.15)")
    parser.add_argument("--keypoints", "-k", type=int, default=50,
                        help="Keypoint threshold (default: 50)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Analyze only, don't copy")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-image results")

    args = parser.parse_args()

    config = SkyFilterConfig(
        brightness_threshold=args.brightness,
        saturation_threshold=args.saturation,
        keypoint_threshold=args.keypoints,
    )

    filter = SkyFilter(config)

    if args.verbose or not args.output:
        # Analyze mode
        def progress(curr, total, name):
            pass

        results = filter.analyze_batch(args.input, progress)

        sky_count = sum(1 for m in results.values() if m.is_sky)
        print(f"Analyzed {len(results)} images: {sky_count} sky, {len(results) - sky_count} kept")

        if args.verbose:
            for name, metrics in results.items():
                status = "SKY" if metrics.is_sky else "OK"
                print(f"  {name}: {status} (bright={metrics.brightness:.2f} sat={metrics.saturation:.2f} kp={metrics.keypoint_count})")

    else:
        # Filter mode
        def progress(curr, total, name):
            print(f"[{curr}/{total}] {name}")

        kept, sky = filter.filter_images(
            args.input,
            args.output,
            progress,
            dry_run=args.dry_run
        )

        print(f"Kept: {len(kept)}, Sky: {len(sky)}")

    return 0


if __name__ == "__main__":
    exit(main())

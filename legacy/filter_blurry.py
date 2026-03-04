#!/usr/bin/env python3
"""
Filter blurry frames from an image sequence using Laplacian variance.

Higher Laplacian variance = sharper image.
"""
import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def calculate_blur_score(image_path: str) -> Tuple[str, float]:
    """
    Calculate blur score using Laplacian variance.
    
    Returns (path, score) where higher score = sharper image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_path, 0.0
    
    # Laplacian variance - classic blur detection
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    score = laplacian.var()
    
    return image_path, score


def calculate_blur_score_robust(image_path: str) -> Tuple[str, float, dict]:
    """
    Calculate multiple blur metrics for robust filtering.
    
    Returns (path, primary_score, all_metrics).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_path, 0.0, {}
    
    metrics = {}
    
    # Laplacian variance
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    metrics["laplacian_var"] = laplacian.var()
    
    # Sobel gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    metrics["sobel_mag"] = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # Brenner gradient
    shifted = np.roll(img, 2, axis=1)
    brenner = ((img.astype(float) - shifted.astype(float)) ** 2).mean()
    metrics["brenner"] = brenner
    
    # Combined score (normalized)
    primary_score = metrics["laplacian_var"]
    
    return image_path, primary_score, metrics


def filter_images(
    input_dir: Path,
    output_dir: Path,
    threshold: float = None,
    percentile: float = None,
    keep_top_n: int = None,
    workers: int = 4,
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Filter images based on blur score.
    
    Args:
        input_dir: Directory containing images
        output_dir: Where to copy sharp images (or None to filter in place)
        threshold: Absolute blur score threshold (keep if score > threshold)
        percentile: Keep images above this percentile (0-100)
        keep_top_n: Keep only the top N sharpest images
        workers: Number of parallel workers
        dry_run: Don't copy, just report
        verbose: Print detailed info
    
    Returns:
        (kept_images, rejected_images)
    """
    # Find images
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = [f for f in input_dir.iterdir() if f.suffix in extensions]
    images = sorted(images)
    
    if not images:
        print(f"No images found in {input_dir}")
        return [], []
    
    print(f"Analyzing {len(images)} images...")
    
    # Calculate blur scores in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(calculate_blur_score, [str(p) for p in images]))
    
    # Sort by score
    scored = [(Path(path), score) for path, score in results]
    scored.sort(key=lambda x: x[1], reverse=True)  # Highest (sharpest) first
    
    scores = [s for _, s in scored]
    
    if verbose:
        print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"Mean score: {np.mean(scores):.1f}")
        print(f"Median score: {np.median(scores):.1f}")
    
    # Determine threshold
    if keep_top_n:
        cutoff_score = scored[min(keep_top_n, len(scored)) - 1][1]
    elif percentile:
        cutoff_score = np.percentile(scores, 100 - percentile)
    elif threshold:
        cutoff_score = threshold
    else:
        # Default: keep top 80%
        cutoff_score = np.percentile(scores, 20)
    
    # Split into keep/reject
    kept = [(p, s) for p, s in scored if s >= cutoff_score]
    rejected = [(p, s) for p, s in scored if s < cutoff_score]
    
    print(f"Keeping {len(kept)}/{len(images)} images (threshold: {cutoff_score:.1f})")
    
    if verbose:
        print("\nTop 5 sharpest:")
        for p, s in scored[:5]:
            print(f"  {p.name}: {s:.1f}")
        print("\nBottom 5 (blurriest):")
        for p, s in scored[-5:]:
            print(f"  {p.name}: {s:.1f}")
    
    # Copy kept images
    if not dry_run and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, score in kept:
            shutil.copy2(path, output_dir / path.name)
        print(f"Copied {len(kept)} images to {output_dir}")
    
    return [str(p) for p, _ in kept], [str(p) for p, _ in rejected]


def main():
    parser = argparse.ArgumentParser(
        description="Filter blurry frames using Laplacian variance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep top 80%% sharpest (default)
  python filter_blurry.py ./frames ./sharp_frames
  
  # Keep top 100 sharpest images
  python filter_blurry.py ./frames ./sharp_frames --top 100
  
  # Keep images with score > 500
  python filter_blurry.py ./frames ./sharp_frames --threshold 500
  
  # Keep top 90%%
  python filter_blurry.py ./frames ./sharp_frames --percentile 90
  
  # Dry run - just analyze
  python filter_blurry.py ./frames --dry-run --verbose
        """
    )
    
    parser.add_argument("input", help="Input directory with images")
    parser.add_argument("output", nargs="?", help="Output directory for sharp images")
    parser.add_argument("--threshold", "-t", type=float, help="Absolute score threshold")
    parser.add_argument("--percentile", "-p", type=float, help="Keep top N percent (e.g., 80)")
    parser.add_argument("--top", "-n", type=int, help="Keep only top N sharpest")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Analyze only, don't copy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: {input_dir} not found", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else None
    
    if not args.dry_run and not output_dir:
        print("Error: Output directory required (or use --dry-run)", file=sys.stderr)
        sys.exit(1)
    
    kept, rejected = filter_images(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        percentile=args.percentile,
        keep_top_n=args.top,
        workers=args.workers,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    if args.verbose and rejected:
        print(f"\nRejected {len(rejected)} blurry images")


if __name__ == "__main__":
    main()

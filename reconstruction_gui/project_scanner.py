"""
Project Scanner — Discover photogrammetry projects on disk.

Scans specified root directories for:
- .psx files (Metashape projects)
- Directories containing image files (potential source media)
- Mask directories (siblings named 'masks')

Returns discovery results that can be matched against existing projects
or used to create new ones.

GUI-independent — can be used from CLI or background threads.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set
from datetime import datetime

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class ScanResult:
    """A discovered project-like cluster on disk."""
    psx_path: Optional[str] = None
    psx_modified: Optional[str] = None
    image_dirs: List[str] = field(default_factory=list)
    mask_dirs: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    root_dir: str = ""
    image_counts: dict = field(default_factory=dict)  # dir -> count

    def suggested_title(self) -> str:
        """Derive a project title from the directory structure."""
        if self.psx_path:
            return Path(self.psx_path).stem
        if self.root_dir:
            return Path(self.root_dir).name
        return "Untitled"


def scan_directory(
    root: str,
    max_depth: int = 5,
    progress_callback=None,
) -> List[ScanResult]:
    """Scan a directory tree for photogrammetry projects.

    Heuristic: A .psx file anchors a project. Nearby image/mask directories
    (within 2 levels up or down) are associated with it. Image directories
    without a nearby .psx are reported as standalone discoveries.

    Args:
        root: Directory to scan
        max_depth: Maximum recursion depth
        progress_callback: Optional callable(status_str) for progress updates

    Returns:
        List of ScanResult, one per discovered project cluster
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    psx_files: List[Path] = []
    image_dirs: List[Path] = []
    mask_dirs: List[Path] = []
    video_files: List[Path] = []
    claimed_dirs: Set[str] = set()

    # Walk the tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        depth = len(Path(dirpath).relative_to(root_path).parts)
        if depth > max_depth:
            dirnames.clear()
            continue

        if progress_callback and depth <= 2:
            progress_callback(f"Scanning: {dirpath}")

        # Skip hidden dirs and common non-project dirs
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d.lower() not in (
                "node_modules", "__pycache__", ".git", "venv", "env",
            )
        ]

        dir_p = Path(dirpath)

        for f in filenames:
            fl = f.lower()
            fp = dir_p / f
            if fl.endswith(".psx"):
                psx_files.append(fp)
            elif Path(fl).suffix in VIDEO_EXTS:
                video_files.append(fp)

        # Check if this dir contains images
        img_count = sum(1 for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS)
        if img_count >= 5:  # Threshold: at least 5 images to be interesting
            if dir_p.name.lower() in ("masks", "mask"):
                mask_dirs.append(dir_p)
            else:
                image_dirs.append(dir_p)

    # Cluster around .psx files
    results: List[ScanResult] = []

    for psx in psx_files:
        result = ScanResult(
            psx_path=str(psx),
            psx_modified=datetime.fromtimestamp(psx.stat().st_mtime).isoformat(),
            root_dir=str(psx.parent),
        )

        # Find image/mask dirs within 3 levels of the .psx
        psx_parent = psx.parent
        for img_dir in image_dirs:
            try:
                rel = img_dir.relative_to(psx_parent)
                if len(rel.parts) <= 3:
                    result.image_dirs.append(str(img_dir))
                    result.image_counts[str(img_dir)] = sum(
                        1 for f in img_dir.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS
                    )
                    claimed_dirs.add(str(img_dir))
            except ValueError:
                # Also check if psx is inside image dir's parent tree
                try:
                    rel = psx_parent.relative_to(img_dir.parent)
                    if len(rel.parts) <= 2:
                        result.image_dirs.append(str(img_dir))
                        result.image_counts[str(img_dir)] = sum(
                            1 for f in img_dir.iterdir()
                            if f.suffix.lower() in IMAGE_EXTS
                        )
                        claimed_dirs.add(str(img_dir))
                except ValueError:
                    pass

        for msk_dir in mask_dirs:
            try:
                rel = msk_dir.relative_to(psx_parent)
                if len(rel.parts) <= 3:
                    result.mask_dirs.append(str(msk_dir))
                    claimed_dirs.add(str(msk_dir))
            except ValueError:
                pass

        # Videos near .psx
        for vid in video_files:
            try:
                rel = vid.relative_to(psx_parent)
                if len(rel.parts) <= 3:
                    result.video_files.append(str(vid))
            except ValueError:
                pass

        results.append(result)

    # Report unclaimed image dirs as orphans
    for img_dir in image_dirs:
        if str(img_dir) not in claimed_dirs:
            result = ScanResult(
                root_dir=str(img_dir.parent),
                image_dirs=[str(img_dir)],
                image_counts={str(img_dir): sum(
                    1 for f in img_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTS
                )},
            )
            # Check for sibling mask dir
            sibling_mask = img_dir.parent / "masks"
            if sibling_mask.is_dir():
                result.mask_dirs.append(str(sibling_mask))
            results.append(result)

    return results


def scan_multiple_roots(
    roots: List[str],
    max_depth: int = 5,
    progress_callback=None,
) -> List[ScanResult]:
    """Scan multiple root directories."""
    all_results = []
    for root in roots:
        if progress_callback:
            progress_callback(f"Scanning root: {root}")
        all_results.extend(scan_directory(root, max_depth, progress_callback))
    return all_results

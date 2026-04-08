"""
Project Scanner — Discover photogrammetry projects on disk.

Scans specified root directories for:
- .psx files (Metashape projects)
- .rsproj / .rcproj files (RealityScan / RealityCapture projects)
- COLMAP workspaces (sparse/ directories with cameras.txt or cameras.bin)
- Directories containing image files (potential source media)
- Mask directories (siblings named 'masks')

Returns discovery results that can be matched against existing projects
or used to create new ones.

GUI-independent — can be used from CLI or background threads.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

# File extensions that identify reconstruction tool projects
TOOL_EXTS = {
    ".psx": "Metashape",
    ".rsproj": "RealityScan",
    ".rcproj": "RealityScan",
}

# Files that identify a COLMAP sparse reconstruction directory
COLMAP_INDICATORS = {"cameras.txt", "cameras.bin"}


@dataclass
class ScanResult:
    """A discovered project-like cluster on disk."""
    tool_files: Dict[str, str] = field(default_factory=dict)  # tool name -> path
    image_dirs: List[str] = field(default_factory=list)
    mask_dirs: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    root_dir: str = ""
    image_counts: dict = field(default_factory=dict)  # dir -> count

    # Backward compat for code that reads psx_path directly
    @property
    def psx_path(self) -> Optional[str]:
        return self.tool_files.get("Metashape")

    def suggested_title(self) -> str:
        """Derive a project title from the project root or tool file.

        Prefers the root directory name (the common ancestor of the cluster).
        Falls back to a tool project file stem (.psx, .rsproj) if it differs
        from the root dir name — COLMAP sparse dirs (named '0') are never
        used as titles.
        """
        root_name = Path(self.root_dir).name if self.root_dir else ""

        # Check if any tool file has a meaningful name different from the dir
        for tool, path in self.tool_files.items():
            if tool == "COLMAP":
                continue  # sparse/0/ is never a good title
            stem = Path(path).stem
            if stem and stem != root_name:
                return stem

        return root_name or "Untitled"

    def tool_summary(self) -> str:
        """One-line summary of which tools were found."""
        if self.tool_files:
            return ", ".join(sorted(self.tool_files.keys()))
        return "no tools"


def _detect_colmap(dir_p: Path, filenames: List[str]):
    """Check if a directory is (or contains) a COLMAP sparse reconstruction.

    Returns (sparse_path, project_root) if found, else None.
    sparse_path  = the directory with cameras.txt/bin (stored in tool_files)
    project_root = the project directory to use as the cluster anchor
    """
    # Check for sparse/N/ pattern (standard COLMAP layout)
    sparse = dir_p / "sparse"
    if sparse.is_dir():
        for child in sparse.iterdir():
            if child.is_dir():
                child_files = {f.name.lower() for f in child.iterdir() if f.is_file()}
                if child_files & COLMAP_INDICATORS:
                    return str(child), dir_p  # anchor is the project dir

    # Direct match: cameras.txt/cameras.bin in this directory itself
    lower_names = {f.lower() for f in filenames}
    if lower_names & COLMAP_INDICATORS:
        return str(dir_p), dir_p.parent  # anchor is one level up

    return None


def scan_directory(
    root: str,
    max_depth: int = 5,
    progress_callback=None,
) -> List[ScanResult]:
    """Scan a directory tree for photogrammetry projects.

    Heuristic: Tool project files (.psx, .rsproj, .rcproj) and COLMAP
    workspaces serve as anchors. Nearby image/mask directories (within 3
    levels) are associated with the nearest anchor. Image directories
    without a nearby anchor are reported as standalone discoveries.

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

    # Collected anchors: (tool_name, file_path, anchor_parent_dir)
    anchors: List[tuple] = []
    image_dirs: List[Path] = []
    mask_dirs: List[Path] = []
    video_files: List[Path] = []
    claimed_dirs: Set[str] = set()
    # Track which directories already have a COLMAP detection to avoid dupes
    colmap_roots: Set[str] = set()

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

        # Detect tool project files
        for f in filenames:
            fp = dir_p / f
            ext = fp.suffix.lower()
            if ext in TOOL_EXTS:
                anchors.append((TOOL_EXTS[ext], str(fp), dir_p))
            elif ext in VIDEO_EXTS:
                video_files.append(fp)

        # Detect COLMAP workspaces
        colmap_result = _detect_colmap(dir_p, filenames)
        if colmap_result and str(dir_p) not in colmap_roots:
            sparse_path, project_root = colmap_result
            anchors.append(("COLMAP", sparse_path, project_root))
            colmap_roots.add(str(dir_p))

        # Check if this dir contains images
        img_count = sum(1 for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS)
        if img_count >= 5:  # Threshold: at least 5 images to be interesting
            if dir_p.name.lower() in ("masks", "mask"):
                mask_dirs.append(dir_p)
            else:
                image_dirs.append(dir_p)

    # Cluster anchors that share a common ancestor within 2 levels.
    # This merges sibling tools (e.g. house/metashape/scene.psx +
    # house/colmap/sparse/0/) into one ScanResult with root_dir at
    # the common ancestor (house/).
    results = _cluster_anchors(anchors, merge_depth=2)

    # Associate nearby image/mask/video dirs with each cluster
    for result in results:
        anchor_parent = Path(result.root_dir)

        for img_dir in image_dirs:
            if _is_nearby(img_dir, anchor_parent, max_levels=3):
                result.image_dirs.append(str(img_dir))
                result.image_counts[str(img_dir)] = sum(
                    1 for f in img_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTS
                )
                claimed_dirs.add(str(img_dir))

        for msk_dir in mask_dirs:
            if _is_nearby(msk_dir, anchor_parent, max_levels=3):
                result.mask_dirs.append(str(msk_dir))
                claimed_dirs.add(str(msk_dir))

        for vid in video_files:
            if _is_nearby(vid, anchor_parent, max_levels=3):
                result.video_files.append(str(vid))

    return results


def _common_ancestor(a: Path, b: Path, max_up: int) -> Optional[Path]:
    """Find the nearest common ancestor of *a* and *b*, searching up to
    *max_up* levels above each.  Returns None if no common ancestor exists
    within that range."""
    a_parts = a.resolve().parts
    b_parts = b.resolve().parts

    # Find the longest shared prefix
    shared = 0
    for pa, pb in zip(a_parts, b_parts):
        if pa.lower() == pb.lower():  # case-insensitive for Windows
            shared += 1
        else:
            break

    if shared == 0:
        return None  # different drives

    ancestor = Path(*a_parts[:shared]) if shared > 0 else None
    if ancestor is None:
        return None

    # Check that neither path is more than max_up levels below the ancestor
    a_depth = len(a_parts) - shared
    b_depth = len(b_parts) - shared
    if a_depth > max_up or b_depth > max_up:
        return None

    return ancestor


def _cluster_anchors(
    anchors: List[tuple],
    merge_depth: int = 2,
) -> List[ScanResult]:
    """Group anchors whose parent directories share a common ancestor
    within *merge_depth* levels into single ScanResult clusters.

    Each anchor is (tool_name, tool_path, anchor_parent_dir).
    The merged cluster's root_dir is set to the common ancestor.
    """
    # Start with each anchor in its own cluster
    clusters: List[ScanResult] = []
    for tool_name, tool_path, anchor_parent in anchors:
        r = ScanResult(root_dir=str(anchor_parent))
        r.tool_files[tool_name] = tool_path
        clusters.append(r)

    # Iteratively merge clusters whose root_dirs share a common ancestor
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                ancestor = _common_ancestor(
                    Path(clusters[i].root_dir),
                    Path(clusters[j].root_dir),
                    max_up=merge_depth,
                )
                if ancestor is not None:
                    # Merge j into i
                    clusters[i].tool_files.update(clusters[j].tool_files)
                    clusters[i].root_dir = str(ancestor)
                    clusters.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    return clusters


def _is_nearby(path: Path, anchor: Path, max_levels: int = 3) -> bool:
    """Check if *path* is within *max_levels* of *anchor* in either direction."""
    try:
        rel = path.relative_to(anchor)
        return len(rel.parts) <= max_levels
    except ValueError:
        pass
    try:
        rel = anchor.relative_to(path.parent)
        return len(rel.parts) <= 2
    except ValueError:
        return False


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

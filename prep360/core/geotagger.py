"""
Frame Geotagger

Write EXIF GPS coordinates, focal length, and capture datetime into extracted
frames using metadata from DJI SRT telemetry files. Uses exiftool for robust
EXIF writing.

Two modes:
  1. Manifest-based: reads extraction_manifest.json to map frames to timestamps
  2. Interval-based: computes timestamps from frame numbering + extraction interval
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Dict

from .srt_parser import SrtData, SrtEntry, parse_srt, find_srt_for_video

_SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# Manifest filename written by FrameExtractor alongside extracted frames
MANIFEST_FILENAME = "extraction_manifest.json"


@dataclass
class GeotagResult:
    """Result of geotagging operation."""
    success: bool
    tagged_count: int
    skipped_count: int       # Frames with no GPS data at their timestamp
    total_frames: int
    errors: List[str]


def _find_exiftool() -> Optional[str]:
    """Find exiftool on PATH."""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True, text=True,
            **_SUBPROCESS_FLAGS,
        )
        if result.returncode == 0:
            return "exiftool"
    except FileNotFoundError:
        pass
    return None


def _build_exiftool_args(entry: SrtEntry) -> List[str]:
    """Build exiftool arguments for a single frame's metadata.

    Writes: GPS coordinates, altitude, focal length, capture datetime.
    """
    args = []

    if entry.has_gps:
        lat = entry.latitude
        lon = entry.longitude
        args.append(f"-GPSLatitude={abs(lat)}")
        args.append(f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}")
        args.append(f"-GPSLongitude={abs(lon)}")
        args.append(f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}")

    if entry.abs_alt is not None:
        alt = entry.abs_alt
        args.append(f"-GPSAltitude={abs(alt)}")
        args.append(f"-GPSAltitudeRef={'Above Sea Level' if alt >= 0 else 'Below Sea Level'}")

    if entry.focal_len is not None:
        args.append(f"-FocalLength={entry.focal_len}")
        # FocalLengthIn35mmFormat — Mavic 3 Pro sensor crop varies by lens,
        # but the SRT focal_len is already the 35mm equivalent
        args.append(f"-FocalLengthIn35mmFormat={entry.focal_len}")

    if entry.datetime_str:
        # Convert "2025-10-23 10:59:27.465" to EXIF format "2025:10:23 10:59:27"
        dt = entry.datetime_str.split(".")[0].replace("-", ":")
        args.append(f"-DateTimeOriginal={dt}")
        args.append(f"-CreateDate={dt}")

    return args


def geotag_from_manifest(
    frames_dir: str,
    srt_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> GeotagResult:
    """Geotag frames using extraction manifest + SRT telemetry.

    Reads extraction_manifest.json from frames_dir, looks up each frame's
    timestamp in the SRT data, and writes EXIF metadata via exiftool.

    Args:
        frames_dir: Directory containing extracted frames + manifest
        srt_path: Path to DJI .SRT file
        progress_callback: Called with (current, total, message)

    Returns:
        GeotagResult with counts and any errors
    """
    frames_dir = Path(frames_dir)
    manifest_path = frames_dir / MANIFEST_FILENAME

    if not manifest_path.exists():
        return GeotagResult(
            success=False, tagged_count=0, skipped_count=0, total_frames=0,
            errors=[f"Manifest not found: {manifest_path}"],
        )

    exiftool = _find_exiftool()
    if not exiftool:
        return GeotagResult(
            success=False, tagged_count=0, skipped_count=0, total_frames=0,
            errors=["exiftool not found on PATH. Install from https://exiftool.org"],
        )

    # Load manifest and SRT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    srt = parse_srt(srt_path)
    frames = manifest.get("frames", [])

    return _geotag_frames(exiftool, frames_dir, frames, srt, progress_callback)


def geotag_from_interval(
    frames_dir: str,
    srt_path: str,
    interval: float,
    start_sec: float = 0.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> GeotagResult:
    """Geotag frames by computing timestamps from interval.

    Fallback when no manifest exists. Assumes frames are sequentially numbered
    starting at 00001 and extracted at a fixed interval.

    Args:
        frames_dir: Directory containing extracted frames
        srt_path: Path to DJI .SRT file
        interval: Extraction interval in seconds
        start_sec: Start time offset in seconds
        progress_callback: Called with (current, total, message)
    """
    frames_dir = Path(frames_dir)

    exiftool = _find_exiftool()
    if not exiftool:
        return GeotagResult(
            success=False, tagged_count=0, skipped_count=0, total_frames=0,
            errors=["exiftool not found on PATH. Install from https://exiftool.org"],
        )

    srt = parse_srt(srt_path)

    # Find all image files, sorted by name
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_exts and f.name != MANIFEST_FILENAME
    )

    if not image_files:
        return GeotagResult(
            success=False, tagged_count=0, skipped_count=0, total_frames=0,
            errors=[f"No image files found in {frames_dir}"],
        )

    # Build frame list with computed timestamps
    frames = []
    for i, img in enumerate(image_files):
        frames.append({
            "filename": img.name,
            "time_sec": start_sec + i * interval,
        })

    return _geotag_frames(exiftool, frames_dir, frames, srt, progress_callback)


def _geotag_frames(
    exiftool: str,
    frames_dir: Path,
    frames: List[Dict],
    srt: SrtData,
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> GeotagResult:
    """Core geotagging: match frames to SRT entries and write EXIF in batch."""
    total = len(frames)
    tagged = 0
    skipped = 0
    errors = []

    # Build a batch argfile for exiftool — one command for all frames
    # Format: one arg per line, frames separated by -execute
    argfile_lines = []
    frames_to_tag = []

    for i, frame_info in enumerate(frames):
        filename = frame_info["filename"]
        time_sec = frame_info.get("time_sec", 0.0)
        frame_path = frames_dir / filename

        if not frame_path.exists():
            skipped += 1
            continue

        entry = srt.lookup(time_sec)
        if entry is None or not entry.has_gps:
            skipped += 1
            continue

        exif_args = _build_exiftool_args(entry)
        if not exif_args:
            skipped += 1
            continue

        # Add this frame's args to the batch
        for arg in exif_args:
            argfile_lines.append(arg)
        argfile_lines.append("-overwrite_original")
        argfile_lines.append(str(frame_path))
        argfile_lines.append("-execute")
        frames_to_tag.append(filename)

    if not frames_to_tag:
        return GeotagResult(
            success=True, tagged_count=0, skipped_count=skipped,
            total_frames=total, errors=errors,
        )

    # Write argfile and run exiftool in batch mode
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n".join(argfile_lines))
            argfile_path = f.name

        if progress_callback:
            progress_callback(0, len(frames_to_tag), "Writing EXIF metadata...")

        result = subprocess.run(
            [exiftool, "-@", argfile_path],
            capture_output=True, text=True,
            **_SUBPROCESS_FLAGS,
        )

        # Count successes from exiftool output
        # Each successful write produces "1 image files updated"
        tagged = result.stdout.count("1 image files updated")

        if result.returncode != 0 and tagged == 0:
            errors.append(f"exiftool error: {result.stderr.strip()}")

        if progress_callback:
            progress_callback(tagged, len(frames_to_tag), f"Tagged {tagged} frames")

    finally:
        try:
            os.unlink(argfile_path)
        except OSError:
            pass

    return GeotagResult(
        success=len(errors) == 0,
        tagged_count=tagged,
        skipped_count=skipped,
        total_frames=total,
        errors=errors,
    )

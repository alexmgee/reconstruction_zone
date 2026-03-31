"""
DJI SRT Telemetry Parser

Parse DJI drone SRT subtitle files to extract per-frame GPS coordinates,
camera settings, and other telemetry. DJI records telemetry at 50fps (20ms
intervals) in standard SRT subtitle format with metadata in bracketed fields.

Supported fields: iso, shutter, fnum, ev, color_md, focal_len,
                  latitude, longitude, rel_alt, abs_alt, ct (color temp)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class SrtEntry:
    """Single telemetry entry from a DJI SRT file."""
    index: int                          # SRT sequence number (1-based)
    timestamp_ms: int                   # Start timestamp in milliseconds
    datetime_str: Optional[str] = None  # e.g. "2025-10-23 10:59:27.465"

    # GPS
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rel_alt: Optional[float] = None     # Relative altitude (above takeoff)
    abs_alt: Optional[float] = None     # Absolute altitude (above sea level)

    # Camera
    iso: Optional[int] = None
    shutter: Optional[str] = None       # e.g. "1/8000.0"
    fnum: Optional[float] = None        # f-number (aperture)
    ev: Optional[float] = None          # Exposure value
    focal_len: Optional[float] = None   # Focal length in mm
    color_md: Optional[str] = None      # Color mode (e.g. "dlog_m")
    ct: Optional[int] = None            # Color temperature

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ms / 1000.0


@dataclass
class SrtData:
    """Parsed SRT file with time-indexed lookup."""
    entries: List[SrtEntry]
    source_path: str
    total_duration_ms: int = 0

    def lookup(self, time_sec: float) -> Optional[SrtEntry]:
        """Find the nearest SRT entry to a given timestamp in seconds.

        Uses binary search for efficiency (SRT files can have 5000+ entries).
        """
        if not self.entries:
            return None

        target_ms = int(time_sec * 1000)

        # Binary search for nearest entry
        lo, hi = 0, len(self.entries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.entries[mid].timestamp_ms < target_ms:
                lo = mid + 1
            else:
                hi = mid

        # Check neighbors to find true nearest
        best = lo
        if best > 0:
            diff_lo = abs(self.entries[best - 1].timestamp_ms - target_ms)
            diff_hi = abs(self.entries[best].timestamp_ms - target_ms)
            if diff_lo < diff_hi:
                best = best - 1

        return self.entries[best]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the SRT data."""
        gps_entries = [e for e in self.entries if e.has_gps]
        focal_lengths = sorted(set(e.focal_len for e in self.entries if e.focal_len))
        return {
            "total_entries": len(self.entries),
            "duration_sec": self.total_duration_ms / 1000.0,
            "gps_entries": len(gps_entries),
            "gps_coverage": f"{len(gps_entries) / len(self.entries) * 100:.0f}%" if self.entries else "0%",
            "focal_lengths_mm": focal_lengths,
            "lat_range": (
                min(e.latitude for e in gps_entries),
                max(e.latitude for e in gps_entries),
            ) if gps_entries else None,
            "lon_range": (
                min(e.longitude for e in gps_entries),
                max(e.longitude for e in gps_entries),
            ) if gps_entries else None,
        }


# Regex patterns for parsing
_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
_FIELD_RE = re.compile(r"\[(\w+):\s*([^\]]+)\]")
_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+")


def parse_srt(path: str) -> SrtData:
    """Parse a DJI SRT file into structured telemetry data.

    Args:
        path: Path to the .SRT file

    Returns:
        SrtData with all parsed entries and time-indexed lookup
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", text.strip())
    entries = []

    for block in blocks:
        entry = _parse_block(block.strip())
        if entry is not None:
            entries.append(entry)

    # Sort by timestamp (should already be sorted, but be safe)
    entries.sort(key=lambda e: e.timestamp_ms)

    total_duration = entries[-1].timestamp_ms if entries else 0

    return SrtData(
        entries=entries,
        source_path=str(path),
        total_duration_ms=total_duration,
    )


def _parse_block(block: str) -> Optional[SrtEntry]:
    """Parse a single SRT subtitle block into an SrtEntry."""
    lines = block.split("\n")
    if len(lines) < 2:
        return None

    # Line 1: sequence number
    try:
        index = int(lines[0].strip())
    except ValueError:
        return None

    # Line 2: timestamp range
    ts_match = _TIMESTAMP_RE.search(lines[1] if len(lines) > 1 else "")
    if not ts_match:
        return None

    h, m, s, ms = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3)), int(ts_match.group(4))
    timestamp_ms = h * 3600000 + m * 60000 + s * 1000 + ms

    entry = SrtEntry(index=index, timestamp_ms=timestamp_ms)

    # Remaining lines: metadata
    content = "\n".join(lines[2:])

    # Extract datetime
    dt_match = _DATETIME_RE.search(content)
    if dt_match:
        entry.datetime_str = dt_match.group(0)

    # Extract bracketed fields
    for key, value in _FIELD_RE.findall(content):
        value = value.strip()
        try:
            if key == "latitude":
                entry.latitude = float(value)
            elif key == "longitude":
                entry.longitude = float(value)
            elif key == "rel_alt":
                # DJI format: "[rel_alt: 7.900 abs_alt: 651.867]" — both values in one bracket
                tokens = value.split()
                entry.rel_alt = float(tokens[0])
                if "abs_alt:" in value:
                    abs_idx = tokens.index("abs_alt:")
                    entry.abs_alt = float(tokens[abs_idx + 1])
            elif key == "abs_alt":
                entry.abs_alt = float(value)
            elif key == "iso":
                entry.iso = int(value)
            elif key == "shutter":
                entry.shutter = value
            elif key == "fnum":
                entry.fnum = float(value)
            elif key == "ev":
                entry.ev = float(value)
            elif key == "focal_len":
                entry.focal_len = float(value)
            elif key == "color_md":
                entry.color_md = value
            elif key == "ct":
                entry.ct = int(value)
        except (ValueError, IndexError):
            pass

    return entry


def find_srt_for_video(video_path: str) -> Optional[str]:
    """Auto-detect an SRT file matching a video file by stem.

    Checks for both .SRT and .srt extensions in the same directory,
    then the parent directory (e.g. video in adjusted/ subfolder,
    SRT alongside the original in the parent).

    Args:
        video_path: Path to the video file

    Returns:
        Path to matching SRT file, or None
    """
    video = Path(video_path)
    dirs = [video.parent]
    if video.parent.parent != video.parent:
        dirs.append(video.parent.parent)

    for d in dirs:
        for ext in (".SRT", ".srt"):
            srt_path = d / (video.stem + ext)
            if srt_path.exists():
                return str(srt_path)
    return None

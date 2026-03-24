"""
Sharpest Frame Extractor Module

Extracts the sharpest frame from each time-interval chunk of a video
using FFmpeg's blurdetect filter.  Instead of extracting all frames and
then discarding blurry ones, this analyses blur on every frame *first*
and only extracts the winners.

Scene-aware chunking: a ``select='gte(scene,0)'`` filter in the same
pass populates ``lavfi.scene_score`` for free.  When a score exceeds
``scene_threshold``, the interval chunk is split so both sides of the
transition get a representative sharp frame.

Algorithm (adapted from github.com/Kotohibi/Extract_sharpest_frame):
  1. Run ffmpeg blurdetect + scene scoring → per-frame metadata
  2. Divide frames into interval chunks; split at scene boundaries
  3. Pick lowest-blur frame per (sub-)chunk
  4. Extract only those frames with ffmpeg select filter
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .extractor import MANIFEST_FILENAME

# Hide console windows on Windows for subprocess calls
_SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# ── data classes ─────────────────────────────────────────────────────

@dataclass
class SharpestConfig:
    """Configuration for sharpest-frame extraction."""
    interval: float = 2.0            # seconds between selections
    scene_threshold: float = 0.3     # scene-change score to split chunks
    scale_width: int = 1920          # resolution for blur analysis
    block_size: int = 32             # blurdetect block dimensions
    quality: int = 95                # JPEG quality (1-100)
    output_format: str = "jpg"       # jpg or png
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclass
class SharpestResult:
    """Result of sharpest-frame extraction."""
    success: bool
    total_frames_analyzed: int = 0
    frames_extracted: int = 0
    output_dir: str = ""
    frame_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ── main class ───────────────────────────────────────────────────────

class SharpestExtractor:
    """Extract the sharpest frame per interval from a video."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    # ── public API ───────────────────────────────────────────────────

    def extract(
        self,
        video_path: str,
        output_dir: str,
        config: Optional[SharpestConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
    ) -> SharpestResult:
        """Full pipeline: probe fps → blurdetect → parse → extract.

        Args:
            video_path:  Path to input video file.
            output_dir:  Directory for extracted frames.
            config:      Extraction settings (defaults to SharpestConfig()).
            progress_callback: Called with ``(current, total, message)``.

        Returns:
            SharpestResult with extraction details.
        """
        if config is None:
            config = SharpestConfig()

        video = Path(video_path)
        out = Path(output_dir)

        if not video.exists():
            return SharpestResult(success=False, error=f"Video not found: {video}")

        out.mkdir(parents=True, exist_ok=True)

        def _progress(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        # Step 1 — probe FPS and duration
        _progress(0, 100, "Probing video...")
        fps = self._probe_fps(str(video))
        if fps <= 0:
            return SharpestResult(success=False, error="Could not determine video FPS")
        duration_sec = self._probe_duration(str(video))

        chunk_size = max(1, round(fps * config.interval))

        # Step 2 — blurdetect (0-85% of progress)
        _progress(0, 100, f"Running blurdetect (chunk={chunk_size} frames)...")
        metadata_path = None
        try:
            metadata_path = Path(tempfile.mktemp(suffix="_blurdetect.txt",
                                                  dir=str(out)))
            ok, err = self._run_blurdetect(str(video), metadata_path, config,
                                            progress_callback=progress_callback,
                                            duration_sec=duration_sec,
                                            cancel_check=cancel_check)
            if not ok:
                return SharpestResult(success=False, error=f"blurdetect failed: {err}")

            # Step 3 — parse best frames
            _progress(85, 100, "Selecting sharpest frames...")
            best_frames = self._parse_best_frames(
                metadata_path, chunk_size, config.scene_threshold,
            )
            total_analyzed = self._count_analyzed_frames(metadata_path)

            if not best_frames:
                return SharpestResult(
                    success=False,
                    total_frames_analyzed=total_analyzed,
                    error="No frames selected (blurdetect returned no data)",
                )

            if cancel_check and cancel_check():
                return SharpestResult(success=False, error="Cancelled")

            # Step 4 — extract (85-100% of progress)
            stem = video.stem + "_" if prefix_source else ""
            _progress(85, 100, f"Extracting {len(best_frames)} sharpest frames...")
            frame_paths = self._extract_frames(
                str(video), best_frames, str(out), config,
                stem=stem,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            # Write extraction manifest for geotagging
            start = config.start_sec or 0.0
            self._write_manifest(out, frame_paths, video, config, best_frames, fps, start)

            return SharpestResult(
                success=True,
                total_frames_analyzed=total_analyzed,
                frames_extracted=len(frame_paths),
                output_dir=str(out),
                frame_paths=frame_paths,
            )

        finally:
            if metadata_path and metadata_path.exists():
                metadata_path.unlink(missing_ok=True)

    # ── internals ────────────────────────────────────────────────────

    def _probe_fps(self, video_path: str) -> float:
        """Get video FPS via ffprobe."""
        cmd = [
            self.ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return 0.0
        try:
            data = json.loads(result.stdout)
            rate = data["streams"][0]["r_frame_rate"]
            num, den = rate.split("/")
            return float(num) / float(den)
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            return 0.0

    def _probe_duration(self, video_path: str) -> float:
        """Get video duration in seconds via ffprobe."""
        cmd = [
            self.ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            return 0.0
        try:
            data = json.loads(result.stdout)
            # Try stream duration first, fall back to format duration
            dur = data.get("streams", [{}])[0].get("duration")
            if not dur:
                dur = data.get("format", {}).get("duration")
            return float(dur) if dur else 0.0
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.0

    def _run_blurdetect(
        self,
        video_path: str,
        metadata_path: Path,
        config: SharpestConfig,
        progress_callback: Optional[Callable] = None,
        duration_sec: float = 0.0,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """Run ffmpeg blurdetect + scene scoring, write per-frame metadata."""
        # select='gte(scene,0)' passes every frame but populates
        # lavfi.scene_score in metadata at zero extra cost.
        vf = (
            f"scale={config.scale_width}:-1,"
            f"select='gte(scene\\,0)',"
            f"blurdetect=block_width={config.block_size}"
            f":block_height={config.block_size},"
            f"metadata=print:file={metadata_path.name}"
        )
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-y",
            "-progress", "pipe:1", "-nostats",
        ]
        if config.start_sec is not None:
            cmd.extend(["-ss", str(config.start_sec)])
        cmd.extend(["-i", video_path])
        if config.end_sec is not None:
            cmd.extend(["-to", str(config.end_sec)])
        cmd.extend(["-vf", vf, "-an", "-f", "null", "-"])

        # Effective duration for percentage calculation
        eff_duration = duration_sec
        if config.start_sec or config.end_sec:
            start = config.start_sec or 0.0
            end = config.end_sec or duration_sec
            eff_duration = max(0.0, end - start)

        # Format duration as MM:SS for display
        def _fmt_time(secs):
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        total_display = _fmt_time(eff_duration) if eff_duration > 0 else ""

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(metadata_path.parent), **_SUBPROCESS_FLAGS,
        )

        # Parse ffmpeg -progress output for time/speed updates
        last_pct = -1
        current_time_us = 0
        current_speed = ""
        for line in proc.stdout:
            if cancel_check and cancel_check():
                proc.terminate()
                proc.wait()
                return False, "cancelled"
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    current_time_us = int(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("speed="):
                current_speed = line.split("=", 1)[1].strip()
            elif line.startswith("progress="):
                # Each progress block ends with "progress=continue" or "progress=end"
                # — emit a callback once per block with accumulated values
                if not progress_callback:
                    continue
                elapsed_sec = current_time_us / 1_000_000
                elapsed_display = _fmt_time(elapsed_sec)

                if eff_duration > 0:
                    # Blurdetect occupies 0-85% of overall progress
                    raw_pct = min(elapsed_sec / eff_duration, 1.0)
                    pct = int(raw_pct * 85)
                    if pct == last_pct:
                        continue
                    last_pct = pct
                    speed_str = f" [{current_speed}]" if current_speed and current_speed != "N/A" else ""
                    msg = f"Analyzing: {elapsed_display} / {total_display} ({int(raw_pct * 100)}%){speed_str}"
                    progress_callback(pct, 100, msg)
                else:
                    msg = f"Analyzing: {elapsed_display}"
                    progress_callback(0, 0, msg)

        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            return False, err[-500:] if err else "unknown error"
        return True, ""

    def _parse_best_frames(
        self,
        metadata_path: Path,
        chunk_size: int,
        scene_threshold: float = 0.3,
    ) -> List[int]:
        """Parse blurdetect metadata with scene-aware chunk splitting.

        1. Collect ``(frame, blur, scene_score)`` from metadata.
        2. Divide into interval-based chunks of *chunk_size*.
        3. If a frame inside a chunk has ``scene_score >= threshold``,
           split the chunk at that boundary.
        4. Pick the lowest-blur frame from each (sub-)chunk.
        """
        pat_frame = re.compile(r"frame:(\d+)")
        pat_blur = re.compile(r"lavfi\.blur=([0-9.]+)")
        pat_scene = re.compile(r"lavfi\.scene_score=([0-9.]+)")

        # Collect per-frame data: (frame_num, blur, scene_score)
        frame_data: List[Tuple[int, float, float]] = []
        current_frame = -1
        current_blur = -1.0
        current_scene = 0.0

        try:
            lines = metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except FileNotFoundError:
            return []

        for line in lines:
            line = line.strip()
            m = pat_frame.search(line)
            if m:
                # Emit previous frame if complete
                if current_frame >= 0 and current_blur >= 0:
                    frame_data.append((current_frame, current_blur, current_scene))
                current_frame = int(m.group(1))
                current_blur = -1.0
                current_scene = 0.0
                continue
            m = pat_blur.search(line)
            if m and current_frame >= 0:
                try:
                    current_blur = float(m.group(1))
                except ValueError:
                    pass
                continue
            m = pat_scene.search(line)
            if m and current_frame >= 0:
                try:
                    current_scene = float(m.group(1))
                except ValueError:
                    pass

        # Don't forget the last frame
        if current_frame >= 0 and current_blur >= 0:
            frame_data.append((current_frame, current_blur, current_scene))

        if not frame_data:
            return []

        # Build interval chunks, then split at scene boundaries
        best: List[int] = []
        for i in range(0, len(frame_data), chunk_size):
            chunk = frame_data[i : i + chunk_size]
            sub_chunks = self._split_at_scenes(chunk, scene_threshold)
            for sc in sub_chunks:
                winner = min(sc, key=lambda x: x[1])
                best.append(winner[0])

        return best

    @staticmethod
    def _split_at_scenes(
        chunk: List[Tuple[int, float, float]],
        threshold: float,
    ) -> List[List[Tuple[int, float, float]]]:
        """Split a chunk into sub-chunks at scene boundaries."""
        if not chunk:
            return []
        sub_chunks: List[List[Tuple[int, float, float]]] = []
        current: List[Tuple[int, float, float]] = []
        for entry in chunk:
            # Scene change → flush current sub-chunk, start new one
            if entry[2] >= threshold and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    def _count_analyzed_frames(self, metadata_path: Path) -> int:
        """Count how many frames were analyzed (quick scan)."""
        try:
            text = metadata_path.read_text(encoding="utf-8", errors="ignore")
            return len(re.findall(r"lavfi\.blur=", text))
        except FileNotFoundError:
            return 0

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        frame_paths: List[str],
        video_path: Path,
        config: SharpestConfig,
        frame_numbers: List[int],
        fps: float,
        start_sec: float,
    ):
        """Write extraction manifest mapping frames to source video timestamps.

        Uses exact frame numbers and FPS for precise timestamp computation,
        unlike FrameExtractor which estimates from interval index.
        """
        manifest = {
            "video": str(video_path.absolute()),
            "video_stem": video_path.stem,
            "extraction_mode": "sharpest",
            "interval": config.interval,
            "start_sec": start_sec,
            "end_sec": config.end_sec,
            "fps": fps,
            "frames": [],
        }

        for i, path in enumerate(frame_paths):
            filename = Path(path).name
            # frame_numbers[i] is the source video frame index
            frame_num = frame_numbers[i] if i < len(frame_numbers) else 0
            time_sec = round(start_sec + frame_num / fps, 3)
            manifest["frames"].append({
                "filename": filename,
                "index": i + 1,
                "source_frame": frame_num,
                "time_sec": time_sec,
            })

        manifest_path = output_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _extract_frames(
        self,
        video_path: str,
        frame_numbers: List[int],
        output_dir: str,
        config: SharpestConfig,
        stem: str = "",
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        """Extract only the selected frames."""
        out = Path(output_dir)
        ext = config.output_format
        pattern = str(out / f"{stem}%05d.{ext}")
        total_frames = len(frame_numbers)

        select_expr = "+".join(f"eq(n\\,{f})" for f in frame_numbers)
        vf = f"select='{select_expr}'"

        # On Windows the select expression can exceed the command-line
        # length limit (~8 KB for cmd.exe / 32 KB for CreateProcess).
        # Fall back to a temporary filter_script file when that happens.
        filter_script_path = None
        use_filter_script = len(vf) > 7000

        if use_filter_script:
            filter_script_path = Path(tempfile.mktemp(
                suffix="_select.txt", dir=str(out)))
            filter_script_path.write_text(
                f"select='{select_expr}'", encoding="utf-8")

        cmd = [
            self.ffmpeg_path, "-hide_banner", "-y",
            "-progress", "pipe:1", "-nostats",
            "-i", video_path,
        ]
        if use_filter_script:
            cmd.extend(["-filter_script:v", str(filter_script_path)])
        else:
            cmd.extend(["-vf", vf])
        cmd.append("-vsync")
        cmd.append("0")

        if ext in ("jpg", "jpeg"):
            qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
            cmd.extend(["-qscale:v", str(qscale)])
        elif ext == "png":
            cmd.extend(["-compression_level", "6"])

        cmd.append(pattern)

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, **_SUBPROCESS_FLAGS)

            last_pct = -1
            current_frame = 0
            for line in proc.stdout:
                if cancel_check and cancel_check():
                    proc.terminate()
                    proc.wait()
                    return []
                line = line.strip()
                if line.startswith("frame="):
                    try:
                        current_frame = int(line.split("=", 1)[1])
                    except ValueError:
                        pass
                elif line.startswith("progress=") and progress_callback:
                    # Extraction occupies 85-100% of overall progress
                    if total_frames > 0:
                        raw_pct = min(current_frame / total_frames, 1.0)
                        pct = 85 + int(raw_pct * 15)
                        if pct != last_pct:
                            last_pct = pct
                            progress_callback(
                                pct, 100,
                                f"Extracting: {current_frame}/{total_frames} frames")

            proc.wait()
            if proc.returncode != 0:
                return []
        finally:
            if filter_script_path and filter_script_path.exists():
                filter_script_path.unlink()

        # Collect extracted files
        return sorted(str(p) for p in out.glob(f"{stem}*.{ext}"))

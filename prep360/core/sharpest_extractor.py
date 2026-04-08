"""
Sharpest Frame Extractor Module

Extracts the sharpest frame from each time-interval chunk of a video
using Tenengrad (Sobel gradient energy) scoring via OpenCV.

Two tiers:
  Basic — score every frame, pick sharpest per interval window.
  Best  — same scoring + histogram-based scene-change detection;
          interval windows are split at scene boundaries so both
          sides of a cut get a sharp representative.

Algorithm:
  1. Read every frame with OpenCV, score sharpness via Tenengrad
  2. (Best only) Detect scene changes via HSV histogram correlation
  3. Divide frames into interval chunks; split at scene boundaries
  4. Pick highest-sharpness frame per (sub-)chunk
  5. Extract winners (Basic: OpenCV seek+write; Best: ffmpeg select)
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .extractor import MANIFEST_FILENAME

# Hide console windows on Windows for subprocess calls
_SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

# Backward-compat mapping for old tier names
_TIER_ALIASES = {"fast": "basic", "balanced": "basic", "quality": "best"}


# ── data classes ─────────────────────────────────────────────────────

@dataclass
class SharpestConfig:
    """Configuration for sharpest-frame extraction."""
    interval: float = 2.0            # seconds between selections
    scene_threshold: float = 0.3     # scene-change sensitivity (0-1, higher = more sensitive)
    scale_width: int = 1920          # resolution for sharpness analysis
    quality: int = 95                # JPEG quality (1-100)
    output_format: str = "jpg"       # jpg or png
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    scoring_method: str = "laplacian"  # "laplacian" or "tenengrad"
    scene_detection: bool = True       # split windows at scene cuts


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
        log: Optional[Callable[[str], None]] = None,
    ) -> SharpestResult:
        """Full pipeline: score frames → select winners → extract.

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

        # Backward compat: old configs may have tier instead of new fields
        if hasattr(config, 'tier') and not hasattr(config, 'scoring_method'):
            tier = _TIER_ALIASES.get(config.tier.lower(), config.tier.lower())
            config.scene_detection = (tier == "best")
            config.scoring_method = "laplacian"

        scene_aware = config.scene_detection
        use_ffmpeg_extract = config.scene_detection

        return self._extract_opencv(
            video_path=str(video), output_dir=str(out), config=config,
            scene_aware=scene_aware, use_ffmpeg_extract=use_ffmpeg_extract,
            progress_callback=progress_callback, cancel_check=cancel_check,
            prefix_source=prefix_source, log=log,
        )

    # ── scoring helpers ──────────────────────────────────────────────

    @staticmethod
    def _laplacian_sharpness(gray) -> float:
        """Laplacian variance — fast, ecosystem standard for 3DGS/NeRF."""
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _tenengrad_sharpness(gray) -> float:
        """Tenengrad focus measure: mean Sobel gradient energy.

        Better noise robustness than Laplacian due to Sobel's implicit
        Gaussian smoothing (Pertuz et al. 2013, PMC 2025).
        """
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float((gx * gx + gy * gy).mean())

    @staticmethod
    def _detect_scene_change(prev_bgr, curr_bgr, threshold: float) -> bool:
        """Detect scene change via HSV histogram correlation.

        Args:
            prev_bgr: Previous frame (BGR, any size).
            curr_bgr: Current frame (BGR, same size).
            threshold: Sensitivity 0-1 (higher = more sensitive).
                       Mapped to correlation threshold = 1.0 - threshold.

        Returns:
            True if a scene change is detected.
        """
        import cv2
        corr_threshold = 1.0 - threshold
        prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
        curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
        hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_prev, hist_prev)
        cv2.normalize(hist_curr, hist_curr)
        corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return corr < corr_threshold

    # ── internals ────────────────────────────────────────────────────

    def _extract_opencv(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        scene_aware: bool = False,
        use_ffmpeg_extract: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> SharpestResult:
        """Unified OpenCV scoring path for all tiers.

        Pass 1: Read every frame, score with Tenengrad, optionally detect
                 scene changes via histogram correlation.
        Pass 2: Extract winners — OpenCV seek+write (basic) or ffmpeg
                 select filter (best, frame-accurate).
        """
        import cv2
        import time

        def _log(msg):
            if log:
                log(msg)

        def _progress(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        out = Path(output_dir)
        video = Path(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return SharpestResult(success=False, error=f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return SharpestResult(success=False, error="Could not determine video FPS")

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        window_size = max(1, int(fps * config.interval))

        # Determine frame range from start_sec / end_sec
        start_frame = int((config.start_sec or 0.0) * fps)
        if config.end_sec is not None:
            end_frame = min(int(config.end_sec * fps), total_frame_count)
        else:
            end_frame = total_frame_count

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        _log(f"  Scoring: {config.scoring_method}, scene detection: {scene_aware}, analysis width: {config.scale_width}px")
        _log(f"  FPS: {fps:.2f}, window: {window_size} frames ({config.interval:.1f}s)")

        # ── Pass 1: score every frame ────────────────────────────────
        # Tuples: (frame_idx, sharpness, is_scene_boundary)
        scores: List[Tuple[int, float, bool]] = []
        frames_in_range = end_frame - start_frame
        t_score = time.perf_counter()
        frame_idx = start_frame
        prev_small_bgr = None  # for scene detection

        while frame_idx < end_frame:
            if cancel_check and cancel_check():
                cap.release()
                return SharpestResult(success=False, error="Cancelled")

            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for consistent scoring
            h, w = frame.shape[:2]
            if config.scale_width and w > config.scale_width:
                scale = config.scale_width / w
                new_h = int(h * scale)
                small = cv2.resize(frame, (config.scale_width, new_h))
            else:
                small = frame

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if config.scoring_method == "tenengrad":
                sharpness = self._tenengrad_sharpness(gray)
            else:
                sharpness = self._laplacian_sharpness(gray)

            # Scene detection (Best tier only)
            is_scene = False
            if scene_aware and prev_small_bgr is not None:
                is_scene = self._detect_scene_change(
                    prev_small_bgr, small, config.scene_threshold)

            scores.append((frame_idx, sharpness, is_scene))

            if scene_aware:
                prev_small_bgr = small

            # Progress: scoring occupies 0-85%
            if frames_in_range > 0 and (frame_idx - start_frame) % max(1, frames_in_range // 100) == 0:
                pct = int((frame_idx - start_frame) / frames_in_range * 85)
                _progress(pct, 100, f"Scoring: frame {frame_idx - start_frame}/{frames_in_range}")

            frame_idx += 1

        cap.release()

        if not scores:
            return SharpestResult(success=False, error="No frames scored")

        sharp_vals = [s for _, s, _ in scores]
        elapsed = time.perf_counter() - t_score
        _log(f"  {config.scoring_method.title()} scoring: {len(scores)} frames ({elapsed:.1f}s)")
        _log(f"  Sharpness: min={min(sharp_vals):.1f} max={max(sharp_vals):.1f} "
             f"mean={sum(sharp_vals)/len(sharp_vals):.1f}")

        if scene_aware:
            scene_count = sum(1 for _, _, sc in scores if sc)
            _log(f"  Scene changes detected: {scene_count}")

        # ── Select winners ───────────────────────────────────────────
        best_frames: List[int] = []
        for i in range(0, len(scores), window_size):
            window = scores[i : i + window_size]
            if scene_aware:
                sub_chunks = self._split_at_scenes(window)
            else:
                sub_chunks = [window]
            for sc in sub_chunks:
                winner = max(sc, key=lambda x: x[1])
                best_frames.append(winner[0])

        total_windows = len(range(0, len(scores), window_size))
        if scene_aware:
            scene_count = sum(1 for _, _, sc in scores if sc)
            if scene_count > 0:
                _log(f"  Chunks: {total_windows} intervals → {len(best_frames)} winners "
                     f"({scene_count} scene splits)")
            else:
                _log(f"  Chunks: {total_windows} intervals → {len(best_frames)} winners")
        else:
            _log(f"  Windows: {len(best_frames)} winners from {total_windows} windows "
                 f"(window_size={window_size} frames)")

        if not best_frames:
            return SharpestResult(
                success=False, total_frames_analyzed=len(scores),
                error="No frames selected",
            )

        if cancel_check and cancel_check():
            return SharpestResult(success=False, error="Cancelled")

        # ── Pass 2: extract winners ──────────────────────────────────
        stem = video.stem + "_" if prefix_source else ""

        if use_ffmpeg_extract:
            # Frame-accurate extraction via ffmpeg select filter
            _progress(85, 100, f"Extracting {len(best_frames)} sharpest frames...")
            frame_paths = self._extract_frames(
                video_path, best_frames, output_dir, config,
                stem=stem,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        else:
            # Fast extraction via OpenCV seek+write
            frame_paths = self._extract_opencv_pass2(
                video_path, best_frames, out, config,
                stem=stem,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

        if frame_paths is None:
            return SharpestResult(success=False, error="Cancelled")

        # Write manifest
        start = config.start_sec or 0.0
        self._write_manifest(out, frame_paths, video, config, best_frames, fps, start)

        return SharpestResult(
            success=True,
            total_frames_analyzed=len(scores),
            frames_extracted=len(frame_paths),
            output_dir=str(out),
            frame_paths=frame_paths,
        )

    @staticmethod
    def _split_at_scenes(
        chunk: List[Tuple[int, float, bool]],
    ) -> List[List[Tuple[int, float, bool]]]:
        """Split a chunk into sub-chunks at scene boundaries."""
        if not chunk:
            return []
        sub_chunks: List[List[Tuple[int, float, bool]]] = []
        current: List[Tuple[int, float, bool]] = []
        for entry in chunk:
            if entry[2] and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    def _extract_opencv_pass2(
        self,
        video_path: str,
        frame_numbers: List[int],
        output_dir: Path,
        config: SharpestConfig,
        stem: str = "",
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[List[str]]:
        """Pass 2 for Basic tier: seek to winners and write via OpenCV."""
        import cv2

        ext = config.output_format
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_paths: List[str] = []
        total_to_extract = len(frame_numbers)
        for idx, fnum in enumerate(frame_numbers):
            if cancel_check and cancel_check():
                cap.release()
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if not ret:
                continue

            filename = f"{stem}{idx + 1:05d}.{ext}"
            filepath = output_dir / filename

            if ext in ("jpg", "jpeg"):
                cv2.imwrite(str(filepath), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, config.quality])
            else:
                cv2.imwrite(str(filepath), frame)

            frame_paths.append(str(filepath))

            if progress_callback:
                pct = 85 + int((idx + 1) / total_to_extract * 15)
                progress_callback(pct, 100, f"Extracting: {idx + 1}/{total_to_extract} frames")

        cap.release()
        return frame_paths

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
        """Write extraction manifest mapping frames to source video timestamps."""
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
        """Frame-accurate extraction via ffmpeg select filter (Best tier)."""
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

"""
Sharpest Frame Extractor Module

Extracts the sharpest frame from each time-interval chunk of a video.
Supports Laplacian variance and Tenengrad (Sobel gradient energy) scoring.

Architecture:
  Single-pass streaming: decode each frame, score it, buffer the current
  window's best, and write the winner when the window closes. Optionally
  splits windows at scene boundaries (HSV histogram correlation) so both
  sides of a cut get a sharp representative.

  An optional GPU path (cv2.cudacodec + cv2.cuda) uses NVDEC hardware
  decode and GPU-accelerated scoring. Falls back to CPU automatically
  when CUDA OpenCV is unavailable or NVDEC cannot handle the codec/resolution.
"""

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .extractor import MANIFEST_FILENAME

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
    gpu_accelerated: bool = False


# ── main class ───────────────────────────────────────────────────────

class SharpestExtractor:
    """Extract the sharpest frame per interval from a video."""

    _gpu_ok: Optional[bool] = None  # class-level cache for GPU availability

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

        # Refuse non-empty output directory to prevent cancellation
        # cleanup from deleting pre-existing files
        existing_files = [p for p in out.iterdir() if p.is_file()]
        if existing_files:
            return SharpestResult(
                success=False,
                error=f"Output directory is not empty ({len(existing_files)} files). "
                      f"Delete existing output before re-extracting.",
            )

        # Backward compat: old configs may have tier instead of new fields
        if hasattr(config, 'tier') and not hasattr(config, 'scoring_method'):
            tier = _TIER_ALIASES.get(config.tier.lower(), config.tier.lower())
            config.scene_detection = (tier == "best")
            config.scoring_method = "laplacian"

        scene_aware = config.scene_detection

        # Try GPU path first
        if self._gpu_available():
            gpu_result = self._extract_gpu_single_pass(
                video_path=str(video), output_dir=str(out), config=config,
                scene_aware=scene_aware,
                progress_callback=progress_callback, cancel_check=cancel_check,
                prefix_source=prefix_source, log=log,
            )
            if gpu_result is not None:
                return gpu_result
            # gpu_result is None → GPU init or runtime failure → CPU fallback
            if log:
                log("  Falling back to CPU extraction")

        # CPU single-pass fallback
        return self._extract_opencv(
            video_path=str(video), output_dir=str(out), config=config,
            scene_aware=scene_aware,
            progress_callback=progress_callback, cancel_check=cancel_check,
            prefix_source=prefix_source, log=log,
        )

    # ── GPU availability ─────────────────────────────────────────────

    def _gpu_available(self) -> bool:
        """Check if CUDA OpenCV with cudacodec and required ops is available."""
        if SharpestExtractor._gpu_ok is None:
            try:
                import cv2
                SharpestExtractor._gpu_ok = (
                    cv2.cuda.getCudaEnabledDeviceCount() > 0
                    and callable(getattr(cv2.cudacodec, 'createVideoReader', None))
                    and callable(getattr(cv2.cudacodec, 'VideoReaderInitParams', None))
                    and callable(getattr(cv2.cuda, 'createSobelFilter', None))
                    and callable(getattr(cv2.cuda, 'createLaplacianFilter', None))
                    and callable(getattr(cv2.cuda, 'resize', None))
                    and callable(getattr(cv2.cuda, 'cvtColor', None))
                    and callable(getattr(cv2.cuda, 'sum', None))
                    and callable(getattr(cv2.cuda, 'multiply', None))
                    and callable(getattr(cv2.cuda, 'add', None))
                    and hasattr(cv2.cuda, 'GpuMat')
                )
            except Exception:
                SharpestExtractor._gpu_ok = False
        return SharpestExtractor._gpu_ok

    @staticmethod
    def _frame_range_for_seconds(
        fps: float,
        total_frame_count: int,
        start_sec: Optional[float],
        end_sec: Optional[float],
    ) -> Tuple[float, int, int]:
        """Convert a user time range to an exclusive frame range."""
        range_start_sec = max(0.0, float(start_sec or 0.0))
        start_frame = max(0, int(math.ceil(range_start_sec * fps - 1e-9)))
        if end_sec is None:
            end_frame = total_frame_count
        else:
            range_end_sec = max(range_start_sec, float(end_sec))
            end_frame = min(
                total_frame_count,
                int(math.ceil(range_end_sec * fps - 1e-9)),
            )
        return range_start_sec, start_frame, end_frame

    @staticmethod
    def _time_window_index(
        frame_idx: int,
        fps: float,
        range_start_sec: float,
        interval_sec: float,
    ) -> int:
        """Return the user-time interval window containing frame_idx."""
        interval = max(float(interval_sec), 1e-9)
        frame_time = frame_idx / fps
        elapsed = max(0.0, frame_time - range_start_sec)
        return int(math.floor((elapsed + 1e-9) / interval))

    def _media_summary_for_gpu_failure(self, video_path: str) -> str:
        """Return a compact video summary for explaining GPU decode fallback."""
        try:
            try:
                from prep360.core.subprocess_utils import subprocess_kwargs_for_binary
                _env_kw = subprocess_kwargs_for_binary(self.ffprobe_path)
            except ImportError:
                _env_kw = {}
            proc = subprocess.run(
                [
                    self.ffprobe_path,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries",
                    "stream=codec_name,codec_long_name,profile,codec_tag_string,"
                    "pix_fmt,width,height,avg_frame_rate,r_frame_rate",
                    "-of", "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                **_env_kw,
            )
            data = json.loads(proc.stdout)
            streams = data.get("streams") or []
            if streams:
                stream = streams[0]
                codec = stream.get("codec_name") or "unknown codec"
                profile = stream.get("profile")
                tag = stream.get("codec_tag_string")
                pix_fmt = stream.get("pix_fmt")
                width = stream.get("width")
                height = stream.get("height")
                rate = (
                    stream.get("avg_frame_rate")
                    or stream.get("r_frame_rate")
                    or "unknown fps"
                )
                codec_bits = []
                if profile and profile not in ("unknown", "N/A"):
                    codec_bits.append(str(profile))
                codec_bits.append(str(codec))
                if tag and tag not in ("unknown", "N/A"):
                    codec_bits.append(str(tag))
                parts = [" / ".join(codec_bits)]
                if pix_fmt:
                    parts.append(str(pix_fmt))
                if width and height:
                    parts.append(f"{width}x{height}")
                if rate and rate != "0/0":
                    parts.append(f"{rate} fps")
                return ", ".join(parts)
        except Exception:
            pass

        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return ""
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc = "".join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip("\x00")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            parts = []
            if fourcc:
                parts.append(fourcc)
            if width and height:
                parts.append(f"{width}x{height}")
            if fps:
                parts.append(f"{fps:.3f} fps")
            return ", ".join(parts)
        except Exception:
            return ""

    @staticmethod
    def _nvdec_failure_hint(media_summary: str) -> str:
        """Return a concise hint for common non-NVDEC media formats."""
        summary = media_summary.lower()
        if any(token in summary for token in ("dnxhd", "dnxhr", "vc3", "avdh")):
            return (
                "DNxHD/DNxHR is not supported by NVDEC; CPU fallback is expected. "
                "Use HEVC/H.265 or H.264 for GPU extraction."
            )
        if "prores" in summary or "apch" in summary or "apcn" in summary:
            return (
                "ProRes is not supported by NVDEC; CPU fallback is expected. "
                "Use HEVC/H.265 or H.264 for GPU extraction."
            )
        if media_summary:
            return (
                "NVDEC may not support this codec, profile, chroma format, "
                "container, or resolution on this GPU."
            )
        return "NVDEC could not open this stream; CPU fallback is expected."

    def _gpu_reader_failure_messages(
        self,
        video_path: str,
        stream_label: str = "",
        exc: Optional[BaseException] = None,
    ) -> List[str]:
        """Build user-facing log lines for cudacodec reader failures."""
        prefix = f"{stream_label} " if stream_label else ""
        messages = [f"  GPU: {prefix}cudacodec reader failed"]
        summary = self._media_summary_for_gpu_failure(video_path)
        if summary:
            messages.append(f"       media: {summary}")
        messages.append(f"       note: {self._nvdec_failure_hint(summary)}")
        if exc is not None:
            detail = str(exc).strip().splitlines()
            if detail:
                messages.append(f"       OpenCV: {detail[0]}")
        return messages

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

    @staticmethod
    def _split_at_scenes(
        chunk: List[Tuple[int, float, bool]],
    ) -> List[List[Tuple[int, float, bool]]]:
        """Split a chunk into sub-chunks at scene boundaries.

        Kept as a helper for sharpness_benchmark.py. Production extraction
        uses inline streaming logic with equivalent behavior.
        """
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

    # ── GPU extraction ────────────────────────────────────────────────

    def _extract_gpu_single_pass(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        scene_aware: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> Optional[SharpestResult]:
        """GPU-accelerated single-pass extraction using NVDEC + CUDA scoring.

        Returns SharpestResult on success or cancellation.
        Returns None on GPU init/runtime failure (caller falls back to CPU).
        """
        import time

        import cv2
        import numpy as np

        def _log(msg):
            if log:
                log(msg)

        def _progress(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        out = Path(output_dir)
        video = Path(video_path)

        # Get video metadata from CPU VideoCapture (reliable)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            return None

        approx_window_frames = max(1, int(round(fps * config.interval)))
        range_start_sec, start_frame, end_frame = self._frame_range_for_seconds(
            fps, total_frame_count, config.start_sec, config.end_sec)
        frames_in_range = end_frame - start_frame
        if frames_in_range <= 0:
            return SharpestResult(success=False, error="Invalid time range")

        # Open cudacodec reader with firstFrameIdx for seeking
        params = cv2.cudacodec.VideoReaderInitParams()
        if start_frame > 0:
            params.firstFrameIdx = start_frame
        try:
            reader = cv2.cudacodec.createVideoReader(video_path, params=params)
        except (cv2.error, TypeError) as e:
            for msg in self._gpu_reader_failure_messages(video_path, exc=e):
                _log(msg)
            return None

        # Everything from here can fail with cv2.error — wrap for CPU fallback
        gpu_written_paths: List[str] = []

        try:
            # Read first frame for format detection — this IS start_frame
            ret, first_gpu = reader.nextFrame()
            if not ret:
                _log("  GPU: could not read first frame")
                return None

            # Detect frame format
            channels = first_gpu.channels()
            gpu_type = first_gpu.type()
            depth = gpu_type & 7  # CV_8U=0, CV_16U=2, CV_32F=5
            is_16bit = (depth == 2)
            is_8bit = (depth == 0)

            if not (is_8bit or is_16bit):
                _log(f"  GPU: unsupported frame depth (type={gpu_type}, depth={depth})")
                return None

            if channels == 4:
                gray_code = cv2.COLOR_BGRA2GRAY
                bgr_code = cv2.COLOR_BGRA2BGR
            elif channels == 3:
                gray_code = cv2.COLOR_BGR2GRAY
                bgr_code = None
            else:
                _log(f"  GPU: unsupported frame format ({channels} channels, type={gpu_type})")
                return None

            _log(f"  GPU: NVDEC decode, {channels}ch {'uint16' if is_16bit else 'uint8'}, "
                 f"scoring: {config.scoring_method}, scene detection: {scene_aware}")
            _log(f"  FPS: {fps:.2f}, window: {config.interval:.1f}s "
                 f"(~{approx_window_frames} frames)")

            # Pre-create GPU filters
            if config.scoring_method == "tenengrad":
                sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)
            else:
                lap_filter = cv2.cuda.createLaplacianFilter(cv2.CV_32FC1, cv2.CV_32FC1, ksize=1)

            # ── Helper functions ─────────────────────────────────────

            def _prepare_gray(gpu_frm):
                w, h = gpu_frm.size()
                if config.scale_width and w > config.scale_width:
                    gpu_small = cv2.cuda.resize(
                        gpu_frm, (config.scale_width, int(h * config.scale_width / w)))
                else:
                    gpu_small = gpu_frm
                gpu_gray = cv2.cuda.cvtColor(gpu_small, gray_code)
                if is_16bit:
                    gpu_8u = cv2.cuda.GpuMat(gpu_gray.size(), cv2.CV_8UC1)
                    gpu_gray.convertTo(cv2.CV_8UC1, gpu_8u, alpha=1.0 / 256.0)
                else:
                    gpu_8u = gpu_gray
                return gpu_8u

            def _score_tenengrad(gpu_8u):
                gx = sobel_x.apply(gpu_8u)
                gy = sobel_y.apply(gpu_8u)
                gx2 = cv2.cuda.multiply(gx, gx)
                gy2 = cv2.cuda.multiply(gy, gy)
                energy = cv2.cuda.add(gx2, gy2)
                s = cv2.cuda.sum(energy)
                n = energy.size()[0] * energy.size()[1]
                return s[0] / n

            def _score_laplacian(gpu_8u):
                gpu_32f = cv2.cuda.GpuMat(gpu_8u.size(), cv2.CV_32FC1)
                gpu_8u.convertTo(cv2.CV_32FC1, gpu_32f)
                dst = lap_filter.apply(gpu_32f)
                sq = cv2.cuda.multiply(dst, dst)
                sum_sq = cv2.cuda.sum(sq)
                sum_val = cv2.cuda.sum(dst)
                n = dst.size()[0] * dst.size()[1]
                mean = sum_val[0] / n
                return sum_sq[0] / n - mean * mean

            score_fn = _score_tenengrad if config.scoring_method == "tenengrad" else _score_laplacian

            def _prepare_scene_bgr(gpu_frm):
                w, h = gpu_frm.size()
                scene_w = min(480, w)
                scene_h = int(h * scene_w / w)
                gpu_small = cv2.cuda.resize(gpu_frm, (scene_w, scene_h))
                gpu_bgr = cv2.cuda.cvtColor(gpu_small, cv2.COLOR_BGRA2BGR) if channels == 4 else gpu_small
                if is_16bit:
                    gpu_bgr_8u = cv2.cuda.GpuMat(gpu_bgr.size(), cv2.CV_8UC3)
                    gpu_bgr.convertTo(cv2.CV_8UC3, gpu_bgr_8u, alpha=1.0 / 256.0)
                    return gpu_bgr_8u.download()
                return gpu_bgr.download()

            def _save_winner_gpu(gpu_frm, filepath):
                frame = gpu_frm.download()
                if is_16bit:
                    frame = (frame >> 8).astype(np.uint8)
                if bgr_code is not None:
                    frame = cv2.cvtColor(frame, bgr_code)
                ext = config.output_format
                if ext in ("jpg", "jpeg"):
                    cv2.imwrite(str(filepath), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                else:
                    cv2.imwrite(str(filepath), frame)

            # ── Output setup ─────────────────────────────────────────

            ext = config.output_format
            stem = video.stem + "_" if prefix_source else ""
            winners_written = 0
            frame_paths: List[str] = []
            best_frame_numbers: List[int] = []

            def _write_gpu_winner(gpu_frm, frame_num):
                nonlocal winners_written
                winners_written += 1
                filename = f"{stem}{winners_written:05d}.{ext}"
                filepath = out / filename
                _save_winner_gpu(gpu_frm, filepath)
                path_str = str(filepath)
                frame_paths.append(path_str)
                best_frame_numbers.append(frame_num)
                gpu_written_paths.append(path_str)

            # ── Single-pass streaming loop ───────────────────────────

            t_start = time.perf_counter()
            total_scored = 0
            scene_count = 0
            all_sharpness: List[float] = []
            prev_scene_bgr = None

            best_score = -1.0
            best_gpu_frame = None
            best_frame_idx = -1
            current_window_idx: Optional[int] = None

            # Check cancel before first-frame processing
            if cancel_check and cancel_check():
                return SharpestResult(success=False, error="Cancelled")

            # Process first frame (already read for format detection)
            gpu_8u = _prepare_gray(first_gpu)
            sharpness = score_fn(gpu_8u)
            total_scored += 1
            all_sharpness.append(sharpness)

            if scene_aware:
                prev_scene_bgr = _prepare_scene_bgr(first_gpu)

            best_score = sharpness
            best_gpu_frame = first_gpu.clone()
            best_frame_idx = start_frame
            current_window_idx = self._time_window_index(
                start_frame, fps, range_start_sec, config.interval)

            # Continue with remaining frames
            for frame_idx in range(start_frame + 1, end_frame):
                if cancel_check and cancel_check():
                    for p in gpu_written_paths:
                        Path(p).unlink(missing_ok=True)
                    return SharpestResult(success=False, error="Cancelled")

                ret, gpu_frame = reader.nextFrame()
                if not ret:
                    break

                gpu_8u = _prepare_gray(gpu_frame)
                sharpness = score_fn(gpu_8u)
                total_scored += 1
                all_sharpness.append(sharpness)

                relative_idx = frame_idx - start_frame
                window_idx = self._time_window_index(
                    frame_idx, fps, range_start_sec, config.interval)

                is_scene = False
                if scene_aware:
                    scene_bgr = _prepare_scene_bgr(gpu_frame)
                    if prev_scene_bgr is not None:
                        is_scene = self._detect_scene_change(
                            prev_scene_bgr, scene_bgr, config.scene_threshold)
                    prev_scene_bgr = scene_bgr

                if current_window_idx is None:
                    current_window_idx = window_idx
                elif window_idx != current_window_idx:
                    if best_gpu_frame is not None:
                        _write_gpu_winner(best_gpu_frame, best_frame_idx)
                    best_score = -1.0
                    best_gpu_frame = None
                    best_frame_idx = -1
                    current_window_idx = window_idx

                if is_scene:
                    scene_count += 1
                    if best_gpu_frame is not None:
                        _write_gpu_winner(best_gpu_frame, best_frame_idx)
                        best_score = -1.0
                        best_gpu_frame = None
                        best_frame_idx = -1

                if sharpness > best_score:
                    best_score = sharpness
                    best_gpu_frame = gpu_frame.clone()
                    best_frame_idx = frame_idx

                if frames_in_range > 0:
                    pct = int(relative_idx / frames_in_range * 100)
                    if relative_idx % max(1, frames_in_range // 10) == 0:
                        _progress(pct, 100, f"Processing (GPU): {pct}% ({relative_idx}/{frames_in_range})")

            # Final flush (inside try — a cv2.error here still falls back)
            if best_gpu_frame is not None:
                _write_gpu_winner(best_gpu_frame, best_frame_idx)

        except cv2.error as e:
            _log(f"  GPU error during processing: {e}")
            for p in gpu_written_paths:
                Path(p).unlink(missing_ok=True)
            return None  # triggers CPU fallback

        elapsed = time.perf_counter() - t_start

        if total_scored == 0:
            return SharpestResult(success=False, error="No frames scored")

        # Log summary
        _log(f"  GPU {config.scoring_method.title()} scoring: {total_scored} frames ({elapsed:.1f}s)")
        _log(f"  Sharpness: min={min(all_sharpness):.1f} max={max(all_sharpness):.1f} "
             f"mean={sum(all_sharpness)/len(all_sharpness):.1f}")
        if scene_aware:
            _log(f"  Scene changes detected: {scene_count}")
        _log(f"  Winners: {winners_written} frames extracted")

        # Write manifest
        self._write_manifest(out, frame_paths, video, config, best_frame_numbers, fps)

        return SharpestResult(
            success=True,
            total_frames_analyzed=total_scored,
            frames_extracted=winners_written,
            output_dir=str(out),
            frame_paths=frame_paths,
            gpu_accelerated=True,
        )

    # ── CPU extraction ───────────────────────────────────────────────

    def _extract_opencv(
        self,
        video_path: str,
        output_dir: str,
        config: SharpestConfig,
        scene_aware: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        prefix_source: bool = True,
        log: Optional[Callable[[str], None]] = None,
    ) -> SharpestResult:
        """Single-pass streaming extraction: score + buffer + write inline.

        For each frame: score sharpness, optionally detect scene changes,
        buffer the current window's best frame, and write the winner when
        the window closes. No second decode pass.
        """
        import time

        import cv2

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
        approx_window_frames = max(1, int(round(fps * config.interval)))

        # Determine exclusive frame range from start_sec / end_sec.
        range_start_sec, start_frame, end_frame = self._frame_range_for_seconds(
            fps, total_frame_count, config.start_sec, config.end_sec)
        if end_frame <= start_frame:
            cap.release()
            return SharpestResult(success=False, error="Invalid time range")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_in_range = end_frame - start_frame
        _log(f"  Scoring: {config.scoring_method}, scene detection: {scene_aware}, analysis width: {config.scale_width}px")
        _log(f"  FPS: {fps:.2f}, window: {config.interval:.1f}s "
             f"(~{approx_window_frames} frames)")

        # Select scoring function
        if config.scoring_method == "tenengrad":
            score_fn = self._tenengrad_sharpness
        else:
            score_fn = self._laplacian_sharpness

        # Output setup
        ext = config.output_format
        stem = video.stem + "_" if prefix_source else ""
        winners_written = 0
        frame_paths: List[str] = []
        best_frame_numbers: List[int] = []
        written_paths: List[str] = []  # for cancel cleanup

        def _write_winner(frame_data, frame_num):
            nonlocal winners_written
            winners_written += 1
            filename = f"{stem}{winners_written:05d}.{ext}"
            filepath = out / filename
            if ext in ("jpg", "jpeg"):
                cv2.imwrite(str(filepath), frame_data,
                            [cv2.IMWRITE_JPEG_QUALITY, config.quality])
            else:
                cv2.imwrite(str(filepath), frame_data)
            path_str = str(filepath)
            frame_paths.append(path_str)
            best_frame_numbers.append(frame_num)
            written_paths.append(path_str)

        # ── Single-pass streaming loop ───────────────────────────────
        t_start = time.perf_counter()
        frame_idx = start_frame
        prev_small_bgr = None
        total_scored = 0
        scene_count = 0
        all_sharpness: List[float] = []

        # Sub-chunk state: best frame in current sub-chunk
        best_score = -1.0
        best_frame = None
        best_frame_idx = -1
        current_window_idx: Optional[int] = None

        while frame_idx < end_frame:
            if cancel_check and cancel_check():
                cap.release()
                for p in written_paths:
                    Path(p).unlink(missing_ok=True)
                return SharpestResult(success=False, error="Cancelled")

            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for scoring
            h, w = frame.shape[:2]
            if config.scale_width and w > config.scale_width:
                scale = config.scale_width / w
                small = cv2.resize(frame, (config.scale_width, int(h * scale)))
            else:
                small = frame

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharpness = score_fn(gray)
            total_scored += 1
            all_sharpness.append(sharpness)

            # Scene detection
            is_scene = False
            if scene_aware and prev_small_bgr is not None:
                is_scene = self._detect_scene_change(
                    prev_small_bgr, small, config.scene_threshold)
            if scene_aware:
                prev_small_bgr = small

            relative_idx = frame_idx - start_frame
            window_idx = self._time_window_index(
                frame_idx, fps, range_start_sec, config.interval)

            # Step 2: Time-window boundary — flush previous sub-chunk
            # BEFORE considering this frame (boundary frame belongs to
            # the new user-time interval).
            if current_window_idx is None:
                current_window_idx = window_idx
            elif window_idx != current_window_idx:
                if best_frame is not None:
                    _write_winner(best_frame, best_frame_idx)
                best_score = -1.0
                best_frame = None
                best_frame_idx = -1
                current_window_idx = window_idx

            # Step 3: Scene boundary — flush previous sub-chunk BEFORE
            # considering this frame (boundary frame belongs to new sub-chunk)
            if is_scene:
                scene_count += 1
                if best_frame is not None:
                    _write_winner(best_frame, best_frame_idx)
                    best_score = -1.0
                    best_frame = None
                    best_frame_idx = -1

            # Step 4: Consider current frame for (possibly new) sub-chunk
            if sharpness > best_score:
                best_score = sharpness
                best_frame = frame
                best_frame_idx = frame_idx

            # Progress (linear 0-100%)
            if frames_in_range > 0:
                pct = int(relative_idx / frames_in_range * 100)
                if relative_idx % max(1, frames_in_range // 10) == 0:
                    _progress(pct, 100, f"Processing: {pct}% ({relative_idx}/{frames_in_range})")

            frame_idx += 1

        # Final flush: partial window at EOF
        if best_frame is not None:
            _write_winner(best_frame, best_frame_idx)

        cap.release()
        elapsed = time.perf_counter() - t_start

        if total_scored == 0:
            return SharpestResult(success=False, error="No frames scored")

        # Log summary
        _log(f"  {config.scoring_method.title()} scoring: {total_scored} frames ({elapsed:.1f}s)")
        _log(f"  Sharpness: min={min(all_sharpness):.1f} max={max(all_sharpness):.1f} "
             f"mean={sum(all_sharpness)/len(all_sharpness):.1f}")
        if scene_aware:
            _log(f"  Scene changes detected: {scene_count}")
        _log(f"  Winners: {winners_written} frames extracted")

        # Write manifest
        self._write_manifest(out, frame_paths, video, config, best_frame_numbers, fps)

        return SharpestResult(
            success=True,
            total_frames_analyzed=total_scored,
            frames_extracted=winners_written,
            output_dir=str(out),
            frame_paths=frame_paths,
        )

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        frame_paths: List[str],
        video_path: Path,
        config: SharpestConfig,
        frame_numbers: List[int],
        fps: float,
    ):
        """Write extraction manifest mapping frames to source video timestamps."""
        manifest = {
            "video": str(video_path.absolute()),
            "video_stem": video_path.stem,
            "extraction_mode": "sharpest",
            "interval": config.interval,
            "start_sec": config.start_sec,
            "end_sec": config.end_sec,
            "fps": fps,
            "frames": [],
        }

        for i, path in enumerate(frame_paths):
            filename = Path(path).name
            frame_num = frame_numbers[i] if i < len(frame_numbers) else 0
            # Frame numbers are absolute — frame_num / fps is the true timestamp
            time_sec = round(frame_num / fps, 3)
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

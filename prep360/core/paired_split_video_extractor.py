"""
Pair-aware frame extraction from front/back split lens videos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from .sharpest_extractor import SharpestExtractor


@dataclass
class PairedSplitConfig:
    """Configuration for shared front/back frame extraction."""

    mode: str = "sharpest"  # "fixed" or "sharpest"
    scoring_method: str = "laplacian"  # "laplacian" or "tenengrad"
    scene_detection: bool = True
    interval_sec: float = 2.0
    quality: int = 95
    output_format: str = "jpg"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    scene_threshold: float = 0.3
    scale_width: int = 1920


@dataclass
class PairedSplitResult:
    """Result of pair-aware split-video extraction."""

    success: bool
    pair_count: int = 0
    output_dir: str = ""
    front_paths: list[str] = field(default_factory=list)
    back_paths: list[str] = field(default_factory=list)
    selected_times: list[float] = field(default_factory=list)
    source_front_frames: list[int] = field(default_factory=list)
    source_back_frames: list[int] = field(default_factory=list)
    error: Optional[str] = None
    gpu_accelerated: bool = False


class PairedSplitVideoExtractor:
    """Extract shared front/back frames from graded split videos."""

    def __init__(self):
        self._sharpest = SharpestExtractor()

    @staticmethod
    def _format_clock(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _open_video(path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Could not read FPS/frame count for {path}")
        return cap, fps, frame_count

    @staticmethod
    def _shared_stream_info(front_fps: float, back_fps: float,
                            front_count: int, back_count: int) -> tuple[float, int]:
        fps_delta = abs(front_fps - back_fps)
        if fps_delta > 0.05:
            raise RuntimeError(
                f"Front/back FPS mismatch too large: {front_fps:.3f} vs {back_fps:.3f}"
            )
        return min(front_fps, back_fps), min(front_count, back_count)

    @staticmethod
    def _ensure_empty_output(path: Path) -> None:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite non-empty paired extraction folder: {path}"
            )

    @staticmethod
    def _write_pair_image(path: Path, frame, config: PairedSplitConfig) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = config.output_format.lower()
        if ext == "jpg" or ext == "jpeg":
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(config.quality)])
        elif ext == "png":
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(path), frame)

    @staticmethod
    def _fixed_frame_indices(start_frame: int, end_frame: int, window_size: int) -> list[int]:
        return list(range(start_frame, end_frame, window_size))

    @staticmethod
    def _laplacian_sharpness(frame, analysis_width: int = 1920) -> float:
        """Laplacian variance on a BGR frame, with optional downscale."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width > analysis_width > 0:
            resized_height = max(1, int(round(height * (analysis_width / width))))
            gray = cv2.resize(
                gray,
                (analysis_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _tenengrad_sharpness(frame, analysis_width: int = 1920) -> float:
        """Tenengrad focus measure on a BGR frame, with optional downscale."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width > analysis_width > 0:
            resized_height = max(1, int(round(height * (analysis_width / width))))
            gray = cv2.resize(
                gray,
                (analysis_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float((gx * gx + gy * gy).mean())

    @staticmethod
    def _pair_sharpness_sort_key(entry: dict[str, float]) -> tuple[float, float, float]:
        min_sharp = min(entry["front_score"], entry["back_score"])
        avg_sharp = (entry["front_score"] + entry["back_score"]) * 0.5
        imbalance = abs(entry["front_score"] - entry["back_score"])
        return (min_sharp, avg_sharp, -imbalance)

    @staticmethod
    def _split_pair_chunk_at_scenes(
        chunk: list[dict],
    ) -> list[list[dict]]:
        """Split a chunk at frames where scene_change is True."""
        if not chunk:
            return []
        sub_chunks: list[list[dict]] = []
        current: list[dict] = []
        for entry in chunk:
            if entry.get("scene_change") and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    def _select_from_paired_entries(
        self,
        paired_entries: list[dict[str, float]],
        scene_aware: bool,
        log: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[int], list[float], list[float]]:
        def _log(msg):
            if log:
                log(msg)

        selected_indices: list[int] = []
        front_scores: list[float] = []
        back_scores: list[float] = []
        current_chunk: list[dict[str, float]] = []
        current_chunk_index: Optional[int] = None
        total_chunks = 0

        def _flush_chunk(chunk_entries: list[dict[str, float]]) -> None:
            nonlocal total_chunks
            if not chunk_entries:
                return
            total_chunks += 1
            sub_chunks = (
                self._split_pair_chunk_at_scenes(chunk_entries)
                if scene_aware else
                [chunk_entries]
            )
            for sub_chunk in sub_chunks:
                winner = max(sub_chunk, key=self._pair_sharpness_sort_key)
                selected_indices.append(int(winner["absolute_frame"]))
                front_scores.append(float(winner["front_score"]))
                back_scores.append(float(winner["back_score"]))

        for entry in paired_entries:
            chunk_index = int(entry["window_index"])
            if current_chunk_index is None:
                current_chunk_index = chunk_index
            if chunk_index != current_chunk_index:
                _flush_chunk(current_chunk)
                current_chunk = []
                current_chunk_index = chunk_index
            current_chunk.append(entry)

        _flush_chunk(current_chunk)
        _log(f"  Selection: {len(selected_indices)} winners from {total_chunks} chunks")
        return selected_indices, front_scores, back_scores

    def _select_sharpest_frame_indices(
        self,
        front_video: str,
        back_video: str,
        out_root: Path,
        start_frame: int,
        end_frame: int,
        range_start_sec: float,
        shared_fps: float,
        total_frames: int,
        config: PairedSplitConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        duration_sec: float = 0.0,
        _log: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[int], list[float], list[float]]:
        """Score both videos with Tenengrad and pick the best shared frame indices.

        Optionally detects scene changes via histogram correlation (Best tier).
        """

        if _log is None:
            _log = lambda _msg: None

        scene_aware = config.scene_detection
        score_fn = self._tenengrad_sharpness if config.scoring_method == "tenengrad" else self._laplacian_sharpness

        front_cap = back_cap = None
        try:
            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            total_to_score = max(1, end_frame - start_frame)
            total_duration = total_to_score / shared_fps if shared_fps > 0 else 0.0
            paired_entries: list[dict] = []
            prev_front_small = None
            prev_back_small = None

            method_label = config.scoring_method.title()

            for offset, frame_idx in enumerate(range(start_frame, end_frame), start=1):
                if cancel_check and cancel_check():
                    raise RuntimeError("Cancelled")

                ok_front, front_frame = front_cap.read()
                ok_back, back_frame = back_cap.read()
                if not ok_front or not ok_back:
                    break

                front_score = score_fn(front_frame, config.scale_width)
                back_score = score_fn(back_frame, config.scale_width)

                # Scene detection via histogram correlation on downscaled frames
                is_scene = False
                if scene_aware:
                    # Downscale for histogram (reuse analysis width)
                    h, w = front_frame.shape[:2]
                    if config.scale_width and w > config.scale_width:
                        scale = config.scale_width / w
                        new_h = int(h * scale)
                        front_small = cv2.resize(front_frame, (config.scale_width, new_h))
                        back_small = cv2.resize(back_frame, (config.scale_width, new_h))
                    else:
                        front_small = front_frame
                        back_small = back_frame

                    if prev_front_small is not None:
                        front_scene = SharpestExtractor._detect_scene_change(
                            prev_front_small, front_small, config.scene_threshold)
                        back_scene = SharpestExtractor._detect_scene_change(
                            prev_back_small, back_small, config.scene_threshold)
                        is_scene = front_scene or back_scene

                    prev_front_small = front_small
                    prev_back_small = back_small

                paired_entries.append(
                    {
                        "relative_frame": frame_idx - start_frame,
                        "absolute_frame": frame_idx,
                        "window_index": SharpestExtractor._time_window_index(
                            frame_idx,
                            shared_fps,
                            range_start_sec,
                            config.interval_sec,
                        ),
                        "front_score": front_score,
                        "back_score": back_score,
                        "scene_change": is_scene,
                    }
                )

                if progress_callback and (
                    offset == 1
                    or offset == total_to_score
                    or offset % 30 == 0
                ):
                    raw_pct = min(offset / total_to_score, 1.0)
                    elapsed_sec = offset / shared_fps if shared_fps > 0 else 0.0
                    msg = (
                        "Analyzing pair timeline: "
                        f"{self._format_clock(elapsed_sec)} / {self._format_clock(total_duration)} "
                        f"({int(raw_pct * 100)}%) [{method_label}]"
                    )
                    progress_callback(int(raw_pct * 80), 100, msg)

            if paired_entries:
                front_vals = [e["front_score"] for e in paired_entries]
                back_vals = [e["back_score"] for e in paired_entries]
                imbalances = [abs(e["front_score"] - e["back_score"]) for e in paired_entries]
                _log(f"  {config.scoring_method.title()} scoring: {len(paired_entries)} frame pairs")
                _log(f"  Front sharpness: min={min(front_vals):.1f} max={max(front_vals):.1f} mean={sum(front_vals)/len(front_vals):.1f}")
                _log(f"  Back sharpness:  min={min(back_vals):.1f} max={max(back_vals):.1f} mean={sum(back_vals)/len(back_vals):.1f}")
                _log(f"  Imbalance: min={min(imbalances):.1f} max={max(imbalances):.1f} mean={sum(imbalances)/len(imbalances):.1f}")
                if scene_aware:
                    scene_count = sum(1 for e in paired_entries if e["scene_change"])
                    _log(f"  Scene changes detected: {scene_count}")

            if not paired_entries:
                raise RuntimeError("Pair analysis found no overlapping frames")

            if progress_callback:
                label = "Selecting scene-aware pair winners..." if scene_aware else "Selecting pair winners..."
                progress_callback(82, 100, label)

            return self._select_from_paired_entries(
                paired_entries,
                scene_aware=scene_aware,
                log=_log,
            )
        finally:
            if front_cap is not None:
                front_cap.release()
            if back_cap is not None:
                back_cap.release()

    def _extract_selected_pairs(
        self,
        front_cap,
        back_cap,
        frame_indices: list[int],
        start_frame: int,
        end_frame: int,
        shared_fps: float,
        front_out: Path,
        back_out: Path,
        config: PairedSplitConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> tuple[list[str], list[str], list[float], list[int], list[int]]:
        """Second pass: extract the previously selected shared frame indices."""

        frame_set = set(frame_indices)
        front_paths: list[str] = []
        back_paths: list[str] = []
        selected_times: list[float] = []
        source_front_frames: list[int] = []
        source_back_frames: list[int] = []

        total_selected = len(frame_indices)
        written = 0

        for frame_idx in range(start_frame, end_frame):
            if cancel_check and cancel_check():
                # Clean up files written so far before raising
                for p in front_paths + back_paths:
                    Path(p).unlink(missing_ok=True)
                raise RuntimeError("Cancelled")

            ok_front, front_frame = front_cap.read()
            ok_back, back_frame = back_cap.read()
            if not ok_front or not ok_back:
                break

            if frame_idx not in frame_set:
                continue

            written += 1
            frame_id = f"{written:06d}"
            front_path = front_out / f"{frame_id}.{config.output_format.lower()}"
            back_path = back_out / f"{frame_id}.{config.output_format.lower()}"
            self._write_pair_image(front_path, front_frame, config)
            self._write_pair_image(back_path, back_frame, config)
            front_paths.append(str(front_path))
            back_paths.append(str(back_path))
            selected_times.append(frame_idx / shared_fps)
            source_front_frames.append(frame_idx)
            source_back_frames.append(frame_idx)

            if progress_callback and (
                written == 1
                or written == total_selected
                or written % 10 == 0
            ):
                progress_callback(
                    written,
                    total_selected,
                    f"extracting {written}/{total_selected} selected pairs",
                )

            if written >= total_selected:
                break

        return (
            front_paths,
            back_paths,
            selected_times,
            source_front_frames,
            source_back_frames,
        )

    # ── GPU streaming paired extraction ────────────────────────────

    def _extract_sharpest_gpu(
        self,
        front_video: str,
        back_video: str,
        out_root: Path,
        front_out: Path,
        back_out: Path,
        start_frame: int,
        end_frame: int,
        range_start_sec: float,
        shared_fps: float,
        config: PairedSplitConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        _log: Optional[Callable[[str], None]] = None,
    ) -> Optional[PairedSplitResult]:
        """GPU streaming single-pass paired extraction.

        Scores and writes winners directly from GPU memory to avoid
        GPU/CPU decoder frame-index misalignment.

        Returns PairedSplitResult on success or cancellation.
        Returns None on GPU init/runtime failure (caller falls back to CPU).
        """
        import numpy as np
        import time

        if _log is None:
            _log = lambda _msg: None

        if not SharpestExtractor._gpu_available(SharpestExtractor()):
            return None

        total_frames = end_frame - start_frame
        scene_aware = config.scene_detection

        # Open both cudacodec readers with firstFrameIdx
        params = cv2.cudacodec.VideoReaderInitParams()
        if start_frame > 0:
            params.firstFrameIdx = start_frame
        try:
            front_reader = cv2.cudacodec.createVideoReader(front_video, params=params)
        except (cv2.error, TypeError) as e:
            for msg in self._sharpest._gpu_reader_failure_messages(
                front_video, "front", e):
                _log(msg)
            return None
        try:
            back_reader = cv2.cudacodec.createVideoReader(back_video, params=params)
        except (cv2.error, TypeError) as e:
            for msg in self._sharpest._gpu_reader_failure_messages(
                back_video, "back", e):
                _log(msg)
            return None

        # Everything from here can fail with cv2.error — wrap for CPU fallback
        gpu_written_paths: list[str] = []

        try:
            # Read first frame pair for format detection
            ret_f, first_front = front_reader.nextFrame()
            ret_b, first_back = back_reader.nextFrame()
            if not ret_f or not ret_b:
                _log("  GPU: could not read first frame pair")
                return None

            # Format detection — validate BOTH streams
            f_ch, b_ch = first_front.channels(), first_back.channels()
            f_depth, b_depth = first_front.type() & 7, first_back.type() & 7

            if f_ch != b_ch or f_depth != b_depth:
                _log(f"  GPU: front/back format mismatch "
                     f"(front: {f_ch}ch depth={f_depth}, back: {b_ch}ch depth={b_depth})")
                return None

            channels = f_ch
            is_16bit = (f_depth == 2)
            is_8bit = (f_depth == 0)

            if not (is_8bit or is_16bit):
                _log(f"  GPU: unsupported frame depth (depth={f_depth})")
                return None
            if channels not in (3, 4):
                _log(f"  GPU: unsupported frame format ({channels} channels)")
                return None

            gray_code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
            bgr_code = cv2.COLOR_BGRA2BGR if channels == 4 else None

            _log(f"  GPU: paired NVDEC decode, {channels}ch "
                 f"{'uint16' if is_16bit else 'uint8'}, "
                 f"scoring: {config.scoring_method}, scene detection: {scene_aware}")

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
                if channels == 4:
                    gpu_bgr = cv2.cuda.cvtColor(gpu_small, cv2.COLOR_BGRA2BGR)
                else:
                    gpu_bgr = gpu_small
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
                                [cv2.IMWRITE_JPEG_QUALITY, int(config.quality)])
                else:
                    cv2.imwrite(str(filepath), frame)

            # ── Output tracking ──────────────────────────────────────

            ext = config.output_format
            winners_written = 0
            front_paths: list[str] = []
            back_paths: list[str] = []
            selected_times: list[float] = []
            source_front_frames: list[int] = []
            source_back_frames: list[int] = []
            selected_front_scores: list[float] = []
            selected_back_scores: list[float] = []

            def _write_pair_winner(best_f_gpu, best_b_gpu, frame_num, f_score, b_score):
                nonlocal winners_written
                winners_written += 1
                idx_str = f"{winners_written:06d}"
                f_path = front_out / f"{idx_str}.{ext}"
                b_path = back_out / f"{idx_str}.{ext}"
                _save_winner_gpu(best_f_gpu, f_path)
                _save_winner_gpu(best_b_gpu, b_path)
                front_paths.append(str(f_path))
                back_paths.append(str(b_path))
                gpu_written_paths.extend([str(f_path), str(b_path)])
                selected_times.append(round(frame_num / shared_fps, 3))
                source_front_frames.append(frame_num)
                source_back_frames.append(frame_num)
                selected_front_scores.append(f_score)
                selected_back_scores.append(b_score)

            # ── Streaming single-pass loop ───────────────────────────

            t_start = time.perf_counter()
            total_scored = 0
            scene_count = 0

            best_key = None
            best_front_gpu = None
            best_back_gpu = None
            best_frame_num = -1
            best_f_score = 0.0
            best_b_score = 0.0
            prev_front_scene = None
            prev_back_scene = None
            current_window_idx: Optional[int] = None

            def _flush_best():
                if best_front_gpu is not None:
                    _write_pair_winner(best_front_gpu, best_back_gpu,
                                       best_frame_num, best_f_score, best_b_score)

            def _process_pair(f_gpu, b_gpu, frame_num, relative_idx):
                nonlocal best_key, best_front_gpu, best_back_gpu, best_frame_num
                nonlocal best_f_score, best_b_score, scene_count, total_scored
                nonlocal prev_front_scene, prev_back_scene
                nonlocal current_window_idx

                f_score = score_fn(_prepare_gray(f_gpu))
                b_score = score_fn(_prepare_gray(b_gpu))
                total_scored += 1
                window_idx = SharpestExtractor._time_window_index(
                    frame_num, shared_fps, range_start_sec, config.interval_sec)

                is_scene = False
                if scene_aware:
                    f_scene_bgr = _prepare_scene_bgr(f_gpu)
                    b_scene_bgr = _prepare_scene_bgr(b_gpu)
                    if prev_front_scene is not None:
                        f_sc = SharpestExtractor._detect_scene_change(
                            prev_front_scene, f_scene_bgr, config.scene_threshold)
                        b_sc = SharpestExtractor._detect_scene_change(
                            prev_back_scene, b_scene_bgr, config.scene_threshold)
                        is_scene = f_sc or b_sc
                    prev_front_scene = f_scene_bgr
                    prev_back_scene = b_scene_bgr

                if current_window_idx is None:
                    current_window_idx = window_idx
                elif window_idx != current_window_idx:
                    _flush_best()
                    best_key = None
                    best_front_gpu = None
                    best_back_gpu = None
                    best_frame_num = -1
                    current_window_idx = window_idx

                if is_scene:
                    scene_count += 1
                    _flush_best()
                    best_key = None
                    best_front_gpu = None
                    best_back_gpu = None
                    best_frame_num = -1

                pair_key = self._pair_sharpness_sort_key({
                    "front_score": f_score, "back_score": b_score,
                })
                if best_key is None or pair_key > best_key:
                    best_key = pair_key
                    best_front_gpu = f_gpu.clone()
                    best_back_gpu = b_gpu.clone()
                    best_frame_num = frame_num
                    best_f_score = f_score
                    best_b_score = b_score

                if total_frames > 0 and relative_idx % max(1, total_frames // 10) == 0:
                    pct = int(relative_idx / total_frames * 80)
                    if progress_callback:
                        progress_callback(pct, 100,
                            f"Analyzing pair (GPU): {int(relative_idx / total_frames * 100)}% "
                            f"({relative_idx}/{total_frames})")

            # Check cancel before first-frame processing
            if cancel_check and cancel_check():
                raise RuntimeError("Cancelled")

            # Process first pair (already read for format detection)
            _process_pair(first_front, first_back, start_frame, 0)

            # Continue with remaining frames
            for frame_idx in range(start_frame + 1, end_frame):
                if cancel_check and cancel_check():
                    for p in gpu_written_paths:
                        Path(p).unlink(missing_ok=True)
                    raise RuntimeError("Cancelled")

                ret_f, f_gpu = front_reader.nextFrame()
                ret_b, b_gpu = back_reader.nextFrame()
                if not ret_f or not ret_b:
                    break

                relative_idx = frame_idx - start_frame
                _process_pair(f_gpu, b_gpu, frame_idx, relative_idx)

            # Final flush (inside try — cv2.error here still falls back)
            _flush_best()

        except RuntimeError:
            raise  # re-raise cancel — do NOT catch as GPU error

        except cv2.error as e:
            _log(f"  GPU error during paired processing: {e}")
            for p in gpu_written_paths:
                Path(p).unlink(missing_ok=True)
            return None

        elapsed = time.perf_counter() - t_start
        _log(f"  GPU paired {config.scoring_method.title()} scoring: "
             f"{total_scored} frame pairs ({elapsed:.1f}s)")
        if scene_aware:
            _log(f"  Scene changes detected: {scene_count}")
        _log(f"  Winners: {winners_written} pairs extracted")

        if winners_written == 0:
            return PairedSplitResult(
                success=False,
                output_dir=str(out_root),
                error="No pairs selected (GPU path)",
            )

        # Build manifest
        scoring_method = config.scoring_method
        selection_method = (
            f"{scoring_method}_scene_aware_pair"
            if scene_aware else f"{scoring_method}_pair"
        )
        manifest = {
            "schema_version": 1,
            "dataset_type": "paired_split_frames",
            "front_video": str(Path(front_video).resolve()),
            "back_video": str(Path(back_video).resolve()),
            "mode": "sharpest",
            "scoring_method": scoring_method,
            "scene_detection": scene_aware,
            "interval_sec": config.interval_sec,
            "fps": shared_fps,
            "selection_method": selection_method,
            "gpu_accelerated": True,
            "pairs": [],
        }
        for index, (fp, bp, t, sf, sb, fs, bs) in enumerate(
            zip(front_paths, back_paths, selected_times,
                source_front_frames, source_back_frames,
                selected_front_scores, selected_back_scores),
            start=1,
        ):
            manifest["pairs"].append({
                "pair_index": index,
                "frame_id": f"{index:06d}",
                "front_image": Path(fp).relative_to(out_root).as_posix(),
                "back_image": Path(bp).relative_to(out_root).as_posix(),
                "time_sec": t,
                "source_front_frame": sf,
                "source_back_frame": sb,
                "score_kind": f"{scoring_method}_sharpness",
                "front_score": fs,
                "back_score": bs,
                "pair_score": min(fs, bs),
            })

        manifest_path = out_root / "paired_extraction_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

        return PairedSplitResult(
            success=True,
            pair_count=winners_written,
            output_dir=str(out_root),
            front_paths=front_paths,
            back_paths=back_paths,
            selected_times=selected_times,
            source_front_frames=source_front_frames,
            source_back_frames=source_back_frames,
            gpu_accelerated=True,
        )

    # ── main public API ──────────────────────────────────────────────

    def extract(
        self,
        front_video: str,
        back_video: str,
        output_dir: str,
        config: Optional[PairedSplitConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> PairedSplitResult:
        _log = log or (lambda _msg: None)
        if config is None:
            config = PairedSplitConfig()

        mode = config.mode.lower()
        if mode not in {"fixed", "sharpest"}:
            return PairedSplitResult(
                success=False,
                output_dir=output_dir,
                error=f"Unsupported paired extraction mode: {config.mode}",
            )

        scoring_method = config.scoring_method
        scene_detection = config.scene_detection

        out_root = Path(output_dir)
        front_out = out_root / "front" / "frames"
        back_out = out_root / "back" / "frames"
        self._ensure_empty_output(front_out)
        self._ensure_empty_output(back_out)
        manifest_path = out_root / "paired_extraction_manifest.json"
        if manifest_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing paired extraction manifest: {manifest_path}"
            )
        front_out.mkdir(parents=True, exist_ok=True)
        back_out.mkdir(parents=True, exist_ok=True)

        front_cap = back_cap = None
        try:
            front_cap, front_fps, front_count = self._open_video(front_video)
            back_cap, back_fps, back_count = self._open_video(back_video)
            shared_fps, shared_count = self._shared_stream_info(
                front_fps, back_fps, front_count, back_count
            )

            range_start_sec, start_frame, end_frame = SharpestExtractor._frame_range_for_seconds(
                shared_fps, shared_count, config.start_sec, config.end_sec)
            if end_frame <= start_frame:
                return PairedSplitResult(
                    success=False,
                    output_dir=str(out_root),
                    error="Invalid time range for paired extraction",
                )

            window_size = max(1, int(round(config.interval_sec * shared_fps)))
            total_frames = end_frame - start_frame
            effective_duration = total_frames / shared_fps if shared_fps > 0 else 0.0

            # Release CPU captures before GPU path to avoid 4 simultaneous readers
            front_cap.release()
            back_cap.release()
            front_cap = back_cap = None

            # Try GPU streaming path for sharpest mode
            if mode == "sharpest" and SharpestExtractor._gpu_available(SharpestExtractor()):
                gpu_result = self._extract_sharpest_gpu(
                    front_video, back_video, out_root, front_out, back_out,
                    start_frame, end_frame, range_start_sec, shared_fps, config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    _log=_log,
                )
                if gpu_result is not None:
                    return gpu_result
                _log("  Falling back to CPU paired extraction")

            # Re-open CPU captures for CPU path
            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if mode == "fixed":
                selected_frame_indices = self._fixed_frame_indices(
                    start_frame, end_frame, window_size
                )
                selected_front_scores: list[float] = []
                selected_back_scores: list[float] = []
            else:
                selected_frame_indices, selected_front_scores, selected_back_scores = self._select_sharpest_frame_indices(
                    front_video,
                    back_video,
                    out_root,
                    start_frame,
                    end_frame,
                    range_start_sec,
                    shared_fps,
                    total_frames,
                    config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    duration_sec=effective_duration,
                    _log=_log,
                )

            if not selected_frame_indices:
                return PairedSplitResult(
                    success=False,
                    output_dir=str(out_root),
                    error="No shared frame pairs were selected",
                )

            front_cap.release()
            back_cap.release()
            front_cap = back_cap = None

            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if progress_callback:
                progress_callback(
                    0,
                    len(selected_frame_indices),
                    f"extracting 0/{len(selected_frame_indices)} selected pairs",
                )

            (
                front_paths,
                back_paths,
                selected_times,
                source_front_frames,
                source_back_frames,
            ) = self._extract_selected_pairs(
                front_cap,
                back_cap,
                selected_frame_indices,
                start_frame,
                end_frame,
                shared_fps,
                front_out,
                back_out,
                config,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )

            # Build manifest
            selection_method = (
                f"{scoring_method}_scene_aware_pair"
                if mode == "sharpest" and scene_detection else
                f"{scoring_method}_pair"
                if mode == "sharpest" else
                "fixed_interval_pair"
            )

            manifest = {
                "schema_version": 1,
                "dataset_type": "paired_split_frames",
                "front_video": str(Path(front_video).resolve()),
                "back_video": str(Path(back_video).resolve()),
                "mode": mode,
                "scoring_method": scoring_method if mode == "sharpest" else None,
                "scene_detection": scene_detection if mode == "sharpest" else None,
                "interval_sec": config.interval_sec,
                "fps": shared_fps,
                "selection_method": selection_method,
                "pairs": [],
            }
            front_metric_values = (
                selected_front_scores if mode == "sharpest" else [None] * len(front_paths)
            )
            back_metric_values = (
                selected_back_scores if mode == "sharpest" else [None] * len(front_paths)
            )
            for index, (
                front_path,
                back_path,
                time_sec,
                src_front,
                src_back,
                front_metric,
                back_metric,
            ) in enumerate(
                zip(
                    front_paths,
                    back_paths,
                    selected_times,
                    source_front_frames,
                    source_back_frames,
                    front_metric_values,
                    back_metric_values,
                ),
                start=1,
            ):
                manifest["pairs"].append(
                    {
                        "pair_index": index,
                        "frame_id": f"{index:06d}",
                        "front_image": Path(front_path).relative_to(out_root).as_posix(),
                        "back_image": Path(back_path).relative_to(out_root).as_posix(),
                        "time_sec": round(time_sec, 3),
                        "source_front_frame": src_front,
                        "source_back_frame": src_back,
                        "score_kind": f"{scoring_method}_sharpness" if mode == "sharpest" else None,
                        "front_score": front_metric,
                        "back_score": back_metric,
                        "pair_score": (
                            min(front_metric, back_metric)
                            if mode == "sharpest"
                            and front_metric is not None and back_metric is not None else
                            None
                        ),
                    }
                )
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            return PairedSplitResult(
                success=True,
                pair_count=len(front_paths),
                output_dir=str(out_root),
                front_paths=front_paths,
                back_paths=back_paths,
                selected_times=selected_times,
                source_front_frames=source_front_frames,
                source_back_frames=source_back_frames,
            )
        except Exception as exc:
            return PairedSplitResult(
                success=False,
                output_dir=str(out_root),
                error=str(exc),
            )
        finally:
            if front_cap is not None:
                front_cap.release()
            if back_cap is not None:
                back_cap.release()

"""
Pair-aware frame extraction from front/back split lens videos.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from .sharpest_extractor import SharpestConfig, SharpestExtractor


@dataclass
class PairedSplitConfig:
    """Configuration for shared front/back frame extraction."""

    mode: str = "sharpest"  # "fixed" or "sharpest"
    sharpest_tier: str = "best"  # "fast", "balanced", or "best"
    interval_sec: float = 2.0
    quality: int = 95
    output_format: str = "jpg"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    scene_threshold: float = 0.3
    scale_width: int = 1920
    block_size: int = 32


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
    def _laplacian_sharpness(frame, analysis_width: int = 960) -> float:
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
    def _sharpest_config(config: PairedSplitConfig) -> SharpestConfig:
        return SharpestConfig(
            interval=config.interval_sec,
            scene_threshold=config.scene_threshold,
            scale_width=config.scale_width,
            block_size=config.block_size,
            quality=config.quality,
            output_format=config.output_format,
            start_sec=config.start_sec,
            end_sec=config.end_sec,
        )

    @staticmethod
    def _parse_blur_metadata(metadata_path: Path) -> list[tuple[int, float, float]]:
        """Parse ffmpeg blurdetect metadata into (frame_num, blur, scene_score)."""

        pat_frame = re.compile(r"frame:(\d+)")
        pat_blur = re.compile(r"lavfi\.blur=([0-9.]+)")
        pat_scene = re.compile(r"lavfi\.scene_score=([0-9.]+)")

        frame_data: list[tuple[int, float, float]] = []
        current_frame = -1
        current_blur = -1.0
        current_scene = 0.0

        lines = metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            line = line.strip()
            m = pat_frame.search(line)
            if m:
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

        if current_frame >= 0 and current_blur >= 0:
            frame_data.append((current_frame, current_blur, current_scene))

        return frame_data

    def _run_blurdetect_analysis(
        self,
        video_path: str,
        out_root: Path,
        config: PairedSplitConfig,
        label: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_offset: int = 0,
        progress_span: int = 40,
        duration_sec: float = 0.0,
    ) -> list[tuple[int, float, float]]:
        """Run blurdetect + scene scoring and return per-frame metadata."""

        metadata_path = Path(tempfile.mktemp(suffix=f"_{label}_blurdetect.txt", dir=str(out_root)))
        sharp_cfg = self._sharpest_config(config)

        def _scaled_progress(cur, tot, msg):
            if not progress_callback:
                return
            if tot:
                baseline_total = 85 if tot == 100 else tot
                pct = progress_offset + int((cur / baseline_total) * progress_span)
                progress_callback(
                    pct,
                    100,
                    msg.replace("Analyzing:", f"Analyzing {label}:"),
                )
            else:
                progress_callback(progress_offset, 100, f"Analyzing {label}: {msg}")

        try:
            ok, err = self._sharpest._run_blurdetect(
                video_path,
                metadata_path,
                sharp_cfg,
                progress_callback=_scaled_progress if progress_callback else None,
                duration_sec=duration_sec,
                cancel_check=cancel_check,
            )
            if not ok:
                raise RuntimeError(f"{label} blurdetect failed: {err}")
            frame_data = self._parse_blur_metadata(metadata_path)
            if not frame_data:
                raise RuntimeError(f"{label} blurdetect produced no frame metadata")
            return frame_data
        finally:
            metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _split_pair_chunk_at_scenes(
        chunk: list[dict[str, float]],
        threshold: float,
    ) -> list[list[dict[str, float]]]:
        if not chunk:
            return []
        sub_chunks: list[list[dict[str, float]]] = []
        current: list[dict[str, float]] = []
        for entry in chunk:
            if entry["scene_score"] >= threshold and current:
                sub_chunks.append(current)
                current = []
            current.append(entry)
        if current:
            sub_chunks.append(current)
        return sub_chunks

    @staticmethod
    def _pair_blur_sort_key(entry: dict[str, float]) -> tuple[float, float, float]:
        max_blur = max(entry["front_blur"], entry["back_blur"])
        avg_blur = (entry["front_blur"] + entry["back_blur"]) * 0.5
        imbalance = abs(entry["front_blur"] - entry["back_blur"])
        return (max_blur, avg_blur, imbalance)

    @staticmethod
    def _pair_sharpness_sort_key(entry: dict[str, float]) -> tuple[float, float, float]:
        min_sharp = min(entry["front_score"], entry["back_score"])
        avg_sharp = (entry["front_score"] + entry["back_score"]) * 0.5
        imbalance = abs(entry["front_score"] - entry["back_score"])
        return (min_sharp, avg_sharp, -imbalance)

    def _select_from_paired_entries(
        self,
        paired_entries: list[dict[str, float]],
        window_size: int,
        scene_threshold: float,
        scene_aware: bool,
        use_blur_scores: bool,
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
                self._split_pair_chunk_at_scenes(chunk_entries, scene_threshold)
                if scene_aware else
                [chunk_entries]
            )
            for sub_chunk in sub_chunks:
                if use_blur_scores:
                    winner = min(sub_chunk, key=self._pair_blur_sort_key)
                    selected_indices.append(int(winner["absolute_frame"]))
                    front_scores.append(float(winner["front_blur"]))
                    back_scores.append(float(winner["back_blur"]))
                else:
                    winner = max(sub_chunk, key=self._pair_sharpness_sort_key)
                    selected_indices.append(int(winner["absolute_frame"]))
                    front_scores.append(float(winner["front_score"]))
                    back_scores.append(float(winner["back_score"]))

        for entry in paired_entries:
            chunk_index = int(entry["relative_frame"]) // window_size
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

    def _select_fast_frame_indices(
        self,
        front_video: str,
        back_video: str,
        start_frame: int,
        end_frame: int,
        window_size: int,
        shared_fps: float,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[int], list[float], list[float]]:
        def _log(msg):
            if log:
                log(msg)

        front_cap = back_cap = None
        try:
            front_cap, _, _ = self._open_video(front_video)
            back_cap, _, _ = self._open_video(back_video)
            front_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            back_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            total_frames = max(1, end_frame - start_frame)
            total_duration = total_frames / shared_fps if shared_fps > 0 else 0.0
            paired_entries: list[dict[str, float]] = []

            for offset, frame_idx in enumerate(range(start_frame, end_frame), start=1):
                if cancel_check and cancel_check():
                    raise RuntimeError("Cancelled")

                ok_front, front_frame = front_cap.read()
                ok_back, back_frame = back_cap.read()
                if not ok_front or not ok_back:
                    break

                front_score = self._laplacian_sharpness(front_frame)
                back_score = self._laplacian_sharpness(back_frame)
                paired_entries.append(
                    {
                        "relative_frame": frame_idx - start_frame,
                        "absolute_frame": frame_idx,
                        "front_score": front_score,
                        "back_score": back_score,
                    }
                )

                if progress_callback and (
                    offset == 1
                    or offset == total_frames
                    or offset % 30 == 0
                ):
                    raw_pct = min(offset / total_frames, 1.0)
                    elapsed_sec = offset / shared_fps if shared_fps > 0 else 0.0
                    msg = (
                        "Analyzing pair timeline: "
                        f"{self._format_clock(elapsed_sec)} / {self._format_clock(total_duration)} "
                        f"({int(raw_pct * 100)}%) [Fast]"
                    )
                    progress_callback(int(raw_pct * 80), 100, msg)

            if paired_entries:
                front_vals = [e["front_score"] for e in paired_entries]
                back_vals = [e["back_score"] for e in paired_entries]
                imbalances = [abs(e["front_score"] - e["back_score"]) for e in paired_entries]
                _log(f"  Fast scoring: {len(paired_entries)} frame pairs")
                _log(f"  Front sharpness: min={min(front_vals):.1f} max={max(front_vals):.1f} mean={sum(front_vals)/len(front_vals):.1f}")
                _log(f"  Back sharpness:  min={min(back_vals):.1f} max={max(back_vals):.1f} mean={sum(back_vals)/len(back_vals):.1f}")
                _log(f"  Imbalance: min={min(imbalances):.1f} max={max(imbalances):.1f} mean={sum(imbalances)/len(imbalances):.1f}")

            if not paired_entries:
                raise RuntimeError("Fast pair analysis found no overlapping frames")

            if progress_callback:
                progress_callback(82, 100, "Selecting pair winners...")

            return self._select_from_paired_entries(
                paired_entries,
                window_size,
                scene_threshold=0.0,
                scene_aware=False,
                use_blur_scores=False,
                log=_log,
            )
        finally:
            if front_cap is not None:
                front_cap.release()
            if back_cap is not None:
                back_cap.release()

    def _select_sharpest_frame_indices(
        self,
        front_video: str,
        back_video: str,
        out_root: Path,
        start_frame: int,
        end_frame: int,
        window_size: int,
        shared_fps: float,
        total_frames: int,
        config: PairedSplitConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        duration_sec: float = 0.0,
        _log: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[int], list[float], list[float]]:
        """Analyze both videos with blurdetect and pick the best shared frame indices."""

        if _log is None:
            _log = lambda _msg: None

        tier = (config.sharpest_tier or "best").lower()
        if tier not in {"fast", "balanced", "best"}:
            raise RuntimeError(f"Unsupported sharpest tier: {config.sharpest_tier}")

        if tier == "fast":
            return self._select_fast_frame_indices(
                front_video,
                back_video,
                start_frame,
                end_frame,
                window_size,
                shared_fps,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                log=_log,
            )

        front_data = self._run_blurdetect_analysis(
            front_video,
            out_root,
            config,
            "front",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            progress_offset=0,
            progress_span=40,
            duration_sec=duration_sec,
        )
        back_data = self._run_blurdetect_analysis(
            back_video,
            out_root,
            config,
            "back",
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            progress_offset=40,
            progress_span=40,
            duration_sec=duration_sec,
        )

        back_map = {frame_num: (blur, scene) for frame_num, blur, scene in back_data}
        paired_entries: list[dict[str, float]] = []
        for frame_num, front_blur, front_scene in front_data:
            if frame_num not in back_map:
                continue
            back_blur, back_scene = back_map[frame_num]
            absolute_frame = start_frame + frame_num
            if absolute_frame >= end_frame:
                continue
            paired_entries.append(
                {
                    "relative_frame": frame_num,
                    "absolute_frame": absolute_frame,
                    "front_blur": front_blur,
                    "back_blur": back_blur,
                    "front_scene": front_scene,
                    "back_scene": back_scene,
                    "scene_score": max(front_scene, back_scene),
                }
            )

        if paired_entries:
            _log(f"  Paired blurdetect: {len(paired_entries)} matched frames")
            if tier == "best":
                scene_count = sum(1 for e in paired_entries if e.get("scene_score", 0) >= config.scene_threshold)
                _log(f"  Scene changes: {scene_count}")

        if not paired_entries:
            raise RuntimeError("No overlapping front/back blur metadata was found")

        if progress_callback:
            label = "Selecting scene-aware pair winners..." if tier == "best" else "Selecting pair winners..."
            progress_callback(82, 100, label)

        return self._select_from_paired_entries(
            paired_entries,
            window_size,
            scene_threshold=config.scene_threshold,
            scene_aware=(tier == "best"),
            use_blur_scores=True,
            log=_log,
        )

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
        sharpest_tier = (config.sharpest_tier or "best").lower()
        if sharpest_tier not in {"fast", "balanced", "best"}:
            return PairedSplitResult(
                success=False,
                output_dir=output_dir,
                error=f"Unsupported sharpest tier: {config.sharpest_tier}",
            )

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

            start_frame = max(0, int(round((config.start_sec or 0.0) * shared_fps)))
            end_frame = shared_count
            if config.end_sec is not None:
                end_frame = min(end_frame, int(round(config.end_sec * shared_fps)))
            if end_frame <= start_frame:
                return PairedSplitResult(
                    success=False,
                    output_dir=str(out_root),
                    error="Invalid time range for paired extraction",
                )

            window_size = max(1, int(round(config.interval_sec * shared_fps)))
            total_frames = end_frame - start_frame
            effective_duration = total_frames / shared_fps if shared_fps > 0 else 0.0

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
                    window_size,
                    shared_fps,
                    total_frames,
                    config,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    duration_sec=effective_duration,
                    log=_log,
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

            manifest = {
                "schema_version": 1,
                "dataset_type": "paired_split_frames",
                "front_video": str(Path(front_video).resolve()),
                "back_video": str(Path(back_video).resolve()),
                "mode": mode,
                "sharpest_tier": sharpest_tier if mode == "sharpest" else None,
                "interval_sec": config.interval_sec,
                "fps": shared_fps,
                "selection_method": (
                    "two_pass_laplacian_pair"
                    if mode == "sharpest" and sharpest_tier == "fast" else
                    "two_pass_blurdetect_pair"
                    if mode == "sharpest" and sharpest_tier == "balanced" else
                    "two_pass_blurdetect_scene_aware_pair"
                    if mode == "sharpest" else
                    "fixed_interval_pair"
                ),
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
                score_kind = (
                    "laplacian_sharpness"
                    if mode == "sharpest" and sharpest_tier == "fast" else
                    "blurdetect_blur"
                    if mode == "sharpest" else
                    None
                )
                manifest["pairs"].append(
                    {
                        "pair_index": index,
                        "frame_id": f"{index:06d}",
                        "front_image": Path(front_path).relative_to(out_root).as_posix(),
                        "back_image": Path(back_path).relative_to(out_root).as_posix(),
                        "time_sec": round(time_sec, 3),
                        "source_front_frame": src_front,
                        "source_back_frame": src_back,
                        "score_kind": score_kind,
                        "front_score": front_metric,
                        "back_score": back_metric,
                        "pair_score": (
                            min(front_metric, back_metric)
                            if mode == "sharpest" and sharpest_tier == "fast"
                            and front_metric is not None and back_metric is not None else
                            max(front_metric, back_metric)
                            if mode == "sharpest"
                            and front_metric is not None and back_metric is not None else
                            None
                        ),
                        "front_blur": (
                            front_metric
                            if mode == "sharpest" and sharpest_tier in {"balanced", "best"} else
                            None
                        ),
                        "back_blur": (
                            back_metric
                            if mode == "sharpest" and sharpest_tier in {"balanced", "best"} else
                            None
                        ),
                        "pair_blur": (
                            max(front_metric, back_metric)
                            if mode == "sharpest" and sharpest_tier in {"balanced", "best"}
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

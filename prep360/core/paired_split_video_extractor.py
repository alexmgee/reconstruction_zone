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
        window_size: int,
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
                window_size,
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

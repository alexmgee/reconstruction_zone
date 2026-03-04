"""
Motion-Aware Frame Selection

Select optimal frames from video or image sequences using a combination
of sharpness scoring and optical flow analysis. Upgrades fixed-interval
extraction to motion-aware selection.

Strategy:
    1. Score all candidate frames for sharpness (Laplacian variance)
    2. Reject frames below a sharpness threshold
    3. Compute inter-frame optical flow (DIS) between consecutive survivors
    4. Greedily select frames maintaining a target motion baseline
    5. If still too many, prefer sharper frames

Works with both fisheye frames and ERP frames — the scoring is
content-agnostic.

Usage:
    selector = MotionSelector()

    # From pre-extracted frames
    selected = selector.select_from_paths(frame_paths, max_frames=100)

    # From video (samples densely, selects intelligently)
    selected = selector.select_from_video(
        "input.mp4", output_dir="./selected/",
        sample_fps=5.0, max_frames=200,
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameScore:
    """Score for a single frame candidate."""
    path: str
    index: int
    timestamp: float            # seconds into video (or frame index for images)
    sharpness: float            # Laplacian variance
    flow_from_prev: float       # average optical flow magnitude from previous frame
    selected: bool = False


@dataclass
class SelectionResult:
    """Result of motion-aware frame selection."""
    total_candidates: int
    sharpness_rejected: int
    selected_count: int
    selected_frames: List[FrameScore]
    all_scores: List[FrameScore]

    def summary(self) -> str:
        if not self.selected_frames:
            return "No frames selected"
        sharpness_vals = [f.sharpness for f in self.selected_frames]
        flow_vals = [f.flow_from_prev for f in self.selected_frames if f.flow_from_prev > 0]
        lines = [
            f"Selected {self.selected_count}/{self.total_candidates} frames "
            f"({self.sharpness_rejected} rejected for blur)",
        ]
        if sharpness_vals:
            lines.append(
                f"Sharpness: min={min(sharpness_vals):.1f} "
                f"max={max(sharpness_vals):.1f} "
                f"mean={np.mean(sharpness_vals):.1f}"
            )
        if flow_vals:
            lines.append(
                f"Flow: min={min(flow_vals):.1f} "
                f"max={max(flow_vals):.1f} "
                f"mean={np.mean(flow_vals):.1f}"
            )
        return "\n".join(lines)


class MotionSelector:
    """Select optimal frames using sharpness + optical flow."""

    def __init__(
        self,
        min_sharpness: float = 50.0,
        min_flow: float = 2.0,
        max_flow: float = 50.0,
        target_flow: float = 10.0,
        flow_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        downscale: int = 4,
    ):
        """Initialize the motion-aware selector.

        Args:
            min_sharpness: Laplacian variance threshold — reject below this.
            min_flow: Minimum average flow (pixels) between selected frames.
                      Prevents selecting redundant near-identical frames.
            max_flow: Maximum average flow. Frames with excessive motion
                      relative to the previous selected frame are skipped
                      (likely extreme motion blur).
            target_flow: Ideal inter-frame flow. The greedy selector tries
                         to maintain roughly this much motion between picks.
            flow_preset: DIS optical flow quality preset.
                         PRESET_ULTRAFAST, PRESET_FAST, PRESET_MEDIUM.
            downscale: Factor to downscale images before flow computation.
                       4x reduces 3840px to 960px — much faster, still accurate.
        """
        self.min_sharpness = min_sharpness
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.target_flow = target_flow
        self.downscale = downscale
        self._flow_preset = flow_preset

    def _create_dis(self):
        """Create DIS optical flow instance (not picklable, so recreate)."""
        return cv2.DISOpticalFlow.create(self._flow_preset)

    def compute_sharpness(self, image: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _to_gray_downscaled(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and downscale for flow computation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(
                gray, (w // self.downscale, h // self.downscale),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def compute_flow_magnitude(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> float:
        """Average optical flow magnitude between two frames.

        Input should be grayscale at the desired scale (use _to_gray_downscaled).
        Returns average flow in pixels at the downscaled resolution.
        """
        dis = self._create_dis()
        flow = dis.calc(prev_gray, curr_gray, None)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        # Scale back to original resolution
        return float(np.mean(mag)) * self.downscale

    def select_from_paths(
        self,
        frame_paths: List[str],
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> SelectionResult:
        """Select optimal frames from a list of image paths.

        Args:
            frame_paths: Ordered list of frame image paths.
            max_frames: Maximum number of frames to select.
            progress_callback: Called with (current, total, message).

        Returns:
            SelectionResult with selected frames and scores.
        """
        if not frame_paths:
            return SelectionResult(0, 0, 0, [], [])

        # Phase 1: Score all frames for sharpness
        all_scores = []
        for i, path in enumerate(frame_paths):
            img = cv2.imread(str(path))
            if img is None:
                continue
            sharpness = self.compute_sharpness(img)
            all_scores.append(FrameScore(
                path=str(path),
                index=i,
                timestamp=float(i),
                sharpness=sharpness,
                flow_from_prev=0.0,
            ))
            if progress_callback:
                progress_callback(i + 1, len(frame_paths), f"scoring {Path(path).name}")

        # Phase 2: Reject blurry frames
        sharpness_rejected = 0
        candidates = []
        for fs in all_scores:
            if fs.sharpness >= self.min_sharpness:
                candidates.append(fs)
            else:
                sharpness_rejected += 1

        if not candidates:
            return SelectionResult(
                len(all_scores), sharpness_rejected, 0, [], all_scores,
            )

        # Phase 3: Compute optical flow between consecutive candidates
        prev_gray = None
        for i, fs in enumerate(candidates):
            img = cv2.imread(fs.path)
            gray = self._to_gray_downscaled(img)
            if prev_gray is not None:
                fs.flow_from_prev = self.compute_flow_magnitude(prev_gray, gray)
            prev_gray = gray
            if progress_callback:
                progress_callback(i + 1, len(candidates), f"flow {Path(fs.path).name}")

        # Phase 4: Greedy selection maintaining target baseline
        selected = self._greedy_select(candidates, max_frames)

        return SelectionResult(
            total_candidates=len(all_scores),
            sharpness_rejected=sharpness_rejected,
            selected_count=len(selected),
            selected_frames=selected,
            all_scores=all_scores,
        )

    def select_from_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        sample_fps: float = 5.0,
        max_frames: Optional[int] = None,
        start_sec: float = 0,
        end_sec: Optional[float] = None,
        stream_index: Optional[int] = None,
        progress_callback=None,
    ) -> SelectionResult:
        """Select optimal frames directly from a video file.

        Samples at sample_fps, scores for sharpness + flow,
        and selects the best subset. Optionally saves selected
        frames to output_dir.

        Args:
            video_path: Path to video file (MP4, OSV, etc.)
            output_dir: If set, save selected frames as JPEGs here.
            sample_fps: Sampling rate (frames per second) from the video.
            max_frames: Maximum frames to select.
            start_sec: Start time in the video.
            end_sec: End time (None = end of video).
            stream_index: Specific stream index for multi-stream containers.
            progress_callback: Called with (current, total, message).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        if end_sec is None:
            end_sec = duration

        # Calculate frame sampling interval
        sample_interval = max(1, int(video_fps / sample_fps))

        # Seek to start
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Phase 1+2: Sample and score
        all_scores = []
        prev_gray = None
        frame_idx = start_frame
        dis = self._create_dis()

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % sample_interval == 0:
                timestamp = frame_idx / video_fps
                sharpness = self.compute_sharpness(frame)

                flow_mag = 0.0
                gray = self._to_gray_downscaled(frame)
                if prev_gray is not None:
                    flow = dis.calc(prev_gray, gray, None)
                    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    flow_mag = float(np.mean(mag)) * self.downscale
                prev_gray = gray

                fs = FrameScore(
                    path="",
                    index=len(all_scores),
                    timestamp=timestamp,
                    sharpness=sharpness,
                    flow_from_prev=flow_mag,
                )
                # Temporarily store frame data for saving later
                fs._frame_data = frame  # noqa: attribute assignment
                all_scores.append(fs)

                if progress_callback:
                    progress_callback(
                        len(all_scores), 0,
                        f"t={timestamp:.1f}s sharp={sharpness:.0f} flow={flow_mag:.1f}",
                    )

            frame_idx += 1

        cap.release()

        # Phase 2: Filter by sharpness
        sharpness_rejected = 0
        candidates = []
        for fs in all_scores:
            if fs.sharpness >= self.min_sharpness:
                candidates.append(fs)
            else:
                sharpness_rejected += 1

        # Phase 3: Greedy select
        selected = self._greedy_select(candidates, max_frames)

        # Phase 4: Save selected frames if output_dir provided
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fs in selected:
                frame_data = getattr(fs, '_frame_data', None)
                if frame_data is not None:
                    filename = f"frame_{fs.index:05d}_t{fs.timestamp:.2f}s.jpg"
                    filepath = out / filename
                    cv2.imwrite(str(filepath), frame_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    fs.path = str(filepath)

        # Clean up frame data references
        for fs in all_scores:
            if hasattr(fs, '_frame_data'):
                del fs._frame_data

        return SelectionResult(
            total_candidates=len(all_scores),
            sharpness_rejected=sharpness_rejected,
            selected_count=len(selected),
            selected_frames=selected,
            all_scores=all_scores,
        )

    def analyze_segment(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        sample_fps: float = 5.0,
    ) -> List[FrameScore]:
        """Analyze a video segment and return scored frame candidates.

        Used by bridge extraction — analyze a specific time range
        and return all scores without selecting. The caller decides
        which frames to keep.

        Returns list of FrameScore sorted by timestamp.
        """
        result = self.select_from_video(
            video_path,
            sample_fps=sample_fps,
            max_frames=None,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        return result.all_scores

    def _greedy_select(
        self,
        candidates: List[FrameScore],
        max_frames: Optional[int],
    ) -> List[FrameScore]:
        """Greedy frame selection maintaining target motion baseline.

        Strategy:
        - Always take the first candidate
        - Accumulate flow from the last selected frame
        - When accumulated flow >= target_flow, select the sharpest
          frame in the recent window
        - Skip if accumulated flow > max_flow (too much motion)
        - Continue until we hit max_frames or run out of candidates
        """
        if not candidates:
            return []

        # Always select the first frame
        selected = [candidates[0]]
        candidates[0].selected = True

        if max_frames is not None and max_frames <= 1:
            return selected

        accumulated_flow = 0.0
        window = []  # frames since last selection

        for i in range(1, len(candidates)):
            fs = candidates[i]
            accumulated_flow += fs.flow_from_prev

            # Too little motion — skip (too similar to last selected)
            if accumulated_flow < self.min_flow:
                continue

            window.append((fs, accumulated_flow))

            # Enough motion accumulated — pick best from window
            if accumulated_flow >= self.target_flow:
                # Pick the sharpest frame in the window
                best = max(window, key=lambda x: x[0].sharpness)
                best_fs = best[0]
                best_fs.selected = True
                selected.append(best_fs)

                # Reset accumulation from the selected frame's position
                # Approximate: use remaining flow after selected frame
                idx_in_window = window.index(best)
                remaining_flow = sum(
                    w[0].flow_from_prev for w in window[idx_in_window + 1:]
                )
                accumulated_flow = remaining_flow
                window = [w for w in window[idx_in_window + 1:]]

                if max_frames is not None and len(selected) >= max_frames:
                    break

            # Way too much motion — forced selection even if not ideal
            elif accumulated_flow > self.max_flow:
                best = max(window, key=lambda x: x[0].sharpness)
                best_fs = best[0]
                best_fs.selected = True
                selected.append(best_fs)
                accumulated_flow = 0.0
                window = []

                if max_frames is not None and len(selected) >= max_frames:
                    break

        return selected


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Motion-aware frame selection from video or image sequences"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Select from video
    vid_p = sub.add_parser("video", help="Select frames from video")
    vid_p.add_argument("input", help="Video file path")
    vid_p.add_argument("-o", "--output", help="Output directory for selected frames")
    vid_p.add_argument("--fps", type=float, default=5.0, help="Sample FPS (default: 5)")
    vid_p.add_argument("--max", type=int, help="Maximum frames to select")
    vid_p.add_argument("--start", type=float, default=0, help="Start time (seconds)")
    vid_p.add_argument("--end", type=float, help="End time (seconds)")
    vid_p.add_argument("--min-sharpness", type=float, default=50.0)
    vid_p.add_argument("--target-flow", type=float, default=10.0)
    vid_p.add_argument("--report", help="Save report JSON to this path")

    # Select from image directory
    img_p = sub.add_parser("images", help="Select from pre-extracted frames")
    img_p.add_argument("input", help="Directory of images")
    img_p.add_argument("--max", type=int, help="Maximum frames to select")
    img_p.add_argument("--min-sharpness", type=float, default=50.0)
    img_p.add_argument("--target-flow", type=float, default=10.0)
    img_p.add_argument("--report", help="Save report JSON to this path")

    # Analyze video segment (for bridge extraction)
    seg_p = sub.add_parser("analyze", help="Analyze a video segment")
    seg_p.add_argument("input", help="Video file path")
    seg_p.add_argument("--start", type=float, required=True, help="Start time (sec)")
    seg_p.add_argument("--end", type=float, required=True, help="End time (sec)")
    seg_p.add_argument("--fps", type=float, default=5.0)

    args = parser.parse_args()

    try:
        if args.command == "video":
            selector = MotionSelector(
                min_sharpness=args.min_sharpness,
                target_flow=args.target_flow,
            )

            def progress(cur, total, msg):
                print(f"  [{cur}] {msg}")

            result = selector.select_from_video(
                args.input,
                output_dir=args.output,
                sample_fps=args.fps,
                max_frames=args.max,
                start_sec=args.start,
                end_sec=args.end,
                progress_callback=progress,
            )

            print(result.summary())

            if args.output:
                print(f"Saved {result.selected_count} frames to {args.output}")

            if args.report:
                _save_report(result, args.report)
                print(f"Report saved to {args.report}")

        elif args.command == "images":
            selector = MotionSelector(
                min_sharpness=args.min_sharpness,
                target_flow=args.target_flow,
            )

            input_dir = Path(args.input)
            exts = {'.jpg', '.jpeg', '.png'}
            paths = sorted(
                str(f) for f in input_dir.iterdir()
                if f.suffix.lower() in exts
            )

            if not paths:
                print("No images found")
                return 1

            def progress(cur, total, msg):
                print(f"  [{cur}/{total}] {msg}")

            result = selector.select_from_paths(
                paths, max_frames=args.max, progress_callback=progress,
            )

            print(result.summary())
            print("\nSelected frames:")
            for fs in result.selected_frames:
                print(f"  {Path(fs.path).name}: "
                      f"sharp={fs.sharpness:.1f} flow={fs.flow_from_prev:.1f}")

            if args.report:
                _save_report(result, args.report)

        elif args.command == "analyze":
            selector = MotionSelector()
            scores = selector.analyze_segment(
                args.input, args.start, args.end, args.fps,
            )
            print(f"Analyzed {len(scores)} frames from "
                  f"{args.start:.1f}s to {args.end:.1f}s")
            for fs in scores:
                flag = "BLUR" if fs.sharpness < selector.min_sharpness else "    "
                print(f"  t={fs.timestamp:.2f}s "
                      f"sharp={fs.sharpness:.0f} "
                      f"flow={fs.flow_from_prev:.1f} {flag}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def _save_report(result: SelectionResult, path: str):
    """Save selection report as JSON."""
    data = {
        "total_candidates": result.total_candidates,
        "sharpness_rejected": result.sharpness_rejected,
        "selected_count": result.selected_count,
        "selected": [
            {
                "path": fs.path,
                "index": fs.index,
                "timestamp": fs.timestamp,
                "sharpness": fs.sharpness,
                "flow_from_prev": fs.flow_from_prev,
            }
            for fs in result.selected_frames
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    exit(main())

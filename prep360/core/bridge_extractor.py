"""
Bridge Frame Extraction

Extract targeted frames from source video to fill spatial gaps
detected in COLMAP reconstructions (Phase 5 → Phase 6 feedback loop).

Workflow:
    1. Load a GapReport from gap_detector.analyze() or JSON
    2. Plan extraction — map each gap's timestamps to video segments
    3. Extract candidate frames using MotionSelector
    4. Optionally reframe fisheye → perspective via FisheyeReframer
    5. Save bridge frames + report

Usage:
    from prep360.core import BridgeExtractor, GapReport

    report = GapReport.load("gaps.json")
    extractor = BridgeExtractor()
    result = extractor.extract(
        gap_report=report,
        video_path="input.osv",
        output_dir="./bridges/",
    )
    print(result.summary())

CLI:
    python -m prep360.core.bridge_extractor gaps.json input.osv ./bridges/
    python -m prep360.core.bridge_extractor gaps.json input.osv ./bridges/ --reframe
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .gap_detector import GapReport, SpatialGap
from .motion_selector import MotionSelector, FrameScore


@dataclass
class BridgeRequest:
    """Extraction plan for a single gap."""
    gap_index: int
    gap: SpatialGap
    video_path: str
    start_sec: float
    end_sec: float
    target_frames: int
    padding_sec: float = 0.0     # extra seconds added around the gap window


@dataclass
class BridgeFrameInfo:
    """Info about a single extracted bridge frame."""
    path: str
    timestamp: float
    sharpness: float
    flow_from_prev: float
    gap_index: int
    reframed_paths: List[str] = field(default_factory=list)


@dataclass
class BridgeResult:
    """Result of bridge extraction for all gaps."""
    requests: List[BridgeRequest]
    frames: List[BridgeFrameInfo]
    total_extracted: int
    total_reframed: int
    output_dir: str

    def summary(self) -> str:
        lines = [
            f"Bridge extraction: {len(self.requests)} gaps processed",
            f"  Extracted: {self.total_extracted} frames",
        ]
        if self.total_reframed > 0:
            lines.append(f"  Reframed:  {self.total_reframed} perspective crops")
        lines.append(f"  Output:    {self.output_dir}")

        for req in self.requests:
            gap_frames = [f for f in self.frames if f.gap_index == req.gap_index]
            ts = f"[{req.start_sec:.1f}s .. {req.end_sec:.1f}s]"
            lines.append(
                f"  Gap {req.gap_index} ({req.gap.gap_type}): "
                f"{len(gap_frames)}/{req.target_frames} frames {ts}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_extracted": self.total_extracted,
            "total_reframed": self.total_reframed,
            "output_dir": self.output_dir,
            "requests": [
                {
                    "gap_index": r.gap_index,
                    "gap_type": r.gap.gap_type,
                    "start_sec": r.start_sec,
                    "end_sec": r.end_sec,
                    "target_frames": r.target_frames,
                }
                for r in self.requests
            ],
            "frames": [
                {
                    "path": f.path,
                    "timestamp": f.timestamp,
                    "sharpness": f.sharpness,
                    "flow_from_prev": f.flow_from_prev,
                    "gap_index": f.gap_index,
                    "reframed_paths": f.reframed_paths,
                }
                for f in self.frames
            ],
        }

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class BridgeExtractor:
    """Extract bridge frames from source video to fill reconstruction gaps.

    Connects Phase 5 (gap detection) → Phase 4 (motion-aware selection)
    → Phase 3 (fisheye reframing) into a single extraction step.
    """

    def __init__(
        self,
        sample_fps: float = 5.0,
        min_sharpness: float = 50.0,
        target_flow: float = 8.0,
        padding_sec: float = 2.0,
        frames_per_disconnected: int = 8,
        frames_per_sparse: int = 3,
    ):
        """Initialize bridge extractor.

        Args:
            sample_fps: How densely to sample the video in gap regions.
            min_sharpness: Laplacian variance threshold for blur rejection.
            target_flow: Target optical flow between selected bridge frames.
                         Lower than normal extraction since we want denser
                         coverage in gap regions.
            padding_sec: Extra seconds to add before/after each gap's
                         timestamp window. Gives margin for the motion
                         selector to find good frames near the boundary.
            frames_per_disconnected: Target bridge frames for disconnected
                                     component gaps (most severe).
            frames_per_sparse: Target bridge frames for sparse region gaps.
        """
        self.sample_fps = sample_fps
        self.min_sharpness = min_sharpness
        self.target_flow = target_flow
        self.padding_sec = padding_sec
        self.frames_per_disconnected = frames_per_disconnected
        self.frames_per_sparse = frames_per_sparse

    def plan(
        self,
        gap_report: GapReport,
        video_path: str,
        video_duration: Optional[float] = None,
    ) -> List[BridgeRequest]:
        """Create extraction plan from a gap report.

        Maps each gap's estimated timestamps to a video segment.
        Gaps without timestamps are skipped (cannot map to video).

        Args:
            gap_report: Output from GapDetector.analyze().
            video_path: Source video file.
            video_duration: Total video length in seconds. If not
                            provided, probed from the video file.

        Returns:
            List of BridgeRequest, one per actionable gap.
        """
        if video_duration is None:
            video_duration = self._probe_duration(video_path)

        requests = []
        for i, gap in enumerate(gap_report.gaps):
            if gap.estimated_timestamps is None:
                continue

            t_start, t_end = gap.estimated_timestamps

            # Scale target frames by gap severity
            if gap.gap_type == "disconnected":
                target = self.frames_per_disconnected
            else:
                target = self.frames_per_sparse

            # Add padding and clamp to video bounds
            start = max(0.0, t_start - self.padding_sec)
            end = min(video_duration, t_end + self.padding_sec)

            # Ensure we have at least a minimal window
            if end - start < 1.0:
                mid = (t_start + t_end) / 2.0
                start = max(0.0, mid - 2.0)
                end = min(video_duration, mid + 2.0)

            requests.append(BridgeRequest(
                gap_index=i,
                gap=gap,
                video_path=video_path,
                start_sec=start,
                end_sec=end,
                target_frames=target,
                padding_sec=self.padding_sec,
            ))

        return requests

    def extract(
        self,
        gap_report: GapReport,
        video_path: str,
        output_dir: str,
        reframe: bool = False,
        reframer=None,
        view_config=None,
        stream_index: Optional[int] = None,
        video_duration: Optional[float] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> BridgeResult:
        """Extract bridge frames for all gaps in a report.

        Args:
            gap_report: Gap analysis result.
            video_path: Source video (MP4, OSV, etc.).
            output_dir: Root output directory. Creates subdirs per gap.
            reframe: If True, reframe fisheye frames to perspective.
                     Requires reframer and view_config.
            reframer: FisheyeReframer instance (required if reframe=True).
            view_config: FisheyeViewConfig (required if reframe=True).
            stream_index: Video stream index for multi-stream containers.
            video_duration: Total video duration (probed if not given).
            progress_callback: Called with status messages.

        Returns:
            BridgeResult with all extracted frames and metadata.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Plan extraction
        requests = self.plan(gap_report, video_path, video_duration)

        if not requests:
            if progress_callback:
                progress_callback("No actionable gaps (no timestamps available)")
            return BridgeResult(
                requests=[],
                frames=[],
                total_extracted=0,
                total_reframed=0,
                output_dir=str(out_path),
            )

        if progress_callback:
            progress_callback(
                f"Planned {len(requests)} gap extractions "
                f"from {len(gap_report.gaps)} total gaps"
            )

        # Create motion selector with bridge-tuned parameters
        selector = MotionSelector(
            min_sharpness=self.min_sharpness,
            target_flow=self.target_flow,
        )

        all_frames: List[BridgeFrameInfo] = []
        total_reframed = 0

        for req in requests:
            gap_dir = out_path / f"gap_{req.gap_index:02d}"
            gap_dir.mkdir(exist_ok=True)

            if progress_callback:
                progress_callback(
                    f"Gap {req.gap_index} ({req.gap.gap_type}): "
                    f"sampling {req.start_sec:.1f}s-{req.end_sec:.1f}s "
                    f"for {req.target_frames} frames"
                )

            # Use motion selector to extract best frames from this segment
            result = selector.select_from_video(
                video_path=req.video_path,
                output_dir=str(gap_dir),
                sample_fps=self.sample_fps,
                max_frames=req.target_frames,
                start_sec=req.start_sec,
                end_sec=req.end_sec,
                stream_index=stream_index,
            )

            for fs in result.selected_frames:
                info = BridgeFrameInfo(
                    path=fs.path,
                    timestamp=fs.timestamp,
                    sharpness=fs.sharpness,
                    flow_from_prev=fs.flow_from_prev,
                    gap_index=req.gap_index,
                )

                # Optionally reframe fisheye → perspective
                if reframe and reframer is not None and view_config is not None:
                    reframed = self._reframe_bridge_frame(
                        fs.path, reframer, view_config, gap_dir,
                    )
                    info.reframed_paths = reframed
                    total_reframed += len(reframed)

                all_frames.append(info)

            if progress_callback:
                progress_callback(
                    f"  → extracted {len(result.selected_frames)} frames"
                )

        result = BridgeResult(
            requests=requests,
            frames=all_frames,
            total_extracted=len(all_frames),
            total_reframed=total_reframed,
            output_dir=str(out_path),
        )

        # Save report
        result.save(str(out_path / "bridge_report.json"))

        if progress_callback:
            progress_callback(result.summary())

        return result

    def extract_for_gap(
        self,
        gap: SpatialGap,
        gap_index: int,
        video_path: str,
        output_dir: str,
        video_duration: Optional[float] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[BridgeFrameInfo]:
        """Extract bridge frames for a single gap.

        Convenience method for interactive use or GUI integration.
        """
        if gap.estimated_timestamps is None:
            return []

        if video_duration is None:
            video_duration = self._probe_duration(video_path)

        # Create a minimal report with just this gap
        mini_report = GapReport(
            num_components=0,
            num_cameras_aligned=0,
            num_cameras_total=0,
            gaps=[gap],
            failed_images=[],
            camera_positions=[],
            component_labels=np.zeros(0),
        )

        # Override gap index in the plan
        requests = self.plan(mini_report, video_path, video_duration)
        if not requests:
            return []
        requests[0].gap_index = gap_index

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        selector = MotionSelector(
            min_sharpness=self.min_sharpness,
            target_flow=self.target_flow,
        )

        result = selector.select_from_video(
            video_path=video_path,
            output_dir=str(out_path),
            sample_fps=self.sample_fps,
            max_frames=requests[0].target_frames,
            start_sec=requests[0].start_sec,
            end_sec=requests[0].end_sec,
        )

        frames = []
        for fs in result.selected_frames:
            frames.append(BridgeFrameInfo(
                path=fs.path,
                timestamp=fs.timestamp,
                sharpness=fs.sharpness,
                flow_from_prev=fs.flow_from_prev,
                gap_index=gap_index,
            ))

        return frames

    def _reframe_bridge_frame(
        self,
        frame_path: str,
        reframer,
        view_config,
        output_dir: Path,
    ) -> List[str]:
        """Reframe a single bridge frame through fisheye → perspective.

        Assumes the bridge frame is a fisheye image from one lens.
        Extracts all views configured for that lens.
        """
        img = cv2.imread(str(frame_path))
        if img is None:
            return []

        stem = Path(frame_path).stem
        reframed_dir = output_dir / "reframed"
        reframed_dir.mkdir(exist_ok=True)

        saved = []
        for view in view_config.views:
            crop = reframer.extract_view(img, view, view_config.crop_size)
            if crop is None:
                continue

            # Check content coverage
            h, w = crop.shape[:2]
            nonblack = np.count_nonzero(crop.sum(axis=2) > 0)
            if nonblack / (h * w) < 0.5:
                continue  # skip mostly-empty crops

            out_name = f"{stem}_{view.name}.jpg"
            out_path = reframed_dir / out_name
            cv2.imwrite(
                str(out_path), crop,
                [cv2.IMWRITE_JPEG_QUALITY, view_config.quality],
            )
            saved.append(str(out_path))

        return saved

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        """Get video duration in seconds using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0:
            raise RuntimeError(f"Cannot determine FPS for: {video_path}")
        return total / fps


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract bridge frames to fill reconstruction gaps"
    )
    parser.add_argument("gaps", help="Gap report JSON (from gap_detector)")
    parser.add_argument("video", help="Source video file")
    parser.add_argument("output", help="Output directory for bridge frames")
    parser.add_argument("--reframe", action="store_true",
                        help="Reframe fisheye to perspective crops")
    parser.add_argument("--preset", default="osv-full-f90-dual-26",
                        help="Fisheye reframe preset (default: osv-full-f90-dual-26)")
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="Video sampling rate in gap regions (default: 5)")
    parser.add_argument("--min-sharpness", type=float, default=50.0,
                        help="Minimum sharpness for bridge frames (default: 50)")
    parser.add_argument("--target-flow", type=float, default=8.0,
                        help="Target optical flow between selections (default: 8)")
    parser.add_argument("--padding", type=float, default=2.0,
                        help="Extra seconds around gap timestamps (default: 2)")
    parser.add_argument("--disconnected-frames", type=int, default=8,
                        help="Target frames per disconnected gap (default: 8)")
    parser.add_argument("--sparse-frames", type=int, default=3,
                        help="Target frames per sparse gap (default: 3)")

    args = parser.parse_args()

    try:
        # Load gap report
        report = GapReport.load(args.gaps)
        print(f"Loaded gap report: {len(report.gaps)} gaps, "
              f"{report.num_cameras_aligned} cameras")

        extractor = BridgeExtractor(
            sample_fps=args.sample_fps,
            min_sharpness=args.min_sharpness,
            target_flow=args.target_flow,
            padding_sec=args.padding,
            frames_per_disconnected=args.disconnected_frames,
            frames_per_sparse=args.sparse_frames,
        )

        # Set up reframing if requested
        reframer = None
        view_config = None
        if args.reframe:
            from .fisheye_reframer import (
                FisheyeReframer, FISHEYE_PRESETS,
            )
            reframer = FisheyeReframer.with_defaults()
            view_config = FISHEYE_PRESETS.get(args.preset)
            if view_config is None:
                print(f"Unknown preset: {args.preset}")
                print(f"Available: {', '.join(FISHEYE_PRESETS.keys())}")
                return 1

        result = extractor.extract(
            gap_report=report,
            video_path=args.video,
            output_dir=args.output,
            reframe=args.reframe,
            reframer=reframer,
            view_config=view_config,
            progress_callback=print,
        )

        print(f"\n{result.summary()}")
        print(f"Report saved to {Path(args.output) / 'bridge_report.json'}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

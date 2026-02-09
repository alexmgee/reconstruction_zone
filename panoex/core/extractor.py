"""
Frame Extractor Module

Extract frames from video with multiple selection modes.
"""

import subprocess
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List


class ExtractionMode(Enum):
    """Frame extraction modes."""
    FIXED = "fixed"          # Extract at exact intervals
    ADAPTIVE = "adaptive"    # Interval with max gap enforcement
    SCENE = "scene"          # Scene detection based


@dataclass
class ExtractionConfig:
    """Configuration for frame extraction."""
    interval: float = 2.0        # Base extraction interval in seconds
    max_gap: float = 3.0         # Maximum allowed gap (for adaptive mode)
    mode: ExtractionMode = ExtractionMode.FIXED
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    quality: int = 95            # JPEG quality (1-100)
    output_format: str = "jpg"   # jpg or png
    scene_threshold: float = 0.3  # For scene detection mode


@dataclass
class ExtractionResult:
    """Result of frame extraction."""
    success: bool
    frame_count: int
    output_dir: str
    frames: List[str]
    error: Optional[str] = None


class FrameExtractor:
    """Extract frames from video files using ffmpeg."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def extract(
        self,
        video_path: str,
        output_dir: str,
        config: Optional[ExtractionConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        dry_run: bool = False
    ) -> ExtractionResult:
        """
        Extract frames from video.

        Args:
            video_path: Path to input video
            output_dir: Directory for extracted frames
            config: Extraction configuration
            progress_callback: Called with (current, total, message)
            dry_run: If True, only print command without executing

        Returns:
            ExtractionResult with extraction details
        """
        if config is None:
            config = ExtractionConfig()

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            return ExtractionResult(
                success=False,
                frame_count=0,
                output_dir=str(output_dir),
                frames=[],
                error=f"Video not found: {video_path}"
            )

        # Create output directory
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command based on mode
        if config.mode == ExtractionMode.FIXED:
            cmd = self._build_fixed_command(video_path, output_dir, config)
        elif config.mode == ExtractionMode.SCENE:
            cmd = self._build_scene_command(video_path, output_dir, config)
        elif config.mode == ExtractionMode.ADAPTIVE:
            # Adaptive mode uses fixed extraction then fills gaps
            cmd = self._build_fixed_command(video_path, output_dir, config)
        else:
            cmd = self._build_fixed_command(video_path, output_dir, config)

        if dry_run:
            print(f"Command: {' '.join(cmd)}")
            return ExtractionResult(
                success=True,
                frame_count=0,
                output_dir=str(output_dir),
                frames=[],
                error="Dry run - command not executed"
            )

        # Run extraction
        try:
            if progress_callback:
                progress_callback(0, 100, "Starting extraction...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return ExtractionResult(
                    success=False,
                    frame_count=0,
                    output_dir=str(output_dir),
                    frames=[],
                    error=result.stderr
                )

            # Count extracted frames
            ext = config.output_format
            frames = sorted([f.name for f in output_dir.glob(f"*.{ext}")])

            if progress_callback:
                progress_callback(100, 100, f"Extracted {len(frames)} frames")

            return ExtractionResult(
                success=True,
                frame_count=len(frames),
                output_dir=str(output_dir),
                frames=frames
            )

        except Exception as e:
            return ExtractionResult(
                success=False,
                frame_count=0,
                output_dir=str(output_dir),
                frames=[],
                error=str(e)
            )

    def _build_fixed_command(
        self,
        video_path: Path,
        output_dir: Path,
        config: ExtractionConfig
    ) -> List[str]:
        """Build ffmpeg command for fixed interval extraction."""
        cmd = [self.ffmpeg_path, "-y"]

        # Input options (time range)
        if config.start_sec is not None:
            cmd.extend(["-ss", str(config.start_sec)])

        cmd.extend(["-i", str(video_path)])

        if config.end_sec is not None:
            cmd.extend(["-to", str(config.end_sec)])

        # Filter: fps extraction
        fps = 1.0 / config.interval
        cmd.extend(["-vf", f"fps={fps}"])

        # Output quality
        if config.output_format == "jpg":
            # qscale: 2 = high quality, maps roughly to quality 95
            qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
            cmd.extend(["-qscale:v", str(qscale)])
        elif config.output_format == "png":
            cmd.extend(["-compression_level", "6"])

        # Output pattern
        output_pattern = str(output_dir / f"%05d.{config.output_format}")
        cmd.append(output_pattern)

        return cmd

    def _build_scene_command(
        self,
        video_path: Path,
        output_dir: Path,
        config: ExtractionConfig
    ) -> List[str]:
        """Build ffmpeg command for scene detection extraction."""
        cmd = [self.ffmpeg_path, "-y"]

        # Input options
        if config.start_sec is not None:
            cmd.extend(["-ss", str(config.start_sec)])

        cmd.extend(["-i", str(video_path)])

        if config.end_sec is not None:
            cmd.extend(["-to", str(config.end_sec)])

        # Scene detection filter with minimum interval
        # select='gt(scene,0.3)*gte(t-prev_selected_t,2)'
        threshold = config.scene_threshold
        interval = config.interval
        filter_str = f"select='gt(scene,{threshold})*gte(t-prev_selected_t,{interval})'"

        cmd.extend(["-vf", filter_str])
        cmd.extend(["-vsync", "vfr"])

        # Output quality
        if config.output_format == "jpg":
            qscale = max(1, min(31, int(32 - (config.quality / 100 * 30))))
            cmd.extend(["-qscale:v", str(qscale)])
        elif config.output_format == "png":
            cmd.extend(["-compression_level", "6"])

        # Output pattern
        output_pattern = str(output_dir / f"%05d.{config.output_format}")
        cmd.append(output_pattern)

        return cmd

    def estimate_frames(self, duration: float, config: ExtractionConfig) -> int:
        """Estimate number of frames that will be extracted."""
        effective_duration = duration

        if config.start_sec is not None:
            effective_duration -= config.start_sec
        if config.end_sec is not None:
            effective_duration = min(effective_duration, config.end_sec - (config.start_sec or 0))

        return max(1, int(effective_duration / config.interval))


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--interval", "-i", type=float, default=2.0,
                        help="Extraction interval in seconds (default: 2.0)")
    parser.add_argument("--mode", "-m", choices=["fixed", "scene", "adaptive"],
                        default="fixed", help="Extraction mode")
    parser.add_argument("--start", "-ss", type=float, help="Start time in seconds")
    parser.add_argument("--end", "-to", type=float, help="End time in seconds")
    parser.add_argument("--quality", "-q", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--format", "-f", choices=["jpg", "png"], default="jpg",
                        help="Output format (default: jpg)")
    parser.add_argument("--scene-threshold", type=float, default=0.3,
                        help="Scene detection threshold (default: 0.3)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show command without executing")

    args = parser.parse_args()

    config = ExtractionConfig(
        interval=args.interval,
        mode=ExtractionMode(args.mode),
        start_sec=args.start,
        end_sec=args.end,
        quality=args.quality,
        output_format=args.format,
        scene_threshold=args.scene_threshold,
    )

    extractor = FrameExtractor()

    def progress(current, total, msg):
        print(f"[{current}/{total}] {msg}")

    result = extractor.extract(
        args.video,
        args.output,
        config,
        progress_callback=progress,
        dry_run=args.dry_run
    )

    if result.success:
        print(f"Extracted {result.frame_count} frames to {result.output_dir}")
        return 0
    else:
        print(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    exit(main())

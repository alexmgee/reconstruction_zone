"""
Video Analyzer Module

Analyze video files to extract metadata and recommend extraction parameters.
"""

import json
import subprocess
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class VideoInfo:
    """Video metadata and analysis results."""
    path: str
    filename: str
    format: str
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    bitrate: Optional[int] = None
    pixel_format: Optional[str] = None

    # Derived properties
    is_equirectangular: bool = False
    is_log_format: bool = False
    detected_log_type: Optional[str] = None

    # Recommendations
    recommended_interval: float = 2.0
    recommended_lut: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "format": self.format,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "bitrate": self.bitrate,
            "pixel_format": self.pixel_format,
            "is_equirectangular": self.is_equirectangular,
            "is_log_format": self.is_log_format,
            "detected_log_type": self.detected_log_type,
            "recommended_interval": self.recommended_interval,
            "recommended_lut": self.recommended_lut,
        }


# Known log formats and their recommended LUTs
LOG_FORMATS = {
    "dlog": {
        "name": "D-Log M",
        "cameras": ["DJI"],
        "lut": "DJI_DLog_M_to_Rec709.cube",
    },
    "ilog": {
        "name": "I-Log",
        "cameras": ["Insta360"],
        "lut": "Insta360_ILog_to_Rec709.cube",
    },
    "protune": {
        "name": "Protune Flat",
        "cameras": ["GoPro"],
        "lut": "GoPro_Protune_to_Rec709.cube",
    },
    "vlog": {
        "name": "V-Log",
        "cameras": ["Panasonic"],
        "lut": "VLog_to_Rec709.cube",
    },
}


class VideoAnalyzer:
    """Analyze video files for 360° processing."""

    def __init__(self, ffprobe_path: str = "ffprobe"):
        self.ffprobe_path = ffprobe_path

    def analyze(self, video_path: str) -> VideoInfo:
        """
        Analyze a video file and return metadata with recommendations.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo with metadata and recommendations
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get raw metadata from ffprobe
        raw_data = self._run_ffprobe(video_path)

        # Parse into structured format
        info = self._parse_metadata(raw_data, path)

        # Analyze for 360/log characteristics
        self._detect_360(info)
        self._detect_log_format(info, path)

        # Generate recommendations
        self._generate_recommendations(info)

        return info

    def _run_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON output."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames,codec_name,pix_fmt,bit_rate",
            "-show_entries", "format=duration,size,bit_rate,format_name",
            "-of", "json",
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")

    def _parse_metadata(self, data: Dict[str, Any], path: Path) -> VideoInfo:
        """Parse ffprobe output into VideoInfo."""
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        # Parse frame rate (can be "30/1" or "29.97")
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        # Parse duration
        duration = float(stream.get("duration", 0) or fmt.get("duration", 0))

        # Parse frame count
        nb_frames = stream.get("nb_frames")
        if nb_frames:
            frame_count = int(nb_frames)
        else:
            frame_count = int(duration * fps) if duration else 0

        # Parse bitrate
        bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
        if bitrate:
            bitrate = int(bitrate)

        return VideoInfo(
            path=str(path.absolute()),
            filename=path.name,
            format=fmt.get("format_name", path.suffix[1:]),
            codec=stream.get("codec_name", "unknown"),
            width=int(stream.get("width", 0)),
            height=int(stream.get("height", 0)),
            fps=fps,
            duration_seconds=duration,
            frame_count=frame_count,
            bitrate=bitrate,
            pixel_format=stream.get("pix_fmt"),
        )

    def _detect_360(self, info: VideoInfo):
        """Detect if video is equirectangular 360°."""
        if info.width > 0 and info.height > 0:
            aspect_ratio = info.width / info.height
            # 2:1 aspect ratio indicates equirectangular
            info.is_equirectangular = 1.9 < aspect_ratio < 2.1

    def _detect_log_format(self, info: VideoInfo, path: Path):
        """Detect if video uses log color profile."""
        filename_lower = path.name.lower()

        # Check filename for hints
        log_hints = {
            "dlog": ["dlog", "d-log", "dji"],
            "ilog": ["ilog", "i-log", "insta360", "insv"],
            "protune": ["protune", "gopro", "flat"],
            "vlog": ["vlog", "v-log"],
        }

        for log_type, hints in log_hints.items():
            if any(hint in filename_lower for hint in hints):
                info.is_log_format = True
                info.detected_log_type = log_type
                info.recommended_lut = LOG_FORMATS[log_type]["lut"]
                return

        # Check file extension for camera-specific formats
        ext = path.suffix.lower()
        if ext == ".insv":
            info.is_log_format = True
            info.detected_log_type = "ilog"
            info.recommended_lut = LOG_FORMATS["ilog"]["lut"]
        elif ext == ".360":
            # GoPro MAX format
            info.is_log_format = True
            info.detected_log_type = "protune"
            info.recommended_lut = LOG_FORMATS["protune"]["lut"]

    def _generate_recommendations(self, info: VideoInfo):
        """Generate extraction recommendations based on video properties."""
        # Base interval on resolution and typical walking speed
        if info.is_equirectangular:
            # Higher res = can use longer intervals
            if info.width >= 7680:  # 8K
                info.recommended_interval = 2.0
            elif info.width >= 5760:  # 6K
                info.recommended_interval = 1.5
            else:  # 4K and below
                info.recommended_interval = 1.0
        else:
            # Non-360 video, use shorter intervals
            info.recommended_interval = 0.5

    def get_duration_formatted(self, info: VideoInfo) -> str:
        """Format duration as HH:MM:SS."""
        total_seconds = int(info.duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def estimate_frame_count(self, info: VideoInfo, interval: float) -> int:
        """Estimate number of frames that will be extracted."""
        return int(info.duration_seconds / interval)


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze video files for 360° processing")
    parser.add_argument("video", help="Video file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = VideoAnalyzer()

    try:
        info = analyzer.analyze(args.video)

        if args.json:
            print(json.dumps(info.to_dict(), indent=2))
        else:
            print(f"File: {info.filename}")
            print(f"Format: {info.format} ({info.codec})")
            print(f"Resolution: {info.width}x{info.height}")
            print(f"FPS: {info.fps:.2f}")
            print(f"Duration: {analyzer.get_duration_formatted(info)} ({info.duration_seconds:.1f}s)")
            print(f"Frames: {info.frame_count}")
            print(f"Equirectangular: {info.is_equirectangular}")
            print(f"Log Format: {info.detected_log_type or 'None detected'}")
            print(f"Recommended Interval: {info.recommended_interval}s")
            if info.recommended_lut:
                print(f"Recommended LUT: {info.recommended_lut}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

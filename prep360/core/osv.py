"""
Dual Fisheye Container Handler

Parse and extract from dual-fisheye 360 containers (DJI .osv, Insta360 .insv).

DJI OSV format (confirmed via ffprobe on real files):
    Stream 0: HEVC 3840x3840 50fps 10-bit  — fisheye lens 0 (back)
    Stream 1: HEVC 3840x3840 50fps 10-bit  — fisheye lens 1 (front)
    Stream 2: AAC stereo 48kHz             — audio
    Stream 3: djmd data                    — per-frame camera metadata (lens 0)
    Stream 4: djmd data                    — per-frame camera metadata (lens 1)
    Stream 5: dbgi data                    — gyro/debug data (lens 0)
    Stream 6: dbgi data                    — gyro/debug data (lens 1)
    Stream 7: MJPEG 688x344               — thumbnail (attached_pic)
    Container: MOV/MP4, encoder tag "Osmo 360".

Insta360 INSV format (newer dual-stream variant):
    Stream 0: HEVC 3840x3840              — fisheye lens 0 (back)
    Stream 1: HEVC 3840x3840              — fisheye lens 1 (front)
    Stream 2: AAC                          — audio
    Container: MOV/MP4, handler "INS.HVC". No encoder tag.

Both formats: stream 0 = back (away from operator),
stream 1 = front (toward operator).
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

_SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


@dataclass
class OSVStreamInfo:
    """Metadata for a single stream within an OSV file."""
    index: int
    codec_type: str         # "video", "audio", "data"
    codec_name: str         # "hevc", "aac", "mjpeg", or codec_tag_string for data
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bit_depth: int = 0
    bitrate: int = 0
    handler_name: str = ""
    is_default: bool = False
    is_thumbnail: bool = False


@dataclass
class OSVInfo:
    """Parsed metadata from an OSV file."""
    path: str
    filename: str
    duration: float                 # seconds
    total_size: int                 # bytes
    encoder: str                    # e.g. "Osmo 360"
    creation_time: str

    # Fisheye streams
    front_stream: int               # stream index for front lens
    back_stream: int                # stream index for back lens
    width: int                      # per-stream width (3840)
    height: int                     # per-stream height (3840)
    fps: float                      # frame rate (50)
    codec: str                      # "hevc"
    bit_depth: int                  # 10
    frame_count: int                # frames per stream

    # Data streams
    has_metadata: bool = False      # djmd tracks present
    has_gyro: bool = False          # dbgi tracks present
    has_audio: bool = False
    has_thumbnail: bool = False

    # All streams for reference
    streams: List[OSVStreamInfo] = field(default_factory=list)

    @property
    def total_bitrate_mbps(self) -> float:
        return (self.total_size * 8) / (self.duration * 1_000_000) if self.duration else 0

    def summary(self) -> str:
        lines = [
            f"File: {self.filename}",
            f"Encoder: {self.encoder}",
            f"Duration: {self.duration:.1f}s ({self._format_duration()})",
            f"Fisheye: {self.width}x{self.height} {self.fps:.0f}fps {self.bit_depth}-bit {self.codec}",
            f"Streams: front={self.front_stream}, back={self.back_stream}",
            f"Frames: {self.frame_count} per stream",
            f"Size: {self.total_size / (1024**3):.2f} GB ({self.total_bitrate_mbps:.0f} Mbps)",
        ]
        extras = []
        if self.has_metadata:
            extras.append("metadata")
        if self.has_gyro:
            extras.append("gyro")
        if self.has_audio:
            extras.append("audio")
        if extras:
            lines.append(f"Data: {', '.join(extras)}")
        return "\n".join(lines)

    def _format_duration(self) -> str:
        total = int(self.duration)
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class OSVHandler:
    """Parse and extract from DJI Osmo 360 OSV files."""

    def __init__(
        self,
        ffprobe_path: str = "ffprobe",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.ffprobe_path = ffprobe_path
        self.ffmpeg_path = ffmpeg_path

    def probe(self, osv_path: str) -> OSVInfo:
        """Probe an OSV file and return its stream layout.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file doesn't look like a valid OSV.
            RuntimeError: If ffprobe fails.
        """
        path = Path(osv_path)
        if not path.exists():
            raise FileNotFoundError(f"OSV file not found: {osv_path}")

        raw = self._run_ffprobe(osv_path)
        return self._parse_osv(raw, path)

    def demux_streams(
        self,
        osv_path: str,
        output_dir: str,
        streams: str = "both",
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract raw fisheye video streams from OSV container.

        Uses stream copy — no re-encoding, nearly instant.

        Args:
            osv_path: Path to OSV file.
            output_dir: Directory for extracted MP4 files.
            streams: "front", "back", or "both".

        Returns:
            (front_path, back_path) — paths to extracted MP4 files.
            Either may be None if not requested.
        """
        info = self.probe(osv_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = Path(osv_path).stem
        front_path = None
        back_path = None

        if streams in ("front", "both"):
            front_path = str(out / f"{stem}_front.mp4")
            self._demux_one_stream(osv_path, info.front_stream, front_path)

        if streams in ("back", "both"):
            back_path = str(out / f"{stem}_back.mp4")
            self._demux_one_stream(osv_path, info.back_stream, back_path)

        return front_path, back_path

    def extract_frames(
        self,
        osv_path: str,
        output_dir: str,
        interval: float = 2.0,
        streams: str = "both",
        start_sec: float = 0,
        end_sec: Optional[float] = None,
        quality: int = 95,
        output_format: str = "jpg",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, List[str]]:
        """Extract fisheye frames directly from OSV.

        Filenames encode stream and frame number:
            front/00001.jpg, front/00002.jpg, ...
            back/00001.jpg, back/00002.jpg, ...

        Args:
            osv_path: Path to OSV file.
            output_dir: Base output directory.
            interval: Seconds between frames.
            streams: "front", "back", or "both".
            start_sec: Start time in seconds.
            end_sec: End time in seconds (None = entire video).
            quality: JPEG quality 1-100.
            output_format: "jpg" or "png".
            progress_callback: Called with (stream_name, current, total).

        Returns:
            Dict with 'front' and/or 'back' keys mapping to lists of
            extracted frame paths.
        """
        info = self.probe(osv_path)
        result = {}

        stream_map = {}
        if streams in ("front", "both"):
            stream_map["front"] = info.front_stream
        if streams in ("back", "both"):
            stream_map["back"] = info.back_stream

        for name, stream_idx in stream_map.items():
            stream_dir = Path(output_dir) / name
            stream_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback(name, 0, 100)

            frames = self._extract_frames_from_stream(
                osv_path, stream_idx, str(stream_dir),
                interval, start_sec, end_sec,
                quality, output_format,
            )
            result[name] = frames

            if progress_callback:
                progress_callback(name, 100, 100)

        return result

    def extract_thumbnail(self, osv_path: str, output_path: str) -> bool:
        """Extract the embedded thumbnail image."""
        info = self.probe(osv_path)
        thumb_streams = [
            s for s in info.streams
            if s.is_thumbnail
        ]
        if not thumb_streams:
            return False

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", osv_path,
            "-map", f"0:{thumb_streams[0].index}",
            "-frames:v", "1",
            "-update", "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        return result.returncode == 0

    # --- Internal methods ---

    def _run_ffprobe(self, path: str) -> Dict[str, Any]:
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, **_SUBPROCESS_FLAGS)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")

    def _parse_osv(self, raw: Dict[str, Any], path: Path) -> OSVInfo:
        fmt = raw.get("format", {})
        raw_streams = raw.get("streams", [])

        # Validate it's a dual-fisheye container (DJI .osv or Insta360 .insv)
        encoder = fmt.get("tags", {}).get("encoder", "")
        suffix = path.suffix.lower()
        known_suffix = suffix in (".osv", ".insv")
        known_encoder = "osmo" in encoder.lower() or "360" in encoder.lower()
        if not known_suffix and not known_encoder:
            format_name = fmt.get("format_name", "")
            raise ValueError(
                f"Not a recognized dual-fisheye file: encoder='{encoder}', "
                f"format='{format_name}', suffix='{suffix}'"
            )

        # Parse all streams
        parsed_streams = []
        hevc_streams = []

        for s in raw_streams:
            codec_type = s.get("codec_type", "unknown")
            codec_name = s.get("codec_name", s.get("codec_tag_string", "unknown"))
            index = s.get("index", 0)

            tags = s.get("tags", {})
            disp = s.get("disposition", {})

            si = OSVStreamInfo(
                index=index,
                codec_type=codec_type,
                codec_name=codec_name,
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                handler_name=tags.get("handler_name", ""),
                is_default=bool(disp.get("default", 0)),
                is_thumbnail=bool(disp.get("attached_pic", 0)),
            )

            # Parse fps for video streams
            if codec_type == "video":
                fps_str = s.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    si.fps = float(num) / float(den) if float(den) else 0
                else:
                    si.fps = float(fps_str)

            # Parse bit depth
            pix_fmt = s.get("pix_fmt", "")
            if "10" in pix_fmt:
                si.bit_depth = 10
            elif "12" in pix_fmt:
                si.bit_depth = 12
            elif pix_fmt:
                si.bit_depth = 8

            # Parse bitrate
            br = s.get("bit_rate")
            if br:
                si.bitrate = int(br)

            parsed_streams.append(si)

            # Track HEVC video streams (the fisheye pair)
            if codec_type == "video" and codec_name == "hevc":
                hevc_streams.append(si)

        if len(hevc_streams) < 2:
            raise ValueError(
                f"Expected 2 HEVC video streams in OSV, found {len(hevc_streams)}"
            )

        # Identify front/back. DuckbillStudio convention and confirmed
        # by visual inspection: stream 0 = back (away from operator),
        # stream 1 = front (toward operator). The back stream typically
        # has disposition.default=1 (it's the "primary" video stream).
        # We assign by stream index — lower index = back, higher = front.
        back_stream = hevc_streams[0]
        front_stream = hevc_streams[1]

        # Detect metadata and gyro tracks
        has_metadata = any(
            s.codec_name == "djmd" or "CAM meta" in s.handler_name
            for s in parsed_streams
        )
        has_gyro = any(
            s.codec_name == "dbgi" or "dbgi" in s.handler_name
            for s in parsed_streams
        )
        has_audio = any(s.codec_type == "audio" for s in parsed_streams)
        has_thumbnail = any(s.is_thumbnail for s in parsed_streams)

        duration = float(fmt.get("duration", 0))
        size = int(fmt.get("size", 0))
        creation_time = fmt.get("tags", {}).get("creation_time", "")

        return OSVInfo(
            path=str(path.absolute()),
            filename=path.name,
            duration=duration,
            total_size=size,
            encoder=encoder,
            creation_time=creation_time,
            front_stream=front_stream.index,
            back_stream=back_stream.index,
            width=front_stream.width,
            height=front_stream.height,
            fps=front_stream.fps,
            codec=front_stream.codec_name,
            bit_depth=front_stream.bit_depth,
            frame_count=int(duration * front_stream.fps) if duration else 0,
            has_metadata=has_metadata,
            has_gyro=has_gyro,
            has_audio=has_audio,
            has_thumbnail=has_thumbnail,
            streams=parsed_streams,
        )

    def _demux_one_stream(self, osv_path: str, stream_idx: int, output_path: str):
        """Demux a single video stream using stream copy."""
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", osv_path,
            "-map", f"0:{stream_idx}",
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to demux stream {stream_idx}: {result.stderr}"
            )

    def _extract_frames_from_stream(
        self,
        osv_path: str,
        stream_idx: int,
        output_dir: str,
        interval: float,
        start_sec: float,
        end_sec: Optional[float],
        quality: int,
        output_format: str,
    ) -> List[str]:
        """Extract frames from a single video stream."""
        cmd = [self.ffmpeg_path, "-y"]

        if start_sec > 0:
            cmd.extend(["-ss", str(start_sec)])

        cmd.extend(["-i", osv_path])

        if end_sec is not None:
            cmd.extend(["-to", str(end_sec)])

        # Select specific video stream and apply fps filter
        fps = 1.0 / interval
        cmd.extend([
            "-map", f"0:{stream_idx}",
            "-vf", f"fps={fps}",
        ])

        # Quality settings
        if output_format == "jpg":
            qscale = max(1, min(31, int(32 - (quality / 100 * 30))))
            cmd.extend(["-qscale:v", str(qscale)])
        elif output_format == "png":
            cmd.extend(["-compression_level", "6"])

        # Output pattern
        output_pattern = str(Path(output_dir) / f"%05d.{output_format}")
        cmd.append(output_pattern)

        result = subprocess.run(cmd, capture_output=True, text=True, **_SUBPROCESS_FLAGS)
        if result.returncode != 0:
            raise RuntimeError(
                f"Frame extraction failed for stream {stream_idx}: "
                f"{result.stderr[:500]}"
            )

        # Collect extracted frames
        out_path = Path(output_dir)
        frames = sorted(str(f) for f in out_path.glob(f"*.{output_format}"))
        return frames


# --- CLI interface ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DJI Osmo 360 OSV file handler"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Probe command
    probe_p = sub.add_parser("probe", help="Probe OSV file structure")
    probe_p.add_argument("osv", help="OSV file to probe")
    probe_p.add_argument("--json", action="store_true", help="Output raw JSON")

    # Demux command
    demux_p = sub.add_parser("demux", help="Extract fisheye video streams")
    demux_p.add_argument("osv", help="OSV file")
    demux_p.add_argument("output", help="Output directory")
    demux_p.add_argument(
        "--streams", choices=["front", "back", "both"],
        default="both", help="Which streams to extract"
    )

    # Extract frames command
    extract_p = sub.add_parser("extract", help="Extract fisheye frames")
    extract_p.add_argument("osv", help="OSV file")
    extract_p.add_argument("output", help="Output directory")
    extract_p.add_argument(
        "--interval", "-i", type=float, default=2.0,
        help="Seconds between frames (default: 2.0)"
    )
    extract_p.add_argument(
        "--streams", choices=["front", "back", "both"],
        default="both", help="Which streams to extract"
    )
    extract_p.add_argument("--start", type=float, default=0, help="Start time (sec)")
    extract_p.add_argument("--end", type=float, default=None, help="End time (sec)")
    extract_p.add_argument(
        "--quality", "-q", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    extract_p.add_argument(
        "--format", "-f", choices=["jpg", "png"], default="jpg",
        help="Output format"
    )

    args = parser.parse_args()
    handler = OSVHandler()

    try:
        if args.command == "probe":
            info = handler.probe(args.osv)
            if args.json:
                import dataclasses
                d = dataclasses.asdict(info)
                # Remove streams detail for cleaner output
                d.pop("streams", None)
                print(json.dumps(d, indent=2, default=str))
            else:
                print(info.summary())

        elif args.command == "demux":
            front, back = handler.demux_streams(args.osv, args.output, args.streams)
            if front:
                print(f"Front: {front}")
            if back:
                print(f"Back:  {back}")

        elif args.command == "extract":
            def progress(stream, cur, total):
                print(f"  [{stream}] {cur}/{total}")

            result = handler.extract_frames(
                args.osv, args.output,
                interval=args.interval,
                streams=args.streams,
                start_sec=args.start,
                end_sec=args.end,
                quality=args.quality,
                output_format=args.format,
                progress_callback=progress,
            )
            for stream_name, frames in result.items():
                print(f"{stream_name}: {len(frames)} frames")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

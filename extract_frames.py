#!/usr/bin/env python3
"""
Frame extraction wrapper with common presets for photogrammetry/3DGS workflows.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "photogrammetry": {
        "description": "2 fps, high quality JPEG, good for walking-speed capture",
        "vf": "fps=2",
        "quality": 2,
        "ext": "jpg"
    },
    "photogrammetry-hq": {
        "description": "2 fps, PNG lossless, maximum quality",
        "vf": "fps=2",
        "quality": None,
        "ext": "png"
    },
    "3dgs": {
        "description": "Scene detection + 2fps cap, for 3D Gaussian Splatting",
        "vf": "select='gt(scene,0.2)+gte(t-prev_selected_t,0.5)',fps=4",
        "quality": 2,
        "ext": "jpg",
        "vsync": "vfr"
    },
    "scene": {
        "description": "Scene detection only, captures key moments",
        "vf": "select='gt(scene,0.3)'",
        "quality": 2,
        "ext": "jpg",
        "vsync": "vfr"
    },
    "timelapse": {
        "description": "1 frame every 5 seconds",
        "vf": "fps=1/5",
        "quality": 2,
        "ext": "jpg"
    },
    "all": {
        "description": "Extract every frame (warning: large output)",
        "vf": None,
        "quality": 2,
        "ext": "jpg"
    },
    "sparse": {
        "description": "1 fps, for long videos or fast motion",
        "vf": "fps=1",
        "quality": 2,
        "ext": "jpg"
    },
    "dense": {
        "description": "5 fps, for slow capture or detailed coverage",
        "vf": "fps=5",
        "quality": 2,
        "ext": "jpg"
    }
}


def extract_frames(
    video_path: str,
    output_dir: str,
    preset: str = "photogrammetry",
    fps: float = None,
    every_n: int = None,
    start_time: str = None,
    end_time: str = None,
    quality: int = None,
    output_format: str = None,
    scale: str = None,
    dry_run: bool = False
) -> int:
    """
    Extract frames from video.
    
    Returns frame count on success, -1 on error.
    """
    # Get preset settings
    settings = PRESETS.get(preset, PRESETS["photogrammetry"]).copy()
    
    # Override with explicit options
    if fps:
        settings["vf"] = f"fps={fps}"
    if every_n:
        settings["vf"] = f"select=not(mod(n\\,{every_n}))"
        settings["vsync"] = "vfr"
    if quality:
        settings["quality"] = quality
    if output_format:
        settings["ext"] = output_format
    
    # Build filter string
    filters = []
    if settings.get("vf"):
        filters.append(settings["vf"])
    if scale:
        filters.append(f"scale={scale}")
    
    vf_string = ",".join(filters) if filters else None
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    
    # Input options
    if start_time:
        cmd.extend(["-ss", start_time])
    cmd.extend(["-i", video_path])
    if end_time:
        cmd.extend(["-to", end_time])
    
    # Filter
    if vf_string:
        cmd.extend(["-vf", vf_string])
    
    # Output options
    if settings.get("vsync"):
        cmd.extend(["-vsync", settings["vsync"]])
    
    if settings["ext"] == "jpg" and settings.get("quality"):
        cmd.extend(["-qscale:v", str(settings["quality"])])
    elif settings["ext"] == "png":
        cmd.extend(["-compression_level", "6"])
    
    # Output path
    output_pattern = str(Path(output_dir) / f"%05d.{settings['ext']}")
    cmd.append(output_pattern)
    
    # Print command
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("(dry run - not executing)")
        return 0
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run extraction
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return -1
    
    # Count output frames
    ext = settings["ext"]
    frames = list(Path(output_dir).glob(f"*.{ext}"))
    print(f"Extracted {len(frames)} frames to {output_dir}")
    
    return len(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video with presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  photogrammetry    2 fps, high quality JPEG (default)
  photogrammetry-hq 2 fps, PNG lossless
  3dgs              Scene detection + fps cap, for Gaussian Splatting
  scene             Scene detection only
  timelapse         1 frame every 5 seconds
  sparse            1 fps, for long/fast videos
  dense             5 fps, for detailed coverage
  all               Every frame (warning: large output)

Examples:
  python extract_frames.py video.mp4 ./frames
  python extract_frames.py video.mp4 ./frames --preset 3dgs
  python extract_frames.py video.mp4 ./frames --fps 3
  python extract_frames.py video.mp4 ./frames --every 60
  python extract_frames.py video.mp4 ./frames --start 00:01:00 --end 00:02:00
        """
    )
    
    parser.add_argument("video", nargs="?", help="Input video file")
    parser.add_argument("output", nargs="?", help="Output directory for frames")
    parser.add_argument("--preset", "-p", default="photogrammetry",
                        choices=list(PRESETS.keys()), help="Extraction preset")
    parser.add_argument("--fps", "-f", type=float, help="Override: frames per second")
    parser.add_argument("--every", "-n", type=int, help="Override: extract every Nth frame")
    parser.add_argument("--start", "-ss", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", "-to", help="End time (HH:MM:SS or seconds)")
    parser.add_argument("--quality", "-q", type=int, choices=range(1, 32),
                        help="JPEG quality (1=best, 31=worst)")
    parser.add_argument("--format", choices=["jpg", "png"], help="Output format")
    parser.add_argument("--scale", help="Scale filter (e.g., '1920:-1' or '50%%')")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Show command only")
    parser.add_argument("--list-presets", "-l", action="store_true", help="List presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for name, info in PRESETS.items():
            print(f"  {name:18} {info['description']}")
        sys.exit(0)
    
    if not args.video or not args.output:
        parser.error("video and output are required")
    
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    
    count = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        preset=args.preset,
        fps=args.fps,
        every_n=args.every,
        start_time=args.start,
        end_time=args.end,
        quality=args.quality,
        output_format=args.format,
        scale=args.scale,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()

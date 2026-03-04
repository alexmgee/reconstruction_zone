#!/usr/bin/env python3
"""
prep360 CLI - Command-line interface for 360° video processing.

Usage:
    python -m prep360 <command> [options]

Commands:
    analyze     Analyze video file
    extract     Extract frames from video
    reframe     Reframe equirectangular images
    lut         Apply LUT color correction
    sky         Filter sky images
    blur        Filter blurry images
    presets     Manage presets
    pipeline    Run full pipeline
    colmap      Export Metashape project to COLMAP/XMP
"""

import argparse
import sys
from pathlib import Path


def cmd_analyze(args):
    """Analyze video file."""
    from .core.analyzer import VideoAnalyzer
    import json

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

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_extract(args):
    """Extract frames from video."""
    from .core.extractor import FrameExtractor, ExtractionConfig, ExtractionMode

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

    def progress(curr, total, msg):
        print(f"[{curr}/{total}] {msg}")

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
        print(f"Error: {result.error}", file=sys.stderr)
        return 1


def cmd_reframe(args):
    """Reframe equirectangular images to perspective views."""
    from .core.reframer import Reframer, VIEW_PRESETS

    if args.list_presets:
        print("Available presets:")
        for name, config in VIEW_PRESETS.items():
            print(f"  {name}: {config.total_views()} views")
        return 0

    config = VIEW_PRESETS.get(args.preset)
    if config is None:
        print(f"Unknown preset: {args.preset}", file=sys.stderr)
        return 1

    config.output_size = args.size
    config.jpeg_quality = args.quality

    reframer = Reframer(config)

    if args.info:
        print(f"Preset: {args.preset}")
        print(reframer.preview_view_positions())
        return 0

    def progress(curr, total, name):
        print(f"[{curr}/{total}] {name}")

    result = reframer.reframe_batch(
        args.input,
        args.output,
        num_workers=args.workers,
        progress_callback=progress
    )

    if result.success:
        print(f"Reframed {result.input_count} images -> {result.output_count} views")
        return 0
    else:
        print(f"Errors: {len(result.errors)}", file=sys.stderr)
        for err in result.errors[:5]:
            print(f"  {err}", file=sys.stderr)
        return 1


def cmd_lut(args):
    """Apply LUT color correction."""
    from .core.lut import LUTProcessor, list_luts

    if args.list:
        luts = list_luts(args.lut_dir or ".")
        print("Available LUTs:")
        for lut in luts:
            print(f"  {lut.name} ({lut.size}x{lut.size}x{lut.size})")
        return 0

    processor = LUTProcessor()

    if args.info:
        try:
            _, info = processor.load_cube(args.lut)
            print(f"LUT: {info.name}")
            print(f"Size: {info.size}x{info.size}x{info.size}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def progress(curr, total, name):
        print(f"[{curr}/{total}] {name}")

    input_path = Path(args.input)

    if input_path.is_dir():
        success, errors = processor.process_batch(
            str(input_path),
            args.output,
            args.lut,
            args.strength,
            progress
        )
        print(f"Processed {success} images, {errors} errors")
    else:
        if processor.process_image(str(input_path), args.output, args.lut, args.strength):
            print(f"Saved: {args.output}")
        else:
            print("Error processing image", file=sys.stderr)
            return 1

    return 0


def cmd_sky(args):
    """Filter sky images."""
    from .core.sky_filter import SkyFilter, SkyFilterConfig

    config = SkyFilterConfig(
        brightness_threshold=args.brightness,
        saturation_threshold=args.saturation,
        keypoint_threshold=args.keypoints,
    )

    filter = SkyFilter(config)

    def progress(curr, total, name):
        if not args.quiet:
            print(f"[{curr}/{total}] {name}")

    if args.analyze:
        results = filter.analyze_batch(args.input, progress)
        sky_count = sum(1 for m in results.values() if m.is_sky)
        print(f"Analyzed {len(results)} images: {sky_count} sky, {len(results) - sky_count} kept")

        if args.verbose:
            for name, metrics in results.items():
                status = "SKY" if metrics.is_sky else "OK"
                print(f"  {name}: {status}")
    else:
        if not args.output:
            print("Error: output directory required", file=sys.stderr)
            return 1

        kept, sky = filter.filter_images(
            args.input,
            args.output,
            progress,
            dry_run=args.dry_run
        )
        print(f"Kept: {len(kept)}, Sky: {len(sky)}")

    return 0


def cmd_presets(args):
    """Manage presets."""
    from .core.presets import PresetManager
    import json

    manager = PresetManager(args.dir)

    if args.action == "list":
        print("Available presets:")
        for name in manager.list_presets():
            info = manager.get_preset_info(name)
            builtin = "[builtin]" if info.get("is_builtin") else "[user]"
            views = info.get("total_views", "?")
            print(f"  {name:20} {builtin:10} {views:2} views")

    elif args.action == "show":
        if not args.name:
            print("Error: preset name required", file=sys.stderr)
            return 1
        try:
            preset = manager.load(args.name)
            print(json.dumps(preset.to_dict(), indent=2))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.action == "export":
        if not args.name:
            print("Error: preset name required", file=sys.stderr)
            return 1
        try:
            preset = manager.load(args.name)
            path = manager.save(preset)
            print(f"Exported to: {path}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


def cmd_blur(args):
    """Filter blurry images from a directory."""
    from .core.blur_filter import BlurFilter, BlurFilterConfig

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory", file=sys.stderr)
        return 1

    config = BlurFilterConfig(
        threshold=args.threshold,
        percentile=args.percentile if args.threshold is None and args.top is None else 80.0,
        keep_top_n=args.top,
        workers=args.workers,
    )

    bf = BlurFilter(config)

    if args.dry_run or not args.output:
        # Analyze only
        import numpy as np

        def progress(curr, total, name):
            if args.verbose:
                print(f"  [{curr}/{total}] {name}")

        scores = bf.analyze_batch(args.input, progress)
        if not scores:
            print("No images found.")
            return 0

        score_values = [s.score for s in scores]
        print(f"Analyzed {len(scores)} images")
        print(f"Score range: {min(score_values):.1f} - {max(score_values):.1f}")
        print(f"Mean: {np.mean(score_values):.1f}, Median: {np.median(score_values):.1f}")

        if args.verbose:
            print("\nTop 5 sharpest:")
            for s in scores[:5]:
                print(f"  {s.image_name}: {s.score:.1f}")
            print("\nBottom 5 (blurriest):")
            for s in scores[-5:]:
                print(f"  {s.image_name}: {s.score:.1f}")

        return 0

    def progress(curr, total, name):
        print(f"  [{curr}/{total}] {name}")

    result = bf.filter_images(args.input, args.output, config, progress)

    print(f"\nKept {result.kept_count}/{result.total_images} images")
    print(f"Rejected {result.rejected_count} blurry images")
    if result.score_stats:
        print(f"Score cutoff: {result.score_stats['cutoff']:.1f}")
    print(f"Output: {args.output}")

    return 0


def cmd_pipeline(args):
    """Run full pipeline."""
    from .core.analyzer import VideoAnalyzer
    from .core.extractor import FrameExtractor, ExtractionConfig, ExtractionMode
    from .core.reframer import Reframer
    from .core.presets import PresetManager

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load preset if specified
    preset = None
    if args.preset:
        manager = PresetManager()
        try:
            preset = manager.load(args.preset)
        except ValueError:
            print(f"Warning: preset '{args.preset}' not found, using defaults")

    # Step 1: Analyze
    print("\n=== Step 1: Analyze ===")
    analyzer = VideoAnalyzer()
    info = analyzer.analyze(args.video)
    print(f"Video: {info.filename}")
    print(f"Resolution: {info.width}x{info.height}, Duration: {info.duration_seconds:.1f}s")

    # Step 2: Extract
    print("\n=== Step 2: Extract Frames ===")
    frames_dir = output_base / "frames"

    interval = preset.extraction_interval if preset else args.interval
    config = ExtractionConfig(
        interval=interval,
        mode=ExtractionMode.FIXED,
        quality=95,
    )

    extractor = FrameExtractor()

    def progress(curr, total, msg):
        print(f"  [{curr}/{total}] {msg}")

    result = extractor.extract(args.video, str(frames_dir), config, progress)
    if not result.success:
        print(f"Extraction failed: {result.error}", file=sys.stderr)
        return 1
    print(f"Extracted {result.frame_count} frames")

    # Step 2.5: Filter blurry frames (optional)
    reframe_source = frames_dir
    if args.blur_filter:
        from .core.blur_filter import BlurFilter, BlurFilterConfig

        print(f"\n=== Step 2.5: Filter Blurry Frames ===")
        filtered_dir = output_base / "frames_filtered"
        blur_config = BlurFilterConfig(
            percentile=args.blur_percentile,
            workers=args.workers,
        )
        bf = BlurFilter(blur_config)
        blur_result = bf.filter_images(str(frames_dir), str(filtered_dir), blur_config, progress)
        print(f"Kept {blur_result.kept_count}/{blur_result.total_images} sharp frames")
        if blur_result.rejected_count > 0:
            print(f"Rejected {blur_result.rejected_count} blurry frames")
        reframe_source = filtered_dir

    # Step 3: Reframe
    print("\n=== Step 3: Reframe ===")
    perspectives_dir = output_base / "perspectives"

    if preset:
        view_config = preset.get_view_config()
    else:
        from .core.reframer import VIEW_PRESETS
        view_config = VIEW_PRESETS["prep360_default"]

    view_config.output_size = args.size

    reframer = Reframer(view_config)
    result = reframer.reframe_batch(
        str(reframe_source),
        str(perspectives_dir),
        num_workers=args.workers,
        progress_callback=progress
    )

    if not result.success:
        print(f"Reframing had errors", file=sys.stderr)

    print(f"Created {result.output_count} perspective views")

    print(f"\n=== Complete ===")
    print(f"Output: {output_base}")
    print(f"  frames/: {result.input_count} equirectangular frames")
    print(f"  perspectives/: {result.output_count} perspective views")

    return 0


def cmd_segment(args):
    """Segment images to generate masks."""
    from .core.segmenter import (
        Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS,
        get_class_id, HAS_YOLO
    )

    if args.list_classes:
        print("COCO Classes:")
        for cid, name in sorted(COCO_CLASSES.items()):
            print(f"  {cid:2d}: {name}")
        return 0

    if args.list_presets:
        print("Class Presets:")
        for name, ids in CLASS_PRESETS.items():
            class_names = [COCO_CLASSES[i] for i in ids]
            print(f"  {name}: {', '.join(class_names)}")
        return 0

    if not HAS_YOLO:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return 1

    # Parse classes
    if args.preset:
        class_ids = CLASS_PRESETS[args.preset]
    else:
        class_ids = []
        for c in args.classes:
            try:
                class_ids.append(int(c))
            except ValueError:
                cid = get_class_id(c)
                if cid is not None:
                    class_ids.append(cid)
                else:
                    print(f"Warning: Unknown class '{c}'")

    if not class_ids:
        class_ids = [0]

    config = SegmentConfig(
        model_name=args.model,
        classes=class_ids,
        confidence=args.confidence,
        dilate_pixels=args.dilate,
        invert_mask=args.invert,
        device=args.device,
    )

    segmenter = Segmenter(config)

    print(f"Classes: {segmenter.get_class_names(class_ids)}")
    print(f"Model: {config.model_name}")

    def progress(curr, total, name):
        print(f"[{curr}/{total}] {name}")

    from pathlib import Path
    input_path = Path(args.input)

    if input_path.is_dir():
        result = segmenter.segment_batch(args.input, args.output, progress)
        print(f"\nProcessed {result.total_images} images")
        print(f"Images with detections: {result.images_with_detections}")
        print(f"Total detections: {result.total_detections}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
        return 0 if result.success else 1
    else:
        result = segmenter.segment_single(args.input, args.output)
        if result.error:
            print(f"Error: {result.error}")
            return 1
        print(f"Detections: {result.detections}")
        print(f"Classes: {result.classes_found}")
        print(f"Mask: {result.mask_path}")
        return 0


def cmd_colmap(args):
    """Export Metashape XML project to COLMAP + XMP."""
    from .core.colmap_export import (
        ColmapExporter, ColmapExportConfig, print_project_info,
    )

    if args.info:
        try:
            print_project_info(args.xml)
        except FileNotFoundError:
            print(f"Error: file not found: {args.xml}")
            return 1
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return 1
        return 0

    if not args.images or not args.output:
        print("Error: --images and --output are required (unless using --info)")
        return 1

    config = ColmapExportConfig(
        xml_path=args.xml,
        images_dir=args.images,
        output_dir=args.output,
        ply_path=getattr(args, 'ply', None),
        masks_dir=getattr(args, 'masks', None),
        preset_name=getattr(args, 'preset', "prep360_default") or "prep360_default",
        output_size=args.size,
        jpeg_quality=args.quality,
        export_colmap=not args.no_colmap,
        export_xmp=not args.no_xmp,
        export_rig_config=args.rig_config,
        undistort_frame_cameras=not args.no_undistort,
        num_workers=args.workers,
        xmp_pose_prior=args.xmp_pose_prior,
        skip_existing=not args.force,
    )

    exporter = ColmapExporter(config)

    def progress(curr, total, name):
        print(f"  [{curr + 1}/{total}] {name}")

    result = exporter.export(progress_callback=progress)

    if result.success:
        print(f"\nExport complete:")
        print(f"  Images: {result.images_exported}")
        if result.cameras_written > 0:
            print(f"  COLMAP cameras: {result.cameras_written}")
        if result.points_written > 0:
            print(f"  3D points: {result.points_written}")
        if result.xmp_count > 0:
            print(f"  XMP sidecars: {result.xmp_count}")
        if result.skipped_existing > 0:
            print(f"  Skipped (existing): {result.skipped_existing}")
        if result.skipped_unaligned > 0:
            print(f"  Skipped (unaligned): {result.skipped_unaligned}")
        if result.colmap_dir:
            print(f"  COLMAP output: {result.colmap_dir}")
        return 0
    else:
        print(f"\nExport failed:")
        for err in result.errors[:10]:
            print(f"  {err}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="prep360 - 360° Video Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # analyze
    p = subparsers.add_parser("analyze", help="Analyze video file")
    p.add_argument("video", help="Video file to analyze")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    # extract
    p = subparsers.add_parser("extract", help="Extract frames from video")
    p.add_argument("video", help="Input video file")
    p.add_argument("output", help="Output directory")
    p.add_argument("--interval", "-i", type=float, default=2.0, help="Interval in seconds")
    p.add_argument("--mode", "-m", choices=["fixed", "scene", "adaptive"], default="fixed")
    p.add_argument("--start", type=float, help="Start time in seconds")
    p.add_argument("--end", type=float, help="End time in seconds")
    p.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality")
    p.add_argument("--format", "-f", choices=["jpg", "png"], default="jpg")
    p.add_argument("--scene-threshold", type=float, default=0.3)
    p.add_argument("--dry-run", "-n", action="store_true")

    # reframe
    p = subparsers.add_parser("reframe", help="Reframe equirectangular images")
    p.add_argument("input", nargs="?", help="Input directory")
    p.add_argument("output", nargs="?", help="Output directory")
    p.add_argument("--preset", "-p", default="prep360_default", help="View preset")
    p.add_argument("--size", "-s", type=int, default=1920, help="Output size")
    p.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    p.add_argument("--info", action="store_true", help="Show preset info")
    p.add_argument("--list-presets", action="store_true", help="List presets")

    # lut
    p = subparsers.add_parser("lut", help="Apply LUT color correction")
    p.add_argument("input", nargs="?", help="Input image or directory")
    p.add_argument("output", nargs="?", help="Output image or directory")
    p.add_argument("--lut", "-l", help="Path to .cube LUT file")
    p.add_argument("--strength", "-s", type=float, default=1.0, help="LUT strength")
    p.add_argument("--info", action="store_true", help="Show LUT info")
    p.add_argument("--list", action="store_true", help="List available LUTs")
    p.add_argument("--lut-dir", help="LUT directory")

    # sky
    p = subparsers.add_parser("sky", help="Filter sky images")
    p.add_argument("input", help="Input directory")
    p.add_argument("output", nargs="?", help="Output directory")
    p.add_argument("--brightness", "-b", type=float, default=0.85)
    p.add_argument("--saturation", "-s", type=float, default=0.15)
    p.add_argument("--keypoints", "-k", type=int, default=50)
    p.add_argument("--analyze", "-a", action="store_true", help="Analyze only")
    p.add_argument("--dry-run", "-n", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")

    # blur
    p = subparsers.add_parser("blur", help="Filter blurry images")
    p.add_argument("input", help="Input directory with images")
    p.add_argument("output", nargs="?", help="Output directory for sharp images")
    p.add_argument("--percentile", "-p", type=float, default=80.0,
                   help="Keep top N percent (default: 80)")
    p.add_argument("--threshold", "-t", type=float, help="Absolute score threshold")
    p.add_argument("--top", "-n", type=int, help="Keep only top N sharpest")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    p.add_argument("--dry-run", "-d", action="store_true", help="Analyze only, don't copy")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # presets
    p = subparsers.add_parser("presets", help="Manage presets")
    p.add_argument("action", choices=["list", "show", "export"])
    p.add_argument("name", nargs="?", help="Preset name")
    p.add_argument("--dir", help="Presets directory")

    # pipeline
    p = subparsers.add_parser("pipeline", help="Run full pipeline")
    p.add_argument("video", help="Input video file")
    p.add_argument("output", help="Output directory")
    p.add_argument("--preset", "-p", help="Camera preset")
    p.add_argument("--interval", "-i", type=float, default=2.0, help="Extraction interval")
    p.add_argument("--size", "-s", type=int, default=1920, help="Output size")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    p.add_argument("--blur-filter", action="store_true",
                   help="Filter blurry frames after extraction")
    p.add_argument("--blur-percentile", type=float, default=80.0,
                   help="Keep top N percent sharpest (default: 80)")

    # segment
    p = subparsers.add_parser("segment", help="Generate masks using YOLO segmentation")
    p.add_argument("input", nargs="?", help="Input image or directory")
    p.add_argument("output", nargs="?", help="Output directory for masks")
    p.add_argument("--model", "-m", default="yolo11n-seg.pt",
                   help="YOLO model (default: yolo11n-seg.pt)")
    p.add_argument("--classes", "-c", nargs="+", type=str, default=["person"],
                   help="Class names or IDs to detect (default: person)")
    p.add_argument("--preset", "-p", choices=["person", "photographer", "equipment",
                                               "vehicles", "animals", "all_dynamic"],
                   help="Use a class preset")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Detection confidence threshold (default: 0.5)")
    p.add_argument("--dilate", "-d", type=int, default=0,
                   help="Dilate masks by N pixels")
    p.add_argument("--invert", "-i", action="store_true",
                   help="Invert mask (white background)")
    p.add_argument("--device", default="", help="Device (auto, cpu, cuda:0)")
    p.add_argument("--list-classes", action="store_true", help="List COCO classes")
    p.add_argument("--list-presets", action="store_true", help="List class presets")

    # colmap
    p = subparsers.add_parser("colmap", help="Export Metashape project to COLMAP/XMP")
    p.add_argument("--xml", "-x", required=True, help="Metashape cameras.xml")
    p.add_argument("--images", "-i", help="Source images directory")
    p.add_argument("--output", "-o", help="Output directory")
    p.add_argument("--ply", help="PLY pointcloud for points3D.txt")
    p.add_argument("--masks", help="Equirectangular masks directory")
    p.add_argument("--preset", "-p", default="prep360_default", help="View preset name")
    p.add_argument("--size", "-s", type=int, default=1920, help="Output crop size")
    p.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality")
    p.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    p.add_argument("--no-colmap", action="store_true", help="Skip COLMAP output")
    p.add_argument("--no-xmp", action="store_true", help="Skip XMP output")
    p.add_argument("--rig-config", action="store_true", help="Generate rig_config.json")
    p.add_argument("--no-undistort", action="store_true",
                   help="Skip frame camera undistortion")
    p.add_argument("--xmp-pose-prior", default="locked",
                   choices=["initial", "exact", "locked"],
                   help="XMP pose prior (default: locked)")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--info", action="store_true",
                   help="Parse XML and show project info only")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "analyze": cmd_analyze,
        "extract": cmd_extract,
        "reframe": cmd_reframe,
        "lut": cmd_lut,
        "sky": cmd_sky,
        "blur": cmd_blur,
        "presets": cmd_presets,
        "pipeline": cmd_pipeline,
        "segment": cmd_segment,
        "colmap": cmd_colmap,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

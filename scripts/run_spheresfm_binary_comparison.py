"""Run a controlled custom-vs-prebuilt SphereSfM comparison.

This helper exists to support Priority 1 in:
docs/planning/2026-04-11-spheresfm-next-research-plan.md

It runs the same ERP alignment configuration against two SphereSfM binaries:

- the current custom local build
- the official prebuilt V1.2 Windows binary

Each binary gets its own isolated workspace root. The script then emits a
single comparison summary JSON and Markdown file that point back to the
underlying per-run `run_info.json` artifacts written by `ColmapRunner`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruction_gui.alignment_profiles import build_spheresfm_erp_profile
from reconstruction_gui.colmap_runner import ColmapRunner


DEFAULT_CUSTOM_BINARY = Path(r"D:\Tools\SphereSfM\bin\colmap.exe")
DEFAULT_PREBUILT_BINARY = (
    REPO_ROOT / ".tmp" / "SphereSfM-2025-8-18" / "SphereSfM-2024-12-14" / "colmap.exe"
)
DEFAULT_COMPARISON_ROOT = REPO_ROOT / ".tmp" / "spheresfm-prebuilt-comparison"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare custom and prebuilt SphereSfM binaries on one ERP dataset."
    )
    parser.add_argument("--images", required=True, help="Directory containing ERP images.")
    parser.add_argument("--masks", default="", help="Optional per-image mask directory.")
    parser.add_argument("--pose-path", default="", help="Optional pose/POS file for spatial matching.")
    parser.add_argument(
        "--camera-mask-path",
        default="",
        help="Optional global camera mask file passed as ImageReader.camera_mask_path.",
    )
    parser.add_argument(
        "--strategy",
        default="exhaustive",
        choices=("exhaustive", "sequential", "spatial", "vocab_tree"),
        help="Matching strategy to use for both binaries.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        default="",
        help="Vocabulary tree path, required if --strategy vocab_tree.",
    )
    parser.add_argument(
        "--custom-binary",
        default=str(DEFAULT_CUSTOM_BINARY),
        help="Path to the current custom SphereSfM binary.",
    )
    parser.add_argument(
        "--prebuilt-binary",
        default=str(DEFAULT_PREBUILT_BINARY),
        help="Path to the official prebuilt SphereSfM binary.",
    )
    parser.add_argument(
        "--comparison-root",
        default=str(DEFAULT_COMPARISON_ROOT),
        help="Parent directory for the comparison session output.",
    )
    parser.add_argument(
        "--session-name",
        default="",
        help="Optional fixed session name. Defaults to a timestamped folder.",
    )
    parser.add_argument("--image-width", type=int, default=0, help="ERP image width.")
    parser.add_argument("--image-height", type=int, default=0, help="ERP image height.")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--max-image-size", type=int, default=2048)
    parser.add_argument("--max-num-matches", type=int, default=32768)
    parser.add_argument("--guided-matching", action="store_true")
    parser.add_argument("--mapper", default="incremental", choices=("incremental", "global"))
    parser.add_argument("--min-num-inliers", type=int, default=15)
    parser.add_argument("--snapshot-freq", type=int, default=10)
    parser.add_argument("--mapper-num-threads", type=int, default=0)
    parser.add_argument("--spatial-is-gps", type=int, choices=(0, 1), default=0)
    parser.add_argument("--spatial-max-distance", type=float, default=50.0)
    parser.add_argument("--sift-match-max-error", type=float, default=0.0)
    parser.add_argument("--sift-match-min-num-inliers", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate both binaries and write the comparison summary.",
    )
    return parser.parse_args()


def _ensure_path(path_str: str, label: str, *, must_be_file: bool = False) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if must_be_file and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if not must_be_file and not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    return path.resolve()


def _infer_image_size(images_dir: Path) -> Tuple[int, int]:
    candidates = [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    ]
    if not candidates:
        raise RuntimeError(f"No image files found under {images_dir}")

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Could not infer image size automatically because Pillow is unavailable. "
            "Pass --image-width and --image-height explicitly."
        ) from exc

    with Image.open(candidates[0]) as image:
        width, height = image.size
    return int(width), int(height)


def _run_binary(
    *,
    label: str,
    binary_path: Path,
    workspace_root: Path,
    images_dir: Path,
    masks_dir: Optional[Path],
    profile_args: Dict[str, Any],
    runner_args: Dict[str, Any],
    validate_only: bool,
) -> Dict[str, Any]:
    runner = ColmapRunner(
        binary_path=str(binary_path),
        camera_model="SPHERE",
        workspace_root=str(workspace_root),
        engine_name=f"spheresfm-{label}",
    )
    validation = runner.validate_binary()
    binary_result: Dict[str, Any] = {
        "label": label,
        "binary_path": str(binary_path),
        "workspace_root": str(workspace_root),
        "validation": {
            "success": validation.success,
            "resolved_binary": validation.resolved_binary,
            "binary_flavor": validation.binary_flavor,
            "supports_sphere_workflow": validation.supports_sphere_workflow,
            "detected_commands": validation.detected_commands,
            "detected_capabilities": validation.detected_capabilities,
            "error": validation.error,
        },
    }
    if validate_only:
        return binary_result

    profile = build_spheresfm_erp_profile(**profile_args)
    results = runner.run_all(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir) if masks_dir else None,
        max_features=runner_args["max_features"],
        max_image_size=runner_args["max_image_size"],
        feature_type="SIFT",
        strategy=profile.matching_strategy,
        guided=profile.guided_matching,
        max_num_matches=runner_args["max_num_matches"],
        vocab_tree_path=profile.vocab_tree_path or None,
        mapper=runner_args["mapper"],
        min_num_inliers=runner_args["min_num_inliers"],
        extract_extra_args=profile.extract_extra_args,
        match_extra_args=profile.match_extra_args,
        reconstruct_extra_args=profile.reconstruct_extra_args,
    )
    run_info_path = runner.current_run_dir / runner.RUN_INFO_FILENAME if runner.current_run_dir else None
    run_info = {}
    if run_info_path and run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))

    binary_result.update(
        {
            "run_dir": str(runner.current_run_dir) if runner.current_run_dir else "",
            "run_info_path": str(run_info_path) if run_info_path else "",
            "run_info": run_info,
            "stages": [
                {
                    "stage": result.stage,
                    "status": result.status,
                    "returncode": result.returncode,
                    "error": result.error,
                    "selected_model_dir": result.selected_model_dir,
                    "stats": result.stats,
                }
                for result in results
            ],
        }
    )
    return binary_result


def _summarize_binary_result(result: Dict[str, Any]) -> Dict[str, Any]:
    run_info = result.get("run_info") or {}
    stages = run_info.get("stages", {})
    reconstruction = stages.get("reconstruction", {})
    reconstruction_stats = reconstruction.get("stats", {})
    return {
        "label": result.get("label", ""),
        "binary_path": result.get("binary_path", ""),
        "validation_ok": bool(result.get("validation", {}).get("success")),
        "binary_flavor": result.get("validation", {}).get("binary_flavor", ""),
        "supports_sphere_workflow": bool(
            result.get("validation", {}).get("supports_sphere_workflow")
        ),
        "run_dir": result.get("run_dir", ""),
        "reconstruction_status": reconstruction.get("status", ""),
        "reconstruction_returncode": reconstruction.get("returncode"),
        "reconstruction_error": reconstruction.get("error", ""),
        "progress_summary": reconstruction_stats.get("progress_summary", ""),
        "selected_model_dir": reconstruction.get("selected_model_dir", "")
        or run_info.get("selected_model_dir", ""),
        "selected_model_stats": reconstruction_stats.get("selected_model_stats", {}),
        "recovered_model_source": reconstruction_stats.get("recovered_model_source", ""),
        "recovered_from_snapshot": reconstruction_stats.get("recovered_from_snapshot", False),
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# SphereSfM Binary Comparison")
    lines.append("")
    lines.append(f"Created: {summary['created_at']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Images: `{summary['inputs']['images_dir']}`")
    lines.append(f"- Masks: `{summary['inputs']['masks_dir']}`")
    lines.append(f"- Pose path: `{summary['inputs']['pose_path']}`")
    lines.append(f"- Camera mask path: `{summary['inputs']['camera_mask_path']}`")
    lines.append(f"- Image size: `{summary['inputs']['image_width']}x{summary['inputs']['image_height']}`")
    lines.append(f"- Strategy: `{summary['inputs']['strategy']}`")
    lines.append(f"- Mapper: `{summary['inputs']['mapper']}`")
    lines.append(f"- Snapshot freq: `{summary['inputs']['snapshot_freq']}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for binary in summary["binary_summaries"]:
        lines.append(f"### {binary['label']}")
        lines.append("")
        lines.append(f"- Binary: `{binary['binary_path']}`")
        lines.append(f"- Validation ok: `{binary['validation_ok']}`")
        lines.append(f"- Binary flavor: `{binary['binary_flavor']}`")
        lines.append(f"- Supports sphere workflow: `{binary['supports_sphere_workflow']}`")
        lines.append(f"- Run dir: `{binary['run_dir']}`")
        lines.append(f"- Reconstruction status: `{binary['reconstruction_status']}`")
        lines.append(f"- Reconstruction return code: `{binary['reconstruction_returncode']}`")
        lines.append(f"- Reconstruction error: `{binary['reconstruction_error']}`")
        lines.append(f"- Progress summary: `{binary['progress_summary']}`")
        lines.append(f"- Selected model dir: `{binary['selected_model_dir']}`")
        lines.append(
            f"- Recovered model source: `{binary['recovered_model_source']}` "
            f"(snapshot: `{binary['recovered_from_snapshot']}`)"
        )
        selected_stats = binary.get("selected_model_stats") or {}
        if selected_stats:
            lines.append(
                "- Selected model stats: "
                f"`registered={selected_stats.get('num_registered', '')}`, "
                f"`points={selected_stats.get('num_points', '')}`, "
                f"`mean_reproj={selected_stats.get('mean_reproj_error', '')}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()

    images_dir = _ensure_path(args.images, "Images directory")
    masks_dir = _ensure_path(args.masks, "Masks directory") if args.masks else None
    pose_path = _ensure_path(args.pose_path, "Pose path", must_be_file=True) if args.pose_path else None
    camera_mask_path = (
        _ensure_path(args.camera_mask_path, "Camera mask path", must_be_file=True)
        if args.camera_mask_path
        else None
    )
    custom_binary = _ensure_path(args.custom_binary, "Custom SphereSfM binary", must_be_file=True)
    prebuilt_binary = _ensure_path(args.prebuilt_binary, "Prebuilt SphereSfM binary", must_be_file=True)

    if args.strategy == "vocab_tree" and not args.vocab_tree_path:
        raise SystemExit("--vocab-tree-path is required when --strategy vocab_tree")

    if args.image_width > 0 and args.image_height > 0:
        image_width, image_height = args.image_width, args.image_height
    else:
        image_width, image_height = _infer_image_size(images_dir)

    session_name = args.session_name.strip() or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_root = Path(args.comparison_root).expanduser() / session_name
    session_root.mkdir(parents=True, exist_ok=True)

    profile_args: Dict[str, Any] = {
        "image_width": image_width,
        "image_height": image_height,
        "pose_path": str(pose_path) if pose_path else None,
        "camera_mask_path": str(camera_mask_path) if camera_mask_path else None,
        "matching_strategy": args.strategy,
        "spatial_is_gps": args.spatial_is_gps,
        "spatial_max_distance": args.spatial_max_distance if args.strategy == "spatial" else None,
        "guided_matching": args.guided_matching,
        "vocab_tree_path": args.vocab_tree_path,
        "match_extra_args": {},
        "reconstruct_extra_args": {
            "Mapper.snapshot_images_freq": args.snapshot_freq,
        },
    }
    if args.sift_match_max_error > 0:
        profile_args["match_extra_args"]["SiftMatching.max_error"] = args.sift_match_max_error
    if args.sift_match_min_num_inliers > 0:
        profile_args["match_extra_args"]["SiftMatching.min_num_inliers"] = (
            args.sift_match_min_num_inliers
        )
    if args.mapper_num_threads > 0:
        profile_args["reconstruct_extra_args"]["Mapper.num_threads"] = args.mapper_num_threads

    runner_args = {
        "max_features": args.max_features,
        "max_image_size": args.max_image_size,
        "max_num_matches": args.max_num_matches,
        "mapper": args.mapper,
        "min_num_inliers": args.min_num_inliers,
    }

    results = []
    for label, binary_path in (
        ("custom", custom_binary),
        ("prebuilt", prebuilt_binary),
    ):
        workspace_root = session_root / label
        workspace_root.mkdir(parents=True, exist_ok=True)
        result = _run_binary(
            label=label,
            binary_path=binary_path,
            workspace_root=workspace_root,
            images_dir=images_dir,
            masks_dir=masks_dir,
            profile_args=profile_args,
            runner_args=runner_args,
            validate_only=args.validate_only,
        )
        results.append(result)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_root": str(session_root),
        "inputs": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir) if masks_dir else "",
            "pose_path": str(pose_path) if pose_path else "",
            "camera_mask_path": str(camera_mask_path) if camera_mask_path else "",
            "image_width": image_width,
            "image_height": image_height,
            "strategy": args.strategy,
            "mapper": args.mapper,
            "max_features": args.max_features,
            "max_image_size": args.max_image_size,
            "max_num_matches": args.max_num_matches,
            "min_num_inliers": args.min_num_inliers,
            "snapshot_freq": args.snapshot_freq,
            "guided_matching": args.guided_matching,
            "mapper_num_threads": args.mapper_num_threads,
            "sift_match_max_error": args.sift_match_max_error,
            "sift_match_min_num_inliers": args.sift_match_min_num_inliers,
            "validate_only": args.validate_only,
        },
        "binaries": results,
        "binary_summaries": [_summarize_binary_result(result) for result in results],
    }

    json_path = session_root / "comparison_summary.json"
    md_path = session_root / "comparison_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"Wrote comparison summary: {json_path}")
    print(f"Wrote comparison summary: {md_path}")
    for binary in summary["binary_summaries"]:
        print(
            f"[{binary['label']}] validation_ok={binary['validation_ok']} "
            f"status={binary['reconstruction_status']} "
            f"run_dir={binary['run_dir']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

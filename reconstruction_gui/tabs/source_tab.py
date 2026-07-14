"""
Source Tab — Analyze, Extract (with video queue), and Fisheye.

Ported from panoex_gui.py into a single scrollable tab for Reconstruction Zone.
All GUI state is stored on the *app* instance so the preview panel and other
tabs can read it.

Usage::

    from tabs.source_tab import build_source_tab
    build_source_tab(app, tab_frame)
"""

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk
from widgets import (
    BROWSE_BUTTON_WIDTH,
    COLOR_ACTION_DANGER,
    COLOR_ACTION_DANGER_H,
    COLOR_ACTION_MUTED,
    COLOR_ACTION_MUTED_H,
    COLOR_ACTION_PRIMARY,
    COLOR_ACTION_PRIMARY_H,
    COLOR_ACTION_SECONDARY,
    COLOR_ACTION_SECONDARY_H,
    COLOR_TEXT_DIM,
    COLOR_TEXT_DISABLED,
    FONT_TEXT_BTN_PRIMARY,
    FONT_TEXT_MONO_VALUE,
    FONT_TEXT_STATUS,
    HEIGHT_ACTION_BAR,
    LABEL_FIELD_WIDTH,
    CollapsibleSection,
    Section,
    Tooltip,
)

# ── prep360 core (optional) ────────────────────────────────────────────

try:
    from prep360.core import (
        FISHEYE_PRESETS,
        DualFisheyeCalibration,
        ExtractionConfig,
        ExtractionMode,
        FisheyeViewConfig,
        FrameExtractor,
        OSVHandler,
        SharpestConfig,
        SharpestExtractor,
        VideoAnalyzer,
    )
    from prep360.core.fisheye_reframer import (
        batch_extract as fisheye_batch_extract,
    )
    from prep360.core.fisheye_reframer import (
        default_osmo360_calibration,
    )
    from prep360.core.geotagger import (
        derive_ground_elevation,
        geotag_from_interval,
        geotag_from_manifest,
    )
    from prep360.core.paired_split_video_extractor import (
        PairedSplitConfig,
        PairedSplitVideoExtractor,
    )
    from prep360.core.queue_manager import ExtractionSettings, VideoQueue
    from prep360.core.srt_parser import find_srt_for_video, parse_srt

    HAS_PREP360 = True
except ImportError:
    HAS_PREP360 = False

# GPU extraction availability (uses extractor's hardened check)
_HAS_CUDACODEC = False
try:
    from prep360.core.sharpest_extractor import SharpestExtractor as _SE
    _HAS_CUDACODEC = _SE()._gpu_available()
    del _SE
except Exception:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── 360 preset display labels ─────────────────────────────────────────

def _build_preset_labels():
    """Build display-label ↔ preset-key mappings from FISHEYE_PRESETS."""
    if not HAS_PREP360:
        return {}, {}
    label_to_key = {}
    key_to_label = {}
    for key, cfg in FISHEYE_PRESETS.items():
        label = _preset_description_oneline(cfg)
        label_to_key[label] = key
        key_to_label[key] = label
    return label_to_key, key_to_label


def _preset_description_oneline(cfg):
    """One-line description: '13 views per lens: 1 looking up (45°), ...'"""
    front = cfg.views_for_lens("front")
    per_lens = len(front)
    pitch_groups = {}
    for v in front:
        p = round(v.pitch_deg)
        pitch_groups.setdefault(p, []).append(v)
    tiers = []
    for pitch in sorted(pitch_groups.keys(), reverse=True):
        n = len(pitch_groups[pitch])
        if pitch > 0:
            tiers.append(f"{n} looking up ({pitch}\u00b0)")
        elif pitch < 0:
            tiers.append(f"{n} looking down ({pitch}\u00b0)")
        else:
            tiers.append(f"{n} at horizon")
    return f"{per_lens} views per lens: {', '.join(tiers)}"

_PRESET_LABEL_TO_KEY, _PRESET_KEY_TO_LABEL = _build_preset_labels()

def _get_preset_key(app) -> str:
    """Map the display label back to the internal preset key."""
    return _PRESET_LABEL_TO_KEY.get(app.fisheye_preset_var.get(), "osv-pinhole-f90-dual-16")


# ── helper dataclass ──────────────────────────────────────────────────

@dataclass
class PlannedFrame:
    """Represents a frame planned for extraction."""
    index: int
    frame_number: int
    timestamp: float
    reason: str          # "interval", "scene", "manual"
    included: bool = True


@dataclass
class SourceAnalysisSummary:
    """Minimal analysis data for the currently selected extraction source."""

    source_label: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int
    recommended_interval: float = 2.0
    format: str = ""
    codec: str = ""
    pixel_format: str = ""
    is_equirectangular: bool = False
    is_log_format: bool = False
    detected_log_type: Optional[str] = None
    pair_mode: bool = False
    warning_lines: List[str] = field(default_factory=list)


# ── extraction sharpness + scene detection ────────────────────────────

def _get_sharpness_method(app) -> str:
    """Get the selected sharpness scoring method."""
    var = getattr(app, "extract_scoring_var", None)
    return var.get().lower() if var else "laplacian"


def _get_scene_detection(app) -> bool:
    """Get whether scene detection is enabled."""
    var = getattr(app, "extract_scene_var", None)
    return var.get() if var else True


def _snapshot_settings(app) -> "ExtractionSettings":
    """Capture current GUI extraction settings into an ExtractionSettings object."""
    start = app.extract_start_entry.get().strip()
    end = app.extract_end_entry.get().strip()
    sharpness = _get_sharpness_method(app)
    return ExtractionSettings(
        mode="sharpest" if sharpness != "none" else "fixed",
        interval=round(app.extract_interval_var.get(), 1),
        quality=app.extract_quality_var.get(),
        format=app.extract_format_var.get(),
        start_sec=_parse_time(start) if start else None,
        end_sec=_parse_time(end) if end else None,
        sharpness_method=sharpness,
        scene_detection=_get_scene_detection(app),
    )


def _update_sharpness_ui(app):
    """Update UI state based on sharpness/scene selection."""
    sharpness = _get_sharpness_method(app)
    is_sharpest = sharpness != "none"

    # Grey out scene detection when sharpness = none
    scene_cb = getattr(app, "extract_scene_cb", None)
    if scene_cb:
        if is_sharpest:
            scene_cb.configure(state="normal")
        else:
            scene_cb.configure(state="disabled")
            app.extract_scene_var.set(False)

    _update_estimate(app)


# ── time parser ───────────────────────────────────────────────────────

def _parse_time(time_str: str) -> Optional[float]:
    """Parse *time_str* (seconds or ``MM:SS`` or ``HH:MM:SS``) → float seconds.

    Returns None if the string is incomplete or unparseable (e.g. mid-typing "0:").
    """
    time_str = time_str.strip()
    if not time_str:
        return None
    try:
        return float(time_str)
    except ValueError:
        pass
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        return None
    return None


# ── live estimation ────────────────────────────────────────────────────

def _format_size(nbytes: float) -> str:
    """Human-readable file size."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / (1 << 10):.0f} KB"


def _format_duration(seconds: float) -> str:
    """Seconds → MM:SS or H:MM:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _split_source_enabled(app) -> bool:
    return bool(
        getattr(app, "split_source_var", None)
        and app.split_source_var.get()
    )


def _strip_lens_suffix(stem: str) -> str:
    text = stem.strip()
    patterns = [
        r"(?i)[_-](front|back)$",
        r"(?i)^(front|back)[_-]",
        r"(?i)\s(front|back)$",
        r"(?i)^(front|back)\s",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text.rstrip("_- ").strip()


def _shared_clip_stem(front_path: str, back_path: str = "") -> str:
    front_stem = _strip_lens_suffix(Path(front_path).stem)
    if not back_path:
        return front_stem or Path(front_path).stem

    back_stem = _strip_lens_suffix(Path(back_path).stem)
    if front_stem and back_stem and front_stem == back_stem:
        return front_stem

    common = Path(front_path).stem
    other = Path(back_path).stem
    prefix = []
    for a, b in zip(common, other):
        if a != b:
            break
        prefix.append(a)
    merged = "".join(prefix).rstrip("_- ")
    return merged or front_stem or back_stem or Path(front_path).stem


def _set_entry_text(entry, value: str):
    entry.delete(0, "end")
    entry.insert(0, value)


def _maybe_autofill_split_output(app):
    if not hasattr(app, "split_output_entry"):
        return

    front = app.split_front_video_entry.get().strip()
    back = app.split_back_video_entry.get().strip()
    source = front or back
    if not source:
        return

    stem = _shared_clip_stem(front or back, back)
    parent_dir = Path(source).parent
    new_value = str(parent_dir / stem)
    current_value = app.split_output_entry.get().strip()
    last_auto_value = getattr(app, "_last_split_auto_output", "")
    if current_value and current_value != last_auto_value:
        return
    _set_entry_text(app.split_output_entry, new_value)
    app._last_split_auto_output = new_value


def _browse_split_video(app, entry, title: str):
    app._browse_file_for(
        entry,
        title,
        [("Video Files", "*.mp4 *.mov *.mkv *.avi"), ("All Files", "*.*")],
    )
    _maybe_autofill_split_output(app)


def _analysis_placeholder_text(app) -> str:
    if _split_source_enabled(app):
        return (
            "Shows: shared duration, fps,\n"
            "pair compatibility warnings,\n"
            "recommended interval & source notes"
        )
    return (
        "Shows: resolution, fps, duration,\n"
        "pair compatibility warnings,\n"
        "recommended interval & source notes"
    )


def _on_split_source_toggle(app):
    app.current_video_info = None
    app.current_video_path = None
    if hasattr(app, "analyze_results"):
        app.analyze_results.delete("1.0", "end")
        app.analyze_results.insert("1.0", _analysis_placeholder_text(app))
    _update_estimate(app)
    _update_source_mode_ui(app)


def _update_source_mode_ui(app):
    split_mode = _split_source_enabled(app)

    for widget in getattr(app, "_split_source_widgets", []):
        widget.configure(state="normal" if split_mode else "disabled")

    queue_state = "disabled" if split_mode else "normal"
    for attr in (
        "extract_queue_add_btn",
        "queue_run_btn",
        "queue_add_videos_btn",
        "queue_add_folder_btn",
        "queue_remove_btn",
        "queue_clear_done_btn",
    ):
        if hasattr(app, attr):
            getattr(app, attr).configure(state=queue_state)

    _update_sharpness_ui(app)


def _update_estimate(app, *_args):
    """Recalculate and display extraction estimates from current settings."""
    if not hasattr(app, "extract_estimate_label"):
        return
    info = getattr(app, "current_video_info", None)
    if info is None:
        app.extract_estimate_label.configure(text="")
        return

    # effective duration
    total_dur = info.duration_seconds
    start_str = app.extract_start_entry.get().strip()
    end_str = app.extract_end_entry.get().strip()
    start_sec = _parse_time(start_str) if start_str else 0.0
    if start_sec is None:
        start_sec = 0.0
    end_sec = _parse_time(end_str) if end_str and end_str.lower() != "end" else total_dur
    if end_sec is None:
        end_sec = total_dur
    end_sec = min(end_sec, total_dur)
    eff_dur = max(0.0, end_sec - start_sec)

    sharpness = _get_sharpness_method(app)
    scene = _get_scene_detection(app)
    interval = app.extract_interval_var.get()

    # frame count estimate
    frames = max(1, int(eff_dur / interval))
    if sharpness != "none" and scene:
        # Scene detection may produce more frames than fixed interval
        if getattr(info, "pair_mode", False):
            frame_str = f"~{frames:,} frame pairs (varies by scene cuts)"
        else:
            frame_str = f"~{frames:,} frames (varies by scene cuts)"
    else:
        if getattr(info, "pair_mode", False):
            frame_str = f"~{frames:,} frame pairs (~{frames * 2:,} images)"
        else:
            frame_str = f"~{frames:,} frames"

    # size estimate per frame (pixels × bytes-per-pixel × compression ratio)
    pixels = info.width * info.height
    fmt = app.extract_format_var.get()
    quality = app.extract_quality_var.get()
    if fmt == "png":
        bytes_per_frame = pixels * 3 * 0.5  # typical PNG compression
        fmt_label = "png"
    else:
        # JPEG: quality 95 ≈ 15% of raw, scales roughly linearly
        bytes_per_frame = pixels * 3 * (quality / 100) * 0.15
        fmt_label = f"jpg q{quality}"

    total_bytes = frames * bytes_per_frame * (2 if getattr(info, "pair_mode", False) else 1)

    caveats = []
    if sharpness != "none":
        parts_s = [sharpness]
        if scene:
            parts_s.append("scene-aware")
        caveats.append(f"sharpest per window ({', '.join(parts_s)})")

    parts = [frame_str, f"~{_format_size(total_bytes)} ({fmt_label})", _format_duration(eff_dur)]
    text = "  ·  ".join(parts)
    if caveats:
        text += "\n" + "  ·  ".join(caveats)

    app.extract_estimate_label.configure(text=text)


# ======================================================================
# Public entry point
# ======================================================================

def build_source_tab(app, parent):
    """Populate *parent* (the "Source" tab frame) with Analyze, Extract, and
    Fisheye sections.  All widget references are stored on *app*.
    """
    scroll = ctk.CTkScrollableFrame(parent)
    scroll.pack(fill="both", expand=True)

    _build_video_selection_section(app, scroll)
    _build_extract_section(app, scroll)
    _build_metadata_section(app, scroll)
    # Metashape Session section archived to legacy/ (2026-03-24)
    _update_source_mode_ui(app)


# ======================================================================
#  1. VIDEO SELECTION (input + output + optional analysis)
# ======================================================================

def _build_video_selection_section(app, parent):
    sec = Section(parent, "Video Selection")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Input video row
    vid_frame = ctk.CTkFrame(c, fg_color="transparent")
    vid_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(vid_frame, text="Input:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.analyze_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="Video file...")
    app.analyze_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(vid_frame, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_video_for(app.analyze_video_entry)
                  ).pack(side="left")

    # Output folder row
    out_frame = ctk.CTkFrame(c, fg_color="transparent")
    out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(out_frame, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.extract_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Frames output folder...")
    app.extract_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(out_frame, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.extract_output_entry)
                  ).pack(side="left", padx=(0, 4))

    split_sec = CollapsibleSection(
        c,
        "Split Lens Videos",
        subtitle="use graded front/back exports as the extraction source",
        expanded=False,
    )
    split_sec.pack(fill="x", pady=(4, 0), padx=2)
    sc = split_sec.content

    app.split_source_var = ctk.BooleanVar(value=False)
    split_mode_cb = ctk.CTkCheckBox(
        sc,
        text="Use Split Lens Videos",
        variable=app.split_source_var,
        command=lambda: _on_split_source_toggle(app),
        width=180,
    )
    split_mode_cb.pack(pady=(2, 4), padx=6, anchor="w")
    Tooltip(
        split_mode_cb,
        "When enabled, Analyze and Extract use the graded front/back videos below.\n"
        "Use this after splitting and color-grading outside the GUI.",
    )

    split_front_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_front_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_front_frame, text="Front:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.split_front_video_entry = ctk.CTkEntry(
        split_front_frame,
        placeholder_text="Graded front lens video (.mp4 / .mov)...",
    )
    app.split_front_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.split_front_video_entry.bind("<FocusOut>", lambda _e: _maybe_autofill_split_output(app))
    split_front_btn = ctk.CTkButton(
        split_front_frame,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: _browse_split_video(app, app.split_front_video_entry, "Select Front Lens Video"),
    )
    split_front_btn.pack(side="left")

    split_back_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_back_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_back_frame, text="Back:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.split_back_video_entry = ctk.CTkEntry(
        split_back_frame,
        placeholder_text="Graded back lens video (.mp4 / .mov)...",
    )
    app.split_back_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.split_back_video_entry.bind("<FocusOut>", lambda _e: _maybe_autofill_split_output(app))
    split_back_btn = ctk.CTkButton(
        split_back_frame,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: _browse_split_video(app, app.split_back_video_entry, "Select Back Lens Video"),
    )
    split_back_btn.pack(side="left")

    split_out_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_out_frame, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.split_output_entry = ctk.CTkEntry(
        split_out_frame,
        placeholder_text="Per-clip working folder (auto-filled from the filename stem)...",
    )
    app.split_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    split_out_btn = ctk.CTkButton(
        split_out_frame,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_folder_for(app.split_output_entry),
    )
    split_out_btn.pack(side="left")
    Tooltip(
        app.split_output_entry,
        "The main Extract button writes front/frames + back/frames here for this one clip.",
    )

    app._split_source_widgets = [
        app.split_front_video_entry,
        split_front_btn,
        app.split_back_video_entry,
        split_back_btn,
        app.split_output_entry,
        split_out_btn,
    ]

    # Analysis moved to Frame Extraction section


def _run_analyze(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    if _split_source_enabled(app):
        front_video = app.split_front_video_entry.get().strip()
        back_video = app.split_back_video_entry.get().strip()
        if not front_video or not back_video:
            app.log("Error: Select both front and back split videos first")
            return
        if not Path(front_video).exists():
            app.log(f"Error: Front video not found: {front_video}")
            return
        if not Path(back_video).exists():
            app.log(f"Error: Back video not found: {back_video}")
            return
    else:
        video = app.analyze_video_entry.get().strip()
        if not video:
            app.log("Error: Please select a video file")
            return
        if not Path(video).exists():
            app.log(f"Error: Video not found: {video}")
            return
    app.analyze_run_btn.configure(state="disabled")
    if _split_source_enabled(app):
        threading.Thread(
            target=_analyze_split_worker,
            args=(app, front_video, back_video),
            daemon=True,
        ).start()
    else:
        threading.Thread(target=_analyze_worker, args=(app, video), daemon=True).start()


def _analyze_worker(app, video):
    try:
        app.log(f"Analyzing: {Path(video).name}")
        analyzer = VideoAnalyzer()
        info = analyzer.analyze(video)
        summary = SourceAnalysisSummary(
            source_label=Path(video).name,
            width=info.width,
            height=info.height,
            fps=info.fps,
            duration_seconds=info.duration_seconds,
            frame_count=info.frame_count,
            recommended_interval=info.recommended_interval,
            format=info.format,
            codec=info.codec,
            pixel_format=info.pixel_format or "Unknown",
            is_equirectangular=info.is_equirectangular,
            is_log_format=info.is_log_format,
            detected_log_type=info.detected_log_type,
            pair_mode=False,
        )
        result_text = (
            f"File: {info.filename}\n"
            f"Format: {info.format} ({info.codec})\n"
            f"Resolution: {info.width}x{info.height}\n"
            f"FPS: {info.fps:.2f}\n"
            f"Duration: {analyzer.get_duration_formatted(info)} ({info.duration_seconds:.1f}s)\n"
            f"Frames: {info.frame_count}\n"
            f"Bitrate: {info.bitrate // 1000 if info.bitrate else '?'} kbps\n"
            f"Pixel Format: {info.pixel_format or 'Unknown'}\n\n"
            f"=== 360 Detection ===\n"
            f"Equirectangular: {'Yes' if info.is_equirectangular else 'No'}\n"
            f"Aspect Ratio: {info.width / info.height:.2f}:1\n\n"
            f"=== Color Profile ===\n"
            f"Log Format: {info.detected_log_type or 'None detected'}\n"
            f"Is Log: {'Yes' if info.is_log_format else 'No'}\n\n"
            f"=== Recommendations ===\n"
            f"Extraction Interval: {info.recommended_interval}s\n"
            f"Estimated Frames @ {info.recommended_interval}s: "
            f"{int(info.duration_seconds / info.recommended_interval)}\n"
        )
        if info.recommended_lut:
            result_text += f"Recommended LUT: {info.recommended_lut}\n"
        app.current_video_info = summary
        app.current_video_path = video
        app.after(0, lambda: _update_analyze_results(app, result_text))
        app.log("Analysis complete")
    except Exception as e:
        app.log(f"Error: {e}")
    finally:
        app.after(0, lambda: app.analyze_run_btn.configure(state="normal"))


def _update_analyze_results(app, text):
    app.analyze_results.delete("1.0", "end")
    app.analyze_results.insert("1.0", text)
    _update_estimate(app)


def _analyze_split_worker(app, front_video: str, back_video: str):
    try:
        app.log(f"Analyzing split pair: {Path(front_video).name} + {Path(back_video).name}")
        analyzer = VideoAnalyzer()
        front_info = analyzer.analyze(front_video)
        back_info = analyzer.analyze(back_video)

        warnings = []
        if front_info.width != back_info.width or front_info.height != back_info.height:
            warnings.append(
                f"Resolution mismatch: front {front_info.width}x{front_info.height}, "
                f"back {back_info.width}x{back_info.height}"
            )
        fps_delta = abs(front_info.fps - back_info.fps)
        if fps_delta > 0.05:
            warnings.append(
                f"FPS mismatch: front {front_info.fps:.3f}, back {back_info.fps:.3f}"
            )
        duration_delta = abs(front_info.duration_seconds - back_info.duration_seconds)
        if duration_delta > 0.1:
            warnings.append(
                f"Duration mismatch: front {front_info.duration_seconds:.2f}s, "
                f"back {back_info.duration_seconds:.2f}s"
            )

        shared_duration = min(front_info.duration_seconds, back_info.duration_seconds)
        shared_fps = min(front_info.fps, back_info.fps)
        shared_frames = min(front_info.frame_count, back_info.frame_count)
        shared_interval = min(front_info.recommended_interval, back_info.recommended_interval)

        summary = SourceAnalysisSummary(
            source_label=f"{Path(front_video).name} + {Path(back_video).name}",
            width=min(front_info.width, back_info.width),
            height=min(front_info.height, back_info.height),
            fps=shared_fps,
            duration_seconds=shared_duration,
            frame_count=shared_frames,
            recommended_interval=shared_interval,
            format=f"{front_info.format}/{back_info.format}",
            codec=f"{front_info.codec}/{back_info.codec}",
            pixel_format=front_info.pixel_format or back_info.pixel_format or "Unknown",
            is_equirectangular=False,
            is_log_format=front_info.is_log_format or back_info.is_log_format,
            detected_log_type=front_info.detected_log_type or back_info.detected_log_type,
            pair_mode=True,
            warning_lines=warnings,
        )

        result_lines = [
            "Source: Split lens pair",
            f"Front: {Path(front_video).name}",
            f"Back:  {Path(back_video).name}",
            "",
            "=== Shared Pairing ===",
            f"Resolution: {summary.width}x{summary.height} per lens",
            f"FPS: {shared_fps:.2f}",
            f"Duration: {_format_duration(shared_duration)} ({shared_duration:.1f}s shared)",
            f"Frames: {shared_frames} shared",
            "",
            "=== Recommendations ===",
            f"Extraction Interval: {shared_interval}s",
            f"Estimated Frame Pairs @ {shared_interval}s: "
            f"{max(1, int(shared_duration / shared_interval))}",
        ]
        if warnings:
            result_lines.extend(["", "=== Warnings ===", *warnings])

        app.current_video_info = summary
        app.current_video_path = None
        app.after(0, lambda: _update_analyze_results(app, "\n".join(result_lines)))
        app.log("Split-pair analysis complete")
    except Exception as e:
        app.log(f"Error: {e}")
    finally:
        app.after(0, lambda: app.analyze_run_btn.configure(state="normal"))


# ======================================================================
#  2. EXTRACT (with video queue)
# ======================================================================

def _build_extract_section(app, parent):
    sec = Section(parent, "Frame Extraction")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # -- initialise queue state on app --
    if HAS_PREP360:
        app.video_queue = VideoQueue()
    else:
        app.video_queue = None
    app.extract_queue_processing = False
    app.extract_current_item_id = None
    app.extract_planned_frames: List[PlannedFrame] = []
    app.queue_item_widgets = {}

    # -- Analysis (collapsible subsection inside Frame Extraction) --
    analysis_sec = CollapsibleSection(c, "Analysis", expanded=True, core=True)
    analysis_sec.pack(fill="x", pady=(0, 6), padx=2)
    # Static subtitle in header
    ctk.CTkLabel(
        analysis_sec.header, text="analyze the selected source to see estimates",
        anchor="w", text_color=COLOR_TEXT_DIM, font=FONT_TEXT_STATUS
    ).pack(side="left", padx=(8, 0))
    ac = analysis_sec.content

    analysis_row = ctk.CTkFrame(ac, fg_color="transparent")
    analysis_row.pack(fill="x", pady=3, padx=4)

    app.analyze_run_btn = ctk.CTkButton(
        analysis_row, text="Analyze", command=lambda: _run_analyze(app),
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12), width=90,
    )
    app.analyze_run_btn.pack(side="left", padx=(0, 6), anchor="n")

    app.analyze_results = ctk.CTkTextbox(analysis_row, height=160,
                                         font=ctk.CTkFont(family="Consolas", size=11))
    app.analyze_results.pack(side="left", fill="x", expand=True)
    app.analyze_results.insert("1.0", _analysis_placeholder_text(app))

    # -- live estimate (above sliders) --
    app.current_video_info = None
    app.extract_estimate_label = ctk.CTkLabel(
        c, text="", anchor="w", justify="left",
        text_color=COLOR_TEXT_DIM, font=FONT_TEXT_STATUS)
    app.extract_estimate_label.pack(fill="x", padx=8, pady=(0, 4))

    # -- extraction settings --
    int_frame = ctk.CTkFrame(c, fg_color="transparent")
    int_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(int_frame, text="Every:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.extract_interval_var = ctk.DoubleVar(value=2.0)
    def _fmt_interval(v):
        v = float(v)
        fps = 1.0 / v if v > 0 else 0
        app.extract_interval_label.configure(text=f"{v:.1f}s ({fps:.1f} fps)")
    app.extract_interval_label = ctk.CTkLabel(int_frame, text="2.0s (0.5 fps)", width=0,
                                              font=FONT_TEXT_MONO_VALUE)
    app.extract_interval_label.pack(side="right")
    _int_slider = ctk.CTkSlider(int_frame, from_=0.1, to=10,
                                variable=app.extract_interval_var,
                                command=_fmt_interval)
    _int_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(_int_slider, "Time between extracted frames.\n"
            "Lower = more frames, larger dataset.\n"
            "2.0s is typical for walking-speed capture.")

    qual_frame = ctk.CTkFrame(c, fg_color="transparent")
    qual_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(qual_frame, text="Quality:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.extract_quality_var = ctk.IntVar(value=95)
    app.extract_quality_label = ctk.CTkLabel(qual_frame, text="95", width=40,
                                             font=FONT_TEXT_MONO_VALUE)
    app.extract_quality_label.pack(side="right")
    ctk.CTkSlider(qual_frame, from_=70, to=100, variable=app.extract_quality_var,
                  command=lambda v: app.extract_quality_label.configure(text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))

    # -- Rows 3–4: grid with left/right halves --
    # Left half: Start/End, Format+PNG    Right half: Sharpness radios, Scene Detection + Auto-geotag
    settings_grid = ctk.CTkFrame(c, fg_color="transparent")
    settings_grid.pack(fill="x", pady=3, padx=6)
    settings_grid.grid_columnconfigure(0, weight=2, uniform="half")   # left ~40%
    settings_grid.grid_columnconfigure(1, weight=3, uniform="half")   # right ~60%

    # Row 0, Col 0: Start/End
    time_grp = ctk.CTkFrame(settings_grid, fg_color="transparent")
    time_grp.grid(row=0, column=0, sticky="w", pady=(0, 8))
    _start_lbl = ctk.CTkLabel(time_grp, text="Start:", font=ctk.CTkFont(weight="bold"))
    _start_lbl.pack(side="left")
    Tooltip(_start_lbl, "Trim the extraction window.\n"
            "Accepts seconds (45.5), MM:SS (1:30),\n"
            "or HH:MM:SS (1:02:30).")
    app.extract_start_entry = ctk.CTkEntry(time_grp, width=55, placeholder_text="0:00")
    app.extract_start_entry.pack(side="left", padx=(4, 8))
    ctk.CTkLabel(time_grp, text="End:", font=ctk.CTkFont(weight="bold")).pack(side="left")
    app.extract_end_entry = ctk.CTkEntry(time_grp, width=55, placeholder_text="end")
    app.extract_end_entry.pack(side="left", padx=(4, 0))

    # Row 0, Col 1: Sharpness radios
    sharp_grp = ctk.CTkFrame(settings_grid, fg_color="transparent")
    sharp_grp.grid(row=0, column=1, sticky="w", pady=(0, 8))
    ctk.CTkLabel(sharp_grp, text="Sharpness:", font=ctk.CTkFont(weight="bold")).pack(side="left")
    app.extract_scoring_var = ctk.StringVar(value="laplacian")
    for val, label in [("tenengrad", "Tenengrad"), ("laplacian", "Laplacian"), ("none", "None")]:
        ctk.CTkRadioButton(
            sharp_grp, text=label, variable=app.extract_scoring_var,
            value=val, width=0, radiobutton_width=16, radiobutton_height=16,
            command=lambda: _update_sharpness_ui(app),
        ).pack(side="left", padx=(8, 0))

    # Row 1, Col 0: Format
    fmt_grp = ctk.CTkFrame(settings_grid, fg_color="transparent")
    fmt_grp.grid(row=1, column=0, sticky="w")
    ctk.CTkLabel(fmt_grp, text="Format:").pack(side="left", padx=(0, 4))
    app.extract_format_var = ctk.StringVar(value="jpg")
    ctk.CTkRadioButton(fmt_grp, text="JPEG", variable=app.extract_format_var,
                       value="jpg", width=0, radiobutton_width=16, radiobutton_height=16
                       ).pack(side="left", padx=(0, 9))
    ctk.CTkRadioButton(fmt_grp, text="PNG", variable=app.extract_format_var,
                       value="png", width=0, radiobutton_width=16, radiobutton_height=16
                       ).pack(side="left")

    # Row 1, Col 1: Scene Detection + Auto-geotag
    bottom_right = ctk.CTkFrame(settings_grid, fg_color="transparent")
    bottom_right.grid(row=1, column=1, sticky="ew")
    app.extract_scene_var = ctk.BooleanVar(value=True)
    app.extract_scene_cb = ctk.CTkCheckBox(
        bottom_right, text="Scene Detection",
        variable=app.extract_scene_var, width=0,
        command=lambda: _update_sharpness_ui(app),
    )
    app.extract_scene_cb.pack(side="left")
    Tooltip(app.extract_scene_cb,
            "Split time windows at scene boundaries so both\n"
            "sides of a cut get a sharp representative frame.\n\n"
            "Requires a sharpness method (Laplacian or Tenengrad)\n"
            "since scene detection runs during the scoring loop.")
    app.extract_geotag_enabled_var = ctk.BooleanVar(value=True)
    _gt = ctk.CTkCheckBox(bottom_right, text="Auto-geotag from SRT",
                           variable=app.extract_geotag_enabled_var, width=0)
    _gt.pack(side="left", padx=(20, 0))
    Tooltip(_gt,
            "After extraction, automatically write GPS coordinates,\n"
            "altitude, focal length, and capture datetime into each\n"
            "frame's EXIF metadata using DJI SRT telemetry.\n\n"
            "The SRT file is auto-detected by matching the video\n"
            "filename (e.g. my_video.mp4 → my_video.SRT).\n"
            "Override in Advanced → Metadata if auto-detect fails.\n\n"
            "Requires exiftool on PATH (https://exiftool.org).")
    gpu_label = ctk.CTkLabel(
        bottom_right,
        text="GPU ready" if _HAS_CUDACODEC else "CPU only",
        text_color="#4CAF50" if _HAS_CUDACODEC else COLOR_TEXT_DIM,
        font=ctk.CTkFont(size=11),
    )
    gpu_label.pack(side="right", padx=(8, 12))
    Tooltip(gpu_label,
            "GPU acceleration for sharpest-frame scoring.\n"
            "Requires CUDA OpenCV with cudacodec (NVDEC).\n"
            "Falls back to CPU automatically when unavailable."
            if _HAS_CUDACODEC else
            "Install CUDA OpenCV for GPU-accelerated extraction.\n"
            "See benchmarks/BENCHMARK_RESULTS.md for setup.")

    # -- 360 Processing (collapsible, inside Frame Extraction) --
    _build_fisheye_section(app, c)

    # -- Primary action: Extract / Add to Queue / Stop --
    action_row = ctk.CTkFrame(c, fg_color="transparent")
    action_row.pack(fill="x", pady=(16, 4), padx=6)

    app.extract_run_btn = ctk.CTkButton(
        action_row, text="Extract", command=lambda: _run_extract_single(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.extract_run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    app.extract_queue_add_btn = ctk.CTkButton(
        action_row, text="Add to Queue", command=lambda: _add_current_to_queue(app),
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12), height=38, width=110,
    )
    app.extract_queue_add_btn.pack(side="left", padx=(0, 4))

    app.queue_run_btn = ctk.CTkButton(
        action_row, text="Process Queue", command=lambda: _run_extract_queue(app),
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=38, width=120,
    )
    app.queue_run_btn.pack(side="left", padx=(0, 4))

    app.extract_stop_btn = ctk.CTkButton(
        action_row, text="Stop", command=lambda: _queue_stop(app),
        fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
        font=ctk.CTkFont(size=12), height=38, width=70,
    )
    app.extract_stop_btn.pack(side="left")
    app.extract_stop_btn.pack_forget()

    app.extract_progress_bar = ctk.CTkProgressBar(c, height=8)
    app.extract_progress_bar.pack(fill="x", padx=8, pady=(0, 4))
    app.extract_progress_bar.set(0)

    # -- Batch Queue --
    q_sec = CollapsibleSection(c, "Batch Queue", expanded=True)
    q_sec.pack(fill="x", pady=(6, 0), padx=2)
    qc = q_sec.content

    # queue controls
    qctrl = ctk.CTkFrame(qc, fg_color="transparent")
    qctrl.pack(fill="x", pady=(2, 4))
    app.queue_add_videos_btn = ctk.CTkButton(
        qctrl, text="Add Videos", width=80,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _queue_add_videos(app))
    app.queue_add_videos_btn.pack(side="left", padx=(0, 4))
    app.queue_add_folder_btn = ctk.CTkButton(
        qctrl, text="Add Folder", width=80,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _queue_add_folder(app))
    app.queue_add_folder_btn.pack(side="left", padx=(0, 4))
    app.queue_remove_btn = ctk.CTkButton(
        qctrl, text="Remove", width=60,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _queue_remove_selected(app))
    app.queue_remove_btn.pack(side="left", padx=(0, 4))
    app.queue_clear_done_btn = ctk.CTkButton(
        qctrl, text="Clear Done", width=72,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _queue_clear_done(app))
    app.queue_clear_done_btn.pack(side="left")

    # queue list — starts at minimal height, grows with content
    app.queue_scroll = ctk.CTkScrollableFrame(qc, height=0, fg_color="transparent")
    app.queue_scroll.pack(fill="x", pady=(0, 4))

    app.queue_stats_label = ctk.CTkLabel(qc, text="Queue: 0 pending, 0 done",
                                         text_color="gray", font=ctk.CTkFont(size=10))
    app.queue_stats_label.pack(anchor="w")

    # refresh display after build
    _queue_refresh(app)

    # -- wire up live estimate updates --
    _est = lambda *_a: _update_estimate(app)
    app.extract_scoring_var.trace_add("write", _est)
    app.extract_scene_var.trace_add("write", _est)
    app.extract_interval_var.trace_add("write", _est)
    app.extract_quality_var.trace_add("write", _est)
    app.extract_format_var.trace_add("write", _est)
    # Start/end entries don't have trace — use KeyRelease instead
    app.extract_start_entry.bind("<KeyRelease>", _est)
    app.extract_end_entry.bind("<KeyRelease>", _est)
    _update_sharpness_ui(app)


# ── SRT geotagging helper ─────────────────────────────────────────────

def _resolve_ground_elevation(app, srt_path):
    """Pick the altitude datum (metres) for geotagging, or None.

    A manual value in the Ground field wins. Otherwise the takeoff elevation is
    auto-derived from the FIRST clip of this run and reused for every later clip
    (cached on ``app._geotag_ground_lock``), so a multi-clip shoot stays on one
    vertical datum despite the drone re-zeroing its barometer between clips.
    Returns None only when no rel_alt is available (then raw abs_alt is used).
    """
    raw = app.metadata_ground_elev_entry.get().strip()
    if raw:
        try:
            g = float(raw)
            app.log(f"Geotag datum: {g:.2f} m (manual)")
            return g
        except ValueError:
            app.log(f"Geotag: invalid Ground value '{raw}' — auto-deriving instead")

    locked = getattr(app, "_geotag_ground_lock", None)
    if locked is not None:
        return locked

    try:
        g = derive_ground_elevation(parse_srt(srt_path))
    except Exception:
        g = None
    app._geotag_ground_lock = g
    if g is not None:
        app.log(f"Geotag datum: {g:.2f} m (auto, from {Path(srt_path).name})")
    return g


def _maybe_geotag(app, video_path, output_dir):
    """Auto-geotag extracted frames if SRT Geotag is enabled.

    Tries explicit SRT path from the GUI first, then auto-detects by stem.
    Uses manifest-based geotagging (reads extraction_manifest.json).
    Falls back to interval-based if no manifest exists.
    """
    if not app.extract_geotag_enabled_var.get():
        return
    if app.cancel_flag.is_set():
        return

    # Resolve SRT path: explicit entry from Metadata section > auto-detect
    srt_path = app.metadata_srt_entry.get().strip()
    if not srt_path:
        srt_path = find_srt_for_video(video_path)
    if not srt_path:
        app.log("SRT geotag: no matching .SRT file found — skipping")
        return

    app.log(f"Geotagging with: {Path(srt_path).name}")

    ground = _resolve_ground_elevation(app, srt_path)

    # Try manifest-based first, fall back to interval
    manifest = output_dir / "extraction_manifest.json"
    if manifest.exists():
        result = geotag_from_manifest(str(output_dir), srt_path, ground_elevation=ground)
    else:
        interval = app.extract_interval_var.get()
        start = app.extract_start_entry.get().strip()
        start_sec = _parse_time(start) if start else 0.0
        if start_sec is None:
            start_sec = 0.0
        result = geotag_from_interval(str(output_dir), srt_path, interval, start_sec,
                                      ground_elevation=ground)

    if result.success:
        app.log(f"Geotagged {result.tagged_count}/{result.total_frames} frames"
                f" ({result.skipped_count} skipped — no GPS)")
    else:
        for err in result.errors:
            app.log(f"Geotag error: {err}")


# ── single-video extract ──────────────────────────────────────────────

def _add_current_to_queue(app):
    """Add the current video to the batch queue with a snapshot of GUI settings."""
    if _split_source_enabled(app):
        app.log("Batch queue is not available for split lens video pairs. Use Extract instead.")
        return
    video = app.analyze_video_entry.get().strip()
    if not video:
        app.log("Select a video first")
        return
    if not Path(video).exists():
        app.log(f"Video not found: {video}")
        return
    if not app.video_queue:
        return
    settings = _snapshot_settings(app)
    added = app.video_queue.add_video(video, settings=settings)
    if added:
        app.log(f"Queued: {Path(video).name}  [{settings.summary()}]")
    else:
        app.log(f"Already in queue: {Path(video).name}")
    _queue_refresh(app)


def _run_extract_single(app):
    """Extract frames from the currently analyzed video directly (no queue)."""
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    if app.is_running:
        app.log("A process is already running.")
        return

    split_mode = _split_source_enabled(app)
    if split_mode:
        front_video = app.split_front_video_entry.get().strip()
        back_video = app.split_back_video_entry.get().strip()
        output = app.split_output_entry.get().strip()
        if not front_video or not back_video:
            app.log("Error: Select both front and back split videos first")
            return
        if not Path(front_video).exists():
            app.log(f"Error: Front video not found: {front_video}")
            return
        if not Path(back_video).exists():
            app.log(f"Error: Back video not found: {back_video}")
            return
        if not output:
            app.log("Error: Select a clip folder for paired extraction")
            return
    else:
        video = app.analyze_video_entry.get().strip()
        if not video:
            app.log("Error: Select a video first")
            return
        if not Path(video).exists():
            app.log(f"Error: Video not found: {video}")
            return
        output = app.extract_output_entry.get()
        if not output:
            app.log("Error: Please select output folder")
            return

    app.cancel_flag.clear()
    app.is_running = True
    app.extract_run_btn.configure(state="disabled")
    app.extract_stop_btn.pack(side="left")
    if hasattr(app, "extract_progress_bar"):
        app.extract_progress_bar.set(0)

    if split_mode:
        threading.Thread(
            target=_extract_split_pair_worker,
            args=(app, front_video, back_video, output),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_extract_single_worker, args=(app, video, output),
            daemon=True,
        ).start()


def _extract_split_pair_worker(app, front_video, back_video, clip_root):
    try:
        _paired_split_video_worker(app, front_video, back_video, clip_root)
    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: _extract_single_done(app))


def _extract_single_worker(app, video_path, base_output):
    """Run extraction for a single video directly — no queue involvement."""
    try:
        import time
        t_start = time.perf_counter()
        app._geotag_ground_lock = None   # fresh altitude datum per run

        video_name = Path(video_path).stem
        output_dir = Path(base_output) / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        sharpness = _get_sharpness_method(app)
        scene = _get_scene_detection(app)
        interval = app.extract_interval_var.get()
        quality = app.extract_quality_var.get()
        fmt = app.extract_format_var.get()

        start = app.extract_start_entry.get().strip()
        end = app.extract_end_entry.get().strip()
        start_sec = _parse_time(start) if start else None
        end_sec = _parse_time(end) if end else None
        settings = _snapshot_settings(app)

        # Detailed header
        app.log(f"\n{'='*50}")
        app.log(f"Processing: {Path(video_path).name}")
        app.log(f"Output: {output_dir}")
        if sharpness != "none":
            mode_display = f"sharpest ({sharpness}" + (", scene-aware" if scene else "") + ")"
        else:
            mode_display = "fixed"
        app.log(f"Mode: {mode_display}, Interval: {interval:.1f}s")
        app.log(f"Format: {fmt} q{quality}"
                + (f", Time range: {start or '0:00'}\u2013{end or 'end'}" if start or end else ""))

        _last_logged_pct = [-1]  # mutable for closure

        def progress(curr, total, msg):
            if total > 0 and hasattr(app, "extract_progress_bar"):
                app.after(0, lambda p=curr/total: app.extract_progress_bar.set(p))
            if msg:
                # Throttle log output: only log at 10% intervals for analysis,
                # but always log non-percentage messages (extraction progress, etc.)
                pct = int(curr / total * 100) if total > 0 else -1
                if pct >= 0 and "Analyzing:" in msg:
                    if pct // 10 == _last_logged_pct[0] // 10 and pct != 0:
                        return  # skip — same 10% band
                    _last_logged_pct[0] = pct
                app.after(0, lambda m=msg: app.log(m))

        used_sharpest = False
        app.log("\nExtraction...")
        t_extract = time.perf_counter()

        if sharpness != "none":
            used_sharpest = True
            sharp_cfg = SharpestConfig(
                interval=interval,
                quality=quality,
                output_format=fmt,
                start_sec=start_sec,
                end_sec=end_sec,
                scoring_method=sharpness,
                scene_detection=scene,
            )
            sharp_ext = SharpestExtractor()
            sharp_result = sharp_ext.extract(
                video_path, str(output_dir), sharp_cfg,
                progress_callback=progress,
                cancel_check=app.cancel_flag.is_set,
                log=app.log,
            )
            from prep360.core.extractor import ExtractionResult
            if sharp_result.success:
                result = ExtractionResult(
                    success=True,
                    frame_count=sharp_result.frames_extracted,
                    output_dir=str(output_dir),
                    frames=sharp_result.frame_paths,
                )
                app.log(f"Analyzed {sharp_result.total_frames_analyzed} frames, "
                        f"selected {sharp_result.frames_extracted} sharpest")
                if sharp_result.gpu_accelerated:
                    app.log("  Acceleration: GPU (NVDEC + CUDA scoring)")
                else:
                    app.log("  Acceleration: CPU")
            else:
                result = ExtractionResult(
                    success=False, frame_count=0,
                    output_dir=str(output_dir), frames=[],
                    error=sharp_result.error,
                )
        else:
            config = ExtractionConfig(
                interval=interval,
                mode=ExtractionMode.FIXED,
                quality=quality,
                output_format=fmt,
            )
            if start_sec is not None:
                config.start_sec = start_sec
            if end_sec is not None:
                config.end_sec = end_sec

            extractor = FrameExtractor()
            result = extractor.extract(
                video_path, str(output_dir), config,
                progress_callback=progress,
            )

        extract_elapsed = time.perf_counter() - t_extract

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        if result.success:
            app.log(f"  {result.frame_count} frames extracted ({extract_elapsed:.1f}s)")

            final_count = result.frame_count

            # SRT geotag
            _maybe_geotag(app, video_path, output_dir)

            total_elapsed = time.perf_counter() - t_start

            app.log(f"\nDone: {final_count} frames in {total_elapsed:.1f}s")
            app.log(f"{'='*50}")

            # Record activity for Recent Activity view
            app.record_activity(
                operation="extract",
                input_path=video_path,
                output_path=str(output_dir),
                details={
                    "frame_count": final_count,
                    "video_file": Path(video_path).name,
                },
            )
        else:
            app.log(f"Error: {result.error}")

    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: _extract_single_done(app))


def _extract_single_done(app):
    app.is_running = False
    app.extract_run_btn.configure(state="normal")
    app.extract_stop_btn.pack_forget()
    if hasattr(app, "extract_progress_bar"):
        app.extract_progress_bar.set(1.0)
    _update_source_mode_ui(app)


# ── queue helpers ─────────────────────────────────────────────────────

def _queue_add_videos(app):
    if _split_source_enabled(app):
        app.log("Batch queue is disabled while Use Split Lens Videos is active.")
        return
    files = filedialog.askopenfilenames(
        title="Select Videos",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")],
    )
    if files and app.video_queue:
        settings = _snapshot_settings(app)
        n = app.video_queue.add_videos(list(files), settings=settings)
        app.log(f"Added {n} videos to queue  [{settings.summary()}]")
        _queue_refresh(app)


def _queue_add_folder(app):
    if _split_source_enabled(app):
        app.log("Batch queue is disabled while Use Split Lens Videos is active.")
        return
    folder = filedialog.askdirectory(title="Select Folder with Videos")
    if folder and app.video_queue:
        settings = _snapshot_settings(app)
        n = app.video_queue.add_folder(folder, settings=settings)
        app.log(f"Added {n} videos from folder  [{settings.summary()}]")
        _queue_refresh(app)


def _queue_remove_selected(app):
    if not app.video_queue:
        return
    for item_id, w in list(app.queue_item_widgets.items()):
        if w.get("selected", False):
            app.video_queue.remove_item(item_id)
    _queue_refresh(app)


def _queue_clear_done(app):
    if app.video_queue:
        app.video_queue.clear_completed()
        _queue_refresh(app)


def _queue_refresh(app):
    """Rebuild queue item widgets and resize the scroll area to fit."""
    for w in app.queue_item_widgets.values():
        if "frame" in w:
            w["frame"].destroy()
    app.queue_item_widgets = {}

    if not app.video_queue:
        return

    for item in app.video_queue.items:
        _queue_create_item(app, item)

    # Hide scroll area when empty, grow with items up to 5 visible
    n = len(app.video_queue.items)
    if n == 0:
        app.queue_scroll.pack_forget()
    else:
        row_h = 34
        max_visible = 5
        target_h = min(n, max_visible) * row_h
        app.queue_scroll.configure(height=target_h)
        app.queue_scroll.pack(fill="x", pady=(0, 4))

    stats = app.video_queue.get_stats()
    app.queue_stats_label.configure(
        text=(f"Queue: {stats['pending']} pending, {stats['processing']} processing, "
              f"{stats['done']} done, {stats['error']} errors")
    )

    # Process Queue button: primary when there's work, muted when empty
    if stats["pending"] > 0:
        app.queue_run_btn.configure(
            fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H)
    else:
        app.queue_run_btn.configure(
            fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H)


_STATUS_COLORS = {
    "pending":    "#888888",
    "processing": "#FFA500",
    "done":       "#4CAF50",
    "error":      "#F44336",
    "cancelled":  "#9E9E9E",
}


def _queue_create_item(app, item):
    frame = ctk.CTkFrame(app.queue_scroll, fg_color="#2b2b2b", corner_radius=5)
    frame.pack(fill="x", pady=2, padx=2)

    selected = False

    def toggle(event=None):
        nonlocal selected
        selected = not selected
        app.queue_item_widgets[item.id]["selected"] = selected
        frame.configure(fg_color="#3d5a80" if selected else "#2b2b2b")

    frame.bind("<Button-1>", toggle)

    color = _STATUS_COLORS.get(item.status, "#888")
    ctk.CTkLabel(frame, text="\u25cf", text_color=color, width=18).pack(side="left", padx=(8, 4))

    # Truncate long filenames — the label stays fixed width
    display_name = item.filename
    if len(display_name) > 28:
        display_name = display_name[:12] + "…" + display_name[-12:]
    name_lbl = ctk.CTkLabel(frame, text=display_name, anchor="w", width=160)
    name_lbl.pack(side="left", padx=(0, 6))
    name_lbl.bind("<Button-1>", toggle)

    status_text = item.status.capitalize()
    if item.status == "processing":
        status_text = f"Processing {item.progress}%"
    elif item.status == "done":
        status_text = f"Done ({item.frame_count} frames)"

    st_lbl = ctk.CTkLabel(frame, text=status_text, text_color=color, width=110)
    st_lbl.pack(side="left")

    progress_bar = None
    if item.status == "processing":
        progress_bar = ctk.CTkProgressBar(frame, width=80, height=8)
        progress_bar.set(item.progress / 100)
        progress_bar.pack(side="left", padx=(4, 0))

    app.queue_item_widgets[item.id] = {
        "frame": frame,
        "status_label": st_lbl,
        "progress": progress_bar,
        "selected": selected,
    }


def _queue_update_item(app, item_id):
    item = app.video_queue.get_item(item_id)
    w = app.queue_item_widgets.get(item_id)
    if not item or not w:
        return
    color = _STATUS_COLORS.get(item.status, "#888")
    status_text = item.status.capitalize()
    if item.status == "processing":
        status_text = f"Processing {item.progress}%"
    elif item.status == "done":
        status_text = f"Done ({item.frame_count} frames)"
    w["status_label"].configure(text=status_text, text_color=color)
    if w["progress"] and item.status == "processing":
        w["progress"].set(item.progress / 100)


# ── extract queue worker ──────────────────────────────────────────────

def _run_extract_queue(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    if app.is_running:
        app.log("A process is already running.")
        return
    if _split_source_enabled(app):
        app.log("Batch queue is not available for split lens video pairs. Use Extract instead.")
        return
    output = app.extract_output_entry.get()
    if not output:
        app.log("Error: Please select output folder")
        return
    if app.video_queue.get_pending_count() == 0:
        app.log("No pending videos in queue")
        return

    app.extract_queue_processing = True
    app.is_running = True
    app.cancel_flag.clear()
    app.extract_run_btn.configure(state="disabled")
    app.queue_run_btn.configure(state="disabled")
    app.extract_stop_btn.pack(side="left")
    if hasattr(app, "extract_progress_bar"):
        app.extract_progress_bar.set(0)

    threading.Thread(target=_extract_queue_worker, args=(app,), daemon=True).start()


def _extract_queue_worker(app):
    try:
        import time
        t_queue_start = time.perf_counter()
        items_done = 0
        total_frames = 0
        app._geotag_ground_lock = None   # one altitude datum for the whole batch

        while app.extract_queue_processing and not app.cancel_flag.is_set():
            item = app.video_queue.get_next_pending()
            if not item:
                break

            app.extract_current_item_id = item.id
            app.video_queue.set_processing(item.id)
            app.after(0, lambda: _queue_refresh(app))

            t_item = time.perf_counter()
            app.log(f"\n{'='*50}")
            app.log(f"Processing: {item.filename}")

            try:
                # Use per-item settings (or fall back to current GUI for legacy items)
                s = item.settings or _snapshot_settings(app)
                interval = s.interval
                quality = s.quality
                fmt = s.format
                start_sec = s.start_sec
                end_sec = s.end_sec

                base_output = Path(app.extract_output_entry.get())
                video_name = Path(item.filename).stem
                output_dir = base_output / video_name
                output_dir.mkdir(parents=True, exist_ok=True)

                def progress(curr, total, msg):
                    if total > 0:
                        pct = int(curr / total * 100)
                        app.video_queue.set_progress(item.id, pct)
                        app.after(0, lambda: _queue_update_item(app, item.id))
                        if hasattr(app, "extract_progress_bar"):
                            app.after(0, lambda p=pct / 100.0: app.extract_progress_bar.set(p))
                    if msg:
                        app.after(0, lambda m=msg: app.log(m))

                app.log(f"Output: {output_dir}")
                app.log(f"Settings: {s.summary()}")

                app.log("\nExtraction...")
                t_extract = time.perf_counter()
                used_sharpest = False

                if s.sharpness_method != "none":
                    # Sharpest-frame extraction
                    used_sharpest = True
                    sharp_cfg = SharpestConfig(
                        interval=interval,
                        quality=quality,
                        output_format=fmt,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        scoring_method=s.sharpness_method,
                        scene_detection=s.scene_detection,
                    )
                    sharp_ext = SharpestExtractor()
                    sharp_result = sharp_ext.extract(
                        item.video_path, str(output_dir), sharp_cfg,
                        progress_callback=progress,
                        cancel_check=app.cancel_flag.is_set,
                        log=app.log,
                    )
                    from prep360.core.extractor import ExtractionResult
                    if sharp_result.success:
                        result = ExtractionResult(
                            success=True,
                            frame_count=sharp_result.frames_extracted,
                            output_dir=str(output_dir),
                            frames=sharp_result.frame_paths,
                        )
                        app.log(f"Analyzed {sharp_result.total_frames_analyzed} frames, "
                                f"selected {sharp_result.frames_extracted} sharpest")
                        if sharp_result.gpu_accelerated:
                            app.log("  Acceleration: GPU (NVDEC + CUDA scoring)")
                        else:
                            app.log("  Acceleration: CPU")
                    else:
                        result = ExtractionResult(
                            success=False, frame_count=0,
                            output_dir=str(output_dir), frames=[],
                            error=sharp_result.error,
                        )
                else:
                    # Fixed interval extraction
                    config = ExtractionConfig(
                        interval=interval,
                        mode=ExtractionMode.FIXED,
                        quality=quality,
                        output_format=fmt,
                    )
                    if start_sec is not None:
                        config.start_sec = start_sec
                    if end_sec is not None:
                        config.end_sec = end_sec

                    extractor = FrameExtractor()
                    result = extractor.extract(
                        item.video_path, str(output_dir), config,
                        progress_callback=progress,
                    )

                extract_elapsed = time.perf_counter() - t_extract

                if app.cancel_flag.is_set():
                    app.video_queue.set_cancelled(item.id)
                    app.log(f"Cancelled: {item.filename}")
                    break

                if result.success:
                    app.log(f"  {result.frame_count} frames extracted ({extract_elapsed:.1f}s)")

                    final_count = result.frame_count

                    # SRT geotag
                    _maybe_geotag(app, item.video_path, output_dir)

                    item_elapsed = time.perf_counter() - t_item
                    app.video_queue.set_done(item.id, final_count)

                    app.log(f"\nDone: {final_count} frames in {item_elapsed:.1f}s")

                    # Record activity for Recent Activity view
                    app.record_activity(
                        operation="extract",
                        input_path=item.video_path,
                        output_path=str(output_dir),
                        details={
                            "frame_count": final_count,
                            "interval": str(interval),
                            "video_file": item.filename,
                        },
                    )

                    items_done += 1
                    total_frames += final_count
                else:
                    app.video_queue.set_error(item.id, result.error or "Unknown error")
                    app.log(f"Error: {result.error}")

            except Exception as e:
                import traceback
                app.video_queue.set_error(item.id, str(e))
                app.log(f"Error: {e}")
                app.log(traceback.format_exc())

            app.after(0, lambda: _queue_refresh(app))

        queue_elapsed = time.perf_counter() - t_queue_start
        stats = app.video_queue.get_stats()
        app.log(f"\n{'='*50}")
        app.log(f"Queue complete: {items_done} items, {total_frames} frames in {queue_elapsed:.1f}s")
        app.log(f"Done: {stats['done']}, Errors: {stats['error']}, Cancelled: {stats['cancelled']}")

    except Exception as e:
        app.log(f"Queue error: {e}")
    finally:
        app.extract_queue_processing = False
        app.extract_current_item_id = None
        app.after(0, lambda: _queue_done(app))


def _queue_done(app):
    app.is_running = False
    app.extract_run_btn.configure(state="normal")
    if hasattr(app, "queue_run_btn"):
        app.queue_run_btn.configure(state="normal")
    app.extract_stop_btn.pack_forget()
    if hasattr(app, "extract_progress_bar"):
        app.extract_progress_bar.set(1.0)
    _queue_refresh(app)
    _update_source_mode_ui(app)


def _queue_stop(app):
    app.log("Stopping queue processing...")
    app.cancel_flag.set()
    app.extract_queue_processing = False


# ======================================================================
#  3. ADVANCED — power-user tools (SRT geotagging, future EXIF/XMP)
# ======================================================================

def _build_metadata_section(app, parent):
    """Advanced section — power-user tools tucked out of sight for beginners."""
    adv_sec = CollapsibleSection(parent, "Advanced",
                                 subtitle="frame quality filter, SRT geotagging",
                                 expanded=False, core=True)
    adv_sec.pack(fill="x", pady=(0, 6), padx=4)
    adv = adv_sec.content

    # -- Metadata / SRT Geotagging (collapsible inside Advanced) --
    meta_sec = CollapsibleSection(adv, "Metadata",
                                  subtitle="SRT override and standalone geotagging",
                                  expanded=False)
    meta_sec.pack(fill="x", padx=2, pady=(4, 0))
    c = meta_sec.content

    # -- SRT file override (used by auto-geotag when set) --
    srt_frame = ctk.CTkFrame(c, fg_color="transparent")
    srt_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(srt_frame, text="SRT:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.metadata_srt_entry = ctk.CTkEntry(srt_frame,
                                          placeholder_text="Auto-detect from video name...")
    app.metadata_srt_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(srt_frame, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_file_for(
                      app.metadata_srt_entry, "Select SRT File",
                      [("SRT Files", "*.srt *.SRT"), ("All Files", "*.*")]
                  )).pack(side="left")
    Tooltip(app.metadata_srt_entry,
            "Override the auto-detected SRT file.\n"
            "Leave blank to auto-detect by video filename.\n"
            "Used by both auto-geotag and standalone geotagging.")

    # -- Ground elevation datum (fixes DJI mid-flight barometric re-zeroing) --
    ge_frame = ctk.CTkFrame(c, fg_color="transparent")
    ge_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(ge_frame, text="Ground:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.metadata_ground_elev_entry = ctk.CTkEntry(ge_frame,
                                                  placeholder_text="Auto from first clip — or metres MSL")
    app.metadata_ground_elev_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(app.metadata_ground_elev_entry,
            "Altitude datum for geotagging, in metres.\n"
            "When set, each frame's altitude is written as\n"
            "height-above-takeoff + this value, so every clip\n"
            "shares ONE vertical datum — this fixes DJI re-zeroing\n"
            "its barometer between clips (frames sinking below ground).\n\n"
            "Leave blank to auto-derive from the first clip of the run\n"
            "(recommended). The value used is printed in the log.")

    # -- Standalone: geotag existing frames --
    sep = ctk.CTkFrame(c, height=1, fg_color="gray30")
    sep.pack(fill="x", padx=6, pady=(2, 6))

    ctk.CTkLabel(c, text="Apply to folder",
                 font=ctk.CTkFont(size=12, weight="bold")
                 ).pack(anchor="w", padx=6, pady=(0, 4))

    # Frames folder
    fr_frame = ctk.CTkFrame(c, fg_color="transparent")
    fr_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(fr_frame, text="Frames:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.geotag_frames_entry = ctk.CTkEntry(fr_frame, placeholder_text="Folder with extracted frames...")
    app.geotag_frames_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(fr_frame, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.geotag_frames_entry)
                  ).pack(side="left")

    # Action button
    btn_row = ctk.CTkFrame(c, fg_color="transparent")
    btn_row.pack(fill="x", pady=(8, 4), padx=6)
    app.geotag_run_btn = ctk.CTkButton(
        btn_row, text="Geotag", command=lambda: _run_geotag_standalone(app),
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12), height=38,
    )
    app.geotag_run_btn.pack(side="left", fill="x", expand=True)
    Tooltip(app.geotag_run_btn,
            "Geotag frames that were extracted without auto-geotag.\n"
            "Requires extraction_manifest.json in the frames folder\n"
            "(created automatically by prep360 during extraction).")

    # -- Frame Quality Filter (collapsible inside Advanced) --
    _build_quality_filter_section(app, adv)


def _build_quality_filter_section(app, parent):
    """Frame Quality Filter — standalone sharpness/motion filtering for existing frames."""
    fqf_sec = CollapsibleSection(parent, "Frame Quality Filter",
                                  subtitle="score and filter existing frames",
                                  expanded=False)
    fqf_sec.pack(fill="x", padx=2, pady=(4, 0))
    fc = fqf_sec.content

    # Cache for analyze results (scores stored between analyze and filter)
    app._fqf_scores = []  # List[BlurScore] — cached after Analyze
    app._fqf_folder = None  # str — folder that was analyzed

    # -- Folder picker --
    folder_row = ctk.CTkFrame(fc, fg_color="transparent")
    folder_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(folder_row, text="Folder:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fqf_folder_entry = ctk.CTkEntry(folder_row,
                                         placeholder_text="Folder of extracted frames...")
    app.fqf_folder_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(folder_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fqf_folder_entry)
                  ).pack(side="left")

    # -- Sharpness filter (collapsible, checkbox-enabled) --
    sharp_sec = CollapsibleSection(fc, "Sharpness",
                                    subtitle="reject blurry frames by Laplacian variance",
                                    expanded=True)
    sharp_sec.pack(fill="x", padx=(10, 2), pady=(6, 0))
    sc = sharp_sec.content

    app.fqf_sharp_var = ctk.BooleanVar(value=True)
    def _on_sharp_toggle():
        sharp_sec.set_active(app.fqf_sharp_var.get())
        _fqf_sync_motion_sharpness(app)
    ctk.CTkCheckBox(sc, text="Enable sharpness filter",
                    variable=app.fqf_sharp_var, width=0,
                    command=_on_sharp_toggle,
                    ).pack(pady=(2, 4), anchor="w", padx=6)
    sharp_sec.set_active(True)

    # Mode toggle: Percentile vs Absolute
    mode_row = ctk.CTkFrame(sc, fg_color="transparent")
    mode_row.pack(fill="x", pady=2, padx=6)
    ctk.CTkLabel(mode_row, text="Mode:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fqf_sharp_mode_var = ctk.StringVar(value="percentile")
    app._fqf_sharp_mode_btn = ctk.CTkSegmentedButton(
        mode_row, values=["Percentile", "Absolute"],
        command=lambda v: _fqf_on_mode_change(app, v),
    )
    app._fqf_sharp_mode_btn.set("Percentile")
    app._fqf_sharp_mode_btn.pack(side="left", padx=(4, 0))

    # Percentile slider
    pct_row = ctk.CTkFrame(sc, fg_color="transparent")
    pct_row.pack(fill="x", pady=2, padx=6)
    app._fqf_pct_label_name = ctk.CTkLabel(pct_row, text="Keep:", width=LABEL_FIELD_WIDTH, anchor="e")
    app._fqf_pct_label_name.pack(side="left")
    Tooltip(app._fqf_pct_label_name, "Keep the sharpest N% of frames.\n"
            "80% removes the worst 20%.")
    app.fqf_percentile_var = ctk.DoubleVar(value=80.0)
    app._fqf_pct_label = ctk.CTkLabel(pct_row, text="80%", width=0, font=FONT_TEXT_MONO_VALUE)
    app._fqf_pct_label.pack(side="right")
    app._fqf_pct_slider = ctk.CTkSlider(
        pct_row, from_=50, to=99, variable=app.fqf_percentile_var,
        command=lambda v: _fqf_update_pct_label(app, v),
    )
    app._fqf_pct_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))

    # Absolute threshold slider
    abs_row = ctk.CTkFrame(sc, fg_color="transparent")
    abs_row.pack(fill="x", pady=2, padx=6)
    app._fqf_abs_label_name = ctk.CTkLabel(abs_row, text="Min:", width=LABEL_FIELD_WIDTH, anchor="e",
                                            text_color=COLOR_TEXT_DISABLED)
    app._fqf_abs_label_name.pack(side="left")
    Tooltip(app._fqf_abs_label_name, "Reject any frame with Laplacian variance below this.\n"
            "50 is a safe default. Use Analyze to see your score range.")
    app.fqf_abs_threshold_var = ctk.DoubleVar(value=50.0)
    app._fqf_abs_label = ctk.CTkLabel(abs_row, text="50", width=0,
                                       font=FONT_TEXT_MONO_VALUE, text_color=COLOR_TEXT_DISABLED)
    app._fqf_abs_label.pack(side="right")
    app._fqf_abs_slider = ctk.CTkSlider(
        abs_row, from_=10, to=500, variable=app.fqf_abs_threshold_var,
        command=lambda v: _fqf_update_abs_label(app, v),
        state="disabled",
    )
    app._fqf_abs_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))

    # -- Motion filter (collapsible, checkbox-enabled) --
    mot_sec = CollapsibleSection(fc, "Motion",
                                  subtitle="thin redundant frames by optical flow",
                                  expanded=False)
    mot_sec.pack(fill="x", padx=(10, 2), pady=(4, 0))
    mc = mot_sec.content

    app.fqf_motion_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(mc, text="Enable motion filter",
                    variable=app.fqf_motion_var, width=0,
                    command=lambda: mot_sec.set_active(app.fqf_motion_var.get()),
                    ).pack(pady=(2, 4), anchor="w", padx=6)
    mot_sec.set_active(False)

    sharp_row = ctk.CTkFrame(mc, fg_color="transparent")
    sharp_row.pack(fill="x", pady=2, padx=6)
    _ms_lbl = ctk.CTkLabel(sharp_row, text="Sharpness:", width=LABEL_FIELD_WIDTH, anchor="e")
    _ms_lbl.pack(side="left")
    Tooltip(_ms_lbl, "Minimum Laplacian sharpness score.\n"
            "Auto-synced from Sharpness filter when both are enabled.")
    app.fqf_motion_sharpness_var = ctk.DoubleVar(value=50.0)
    app._fqf_mot_sharp_label = ctk.CTkLabel(sharp_row, text="50 (synced)", width=0,
                                             font=FONT_TEXT_MONO_VALUE)
    app._fqf_mot_sharp_label.pack(side="right")
    app._fqf_mot_sharp_slider = ctk.CTkSlider(
        sharp_row, from_=10, to=200, variable=app.fqf_motion_sharpness_var,
        command=lambda v: app._fqf_mot_sharp_label.configure(text=f"{int(float(v))}"),
        state="disabled",
    )
    app._fqf_mot_sharp_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))

    flow_row = ctk.CTkFrame(mc, fg_color="transparent")
    flow_row.pack(fill="x", pady=2, padx=6)
    _fl_lbl = ctk.CTkLabel(flow_row, text="Target flow:", width=LABEL_FIELD_WIDTH, anchor="e")
    _fl_lbl.pack(side="left")
    Tooltip(_fl_lbl, "Optical flow magnitude between kept frames.\n"
            "Higher = more camera movement required between selections.\n"
            "10 is typical for walking-speed capture.")
    app.fqf_motion_flow_var = ctk.DoubleVar(value=10.0)
    app._fqf_mot_flow_label = ctk.CTkLabel(flow_row, text="10", width=0,
                                            font=FONT_TEXT_MONO_VALUE)
    app._fqf_mot_flow_label.pack(side="right")
    ctk.CTkSlider(flow_row, from_=2, to=30, variable=app.fqf_motion_flow_var,
                  command=lambda v: app._fqf_mot_flow_label.configure(text=f"{int(float(v))}")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))

    # -- Safety toggle --
    safe_row = ctk.CTkFrame(fc, fg_color="transparent")
    safe_row.pack(fill="x", pady=(8, 2), padx=6)
    ctk.CTkLabel(safe_row, text="Rejected frames:").pack(side="left", padx=(0, 6))
    app.fqf_safety_var = ctk.StringVar(value="Move to _rejected/")
    _safe_btn = ctk.CTkSegmentedButton(
        safe_row, values=["Move to _rejected/", "Delete permanently"],
        variable=app.fqf_safety_var,
    )
    _safe_btn.set("Move to _rejected/")
    _safe_btn.pack(side="left")

    # -- Analysis summary (recessed container) --
    summary_frame = ctk.CTkFrame(fc, fg_color="#2a2a2a", corner_radius=4)
    summary_frame.pack(fill="x", padx=6, pady=(8, 2))
    app._fqf_summary_header = ctk.CTkLabel(
        summary_frame, text="", anchor="w", justify="left",
        font=ctk.CTkFont(size=11), text_color="#aaa")
    app._fqf_summary_header.pack(fill="x", padx=10, pady=(6, 0))
    app._fqf_summary_detail = ctk.CTkLabel(
        summary_frame, text="", anchor="w", justify="left",
        font=ctk.CTkFont(size=11), text_color="#e8a84c")
    app._fqf_summary_detail.pack(fill="x", padx=10, pady=(0, 6))

    # -- Action buttons --
    btn_row = ctk.CTkFrame(fc, fg_color="transparent")
    btn_row.pack(fill="x", pady=(6, 4), padx=6)
    app._fqf_analyze_btn = ctk.CTkButton(
        btn_row, text="Analyze", command=lambda: _fqf_run_analyze(app),
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12), height=36,
    )
    app._fqf_analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
    Tooltip(app._fqf_analyze_btn,
            "Score all frames for sharpness without changing anything.\n"
            "Results update live as you adjust the threshold slider.")

    app._fqf_filter_btn = ctk.CTkButton(
        btn_row, text="Filter", command=lambda: _fqf_run_filter(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
        state="disabled",
    )
    app._fqf_filter_btn.pack(side="left", fill="x", expand=True)
    Tooltip(app._fqf_filter_btn,
            "Run the enabled filters and move/delete rejected frames.\n"
            "Click Analyze first to see what would be filtered.")


def _fqf_sync_motion_sharpness(app):
    """Sync motion filter's sharpness floor from sharpness filter settings."""
    sharp_enabled = app.fqf_sharp_var.get()
    slider = getattr(app, "_fqf_mot_sharp_slider", None)
    label = getattr(app, "_fqf_mot_sharp_label", None)
    if not slider or not label:
        return

    if sharp_enabled:
        # Sync value from sharpness filter
        mode = app.fqf_sharp_mode_var.get()
        if mode == "absolute":
            val = app.fqf_abs_threshold_var.get()
        else:
            # For percentile mode, use the computed cutoff if scores exist
            import numpy as np
            scores = app._fqf_scores
            if scores:
                vals = [s.score for s in scores]
                val = float(np.percentile(vals, 100 - app.fqf_percentile_var.get()))
            else:
                val = 50.0  # fallback before analysis
        app.fqf_motion_sharpness_var.set(val)
        slider.configure(state="disabled")
        label.configure(text=f"{int(val)} (synced)")
    else:
        # Independent — let user control it
        slider.configure(state="normal")
        val = app.fqf_motion_sharpness_var.get()
        label.configure(text=f"{int(val)}")


def _fqf_on_mode_change(app, value):
    """Toggle between Percentile and Absolute sharpness mode."""
    is_pct = value == "Percentile"
    # Enable/disable sliders
    app._fqf_pct_slider.configure(state="normal" if is_pct else "disabled")
    app._fqf_abs_slider.configure(state="normal" if not is_pct else "disabled")
    # Dim/brighten labels
    pct_clr = "#c0c0c0" if is_pct else "#666"
    abs_clr = "#666" if is_pct else "#c0c0c0"
    app._fqf_pct_label_name.configure(text_color=pct_clr)
    app._fqf_pct_label.configure(text_color=pct_clr)
    app._fqf_abs_label_name.configure(text_color=abs_clr)
    app._fqf_abs_label.configure(text_color=abs_clr)
    app.fqf_sharp_mode_var.set("percentile" if is_pct else "absolute")
    # Update summary if scores are cached
    _fqf_update_summary(app)
    _fqf_sync_motion_sharpness(app)


def _fqf_update_pct_label(app, v):
    app._fqf_pct_label.configure(text=f"{int(float(v))}%")
    _fqf_update_summary(app)
    _fqf_sync_motion_sharpness(app)


def _fqf_update_abs_label(app, v):
    app._fqf_abs_label.configure(text=f"{int(float(v))}")
    _fqf_update_summary(app)
    _fqf_sync_motion_sharpness(app)


def _fqf_update_summary(app):
    """Update the inline summary from cached scores and current slider values."""
    scores = app._fqf_scores
    if not scores:
        return
    import numpy as np
    vals = [s.score for s in scores]
    n = len(vals)
    mn, mx, med = min(vals), max(vals), float(np.median(vals))
    header = f"{n} frames \u00b7 sharpness {mn:.0f}\u2013{mx:.0f} \u00b7 median {med:.0f}"

    mode = app.fqf_sharp_mode_var.get()
    if mode == "percentile":
        pct = app.fqf_percentile_var.get()
        cutoff = float(np.percentile(vals, 100 - pct))
        rejected = sum(1 for v in vals if v < cutoff)
        detail = f"At {pct:.0f}% keep: {rejected} would be rejected ({rejected/n*100:.1f}%)"
    else:
        thresh = app.fqf_abs_threshold_var.get()
        rejected = sum(1 for v in vals if v < thresh)
        detail = f"At threshold {thresh:.0f}: {rejected} would be rejected ({rejected/n*100:.1f}%)"

    app._fqf_summary_header.configure(text=header)
    app._fqf_summary_detail.configure(text=detail)


def _fqf_run_analyze(app):
    """Score all frames in the selected folder (background thread)."""
    folder = app.fqf_folder_entry.get().strip()
    if not folder:
        app.log("Frame Quality Filter: select a folder first")
        return
    if not Path(folder).is_dir():
        app.log(f"Frame Quality Filter: not a directory: {folder}")
        return

    app._fqf_analyze_btn.configure(state="disabled")
    app._fqf_filter_btn.configure(state="disabled")
    app._fqf_summary_header.configure(text="Scoring frames...")
    app._fqf_summary_detail.configure(text="")

    def worker():
        try:
            from prep360.core.blur_filter import BlurFilter, BlurFilterConfig
            bf = BlurFilter(BlurFilterConfig(workers=4))
            scores = bf.analyze_batch(folder, log=app.log)
            app._fqf_scores = scores
            app._fqf_folder = folder
            app.after(0, lambda: _fqf_analyze_done(app, scores))
        except Exception as e:
            app.log(f"Frame Quality Filter error: {e}")
            app.after(0, lambda: app._fqf_analyze_btn.configure(state="normal"))

    threading.Thread(target=worker, daemon=True).start()


def _fqf_analyze_done(app, scores):
    """Called on main thread after analyze completes."""
    app._fqf_analyze_btn.configure(state="normal")
    if scores:
        app._fqf_filter_btn.configure(state="normal")
        _fqf_update_summary(app)
        app.log(f"Frame Quality Filter: scored {len(scores)} frames")
    else:
        app._fqf_summary_header.configure(text="No images found in folder")
        app._fqf_summary_detail.configure(text="")


def _fqf_run_filter(app):
    """Run the enabled filters on the analyzed folder (background thread)."""
    scores = app._fqf_scores
    folder = app._fqf_folder
    if not scores or not folder:
        app.log("Frame Quality Filter: run Analyze first")
        return

    app._fqf_filter_btn.configure(state="disabled")
    app._fqf_analyze_btn.configure(state="disabled")

    def worker():
        try:
            import shutil

            import numpy as np
            folder_path = Path(folder)
            safety = app.fqf_safety_var.get()
            use_move = "move" in safety.lower() if isinstance(safety, str) else True

            rejected_dir = folder_path / "_rejected"
            total_rejected = 0
            surviving_paths = set()

            # -- Sharpness filter --
            if app.fqf_sharp_var.get() and scores:
                vals = [s.score for s in scores]
                mode = app.fqf_sharp_mode_var.get()
                if mode == "percentile":
                    pct = app.fqf_percentile_var.get()
                    cutoff = float(np.percentile(vals, 100 - pct))
                    app.log(f"  Sharpness filter: percentile {pct:.0f}%, cutoff={cutoff:.1f}")
                else:
                    cutoff = app.fqf_abs_threshold_var.get()
                    app.log(f"  Sharpness filter: absolute threshold={cutoff:.1f}")

                kept = []
                rejected = []
                for s in scores:
                    if s.score >= cutoff:
                        kept.append(s)
                    else:
                        rejected.append(s)

                if rejected:
                    if use_move:
                        rejected_dir.mkdir(parents=True, exist_ok=True)
                    for s in rejected:
                        p = folder_path / s.image_name
                        if p.exists():
                            if use_move:
                                shutil.move(str(p), str(rejected_dir / s.image_name))
                            else:
                                p.unlink()
                            total_rejected += 1

                app.log(f"  Sharpness: kept {len(kept)}, rejected {len(rejected)}")
                surviving_paths = {(folder_path / s.image_name) for s in kept}
            else:
                # All frames survive sharpness
                surviving_paths = {
                    folder_path / s.image_name for s in scores
                    if (folder_path / s.image_name).exists()
                }

            # -- Motion filter --
            if app.fqf_motion_var.get() and surviving_paths:
                from prep360.core.motion_selector import MotionSelector
                selector = MotionSelector(
                    min_sharpness=app.fqf_motion_sharpness_var.get(),
                    target_flow=app.fqf_motion_flow_var.get(),
                )
                paths_sorted = sorted(str(p) for p in surviving_paths if p.exists())
                app.log(f"  Motion filter: analyzing {len(paths_sorted)} frames...")
                sel_result = selector.select_from_paths(paths_sorted, log=app.log)
                selected_set = {f.path for f in sel_result.selected_frames}

                motion_rejected = 0
                for p in paths_sorted:
                    if p not in selected_set:
                        pp = Path(p)
                        if pp.exists():
                            if use_move:
                                rejected_dir.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(pp), str(rejected_dir / pp.name))
                            else:
                                pp.unlink()
                            motion_rejected += 1
                            total_rejected += 1

                app.log(f"  Motion: kept {sel_result.selected_count}, "
                        f"rejected {motion_rejected}")

            # Summary
            action = "moved to _rejected/" if use_move else "deleted"
            app.log(f"Frame Quality Filter complete: {total_rejected} frames {action}")

            # Clear cache since folder contents changed
            app._fqf_scores = []
            app._fqf_folder = None
            app.after(0, lambda: _fqf_filter_done(app, total_rejected))

        except Exception as e:
            import traceback
            app.log(f"Frame Quality Filter error: {e}")
            app.log(traceback.format_exc())
            app.after(0, lambda: _fqf_filter_done(app, 0))

    threading.Thread(target=worker, daemon=True).start()


def _fqf_filter_done(app, total_rejected):
    """Called on main thread after filter completes."""
    app._fqf_analyze_btn.configure(state="normal")
    app._fqf_filter_btn.configure(state="disabled")  # Need to re-analyze after filtering
    done_msg = f"Done \u2014 {total_rejected} frames filtered" if total_rejected > 0 else "Done \u2014 no frames were filtered"
    app._fqf_summary_header.configure(text=done_msg)
    app._fqf_summary_detail.configure(text="")


def _run_geotag_standalone(app):
    """Launch standalone geotagging in a background thread."""
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    frames_dir = app.geotag_frames_entry.get().strip()
    srt_path = app.metadata_srt_entry.get().strip()
    if not frames_dir:
        app.log("Error: Please select a frames folder")
        return
    if not Path(frames_dir).is_dir():
        app.log(f"Error: Not a directory: {frames_dir}")
        return
    if not srt_path:
        app.log("Error: Please select an SRT file")
        return
    if not Path(srt_path).exists():
        app.log(f"Error: SRT file not found: {srt_path}")
        return

    app.geotag_run_btn.configure(state="disabled")
    threading.Thread(
        target=_geotag_standalone_worker,
        args=(app, frames_dir, srt_path),
        daemon=True,
    ).start()


def _geotag_standalone_worker(app, frames_dir, srt_path):
    """Worker thread for standalone geotagging."""
    try:
        frames_dir = Path(frames_dir)

        app.log(f"\n{'='*50}")
        app.log(f"Geotagging: {frames_dir.name}")
        app.log(f"SRT: {Path(srt_path).name}")

        # Show SRT summary
        srt = parse_srt(srt_path)
        summary = srt.summary()
        app.log(f"SRT: {summary['total_entries']} entries, "
                f"{summary['gps_coverage']} GPS coverage, "
                f"{summary['duration_sec']:.0f}s duration")
        if summary['focal_lengths_mm']:
            app.log(f"Focal lengths: {summary['focal_lengths_mm']}")

        # Require extraction manifest for accurate timestamp mapping
        manifest = frames_dir / "extraction_manifest.json"
        if not manifest.exists():
            app.log("Error: No extraction_manifest.json found in folder")
            app.log("Frames must be extracted with prep360 to generate a manifest")
            return

        app.log("Using extraction manifest for timestamp mapping")

        # Altitude datum: manual Ground field wins, else auto-derive from this clip
        raw = app.metadata_ground_elev_entry.get().strip()
        ground = None
        if raw:
            try:
                ground = float(raw)
                app.log(f"Geotag datum: {ground:.2f} m (manual)")
            except ValueError:
                app.log(f"Invalid Ground value '{raw}' — auto-deriving from clip")
        if ground is None:
            ground = derive_ground_elevation(srt)
            if ground is not None:
                app.log(f"Geotag datum: {ground:.2f} m (auto, from this clip)")

        result = geotag_from_manifest(str(frames_dir), srt_path, ground_elevation=ground)

        if result.success:
            app.log(f"Geotagged {result.tagged_count}/{result.total_frames} frames"
                    f" ({result.skipped_count} skipped)")
        else:
            for err in result.errors:
                app.log(f"Error: {err}")

    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app.geotag_run_btn.configure(state="normal"))


# ======================================================================
#  4. FISHEYE (DJI Osmo 360)
# ======================================================================

def _build_fisheye_section(app, parent):
    sec = CollapsibleSection(parent, "360 Processing", expanded=False, core=True)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    split_sec = CollapsibleSection(
        c,
        "Split Lenses",
        subtitle="lossless demux from a 360 container into front/back raw lens videos",
        expanded=False,
    )
    split_sec.pack(fill="x", padx=2, pady=(4, 0))
    sc = split_sec.content

    app.fisheye_split_btn = ctk.CTkButton(
        sc, text="Split Lenses", command=lambda: _run_split_lenses(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=FONT_TEXT_BTN_PRIMARY, height=HEIGHT_ACTION_BAR,
    )
    app.fisheye_split_btn.pack(fill="x", padx=6, pady=(2, 4))
    Tooltip(app.fisheye_split_btn,
            "Uses the Video Selection Input + Output paths.\n"
            "Creates raw front/back lens videos for external grading.\n"
            "After grading, enable Split Lens Videos in Video Selection.")

    reframe_sec = CollapsibleSection(
        c, "Reframing",
        subtitle="convert 360\u00b0 or fisheye to pinhole perspectives",
        expanded=False, scroll_on_expand=True,
    )
    reframe_sec.pack(fill="x", padx=2, pady=(4, 0))
    rc = reframe_sec.content

    # ── Fisheye subsection ─────────────────────────────────────────
    fish_sec = CollapsibleSection(
        rc, "Fisheye",
        subtitle="dual-fisheye \u2192 pinhole perspectives",
        expanded=False, scroll_on_expand=True,
    )
    fish_sec.pack(fill="x", padx=2, pady=(4, 0))
    fc = fish_sec.content

    # ── Tab strip: Metashape | Standard ──
    app._fisheye_tab_var = ctk.StringVar(value="metashape")
    tab_frame = ctk.CTkFrame(fc, fg_color="transparent")
    tab_frame.pack(fill="x", padx=6, pady=(4, 2))

    app._fisheye_meta_tab_btn = ctk.CTkButton(
        tab_frame, text="Metashape", width=100, height=28,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12, weight="bold"),
        command=lambda: _switch_fisheye_tab(app, "metashape"),
    )
    app._fisheye_meta_tab_btn.pack(side="left")

    app._fisheye_std_tab_btn = ctk.CTkButton(
        tab_frame, text="Standard", width=100, height=28,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _switch_fisheye_tab(app, "standard"),
    )
    app._fisheye_std_tab_btn.pack(side="left", padx=(2, 0))

    # ── Metashape tab content ──────────────────────────────────────
    app._fisheye_meta_frame = ctk.CTkFrame(fc, fg_color="transparent")
    app._fisheye_meta_frame.pack(fill="x")
    mc = app._fisheye_meta_frame

    # Input (cameras.xml)
    _row = ctk.CTkFrame(mc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Input:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_xml_entry = ctk.CTkEntry(
        _row, placeholder_text="cameras.xml...",
    )
    app.fisheye_xml_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _browse_xml(app)).pack(side="left")

    # Output (pinhole output dir)
    _row = ctk.CTkFrame(mc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_output_entry = ctk.CTkEntry(
        _row, placeholder_text="pinhole output directory...",
    )
    app.fisheye_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_output_entry)).pack(side="left")

    # Images directory
    _row = ctk.CTkFrame(mc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Images:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_images_entry = ctk.CTkEntry(
        _row, placeholder_text="fisheye images directory...",
    )
    app.fisheye_images_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_images_entry)).pack(side="left")

    # Masks directory
    _row = ctk.CTkFrame(mc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_masks_entry = ctk.CTkEntry(
        _row, placeholder_text="masks directory (optional)...",
    )
    app.fisheye_masks_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_masks_entry)).pack(side="left")

    # Face width + Format
    settings_row = ctk.CTkFrame(mc, fg_color="transparent")
    settings_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(settings_row, text="Face width", font=ctk.CTkFont(size=11)).pack(side="left")
    app.fisheye_face_width_var = ctk.StringVar(value="0")
    ctk.CTkEntry(settings_row, textvariable=app.fisheye_face_width_var,
                 width=50, placeholder_text="0").pack(side="left", padx=(4, 0))
    ctk.CTkLabel(settings_row, text="(auto)", font=ctk.CTkFont(size=10),
                 text_color=COLOR_TEXT_DIM).pack(side="left", padx=(2, 8))
    ctk.CTkLabel(settings_row, text="Format:", font=ctk.CTkFont(size=11)).pack(side="left")
    app.fisheye_format_var = ctk.StringVar(value="png")
    for fmt in ("png", "tiff", "jpg"):
        ctk.CTkRadioButton(settings_row, text=fmt, variable=app.fisheye_format_var,
                           value=fmt, width=50, font=ctk.CTkFont(size=11),
                           radiobutton_width=14, radiobutton_height=14,
                           ).pack(side="left", padx=(4, 0))

    # Force reprocess + Station/Rig
    opts_row = ctk.CTkFrame(mc, fg_color="transparent")
    opts_row.pack(fill="x", pady=3, padx=6)
    app.fisheye_force_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(opts_row, text="Force reprocess",
                    variable=app.fisheye_force_var, width=130,
                    font=ctk.CTkFont(size=11),
                    checkbox_width=16, checkbox_height=16).pack(side="left")

    app.fisheye_layout_var = ctk.StringVar(value="rig")
    ctk.CTkRadioButton(opts_row, text="Station", variable=app.fisheye_layout_var,
                       value="station", width=70, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14,
                       ).pack(side="left", padx=(16, 0))
    ctk.CTkRadioButton(opts_row, text="Rig", variable=app.fisheye_layout_var,
                       value="rig", width=50, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14,
                       ).pack(side="left", padx=(4, 0))

    # Convert button
    app.fisheye_convert_btn = ctk.CTkButton(
        mc, text="Convert to Cubefaces",
        command=lambda: _run_cubeface_convert(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_convert_btn.pack(fill="x", padx=6, pady=(6, 2))

    # ── Standard tab content ──────────────────────────────────────
    app._fisheye_std_frame = ctk.CTkFrame(fc, fg_color="transparent")
    # Hidden initially — Metashape tab is default
    sc = app._fisheye_std_frame

    # Images directory
    _row = ctk.CTkFrame(sc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Images:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_std_images_entry = ctk.CTkEntry(
        _row, placeholder_text="fisheye frames directory...",
    )
    app.fisheye_std_images_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_std_images_entry)).pack(side="left")

    # Masks directory
    _row = ctk.CTkFrame(sc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_std_masks_entry = ctk.CTkEntry(
        _row, placeholder_text="masks directory (optional)...",
    )
    app.fisheye_std_masks_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_std_masks_entry)).pack(side="left")

    # Output directory
    _row = ctk.CTkFrame(sc, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_std_output_entry = ctk.CTkEntry(
        _row, placeholder_text="perspective output directory...",
    )
    app.fisheye_std_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_std_output_entry)).pack(side="left")

    # Preset dropdown
    preset_frame = ctk.CTkFrame(sc, fg_color="transparent")
    preset_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(preset_frame, text="Preset:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    preset_labels = list(_PRESET_KEY_TO_LABEL.values()) if HAS_PREP360 else ["Pinhole 90\u00b0 \u2014 16 views"]
    # Default to the 8-view-per-lens preset
    default_label = _PRESET_KEY_TO_LABEL.get("osv-pinhole-f90-dual-16", preset_labels[0]) if HAS_PREP360 else preset_labels[0]
    app.fisheye_preset_var = ctk.StringVar(value=default_label)
    preset_combo = ctk.CTkComboBox(preset_frame, variable=app.fisheye_preset_var,
                    values=preset_labels, state="readonly")
    preset_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))

    # Crop + Quality
    cq_frame = ctk.CTkFrame(sc, fg_color="transparent")
    cq_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(cq_frame, text="Crop:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_std_crop_var = ctk.StringVar(value="1920")
    ctk.CTkComboBox(cq_frame, variable=app.fisheye_std_crop_var,
                    values=["1280", "1600", "1920"],
                    state="readonly", width=80).pack(side="left", padx=(6, 0))
    ctk.CTkLabel(cq_frame, text="Quality:").pack(side="left", padx=(12, 2))
    app.fisheye_std_quality_var = ctk.IntVar(value=95)
    app.fisheye_std_quality_label = ctk.CTkLabel(cq_frame, text="95", width=30,
                                                  font=FONT_TEXT_MONO_VALUE)
    ctk.CTkSlider(cq_frame, from_=70, to=100, variable=app.fisheye_std_quality_var,
                  width=80,
                  command=lambda v: app.fisheye_std_quality_label.configure(text=f"{int(v)}")
                  ).pack(side="left", padx=2)
    app.fisheye_std_quality_label.pack(side="left")

    # Station / Rig
    layout_row = ctk.CTkFrame(sc, fg_color="transparent")
    layout_row.pack(fill="x", pady=3, padx=6)
    app.fisheye_std_layout_var = ctk.StringVar(value="station")
    ctk.CTkRadioButton(layout_row, text="Station", variable=app.fisheye_std_layout_var,
                       value="station", width=70, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14).pack(side="left")
    ctk.CTkRadioButton(layout_row, text="Rig", variable=app.fisheye_std_layout_var,
                       value="rig", width=50, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14).pack(side="left", padx=(4, 0))

    # Reframe button
    app.fisheye_std_reframe_btn = ctk.CTkButton(
        sc, text="Reframe",
        command=lambda: _run_std_reframe(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_std_reframe_btn.pack(fill="x", padx=6, pady=(6, 2))

    # Info
    ctk.CTkLabel(sc, text="8 views per lens \u00b7 90\u00b0 FOV \u00b7 built-in equidistant calibration",
                 font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_DIM,
                 anchor="w").pack(fill="x", padx=12, pady=(0, 4))

    # ── Shared stop button for Fisheye subsection ──
    app.fisheye_stop_btn = ctk.CTkButton(
        fc, text="Stop", command=app.stop_operation,
        fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
        font=ctk.CTkFont(size=12), height=32,
    )
    # hidden initially — shown via _start_operation

    # ── Equirectangular / Spherical subsection ───────────────────────
    erp_sec = CollapsibleSection(
        rc, "Equirectangular / Spherical",
        subtitle="split 360\u00b0 panoramas into perspectives",
        expanded=False, scroll_on_expand=True,
    )
    erp_sec.pack(fill="x", padx=2, pady=(4, 0))
    ec = erp_sec.content

    # Video input (optional — for extracting ERP frames from 360 video)
    _row = ctk.CTkFrame(ec, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Video:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_osv_entry = ctk.CTkEntry(
        _row, placeholder_text=".osv / .360 / .insv file (optional)...",
    )
    app.fisheye_osv_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _browse_osv(app)).pack(side="left")

    # Frames input
    _row = ctk.CTkFrame(ec, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Frames:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_reframe_frames_entry = ctk.CTkEntry(
        _row, placeholder_text="Existing ERP frames folder (optional)...",
    )
    app.fisheye_reframe_frames_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_reframe_frames_entry)).pack(side="left")

    # Masks input
    _row = ctk.CTkFrame(ec, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_reframe_masks_entry = ctk.CTkEntry(
        _row, placeholder_text="Masks for ERP or fisheye frames (optional)...",
    )
    app.fisheye_reframe_masks_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_reframe_masks_entry)).pack(side="left")

    # Output
    _row = ctk.CTkFrame(ec, fg_color="transparent")
    _row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(_row, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_reframe_output_entry = ctk.CTkEntry(
        _row, placeholder_text="Perspective output directory...",
    )
    app.fisheye_reframe_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_folder_for(app.fisheye_reframe_output_entry)).pack(side="left")

    # Custom Calibration (collapsible)
    adv_sec = CollapsibleSection(ec, "Custom Calibration",
                                  subtitle="override built-in lens model",
                                  expanded=False)
    adv_sec.pack(fill="x", padx=2, pady=(4, 0))
    ac = adv_sec.content

    app.fisheye_calib_default_var = ctk.BooleanVar(value=True)
    app.fisheye_calib_entry = ctk.CTkEntry(ac, width=0)  # hidden storage

    calib_row = ctk.CTkFrame(ac, fg_color="transparent")
    calib_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(calib_row, text="File:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._fisheye_custom_entry = ctk.CTkEntry(
        calib_row, placeholder_text="calibration.json...")
    app._fisheye_custom_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(app._fisheye_custom_entry,
            "Load a custom calibration JSON if the built-in\n"
            "equidistant model produces warped straight lines.\n\n"
            "JSON format: {front: {K, D, image_size, rms_error},\n"
            "              back: {K, D, image_size, rms_error}}\n"
            "K = 3x3 camera matrix, D = 4 distortion coefficients.")
    ctk.CTkButton(calib_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _load_calibration(app)).pack(side="left")
    reset_btn = ctk.CTkButton(
        calib_row, text="Reset", width=50,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _reset_calibration(app))
    reset_btn.pack(side="left", padx=(4, 0))
    Tooltip(reset_btn, "Revert to the built-in equidistant fisheye model.")

    app._fisheye_calib_label = ctk.CTkLabel(
        ac, text="Using built-in calibration",
        font=ctk.CTkFont(size=10),
        text_color="#9ca3af", anchor="w")
    app._fisheye_calib_label.pack(fill="x", padx=12, pady=(0, 2))

    # Preset
    preset_frame = ctk.CTkFrame(ec, fg_color="transparent")
    preset_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(preset_frame, text="Preset:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    erp_preset_labels = list(_PRESET_KEY_TO_LABEL.values()) if HAS_PREP360 else ["Full 90\u00b0 \u2014 26 views"]
    app.fisheye_erp_preset_var = ctk.StringVar(value=erp_preset_labels[0])
    erp_preset_combo = ctk.CTkComboBox(preset_frame, variable=app.fisheye_erp_preset_var,
                    values=erp_preset_labels, state="readonly",
                    command=lambda v: _on_preset_change(app, v))
    erp_preset_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))
    Tooltip(erp_preset_combo,
            "View layout preset \u2014 controls how many pinhole\n"
            "perspectives are extracted from each fisheye frame.")

    # Preset description
    app.fisheye_preset_desc = ctk.CTkLabel(ec, text="", text_color="#9ca3af",
                                            font=ctk.CTkFont(size=10),
                                            anchor="w", justify="left")
    app.fisheye_preset_desc.pack(fill="x", pady=(0, 2), padx=12)

    # Crop + Quality
    cq_frame = ctk.CTkFrame(ec, fg_color="transparent")
    cq_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(cq_frame, text="Crop:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_crop_var = ctk.StringVar(value="1600")
    crop_combo = ctk.CTkComboBox(cq_frame, variable=app.fisheye_crop_var,
                    values=["1280", "1600", "1920"],
                    state="readonly", width=80,
                    command=lambda v: _update_fisheye_estimate(app))
    crop_combo.pack(side="left", padx=(6, 0))
    Tooltip(crop_combo, "Output resolution per pinhole crop (square).")

    ctk.CTkLabel(cq_frame, text="Quality:").pack(side="left", padx=(12, 2))
    app.fisheye_quality_var = ctk.IntVar(value=95)
    app.fisheye_quality_label = ctk.CTkLabel(cq_frame, text="95", width=30,
                                             font=FONT_TEXT_MONO_VALUE)
    qual_slider = ctk.CTkSlider(cq_frame, from_=70, to=100, variable=app.fisheye_quality_var,
                  width=80, command=lambda v: app.fisheye_quality_label.configure(text=f"{int(v)}"))
    qual_slider.pack(side="left", padx=2)
    app.fisheye_quality_label.pack(side="left")

    # Interval
    int_frame = ctk.CTkFrame(ec, fg_color="transparent")
    int_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(int_frame, text="Interval:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.fisheye_interval_var = ctk.DoubleVar(value=2.0)
    app.fisheye_interval_label = ctk.CTkLabel(int_frame, text="2.0s", width=40,
                                              font=FONT_TEXT_MONO_VALUE)
    def _on_interval_change(v):
        app.fisheye_interval_label.configure(text=f"{float(v):.1f}s")
        _update_fisheye_estimate(app)
    interval_slider = ctk.CTkSlider(int_frame, from_=0.5, to=10.0,
                                     variable=app.fisheye_interval_var,
                                     command=_on_interval_change)
    interval_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.fisheye_interval_label.pack(side="right")
    Tooltip(interval_slider,
            "Extract one frame every N seconds from the video.\n"
            "Lower = more frames, longer processing, bigger dataset.\n"
            "2.0s is a good default for walking-speed capture.")

    # Station / Rig layout
    erp_layout_row = ctk.CTkFrame(ec, fg_color="transparent")
    erp_layout_row.pack(fill="x", pady=3, padx=6)
    app.erp_layout_var = ctk.StringVar(value="station")
    ctk.CTkRadioButton(erp_layout_row, text="Station", variable=app.erp_layout_var,
                       value="station", width=70, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14).pack(side="left")
    ctk.CTkRadioButton(erp_layout_row, text="Rig", variable=app.erp_layout_var,
                       value="rig", width=50, font=ctk.CTkFont(size=11),
                       radiobutton_width=14, radiobutton_height=14).pack(side="left", padx=(4, 0))
    Tooltip(erp_layout_row,
            "Station: one subdirectory per source image.\n"
            "Rig: one subdirectory per view direction.")

    # Output estimate
    app.fisheye_estimate_label = ctk.CTkLabel(ec, text="",
                                               font=ctk.CTkFont(size=10),
                                               text_color="#9ca3af",
                                               anchor="w", justify="left")
    app.fisheye_estimate_label.pack(fill="x", padx=12, pady=(0, 4))

    # Extract & Reframe button
    app.fisheye_reframe_btn = ctk.CTkButton(
        ec, text="Extract & Reframe", command=lambda: _run_reframe(app),
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_reframe_btn.pack(fill="x", padx=6, pady=(6, 2))
    Tooltip(app.fisheye_reframe_btn,
            "Extract from video and/or reframe ERP frames into perspectives.")

    # ERP stop button
    app.fisheye_erp_stop_btn = ctk.CTkButton(
        ec, text="Stop", command=app.stop_operation,
        fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
        font=ctk.CTkFont(size=12), height=32,
    )
    # hidden initially — shown via _start_operation

    # Init preset display
    if HAS_PREP360:
        _on_preset_change(app, app.fisheye_erp_preset_var.get())


def _switch_fisheye_tab(app, tab_name):
    """Switch between Metashape and Standard tabs in the Fisheye subsection."""
    app._fisheye_tab_var.set(tab_name)
    if tab_name == "metashape":
        app._fisheye_meta_frame.pack(fill="x")
        app._fisheye_std_frame.pack_forget()
        app._fisheye_meta_tab_btn.configure(
            fg_color=COLOR_ACTION_SECONDARY, font=ctk.CTkFont(size=12, weight="bold"))
        app._fisheye_std_tab_btn.configure(
            fg_color=COLOR_ACTION_MUTED, font=ctk.CTkFont(size=12))
    else:
        app._fisheye_std_frame.pack(fill="x")
        app._fisheye_meta_frame.pack_forget()
        app._fisheye_std_tab_btn.configure(
            fg_color=COLOR_ACTION_SECONDARY, font=ctk.CTkFont(size=12, weight="bold"))
        app._fisheye_meta_tab_btn.configure(
            fg_color=COLOR_ACTION_MUTED, font=ctk.CTkFont(size=12))


def _browse_xml(app):
    """Browse for a Metashape cameras.xml file."""
    path = filedialog.askopenfilename(
        title="Select Metashape cameras.xml",
        filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")],
    )
    if path:
        app.fisheye_xml_entry.delete(0, "end")
        app.fisheye_xml_entry.insert(0, path)


def _run_cubeface_convert(app):
    """Validate inputs and launch cubeface conversion thread."""
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return

    xml_path = app.fisheye_xml_entry.get().strip()
    output_dir = app.fisheye_output_entry.get().strip()
    images_dir = app.fisheye_images_entry.get().strip()
    masks_dir = app.fisheye_masks_entry.get().strip()

    if not xml_path or not Path(xml_path).is_file():
        app.log("Error: Select a cameras.xml file")
        return
    if not output_dir:
        app.log("Error: Select a pinhole output directory")
        return
    if not images_dir or not Path(images_dir).is_dir():
        app.log("Error: Select a fisheye images directory")
        return
    if masks_dir and not Path(masks_dir).is_dir():
        app.log(f"Error: Masks directory not found: {masks_dir}")
        return

    face_width = int(app.fisheye_face_width_var.get() or "0")
    output_format = app.fisheye_format_var.get()
    force = app.fisheye_force_var.get()

    app._start_operation(app.fisheye_convert_btn, app.fisheye_stop_btn)
    threading.Thread(
        target=_cubeface_worker,
        args=(app, xml_path, images_dir, masks_dir or None,
              output_dir, face_width, output_format, force),
        daemon=True,
    ).start()


def _cubeface_worker(app, xml_path, images_dir, masks_dir,
                     output_dir, face_width, output_format, force):
    """Thread target: runs process_cubeface_sensor."""
    try:
        from prep360.core.cubeface_processing import process_cubeface_sensor

        app.log("Converting fisheye \u2192 cubefaces")
        app.log(f"  Input XML:  {xml_path}")
        app.log(f"  Images:     {images_dir}")
        if masks_dir:
            app.log(f"  Masks:      {masks_dir}")
        app.log(f"  Output:     {output_dir}")
        app.log(f"  Face width: {face_width}, Format: {output_format}")

        result = process_cubeface_sensor(
            calibration_xml=Path(xml_path),
            image_dirs=[Path(images_dir)],
            output_dir=Path(output_dir),
            face_width=face_width,
            mask_dirs=[Path(masks_dir)] if masks_dir else None,
            output_format=output_format,
            force=force,
            progress_callback=lambda msg: app.log(msg),
        )

        app.log("\nConversion complete")
        app.log(f"  Processed: {result['processed_count']}")
        app.log(f"  Skipped:   {result['skipped_count']}")
        app.log(f"  Face width: {result['face_width']}")
        app.log(f"  Output:    {result['output_dir']}")

        app.record_activity(
            operation="cubeface_convert",
            input_path=images_dir,
            output_path=output_dir,
            details={
                "processed": result["processed_count"],
                "skipped": result["skipped_count"],
                "face_width": result["face_width"],
            },
        )
    except Exception as e:
        import traceback
        app.log(f"Cubeface conversion error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_convert_btn, app.fisheye_stop_btn))


def _run_std_reframe(app):
    """Validate inputs and launch standard fisheye reframe thread."""
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return

    images_dir = app.fisheye_std_images_entry.get().strip()
    masks_dir = app.fisheye_std_masks_entry.get().strip()
    output_dir = app.fisheye_std_output_entry.get().strip()

    if not images_dir or not Path(images_dir).is_dir():
        app.log("Error: Select a fisheye images directory")
        return
    if not output_dir:
        app.log("Error: Select an output directory")
        return
    if masks_dir and not Path(masks_dir).is_dir():
        app.log(f"Error: Masks directory not found: {masks_dir}")
        return

    preset_key = _get_preset_key(app)
    preset_config = FISHEYE_PRESETS.get(preset_key)
    if preset_config is None:
        app.log(f"Error: Unknown preset '{preset_key}'")
        return

    crop_size = int(app.fisheye_std_crop_var.get())
    quality = app.fisheye_std_quality_var.get()
    config = FisheyeViewConfig(
        views=list(preset_config.views),
        crop_size=crop_size,
        quality=quality,
    )
    station_dirs = app.fisheye_std_layout_var.get() == "station"

    app._start_operation(app.fisheye_std_reframe_btn, app.fisheye_stop_btn)
    threading.Thread(
        target=_std_reframe_worker,
        args=(app, images_dir, masks_dir or None, output_dir,
              config, station_dirs, preset_key),
        daemon=True,
    ).start()


def _std_reframe_worker(app, images_dir, masks_dir, output_dir,
                        config, station_dirs, preset_key):
    """Thread target: standard fisheye reframe using FisheyeReframer."""
    try:
        calib = default_osmo360_calibration()
        app.log("Reframing fisheye \u2192 perspectives")
        app.log(f"  Preset: {preset_key} ({config.total_views()} views)")
        app.log(f"  Crop: {config.crop_size}x{config.crop_size}")
        app.log(f"  Images: {images_dir}")
        if masks_dir:
            app.log(f"  Masks: {masks_dir}")
        app.log(f"  Output: {output_dir}")

        frames_path = Path(images_dir)
        front = sorted(str(f) for f in frames_path.glob("front_*.jpg"))
        back = sorted(str(f) for f in frames_path.glob("back_*.jpg"))
        if not front or not back:
            app.log("Error: Need both front_*.jpg and back_*.jpg frames")
            return
        app.log(f"Found {len(front)} front + {len(back)} back frames")

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        pairs = list(zip(front, back))
        reframe_out = output_dir if station_dirs else str(Path(output_dir) / "images")

        app.log(f"\nReframing {len(pairs)} pairs \u2192 {config.total_views()} views each...")

        def rf_progress(current, total, msg):
            if not app.cancel_flag.is_set():
                app.log(f"  [{current}/{total}] {msg}")

        total_crops, errors = fisheye_batch_extract(
            pairs, config, calib, reframe_out,
            mask_dir=masks_dir,
            num_workers=1, progress_callback=rf_progress,
            station_dirs=station_dirs,
            log=app.log,
        )

        app.log("\nReframe complete")
        app.log(f"  Frame pairs: {len(pairs)}")
        app.log(f"  Total crops: {total_crops}")
        if errors:
            app.log(f"  Errors: {len(errors)}")

        app.record_activity(
            operation="fisheye_reframe",
            input_path=images_dir,
            output_path=output_dir,
            details={
                "frame_pairs": len(pairs),
                "total_crops": total_crops,
                "crop_size": config.crop_size,
            },
        )
    except Exception as e:
        import traceback
        app.log(f"Reframe error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_std_reframe_btn, app.fisheye_stop_btn))


def _browse_osv(app):
    path = filedialog.askopenfilename(
        title="Select OSV / 360 File",
        filetypes=[
            ("DJI 360 Files", "*.osv *.360 *.insv"),
            ("Video Files", "*.mp4 *.mov"),
            ("All Files", "*.*"),
        ],
    )
    if path:
        app.fisheye_osv_entry.delete(0, "end")
        app.fisheye_osv_entry.insert(0, path)
        threading.Thread(target=_probe_osv, args=(app, path), daemon=True).start()


def _probe_osv(app, path):
    try:
        handler = OSVHandler()
        info = handler.probe(path)
        app._fisheye_video_duration = info.duration
        app.after(0, lambda: _update_fisheye_estimate(app))
        app.log(f"Probed: {info.filename} \u2014 {info.duration:.1f}s, {info.fps:.0f}fps")
    except Exception as e:
        app.log(f"Error probing 360 source: {e}")


def _load_calibration(app):
    """Browse for a calibration JSON and update the label."""
    path = filedialog.askopenfilename(
        title="Select Calibration JSON",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
    )
    if not path:
        return
    try:
        calib = DualFisheyeCalibration.load(path)
        app.fisheye_calib_default_var.set(False)
        app.fisheye_calib_entry.delete(0, "end")
        app.fisheye_calib_entry.insert(0, path)
        app._fisheye_custom_entry.delete(0, "end")
        app._fisheye_custom_entry.insert(0, path)
        name = Path(path).name
        rms = calib.front.rms_error
        rms_str = f"RMS {rms:.3f}" if rms > 0 else "no RMS"
        app._fisheye_calib_label.configure(
            text=f"Custom: {name} ({rms_str})")
        app.log(f"Loaded calibration: {name} ({rms_str})")
    except Exception as e:
        app.log(f"Failed to load calibration: {e}")


def _reset_calibration(app):
    """Revert to built-in equidistant fisheye model."""
    app.fisheye_calib_default_var.set(True)
    app.fisheye_calib_entry.delete(0, "end")
    app._fisheye_custom_entry.delete(0, "end")
    app._fisheye_calib_label.configure(text="Using built-in calibration")


def _preset_description(preset_name, cfg):
    """Describe the view layout geometry in plain language."""
    front_views = cfg.views_for_lens("front")
    per_lens = len(front_views)
    fov = sorted(set(v.fov_deg for v in cfg.views))[0]

    # Group views by pitch tier
    pitch_groups = {}
    for v in front_views:
        p = round(v.pitch_deg)
        pitch_groups.setdefault(p, []).append(v)

    # Build tier-by-tier layout string
    tiers = []
    for pitch in sorted(pitch_groups.keys(), reverse=True):
        n = len(pitch_groups[pitch])
        if pitch > 0:
            tiers.append(f"{n} looking up ({pitch}°)")
        elif pitch < 0:
            tiers.append(f"{n} looking down ({pitch}°)")
        else:
            tiers.append(f"{n} at horizon")

    layout = ", ".join(tiers)
    fov_note = "wider crops, more overlap" if fov >= 110 else "standard perspective"
    return (
        f"{per_lens} views per lens: {layout}\n"
        f"Each crop is {fov:.0f}° FOV — {fov_note}"
    )


def _on_preset_change(app, display_label):
    preset_key = _PRESET_LABEL_TO_KEY.get(display_label)
    if not HAS_PREP360 or not preset_key or preset_key not in FISHEYE_PRESETS:
        return
    cfg = FISHEYE_PRESETS[preset_key]
    app.fisheye_preset_desc.configure(text=_preset_description(preset_key, cfg))
    _update_fisheye_estimate(app)


def _update_fisheye_estimate(app):
    """Update the dynamic output estimate label based on current settings."""
    if not HAS_PREP360:
        return
    preset_key = _get_preset_key(app)
    cfg = FISHEYE_PRESETS.get(preset_key)
    if not cfg:
        return
    views = cfg.total_views()
    crop = app.fisheye_crop_var.get()

    # If we have video duration from a probe, compute frame count estimate
    duration = getattr(app, '_fisheye_video_duration', None)
    interval = app.fisheye_interval_var.get()
    if duration and duration > 0:
        pairs = max(1, int(duration / interval))
        total = pairs * views
        app.fisheye_estimate_label.configure(
            text=f"{pairs} frame pairs \u00d7 {views} views = "
                 f"~{total} output images ({crop}x{crop})")
    else:
        app.fisheye_estimate_label.configure(
            text=f"{views} views per frame pair ({crop}x{crop})")


def _get_fisheye_calibration(app):
    if app.fisheye_calib_default_var.get():
        return default_osmo360_calibration()
    calib_path = app.fisheye_calib_entry.get()
    if not calib_path:
        raise ValueError("No calibration file selected")
    return DualFisheyeCalibration.load(calib_path)


# ── split lenses ──────────────────────────────────────────────────────

def _run_split_lenses(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    osv_path = app.analyze_video_entry.get().strip()
    output_dir = app.extract_output_entry.get().strip()
    if not osv_path:
        app.log("Error: Load a 360 video in Video Selection first")
        return
    if not Path(osv_path).exists():
        app.log(f"Error: File not found: {osv_path}")
        return
    if not output_dir:
        app.log("Error: Set an output folder in Video Selection first")
        return

    app._start_operation(app.fisheye_split_btn, app.fisheye_stop_btn)
    threading.Thread(
        target=_split_lenses_worker, args=(app, osv_path, output_dir),
        daemon=True,
    ).start()


def _find_insv_pair(insv_path: str):
    """Find the front/back pair for an Insta360 .insv file.

    Insta360 stores each lens as a separate file:
      VID_DATE_TIME_00_NNN.insv  = front lens
      VID_DATE_TIME_10_NNN.insv  = back lens

    Returns (front_path, back_path) or raises ValueError.
    """
    import re
    p = Path(insv_path)
    name = p.name
    # Match the lens identifier: _00_ (front) or _10_ (back)
    m = re.search(r'_([01]0)_(\d+)\.insv$', name, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot determine lens from filename: {name}\n"
            f"Expected pattern: VID_DATE_TIME_00_NNN.insv or _10_"
        )
    lens_id = m.group(1)
    if lens_id == "00":
        front_path = p
        back_path = p.parent / name.replace(f"_{lens_id}_", "_10_")
    else:
        back_path = p
        front_path = p.parent / name.replace(f"_{lens_id}_", "_00_")

    if not front_path.exists():
        raise FileNotFoundError(f"Front lens file not found: {front_path.name}")
    if not back_path.exists():
        raise FileNotFoundError(f"Back lens file not found: {back_path.name}")

    return str(front_path), str(back_path)


def _split_lenses_worker(app, osv_path, output_dir):
    try:
        app.log(f"\nSplitting lenses: {Path(osv_path).name}")
        is_insv = Path(osv_path).suffix.lower() == ".insv"

        # Try demuxing (works for DJI .osv/.360 and newer Insta360 .insv
        # with dual HEVC streams in one container)
        try:
            handler = OSVHandler()
            front_path, back_path = handler.demux_streams(
                osv_path, output_dir, streams="both",
            )
            app.log(f"Front lens: {front_path}")
            app.log(f"Back lens:  {back_path}")
            app.log("Split complete (lossless stream copy)")
            summary = "\n".join([
                "Split complete",
                f"  Front lens:    {front_path}",
                f"  Back lens:     {back_path}",
                "  Next:          Color-grade these in Resolve if needed,",
                "                 then enable Use Split Lens Videos and select the graded front/back exports.",
            ])
            app.log(summary)
            return
        except ValueError:
            # For older Insta360 .insv with separate files per lens,
            # demux fails ("Expected 2 HEVC streams") — fall back to pair-finding
            if not is_insv:
                raise

        # Older Insta360 format: each lens is already a separate file
        front_path, back_path = _find_insv_pair(osv_path)
        app.log(f"Front lens: {front_path}")
        app.log(f"Back lens:  {back_path}")
        summary = "\n".join([
            "Insta360 pair found (files are already separate)",
            f"  Front lens:    {front_path}",
            f"  Back lens:     {back_path}",
            "  Next:          Color-grade these in Resolve if needed,",
            "                 then enable Use Split Lens Videos and select the graded front/back exports.",
        ])
        app.log(summary)

    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_split_btn, app.fisheye_stop_btn))


# ── fisheye worker ────────────────────────────────────────────────────

def _paired_split_video_worker(app, front_video, back_video, output_dir):
    """Extract shared frame pairs from graded split lens videos."""
    import time
    t_start = time.perf_counter()

    settings = _snapshot_settings(app)
    sharpness = settings.sharpness_method
    scene = settings.scene_detection
    mode = "sharpest" if sharpness != "none" else "fixed"

    clip_root = Path(output_dir)
    if sharpness != "none":
        method_parts = [sharpness]
        if scene:
            method_parts.append("scene-aware")
        settings_summary = (
            f"sharpest/{'/'.join(method_parts)}, "
            f"{settings.interval:.1f}s windows, {settings.format}, q{settings.quality}, "
            f"{sharpness} pair scoring"
            + (" + scene detection" if scene else "")
            + ", then pair extraction"
        )
    else:
        settings_summary = f"fixed, {settings.interval:.1f}s, {settings.format}, q{settings.quality}"
    app.log("Mode: Use Split Lens Videos")
    app.log(f"Front video: {Path(front_video).name}")
    app.log(f"Back video:  {Path(back_video).name}")
    app.log(f"Clip folder: {clip_root}")
    app.log(f"Settings:    {settings_summary}")

    def paired_progress(current, total, msg):
        if hasattr(app, "extract_progress_bar"):
            progress_value = None
            msg_lower = (msg or "").lower()
            if total == 100 and current <= total:
                progress_value = current / 100.0
            elif "extracting" in msg_lower and total > 0:
                progress_value = 0.85 + 0.15 * (current / total)
            elif total > 0 and current <= total:
                progress_value = current / total
            if progress_value is not None:
                progress_value = max(0.0, min(1.0, progress_value))
                app.after(0, lambda p=progress_value: app.extract_progress_bar.set(p))
        if not app.cancel_flag.is_set():
            app.log(f"  [{current}/{total}] {msg}")

    app.log("\nExtraction...")
    t_extract = time.perf_counter()
    extractor = PairedSplitVideoExtractor()
    result = extractor.extract(
        front_video,
        back_video,
        str(clip_root),
        PairedSplitConfig(
            mode=mode,
            scoring_method=sharpness,
            scene_detection=scene,
            interval_sec=settings.interval,
            quality=settings.quality,
            output_format=settings.format,
            start_sec=settings.start_sec,
            end_sec=settings.end_sec,
        ),
        progress_callback=paired_progress,
        cancel_check=lambda: app.cancel_flag.is_set(),
        log=app.log,
    )

    extract_elapsed = time.perf_counter() - t_extract

    if not result.success:
        app.log(f"Error: {result.error}")
        return

    app.log(f"  {result.pair_count} pairs extracted ({extract_elapsed:.1f}s)")
    if result.gpu_accelerated:
        app.log("  Acceleration: GPU (NVDEC + CUDA scoring)")
    else:
        app.log("  Acceleration: CPU")

    total_elapsed = time.perf_counter() - t_start

    summary = "\n".join([
        f"Paired extraction complete ({total_elapsed:.1f}s)",
        f"  Frame pairs:   {result.pair_count}",
        f"  Front frames:  {clip_root / 'front' / 'frames'}",
        f"  Back frames:   {clip_root / 'back' / 'frames'}",
        f"  Manifest:      {clip_root / 'paired_extraction_manifest.json'}",
        "  Next:          Run Mask on this clip folder, then assemble the Metashape session below Metadata.",
    ])
    app.log(f"\n{summary}")

    def _update_ui():
        if hasattr(app, "metashape_clips_root_entry") and not app.metashape_clips_root_entry.get().strip():
            _set_entry_text(app.metashape_clips_root_entry, str(Path(output_dir).parent))
        if hasattr(app, "metashape_output_entry") and not app.metashape_output_entry.get().strip():
            _set_entry_text(app.metashape_output_entry, str(Path(output_dir).parent / "metashape"))
    app.after(0, _update_ui)


def _run_reframe(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return

    video_path = app.fisheye_osv_entry.get().strip()
    frames_dir = app.fisheye_reframe_frames_entry.get().strip()
    masks_dir = app.fisheye_reframe_masks_entry.get().strip()
    output_dir = app.fisheye_reframe_output_entry.get().strip()

    if not output_dir:
        app.log("Error: Select an output folder for reframing")
        return
    if not video_path and not frames_dir:
        app.log("Error: Reframing needs either a 360 video or an existing ERP frames folder")
        return
    if video_path and not Path(video_path).exists():
        app.log(f"Error: Video not found: {video_path}")
        return
    if frames_dir and not Path(frames_dir).is_dir():
        app.log(f"Error: Frames folder not found: {frames_dir}")
        return
    if masks_dir and not Path(masks_dir).is_dir():
        app.log(f"Error: Masks folder not found: {masks_dir}")
        return

    station_dirs = app.erp_layout_var.get() == "station"
    app._start_operation(app.fisheye_reframe_btn, app.fisheye_erp_stop_btn)
    threading.Thread(
        target=_fisheye_worker,
        args=(
            app,
            video_path or None,
            frames_dir or None,
            masks_dir or None,
            output_dir,
            station_dirs,
            app.fisheye_reframe_btn,
        ),
        daemon=True,
    ).start()


def _fisheye_worker(
    app, video_path, frames_dir, masks_dir, output_dir,
    station_dirs=False, action_button=None,
):
    try:
        calib = _get_fisheye_calibration(app)
        app.log(f"Calibration: {calib.camera_model}")

        preset_name = _get_preset_key(app)
        preset_config = FISHEYE_PRESETS.get(preset_name)
        if preset_config is None:
            app.log(f"Error: Unknown preset '{preset_name}'")
            return

        crop_size = int(app.fisheye_crop_var.get())
        quality = app.fisheye_quality_var.get()
        config = FisheyeViewConfig(
            views=list(preset_config.views),
            crop_size=crop_size,
            quality=quality,
        )

        # Determine mode
        has_video = video_path is not None
        has_masks = masks_dir is not None
        mode = "Extract & Reframe" if has_video else ("Reframe with Masks" if has_masks else "Reframe")
        app.log(f"Mode: {mode}")
        app.log(f"Preset: {preset_name} ({config.total_views()} views)")
        app.log(f"Crop: {crop_size}x{crop_size} @ Q{quality}")

        # Step 1: Extract fisheye frames from video (if video provided)
        if has_video:
            interval = app.fisheye_interval_var.get()
            extract_dir = str(Path(output_dir) / "fisheye_frames")
            app.log(f"\nExtracting fisheye frames (interval={interval:.1f}s)...")

            handler = OSVHandler()

            def osv_progress(stream_name, current, total):
                if not app.cancel_flag.is_set():
                    app.log(f"  [{stream_name}] {current}/{total}")

            frame_dict = handler.extract_frames(
                video_path, extract_dir,
                interval=interval, streams="both", quality=95,
                progress_callback=osv_progress,
            )

            front = sorted(frame_dict.get("front", []))
            back = sorted(frame_dict.get("back", []))
            if not front or not back:
                app.log("Error: No frames extracted")
                return
            app.log(f"Extracted {len(front)} front + {len(back)} back frames")
        else:
            # Use existing frames from Frames directory
            frames_path = Path(frames_dir)
            front = sorted(str(f) for f in frames_path.glob("front_*.jpg"))
            back = sorted(str(f) for f in frames_path.glob("back_*.jpg"))
            if not front and not back:
                # Try generic image names (ERP frames, not front/back pairs)
                all_frames = sorted(
                    str(f) for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    for f in frames_path.glob(ext)
                )
                if not all_frames:
                    app.log("Error: No frames found in frames folder")
                    return
                # ERP reframe path (not fisheye pairs)
                app.log(f"Found {len(all_frames)} ERP frames — using equirect reframer")
                _erp_reframe_worker(app, all_frames, masks_dir, output_dir, config, station_dirs)
                return

            if not front or not back:
                app.log("Error: Need both front_*.jpg and back_*.jpg frames")
                return
            app.log(f"Found {len(front)} front + {len(back)} back frames")

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        # Step 2: Reframe → perspectives
        pairs = list(zip(front, back))
        # Station mode: reframer creates images/ and masks/ subdirs itself
        # Flat mode: put crops in output_dir/images/
        reframe_out = output_dir if station_dirs else str(Path(output_dir) / "images")
        app.log(f"\nReframing {len(pairs)} pairs \u2192 {config.total_views()} views each...")
        if has_masks:
            app.log(f"Masks: {masks_dir}")

        def rf_progress(current, total, msg):
            if not app.cancel_flag.is_set():
                app.log(f"  [{current}/{total}] {msg}")

        total_crops, errors = fisheye_batch_extract(
            pairs, config, calib, reframe_out,
            mask_dir=masks_dir,
            num_workers=1, progress_callback=rf_progress,
            station_dirs=station_dirs,
            log=app.log,
        )

        lines = [
            f"{mode} complete",
            f"  Frame pairs:   {len(pairs)}",
            f"  Views/pair:    {config.total_views()}",
            f"  Total crops:   {total_crops}",
            f"  Crop size:     {crop_size}x{crop_size}",
            f"  Images:        {reframe_out}",
        ]
        if has_masks:
            masks_out = str(Path(output_dir) / "masks")
            lines.append(f"  Masks:         {masks_out}")
        if errors:
            lines.append(f"  Errors:        {len(errors)}")
            for err in errors[:5]:
                lines.append(f"    - {err}")

        summary = "\n".join(lines)
        app.log(f"\n{summary}")

        # Record activity for Recent Activity view
        input_src = video_path or frames_dir or ""
        app.record_activity(
            operation="reframe",
            input_path=str(input_src),
            output_path=output_dir,
            details={
                "frame_pairs": len(pairs),
                "total_crops": total_crops,
                "crop_size": crop_size,
            },
        )

    except Exception as e:
        import traceback
        app.log(f"Fisheye error: {e}")
        app.log(traceback.format_exc())
    finally:
        btn = action_button or app.fisheye_reframe_btn
        stop_btn = getattr(app, 'fisheye_erp_stop_btn', app.fisheye_stop_btn)
        app.after(0, lambda: app._stop_operation(btn, stop_btn))


def _erp_reframe_worker(app, frame_paths, masks_dir, output_dir, config, station_dirs=False):
    """Reframe ERP frames using the equirect reframer (not fisheye pairs)."""
    from prep360.core.reframer import OutputLayout, Reframer, Ring, ViewConfig

    # Build a ViewConfig from the fisheye preset's view angles
    # Extract unique rings from the fisheye views
    rings = []
    pitch_groups = {}
    for view in config.views:
        pitch = view.pitch_deg
        if pitch not in pitch_groups:
            pitch_groups[pitch] = {"count": 0, "fov": view.fov_deg}
        pitch_groups[pitch]["count"] += 1

    for pitch, info in sorted(pitch_groups.items()):
        rings.append(Ring(pitch=pitch, count=info["count"], fov=info["fov"]))

    view_config = ViewConfig(
        rings=rings,
        output_size=config.crop_size,
        jpeg_quality=config.quality,
    )
    reframer = Reframer(view_config)

    # Create a temp directory with the frames (reframe_batch expects a directory)
    frames_dir = str(Path(frame_paths[0]).parent)
    # The equirect reframer treats output_dir as the dataset ROOT and creates
    # images/ (plus per-layout subdirs) itself. The ERP layout radio is
    # station|rig: map station -> STATION, else RIG. RIG also emits
    # rig_config.json for Metashape Pro.
    output_layout = OutputLayout.STATION if station_dirs else OutputLayout.RIG

    app.log(f"\nReframing {len(frame_paths)} ERP frames...")

    def rf_progress(current, total, msg):
        if not app.cancel_flag.is_set():
            app.log(f"  [{current}/{total}] {msg}")

    result = reframer.reframe_batch(
        frames_dir, output_dir,
        mask_dir=masks_dir,
        num_workers=1,
        progress_callback=rf_progress,
        output_layout=output_layout,
        log=app.log,
    )

    images_out = str(Path(output_dir) / "images")
    lines = [
        "ERP reframing complete",
        f"  Input frames:  {result.input_count}",
        f"  Total views:   {result.output_count}",
        f"  Output root:   {output_dir}",
        f"  Images:        {images_out}",
    ]
    if output_layout == OutputLayout.RIG:
        lines.append(f"  Rig config:    {Path(output_dir) / 'rig_config.json'}")
    if masks_dir:
        masks_out = str(Path(output_dir) / "masks")
        lines.append(f"  Masks:         {masks_out}")
    if result.errors:
        lines.append(f"  Errors:        {len(result.errors)}")
        for err in result.errors[:5]:
            lines.append(f"    - {err}")

    summary = "\n".join(lines)
    app.log(f"\n{summary}")

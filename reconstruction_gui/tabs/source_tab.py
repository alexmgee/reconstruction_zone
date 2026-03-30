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

from widgets import CollapsibleSection, Tooltip

# ── prep360 core (optional) ────────────────────────────────────────────

try:
    from prep360.core import (
        VideoAnalyzer,
        FrameExtractor, ExtractionConfig, ExtractionMode,
        LUTProcessor,
        SkyFilter, SkyFilterConfig,
        OSVHandler,
        FisheyeViewConfig, FISHEYE_PRESETS,
        DualFisheyeCalibration,
        MotionSelector,
        SharpestExtractor, SharpestConfig,
    )
    from prep360.core.fisheye_reframer import (
        default_osmo360_calibration, batch_extract as fisheye_batch_extract,
    )
    from prep360.core.queue_manager import VideoQueue, ExtractionSettings
    from prep360.core.dual_fisheye_dataset import (
        DualFisheyeClipDataset,
        discover_clip_roots,
        load_clip_dataset,
        write_dual_fisheye_session_manifest,
    )
    from prep360.core.paired_split_video_extractor import (
        PairedSplitConfig,
        PairedSplitVideoExtractor,
    )
    from prep360.core.srt_parser import find_srt_for_video, parse_srt
    from prep360.core.geotagger import geotag_from_manifest, geotag_from_interval

    HAS_PREP360 = True
except ImportError:
    HAS_PREP360 = False

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
    return _PRESET_LABEL_TO_KEY.get(app.fisheye_preset_var.get(), "osv-full-f90-dual-26")


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


# ── extraction mode display labels ────────────────────────────────────

_MODE_INFO = {
    "fixed":    ("Fixed Interval",   "Extract one frame every N seconds"),
    "scene":    ("Scene Detection",   "Extract at scene cuts + interval baseline"),
    "sharpest": ("Sharpest Frame",    "Pick the sharpest frame per time window"),
}
_LABEL_TO_MODE = {info[0]: key for key, info in _MODE_INFO.items()}
_MODE_TO_LABEL = {key: info[0] for key, info in _MODE_INFO.items()}

_TIER_INFO = {
    "fast": ("Fast", "Score each frame with OpenCV Laplacian\nvariance, pick sharpest per window.\nFastest — no ffmpeg analysis pass."),
    "balanced": ("Balanced", "Analyze blur with ffmpeg blurdetect\n(more accurate for motion blur).\nNo scene-cut detection."),
    "best": ("Best", "ffmpeg blurdetect + scene detection.\nSplits time windows at scene boundaries\nso both sides get a sharp representative."),
}
_TIER_KEY_TO_LABEL = {k: v[0] for k, v in _TIER_INFO.items()}
_TIER_LABEL_TO_KEY = {v: k for k, v in _TIER_KEY_TO_LABEL.items()}

def _get_mode_value(app) -> str:
    """Map the display label back to the internal mode value."""
    return _LABEL_TO_MODE.get(app.extract_mode_var.get(), "fixed")


def _get_tier_value(app) -> str:
    """Map the tier dropdown label back to the internal tier key."""
    var = getattr(app, "extract_sharpest_tier_var", None)
    label = var.get() if var is not None else _TIER_KEY_TO_LABEL["best"]
    return _TIER_LABEL_TO_KEY.get(label, "best")


def _snapshot_settings(app) -> "ExtractionSettings":
    """Capture current GUI extraction settings into an ExtractionSettings object."""
    start = app.extract_start_entry.get().strip()
    end = app.extract_end_entry.get().strip()
    return ExtractionSettings(
        mode=_get_mode_value(app),
        interval=round(app.extract_interval_var.get(), 1),
        quality=app.extract_quality_var.get(),
        format=app.extract_format_var.get(),
        start_sec=_parse_time(start) if start else None,
        end_sec=_parse_time(end) if end else None,
        blur_filter=app.extract_blur_enabled_var.get(),
        blur_percentile=app.extract_blur_percentile_var.get(),
        sky_filter=app.extract_sky_enabled_var.get(),
        lut_enabled=app.extract_lut_enabled_var.get(),
        lut_path=app.extract_lut_file_entry.get().strip(),
        lut_strength=app.extract_lut_strength_var.get(),
        shadow=app.extract_shadow_var.get(),
        highlight=app.extract_highlight_var.get(),
        sky_brightness=app.extract_sky_brightness_var.get(),
        sky_keypoints=app.extract_sky_keypoints_var.get(),
        motion_enabled=app.extract_motion_enabled_var.get(),
        motion_sharpness=app.extract_sharpness_var.get(),
        motion_flow=app.extract_flow_var.get(),
        sharpest_tier=_get_tier_value(app),
    )

def _on_mode_change(app, label):
    """Update mode description when combo selection changes."""
    mode = _LABEL_TO_MODE.get(label, "fixed")
    desc = _MODE_INFO.get(mode, ("", ""))[1]
    if hasattr(app, "extract_mode_desc"):
        app.extract_mode_desc.configure(text=desc)
    _update_sharpest_tier_ui(app)


# ── time parser ───────────────────────────────────────────────────────

def _parse_time(time_str: str) -> float:
    """Parse *time_str* (seconds or ``MM:SS`` or ``HH:MM:SS``) → float seconds."""
    try:
        return float(time_str)
    except ValueError:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0.0


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

    _update_sharpest_tier_ui(app)


def _update_sharpest_tier_ui(app):
    combo = getattr(app, "extract_tier_combo", None)
    if combo is None:
        return

    is_sharpest = _get_mode_value(app) == "sharpest"

    # Enable/disable the tier combo
    combo.configure(state="readonly" if is_sharpest else "disabled")

    # Update tier description (inline on mode row, right of tier combo)
    desc = getattr(app, "extract_sharpest_tier_desc", None)
    if desc:
        if is_sharpest:
            tier = _get_tier_value(app)
            desc.configure(text=_TIER_INFO[tier][1])
            if not desc.winfo_manager():
                desc.pack(side="left", padx=(8, 0))
        else:
            if desc.winfo_manager():
                desc.pack_forget()


def _update_estimate(app, *_args):
    """Recalculate and display extraction estimates from current settings."""
    if not hasattr(app, "extract_estimate_label"):
        return
    info = getattr(app, "current_video_info", None)
    if info is None:
        app.extract_estimate_label.configure(text="Analyze the selected source to see estimates")
        return

    # effective duration
    total_dur = info.duration_seconds
    start_str = app.extract_start_entry.get().strip()
    end_str = app.extract_end_entry.get().strip()
    start_sec = _parse_time(start_str) if start_str else 0.0
    end_sec = _parse_time(end_str) if end_str and end_str.lower() != "end" else total_dur
    end_sec = min(end_sec, total_dur)
    eff_dur = max(0.0, end_sec - start_sec)

    mode = _get_mode_value(app)
    interval = app.extract_interval_var.get()
    paired_tier = _get_tier_value(app) if getattr(info, "pair_mode", False) else ""

    # frame count estimate
    if mode == "scene" or (
        mode == "sharpest"
        and getattr(info, "pair_mode", False)
        and paired_tier == "best"
    ):
        frames = int(eff_dur / interval)  # rough baseline
        if getattr(info, "pair_mode", False):
            frame_str = f"~{frames:,} frame pairs (varies by scene cuts)"
        else:
            frame_str = f"~{frames:,} frames (varies by scene cuts)"
    else:
        frames = max(1, int(eff_dur / interval))
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

    # blur/sky filter caveat
    caveats = []
    if mode == "sharpest":
        if getattr(info, "pair_mode", False):
            tier_label = getattr(app, "extract_sharpest_tier_var", None)
            tier_text = tier_label.get() if tier_label else "Best"
            caveats.append(f"sharpest per window ({tier_text.lower()})")
        else:
            caveats.append("sharpest per window")
    if app.extract_blur_enabled_var.get() and mode != "sharpest":
        pct = app.extract_blur_percentile_var.get()
        kept = max(1, int(frames * pct / 100))
        caveats.append(f"~{kept:,} after blur filter")
    if app.extract_sky_enabled_var.get():
        caveats.append("before sky filter")

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
    _build_fisheye_section(app, scroll)
    _build_metadata_section(app, scroll)
    _build_metashape_section(app, scroll)
    _update_source_mode_ui(app)


# ======================================================================
#  1. VIDEO SELECTION (input + output + optional analysis)
# ======================================================================

def _build_video_selection_section(app, parent):
    sec = CollapsibleSection(parent, "Video Selection", expanded=True)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Input video row
    vid_frame = ctk.CTkFrame(c, fg_color="transparent")
    vid_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(vid_frame, text="Input:", width=60, anchor="w").pack(side="left")
    app.analyze_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="Video file...")
    app.analyze_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(vid_frame, text="...", width=36,
                  command=lambda: app._browse_video_for(app.analyze_video_entry)
                  ).pack(side="left")

    # Output folder row
    out_frame = ctk.CTkFrame(c, fg_color="transparent")
    out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(out_frame, text="Output:", width=60, anchor="w").pack(side="left")
    app.extract_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Frames output folder...")
    app.extract_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(out_frame, text="...", width=36,
                  command=lambda: app._browse_folder_for(app.extract_output_entry)
                  ).pack(side="left")

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

    split_desc = ctk.CTkLabel(
        sc,
        text="This is the paired extraction source for the main Extract button.\n"
             "Top Input/Output still belong to single-video extraction and Split Lenses.",
        font=("Consolas", 10),
        text_color="#9ca3af",
        anchor="w",
        justify="left",
    )
    split_desc.pack(fill="x", padx=6, pady=(0, 4))

    split_front_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_front_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_front_frame, text="Front:", width=60, anchor="w").pack(side="left")
    app.split_front_video_entry = ctk.CTkEntry(
        split_front_frame,
        placeholder_text="Graded front lens video (.mp4 / .mov)...",
    )
    app.split_front_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.split_front_video_entry.bind("<FocusOut>", lambda _e: _maybe_autofill_split_output(app))
    split_front_btn = ctk.CTkButton(
        split_front_frame,
        text="...",
        width=36,
        command=lambda: _browse_split_video(app, app.split_front_video_entry, "Select Front Lens Video"),
    )
    split_front_btn.pack(side="left")

    split_back_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_back_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_back_frame, text="Back:", width=60, anchor="w").pack(side="left")
    app.split_back_video_entry = ctk.CTkEntry(
        split_back_frame,
        placeholder_text="Graded back lens video (.mp4 / .mov)...",
    )
    app.split_back_video_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.split_back_video_entry.bind("<FocusOut>", lambda _e: _maybe_autofill_split_output(app))
    split_back_btn = ctk.CTkButton(
        split_back_frame,
        text="...",
        width=36,
        command=lambda: _browse_split_video(app, app.split_back_video_entry, "Select Back Lens Video"),
    )
    split_back_btn.pack(side="left")

    split_out_frame = ctk.CTkFrame(sc, fg_color="transparent")
    split_out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(split_out_frame, text="Clip Folder:", width=80, anchor="w").pack(side="left")
    app.split_output_entry = ctk.CTkEntry(
        split_out_frame,
        placeholder_text="Per-clip working folder (auto-filled from the filename stem)...",
    )
    app.split_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    split_out_btn = ctk.CTkButton(
        split_out_frame,
        text="...",
        width=36,
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

    # Analysis (collapsible subsection)
    analysis_sec = CollapsibleSection(c, "Analysis", expanded=True)
    analysis_sec.pack(fill="x", pady=(4, 0), padx=2)
    ac = analysis_sec.content

    # Button + results side by side
    analysis_row = ctk.CTkFrame(ac, fg_color="transparent")
    analysis_row.pack(fill="x", pady=3, padx=4)

    app.analyze_run_btn = ctk.CTkButton(
        analysis_row, text="Analyze", command=lambda: _run_analyze(app),
        fg_color="#1976D2", hover_color="#1565C0",
        font=ctk.CTkFont(size=12, weight="bold"), width=90,
    )
    app.analyze_run_btn.pack(side="left", padx=(0, 6), anchor="n")

    app.analyze_results = ctk.CTkTextbox(analysis_row, height=160,
                                         font=ctk.CTkFont(family="Consolas", size=11))
    app.analyze_results.pack(side="left", fill="x", expand=True)
    app.analyze_results.insert("1.0", _analysis_placeholder_text(app))


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
    sec = CollapsibleSection(parent, "Frame Extraction", expanded=True)
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

    # -- extraction settings --
    mode_frame = ctk.CTkFrame(c, fg_color="transparent")
    mode_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(mode_frame, text="Mode:", width=60, anchor="w").pack(side="left")
    app.extract_mode_var = ctk.StringVar(value=_MODE_TO_LABEL["fixed"])
    ctk.CTkComboBox(mode_frame, variable=app.extract_mode_var,
                    values=list(_MODE_TO_LABEL.values()),
                    state="readonly", width=160,
                    command=lambda v: _on_mode_change(app, v),
                    ).pack(side="left", padx=(6, 0))

    # Tier dropdown on same row, initially disabled
    ctk.CTkLabel(mode_frame, text="Tier:", width=35, anchor="e").pack(side="left", padx=(12, 0))
    app.extract_sharpest_tier_var = ctk.StringVar(value="Best")
    app.extract_tier_combo = ctk.CTkComboBox(
        mode_frame, variable=app.extract_sharpest_tier_var,
        values=["Fast", "Balanced", "Best"],
        state="disabled", width=110,
        command=lambda _v: _update_sharpest_tier_ui(app),
    )
    app.extract_tier_combo.pack(side="left", padx=(4, 0))
    Tooltip(app.extract_tier_combo,
            "Controls how sharpness is measured and\n"
            "how time windows are handled.")

    # Tier description inline on mode row, right of combo (wraps to 2 lines)
    app.extract_sharpest_tier_desc = ctk.CTkLabel(
        mode_frame, text="", text_color="#9ca3af",
        font=ctk.CTkFont(size=10), anchor="w", justify="left")
    # Starts hidden — _update_sharpest_tier_ui will pack/forget it

    app.extract_mode_desc = ctk.CTkLabel(
        c, text=_MODE_INFO["fixed"][1], text_color="#9ca3af",
        font=ctk.CTkFont(size=10), anchor="w")
    app.extract_mode_desc.pack(fill="x", padx=12, pady=(0, 2))

    int_frame = ctk.CTkFrame(c, fg_color="transparent")
    int_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(int_frame, text="Interval:", width=60, anchor="w").pack(side="left")
    app.extract_interval_var = ctk.DoubleVar(value=2.0)
    app.extract_interval_label = ctk.CTkLabel(int_frame, text="2.0s", width=40,
                                              font=("Consolas", 11))
    app.extract_interval_label.pack(side="right")
    ctk.CTkSlider(int_frame, from_=0.1, to=10, variable=app.extract_interval_var,
                  command=lambda v: app.extract_interval_label.configure(text=f"{float(v):.1f}s")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))

    qual_frame = ctk.CTkFrame(c, fg_color="transparent")
    qual_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(qual_frame, text="Quality:", width=60, anchor="w").pack(side="left")
    app.extract_quality_var = ctk.IntVar(value=95)
    app.extract_quality_label = ctk.CTkLabel(qual_frame, text="95", width=40,
                                             font=("Consolas", 11))
    app.extract_quality_label.pack(side="right")
    ctk.CTkSlider(qual_frame, from_=70, to=100, variable=app.extract_quality_var,
                  command=lambda v: app.extract_quality_label.configure(text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))

    fmt_frame = ctk.CTkFrame(c, fg_color="transparent")
    fmt_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(fmt_frame, text="Format:", width=60, anchor="w").pack(side="left")
    app.extract_format_var = ctk.StringVar(value="jpg")
    ctk.CTkRadioButton(fmt_frame, text="JPEG", variable=app.extract_format_var,
                       value="jpg").pack(side="left", padx=(6, 12))
    ctk.CTkRadioButton(fmt_frame, text="PNG", variable=app.extract_format_var,
                       value="png").pack(side="left")

    # -- time range --
    time_frame = ctk.CTkFrame(c, fg_color="transparent")
    time_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(time_frame, text="Start:", width=40, anchor="w").pack(side="left")
    app.extract_start_entry = ctk.CTkEntry(time_frame, width=65, placeholder_text="0:00")
    app.extract_start_entry.pack(side="left", padx=(4, 8))
    ctk.CTkLabel(time_frame, text="End:", width=30, anchor="w").pack(side="left")
    app.extract_end_entry = ctk.CTkEntry(time_frame, width=65, placeholder_text="end")
    app.extract_end_entry.pack(side="left", padx=(4, 0))

    # -- live estimate --
    app.current_video_info = None
    app.extract_estimate_label = ctk.CTkLabel(
        c, text="Analyze the selected source to see estimates", anchor="w", justify="left",
        text_color="gray", font=ctk.CTkFont(size=11))
    app.extract_estimate_label.pack(fill="x", padx=8, pady=(4, 0))

    # -- Post-Processing (collapsible parent) --
    pp_sec = CollapsibleSection(c, "Post-Processing",
                                subtitle="filters applied after frame extraction",
                                expanded=True)
    pp_sec.pack(fill="x", pady=(4, 0), padx=2)
    pp = pp_sec.content

    # -- Color & LUT (collapsible) --
    lut_sec = CollapsibleSection(pp, "Color & LUT", expanded=False)
    lut_sec.pack(fill="x", pady=(6, 0), padx=2)
    app.extract_lut_section = lut_sec

    app.extract_lut_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(lut_sec.content, text="Apply LUT after extraction",
                    variable=app.extract_lut_enabled_var,
                    command=lambda: lut_sec.expand() if app.extract_lut_enabled_var.get() else None,
                    ).pack(pady=3, anchor="w")

    lut_file_frame = ctk.CTkFrame(lut_sec.content, fg_color="transparent")
    lut_file_frame.pack(fill="x", pady=3)
    ctk.CTkLabel(lut_file_frame, text="LUT:", width=50, anchor="w").pack(side="left")
    app.extract_lut_file_entry = ctk.CTkEntry(lut_file_frame, placeholder_text=".cube file...")
    app.extract_lut_file_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
    ctk.CTkButton(lut_file_frame, text="...", width=34,
                  command=lambda: app._browse_file_for(
                      app.extract_lut_file_entry, "Select LUT File",
                      [("CUBE Files", "*.cube"), ("All Files", "*.*")]
                  )).pack(side="left")

    lut_str_frame = ctk.CTkFrame(lut_sec.content, fg_color="transparent")
    lut_str_frame.pack(fill="x", pady=3)
    ctk.CTkLabel(lut_str_frame, text="Strength:", width=60, anchor="w").pack(side="left")
    app.extract_lut_strength_var = ctk.DoubleVar(value=1.0)
    app.extract_lut_strength_label = ctk.CTkLabel(lut_str_frame, text="100%", width=45,
                                                  font=("Consolas", 11))
    app.extract_lut_strength_label.pack(side="right")
    ctk.CTkSlider(lut_str_frame, from_=0, to=1, variable=app.extract_lut_strength_var,
                  command=lambda v: app.extract_lut_strength_label.configure(
                      text=f"{int(float(v)*100)}%")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    shadow_frame = ctk.CTkFrame(lut_sec.content, fg_color="transparent")
    shadow_frame.pack(fill="x", pady=3)
    ctk.CTkLabel(shadow_frame, text="Shadows:", width=60, anchor="w").pack(side="left")
    app.extract_shadow_var = ctk.IntVar(value=50)
    app.extract_shadow_label = ctk.CTkLabel(shadow_frame, text="50", width=35,
                                            font=("Consolas", 11))
    app.extract_shadow_label.pack(side="right")
    ctk.CTkSlider(shadow_frame, from_=0, to=100, variable=app.extract_shadow_var,
                  command=lambda v: app.extract_shadow_label.configure(text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    hl_frame = ctk.CTkFrame(lut_sec.content, fg_color="transparent")
    hl_frame.pack(fill="x", pady=3)
    ctk.CTkLabel(hl_frame, text="Highlights:", width=60, anchor="w").pack(side="left")
    app.extract_highlight_var = ctk.IntVar(value=50)
    app.extract_highlight_label = ctk.CTkLabel(hl_frame, text="50", width=35,
                                               font=("Consolas", 11))
    app.extract_highlight_label.pack(side="right")
    ctk.CTkSlider(hl_frame, from_=0, to=100, variable=app.extract_highlight_var,
                  command=lambda v: app.extract_highlight_label.configure(text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    ctk.CTkLabel(lut_sec.content, text="50 = neutral",
                 text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")

    # -- Sky Filter (collapsible) --
    sky_sec = CollapsibleSection(pp, "Sky Filter", expanded=False)
    sky_sec.pack(fill="x", pady=(4, 0), padx=2)

    app.extract_sky_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(sky_sec.content, text="Remove sky-dominated images",
                    variable=app.extract_sky_enabled_var).pack(pady=3, anchor="w")

    sky_br = ctk.CTkFrame(sky_sec.content, fg_color="transparent")
    sky_br.pack(fill="x", pady=3)
    ctk.CTkLabel(sky_br, text="Brightness:", width=70, anchor="w").pack(side="left")
    app.extract_sky_brightness_var = ctk.DoubleVar(value=0.85)
    app.extract_sky_brightness_label = ctk.CTkLabel(sky_br, text="0.85", width=40,
                                                    font=("Consolas", 11))
    app.extract_sky_brightness_label.pack(side="right")
    ctk.CTkSlider(sky_br, from_=0.5, to=1.0, variable=app.extract_sky_brightness_var,
                  command=lambda v: app.extract_sky_brightness_label.configure(
                      text=f"{float(v):.2f}")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    sky_kp = ctk.CTkFrame(sky_sec.content, fg_color="transparent")
    sky_kp.pack(fill="x", pady=3)
    ctk.CTkLabel(sky_kp, text="Keypoints:", width=70, anchor="w").pack(side="left")
    app.extract_sky_keypoints_var = ctk.IntVar(value=50)
    app.extract_sky_keypoints_label = ctk.CTkLabel(sky_kp, text="50", width=40,
                                                   font=("Consolas", 11))
    app.extract_sky_keypoints_label.pack(side="right")
    ctk.CTkSlider(sky_kp, from_=10, to=200, variable=app.extract_sky_keypoints_var,
                  command=lambda v: app.extract_sky_keypoints_label.configure(text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    # -- Blur Filter (collapsible) --
    blur_sec = CollapsibleSection(pp, "Blur Filter", expanded=False)
    blur_sec.pack(fill="x", pady=(4, 0), padx=2)

    app.extract_blur_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(blur_sec.content, text="Filter blurry frames after extraction",
                    variable=app.extract_blur_enabled_var).pack(pady=3, anchor="w")

    blur_pct = ctk.CTkFrame(blur_sec.content, fg_color="transparent")
    blur_pct.pack(fill="x", pady=3)
    ctk.CTkLabel(blur_pct, text="Keep:", width=70, anchor="w").pack(side="left")
    app.extract_blur_percentile_var = ctk.IntVar(value=80)
    app.extract_blur_pct_label = ctk.CTkLabel(blur_pct, text="80%", width=40,
                                              font=("Consolas", 11))
    app.extract_blur_pct_label.pack(side="right")
    ctk.CTkSlider(blur_pct, from_=50, to=100, variable=app.extract_blur_percentile_var,
                  command=lambda v: app.extract_blur_pct_label.configure(text=f"{int(v)}%")
                  ).pack(side="left", fill="x", expand=True, padx=(4, 4))

    ctk.CTkLabel(blur_sec.content, text="Percent of sharpest frames to keep",
                 text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")

    # -- Motion Selection (inside Post-Processing) --
    motion_sec = CollapsibleSection(pp, "Motion Selection",
                                     subtitle="filter by camera movement between frames",
                                     expanded=False)
    motion_sec.pack(fill="x", pady=(4, 0), padx=2)
    mc = motion_sec.content

    app.extract_motion_enabled_var = ctk.BooleanVar(value=False)
    motion_cb = ctk.CTkCheckBox(mc, text="Filter by sharpness + optical flow",
                    variable=app.extract_motion_enabled_var)
    motion_cb.pack(pady=3, padx=6, anchor="w")
    Tooltip(motion_cb,
            "After extraction, filter frames to keep only sharp\n"
            "frames with sufficient camera motion between them.\n"
            "Removes redundant near-identical frames.")

    sharp_frame = ctk.CTkFrame(mc, fg_color="transparent")
    sharp_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(sharp_frame, text="Sharpness:", width=70, anchor="w").pack(side="left")
    app.extract_sharpness_var = ctk.DoubleVar(value=50.0)
    app.extract_sharpness_label = ctk.CTkLabel(sharp_frame, text="50", width=35,
                                               font=("Consolas", 11))
    app.extract_sharpness_label.pack(side="right")
    ctk.CTkSlider(sharp_frame, from_=10, to=200,
                  variable=app.extract_sharpness_var,
                  command=lambda v: app.extract_sharpness_label.configure(
                      text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(sharp_frame,
            "Minimum Laplacian sharpness score.\n"
            "Frames below this are discarded as blurry.\n"
            "50 is a safe default; raise for high-quality datasets.")

    flow_frame = ctk.CTkFrame(mc, fg_color="transparent")
    flow_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(flow_frame, text="Target Flow:", width=70, anchor="w").pack(side="left")
    app.extract_flow_var = ctk.DoubleVar(value=10.0)
    app.extract_flow_label = ctk.CTkLabel(flow_frame, text="10", width=35,
                                          font=("Consolas", 11))
    app.extract_flow_label.pack(side="right")
    ctk.CTkSlider(flow_frame, from_=2, to=30,
                  variable=app.extract_flow_var,
                  command=lambda v: app.extract_flow_label.configure(
                      text=f"{int(v)}")
                  ).pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(flow_frame,
            "Target optical flow magnitude between kept frames.\n"
            "Higher = more camera movement required between frames.\n"
            "10 works well for typical walking/driving captures.")

    # -- Primary action: Extract / Add to Queue / Stop --
    action_row = ctk.CTkFrame(c, fg_color="transparent")
    action_row.pack(fill="x", pady=(8, 4), padx=6)

    app.extract_run_btn = ctk.CTkButton(
        action_row, text="Extract", command=lambda: _run_extract_single(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.extract_run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    app.extract_queue_add_btn = ctk.CTkButton(
        action_row, text="Add to Queue", command=lambda: _add_current_to_queue(app),
        fg_color="#1565C0", hover_color="#0D47A1", height=38, width=110,
    )
    app.extract_queue_add_btn.pack(side="left", padx=(0, 4))

    app.queue_run_btn = ctk.CTkButton(
        action_row, text="Process Queue", command=lambda: _run_extract_queue(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=38, width=120,
    )
    app.queue_run_btn.pack(side="left", padx=(0, 4))

    app.extract_stop_btn = ctk.CTkButton(
        action_row, text="Stop", command=lambda: _queue_stop(app),
        fg_color="#C62828", hover_color="#8B0000", height=38, width=70,
    )
    app.extract_stop_btn.pack(side="left")
    app.extract_stop_btn.pack_forget()

    app.extract_progress_bar = ctk.CTkProgressBar(c, height=8)
    app.extract_progress_bar.pack(fill="x", padx=8, pady=(0, 4))
    app.extract_progress_bar.set(0)

    # -- Batch Queue (collapsed by default) --
    q_sec = CollapsibleSection(c, "Batch Queue", expanded=False)
    q_sec.pack(fill="x", pady=(6, 0), padx=2)
    qc = q_sec.content

    # queue controls
    qctrl = ctk.CTkFrame(qc, fg_color="transparent")
    qctrl.pack(fill="x", pady=(2, 4))
    app.queue_add_videos_btn = ctk.CTkButton(
        qctrl, text="Add Videos", width=80,
        command=lambda: _queue_add_videos(app))
    app.queue_add_videos_btn.pack(side="left", padx=(0, 4))
    app.queue_add_folder_btn = ctk.CTkButton(
        qctrl, text="Add Folder", width=80,
        command=lambda: _queue_add_folder(app))
    app.queue_add_folder_btn.pack(side="left", padx=(0, 4))
    app.queue_remove_btn = ctk.CTkButton(
        qctrl, text="Remove", width=60, fg_color="#666",
        command=lambda: _queue_remove_selected(app))
    app.queue_remove_btn.pack(side="left", padx=(0, 4))
    app.queue_clear_done_btn = ctk.CTkButton(
        qctrl, text="Clear Done", width=72, fg_color="#666",
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
    app.extract_mode_var.trace_add("write", _est)
    app.extract_interval_var.trace_add("write", _est)
    app.extract_quality_var.trace_add("write", _est)
    app.extract_format_var.trace_add("write", _est)
    app.extract_sharpest_tier_var.trace_add("write", _est)
    app.extract_blur_enabled_var.trace_add("write", _est)
    app.extract_blur_percentile_var.trace_add("write", _est)
    app.extract_sky_enabled_var.trace_add("write", _est)
    # Start/end entries don't have trace — use KeyRelease instead
    app.extract_start_entry.bind("<KeyRelease>", _est)
    app.extract_end_entry.bind("<KeyRelease>", _est)
    _update_sharpest_tier_ui(app)


# ── SRT geotagging helper ─────────────────────────────────────────────

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

    # Try manifest-based first, fall back to interval
    manifest = output_dir / "extraction_manifest.json"
    if manifest.exists():
        result = geotag_from_manifest(str(output_dir), srt_path)
    else:
        interval = app.extract_interval_var.get()
        start = app.extract_start_entry.get().strip()
        start_sec = _parse_time(start) if start else 0.0
        result = geotag_from_interval(str(output_dir), srt_path, interval, start_sec)

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


def _run_post_processing(app, settings, output_dir):
    """Run post-processing filters on extracted frames in output_dir.

    Applies: LUT, shadow/highlight, sky filter, blur filter, motion selection.
    Reads settings from the ExtractionSettings object (not app GUI vars).
    Deletes rejected frames in-place.
    """
    import cv2
    from pathlib import Path

    ext_patterns = ("*.jpg", "*.jpeg", "*.png")
    images = sorted(
        p for pat in ext_patterns for p in Path(output_dir).glob(pat)
    )
    if not images:
        return

    ext = images[0].suffix.lstrip(".")
    quality = settings.quality

    # LUT
    if settings.lut_enabled and settings.lut_path and Path(settings.lut_path).exists():
        app.log(f"  Applying LUT: {Path(settings.lut_path).name}")
        processor = LUTProcessor()
        for img_path in images:
            if app.cancel_flag.is_set():
                return
            img = cv2.imread(str(img_path))
            if img is not None:
                processed = processor.apply_lut(img, settings.lut_path, settings.lut_strength)
                params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ("jpg", "jpeg") else []
                cv2.imwrite(str(img_path), processed, params)

    # Shadow / highlight
    if settings.shadow != 50 or settings.highlight != 50:
        app.log("  Applying shadow/highlight adjustments...")
        from prep360.core.adjustments import apply_shadow_highlight
        for img_path in images:
            if app.cancel_flag.is_set():
                return
            img = cv2.imread(str(img_path))
            if img is not None:
                adjusted = apply_shadow_highlight(img, settings.shadow, settings.highlight)
                params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ("jpg", "jpeg") else []
                cv2.imwrite(str(img_path), adjusted, params)

    # Sky filter
    if settings.sky_filter:
        app.log("  Running sky filter...")
        sky_cfg = SkyFilterConfig(
            brightness_threshold=settings.sky_brightness,
            keypoint_threshold=settings.sky_keypoints,
        )
        sky = SkyFilter(sky_cfg)
        images = sorted(p for pat in ext_patterns for p in Path(output_dir).glob(pat))
        removed = 0
        for img_path in images:
            if app.cancel_flag.is_set():
                return
            m = sky.analyze_image(str(img_path))
            if m.is_sky:
                img_path.unlink()
                removed += 1
        if removed:
            app.log(f"  Removed {removed} sky-dominated images")

    # Blur filter (skip if sharpest mode was used — already pre-filtered)
    used_sharpest = (settings.mode == "sharpest")
    if settings.blur_filter and not used_sharpest and not app.cancel_flag.is_set():
        app.log("  Running blur filter...")
        from prep360.core.blur_filter import BlurFilter, BlurFilterConfig
        bf = BlurFilter(BlurFilterConfig(percentile=float(settings.blur_percentile), workers=4))
        scores = bf.analyze_batch(str(output_dir))
        if scores:
            import numpy as np
            vals = [s.score for s in scores]
            cutoff = float(np.percentile(vals, 100 - settings.blur_percentile))
            removed = 0
            for s in scores:
                if s.score < cutoff:
                    p = Path(output_dir) / s.image_name
                    if p.exists():
                        p.unlink()
                        removed += 1
            if removed:
                app.log(f"  Removed {removed} blurry frames (kept top {settings.blur_percentile}%)")

    # Motion selection
    if settings.motion_enabled and not app.cancel_flag.is_set():
        app.log("  Running motion-aware selection...")
        from prep360.core.motion_selector import MotionSelector
        selector = MotionSelector(
            min_sharpness=settings.motion_sharpness,
            target_flow=settings.motion_flow,
        )
        images = sorted(str(p) for pat in ext_patterns for p in Path(output_dir).glob(pat))
        sel_result = selector.select_from_paths(images)
        app.log(f"  {sel_result.summary()}")
        selected_set = {f.path for f in sel_result.selected_frames}
        removed = 0
        for p in images:
            if p not in selected_set:
                Path(p).unlink()
                removed += 1
        if removed:
            app.log(f"  Removed {removed} redundant frames via motion selection")


def _extract_single_worker(app, video_path, base_output):
    """Run extraction for a single video directly — no queue involvement."""
    try:
        video_name = Path(video_path).stem
        output_dir = Path(base_output) / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_str = _get_mode_value(app)
        interval = app.extract_interval_var.get()
        quality = app.extract_quality_var.get()
        fmt = app.extract_format_var.get()

        start = app.extract_start_entry.get().strip()
        end = app.extract_end_entry.get().strip()
        start_sec = _parse_time(start) if start else None
        end_sec = _parse_time(end) if end else None

        app.log(f"\n{'='*50}")
        app.log(f"Processing: {Path(video_path).name}")
        app.log(f"Output: {output_dir}")
        app.log(f"Mode: {mode_str}, Interval: {interval}s")

        def progress(curr, total, msg):
            if total > 0 and hasattr(app, "extract_progress_bar"):
                app.after(0, lambda p=curr/total: app.extract_progress_bar.set(p))
            if msg:
                app.after(0, lambda m=msg: app.log(m))

        used_sharpest = False

        if mode_str == "sharpest":
            used_sharpest = True
            tier = _get_tier_value(app)
            sharp_cfg = SharpestConfig(
                interval=interval,
                quality=quality,
                output_format=fmt,
                start_sec=start_sec,
                end_sec=end_sec,
                tier=tier,
            )
            sharp_ext = SharpestExtractor()
            sharp_result = sharp_ext.extract(
                video_path, str(output_dir), sharp_cfg,
                progress_callback=progress,
                cancel_check=app.cancel_flag.is_set,
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
            else:
                result = ExtractionResult(
                    success=False, frame_count=0,
                    output_dir=str(output_dir), frames=[],
                    error=sharp_result.error,
                )
        else:
            config = ExtractionConfig(
                interval=interval,
                mode=ExtractionMode(mode_str),
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

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        if result.success:
            settings = _snapshot_settings(app)
            _run_post_processing(app, settings, output_dir)
            # Recount after filtering
            ext = fmt
            final_count = len(list(output_dir.glob(f"*.{ext}")))

            # SRT geotag (after all filters so we only tag surviving frames)
            _maybe_geotag(app, video_path, output_dir)

            app.log(f"Done: {final_count} frames extracted")
        else:
            app.log(f"Error: {result.error}")

    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: _extract_single_done(app))


def _extract_single_done(app):
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
    app.cancel_flag.clear()
    app.extract_run_btn.configure(state="disabled")
    app.queue_run_btn.configure(state="disabled")
    app.extract_stop_btn.pack(side="left")
    if hasattr(app, "extract_progress_bar"):
        app.extract_progress_bar.set(0)

    threading.Thread(target=_extract_queue_worker, args=(app,), daemon=True).start()


def _extract_queue_worker(app):
    try:
        while app.extract_queue_processing and not app.cancel_flag.is_set():
            item = app.video_queue.get_next_pending()
            if not item:
                break

            app.extract_current_item_id = item.id
            app.video_queue.set_processing(item.id)
            app.after(0, lambda: _queue_refresh(app))

            app.log(f"\n{'='*50}")
            app.log(f"Processing: {item.filename}")

            try:
                # Use per-item settings (or fall back to current GUI for legacy items)
                s = item.settings or _snapshot_settings(app)
                mode_str = s.mode
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

                used_sharpest = False

                if mode_str == "sharpest":
                    # Sharpest-frame extraction via blurdetect
                    used_sharpest = True
                    sharp_cfg = SharpestConfig(
                        interval=interval,
                        quality=quality,
                        output_format=fmt,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        tier=s.sharpest_tier,
                    )
                    sharp_ext = SharpestExtractor()
                    sharp_result = sharp_ext.extract(
                        item.video_path, str(output_dir), sharp_cfg,
                        progress_callback=progress,
                        cancel_check=app.cancel_flag.is_set,
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
                    else:
                        result = ExtractionResult(
                            success=False, frame_count=0,
                            output_dir=str(output_dir), frames=[],
                            error=sharp_result.error,
                        )
                else:
                    # Standard extraction modes (fixed/scene/adaptive)
                    config = ExtractionConfig(
                        interval=interval,
                        mode=ExtractionMode(mode_str),
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

                if app.cancel_flag.is_set():
                    app.video_queue.set_cancelled(item.id)
                    app.log(f"Cancelled: {item.filename}")
                    break

                if result.success:
                    _run_post_processing(app, s, output_dir)
                    ext = s.format
                    final_count = len(list(output_dir.glob(f"*.{ext}")))

                    # SRT geotag (after all filters)
                    _maybe_geotag(app, item.video_path, output_dir)

                    app.video_queue.set_done(item.id, final_count)
                    app.log(f"Done: {final_count} frames extracted")
                else:
                    app.video_queue.set_error(item.id, result.error or "Unknown error")
                    app.log(f"Error: {result.error}")

            except Exception as e:
                import traceback
                app.video_queue.set_error(item.id, str(e))
                app.log(f"Error: {e}")
                app.log(traceback.format_exc())

            app.after(0, lambda: _queue_refresh(app))

        stats = app.video_queue.get_stats()
        app.log(f"\n{'='*50}")
        app.log("Queue processing complete")
        app.log(f"Done: {stats['done']}, Errors: {stats['error']}, Cancelled: {stats['cancelled']}")

    except Exception as e:
        app.log(f"Queue error: {e}")
    finally:
        app.extract_queue_processing = False
        app.extract_current_item_id = None
        app.after(0, lambda: _queue_done(app))


def _queue_done(app):
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
#  3. METADATA — SRT geotagging (+ future EXIF/XMP tools)
# ======================================================================

def _build_metadata_section(app, parent):
    """Metadata injection section — SRT geotagging and future EXIF tools."""
    sec = CollapsibleSection(parent, "Metadata", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # -- Auto-geotag after extraction --
    app.extract_geotag_enabled_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(c, text="Auto-geotag after extraction",
                    variable=app.extract_geotag_enabled_var
                    ).pack(pady=(2, 4), anchor="w", padx=6)

    # -- Shared SRT file path (used by both auto and standalone) --
    srt_frame = ctk.CTkFrame(c, fg_color="transparent")
    srt_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(srt_frame, text="SRT:", width=60, anchor="w").pack(side="left")
    app.metadata_srt_entry = ctk.CTkEntry(srt_frame,
                                          placeholder_text="Auto-detect from video name...")
    app.metadata_srt_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(srt_frame, text="...", width=36,
                  command=lambda: app._browse_file_for(
                      app.metadata_srt_entry, "Select SRT File",
                      [("SRT Files", "*.srt *.SRT"), ("All Files", "*.*")]
                  )).pack(side="left")

    ctk.CTkLabel(c, text="Writes EXIF GPS, altitude, focal length, datetime via exiftool",
                 text_color="gray", font=ctk.CTkFont(size=10)
                 ).pack(anchor="w", padx=8, pady=(0, 6))

    # -- Standalone: geotag existing frames --
    sep = ctk.CTkFrame(c, height=1, fg_color="gray30")
    sep.pack(fill="x", padx=6, pady=(2, 6))

    ctk.CTkLabel(c, text="Apply to folder",
                 font=ctk.CTkFont(size=12, weight="bold")
                 ).pack(anchor="w", padx=6, pady=(0, 4))

    # Frames folder
    fr_frame = ctk.CTkFrame(c, fg_color="transparent")
    fr_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(fr_frame, text="Frames:", width=60, anchor="w").pack(side="left")
    app.geotag_frames_entry = ctk.CTkEntry(fr_frame, placeholder_text="Folder with extracted frames...")
    app.geotag_frames_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(fr_frame, text="...", width=36,
                  command=lambda: app._browse_folder_for(app.geotag_frames_entry)
                  ).pack(side="left")

    # Action button
    btn_row = ctk.CTkFrame(c, fg_color="transparent")
    btn_row.pack(fill="x", pady=(8, 4), padx=6)
    app.geotag_run_btn = ctk.CTkButton(
        btn_row, text="Geotag", command=lambda: _run_geotag_standalone(app),
        fg_color="#1976D2", hover_color="#1565C0",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.geotag_run_btn.pack(side="left", fill="x", expand=True)


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
        result = geotag_from_manifest(str(frames_dir), srt_path)

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
#  4. METASHAPE SESSION
# ======================================================================

def _build_metashape_section(app, parent):
    sec = CollapsibleSection(
        parent,
        "Metashape Session",
        subtitle="assemble a no-copy session manifest from clip working folders",
        expanded=False,
    )
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    desc = ctk.CTkLabel(
        c,
        text="Use this after several clip folders already have front/frames + back/frames\n"
             "and optional front/masks + back/masks. This step writes a session manifest only.\n"
             "It does not duplicate any images on disk.",
        font=("Consolas", 10),
        text_color="#9ca3af",
        anchor="w",
        justify="left",
    )
    desc.pack(fill="x", padx=6, pady=(2, 4))

    clips_frame = ctk.CTkFrame(c, fg_color="transparent")
    clips_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(clips_frame, text="Clips:", width=80, anchor="w").pack(side="left")
    app.metashape_clips_root_entry = ctk.CTkEntry(
        clips_frame,
        placeholder_text="Clip folder or parent folder containing many clip folders...",
    )
    app.metashape_clips_root_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        clips_frame, text="...", width=36,
        command=lambda: app._browse_folder_for(app.metashape_clips_root_entry),
    ).pack(side="left")

    output_frame = ctk.CTkFrame(c, fg_color="transparent")
    output_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(output_frame, text="Session Out:", width=80, anchor="w").pack(side="left")
    app.metashape_output_entry = ctk.CTkEntry(
        output_frame,
        placeholder_text="Folder where the session manifest should be written...",
    )
    app.metashape_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        output_frame, text="...", width=36,
        command=lambda: app._browse_folder_for(app.metashape_output_entry),
    ).pack(side="left")

    name_frame = ctk.CTkFrame(c, fg_color="transparent")
    name_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(name_frame, text="Name:", width=80, anchor="w").pack(side="left")
    app.metashape_session_name_entry = ctk.CTkEntry(
        name_frame,
        placeholder_text="Optional session name (defaults from the output folder)...",
    )
    app.metashape_session_name_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))

    app.metashape_require_masks_var = ctk.BooleanVar(value=False)
    masks_cb = ctk.CTkCheckBox(
        c,
        text="Require masks for every clip pair",
        variable=app.metashape_require_masks_var,
    )
    masks_cb.pack(pady=(4, 2), padx=6, anchor="w")
    Tooltip(
        masks_cb,
        "Enable this if the session should only include clips with complete front/back masks.",
    )

    app.metashape_run_btn = ctk.CTkButton(
        c, text="Assemble Session Manifest", command=lambda: _run_metashape_session(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.metashape_run_btn.pack(fill="x", padx=6, pady=(6, 2))

    app.metashape_status_text = ctk.CTkTextbox(
        c, height=50,
        font=ctk.CTkFont(family="Consolas", size=10),
        fg_color="#1a1a1a", state="disabled",
    )
    app.metashape_status_text.pack(fill="x", padx=6, pady=(4, 2))
    app._set_textbox(
        app.metashape_status_text,
        "Assemble one session manifest from validated clip working folders.\n"
        "This keeps the existing clip files in place and only writes a JSON handoff for Metashape.",
    )


def _run_metashape_session(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return

    clips_root = app.metashape_clips_root_entry.get().strip()
    session_out = app.metashape_output_entry.get().strip()
    session_name = app.metashape_session_name_entry.get().strip() or None
    if not clips_root:
        app.log("Error: Select a clip folder or a parent folder of clip folders")
        return
    if not Path(clips_root).exists():
        app.log(f"Error: Clips folder not found: {clips_root}")
        return
    if not session_out:
        app.log("Error: Select a Session Out folder")
        return

    app.metashape_run_btn.configure(state="disabled")
    threading.Thread(
        target=_metashape_session_worker,
        args=(app, clips_root, session_out, session_name, app.metashape_require_masks_var.get()),
        daemon=True,
    ).start()


def _metashape_session_worker(app, clips_root, session_out, session_name, require_masks):
    try:
        clip_roots = discover_clip_roots(clips_root)
        app.log(f"Assembling Metashape session from {len(clip_roots)} clip folder(s)...")

        clip_datasets: list[DualFisheyeClipDataset] = []
        total_pairs = 0
        masked_pairs = 0
        for clip_root in clip_roots:
            dataset = load_clip_dataset(clip_root, require_masks=require_masks)
            clip_datasets.append(dataset)
            total_pairs += len(dataset.pair_records)
            masked_pairs += sum(
                1 for record in dataset.pair_records
                if record.front_mask is not None and record.back_mask is not None
            )
            app.log(
                f"  {dataset.clip_id}: {len(dataset.pair_records)} pairs"
                + (f", {sum(1 for r in dataset.pair_records if r.front_mask and r.back_mask)} masked"
                   if dataset.masks_root else ", no masks")
            )

        manifest_path = write_dual_fisheye_session_manifest(
            session_out,
            clip_datasets,
            session_name=session_name,
        )

        summary = "\n".join([
            "Metashape session ready",
            f"  Clips:         {len(clip_datasets)}",
            f"  Pair count:    {total_pairs}",
            f"  Mask pairs:    {masked_pairs}",
            f"  Manifest:      {manifest_path}",
            "  Mode:          no-copy session assembly",
        ])
        app.log(f"\n{summary}")
        app.after(0, lambda: app._set_textbox(app.metashape_status_text, summary))
    except Exception as e:
        import traceback
        app.log(f"Metashape session error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app.metashape_run_btn.configure(state="normal"))


# ======================================================================
#  5. FISHEYE (DJI Osmo 360)
# ======================================================================

def _build_fisheye_section(app, parent):
    sec = CollapsibleSection(parent, "360 Video", expanded=True)
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

    split_desc = ctk.CTkLabel(
        sc, text="Uses the top Video Selection Input + Output.\n"
                 "This step only creates the raw front/back lens videos for external grading.\n"
                 "After grading, go back to Video Selection and enable Split Lens Videos.",
        font=("Consolas", 10), text_color="#9ca3af", anchor="w", justify="left")
    split_desc.pack(fill="x", padx=6, pady=(2, 4))

    app.fisheye_split_btn = ctk.CTkButton(
        sc, text="Split Lenses", command=lambda: _run_split_lenses(app),
        fg_color="#1565C0", hover_color="#0D47A1",
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_split_btn.pack(fill="x", padx=6, pady=(0, 4))

    reframe_sec = CollapsibleSection(
        c, "Reframing",
        subtitle="extract pinhole perspectives from a 360 file or existing ERP frames",
        expanded=False,
    )
    reframe_sec.pack(fill="x", padx=2, pady=(4, 0))
    rc = reframe_sec.content

    osv_frame = ctk.CTkFrame(rc, fg_color="transparent")
    osv_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(osv_frame, text="Video:", width=55, anchor="w").pack(side="left")
    app.fisheye_osv_entry = ctk.CTkEntry(
        osv_frame,
        placeholder_text=".osv / .360 / .insv file (optional)...",
    )
    app.fisheye_osv_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(osv_frame, text="...", width=36,
                  command=lambda: _browse_osv(app)).pack(side="left")

    reframe_frames_frame = ctk.CTkFrame(rc, fg_color="transparent")
    reframe_frames_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(reframe_frames_frame, text="Frames:", width=55, anchor="w").pack(side="left")
    app.fisheye_reframe_frames_entry = ctk.CTkEntry(
        reframe_frames_frame,
        placeholder_text="Existing ERP frames folder (optional)...",
    )
    app.fisheye_reframe_frames_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        reframe_frames_frame, text="...", width=36,
        command=lambda: app._browse_folder_for(app.fisheye_reframe_frames_entry)
    ).pack(side="left")

    reframe_masks_frame = ctk.CTkFrame(rc, fg_color="transparent")
    reframe_masks_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(reframe_masks_frame, text="Masks:", width=55, anchor="w").pack(side="left")
    app.fisheye_reframe_masks_entry = ctk.CTkEntry(
        reframe_masks_frame,
        placeholder_text="Masks for ERP or fisheye frames (optional)...",
    )
    app.fisheye_reframe_masks_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        reframe_masks_frame, text="...", width=36,
        command=lambda: app._browse_folder_for(app.fisheye_reframe_masks_entry)
    ).pack(side="left")

    reframe_out_frame = ctk.CTkFrame(rc, fg_color="transparent")
    reframe_out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(reframe_out_frame, text="Output:", width=55, anchor="w").pack(side="left")
    app.fisheye_reframe_output_entry = ctk.CTkEntry(
        reframe_out_frame,
        placeholder_text="Perspective output directory...",
    )
    app.fisheye_reframe_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        reframe_out_frame, text="...", width=36,
        command=lambda: app._browse_folder_for(app.fisheye_reframe_output_entry)
    ).pack(side="left")

    # ── Custom Calibration (collapsible, inside Reframe) ─────────────

    adv_sec = CollapsibleSection(rc, "Custom Calibration",
                                  subtitle="override built-in lens model",
                                  expanded=False)
    adv_sec.pack(fill="x", padx=2, pady=(4, 0))
    ac = adv_sec.content

    app.fisheye_calib_default_var = ctk.BooleanVar(value=True)
    app.fisheye_calib_entry = ctk.CTkEntry(ac, width=0)  # hidden storage

    adv_desc = ctk.CTkLabel(
        ac,
        text="The built-in equidistant fisheye model works for most 360\n"
             "cameras (DJI, Insta360, GoPro). Load a custom JSON if you\n"
             "see warped straight lines in your output crops.\n\n"
             "JSON format: {front: {K, D, image_size, rms_error},\n"
             "              back: {K, D, image_size, rms_error}}\n"
             "K = 3x3 camera matrix, D = 4 distortion coefficients.",
        font=ctk.CTkFont(size=10), text_color="#9ca3af",
        anchor="w", justify="left")
    adv_desc.pack(fill="x", padx=6, pady=(2, 6))

    # Calibration file row
    calib_row = ctk.CTkFrame(ac, fg_color="transparent")
    calib_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(calib_row, text="File:", width=55, anchor="w").pack(side="left")
    app._fisheye_custom_entry = ctk.CTkEntry(
        calib_row, placeholder_text="calibration.json...")
    app._fisheye_custom_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(calib_row, text="...", width=36,
                  command=lambda: _load_calibration(app)).pack(side="left")
    reset_btn = ctk.CTkButton(
        calib_row, text="Reset", width=50,
        fg_color="#6b7280", hover_color="#4b5563",
        command=lambda: _reset_calibration(app))
    reset_btn.pack(side="left", padx=(4, 0))
    Tooltip(reset_btn, "Revert to the built-in equidistant fisheye model.")

    # Status label
    app._fisheye_calib_label = ctk.CTkLabel(
        ac, text="Using built-in calibration",
        font=ctk.CTkFont(family="Consolas", size=10),
        text_color="#9ca3af", anchor="w")
    app._fisheye_calib_label.pack(fill="x", padx=12, pady=(0, 2))

    # Preset
    preset_frame = ctk.CTkFrame(rc, fg_color="transparent")
    preset_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(preset_frame, text="Preset:", width=55, anchor="w").pack(side="left")
    preset_labels = list(_PRESET_KEY_TO_LABEL.values()) if HAS_PREP360 else ["Full 90\u00b0 \u2014 26 views"]
    app.fisheye_preset_var = ctk.StringVar(value=preset_labels[0])
    preset_combo = ctk.CTkComboBox(preset_frame, variable=app.fisheye_preset_var,
                    values=preset_labels, state="readonly",
                    command=lambda v: _on_preset_change(app, v))
    preset_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))
    Tooltip(preset_combo,
            "View layout preset — controls how many pinhole\n"
            "perspectives are extracted from each fisheye frame.\n"
            "More views = better 3D coverage but more images.")

    # Preset description (multi-line, structured)
    app.fisheye_preset_desc = ctk.CTkLabel(rc, text="", text_color="#9ca3af",
                                            font=ctk.CTkFont(family="Consolas", size=10),
                                            anchor="w", justify="left")
    app.fisheye_preset_desc.pack(fill="x", pady=(0, 2), padx=12)

    # Crop + Quality + Interval on compact rows
    cq_frame = ctk.CTkFrame(rc, fg_color="transparent")
    cq_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(cq_frame, text="Crop:", width=55, anchor="w").pack(side="left")
    app.fisheye_crop_var = ctk.StringVar(value="1600")
    crop_combo = ctk.CTkComboBox(cq_frame, variable=app.fisheye_crop_var,
                    values=["1280", "1600", "1920"],
                    state="readonly", width=80,
                    command=lambda v: _update_fisheye_estimate(app))
    crop_combo.pack(side="left", padx=(6, 0))
    Tooltip(crop_combo, "Output resolution per pinhole crop (square).\n"
            "1600 is a good balance of detail and file size.\n"
            "1920 for maximum quality, 1280 for faster processing.")

    ctk.CTkLabel(cq_frame, text="Quality:").pack(side="left", padx=(12, 2))
    app.fisheye_quality_var = ctk.IntVar(value=95)
    app.fisheye_quality_label = ctk.CTkLabel(cq_frame, text="95", width=30,
                                             font=("Consolas", 11))
    qual_slider = ctk.CTkSlider(cq_frame, from_=70, to=100, variable=app.fisheye_quality_var,
                  width=80, command=lambda v: app.fisheye_quality_label.configure(text=f"{int(v)}"))
    qual_slider.pack(side="left", padx=2)
    app.fisheye_quality_label.pack(side="left")
    Tooltip(qual_slider, "JPEG quality for output crops.\n"
            "95 = near-lossless, recommended for photogrammetry.\n"
            "Lower values save disk space but lose detail.")

    int_frame = ctk.CTkFrame(rc, fg_color="transparent")
    int_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(int_frame, text="Interval:", width=55, anchor="w").pack(side="left")
    app.fisheye_interval_var = ctk.DoubleVar(value=2.0)
    app.fisheye_interval_label = ctk.CTkLabel(int_frame, text="2.0s", width=40,
                                              font=("Consolas", 11))
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

    # Station-aware output checkbox
    app.station_dirs_var = ctk.BooleanVar(value=True)
    app.fisheye_station_cb = ctk.CTkCheckBox(rc, text="Station dirs (for Metashape)",
                    variable=app.station_dirs_var, width=200)
    app.fisheye_station_cb.pack(pady=(6, 2), padx=6, anchor="w")
    Tooltip(app.fisheye_station_cb,
            "Organize output into per-source subdirectories.\n"
            "Each subdirectory becomes a Metashape station\n"
            "(shared camera position). Drag all subdirs into\n"
            "an empty chunk, then set group type to Station.\n\n"
            "Also writes reframe_metadata.json with pinhole\n"
            "intrinsics and station-to-view mapping.")

    # Output estimate label (dynamic — updates with preset/interval/crop changes)
    app.fisheye_estimate_label = ctk.CTkLabel(rc, text="",
                                               font=("Consolas", 10),
                                               text_color="#9ca3af",
                                               anchor="w", justify="left")
    app.fisheye_estimate_label.pack(fill="x", padx=12, pady=(0, 4))

    app.fisheye_reframe_btn = ctk.CTkButton(
        rc, text="Extract & Reframe", command=lambda: _run_reframe(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_reframe_btn.pack(fill="x", padx=6, pady=(6, 2))
    Tooltip(
        app.fisheye_reframe_btn,
        "Extract from video and/or reframe ERP frames into perspectives.\n"
        "Use only for pinhole-perspective workflows.",
    )

    app.fisheye_stop_btn = ctk.CTkButton(
        c, text="Stop", command=app.stop_operation,
        fg_color="#C62828", hover_color="#8B0000", height=32,
    )
    # hidden initially — shown via _start_operation

    # Init preset display
    if HAS_PREP360:
        _on_preset_change(app, app.fisheye_preset_var.get())


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


def _split_lenses_worker(app, osv_path, output_dir):
    try:
        app.log(f"\nSplitting lenses: {Path(osv_path).name}")
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
    settings = _snapshot_settings(app)
    mode = settings.mode
    sharpest_tier = _get_tier_value(app)
    if mode not in {"fixed", "sharpest"}:
        app.log(
            "Error: Split lens paired extraction supports only "
            "Fixed Interval and Sharpest Frame modes.\n"
            "Scene Detection produces variable-rate output that "
            "cannot guarantee synchronized front/back pairs."
        )
        return

    clip_root = Path(output_dir)
    settings_summary = (
        f"sharpest/{_TIER_KEY_TO_LABEL[sharpest_tier].lower()}, "
        f"{settings.interval:.1f}s windows, {settings.format}, q{settings.quality}, "
        + (
            "quick shared pair scoring, then pair extraction"
            if sharpest_tier == "fast" else
            "blurdetect pair analysis, then pair extraction"
            if sharpest_tier == "balanced" else
            "blurdetect + scene-aware pair analysis, then pair extraction"
        )
        if mode == "sharpest" else
        f"fixed, {settings.interval:.1f}s, {settings.format}, q{settings.quality}"
    )
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

    extractor = PairedSplitVideoExtractor()
    result = extractor.extract(
        front_video,
        back_video,
        str(clip_root),
        PairedSplitConfig(
            mode=mode,
            sharpest_tier=sharpest_tier,
            interval_sec=settings.interval,
            quality=settings.quality,
            output_format=settings.format,
            start_sec=settings.start_sec,
            end_sec=settings.end_sec,
        ),
        progress_callback=paired_progress,
        cancel_check=lambda: app.cancel_flag.is_set(),
    )

    if not result.success:
        app.log(f"Error: {result.error}")
        return

    # Run post-processing on both front and back frame directories
    settings = _snapshot_settings(app)
    for lens_label, lens_dir in [("front", clip_root / "front" / "frames"),
                                  ("back", clip_root / "back" / "frames")]:
        if lens_dir.exists() and any(lens_dir.iterdir()):
            app.log(f"\nPost-processing {lens_label} frames...")
            _run_post_processing(app, settings, lens_dir)

    summary = "\n".join([
        "Paired extraction complete",
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

    station_dirs = app.station_dirs_var.get()
    app._start_operation(app.fisheye_reframe_btn, app.fisheye_stop_btn)
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

        # Optional motion selection (only when extracting from video)
        if has_video and app.extract_motion_enabled_var.get():
            app.log("\nRunning motion-aware selection...")
            selector = MotionSelector(
                min_sharpness=app.extract_sharpness_var.get(),
                target_flow=app.extract_flow_var.get(),
            )
            sel_result = selector.select_from_paths(front, max_frames=len(front))
            app.log(sel_result.summary())
            idx_set = {fs.index for fs in sel_result.selected_frames}
            front = [f for i, f in enumerate(front) if i in idx_set]
            back = [b for i, b in enumerate(back) if i in idx_set]
            app.log(f"After selection: {len(front)} frame pairs")

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

    except Exception as e:
        import traceback
        app.log(f"Fisheye error: {e}")
        app.log(traceback.format_exc())
    finally:
        btn = action_button or app.fisheye_reframe_btn
        app.after(0, lambda: app._stop_operation(
            btn, app.fisheye_stop_btn))


def _erp_reframe_worker(app, frame_paths, masks_dir, output_dir, config, station_dirs=False):
    """Reframe ERP frames using the equirect reframer (not fisheye pairs)."""
    from prep360.core.reframer import Reframer, ViewConfig, Ring

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
    # Station mode: reframer creates images/ and masks/ subdirs itself
    # Flat mode: put crops in output_dir/images/
    reframe_out = output_dir if station_dirs else str(Path(output_dir) / "images")

    app.log(f"\nReframing {len(frame_paths)} ERP frames...")

    def rf_progress(current, total, msg):
        if not app.cancel_flag.is_set():
            app.log(f"  [{current}/{total}] {msg}")

    result = reframer.reframe_batch(
        frames_dir, reframe_out,
        mask_dir=masks_dir,
        num_workers=1,
        progress_callback=rf_progress,
        station_dirs=station_dirs,
    )

    lines = [
        "ERP reframing complete",
        f"  Input frames:  {result.input_count}",
        f"  Total views:   {result.output_count}",
        f"  Images:        {reframe_out}",
    ]
    if masks_dir:
        masks_out = str(Path(output_dir) / "masks")
        lines.append(f"  Masks:         {masks_out}")
    if result.errors:
        lines.append(f"  Errors:        {len(result.errors)}")
        for err in result.errors[:5]:
            lines.append(f"    - {err}")

    summary = "\n".join(lines)
    app.log(f"\n{summary}")

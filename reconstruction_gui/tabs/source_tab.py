"""
Source Tab — Analyze, Extract (with video queue), and Fisheye.

Ported from panoex_gui.py into a single scrollable tab for Reconstruction Zone.
All GUI state is stored on the *app* instance so the preview panel and other
tabs can read it.

Usage::

    from tabs.source_tab import build_source_tab
    build_source_tab(app, tab_frame)
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from typing import List

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
    from prep360.core.queue_manager import VideoQueue

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


# ── extraction mode display labels ────────────────────────────────────

_MODE_INFO = {
    "fixed":    ("Fixed Interval",   "Extract one frame every N seconds"),
    "scene":    ("Scene Detection",   "Extract at scene cuts + interval baseline"),
    "adaptive": ("Adaptive Density",  "More frames in high-motion segments"),
    "sharpest": ("Sharpest Frame",    "Pick the sharpest frame per time window"),
}
_LABEL_TO_MODE = {info[0]: key for key, info in _MODE_INFO.items()}
_MODE_TO_LABEL = {key: info[0] for key, info in _MODE_INFO.items()}

def _get_mode_value(app) -> str:
    """Map the display label back to the internal mode value."""
    return _LABEL_TO_MODE.get(app.extract_mode_var.get(), "fixed")

def _on_mode_change(app, label):
    """Update mode description when combo selection changes."""
    mode = _LABEL_TO_MODE.get(label, "fixed")
    desc = _MODE_INFO.get(mode, ("", ""))[1]
    if hasattr(app, "extract_mode_desc"):
        app.extract_mode_desc.configure(text=desc)


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


def _update_estimate(app, *_args):
    """Recalculate and display extraction estimates from current settings."""
    if not hasattr(app, "extract_estimate_label"):
        return
    info = getattr(app, "current_video_info", None)
    if info is None:
        app.extract_estimate_label.configure(text="Analyze a video to see estimates")
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

    # frame count estimate
    if mode == "scene":
        frames = int(eff_dur / interval)  # rough baseline
        frame_str = f"~{frames:,} frames (varies by scene cuts)"
    else:
        frames = max(1, int(eff_dur / interval))
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

    total_bytes = frames * bytes_per_frame

    # blur/sky filter caveat
    caveats = []
    if mode == "sharpest":
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

    # Analysis (collapsible subsection)
    analysis_sec = CollapsibleSection(c, "Analysis", expanded=False)
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
    app.analyze_results.insert("1.0",
        "Shows: resolution, fps, duration,\n"
        "360° detection, log format,\n"
        "recommended interval & LUT")


def _run_analyze(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    video = app.analyze_video_entry.get()
    if not video:
        app.log("Error: Please select a video file")
        return
    if not Path(video).exists():
        app.log(f"Error: Video not found: {video}")
        return
    app.analyze_run_btn.configure(state="disabled")
    threading.Thread(target=_analyze_worker, args=(app, video), daemon=True).start()


def _analyze_worker(app, video):
    try:
        app.log(f"Analyzing: {Path(video).name}")
        analyzer = VideoAnalyzer()
        info = analyzer.analyze(video)
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
        app.current_video_info = info
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
        c, text="Analyze a video to see estimates", anchor="w", justify="left",
        text_color="gray", font=ctk.CTkFont(size=11))
    app.extract_estimate_label.pack(fill="x", padx=8, pady=(4, 0))

    # -- Color & LUT (collapsible) --
    lut_sec = CollapsibleSection(c, "Color & LUT", expanded=False)
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
    sky_sec = CollapsibleSection(c, "Sky Filter", expanded=False)
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
    blur_sec = CollapsibleSection(c, "Blur Filter", expanded=False)
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

    # -- Primary action: Extract / Add to Queue / Stop --
    action_row = ctk.CTkFrame(c, fg_color="transparent")
    action_row.pack(fill="x", pady=(8, 4), padx=6)

    app.extract_run_btn = ctk.CTkButton(
        action_row, text="Extract", command=lambda: _run_extract_single(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.extract_run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    ctk.CTkButton(
        action_row, text="Add to Queue", command=lambda: _add_current_to_queue(app),
        fg_color="#1565C0", hover_color="#0D47A1", height=38, width=110,
    ).pack(side="left", padx=(0, 4))

    app.extract_stop_btn = ctk.CTkButton(
        action_row, text="Stop", command=lambda: _queue_stop(app),
        fg_color="#C62828", hover_color="#8B0000", height=38, width=70,
    )
    app.extract_stop_btn.pack(side="left")
    app.extract_stop_btn.pack_forget()

    # -- Batch Queue (collapsed by default) --
    q_sec = CollapsibleSection(c, "Batch Queue", expanded=False)
    q_sec.pack(fill="x", pady=(6, 0), padx=2)
    qc = q_sec.content

    # queue controls
    qctrl = ctk.CTkFrame(qc, fg_color="transparent")
    qctrl.pack(fill="x", pady=(2, 4))
    ctk.CTkButton(qctrl, text="Add Videos", width=80,
                  command=lambda: _queue_add_videos(app)).pack(side="left", padx=(0, 4))
    ctk.CTkButton(qctrl, text="Add Folder", width=80,
                  command=lambda: _queue_add_folder(app)).pack(side="left", padx=(0, 4))
    ctk.CTkButton(qctrl, text="Remove", width=60, fg_color="#666",
                  command=lambda: _queue_remove_selected(app)).pack(side="left", padx=(0, 4))
    ctk.CTkButton(qctrl, text="Clear Done", width=72, fg_color="#666",
                  command=lambda: _queue_clear_done(app)).pack(side="left")

    # queue list — starts at minimal height, grows with content
    app.queue_scroll = ctk.CTkScrollableFrame(qc, height=0, fg_color="transparent")
    app.queue_scroll.pack(fill="x", pady=(0, 4))

    app.queue_stats_label = ctk.CTkLabel(qc, text="Queue: 0 pending, 0 done",
                                         text_color="gray", font=ctk.CTkFont(size=10))
    app.queue_stats_label.pack(anchor="w")

    # process queue button
    q_btn_row = ctk.CTkFrame(qc, fg_color="transparent")
    q_btn_row.pack(fill="x", pady=(6, 4))

    app.queue_run_btn = ctk.CTkButton(
        q_btn_row, text="Process Queue", command=lambda: _run_extract_queue(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.queue_run_btn.pack(side="left", fill="x", expand=True)

    # refresh display after build
    _queue_refresh(app)

    # -- wire up live estimate updates --
    _est = lambda *_a: _update_estimate(app)
    app.extract_mode_var.trace_add("write", _est)
    app.extract_interval_var.trace_add("write", _est)
    app.extract_quality_var.trace_add("write", _est)
    app.extract_format_var.trace_add("write", _est)
    app.extract_blur_enabled_var.trace_add("write", _est)
    app.extract_blur_percentile_var.trace_add("write", _est)
    app.extract_sky_enabled_var.trace_add("write", _est)
    # Start/end entries don't have trace — use KeyRelease instead
    app.extract_start_entry.bind("<KeyRelease>", _est)
    app.extract_end_entry.bind("<KeyRelease>", _est)


# ── single-video extract ──────────────────────────────────────────────

def _add_current_to_queue(app):
    """Add the currently analyzed video to the batch queue."""
    video = getattr(app, "current_video_path", None)
    if not video:
        app.log("Analyze a video first")
        return
    if not app.video_queue:
        return
    added = app.video_queue.add_video(video)
    if added:
        app.log(f"Queued: {Path(video).name}")
    else:
        app.log(f"Already in queue: {Path(video).name}")
    _queue_refresh(app)


def _run_extract_single(app):
    """Extract frames from the currently analyzed video directly (no queue)."""
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    video = getattr(app, "current_video_path", None)
    if not video:
        app.log("Error: Analyze a video first")
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
    if hasattr(app, "progress_bar"):
        app.progress_bar.start()

    threading.Thread(
        target=_extract_single_worker, args=(app, video, output),
        daemon=True,
    ).start()


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
            if total > 0 and hasattr(app, "progress_bar"):
                app.after(0, lambda p=curr/total: app.progress_bar.set(p))
            if msg:
                app.after(0, lambda m=msg: app.log(m))

        used_sharpest = False

        if mode_str == "sharpest":
            used_sharpest = True
            sharp_cfg = SharpestConfig(
                interval=interval,
                quality=quality,
                output_format=fmt,
                start_sec=start_sec,
                end_sec=end_sec,
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
            import cv2
            ext = fmt
            images = sorted(output_dir.glob(f"*.{ext}"))
            final_count = len(images)

            # LUT
            if app.extract_lut_enabled_var.get():
                lut_path = app.extract_lut_file_entry.get()
                if lut_path and Path(lut_path).exists():
                    app.log(f"Applying LUT: {Path(lut_path).name}")
                    processor = LUTProcessor()
                    strength = app.extract_lut_strength_var.get()
                    for img_path in images:
                        if app.cancel_flag.is_set():
                            break
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            processed = processor.apply_lut(img, lut_path, strength)
                            params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                                      if ext in ("jpg", "jpeg") else [])
                            cv2.imwrite(str(img_path), processed, params)

            # Shadow / highlight
            shadow = app.extract_shadow_var.get()
            highlight = app.extract_highlight_var.get()
            if shadow != 50 or highlight != 50:
                app.log("Applying shadow/highlight adjustments...")
                from prep360.core.adjustments import apply_shadow_highlight
                for img_path in images:
                    if app.cancel_flag.is_set():
                        break
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        adjusted = apply_shadow_highlight(img, shadow, highlight)
                        params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                                  if ext in ("jpg", "jpeg") else [])
                        cv2.imwrite(str(img_path), adjusted, params)

            # Sky filter
            if app.extract_sky_enabled_var.get():
                app.log("Running sky filter...")
                sky_cfg = SkyFilterConfig(
                    brightness_threshold=app.extract_sky_brightness_var.get(),
                    keypoint_threshold=app.extract_sky_keypoints_var.get(),
                )
                sky = SkyFilter(sky_cfg)
                images = sorted(output_dir.glob(f"*.{ext}"))
                removed = 0
                for img_path in images:
                    if app.cancel_flag.is_set():
                        break
                    m = sky.analyze_image(str(img_path))
                    if m.is_sky:
                        img_path.unlink()
                        removed += 1
                if removed:
                    app.log(f"Removed {removed} sky-dominated images")
                    final_count -= removed

            # Blur filter — skip if already using sharpest mode
            if (app.extract_blur_enabled_var.get()
                    and not used_sharpest
                    and not app.cancel_flag.is_set()):
                app.log("Running blur filter...")
                from prep360.core.blur_filter import BlurFilter, BlurFilterConfig
                pct = app.extract_blur_percentile_var.get()
                bf = BlurFilter(BlurFilterConfig(percentile=float(pct), workers=4))
                scores = bf.analyze_batch(str(output_dir))
                if scores:
                    import numpy as np
                    vals = [s.score for s in scores]
                    cutoff = float(np.percentile(vals, 100 - pct))
                    blur_removed = 0
                    for s in scores:
                        if s.score < cutoff:
                            p = output_dir / s.image_name
                            if p.exists():
                                p.unlink()
                                blur_removed += 1
                    if blur_removed:
                        app.log(f"Removed {blur_removed} blurry frames (kept top {pct}%)")
                        final_count -= blur_removed

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
    if hasattr(app, "progress_bar"):
        app.progress_bar.stop()
        app.progress_bar.set(1.0)


# ── queue helpers ─────────────────────────────────────────────────────

def _queue_add_videos(app):
    files = filedialog.askopenfilenames(
        title="Select Videos",
        filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")],
    )
    if files and app.video_queue:
        n = app.video_queue.add_videos(list(files))
        app.log(f"Added {n} videos to queue")
        _queue_refresh(app)


def _queue_add_folder(app):
    folder = filedialog.askdirectory(title="Select Folder with Videos")
    if folder and app.video_queue:
        n = app.video_queue.add_folder(folder)
        app.log(f"Added {n} videos from folder")
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
    if hasattr(app, "progress_bar"):
        app.progress_bar.start()

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
                mode_str = _get_mode_value(app)
                interval = app.extract_interval_var.get()
                quality = app.extract_quality_var.get()
                fmt = app.extract_format_var.get()

                start = app.extract_start_entry.get().strip()
                end = app.extract_end_entry.get().strip()
                start_sec = _parse_time(start) if start else None
                end_sec = _parse_time(end) if end else None

                base_output = Path(app.extract_output_entry.get())
                video_name = Path(item.filename).stem
                output_dir = base_output / video_name
                output_dir.mkdir(parents=True, exist_ok=True)

                def progress(curr, total, msg):
                    if total > 0:
                        pct = int(curr / total * 100)
                        app.video_queue.set_progress(item.id, pct)
                        app.after(0, lambda: _queue_update_item(app, item.id))
                    if msg:
                        app.after(0, lambda m=msg: app.log(m))

                app.log(f"Output: {output_dir}")
                app.log(f"Mode: {mode_str}, Interval: {interval}s")

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
                    import cv2
                    ext = fmt
                    images = sorted(output_dir.glob(f"*.{ext}"))
                    final_count = len(images)

                    # LUT
                    if app.extract_lut_enabled_var.get():
                        lut_path = app.extract_lut_file_entry.get()
                        if lut_path and Path(lut_path).exists():
                            app.log(f"Applying LUT: {Path(lut_path).name}")
                            processor = LUTProcessor()
                            strength = app.extract_lut_strength_var.get()
                            for img_path in images:
                                if app.cancel_flag.is_set():
                                    break
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    processed = processor.apply_lut(img, lut_path, strength)
                                    params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                                              if ext in ("jpg", "jpeg") else [])
                                    cv2.imwrite(str(img_path), processed, params)

                    # Shadow / highlight
                    shadow = app.extract_shadow_var.get()
                    highlight = app.extract_highlight_var.get()
                    if shadow != 50 or highlight != 50:
                        app.log("Applying shadow/highlight adjustments...")
                        from prep360.core.adjustments import apply_shadow_highlight
                        for img_path in images:
                            if app.cancel_flag.is_set():
                                break
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                adjusted = apply_shadow_highlight(img, shadow, highlight)
                                params = ([cv2.IMWRITE_JPEG_QUALITY, quality]
                                          if ext in ("jpg", "jpeg") else [])
                                cv2.imwrite(str(img_path), adjusted, params)

                    # Sky filter
                    if app.extract_sky_enabled_var.get():
                        app.log("Running sky filter...")
                        sky_cfg = SkyFilterConfig(
                            brightness_threshold=app.extract_sky_brightness_var.get(),
                            keypoint_threshold=app.extract_sky_keypoints_var.get(),
                        )
                        sky = SkyFilter(sky_cfg)
                        images = sorted(output_dir.glob(f"*.{ext}"))
                        removed = 0
                        for img_path in images:
                            if app.cancel_flag.is_set():
                                break
                            m = sky.analyze_image(str(img_path))
                            if m.is_sky:
                                img_path.unlink()
                                removed += 1
                        if removed:
                            app.log(f"Removed {removed} sky-dominated images")
                            final_count -= removed

                    # Blur filter — skip if already using sharpest mode
                    if (app.extract_blur_enabled_var.get()
                            and not used_sharpest
                            and not app.cancel_flag.is_set()):
                        app.log("Running blur filter...")
                        from prep360.core.blur_filter import BlurFilter, BlurFilterConfig
                        pct = app.extract_blur_percentile_var.get()
                        bf = BlurFilter(BlurFilterConfig(percentile=float(pct), workers=4))
                        scores = bf.analyze_batch(str(output_dir))
                        if scores:
                            import numpy as np
                            vals = [s.score for s in scores]
                            cutoff = float(np.percentile(vals, 100 - pct))
                            blur_removed = 0
                            for s in scores:
                                if s.score < cutoff:
                                    p = output_dir / s.image_name
                                    if p.exists():
                                        p.unlink()
                                        blur_removed += 1
                            if blur_removed:
                                app.log(f"Removed {blur_removed} blurry frames (kept top {pct}%)")
                                final_count -= blur_removed

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
    if hasattr(app, "progress_bar"):
        app.progress_bar.stop()
        app.progress_bar.set(1.0)
    _queue_refresh(app)


def _queue_stop(app):
    app.log("Stopping queue processing...")
    app.cancel_flag.set()
    app.extract_queue_processing = False


# ======================================================================
#  3. FISHEYE (DJI Osmo 360)
# ======================================================================

def _build_fisheye_section(app, parent):
    sec = CollapsibleSection(parent, "360 Video", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # ── Video + Output ────────────────────────────────────────────────

    osv_frame = ctk.CTkFrame(c, fg_color="transparent")
    osv_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(osv_frame, text="Input:", width=55, anchor="w").pack(side="left")
    app.fisheye_osv_entry = ctk.CTkEntry(osv_frame,
                                          placeholder_text=".osv / .360 / .insv file...")
    app.fisheye_osv_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(osv_frame, text="...", width=36,
                  command=lambda: _browse_osv(app)).pack(side="left")

    out_frame = ctk.CTkFrame(c, fg_color="transparent")
    out_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(out_frame, text="Output:", width=55, anchor="w").pack(side="left")
    app.fisheye_output_entry = ctk.CTkEntry(out_frame,
                                             placeholder_text="Output directory...")
    app.fisheye_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(out_frame, text="...", width=36,
                  command=lambda: app._browse_folder_for(app.fisheye_output_entry)
                  ).pack(side="left")

    # ── Custom Calibration (collapsible) ─────────────────────────────

    adv_sec = CollapsibleSection(c, "Custom Calibration",
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

    # ── Split Lenses ──────────────────────────────────────────────────

    split_sec = CollapsibleSection(c, "Split Lenses",
                                    subtitle="lossless demux into front + back streams",
                                    expanded=False)
    split_sec.pack(fill="x", padx=2, pady=(4, 0))
    sc = split_sec.content

    split_desc = ctk.CTkLabel(
        sc, text="Extract front + back lens streams from the 360 container.\n"
                 "No reframing — use this to train on raw 180\u00b0 hemispheres.",
        font=("Consolas", 10), text_color="#9ca3af", anchor="w", justify="left")
    split_desc.pack(fill="x", padx=6, pady=(2, 4))

    app.fisheye_split_btn = ctk.CTkButton(
        sc, text="Split Lenses", command=lambda: _run_split_lenses(app),
        fg_color="#1565C0", hover_color="#0D47A1",
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_split_btn.pack(fill="x", padx=6, pady=(0, 4))

    # ── Reframe to Pinhole Perspectives ───────────────────────────────

    reframe_sec = CollapsibleSection(c, "Reframe to Pinhole Perspectives",
                                     expanded=False)
    reframe_sec.pack(fill="x", padx=2, pady=(4, 0))
    rc = reframe_sec.content

    # Status box — inside reframe section where its feedback is relevant
    app.fisheye_status_text = ctk.CTkTextbox(rc, height=50,
                                              font=ctk.CTkFont(family="Consolas", size=10),
                                              fg_color="#1a1a1a", state="disabled")
    app.fisheye_status_text.pack(fill="x", padx=6, pady=(4, 2))
    app._set_textbox(app.fisheye_status_text, "Select a .osv / .360 file and configure settings.")

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
                    command=lambda v: _update_estimate(app))
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
        _update_estimate(app)
    interval_slider = ctk.CTkSlider(int_frame, from_=0.5, to=10.0,
                                     variable=app.fisheye_interval_var,
                                     command=_on_interval_change)
    interval_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.fisheye_interval_label.pack(side="right")
    Tooltip(interval_slider,
            "Extract one frame every N seconds from the video.\n"
            "Lower = more frames, longer processing, bigger dataset.\n"
            "2.0s is a good default for walking-speed capture.")

    # Motion-aware selection (collapsible)
    motion_sec = CollapsibleSection(rc, "Motion-Aware Selection",
                                     subtitle="keep sharpest frames with enough motion",
                                     expanded=False)
    motion_sec.pack(fill="x", padx=2, pady=(4, 0))
    mc = motion_sec.content

    app.fisheye_motion_var = ctk.BooleanVar(value=False)
    motion_cb = ctk.CTkCheckBox(mc, text="Enable",
                    variable=app.fisheye_motion_var, width=80)
    motion_cb.pack(pady=3, padx=6, anchor="w")
    Tooltip(motion_cb,
            "After extracting frames at the interval rate, filter\n"
            "to keep only sharp frames with sufficient camera motion.\n"
            "Reduces redundant/blurry frames in your dataset.")

    sharp_frame = ctk.CTkFrame(mc, fg_color="transparent")
    sharp_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(sharp_frame, text="Sharpness:", width=70, anchor="w").pack(side="left")
    app.fisheye_sharpness_var = ctk.DoubleVar(value=50.0)
    app.fisheye_sharpness_label = ctk.CTkLabel(sharp_frame, text="50", width=35,
                                               font=("Consolas", 11))
    app.fisheye_sharpness_label.pack(side="right")
    sharp_slider = ctk.CTkSlider(sharp_frame, from_=10, to=200,
                                  variable=app.fisheye_sharpness_var,
                                  command=lambda v: app.fisheye_sharpness_label.configure(
                                      text=f"{int(v)}"))
    sharp_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(sharp_slider, "Minimum Laplacian sharpness score.\n"
            "Frames below this are discarded as blurry.\n"
            "50 is a safe default; raise for high-quality datasets.")

    flow_frame = ctk.CTkFrame(mc, fg_color="transparent")
    flow_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(flow_frame, text="Target Flow:", width=70, anchor="w").pack(side="left")
    app.fisheye_flow_var = ctk.DoubleVar(value=10.0)
    app.fisheye_flow_label = ctk.CTkLabel(flow_frame, text="10", width=35,
                                          font=("Consolas", 11))
    app.fisheye_flow_label.pack(side="right")
    flow_slider = ctk.CTkSlider(flow_frame, from_=2, to=30,
                                 variable=app.fisheye_flow_var,
                                 command=lambda v: app.fisheye_flow_label.configure(
                                     text=f"{int(v)}"))
    flow_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(flow_slider, "Target optical flow magnitude between kept frames.\n"
            "Higher = more camera movement required between frames.\n"
            "10 works well for typical walking/driving captures.")

    # Extract button
    app.fisheye_run_btn = ctk.CTkButton(
        rc, text="Extract Pinhole Views", command=lambda: _run_fisheye(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=36,
    )
    app.fisheye_run_btn.pack(fill="x", padx=6, pady=(6, 2))
    Tooltip(app.fisheye_run_btn,
            "Extract pinhole perspective crops from each fisheye frame pair.\n"
            "Uses the preset view layout and crop settings above.")

    # Output estimate label (dynamic — updates with preset/interval/crop changes)
    app.fisheye_estimate_label = ctk.CTkLabel(rc, text="",
                                               font=("Consolas", 10),
                                               text_color="#9ca3af",
                                               anchor="w", justify="left")
    app.fisheye_estimate_label.pack(fill="x", padx=12, pady=(0, 4))

    app.fisheye_stop_btn = ctk.CTkButton(
        c, text="Stop", command=app.stop_operation,
        fg_color="#C62828", hover_color="#8B0000", height=32,
    )
    # hidden initially — shown via _start_operation

    # Init preset display
    if HAS_PREP360:
        _on_preset_change(app, app.fisheye_preset_var.get())


# ── fisheye helpers ───────────────────────────────────────────────────

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
        def _update():
            app._set_textbox(app.fisheye_status_text, info.summary())
            _update_estimate(app)
        app.after(0, _update)
        app.log(f"Probed: {info.filename} \u2014 {info.duration:.1f}s, {info.fps:.0f}fps")
    except Exception as e:
        app.after(0, lambda: app._set_textbox(app.fisheye_status_text, f"Error: {e}"))


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
    _update_estimate(app)


def _update_estimate(app):
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
    osv_path = app.fisheye_osv_entry.get()
    output_dir = app.fisheye_output_entry.get()
    if not osv_path:
        app.log("Error: Please select a 360 video file")
        return
    if not Path(osv_path).exists():
        app.log(f"Error: File not found: {osv_path}")
        return
    if not output_dir:
        app.log("Error: Please select output folder")
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

        # Auto-load front lens into Video Analysis for immediate use
        if front_path:
            def _load_front():
                app.current_video_path = front_path
                if hasattr(app, "video_entry"):
                    app.video_entry.delete(0, "end")
                    app.video_entry.insert(0, front_path)
                app.log(f"\nLoaded front lens into Video Analysis — click Analyze to continue")
            app.after(0, _load_front)

    except Exception as e:
        import traceback
        app.log(f"Error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_split_btn, app.fisheye_stop_btn))


# ── fisheye worker ────────────────────────────────────────────────────

def _run_fisheye(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    osv_path = app.fisheye_osv_entry.get()
    output_dir = app.fisheye_output_entry.get()
    if not osv_path:
        app.log("Error: Please select an OSV file")
        return
    if not Path(osv_path).exists():
        app.log(f"Error: File not found: {osv_path}")
        return
    if not output_dir:
        app.log("Error: Please select an output folder")
        return

    app._start_operation(app.fisheye_run_btn, app.fisheye_stop_btn)
    threading.Thread(target=_fisheye_worker, args=(app, osv_path, output_dir),
                     daemon=True).start()


def _fisheye_worker(app, osv_path, output_dir):
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

        app.log(f"Preset: {preset_name} ({config.total_views()} views)")
        app.log(f"Crop: {crop_size}x{crop_size} @ Q{quality}")

        # Step 1: extract fisheye frames
        interval = app.fisheye_interval_var.get()
        frames_dir = str(Path(output_dir) / "fisheye_frames")
        app.log(f"\nExtracting fisheye frames (interval={interval:.1f}s)...")

        handler = OSVHandler()

        def osv_progress(stream_name, current, total):
            if not app.cancel_flag.is_set():
                app.log(f"  [{stream_name}] {current}/{total}")

        frame_dict = handler.extract_frames(
            osv_path, frames_dir,
            interval=interval, streams="both", quality=95,
            progress_callback=osv_progress,
        )

        front = sorted(frame_dict.get("front", []))
        back = sorted(frame_dict.get("back", []))
        if not front or not back:
            app.log("Error: No frames extracted")
            return
        app.log(f"Extracted {len(front)} front + {len(back)} back frames")

        if app.cancel_flag.is_set():
            app.log("Cancelled")
            return

        # Optional motion selection
        if app.fisheye_motion_var.get():
            app.log("\nRunning motion-aware selection...")
            selector = MotionSelector(
                min_sharpness=app.fisheye_sharpness_var.get(),
                target_flow=app.fisheye_flow_var.get(),
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

        # Step 2: reframe → perspective
        pairs = list(zip(front, back))
        crops_dir = str(Path(output_dir) / "perspectives")
        app.log(f"\nReframing {len(pairs)} pairs \u2192 {config.total_views()} views each...")

        def rf_progress(current, total, msg):
            if not app.cancel_flag.is_set():
                app.log(f"  [{current}/{total}] {msg}")

        total_crops, errors = fisheye_batch_extract(
            pairs, config, calib, crops_dir,
            num_workers=1, progress_callback=rf_progress,
        )

        lines = [
            "Fisheye extraction complete",
            f"  Frame pairs:   {len(pairs)}",
            f"  Views/pair:    {config.total_views()}",
            f"  Total crops:   {total_crops}",
            f"  Crop size:     {crop_size}x{crop_size}",
            f"  Output:        {crops_dir}",
        ]
        if errors:
            lines.append(f"  Errors:        {len(errors)}")
            for err in errors[:5]:
                lines.append(f"    - {err}")

        summary = "\n".join(lines)
        app.log(f"\n{summary}")
        app.after(0, lambda: app._set_textbox(app.fisheye_status_text, summary))

    except Exception as e:
        app.log(f"Fisheye error: {e}")
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_run_btn, app.fisheye_stop_btn))

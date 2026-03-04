#!/usr/bin/env python3
"""
prep360 GUI - 360 Video Processing Pipeline (Legacy)

A CustomTkinter GUI wrapper for the prep360 core library.
NOTE: This file has been retired. See reconstruction_zone.py for the active app.
Provides tabs for: Analyze, Extract, Reframe, LUT, Sky Filter, Pipeline.
"""

import customtkinter as ctk
import threading
import queue
import sys
from tkinter import filedialog
from pathlib import Path

from dataclasses import dataclass, field
from typing import List, Optional, Callable

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class PlannedFrame:
    """Represents a frame planned for extraction."""
    index: int
    frame_number: int
    timestamp: float
    reason: str  # "interval", "scene", "manual"
    included: bool = True


class CollapsibleSection(ctk.CTkFrame):
    """A collapsible section with toggle button (▼/▶) and show/hide content."""

    def __init__(self, master, title: str, expanded: bool = False, **kwargs):
        super().__init__(master, **kwargs)

        self._expanded = expanded
        self._title = title

        # Header frame with toggle button
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", pady=(0, 5))

        self.toggle_btn = ctk.CTkButton(
            self.header,
            text=f"{'▼' if expanded else '▶'} {title}",
            anchor="w",
            fg_color="transparent",
            hover_color=("gray75", "gray25"),
            text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle,
            height=28
        )
        self.toggle_btn.pack(fill="x")

        # Content frame
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=(15, 0))

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.configure(text=f"{'▼' if self._expanded else '▶'} {self._title}")
        if self._expanded:
            self.content.pack(fill="x", padx=(15, 0))
        else:
            self.content.pack_forget()

    def is_expanded(self) -> bool:
        return self._expanded

    def expand(self):
        if not self._expanded:
            self._toggle()

    def collapse(self):
        if self._expanded:
            self._toggle()

# Import prep360 core modules
try:
    from prep360.core import (
        VideoAnalyzer, VideoInfo,
        FrameExtractor, ExtractionConfig, ExtractionMode,
        Reframer, ViewConfig, Ring,
        LUTProcessor,
        SkyFilter, SkyFilterConfig,
        PresetManager,
        Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS,
        OSVHandler, OSVInfo,
        FisheyeReframer, FisheyeViewConfig, FISHEYE_PRESETS,
        DualFisheyeCalibration,
        MotionSelector, SelectionResult,
        GapDetector, GapReport, SpatialGap,
        BridgeExtractor, BridgeResult,
    )
    from prep360.core.reframer import VIEW_PRESETS
    from prep360.core.segmenter import HAS_YOLO, get_class_id
    from prep360.core.fisheye_reframer import (
        default_osmo360_calibration, fisheye_batch_extract,
    )
    HAS_PREP360 = True
except ImportError as e:
    HAS_PREP360 = False
    HAS_YOLO = False
    IMPORT_ERROR = str(e)


class PanoexApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("prep360 - 360 Video Processing Pipeline")
        self.geometry("1200x900")
        self.minsize(1050, 800)

        self.log_queue = queue.Queue()
        self.is_running = False
        self.cancel_flag = threading.Event()

        self._create_widgets()
        self.process_log_queue()
        self._setup_console_redirect()

        self.log("prep360 - 360 Video Processing Pipeline")
        self.log("=" * 50)

        if not HAS_PREP360:
            self.log(f"WARNING: prep360 core not found: {IMPORT_ERROR}")
            self.log("Run from the reframing directory or install prep360 package.")
        else:
            self.log("Core library loaded successfully")
            self.log("Tabs: Analyze | Extract | Reframe | LUT | Sky Filter | Pipeline")

    def _create_widgets(self):
        # IMPORTANT: Pack bottom frame FIRST so it claims its space
        # Bottom progress + log (fixed height, won't shrink)
        bottom_frame = ctk.CTkFrame(master=self, fg_color="transparent", height=180)
        bottom_frame.pack(fill="x", side="bottom", padx=15, pady=(5, 15))
        bottom_frame.pack_propagate(False)  # CRITICAL: Prevents shrinking below set height

        self.progress_bar = ctk.CTkProgressBar(master=bottom_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 10))

        self.log_textbox = ctk.CTkTextbox(master=bottom_frame, height=150)
        self.log_textbox.pack(fill="both", expand=True)

        # Tab view - packed AFTER bottom frame, fills remaining space
        self.tab_view = ctk.CTkTabview(master=self)
        self.tab_view.pack(fill="both", expand=True, padx=15, pady=(15, 5))

        self.tab_view.add("Analyze")
        self.tab_view.add("Extract")
        self.tab_view.add("Reframe")
        self.tab_view.add("LUT")
        self.tab_view.add("Sky Filter")
        self.tab_view.add("Segment")
        self.tab_view.add("Pipeline")
        self.tab_view.add("Fisheye")
        self.tab_view.add("Gaps")

        self._build_analyze_tab()
        self._build_extract_tab()
        self._build_reframe_tab()
        self._build_lut_tab()
        self._build_sky_tab()
        self._build_segment_tab()
        self._build_pipeline_tab()
        self._build_fisheye_tab()
        self._build_gaps_tab()

    # =========================================================================
    # TAB: Analyze
    # =========================================================================
    def _build_analyze_tab(self):
        tab = self.tab_view.tab("Analyze")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Video File")

        vid_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        vid_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(vid_frame, text="Video:", width=80, anchor="w").pack(side="left")
        self.analyze_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="360 video file...")
        self.analyze_video_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(vid_frame, text="...", width=40,
                      command=self._browse_analyze_video).pack(side="left")

        self.analyze_run_btn = ctk.CTkButton(
            left, text="Analyze Video", command=self.run_analyze,
            fg_color="#1976D2", hover_color="#1565C0",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.analyze_run_btn.pack(pady=20, padx=10, fill="x")

        # Right: Results
        ctk.CTkLabel(right, text="Analysis Results",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        self.analyze_results = ctk.CTkTextbox(right, height=400)
        self.analyze_results.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.analyze_results.insert("1.0", "Select a video file and click 'Analyze Video'\n\n"
                                          "This will show:\n"
                                          "- Video metadata (resolution, fps, duration)\n"
                                          "- Equirectangular detection\n"
                                          "- Log format detection\n"
                                          "- Recommended extraction interval\n"
                                          "- Suggested LUT if applicable")

    def _browse_analyze_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")]
        )
        if path:
            self.analyze_video_entry.delete(0, 'end')
            self.analyze_video_entry.insert(0, path)

    def run_analyze(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        video = self.analyze_video_entry.get()
        if not video:
            self.log("Error: Please select a video file")
            return
        if not Path(video).exists():
            self.log(f"Error: Video not found: {video}")
            return

        self.analyze_run_btn.configure(state="disabled")
        threading.Thread(target=self._analyze_worker, args=(video,), daemon=True).start()

    def _analyze_worker(self, video):
        try:
            self.log(f"Analyzing: {Path(video).name}")
            analyzer = VideoAnalyzer()
            info = analyzer.analyze(video)

            result_text = f"""File: {info.filename}
Format: {info.format} ({info.codec})
Resolution: {info.width}x{info.height}
FPS: {info.fps:.2f}
Duration: {analyzer.get_duration_formatted(info)} ({info.duration_seconds:.1f}s)
Frames: {info.frame_count}
Bitrate: {info.bitrate // 1000 if info.bitrate else '?'} kbps
Pixel Format: {info.pixel_format or 'Unknown'}

=== 360 Detection ===
Equirectangular: {'Yes' if info.is_equirectangular else 'No'}
Aspect Ratio: {info.width/info.height:.2f}:1

=== Color Profile ===
Log Format: {info.detected_log_type or 'None detected'}
Is Log: {'Yes' if info.is_log_format else 'No'}

=== Recommendations ===
Extraction Interval: {info.recommended_interval}s
Estimated Frames @ {info.recommended_interval}s: {int(info.duration_seconds / info.recommended_interval)}
"""
            if info.recommended_lut:
                result_text += f"Recommended LUT: {info.recommended_lut}\n"

            self.after(0, lambda: self._update_analyze_results(result_text))
            self.log("Analysis complete")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self.analyze_run_btn.configure(state="normal"))

    def _update_analyze_results(self, text):
        self.analyze_results.delete("1.0", "end")
        self.analyze_results.insert("1.0", text)

    # =========================================================================
    # TAB: Extract
    # =========================================================================
    def _build_extract_tab(self):
        from prep360.core.queue_manager import VideoQueue

        tab = self.tab_view.tab("Extract")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Initialize video queue and planned frames
        self.video_queue = VideoQueue()
        self.extract_queue_processing = False
        self.extract_current_item_id = None
        self.extract_planned_frames: List[PlannedFrame] = []

        # Create scrollable left panel for settings
        left = ctk.CTkScrollableFrame(master=container, width=380)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # === LEFT: Settings ===
        self._section_header(left, "Output Folder")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        self.extract_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Frames output folder...")
        self.extract_output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.extract_output_entry)).pack(side="left")

        self._section_header(left, "Extraction Settings")

        # Mode
        mode_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        mode_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(mode_frame, text="Mode:", width=70, anchor="w").pack(side="left")
        self.extract_mode_var = ctk.StringVar(value="fixed")
        ctk.CTkComboBox(mode_frame, variable=self.extract_mode_var,
                        values=["fixed", "scene", "adaptive"], state="readonly", width=120
                        ).pack(side="left", padx=(5, 0))

        # Interval
        int_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        int_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(int_frame, text="Interval:", width=70, anchor="w").pack(side="left")
        self.extract_interval_var = ctk.DoubleVar(value=2.0)
        self.extract_interval_label = ctk.CTkLabel(int_frame, text="2.0s", width=40)
        self.extract_interval_label.pack(side="right")
        ctk.CTkSlider(int_frame, from_=0.5, to=10, variable=self.extract_interval_var,
                      command=lambda v: self.extract_interval_label.configure(text=f"{float(v):.1f}s")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Quality
        qual_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        qual_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(qual_frame, text="Quality:", width=70, anchor="w").pack(side="left")
        self.extract_quality_var = ctk.IntVar(value=95)
        self.extract_quality_label = ctk.CTkLabel(qual_frame, text="95", width=40)
        self.extract_quality_label.pack(side="right")
        ctk.CTkSlider(qual_frame, from_=70, to=100, variable=self.extract_quality_var,
                      command=lambda v: self.extract_quality_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Format
        fmt_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        fmt_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(fmt_frame, text="Format:", width=70, anchor="w").pack(side="left")
        self.extract_format_var = ctk.StringVar(value="jpg")
        ctk.CTkRadioButton(fmt_frame, text="JPEG", variable=self.extract_format_var,
                           value="jpg").pack(side="left", padx=(5, 15))
        ctk.CTkRadioButton(fmt_frame, text="PNG", variable=self.extract_format_var,
                           value="png").pack(side="left")

        # Time range (optional)
        self._section_header(left, "Time Range (Optional)")

        time_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        time_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(time_frame, text="Start:", width=40, anchor="w").pack(side="left")
        self.extract_start_entry = ctk.CTkEntry(time_frame, width=70, placeholder_text="0:00")
        self.extract_start_entry.pack(side="left", padx=(5, 10))
        ctk.CTkLabel(time_frame, text="End:", width=35, anchor="w").pack(side="left")
        self.extract_end_entry = ctk.CTkEntry(time_frame, width=70, placeholder_text="end")
        self.extract_end_entry.pack(side="left", padx=(5, 0))

        # === Collapsible: Color & LUT ===
        self.extract_lut_section = CollapsibleSection(left, "Color & LUT", expanded=False)
        self.extract_lut_section.pack(fill="x", pady=(10, 0), padx=5)

        # Enable LUT checkbox
        self.extract_lut_enabled_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.extract_lut_section.content, text="Apply LUT after extraction",
            variable=self.extract_lut_enabled_var,
            command=self._on_extract_lut_toggle
        ).pack(pady=3, anchor="w")

        # LUT file picker
        lut_file_frame = ctk.CTkFrame(master=self.extract_lut_section.content, fg_color="transparent")
        lut_file_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(lut_file_frame, text="LUT:", width=50, anchor="w").pack(side="left")
        self.extract_lut_file_entry = ctk.CTkEntry(lut_file_frame, placeholder_text=".cube file...")
        self.extract_lut_file_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(lut_file_frame, text="...", width=35,
                      command=self._browse_extract_lut_file).pack(side="left")

        # LUT strength slider
        lut_str_frame = ctk.CTkFrame(master=self.extract_lut_section.content, fg_color="transparent")
        lut_str_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(lut_str_frame, text="Strength:", width=60, anchor="w").pack(side="left")
        self.extract_lut_strength_var = ctk.DoubleVar(value=1.0)
        self.extract_lut_strength_label = ctk.CTkLabel(lut_str_frame, text="100%", width=45)
        self.extract_lut_strength_label.pack(side="right")
        ctk.CTkSlider(lut_str_frame, from_=0, to=1, variable=self.extract_lut_strength_var,
                      command=lambda v: self.extract_lut_strength_label.configure(text=f"{int(float(v)*100)}%")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Shadow/Highlight adjustments
        shadow_frame = ctk.CTkFrame(master=self.extract_lut_section.content, fg_color="transparent")
        shadow_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(shadow_frame, text="Shadows:", width=60, anchor="w").pack(side="left")
        self.extract_shadow_var = ctk.IntVar(value=50)
        self.extract_shadow_label = ctk.CTkLabel(shadow_frame, text="50", width=35)
        self.extract_shadow_label.pack(side="right")
        ctk.CTkSlider(shadow_frame, from_=0, to=100, variable=self.extract_shadow_var,
                      command=lambda v: self.extract_shadow_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        highlight_frame = ctk.CTkFrame(master=self.extract_lut_section.content, fg_color="transparent")
        highlight_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(highlight_frame, text="Highlights:", width=60, anchor="w").pack(side="left")
        self.extract_highlight_var = ctk.IntVar(value=50)
        self.extract_highlight_label = ctk.CTkLabel(highlight_frame, text="50", width=35)
        self.extract_highlight_label.pack(side="right")
        ctk.CTkSlider(highlight_frame, from_=0, to=100, variable=self.extract_highlight_var,
                      command=lambda v: self.extract_highlight_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        ctk.CTkLabel(self.extract_lut_section.content, text="50 = neutral",
                     text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")

        # === Collapsible: Sky Filter ===
        self.extract_sky_section = CollapsibleSection(left, "Sky Filter", expanded=False)
        self.extract_sky_section.pack(fill="x", pady=(5, 0), padx=5)

        # Enable Sky Filter checkbox
        self.extract_sky_enabled_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.extract_sky_section.content, text="Remove sky-dominated images",
            variable=self.extract_sky_enabled_var
        ).pack(pady=3, anchor="w")

        # Brightness threshold
        sky_br_frame = ctk.CTkFrame(master=self.extract_sky_section.content, fg_color="transparent")
        sky_br_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(sky_br_frame, text="Brightness:", width=70, anchor="w").pack(side="left")
        self.extract_sky_brightness_var = ctk.DoubleVar(value=0.85)
        self.extract_sky_brightness_label = ctk.CTkLabel(sky_br_frame, text="0.85", width=40)
        self.extract_sky_brightness_label.pack(side="right")
        ctk.CTkSlider(sky_br_frame, from_=0.5, to=1.0, variable=self.extract_sky_brightness_var,
                      command=lambda v: self.extract_sky_brightness_label.configure(text=f"{float(v):.2f}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Keypoints threshold
        sky_kp_frame = ctk.CTkFrame(master=self.extract_sky_section.content, fg_color="transparent")
        sky_kp_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(sky_kp_frame, text="Keypoints:", width=70, anchor="w").pack(side="left")
        self.extract_sky_keypoints_var = ctk.IntVar(value=50)
        self.extract_sky_keypoints_label = ctk.CTkLabel(sky_kp_frame, text="50", width=40)
        self.extract_sky_keypoints_label.pack(side="right")
        ctk.CTkSlider(sky_kp_frame, from_=10, to=200, variable=self.extract_sky_keypoints_var,
                      command=lambda v: self.extract_sky_keypoints_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Pitch threshold (for zenith detection)
        sky_pitch_frame = ctk.CTkFrame(master=self.extract_sky_section.content, fg_color="transparent")
        sky_pitch_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(sky_pitch_frame, text="Min Pitch:", width=70, anchor="w").pack(side="left")
        self.extract_sky_pitch_var = ctk.IntVar(value=60)
        self.extract_sky_pitch_label = ctk.CTkLabel(sky_pitch_frame, text="60°", width=40)
        self.extract_sky_pitch_label.pack(side="right")
        ctk.CTkSlider(sky_pitch_frame, from_=30, to=90, variable=self.extract_sky_pitch_var,
                      command=lambda v: self.extract_sky_pitch_label.configure(text=f"{int(v)}°")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        ctk.CTkLabel(self.extract_sky_section.content, text="Only filter images above pitch angle",
                     text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")

        # === Collapsible: Blur Filter ===
        self.extract_blur_section = CollapsibleSection(left, "Blur Filter", expanded=False)
        self.extract_blur_section.pack(fill="x", pady=(5, 0), padx=5)

        # Enable blur filter checkbox
        self.extract_blur_enabled_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.extract_blur_section.content, text="Filter blurry frames after extraction",
            variable=self.extract_blur_enabled_var
        ).pack(pady=3, anchor="w")

        # Percentile slider
        blur_pct_frame = ctk.CTkFrame(master=self.extract_blur_section.content, fg_color="transparent")
        blur_pct_frame.pack(pady=3, fill="x")
        ctk.CTkLabel(blur_pct_frame, text="Keep:", width=70, anchor="w").pack(side="left")
        self.extract_blur_percentile_var = ctk.IntVar(value=80)
        self.extract_blur_pct_label = ctk.CTkLabel(blur_pct_frame, text="80%", width=40)
        self.extract_blur_pct_label.pack(side="right")
        ctk.CTkSlider(blur_pct_frame, from_=50, to=100, variable=self.extract_blur_percentile_var,
                      command=lambda v: self.extract_blur_pct_label.configure(text=f"{int(v)}%")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        ctk.CTkLabel(self.extract_blur_section.content, text="Percent of sharpest frames to keep",
                     text_color="gray", font=ctk.CTkFont(size=10)).pack(anchor="w")

        # === Launch Mask Tool button ===
        self.extract_mask_btn = ctk.CTkButton(
            left, text="Launch Mask Tool...",
            command=self._launch_segmentation_window,
            fg_color="#555555", hover_color="#666666", height=32
        )
        self.extract_mask_btn.pack(pady=(15, 10), padx=10, fill="x")

        # === RIGHT: Queue and Preview ===
        # Create notebook-style tabs on right side for Queue / Plan / Preview
        self.extract_right_tabs = ctk.CTkTabview(right, height=50)
        self.extract_right_tabs.pack(fill="both", expand=True, padx=5, pady=5)

        self.extract_right_tabs.add("Queue")
        self.extract_right_tabs.add("Plan")
        self.extract_right_tabs.add("LUT Preview")

        # === Queue Tab ===
        queue_tab = self.extract_right_tabs.tab("Queue")
        ctk.CTkLabel(queue_tab, text="Video Queue",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5), padx=15, anchor="w")

        # Queue controls
        queue_ctrl_frame = ctk.CTkFrame(master=queue_tab, fg_color="transparent")
        queue_ctrl_frame.pack(fill="x", padx=15, pady=(0, 5))

        ctk.CTkButton(queue_ctrl_frame, text="Add Videos", width=90,
                      command=self._queue_add_videos).pack(side="left", padx=(0, 5))
        ctk.CTkButton(queue_ctrl_frame, text="Add Folder", width=90,
                      command=self._queue_add_folder).pack(side="left", padx=(0, 5))
        ctk.CTkButton(queue_ctrl_frame, text="Remove", width=70,
                      command=self._queue_remove_selected,
                      fg_color="#666666").pack(side="left", padx=(0, 5))
        ctk.CTkButton(queue_ctrl_frame, text="Clear Done", width=80,
                      command=self._queue_clear_done,
                      fg_color="#666666").pack(side="left", padx=(0, 5))

        # Reorder buttons
        ctk.CTkButton(queue_ctrl_frame, text="Move Up", width=70,
                      command=self._queue_move_up,
                      fg_color="#555555").pack(side="right", padx=(5, 0))
        ctk.CTkButton(queue_ctrl_frame, text="Move Down", width=80,
                      command=self._queue_move_down,
                      fg_color="#555555").pack(side="right", padx=(5, 0))

        # Queue list frame
        queue_list_frame = ctk.CTkFrame(master=queue_tab)
        queue_list_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Create scrollable frame for queue items
        self.queue_scroll = ctk.CTkScrollableFrame(queue_list_frame, fg_color="transparent")
        self.queue_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Queue item widgets will be stored here
        self.queue_item_widgets = {}

        # Queue stats
        self.queue_stats_label = ctk.CTkLabel(queue_tab, text="Queue: 0 pending, 0 done",
                                               text_color="gray")
        self.queue_stats_label.pack(pady=(0, 5), padx=15, anchor="w")

        # Process buttons
        btn_frame = ctk.CTkFrame(master=queue_tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.extract_run_btn = ctk.CTkButton(
            btn_frame, text="Process Queue", command=self.run_extract_queue,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.extract_run_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.extract_stop_btn = ctk.CTkButton(
            btn_frame, text="Stop", command=self._queue_stop_processing,
            fg_color="#C62828", hover_color="#8B0000", height=45, width=80
        )
        self.extract_stop_btn.pack(side="left")
        self.extract_stop_btn.pack_forget()  # Hidden initially

        # === Plan Tab ===
        plan_tab = self.extract_right_tabs.tab("Plan")
        ctk.CTkLabel(plan_tab, text="Extraction Plan",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5), padx=15, anchor="w")

        ctk.CTkLabel(plan_tab, text="Analyze video to see planned frame extraction",
                     text_color="gray").pack(pady=(0, 5), padx=15, anchor="w")

        # Analyze button
        plan_ctrl_frame = ctk.CTkFrame(master=plan_tab, fg_color="transparent")
        plan_ctrl_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.plan_analyze_btn = ctk.CTkButton(
            plan_ctrl_frame, text="Analyze Plan", width=120,
            command=self._analyze_extraction_plan,
            fg_color="#1976D2", hover_color="#1565C0"
        )
        self.plan_analyze_btn.pack(side="left", padx=(0, 10))

        ctk.CTkButton(plan_ctrl_frame, text="Insert Frame", width=100,
                      command=self._plan_insert_frame,
                      fg_color="#555555").pack(side="left", padx=(0, 5))
        ctk.CTkButton(plan_ctrl_frame, text="Delete Selected", width=110,
                      command=self._plan_delete_selected,
                      fg_color="#555555").pack(side="left")

        # Plan list frame
        plan_list_frame = ctk.CTkFrame(master=plan_tab)
        plan_list_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.plan_scroll = ctk.CTkScrollableFrame(plan_list_frame, fg_color="transparent")
        self.plan_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.plan_item_widgets = []

        # Plan stats
        self.plan_stats_label = ctk.CTkLabel(plan_tab, text="Frames: 0 planned",
                                              text_color="gray")
        self.plan_stats_label.pack(pady=(0, 5), padx=15, anchor="w")

        # === LUT Preview Tab ===
        preview_tab = self.extract_right_tabs.tab("LUT Preview")
        ctk.CTkLabel(preview_tab, text="LUT Preview",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5), padx=15, anchor="w")

        ctk.CTkLabel(preview_tab, text="Compare before/after LUT application",
                     text_color="gray").pack(pady=(0, 5), padx=15, anchor="w")

        # Frame slider
        slider_frame = ctk.CTkFrame(master=preview_tab, fg_color="transparent")
        slider_frame.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(slider_frame, text="Frame:", width=50, anchor="w").pack(side="left")
        self.lut_preview_frame_var = ctk.IntVar(value=0)
        self.lut_preview_frame_label = ctk.CTkLabel(slider_frame, text="0", width=40)
        self.lut_preview_frame_label.pack(side="right")
        self.lut_preview_slider = ctk.CTkSlider(
            slider_frame, from_=0, to=100, variable=self.lut_preview_frame_var,
            command=self._on_lut_preview_frame_change
        )
        self.lut_preview_slider.pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Preview images container
        preview_images_frame = ctk.CTkFrame(master=preview_tab)
        preview_images_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Before/After side by side
        self.lut_preview_before_frame = ctk.CTkFrame(preview_images_frame, fg_color="#1a1a1a")
        self.lut_preview_before_frame.pack(side="left", fill="both", expand=True, padx=(5, 2), pady=5)
        ctk.CTkLabel(self.lut_preview_before_frame, text="Before",
                     font=ctk.CTkFont(size=11)).pack(pady=(5, 0))
        self.lut_preview_before = ctk.CTkLabel(self.lut_preview_before_frame, text="No preview",
                                                fg_color="transparent")
        self.lut_preview_before.pack(fill="both", expand=True, padx=5, pady=5)

        self.lut_preview_after_frame = ctk.CTkFrame(preview_images_frame, fg_color="#1a1a1a")
        self.lut_preview_after_frame.pack(side="left", fill="both", expand=True, padx=(2, 5), pady=5)
        ctk.CTkLabel(self.lut_preview_after_frame, text="After",
                     font=ctk.CTkFont(size=11)).pack(pady=(5, 0))
        self.lut_preview_after = ctk.CTkLabel(self.lut_preview_after_frame, text="No preview",
                                               fg_color="transparent")
        self.lut_preview_after.pack(fill="both", expand=True, padx=5, pady=5)

        # Update preview button
        ctk.CTkButton(preview_tab, text="Update Preview",
                      command=self._update_lut_preview,
                      fg_color="#7B1FA2", hover_color="#6A1B9A"
                      ).pack(pady=(0, 10), padx=15)

        # Load saved queue and update display
        self._queue_refresh_display()

    def _queue_add_videos(self):
        """Add video files to the queue."""
        files = filedialog.askopenfilenames(
            title="Select Videos",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")]
        )
        if files:
            count = self.video_queue.add_videos(list(files))
            self.log(f"Added {count} videos to queue")
            self._queue_refresh_display()

    def _queue_add_folder(self):
        """Add all videos from a folder to the queue."""
        folder = filedialog.askdirectory(title="Select Folder with Videos")
        if folder:
            count = self.video_queue.add_folder(folder)
            self.log(f"Added {count} videos from folder")
            self._queue_refresh_display()

    def _queue_remove_selected(self):
        """Remove selected items from queue."""
        for item_id, widgets in list(self.queue_item_widgets.items()):
            if widgets.get("selected", False):
                self.video_queue.remove_item(item_id)
        self._queue_refresh_display()

    def _queue_clear_done(self):
        """Clear completed and error items."""
        self.video_queue.clear_completed()
        self._queue_refresh_display()

    def _queue_move_up(self):
        """Move selected item up."""
        for item_id, widgets in self.queue_item_widgets.items():
            if widgets.get("selected", False):
                self.video_queue.move_up(item_id)
                break
        self._queue_refresh_display()

    def _queue_move_down(self):
        """Move selected item down."""
        for item_id, widgets in self.queue_item_widgets.items():
            if widgets.get("selected", False):
                self.video_queue.move_down(item_id)
                break
        self._queue_refresh_display()

    def _queue_refresh_display(self):
        """Refresh the queue display."""
        # Clear existing widgets
        for widgets in self.queue_item_widgets.values():
            if "frame" in widgets:
                widgets["frame"].destroy()
        self.queue_item_widgets = {}

        # Create widgets for each item
        for item in self.video_queue.items:
            self._queue_create_item_widget(item)

        # Update stats
        stats = self.video_queue.get_stats()
        self.queue_stats_label.configure(
            text=f"Queue: {stats['pending']} pending, {stats['processing']} processing, "
                 f"{stats['done']} done, {stats['error']} errors"
        )

    def _queue_create_item_widget(self, item):
        """Create a widget for a queue item."""
        frame = ctk.CTkFrame(self.queue_scroll, fg_color="#2b2b2b", corner_radius=5)
        frame.pack(fill="x", pady=2, padx=2)

        # Status colors
        status_colors = {
            "pending": "#888888",
            "processing": "#FFA500",
            "done": "#4CAF50",
            "error": "#F44336",
            "cancelled": "#9E9E9E",
        }

        # Selection state
        selected = False

        def toggle_select(event=None):
            nonlocal selected
            selected = not selected
            self.queue_item_widgets[item.id]["selected"] = selected
            frame.configure(fg_color="#3d5a80" if selected else "#2b2b2b")

        frame.bind("<Button-1>", toggle_select)

        # Status indicator
        status_color = status_colors.get(item.status, "#888888")
        status_label = ctk.CTkLabel(frame, text="●", text_color=status_color, width=20)
        status_label.pack(side="left", padx=(10, 5))

        # Filename
        name_label = ctk.CTkLabel(frame, text=item.filename, anchor="w", width=200)
        name_label.pack(side="left", padx=(0, 10))
        name_label.bind("<Button-1>", toggle_select)

        # Status text
        status_text = item.status.capitalize()
        if item.status == "processing":
            status_text = f"Processing {item.progress}%"
        elif item.status == "done":
            status_text = f"Done ({item.frame_count} frames)"
        elif item.status == "error":
            status_text = f"Error"

        status_text_label = ctk.CTkLabel(frame, text=status_text, text_color=status_color, width=120)
        status_text_label.pack(side="left", padx=(0, 10))

        # Progress bar (for processing items)
        if item.status == "processing":
            progress = ctk.CTkProgressBar(frame, width=100, height=10)
            progress.set(item.progress / 100)
            progress.pack(side="left", padx=(0, 10))
        else:
            progress = None

        # Store widget references
        self.queue_item_widgets[item.id] = {
            "frame": frame,
            "status_label": status_label,
            "name_label": name_label,
            "status_text_label": status_text_label,
            "progress": progress,
            "selected": selected,
        }

    def _queue_update_item_widget(self, item_id):
        """Update a single item's widget."""
        item = self.video_queue.get_item(item_id)
        widgets = self.queue_item_widgets.get(item_id)

        if not item or not widgets:
            return

        status_colors = {
            "pending": "#888888",
            "processing": "#FFA500",
            "done": "#4CAF50",
            "error": "#F44336",
            "cancelled": "#9E9E9E",
        }

        status_color = status_colors.get(item.status, "#888888")
        widgets["status_label"].configure(text_color=status_color)

        status_text = item.status.capitalize()
        if item.status == "processing":
            status_text = f"Processing {item.progress}%"
        elif item.status == "done":
            status_text = f"Done ({item.frame_count} frames)"
        elif item.status == "error":
            status_text = "Error"

        widgets["status_text_label"].configure(text=status_text, text_color=status_color)

        if widgets["progress"] and item.status == "processing":
            widgets["progress"].set(item.progress / 100)

    def run_extract_queue(self):
        """Start processing the video queue."""
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        output = self.extract_output_entry.get()
        if not output:
            self.log("Error: Please select output folder")
            return

        if self.video_queue.get_pending_count() == 0:
            self.log("No pending videos in queue")
            return

        self.extract_queue_processing = True
        self.cancel_flag.clear()
        self.extract_run_btn.configure(state="disabled")
        self.extract_stop_btn.pack(side="left")
        self.progress_bar.start()

        threading.Thread(target=self._extract_queue_worker, daemon=True).start()

    def _extract_queue_worker(self):
        """Worker thread for processing the video queue."""
        try:
            while self.extract_queue_processing and not self.cancel_flag.is_set():
                # Get next pending item
                item = self.video_queue.get_next_pending()
                if not item:
                    break

                self.extract_current_item_id = item.id
                self.video_queue.set_processing(item.id)
                self.after(0, lambda: self._queue_refresh_display())

                self.log(f"\n{'='*50}")
                self.log(f"Processing: {item.filename}")

                try:
                    # Build config from current settings
                    config = ExtractionConfig(
                        interval=self.extract_interval_var.get(),
                        mode=ExtractionMode(self.extract_mode_var.get()),
                        quality=self.extract_quality_var.get(),
                        output_format=self.extract_format_var.get(),
                    )

                    # Time range
                    start = self.extract_start_entry.get().strip()
                    end = self.extract_end_entry.get().strip()
                    if start:
                        config.start_sec = self._parse_time(start)
                    if end:
                        config.end_sec = self._parse_time(end)

                    # Create output subfolder for this video
                    base_output = Path(self.extract_output_entry.get())
                    video_name = Path(item.filename).stem
                    output_dir = base_output / video_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    extractor = FrameExtractor()

                    def progress(curr, total, msg):
                        if total > 0:
                            pct = int(curr / total * 100)
                            self.video_queue.set_progress(item.id, pct)
                            self.after(0, lambda: self._queue_update_item_widget(item.id))

                    self.log(f"Output: {output_dir}")
                    self.log(f"Mode: {config.mode.value}, Interval: {config.interval}s")

                    result = extractor.extract(
                        item.video_path, str(output_dir), config,
                        progress_callback=progress
                    )

                    if self.cancel_flag.is_set():
                        self.video_queue.set_cancelled(item.id)
                        self.log(f"Cancelled: {item.filename}")
                        break

                    if result.success:
                        import cv2
                        ext = config.output_format
                        images = sorted(output_dir.glob(f"*.{ext}"))
                        final_count = len(images)

                        # Step 1: Apply LUT if enabled
                        lut_enabled = self.extract_lut_enabled_var.get()
                        lut_path = self.extract_lut_file_entry.get()
                        if lut_enabled and lut_path and Path(lut_path).exists():
                            self.log(f"Applying LUT: {Path(lut_path).name}")
                            processor = LUTProcessor()
                            strength = self.extract_lut_strength_var.get()

                            for i, img_path in enumerate(images):
                                if self.cancel_flag.is_set():
                                    break
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    processed = processor.apply_lut(img, lut_path, strength)
                                    if ext in ('jpg', 'jpeg'):
                                        cv2.imwrite(str(img_path), processed,
                                                   [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                                    else:
                                        cv2.imwrite(str(img_path), processed)

                        # Step 2: Apply shadow/highlight adjustments if needed
                        shadow = self.extract_shadow_var.get()
                        highlight = self.extract_highlight_var.get()

                        if shadow != 50 or highlight != 50:
                            self.log(f"Applying shadow/highlight adjustments...")
                            from prep360.core.adjustments import apply_shadow_highlight

                            for i, img_path in enumerate(images):
                                if self.cancel_flag.is_set():
                                    break
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    adjusted = apply_shadow_highlight(img, shadow, highlight)
                                    if ext in ('jpg', 'jpeg'):
                                        cv2.imwrite(str(img_path), adjusted,
                                                   [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                                    else:
                                        cv2.imwrite(str(img_path), adjusted)

                        # Step 3: Apply sky filter if enabled
                        sky_enabled = self.extract_sky_enabled_var.get()
                        if sky_enabled:
                            self.log(f"Running sky filter...")
                            sky_config = SkyFilterConfig(
                                brightness_threshold=self.extract_sky_brightness_var.get(),
                                keypoint_threshold=self.extract_sky_keypoints_var.get(),
                            )
                            sky_filter = SkyFilter(sky_config)

                            # Re-read current images after processing
                            images = sorted(output_dir.glob(f"*.{ext}"))
                            sky_removed = 0

                            for img_path in images:
                                if self.cancel_flag.is_set():
                                    break
                                # Check if image is sky-dominated
                                metrics = sky_filter.analyze_image(str(img_path))
                                if metrics.is_sky:
                                    img_path.unlink()  # Delete sky image
                                    sky_removed += 1

                            if sky_removed > 0:
                                self.log(f"Removed {sky_removed} sky-dominated images")
                                final_count -= sky_removed

                        # Step 4: Apply blur filter if enabled
                        blur_enabled = self.extract_blur_enabled_var.get()
                        if blur_enabled and not self.cancel_flag.is_set():
                            self.log(f"Running blur filter...")
                            from prep360.core.blur_filter import BlurFilter, BlurFilterConfig

                            pct = self.extract_blur_percentile_var.get()
                            blur_config = BlurFilterConfig(percentile=float(pct), workers=4)
                            bf = BlurFilter(blur_config)

                            # Re-read current images after previous steps
                            ext = config.output_format
                            images = sorted(output_dir.glob(f"*.{ext}"))
                            scores = bf.analyze_batch(str(output_dir))

                            if scores:
                                import numpy as np
                                score_vals = [s.score for s in scores]
                                cutoff = float(np.percentile(score_vals, 100 - pct))
                                blur_removed = 0
                                for s in scores:
                                    if s.score < cutoff:
                                        p = output_dir / s.image_name
                                        if p.exists():
                                            p.unlink()
                                            blur_removed += 1
                                if blur_removed > 0:
                                    self.log(f"Removed {blur_removed} blurry frames (kept top {pct}%)")
                                    final_count -= blur_removed

                        self.video_queue.set_done(item.id, final_count)
                        self.log(f"Done: {final_count} frames extracted")
                    else:
                        self.video_queue.set_error(item.id, result.error or "Unknown error")
                        self.log(f"Error: {result.error}")

                except Exception as e:
                    self.video_queue.set_error(item.id, str(e))
                    self.log(f"Error: {e}")

                self.after(0, lambda: self._queue_refresh_display())

            # Queue complete
            stats = self.video_queue.get_stats()
            self.log(f"\n{'='*50}")
            self.log(f"Queue processing complete")
            self.log(f"Done: {stats['done']}, Errors: {stats['error']}, Cancelled: {stats['cancelled']}")

        except Exception as e:
            self.log(f"Queue error: {e}")
        finally:
            self.extract_queue_processing = False
            self.extract_current_item_id = None
            self.after(0, self._queue_processing_done)

    def _queue_processing_done(self):
        """Called when queue processing is complete."""
        self.extract_run_btn.configure(state="normal")
        self.extract_stop_btn.pack_forget()
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        self._queue_refresh_display()

    def _queue_stop_processing(self):
        """Stop queue processing."""
        self.log("Stopping queue processing...")
        self.cancel_flag.set()
        self.extract_queue_processing = False

    # Legacy single-video extract (kept for compatibility)
    def run_extract(self):
        """Run extraction for single video (legacy)."""
        self.run_extract_queue()

    def _extract_worker(self, video, output):
        """Legacy worker - now handled by queue worker."""
        pass  # Deprecated, use queue worker

    def _parse_time(self, time_str):
        """Parse time string (seconds or HH:MM:SS) to seconds."""
        try:
            return float(time_str)
        except ValueError:
            parts = time_str.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return 0

    # -------------------------------------------------------------------------
    # Extract Tab: LUT, Sky Filter, Plan helpers
    # -------------------------------------------------------------------------
    def _on_extract_lut_toggle(self):
        """Called when LUT enabled checkbox changes."""
        enabled = self.extract_lut_enabled_var.get()
        if enabled:
            self.extract_lut_section.expand()

    def _browse_extract_lut_file(self):
        """Browse for LUT file in extract tab."""
        path = filedialog.askopenfilename(
            title="Select LUT File",
            filetypes=[("CUBE Files", "*.cube"), ("All Files", "*.*")]
        )
        if path:
            self.extract_lut_file_entry.delete(0, 'end')
            self.extract_lut_file_entry.insert(0, path)

    def _analyze_extraction_plan(self):
        """Analyze video to generate extraction plan."""
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        # Get selected video from queue
        selected_video = None
        for item_id, widgets in self.queue_item_widgets.items():
            if widgets.get("selected", False):
                item = self.video_queue.get_item(item_id)
                if item:
                    selected_video = item.video_path
                    break

        if not selected_video:
            # Try first pending video
            for item in self.video_queue.items:
                if item.status == "pending":
                    selected_video = item.video_path
                    break

        if not selected_video:
            self.log("No video selected or queued for analysis")
            return

        self.plan_analyze_btn.configure(state="disabled")
        threading.Thread(target=self._analyze_plan_worker, args=(selected_video,), daemon=True).start()

    def _analyze_plan_worker(self, video_path):
        """Worker thread to analyze video and generate extraction plan."""
        try:
            analyzer = VideoAnalyzer()
            info = analyzer.analyze(video_path)

            interval = self.extract_interval_var.get()
            mode = self.extract_mode_var.get()

            # Parse time range
            start_sec = 0
            end_sec = info.duration_seconds
            start_str = self.extract_start_entry.get().strip()
            end_str = self.extract_end_entry.get().strip()
            if start_str:
                start_sec = self._parse_time(start_str)
            if end_str:
                end_sec = self._parse_time(end_str)

            # Generate planned frames
            self.extract_planned_frames = []
            current_time = start_sec
            idx = 0

            while current_time <= end_sec:
                frame_num = int(current_time * info.fps)
                self.extract_planned_frames.append(PlannedFrame(
                    index=idx,
                    frame_number=frame_num,
                    timestamp=current_time,
                    reason="interval",
                    included=True
                ))
                current_time += interval
                idx += 1

            self.after(0, self._refresh_plan_display)
            self.log(f"Plan: {len(self.extract_planned_frames)} frames @ {interval}s interval")

        except Exception as e:
            self.log(f"Error analyzing plan: {e}")
        finally:
            self.after(0, lambda: self.plan_analyze_btn.configure(state="normal"))

    def _refresh_plan_display(self):
        """Refresh the plan display with current planned frames."""
        # Clear existing widgets
        for widget in self.plan_item_widgets:
            widget.destroy()
        self.plan_item_widgets = []

        # Create widgets for each planned frame
        for pf in self.extract_planned_frames:
            frame = ctk.CTkFrame(self.plan_scroll, fg_color="#2b2b2b", corner_radius=5, height=32)
            frame.pack(fill="x", pady=1, padx=2)
            frame.pack_propagate(False)

            # Include checkbox
            var = ctk.BooleanVar(value=pf.included)
            cb = ctk.CTkCheckBox(frame, text="", variable=var, width=24,
                                 command=lambda idx=pf.index, v=var: self._on_plan_include_change(idx, v))
            cb.pack(side="left", padx=(10, 5))

            # Frame info
            time_str = f"{int(pf.timestamp // 60):02d}:{pf.timestamp % 60:05.2f}"
            info_text = f"#{pf.index:03d}  {time_str}  F{pf.frame_number}"
            ctk.CTkLabel(frame, text=info_text, font=ctk.CTkFont(family="Consolas", size=11),
                         anchor="w", width=180).pack(side="left", padx=(0, 10))

            # Reason badge
            reason_colors = {"interval": "#4CAF50", "scene": "#FFA500", "manual": "#2196F3"}
            ctk.CTkLabel(frame, text=pf.reason, text_color=reason_colors.get(pf.reason, "gray"),
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(0, 10))

            self.plan_item_widgets.append(frame)

        # Update stats
        included_count = sum(1 for pf in self.extract_planned_frames if pf.included)
        self.plan_stats_label.configure(
            text=f"Frames: {included_count} included / {len(self.extract_planned_frames)} total"
        )

    def _on_plan_include_change(self, idx: int, var):
        """Handle checkbox change for planned frame."""
        for pf in self.extract_planned_frames:
            if pf.index == idx:
                pf.included = var.get()
                break
        # Update stats
        included_count = sum(1 for pf in self.extract_planned_frames if pf.included)
        self.plan_stats_label.configure(
            text=f"Frames: {included_count} included / {len(self.extract_planned_frames)} total"
        )

    def _plan_insert_frame(self):
        """Insert a manual frame into the plan."""
        if not self.extract_planned_frames:
            self.log("Generate a plan first")
            return

        # Simple dialog for timestamp
        dialog = ctk.CTkInputDialog(text="Enter timestamp (seconds or MM:SS):", title="Insert Frame")
        result = dialog.get_input()
        if result:
            try:
                timestamp = self._parse_time(result)
                # Find appropriate index
                new_idx = len(self.extract_planned_frames)
                for i, pf in enumerate(self.extract_planned_frames):
                    if pf.timestamp > timestamp:
                        new_idx = i
                        break

                # Insert new frame
                new_frame = PlannedFrame(
                    index=new_idx,
                    frame_number=int(timestamp * 30),  # Estimate, will be recalculated
                    timestamp=timestamp,
                    reason="manual",
                    included=True
                )
                self.extract_planned_frames.insert(new_idx, new_frame)

                # Re-index
                for i, pf in enumerate(self.extract_planned_frames):
                    pf.index = i

                self._refresh_plan_display()
                self.log(f"Inserted frame at {timestamp:.2f}s")
            except Exception as e:
                self.log(f"Error inserting frame: {e}")

    def _plan_delete_selected(self):
        """Delete unchecked frames from the plan."""
        self.extract_planned_frames = [pf for pf in self.extract_planned_frames if pf.included]
        # Re-index
        for i, pf in enumerate(self.extract_planned_frames):
            pf.index = i
        self._refresh_plan_display()
        self.log(f"Kept {len(self.extract_planned_frames)} frames")

    def _on_lut_preview_frame_change(self, value):
        """Handle LUT preview frame slider change."""
        self.lut_preview_frame_label.configure(text=f"{int(value)}")

    def _update_lut_preview(self):
        """Update LUT preview with current settings."""
        if not HAS_PIL or not HAS_PREP360:
            self.log("Preview requires PIL and prep360")
            return

        # Get first video from queue
        video_path = None
        for item in self.video_queue.items:
            video_path = item.video_path
            break

        if not video_path:
            self.log("Add a video to queue for preview")
            return

        lut_path = self.extract_lut_file_entry.get()
        if not lut_path or not Path(lut_path).exists():
            self.log("Select a valid LUT file")
            return

        threading.Thread(target=self._lut_preview_worker, args=(video_path, lut_path), daemon=True).start()

    def _lut_preview_worker(self, video_path, lut_path):
        """Worker to generate LUT preview images."""
        try:
            import cv2
            import numpy as np

            # Extract single frame from video
            cap = cv2.VideoCapture(video_path)
            frame_idx = self.lut_preview_frame_var.get()
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Map slider (0-100) to frame range
            actual_frame = int(frame_idx / 100 * total_frames) if total_frames > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.log("Could not read frame from video")
                return

            # Original image
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply LUT
            processor = LUTProcessor()
            strength = self.extract_lut_strength_var.get()
            shadow = self.extract_shadow_var.get()
            highlight = self.extract_highlight_var.get()

            processed = processor.apply_lut(frame, lut_path, strength)

            # Apply shadow/highlight if not neutral
            if shadow != 50 or highlight != 50:
                from prep360.core.adjustments import apply_shadow_highlight
                processed = apply_shadow_highlight(processed, shadow, highlight)

            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # Resize for display
            preview_size = (280, 140)
            original_resized = cv2.resize(original_rgb, preview_size)
            processed_resized = cv2.resize(processed_rgb, preview_size)

            # Convert to PIL
            original_pil = Image.fromarray(original_resized)
            processed_pil = Image.fromarray(processed_resized)

            # Update GUI
            def update_gui():
                ctk_before = ctk.CTkImage(light_image=original_pil, dark_image=original_pil,
                                          size=preview_size)
                ctk_after = ctk.CTkImage(light_image=processed_pil, dark_image=processed_pil,
                                         size=preview_size)
                self.lut_preview_before.configure(image=ctk_before, text="")
                self.lut_preview_after.configure(image=ctk_after, text="")
                self.lut_preview_before._ctk_image = ctk_before
                self.lut_preview_after._ctk_image = ctk_after

            self.after(0, update_gui)
            self.log(f"Preview updated (frame {actual_frame})")

        except Exception as e:
            self.log(f"Preview error: {e}")

    def _launch_segmentation_window(self):
        """Launch the segmentation window from Extract tab."""
        output_folder = self.extract_output_entry.get()
        SegmentationWindow(self, input_folder=output_folder)

    # =========================================================================
    # TAB: Reframe
    # =========================================================================
    def _build_reframe_tab(self):
        tab = self.tab_view.tab("Reframe")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=420)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Input / Output")

        in_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        in_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(in_frame, text="Input:", width=80, anchor="w").pack(side="left")
        self.reframe_input_entry = ctk.CTkEntry(in_frame, placeholder_text="Equirectangular frames...")
        self.reframe_input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(in_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.reframe_input_entry, self._update_reframe_preview)).pack(side="left")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.reframe_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Perspective views...")
        self.reframe_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.reframe_output_entry)).pack(side="left")

        self._section_header(left, "View Configuration")

        # Preset
        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=80, anchor="w").pack(side="left")
        self.reframe_preset_var = ctk.StringVar(value="prep360_default")
        preset_values = list(VIEW_PRESETS.keys()) if HAS_PREP360 else ["prep360_default"]
        ctk.CTkComboBox(preset_frame, variable=self.reframe_preset_var,
                        values=preset_values, state="readonly",
                        command=self._on_reframe_preset_change).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.reframe_preset_desc = ctk.CTkLabel(left, text="", text_color="gray")
        self.reframe_preset_desc.pack(pady=(0, 5), padx=20, anchor="w")

        # Views Summary (monospace tree display)
        self.reframe_views_summary = ctk.CTkTextbox(
            left, height=100, font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#1a1a1a", state="disabled"
        )
        self.reframe_views_summary.pack(fill="x", padx=10, pady=(0, 10))

        # Output size
        size_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        size_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(size_frame, text="Size:", width=80, anchor="w").pack(side="left")
        self.reframe_size_var = ctk.StringVar(value="1920")
        ctk.CTkComboBox(size_frame, variable=self.reframe_size_var,
                        values=["512", "1024", "1280", "1536", "1920", "2048", "2560", "3840"],
                        state="readonly").pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Quality
        qual_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        qual_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(qual_frame, text="Quality:", width=80, anchor="w").pack(side="left")
        self.reframe_quality_var = ctk.IntVar(value=95)
        self.reframe_quality_label = ctk.CTkLabel(qual_frame, text="95", width=40)
        self.reframe_quality_label.pack(side="right")
        ctk.CTkSlider(qual_frame, from_=70, to=100, variable=self.reframe_quality_var,
                      command=lambda v: self.reframe_quality_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Workers
        workers_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        workers_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(workers_frame, text="Workers:", width=80, anchor="w").pack(side="left")
        self.reframe_workers_var = ctk.IntVar(value=4)
        self.reframe_workers_label = ctk.CTkLabel(workers_frame, text="4", width=40)
        self.reframe_workers_label.pack(side="right")
        ctk.CTkSlider(workers_frame, from_=1, to=16, variable=self.reframe_workers_var,
                      command=lambda v: self.reframe_workers_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self._section_header(left, "Options")

        self.reframe_zenith_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Include zenith (top view)",
                        variable=self.reframe_zenith_var,
                        command=self._on_reframe_option_change).pack(pady=3, padx=20, anchor="w")

        self.reframe_nadir_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Include nadir (bottom view)",
                        variable=self.reframe_nadir_var,
                        command=self._on_reframe_option_change).pack(pady=3, padx=20, anchor="w")

        # Buttons
        self.reframe_run_btn = ctk.CTkButton(
            left, text="Reframe Images", command=self.run_reframe,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.reframe_run_btn.pack(pady=20, padx=10, fill="x")

        self.reframe_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # Right: Preview + Info
        ctk.CTkLabel(right, text="Equirectangular Preview",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5), padx=15, anchor="w")

        ctk.CTkLabel(right, text="View extraction regions shown as colored overlays",
                     text_color="gray").pack(pady=(0, 5), padx=15, anchor="w")

        # Preview container frame for responsive sizing
        self.reframe_preview_container = ctk.CTkFrame(right, fg_color="#1a1a1a", corner_radius=8)
        self.reframe_preview_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Preview label inside container
        self.reframe_preview = ctk.CTkLabel(self.reframe_preview_container,
                                             text="Select input folder to preview\n\nView boundaries will be drawn\non the equirectangular image",
                                             fg_color="transparent")
        self.reframe_preview.pack(expand=True)

        # Store the original image for redrawing
        self._reframe_preview_image = None
        self._reframe_preview_path = None
        self._preview_size = (560, 280)  # Default size

        # Bind resize event for responsive preview
        self.reframe_preview_container.bind("<Configure>", self._on_preview_resize)

        # Legend
        legend_frame = ctk.CTkFrame(master=right)
        legend_frame.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(legend_frame, text="Legend:", font=ctk.CTkFont(weight="bold")
                     ).pack(side="left", padx=(10, 15), pady=8)

        self.reframe_legend_labels = []
        # Will be populated when preset changes

        info_frame = ctk.CTkFrame(master=right)
        info_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.reframe_info = ctk.CTkLabel(info_frame, text="Views: --\nInput: --\nOutput: --",
                                          justify="left", anchor="w")
        self.reframe_info.pack(pady=10, padx=15, anchor="w")

        # Initialize preset info
        self._on_reframe_preset_change("prep360_default")

    def _on_reframe_preset_change(self, preset_name):
        if not HAS_PREP360 or preset_name not in VIEW_PRESETS:
            return

        config = VIEW_PRESETS[preset_name]
        desc = f"{config.total_views()} views: "
        ring_descs = []
        for ring in config.rings:
            ring_descs.append(f"{ring.count}@{ring.pitch:+.0f}deg")
        if config.include_zenith:
            ring_descs.append("zenith")
        if config.include_nadir:
            ring_descs.append("nadir")
        desc += ", ".join(ring_descs)

        self.reframe_preset_desc.configure(text=desc)
        self._update_views_summary(config)
        self._update_reframe_legend(config)
        self._update_reframe_info()
        self._draw_reframe_preview()

    def _update_views_summary(self, config):
        """Update the views summary with a monospace tree display."""
        lines = []
        view_num = 0

        for i, ring in enumerate(config.rings):
            yaw_step = 360 // ring.count if ring.count > 0 else 360
            end_view = view_num + ring.count - 1
            lines.append(
                f"Ring {i}: pitch={ring.pitch:+3.0f}°, {ring.count} views @ {yaw_step}°, "
                f"V{view_num:02d}-V{end_view:02d}"
            )
            view_num += ring.count

        if config.include_zenith:
            lines.append(f"Zenith: V{view_num:02d}, pitch=+90°, FOV={config.zenith_fov}°")
            view_num += 1

        if config.include_nadir:
            lines.append(f"Nadir:  V{view_num:02d}, pitch=-90°, FOV={config.zenith_fov}°")

        # Update textbox
        self.reframe_views_summary.configure(state="normal")
        self.reframe_views_summary.delete("1.0", "end")
        self.reframe_views_summary.insert("1.0", "\n".join(lines))
        self.reframe_views_summary.configure(state="disabled")

    def _update_reframe_legend(self, config):
        """Update the legend with view information."""
        # Clear existing legend labels
        for label in self.reframe_legend_labels:
            label.destroy()
        self.reframe_legend_labels = []

        # Find the legend frame (parent of first legend label would be it)
        legend_frame = None
        for widget in self.tab_view.tab("Reframe").winfo_children():
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkFrame):
                    for subchild in child.winfo_children():
                        if isinstance(subchild, ctk.CTkLabel) and subchild.cget("text") == "Legend:":
                            legend_frame = child
                            break

        if not legend_frame:
            return

        # Distinct colors per ring (matching the preview)
        ring_colors = [
            "#FF7F50",   # Ring 0: Coral
            "#4ECDC4",   # Ring 1: Teal
            "#FFE66D",   # Ring 2: Yellow
            "#95E1D3",   # Ring 3: Mint
            "#DDA0DD",   # Ring 4: Plum
        ]
        zenith_color = "#87CEFA"   # Light blue (light sky blue)
        nadir_color = "#DEB887"    # Tan

        # Count views per ring for legend
        view_num = 0
        for i, ring in enumerate(config.rings):
            color = ring_colors[i % len(ring_colors)]
            count = ring.count
            start_v = view_num
            end_v = view_num + count - 1
            text = f"Ring {i} ({ring.pitch:+.0f}°): V{start_v:02d}-V{end_v:02d}"
            label = ctk.CTkLabel(legend_frame, text=text, text_color=color,
                                 font=ctk.CTkFont(size=11))
            label.pack(side="left", padx=5, pady=8)
            self.reframe_legend_labels.append(label)
            view_num += count

        # Zenith
        if config.include_zenith:
            label = ctk.CTkLabel(legend_frame, text=f"Zenith: V{view_num:02d}", text_color=zenith_color,
                                 font=ctk.CTkFont(size=11))
            label.pack(side="left", padx=5, pady=8)
            self.reframe_legend_labels.append(label)
            view_num += 1

        # Nadir
        if config.include_nadir:
            label = ctk.CTkLabel(legend_frame, text=f"Nadir: V{view_num:02d}", text_color=nadir_color,
                                 font=ctk.CTkFont(size=11))
            label.pack(side="left", padx=5, pady=8)
            self.reframe_legend_labels.append(label)

    def _update_reframe_preview(self):
        """Load the first image from the input folder for preview."""
        if not HAS_PIL:
            return
        folder = self.reframe_input_entry.get()
        if not folder or not Path(folder).exists():
            self._reframe_preview_image = None
            self._reframe_preview_path = None
            return

        folder_path = Path(folder)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            files = list(folder_path.glob(ext))
            if files:
                try:
                    self._reframe_preview_path = files[0]
                    self._reframe_preview_image = Image.open(files[0])
                    self._update_reframe_info()
                    self._draw_reframe_preview()
                    return
                except Exception:
                    pass

        self._reframe_preview_image = None
        self._reframe_preview_path = None
        self.reframe_preview.configure(image=None, text="No images found")

    def _perspective_to_spherical(self, px, py, center_yaw, center_pitch, fov):
        """
        Convert normalized perspective coordinates to spherical.
        px, py in range [-1, 1] representing the perspective image plane.
        """
        import math
        # Focal length for the given FOV
        f = 1.0 / math.tan(math.radians(fov / 2))

        # Direction vector in camera space (looking down +Z)
        dx = px
        dy = py
        dz = f

        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/length, dy/length, dz/length

        # Rotate by pitch (around X axis) - positive pitch looks up
        pitch_rad = math.radians(center_pitch)
        dy2 = dy * math.cos(pitch_rad) + dz * math.sin(pitch_rad)
        dz2 = -dy * math.sin(pitch_rad) + dz * math.cos(pitch_rad)
        dy, dz = dy2, dz2

        # Rotate by yaw (around Y axis)
        yaw_rad = math.radians(center_yaw)
        dx2 = dx * math.cos(yaw_rad) + dz * math.sin(yaw_rad)
        dz2 = -dx * math.sin(yaw_rad) + dz * math.cos(yaw_rad)
        dx, dz = dx2, dz2

        # Convert to spherical
        yaw_out = math.degrees(math.atan2(dx, dz))
        pitch_out = math.degrees(math.asin(max(-1, min(1, dy))))

        return yaw_out, pitch_out

    def _spherical_to_equirect(self, yaw_deg, pitch_deg, img_width, img_height):
        """Convert spherical coordinates to equirectangular pixel coordinates."""
        # Normalize yaw to 0-360 range
        yaw = yaw_deg % 360
        # x: 0° at left edge, 360° at right edge
        x = (yaw / 360.0) * img_width
        # y: +90° (zenith) at top, -90° (nadir) at bottom
        y = ((90.0 - pitch_deg) / 180.0) * img_height
        return (int(x), int(y))

    def _get_view_boundary_points(self, center_yaw, center_pitch, fov, img_width, img_height, num_points=32):
        """
        Generate boundary points for a perspective view projected onto equirectangular.
        Uses proper projection math for curved boundaries.
        """
        points = []

        # Trace the four edges of the perspective rectangle
        # Edge coordinates in perspective view: corners at (±1, ±1) normalized
        edges = [
            # Top edge: y = 1, x from -1 to 1
            [(i/num_points * 2 - 1, 1) for i in range(num_points + 1)],
            # Right edge: x = 1, y from 1 to -1
            [(1, 1 - i/num_points * 2) for i in range(num_points + 1)],
            # Bottom edge: y = -1, x from 1 to -1
            [(1 - i/num_points * 2, -1) for i in range(num_points + 1)],
            # Left edge: x = -1, y from -1 to 1
            [(-1, -1 + i/num_points * 2) for i in range(num_points + 1)],
        ]

        for edge in edges:
            for px, py in edge:
                yaw, pitch = self._perspective_to_spherical(
                    px, py, center_yaw, center_pitch, fov
                )
                x, y = self._spherical_to_equirect(yaw, pitch, img_width, img_height)
                points.append((x, y))

        return points

    def _draw_reframe_preview(self):
        """Draw the equirectangular image with view boundary overlays."""
        if not HAS_PIL or not HAS_PREP360:
            return

        if self._reframe_preview_image is None:
            # No image loaded, show placeholder with view diagram
            self._draw_placeholder_preview()
            return

        try:
            import cv2
            import numpy as np

            # Get current preset config
            preset_name = self.reframe_preset_var.get()
            if preset_name not in VIEW_PRESETS:
                return

            config = VIEW_PRESETS[preset_name]

            # Get current display size (responsive)
            display_w, display_h = getattr(self, '_preview_size', (560, 280))

            # Convert PIL image to OpenCV format (BGR)
            img_pil = self._reframe_preview_image.copy()
            img_pil = img_pil.resize((display_w, display_h), Image.LANCZOS)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Distinct colors per ring (BGR for OpenCV)
            ring_colors = [
                (80, 127, 255),    # Ring 0: Coral (BGR)
                (196, 205, 78),    # Ring 1: Teal (BGR)
                (109, 230, 255),   # Ring 2: Yellow (BGR)
                (211, 225, 149),   # Ring 3: Mint (BGR)
                (221, 160, 221),   # Ring 4: Plum (BGR)
            ]
            zenith_color = (250, 206, 135)   # Light blue (BGR)
            nadir_color = (135, 184, 222)    # Tan (BGR)

            view_num = 0

            # Draw each ring's views with proper alpha blending
            for ring_idx, ring in enumerate(config.rings):
                color = ring_colors[ring_idx % len(ring_colors)]
                yaw_positions = ring.get_yaw_positions()
                for yaw in yaw_positions:
                    self._draw_view_overlay_cv(img_cv, yaw, ring.pitch, ring.fov,
                                               color, f"V{view_num:02d}", fill_alpha=0.2)
                    view_num += 1

            # Draw zenith if enabled
            if config.include_zenith and self.reframe_zenith_var.get():
                self._draw_view_overlay_cv(img_cv, 0, 90, config.zenith_fov,
                                           zenith_color, f"V{view_num:02d}", fill_alpha=0.2)
                view_num += 1

            # Draw nadir if enabled
            if config.include_nadir or self.reframe_nadir_var.get():
                self._draw_view_overlay_cv(img_cv, 0, -90, config.zenith_fov,
                                           nadir_color, f"V{view_num:02d}", fill_alpha=0.2)

            # Convert back to PIL for display
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_final = Image.fromarray(img_rgb)

            # Display using CTkImage for HighDPI support
            ctk_image = ctk.CTkImage(light_image=img_final, dark_image=img_final,
                                     size=(display_w, display_h))
            self.reframe_preview.configure(image=ctk_image, text="")
            self.reframe_preview._ctk_image = ctk_image  # Keep reference

        except Exception as e:
            # Fallback to simple preview
            self.reframe_preview.configure(text=f"Preview error: {e}")

    def _draw_view_overlay_cv(self, image, center_yaw, center_pitch, fov, color, label, fill_alpha=0.2):
        """Draw a semi-transparent view overlay using OpenCV addWeighted.

        Args:
            image: OpenCV BGR image (modified in place)
            center_yaw, center_pitch, fov: View parameters
            color: BGR color tuple
            label: Label string (e.g., "V00")
            fill_alpha: 0.0-1.0, where 0.2 = 20% opacity
        """
        import cv2
        import numpy as np

        img_h, img_w = image.shape[:2]

        # Get boundary points using proper projection
        points = self._get_view_boundary_points(center_yaw, center_pitch, fov, img_w, img_h, num_points=32)

        # Handle wrap-around at 0°/360° boundary - split into segments
        segments = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            prev_x = points[i-1][0]
            curr_x = points[i][0]

            # If jump is more than half the image width, it's a wrap
            if abs(curr_x - prev_x) > img_w / 2:
                segments.append(current_segment)
                current_segment = [points[i]]
            else:
                current_segment.append(points[i])

        segments.append(current_segment)

        # Draw each segment with proper alpha blending
        for segment in segments:
            if len(segment) > 2:
                pts = np.array(segment, dtype=np.int32)

                # Step 1: Create overlay copy
                overlay = image.copy()

                # Step 2: Draw filled polygon on overlay
                cv2.fillPoly(overlay, [pts], color)

                # Step 3: Blend overlay with original (creates transparency)
                cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)

                # Step 4: Draw solid outline on top (full opacity)
                cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)

        # Draw label at center with dark background
        cx, cy = self._spherical_to_equirect(center_yaw, center_pitch, img_w, img_h)

        # Label background
        label_w, label_h = 28, 16
        cv2.rectangle(image,
                      (cx - label_w//2, cy - label_h//2),
                      (cx + label_w//2, cy + label_h//2),
                      (20, 20, 20), -1)

        # Label text
        cv2.putText(image, label,
                    (cx - label_w//2 + 3, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _draw_placeholder_preview(self):
        """Draw a placeholder showing view positions without an actual image."""
        if not HAS_PIL or not HAS_PREP360:
            return

        try:
            import cv2
            import numpy as np

            preset_name = self.reframe_preset_var.get()
            if preset_name not in VIEW_PRESETS:
                return

            config = VIEW_PRESETS[preset_name]

            # Get current display size (responsive)
            display_w, display_h = getattr(self, '_preview_size', (560, 280))

            # Create OpenCV BGR image (dark gray background)
            img_cv = np.full((display_h, display_w, 3), (30, 30, 30), dtype=np.uint8)

            # Draw grid lines for reference
            grid_color = (60, 60, 60)
            for i in range(1, 4):
                y = i * display_h // 4
                cv2.line(img_cv, (0, y), (display_w, y), grid_color, 1)
            for i in range(1, 8):
                x = i * display_w // 8
                cv2.line(img_cv, (x, 0), (x, display_h), grid_color, 1)

            # Draw horizon line
            horizon_y = display_h // 2
            cv2.line(img_cv, (0, horizon_y), (display_w, horizon_y), (80, 80, 80), 2)

            # Distinct colors per ring (BGR for OpenCV) - coral, teal, yellow
            ring_colors_bgr = [
                (80, 127, 255),    # Ring 0: Coral (BGR)
                (196, 205, 78),    # Ring 1: Teal (BGR)
                (109, 230, 255),   # Ring 2: Yellow (BGR)
                (211, 225, 149),   # Ring 3: Mint (BGR)
                (221, 160, 221),   # Ring 4: Plum (BGR)
            ]
            zenith_color_bgr = (250, 206, 135)   # Light blue (BGR)
            nadir_color_bgr = (135, 184, 222)    # Tan (BGR)

            view_num = 0

            # Draw rings with curved boundaries using OpenCV addWeighted
            for ring_idx, ring in enumerate(config.rings):
                color = ring_colors_bgr[ring_idx % len(ring_colors_bgr)]
                yaw_positions = ring.get_yaw_positions()
                for yaw in yaw_positions:
                    self._draw_view_overlay_cv(img_cv, yaw, ring.pitch, ring.fov,
                                               color, f"V{view_num:02d}", fill_alpha=0.2)
                    view_num += 1

            # Zenith
            if config.include_zenith and self.reframe_zenith_var.get():
                self._draw_view_overlay_cv(img_cv, 0, 90, config.zenith_fov,
                                           zenith_color_bgr, f"V{view_num:02d}", fill_alpha=0.2)
                view_num += 1

            # Nadir
            if config.include_nadir or self.reframe_nadir_var.get():
                self._draw_view_overlay_cv(img_cv, 0, -90, config.zenith_fov,
                                           nadir_color_bgr, f"V{view_num:02d}", fill_alpha=0.2)

            # Add text using OpenCV
            cv2.putText(img_cv, "Load images to preview", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(img_cv, f"Preset: {preset_name}", (10, display_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Use CTkImage for HighDPI support
            ctk_image = ctk.CTkImage(light_image=img_pil, dark_image=img_pil,
                                     size=(display_w, display_h))
            self.reframe_preview.configure(image=ctk_image, text="")
            self.reframe_preview._ctk_image = ctk_image  # Keep reference

        except Exception:
            pass

    def _on_reframe_option_change(self):
        """Called when zenith/nadir options change."""
        self._update_reframe_info()
        self._draw_reframe_preview()

    def _on_preview_resize(self, event):
        """Handle preview container resize - fit image entirely within container."""
        # Get container size
        container_w = event.width
        container_h = event.height

        if container_w < 100 or container_h < 50:
            return  # Too small, skip

        # For equirectangular images, we want 2:1 aspect ratio
        img_aspect = 2.0

        # Calculate scale to FIT ENTIRELY within container (not fill/crop)
        scale_w = container_w / (img_aspect * 100)  # Normalize
        scale_h = container_h / 100
        scale = min(scale_w, scale_h)  # Use MINIMUM to ensure full image fits

        # Apply 95% to add padding and prevent edge touching
        scale *= 0.95

        # Calculate display size
        display_w = int(img_aspect * 100 * scale)
        display_h = int(100 * scale)

        # Minimum size
        display_w = max(200, display_w)
        display_h = max(100, display_h)

        # Only redraw if size changed significantly
        if hasattr(self, '_preview_size'):
            old_w, old_h = self._preview_size
            if abs(display_w - old_w) < 10 and abs(display_h - old_h) < 10:
                return  # No significant change

        self._preview_size = (display_w, display_h)
        self._draw_reframe_preview()

    def _update_reframe_info(self):
        if not HAS_PREP360:
            return

        preset_name = self.reframe_preset_var.get()
        if preset_name not in VIEW_PRESETS:
            return

        config = VIEW_PRESETS[preset_name]
        num_views = config.total_views()

        # Adjust for user options
        if not self.reframe_zenith_var.get() and config.include_zenith:
            num_views -= 1
        if self.reframe_nadir_var.get() and not config.include_nadir:
            num_views += 1

        input_folder = self.reframe_input_entry.get()
        input_count = 0
        if input_folder and Path(input_folder).exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                input_count += len(list(Path(input_folder).glob(ext)))

        output_count = input_count * num_views if input_count > 0 else "?"

        self.reframe_info.configure(
            text=f"Views per frame: {num_views}\n"
                 f"Input images: {input_count if input_count > 0 else '?'}\n"
                 f"Output images: {output_count}"
        )

    def run_reframe(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        input_dir = self.reframe_input_entry.get()
        output_dir = self.reframe_output_entry.get()

        if not input_dir:
            self.log("Error: Please select input folder")
            return
        if not Path(input_dir).exists():
            self.log(f"Error: Input folder not found: {input_dir}")
            return
        if not output_dir:
            self.log("Error: Please select output folder")
            return

        self._start_operation(self.reframe_run_btn, self.reframe_stop_btn)
        threading.Thread(target=self._reframe_worker, args=(input_dir, output_dir), daemon=True).start()

    def _reframe_worker(self, input_dir, output_dir):
        try:
            preset_name = self.reframe_preset_var.get()
            config = VIEW_PRESETS.get(preset_name, VIEW_PRESETS["prep360_default"])

            # Apply user settings
            config.output_size = int(self.reframe_size_var.get())
            config.jpeg_quality = self.reframe_quality_var.get()
            config.include_zenith = self.reframe_zenith_var.get()
            config.include_nadir = self.reframe_nadir_var.get()

            reframer = Reframer(config)

            self.log(f"Reframing: {input_dir}")
            self.log(f"Preset: {preset_name}, Size: {config.output_size}, Views: {config.total_views()}")

            def progress(curr, total, name):
                if not self.cancel_flag.is_set():
                    self.log(f"[{curr}/{total}] {name}")

            result = reframer.reframe_batch(
                input_dir, output_dir,
                num_workers=self.reframe_workers_var.get(),
                progress_callback=progress
            )

            if result.success:
                self.log(f"Reframed {result.input_count} images -> {result.output_count} views")
            else:
                self.log(f"Completed with {len(result.errors)} errors")
                for err in result.errors[:5]:
                    self.log(f"  {err}")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(self.reframe_run_btn, self.reframe_stop_btn))

    # =========================================================================
    # TAB: LUT
    # =========================================================================
    def _build_lut_tab(self):
        tab = self.tab_view.tab("LUT")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Input / Output")

        in_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        in_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(in_frame, text="Input:", width=80, anchor="w").pack(side="left")
        self.lut_input_entry = ctk.CTkEntry(in_frame, placeholder_text="Image or folder...")
        self.lut_input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(in_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.lut_input_entry)).pack(side="left")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.lut_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Output folder...")
        self.lut_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.lut_output_entry)).pack(side="left")

        self._section_header(left, "LUT Settings")

        lut_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        lut_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(lut_frame, text="LUT File:", width=80, anchor="w").pack(side="left")
        self.lut_file_entry = ctk.CTkEntry(lut_frame, placeholder_text=".cube LUT file...")
        self.lut_file_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(lut_frame, text="...", width=40,
                      command=self._browse_lut_file).pack(side="left")

        # Strength
        str_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        str_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(str_frame, text="Strength:", width=80, anchor="w").pack(side="left")
        self.lut_strength_var = ctk.DoubleVar(value=1.0)
        self.lut_strength_label = ctk.CTkLabel(str_frame, text="100%", width=50)
        self.lut_strength_label.pack(side="right")
        ctk.CTkSlider(str_frame, from_=0, to=1, variable=self.lut_strength_var,
                      command=lambda v: self.lut_strength_label.configure(text=f"{int(float(v)*100)}%")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self._section_header(left, "Shadow / Highlight")

        # Shadow
        shadow_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        shadow_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(shadow_frame, text="Shadows:", width=80, anchor="w").pack(side="left")
        self.lut_shadow_var = ctk.IntVar(value=50)
        self.lut_shadow_label = ctk.CTkLabel(shadow_frame, text="50", width=40)
        self.lut_shadow_label.pack(side="right")
        ctk.CTkSlider(shadow_frame, from_=0, to=100, variable=self.lut_shadow_var,
                      command=lambda v: self.lut_shadow_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Highlight
        highlight_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        highlight_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(highlight_frame, text="Highlights:", width=80, anchor="w").pack(side="left")
        self.lut_highlight_var = ctk.IntVar(value=50)
        self.lut_highlight_label = ctk.CTkLabel(highlight_frame, text="50", width=40)
        self.lut_highlight_label.pack(side="right")
        ctk.CTkSlider(highlight_frame, from_=0, to=100, variable=self.lut_highlight_var,
                      command=lambda v: self.lut_highlight_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        ctk.CTkLabel(left, text="50 = neutral, applied after LUT",
                     text_color="gray", font=ctk.CTkFont(size=10)).pack(pady=(0, 5), padx=20, anchor="w")

        # Buttons
        self.lut_run_btn = ctk.CTkButton(
            left, text="Apply LUT", command=self.run_lut,
            fg_color="#7B1FA2", hover_color="#6A1B9A",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.lut_run_btn.pack(pady=20, padx=10, fill="x")

        self.lut_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # Right: Info
        ctk.CTkLabel(right, text="LUT Processing",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """LUT color correction converts log footage
to standard color space (Rec.709).

Common LUTs:
- DJI D-Log M, Insta360 I-Log
- GoPro Protune Flat

Shadow/Highlight Adjustments:
- Applied AFTER the LUT
- Shadows: Lift (>50) / crush (<50) darks
- Highlights: Compress (>50) / boost (<50)
- 50 = neutral (no change)

Workflow:
1. Apply LUT to convert log to Rec.709
2. Adjust shadows to recover dark detail
3. Compress highlights if overexposed"""

        self.lut_info = ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw")
        self.lut_info.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _browse_lut_file(self):
        path = filedialog.askopenfilename(
            title="Select LUT File",
            filetypes=[("CUBE Files", "*.cube"), ("All Files", "*.*")]
        )
        if path:
            self.lut_file_entry.delete(0, 'end')
            self.lut_file_entry.insert(0, path)

    def run_lut(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        input_path = self.lut_input_entry.get()
        output_path = self.lut_output_entry.get()
        lut_path = self.lut_file_entry.get()

        if not input_path:
            self.log("Error: Please select input")
            return
        if not Path(input_path).exists():
            self.log(f"Error: Input not found: {input_path}")
            return
        if not output_path:
            self.log("Error: Please select output folder")
            return
        if not lut_path or not Path(lut_path).exists():
            self.log("Error: Please select a valid LUT file")
            return

        self._start_operation(self.lut_run_btn, self.lut_stop_btn)
        threading.Thread(target=self._lut_worker, args=(input_path, output_path, lut_path), daemon=True).start()

    def _lut_worker(self, input_path, output_path, lut_path):
        try:
            processor = LUTProcessor()
            strength = self.lut_strength_var.get()
            shadow = self.lut_shadow_var.get()
            highlight = self.lut_highlight_var.get()

            self.log(f"Applying LUT: {Path(lut_path).name}")
            self.log(f"Strength: {int(strength * 100)}%")
            if shadow != 50 or highlight != 50:
                self.log(f"Shadow/Highlight: {shadow}/{highlight}")

            input_p = Path(input_path)
            output_p = Path(output_path)
            output_p.mkdir(parents=True, exist_ok=True)

            if input_p.is_dir():
                def progress(curr, total, name):
                    if not self.cancel_flag.is_set():
                        self.log(f"[{curr}/{total}] {name}")

                success, errors = processor.process_batch(
                    input_path, output_path, lut_path, strength, progress
                )
                self.log(f"LUT applied to {success} images, {errors} errors")

                # Apply shadow/highlight adjustments if not neutral
                if (shadow != 50 or highlight != 50) and success > 0:
                    self.log(f"\nApplying shadow/highlight adjustments...")
                    from prep360.core.adjustments import apply_shadow_highlight
                    import cv2

                    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    images = []
                    for ext in extensions:
                        images.extend(output_p.glob(ext))
                    images = sorted(images)

                    for i, img_path in enumerate(images):
                        if self.cancel_flag.is_set():
                            self.log("Adjustment cancelled")
                            break

                        self.log(f"[{i+1}/{len(images)}] Adjusting {img_path.name}")

                        img = cv2.imread(str(img_path))
                        if img is not None:
                            adjusted = apply_shadow_highlight(img, shadow, highlight)
                            ext = img_path.suffix.lower()
                            if ext in ('.jpg', '.jpeg'):
                                cv2.imwrite(str(img_path), adjusted,
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                            else:
                                cv2.imwrite(str(img_path), adjusted)

                    self.log("Shadow/highlight adjustments complete")
            else:
                out_file = output_p / input_p.name
                if processor.process_image(input_path, str(out_file), lut_path, strength):
                    # Apply adjustments to single file
                    if shadow != 50 or highlight != 50:
                        from prep360.core.adjustments import apply_shadow_highlight
                        import cv2
                        img = cv2.imread(str(out_file))
                        if img is not None:
                            adjusted = apply_shadow_highlight(img, shadow, highlight)
                            ext = out_file.suffix.lower()
                            if ext in ('.jpg', '.jpeg'):
                                cv2.imwrite(str(out_file), adjusted,
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                            else:
                                cv2.imwrite(str(out_file), adjusted)
                    self.log(f"Saved: {out_file}")
                else:
                    self.log("Error processing image")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(self.lut_run_btn, self.lut_stop_btn))

    # =========================================================================
    # TAB: Sky Filter
    # =========================================================================
    def _build_sky_tab(self):
        tab = self.tab_view.tab("Sky Filter")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Input / Output")

        in_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        in_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(in_frame, text="Input:", width=80, anchor="w").pack(side="left")
        self.sky_input_entry = ctk.CTkEntry(in_frame, placeholder_text="Images folder...")
        self.sky_input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(in_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.sky_input_entry)).pack(side="left")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.sky_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Filtered output...")
        self.sky_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.sky_output_entry)).pack(side="left")

        self._section_header(left, "Detection Thresholds")

        # Brightness
        br_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        br_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(br_frame, text="Brightness:", width=80, anchor="w").pack(side="left")
        self.sky_brightness_var = ctk.DoubleVar(value=0.85)
        self.sky_brightness_label = ctk.CTkLabel(br_frame, text="0.85", width=50)
        self.sky_brightness_label.pack(side="right")
        ctk.CTkSlider(br_frame, from_=0.5, to=1.0, variable=self.sky_brightness_var,
                      command=lambda v: self.sky_brightness_label.configure(text=f"{float(v):.2f}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Saturation
        sat_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        sat_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(sat_frame, text="Saturation:", width=80, anchor="w").pack(side="left")
        self.sky_saturation_var = ctk.DoubleVar(value=0.15)
        self.sky_saturation_label = ctk.CTkLabel(sat_frame, text="0.15", width=50)
        self.sky_saturation_label.pack(side="right")
        ctk.CTkSlider(sat_frame, from_=0.0, to=0.5, variable=self.sky_saturation_var,
                      command=lambda v: self.sky_saturation_label.configure(text=f"{float(v):.2f}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Keypoints
        kp_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        kp_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(kp_frame, text="Keypoints:", width=80, anchor="w").pack(side="left")
        self.sky_keypoints_var = ctk.IntVar(value=50)
        self.sky_keypoints_label = ctk.CTkLabel(kp_frame, text="50", width=50)
        self.sky_keypoints_label.pack(side="right")
        ctk.CTkSlider(kp_frame, from_=10, to=200, variable=self.sky_keypoints_var,
                      command=lambda v: self.sky_keypoints_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self._section_header(left, "Options")

        self.sky_analyze_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Analyze only (don't copy files)",
                        variable=self.sky_analyze_var).pack(pady=3, padx=20, anchor="w")

        self.sky_dry_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Dry run",
                        variable=self.sky_dry_var).pack(pady=3, padx=20, anchor="w")

        # Buttons
        self.sky_run_btn = ctk.CTkButton(
            left, text="Filter Sky Images", command=self.run_sky_filter,
            fg_color="#0097A7", hover_color="#00838F",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.sky_run_btn.pack(pady=20, padx=10, fill="x")

        self.sky_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # Right: Info
        ctk.CTkLabel(right, text="Sky Detection",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """Removes sky-only images that are useless
for photogrammetry reconstruction.

Detection criteria (3 of 4 must match):
- High brightness (default > 0.85)
- Low saturation (default < 0.15)
- Few keypoints (default < 50)
- Low edge density

Tips:
- Lower brightness threshold = stricter
- Higher keypoint threshold = stricter
- Run "Analyze only" first to check results
- Adjust thresholds based on your footage

This is especially useful for zenith views
from 360 cameras that often show empty sky."""

        self.sky_info = ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw")
        self.sky_info.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def run_sky_filter(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        input_dir = self.sky_input_entry.get()
        output_dir = self.sky_output_entry.get()

        if not input_dir:
            self.log("Error: Please select input folder")
            return
        if not Path(input_dir).exists():
            self.log(f"Error: Input not found: {input_dir}")
            return
        if not self.sky_analyze_var.get() and not output_dir:
            self.log("Error: Please select output folder (or enable analyze-only)")
            return

        self._start_operation(self.sky_run_btn, self.sky_stop_btn)
        threading.Thread(target=self._sky_worker, args=(input_dir, output_dir), daemon=True).start()

    def _sky_worker(self, input_dir, output_dir):
        try:
            config = SkyFilterConfig(
                brightness_threshold=self.sky_brightness_var.get(),
                saturation_threshold=self.sky_saturation_var.get(),
                keypoint_threshold=self.sky_keypoints_var.get(),
            )

            sky_filter = SkyFilter(config)

            def progress(curr, total, name):
                if not self.cancel_flag.is_set():
                    self.log(f"[{curr}/{total}] {name}")

            self.log(f"Sky filtering: {input_dir}")

            if self.sky_analyze_var.get():
                results = sky_filter.analyze_batch(input_dir, progress)
                sky_count = sum(1 for m in results.values() if m.is_sky)
                self.log(f"Analyzed {len(results)} images: {sky_count} sky, {len(results) - sky_count} kept")
            else:
                kept, sky = sky_filter.filter_images(
                    input_dir, output_dir, progress,
                    dry_run=self.sky_dry_var.get()
                )
                self.log(f"Kept: {len(kept)}, Sky: {len(sky)}")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(self.sky_run_btn, self.sky_stop_btn))

    # =========================================================================
    # TAB: Segment
    # =========================================================================
    def _build_segment_tab(self):
        tab = self.tab_view.tab("Segment")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=420)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Input / Output")

        in_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        in_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(in_frame, text="Input:", width=80, anchor="w").pack(side="left")
        self.seg_input_entry = ctk.CTkEntry(in_frame, placeholder_text="Images folder...")
        self.seg_input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(in_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.seg_input_entry)).pack(side="left")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.seg_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Masks output folder...")
        self.seg_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.seg_output_entry)).pack(side="left")

        self._section_header(left, "Detection Classes")

        # Class preset
        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=80, anchor="w").pack(side="left")
        self.seg_preset_var = ctk.StringVar(value="photographer")
        preset_values = list(CLASS_PRESETS.keys()) if HAS_PREP360 else ["person"]
        ctk.CTkComboBox(preset_frame, variable=self.seg_preset_var,
                        values=preset_values, state="readonly",
                        command=self._on_seg_preset_change).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.seg_preset_desc = ctk.CTkLabel(left, text="", text_color="gray")
        self.seg_preset_desc.pack(pady=(0, 5), padx=20, anchor="w")

        # Custom classes
        custom_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        custom_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(custom_frame, text="Custom:", width=80, anchor="w").pack(side="left")
        self.seg_custom_entry = ctk.CTkEntry(custom_frame, placeholder_text="person, backpack, car...")
        self.seg_custom_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))

        ctk.CTkLabel(left, text="Leave empty to use preset", text_color="gray"
                     ).pack(pady=(0, 5), padx=20, anchor="w")

        self._section_header(left, "Model Settings")

        # Model
        model_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        model_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(model_frame, text="Model:", width=80, anchor="w").pack(side="left")
        self.seg_model_var = ctk.StringVar(value="yolo11n-seg.pt")
        ctk.CTkComboBox(model_frame, variable=self.seg_model_var,
                        values=["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
                                "yolo11l-seg.pt", "yolo11x-seg.pt"],
                        state="readonly").pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Confidence
        conf_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        conf_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(conf_frame, text="Confidence:", width=80, anchor="w").pack(side="left")
        self.seg_conf_var = ctk.DoubleVar(value=0.5)
        self.seg_conf_label = ctk.CTkLabel(conf_frame, text="0.50", width=50)
        self.seg_conf_label.pack(side="right")
        ctk.CTkSlider(conf_frame, from_=0.1, to=0.9, variable=self.seg_conf_var,
                      command=lambda v: self.seg_conf_label.configure(text=f"{float(v):.2f}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Dilate
        dilate_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        dilate_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(dilate_frame, text="Dilate:", width=80, anchor="w").pack(side="left")
        self.seg_dilate_var = ctk.IntVar(value=0)
        self.seg_dilate_label = ctk.CTkLabel(dilate_frame, text="0 px", width=50)
        self.seg_dilate_label.pack(side="right")
        ctk.CTkSlider(dilate_frame, from_=0, to=50, variable=self.seg_dilate_var,
                      command=lambda v: self.seg_dilate_label.configure(text=f"{int(v)} px")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self._section_header(left, "Options")

        self.seg_invert_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Invert mask (white background)",
                        variable=self.seg_invert_var).pack(pady=3, padx=20, anchor="w")

        # Buttons
        self.seg_run_btn = ctk.CTkButton(
            left, text="Generate Masks", command=self.run_segment,
            fg_color="#E65100", hover_color="#BF360C",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.seg_run_btn.pack(pady=20, padx=10, fill="x")

        self.seg_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # Right: Info
        ctk.CTkLabel(right, text="YOLO Segmentation",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """AI-powered instance segmentation using YOLO.
Generates masks for photogrammetry preprocessing.

Class Presets:
- person: Just people
- photographer: Person + bags, umbrella, suitcase
- equipment: Person + accessories + phone
- vehicles: Cars, bikes, trucks, buses
- animals: Common animals
- all_dynamic: People + vehicles + animals

Model Sizes (speed vs accuracy):
- yolo11n: Fastest, smallest
- yolo11s: Small, fast
- yolo11m: Medium, balanced
- yolo11l: Large, accurate
- yolo11x: Extra large, most accurate

Tips:
- Lower confidence = more detections
- Dilate expands masks to cover edges
- Invert for COLMAP-style masks

Requires: pip install ultralytics"""

        self.seg_info = ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw")
        self.seg_info.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Initialize preset description
        self._on_seg_preset_change("photographer")

    def _on_seg_preset_change(self, preset_name):
        if not HAS_PREP360 or preset_name not in CLASS_PRESETS:
            return
        class_ids = CLASS_PRESETS[preset_name]
        class_names = [COCO_CLASSES.get(cid, f"class_{cid}") for cid in class_ids]
        self.seg_preset_desc.configure(text=", ".join(class_names))

    def run_segment(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return
        if not HAS_YOLO:
            self.log("Error: ultralytics not installed")
            self.log("Install with: pip install ultralytics")
            return

        input_dir = self.seg_input_entry.get()
        output_dir = self.seg_output_entry.get()

        if not input_dir:
            self.log("Error: Please select input folder")
            return
        if not Path(input_dir).exists():
            self.log(f"Error: Input not found: {input_dir}")
            return
        if not output_dir:
            self.log("Error: Please select output folder")
            return

        self._start_operation(self.seg_run_btn, self.seg_stop_btn)
        threading.Thread(target=self._segment_worker, args=(input_dir, output_dir), daemon=True).start()

    def _segment_worker(self, input_dir, output_dir):
        try:
            # Parse classes
            custom_classes = self.seg_custom_entry.get().strip()
            if custom_classes:
                class_ids = []
                for name in custom_classes.split(","):
                    name = name.strip()
                    try:
                        class_ids.append(int(name))
                    except ValueError:
                        cid = get_class_id(name)
                        if cid is not None:
                            class_ids.append(cid)
                        else:
                            self.log(f"Warning: Unknown class '{name}'")
            else:
                preset_name = self.seg_preset_var.get()
                class_ids = CLASS_PRESETS.get(preset_name, [0])

            if not class_ids:
                class_ids = [0]

            config = SegmentConfig(
                model_name=self.seg_model_var.get(),
                classes=class_ids,
                confidence=self.seg_conf_var.get(),
                dilate_pixels=self.seg_dilate_var.get(),
                invert_mask=self.seg_invert_var.get(),
            )

            segmenter = Segmenter(config)
            class_names = segmenter.get_class_names(class_ids)

            self.log(f"Segmenting: {input_dir}")
            self.log(f"Model: {config.model_name}")
            self.log(f"Classes: {', '.join(class_names)}")

            def progress(curr, total, name):
                if not self.cancel_flag.is_set():
                    self.log(f"[{curr}/{total}] {name}")

            result = segmenter.segment_batch(input_dir, output_dir, progress)

            self.log(f"\nProcessed {result.total_images} images")
            self.log(f"Images with detections: {result.images_with_detections}")
            self.log(f"Total detections: {result.total_detections}")
            if result.errors:
                self.log(f"Errors: {len(result.errors)}")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(self.seg_run_btn, self.seg_stop_btn))

    # =========================================================================
    # TAB: Pipeline
    # =========================================================================
    def _build_pipeline_tab(self):
        tab = self.tab_view.tab("Pipeline")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=420)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # Left: Controls
        self._section_header(left, "Input / Output")

        vid_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        vid_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(vid_frame, text="Video:", width=80, anchor="w").pack(side="left")
        self.pipe_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="360 video file...")
        self.pipe_video_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(vid_frame, text="...", width=40,
                      command=lambda: self._browse_video_for(self.pipe_video_entry)).pack(side="left")

        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.pipe_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Output base folder...")
        self.pipe_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.pipe_output_entry)).pack(side="left")

        self._section_header(left, "Pipeline Steps")

        self.pipe_extract_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="1. Extract frames from video",
                        variable=self.pipe_extract_var).pack(pady=3, padx=20, anchor="w")

        self.pipe_reframe_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="2. Reframe to perspectives",
                        variable=self.pipe_reframe_var).pack(pady=3, padx=20, anchor="w")

        self.pipe_sky_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="3. Filter sky images",
                        variable=self.pipe_sky_var).pack(pady=3, padx=20, anchor="w")

        self.pipe_lut_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="4. Apply LUT correction",
                        variable=self.pipe_lut_var).pack(pady=3, padx=20, anchor="w")

        self._section_header(left, "Settings")

        # Interval
        int_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        int_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(int_frame, text="Interval:", width=80, anchor="w").pack(side="left")
        self.pipe_interval_var = ctk.DoubleVar(value=2.0)
        self.pipe_interval_label = ctk.CTkLabel(int_frame, text="2.0s", width=50)
        self.pipe_interval_label.pack(side="right")
        ctk.CTkSlider(int_frame, from_=0.5, to=10, variable=self.pipe_interval_var,
                      command=lambda v: self.pipe_interval_label.configure(text=f"{float(v):.1f}s")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Preset
        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=80, anchor="w").pack(side="left")
        self.pipe_preset_var = ctk.StringVar(value="prep360_default")
        preset_values = list(VIEW_PRESETS.keys()) if HAS_PREP360 else ["prep360_default"]
        ctk.CTkComboBox(preset_frame, variable=self.pipe_preset_var,
                        values=preset_values, state="readonly"
                        ).pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Size
        size_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        size_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(size_frame, text="Size:", width=80, anchor="w").pack(side="left")
        self.pipe_size_var = ctk.StringVar(value="1920")
        ctk.CTkComboBox(size_frame, variable=self.pipe_size_var,
                        values=["1024", "1536", "1920", "2560"],
                        state="readonly").pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Workers
        workers_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        workers_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(workers_frame, text="Workers:", width=80, anchor="w").pack(side="left")
        self.pipe_workers_var = ctk.IntVar(value=4)
        self.pipe_workers_label = ctk.CTkLabel(workers_frame, text="4", width=50)
        self.pipe_workers_label.pack(side="right")
        ctk.CTkSlider(workers_frame, from_=1, to=16, variable=self.pipe_workers_var,
                      command=lambda v: self.pipe_workers_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Buttons
        self.pipe_run_btn = ctk.CTkButton(
            left, text="Run Pipeline", command=self.run_pipeline,
            fg_color="#FF6F00", hover_color="#E65100",
            font=ctk.CTkFont(size=14, weight="bold"), height=50
        )
        self.pipe_run_btn.pack(pady=20, padx=10, fill="x")

        self.pipe_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # Right: Info
        ctk.CTkLabel(right, text="Full Pipeline",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """Run the complete processing pipeline:

1. Extract Frames
   Video -> equirectangular frames at interval

2. Reframe to Perspectives
   Equirect -> multiple pinhole views

3. Filter Sky Images (optional)
   Remove useless sky-only frames

4. Apply LUT (optional)
   Color correct log footage

Output structure:
  output/
    frames/         <- equirectangular frames
    perspectives/   <- reframed views
    filtered/       <- sky-filtered (if enabled)

This automates the entire workflow for
preparing 360 video for photogrammetry."""

        self.pipe_info = ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw")
        self.pipe_info.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def run_pipeline(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        video = self.pipe_video_entry.get()
        output = self.pipe_output_entry.get()

        if not video:
            self.log("Error: Please select a video file")
            return
        if not Path(video).exists():
            self.log(f"Error: Video not found: {video}")
            return
        if not output:
            self.log("Error: Please select output folder")
            return

        self._start_operation(self.pipe_run_btn, self.pipe_stop_btn)
        threading.Thread(target=self._pipeline_worker, args=(video, output), daemon=True).start()

    def _pipeline_worker(self, video, output):
        try:
            output_base = Path(output)
            output_base.mkdir(parents=True, exist_ok=True)

            # Step 1: Extract
            if self.pipe_extract_var.get():
                self.log("\n=== Step 1: Extract Frames ===")
                frames_dir = output_base / "frames"

                config = ExtractionConfig(
                    interval=self.pipe_interval_var.get(),
                    mode=ExtractionMode.FIXED,
                    quality=95,
                )

                extractor = FrameExtractor()

                def progress(curr, total, msg):
                    if not self.cancel_flag.is_set():
                        self.log(f"  [{curr}/{total}] {msg}")

                result = extractor.extract(video, str(frames_dir), config, progress)
                if not result.success:
                    self.log(f"Extraction failed: {result.error}")
                    return
                self.log(f"Extracted {result.frame_count} frames")
            else:
                frames_dir = output_base / "frames"
                self.log("Skipping extraction (using existing frames)")

            if self.cancel_flag.is_set():
                return

            # Step 2: Reframe
            if self.pipe_reframe_var.get():
                self.log("\n=== Step 2: Reframe ===")
                perspectives_dir = output_base / "perspectives"

                preset_name = self.pipe_preset_var.get()
                view_config = VIEW_PRESETS.get(preset_name, VIEW_PRESETS["prep360_default"])
                view_config.output_size = int(self.pipe_size_var.get())

                reframer = Reframer(view_config)

                def progress(curr, total, name):
                    if not self.cancel_flag.is_set():
                        self.log(f"  [{curr}/{total}] {name}")

                result = reframer.reframe_batch(
                    str(frames_dir), str(perspectives_dir),
                    num_workers=self.pipe_workers_var.get(),
                    progress_callback=progress
                )
                self.log(f"Created {result.output_count} perspective views")

            if self.cancel_flag.is_set():
                return

            # Step 3: Sky filter
            if self.pipe_sky_var.get():
                self.log("\n=== Step 3: Sky Filter ===")
                source_dir = output_base / "perspectives"
                filtered_dir = output_base / "filtered"

                sky_filter = SkyFilter()

                def progress(curr, total, name):
                    if not self.cancel_flag.is_set():
                        self.log(f"  [{curr}/{total}] {name}")

                kept, sky = sky_filter.filter_images(str(source_dir), str(filtered_dir), progress)
                self.log(f"Kept: {len(kept)}, Sky: {len(sky)}")

            self.log("\n=== Pipeline Complete ===")
            self.log(f"Output: {output_base}")

        except Exception as e:
            self.log(f"Pipeline error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(self.pipe_run_btn, self.pipe_stop_btn))

    # =========================================================================
    # TAB: Fisheye  (OSV dual-fisheye → perspective crops)
    # =========================================================================
    def _build_fisheye_tab(self):
        tab = self.tab_view.tab("Fisheye")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=420)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # --- Left: Controls ---
        self._section_header(left, "Input / Output")

        # OSV file picker
        osv_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        osv_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(osv_frame, text="OSV File:", width=80, anchor="w").pack(side="left")
        self.fisheye_osv_entry = ctk.CTkEntry(osv_frame, placeholder_text="DJI Osmo 360 .osv / .360 file...")
        self.fisheye_osv_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(osv_frame, text="...", width=40,
                      command=self._browse_osv_file).pack(side="left")

        # Output folder
        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.fisheye_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Perspective crops output...")
        self.fisheye_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.fisheye_output_entry)).pack(side="left")

        # Calibration
        self._section_header(left, "Calibration")

        self.fisheye_calib_default_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Use Default (DJI Osmo 360)",
                        variable=self.fisheye_calib_default_var,
                        command=self._on_fisheye_calib_toggle).pack(pady=3, padx=20, anchor="w")

        calib_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        calib_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(calib_frame, text="Calib:", width=80, anchor="w").pack(side="left")
        self.fisheye_calib_entry = ctk.CTkEntry(calib_frame, placeholder_text="calibration.json", state="disabled")
        self.fisheye_calib_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        self.fisheye_calib_browse_btn = ctk.CTkButton(
            calib_frame, text="...", width=40, state="disabled",
            command=self._browse_calib_file)
        self.fisheye_calib_browse_btn.pack(side="left")

        # Preset
        self._section_header(left, "View Configuration")

        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=80, anchor="w").pack(side="left")
        preset_keys = list(FISHEYE_PRESETS.keys()) if HAS_PREP360 else ["osv-full-f90-dual-26"]
        self.fisheye_preset_var = ctk.StringVar(value=preset_keys[0])
        ctk.CTkComboBox(preset_frame, variable=self.fisheye_preset_var,
                        values=preset_keys, state="readonly",
                        command=self._on_fisheye_preset_change
                        ).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.fisheye_preset_desc = ctk.CTkLabel(left, text="", text_color="gray")
        self.fisheye_preset_desc.pack(pady=(0, 5), padx=20, anchor="w")

        # Crop size
        crop_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        crop_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(crop_frame, text="Crop Size:", width=80, anchor="w").pack(side="left")
        self.fisheye_crop_var = ctk.StringVar(value="1600")
        ctk.CTkComboBox(crop_frame, variable=self.fisheye_crop_var,
                        values=["1280", "1600", "1920"],
                        state="readonly").pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Quality slider
        qual_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        qual_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(qual_frame, text="Quality:", width=80, anchor="w").pack(side="left")
        self.fisheye_quality_var = ctk.IntVar(value=95)
        self.fisheye_quality_label = ctk.CTkLabel(qual_frame, text="95", width=40)
        self.fisheye_quality_label.pack(side="right")
        ctk.CTkSlider(qual_frame, from_=70, to=100, variable=self.fisheye_quality_var,
                      command=lambda v: self.fisheye_quality_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Frame interval
        interval_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        interval_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(interval_frame, text="Interval:", width=80, anchor="w").pack(side="left")
        self.fisheye_interval_var = ctk.DoubleVar(value=2.0)
        self.fisheye_interval_label = ctk.CTkLabel(interval_frame, text="2.0s", width=50)
        self.fisheye_interval_label.pack(side="right")
        ctk.CTkSlider(interval_frame, from_=0.5, to=10.0, variable=self.fisheye_interval_var,
                      command=lambda v: self.fisheye_interval_label.configure(text=f"{v:.1f}s")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Motion selection (collapsible)
        self.fisheye_motion_section = CollapsibleSection(
            left, "Motion-Aware Selection", expanded=False, fg_color="transparent")
        self.fisheye_motion_section.pack(fill="x", padx=10, pady=(5, 0))

        mc = self.fisheye_motion_section.content

        self.fisheye_motion_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(mc, text="Enable motion-aware selection",
                        variable=self.fisheye_motion_var).pack(pady=3, padx=10, anchor="w")

        sharp_frame = ctk.CTkFrame(master=mc, fg_color="transparent")
        sharp_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(sharp_frame, text="Min Sharpness:", width=100, anchor="w").pack(side="left")
        self.fisheye_sharpness_var = ctk.DoubleVar(value=50.0)
        self.fisheye_sharpness_label = ctk.CTkLabel(sharp_frame, text="50", width=40)
        self.fisheye_sharpness_label.pack(side="right")
        ctk.CTkSlider(sharp_frame, from_=10, to=200, variable=self.fisheye_sharpness_var,
                      command=lambda v: self.fisheye_sharpness_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        flow_frame = ctk.CTkFrame(master=mc, fg_color="transparent")
        flow_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(flow_frame, text="Target Flow:", width=100, anchor="w").pack(side="left")
        self.fisheye_flow_var = ctk.DoubleVar(value=10.0)
        self.fisheye_flow_label = ctk.CTkLabel(flow_frame, text="10", width=40)
        self.fisheye_flow_label.pack(side="right")
        ctk.CTkSlider(flow_frame, from_=2, to=30, variable=self.fisheye_flow_var,
                      command=lambda v: self.fisheye_flow_label.configure(text=f"{int(v)}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        sample_frame = ctk.CTkFrame(master=mc, fg_color="transparent")
        sample_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(sample_frame, text="Sample FPS:", width=100, anchor="w").pack(side="left")
        self.fisheye_sample_fps_var = ctk.DoubleVar(value=5.0)
        self.fisheye_sample_fps_label = ctk.CTkLabel(sample_frame, text="5.0", width=40)
        self.fisheye_sample_fps_label.pack(side="right")
        ctk.CTkSlider(sample_frame, from_=1, to=15, variable=self.fisheye_sample_fps_var,
                      command=lambda v: self.fisheye_sample_fps_label.configure(text=f"{v:.1f}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Buttons
        self.fisheye_run_btn = ctk.CTkButton(
            left, text="Extract Fisheye Views", command=self.run_fisheye,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.fisheye_run_btn.pack(pady=20, padx=10, fill="x")

        self.fisheye_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # --- Right: Info + Results ---
        ctk.CTkLabel(right, text="Fisheye Extraction",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5), padx=15, anchor="w")

        ctk.CTkLabel(right, text="Extract perspective views from DJI Osmo 360 dual-fisheye",
                     text_color="gray").pack(pady=(0, 10), padx=15, anchor="w")

        # Preset info display
        self.fisheye_info_text = ctk.CTkTextbox(
            right, height=120, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a", state="disabled"
        )
        self.fisheye_info_text.pack(fill="x", padx=15, pady=(0, 10))

        # OSV probe info
        ctk.CTkLabel(right, text="File Info",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(5, 5), padx=15, anchor="w")

        self.fisheye_probe_text = ctk.CTkTextbox(
            right, height=100, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a", state="disabled"
        )
        self.fisheye_probe_text.pack(fill="x", padx=15, pady=(0, 10))

        # Results
        ctk.CTkLabel(right, text="Results",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(5, 5), padx=15, anchor="w")

        self.fisheye_result_text = ctk.CTkTextbox(
            right, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a", state="disabled"
        )
        self.fisheye_result_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Initialize preset display
        self._on_fisheye_preset_change(self.fisheye_preset_var.get())

    def _browse_osv_file(self):
        path = filedialog.askopenfilename(
            title="Select OSV / 360 File",
            filetypes=[
                ("DJI 360 Files", "*.osv *.360 *.insv"),
                ("Video Files", "*.mp4 *.mov"),
                ("All Files", "*.*"),
            ]
        )
        if path:
            self.fisheye_osv_entry.delete(0, 'end')
            self.fisheye_osv_entry.insert(0, path)
            # Auto-probe in background
            threading.Thread(target=self._probe_osv, args=(path,), daemon=True).start()

    def _probe_osv(self, path):
        try:
            handler = OSVHandler()
            info = handler.probe(path)
            summary = info.summary()
            self.after(0, lambda: self._set_textbox(self.fisheye_probe_text, summary))
            self.log(f"Probed: {info.filename} — {info.duration:.1f}s, {info.fps:.0f}fps")
        except Exception as e:
            self.after(0, lambda: self._set_textbox(self.fisheye_probe_text, f"Error: {e}"))

    def _browse_calib_file(self):
        path = filedialog.askopenfilename(
            title="Select Calibration JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if path:
            self.fisheye_calib_entry.configure(state="normal")
            self.fisheye_calib_entry.delete(0, 'end')
            self.fisheye_calib_entry.insert(0, path)
            if self.fisheye_calib_default_var.get():
                self.fisheye_calib_entry.configure(state="disabled")

    def _on_fisheye_calib_toggle(self):
        if self.fisheye_calib_default_var.get():
            self.fisheye_calib_entry.configure(state="disabled")
            self.fisheye_calib_browse_btn.configure(state="disabled")
        else:
            self.fisheye_calib_entry.configure(state="normal")
            self.fisheye_calib_browse_btn.configure(state="normal")

    def _on_fisheye_preset_change(self, preset_name):
        if not HAS_PREP360 or preset_name not in FISHEYE_PRESETS:
            return
        config = FISHEYE_PRESETS[preset_name]
        desc = f"{config.total_views()} views"
        self.fisheye_preset_desc.configure(text=desc)
        self._set_textbox(self.fisheye_info_text, config.summary())

    def _set_textbox(self, textbox, text):
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")

    def _get_fisheye_calibration(self):
        """Load the fisheye calibration (default or custom)."""
        if self.fisheye_calib_default_var.get():
            return default_osmo360_calibration()
        calib_path = self.fisheye_calib_entry.get()
        if not calib_path:
            raise ValueError("No calibration file selected")
        return DualFisheyeCalibration.load(calib_path)

    def run_fisheye(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        osv_path = self.fisheye_osv_entry.get()
        output_dir = self.fisheye_output_entry.get()

        if not osv_path:
            self.log("Error: Please select an OSV file")
            return
        if not Path(osv_path).exists():
            self.log(f"Error: File not found: {osv_path}")
            return
        if not output_dir:
            self.log("Error: Please select an output folder")
            return

        self._start_operation(self.fisheye_run_btn, self.fisheye_stop_btn)
        threading.Thread(
            target=self._fisheye_worker, args=(osv_path, output_dir),
            daemon=True).start()

    def _fisheye_worker(self, osv_path, output_dir):
        try:
            # Load calibration
            calib = self._get_fisheye_calibration()
            self.log(f"Calibration: {calib.camera_model}")

            # Build view config from preset + user settings
            preset_name = self.fisheye_preset_var.get()
            preset_config = FISHEYE_PRESETS.get(preset_name)
            if preset_config is None:
                self.log(f"Error: Unknown preset '{preset_name}'")
                return

            crop_size = int(self.fisheye_crop_var.get())
            quality = self.fisheye_quality_var.get()
            config = FisheyeViewConfig(
                views=list(preset_config.views),
                crop_size=crop_size,
                quality=quality,
            )

            self.log(f"Preset: {preset_name} ({config.total_views()} views)")
            self.log(f"Crop: {crop_size}x{crop_size} @ Q{quality}")

            # Step 1: Extract fisheye frames from OSV
            interval = self.fisheye_interval_var.get()
            frames_dir = str(Path(output_dir) / "fisheye_frames")

            self.log(f"\nExtracting fisheye frames (interval={interval:.1f}s)...")

            handler = OSVHandler()

            def osv_progress(stream_name, current, total):
                if not self.cancel_flag.is_set():
                    self.log(f"  [{stream_name}] {current}/{total}")

            frame_dict = handler.extract_frames(
                osv_path, frames_dir,
                interval=interval,
                streams="both",
                quality=95,
                progress_callback=osv_progress,
            )

            front_frames = sorted(frame_dict.get("front", []))
            back_frames = sorted(frame_dict.get("back", []))

            if not front_frames or not back_frames:
                self.log("Error: No frames extracted")
                return

            self.log(f"Extracted {len(front_frames)} front + {len(back_frames)} back frames")

            if self.cancel_flag.is_set():
                self.log("Cancelled")
                return

            # Optional: motion-aware selection
            if self.fisheye_motion_var.get():
                self.log("\nRunning motion-aware selection...")
                selector = MotionSelector(
                    min_sharpness=self.fisheye_sharpness_var.get(),
                    target_flow=self.fisheye_flow_var.get(),
                )
                # Select from front stream (representative)
                result = selector.select_from_paths(
                    front_frames,
                    max_frames=len(front_frames),
                )
                self.log(result.summary())

                # Keep only the selected indices
                selected_indices = {fs.index for fs in result.selected_frames}
                front_frames = [f for i, f in enumerate(front_frames) if i in selected_indices]
                back_frames = [b for i, b in enumerate(back_frames) if i in selected_indices]
                self.log(f"After selection: {len(front_frames)} frame pairs")

            if self.cancel_flag.is_set():
                self.log("Cancelled")
                return

            # Step 2: Reframe fisheye → perspective
            frame_pairs = list(zip(front_frames, back_frames))
            crops_dir = str(Path(output_dir) / "perspectives")

            self.log(f"\nReframing {len(frame_pairs)} pairs → {config.total_views()} views each...")

            def reframe_progress(current, total, msg):
                if not self.cancel_flag.is_set():
                    self.log(f"  [{current}/{total}] {msg}")

            total_crops, errors = fisheye_batch_extract(
                frame_pairs, config, calib, crops_dir,
                num_workers=1,
                progress_callback=reframe_progress,
            )

            result_lines = [
                f"Fisheye extraction complete",
                f"  Frame pairs:   {len(frame_pairs)}",
                f"  Views/pair:    {config.total_views()}",
                f"  Total crops:   {total_crops}",
                f"  Crop size:     {crop_size}x{crop_size}",
                f"  Output:        {crops_dir}",
            ]
            if errors:
                result_lines.append(f"  Errors:        {len(errors)}")
                for err in errors[:5]:
                    result_lines.append(f"    - {err}")

            summary = "\n".join(result_lines)
            self.log(f"\n{summary}")
            self.after(0, lambda: self._set_textbox(self.fisheye_result_text, summary))

        except Exception as e:
            self.log(f"Fisheye error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(
                self.fisheye_run_btn, self.fisheye_stop_btn))

    # =========================================================================
    # TAB: Gaps  (Gap detection + bridge frame extraction)
    # =========================================================================
    def _build_gaps_tab(self):
        tab = self.tab_view.tab("Gaps")
        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=420)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # --- Left: Controls ---
        self._section_header(left, "Camera Source")

        # Input source radio buttons
        self.gaps_source_var = ctk.StringVar(value="colmap")
        sources_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        sources_frame.pack(pady=5, padx=20, fill="x")

        ctk.CTkRadioButton(sources_frame, text="COLMAP directory",
                           variable=self.gaps_source_var, value="colmap",
                           command=self._on_gaps_source_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(sources_frame, text="Metashape XML",
                           variable=self.gaps_source_var, value="metashape",
                           command=self._on_gaps_source_change).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(sources_frame, text="XMP sidecars",
                           variable=self.gaps_source_var, value="xmp",
                           command=self._on_gaps_source_change).pack(anchor="w", pady=2)

        # Source path picker (dynamic label)
        src_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        src_frame.pack(pady=5, padx=10, fill="x")
        self.gaps_source_label = ctk.CTkLabel(src_frame, text="COLMAP:", width=80, anchor="w")
        self.gaps_source_label.pack(side="left")
        self.gaps_source_entry = ctk.CTkEntry(src_frame, placeholder_text="Select folder or file...")
        self.gaps_source_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        self.gaps_source_browse_btn = ctk.CTkButton(
            src_frame, text="...", width=40,
            command=self._browse_gaps_source)
        self.gaps_source_browse_btn.pack(side="left")

        # Source images (optional)
        img_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        img_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(img_frame, text="Images:", width=80, anchor="w").pack(side="left")
        self.gaps_images_entry = ctk.CTkEntry(img_frame, placeholder_text="Source images (optional)...")
        self.gaps_images_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(img_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.gaps_images_entry)).pack(side="left")

        # Analysis parameters (collapsible)
        params_section = CollapsibleSection(
            left, "Analysis Parameters", expanded=False, fg_color="transparent")
        params_section.pack(fill="x", padx=10, pady=(5, 0))

        pc = params_section.content

        eps_frame = ctk.CTkFrame(master=pc, fg_color="transparent")
        eps_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(eps_frame, text="Cluster eps:", width=100, anchor="w").pack(side="left")
        self.gaps_eps_var = ctk.DoubleVar(value=5.0)
        self.gaps_eps_label = ctk.CTkLabel(eps_frame, text="5.0", width=40)
        self.gaps_eps_label.pack(side="right")
        ctk.CTkSlider(eps_frame, from_=1.0, to=20.0, variable=self.gaps_eps_var,
                      command=lambda v: self.gaps_eps_label.configure(text=f"{v:.1f}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        sparse_frame = ctk.CTkFrame(master=pc, fg_color="transparent")
        sparse_frame.pack(pady=3, padx=10, fill="x")
        ctk.CTkLabel(sparse_frame, text="Sparse thresh:", width=100, anchor="w").pack(side="left")
        self.gaps_sparse_var = ctk.DoubleVar(value=2.0)
        self.gaps_sparse_label = ctk.CTkLabel(sparse_frame, text="2.0x", width=40)
        self.gaps_sparse_label.pack(side="right")
        ctk.CTkSlider(sparse_frame, from_=1.0, to=5.0, variable=self.gaps_sparse_var,
                      command=lambda v: self.gaps_sparse_label.configure(text=f"{v:.1f}x")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Analyze button
        self.gaps_analyze_btn = ctk.CTkButton(
            left, text="Analyze Gaps", command=self.run_gap_analysis,
            fg_color="#1565C0", hover_color="#0D47A1",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.gaps_analyze_btn.pack(pady=(15, 5), padx=10, fill="x")

        # Bridge extraction (collapsible)
        self.gaps_bridge_section = CollapsibleSection(
            left, "Bridge Extraction", expanded=False, fg_color="transparent")
        self.gaps_bridge_section.pack(fill="x", padx=10, pady=(10, 0))

        bc = self.gaps_bridge_section.content

        # Source video
        vid_frame = ctk.CTkFrame(master=bc, fg_color="transparent")
        vid_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(vid_frame, text="Video:", width=80, anchor="w").pack(side="left")
        self.gaps_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="Source video for bridge frames...")
        self.gaps_video_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(vid_frame, text="...", width=40,
                      command=lambda: self._browse_video_for(self.gaps_video_entry)).pack(side="left")

        # Bridge output
        bout_frame = ctk.CTkFrame(master=bc, fg_color="transparent")
        bout_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(bout_frame, text="Output:", width=80, anchor="w").pack(side="left")
        self.gaps_bridge_output_entry = ctk.CTkEntry(bout_frame, placeholder_text="Bridge frames output...")
        self.gaps_bridge_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(bout_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.gaps_bridge_output_entry)).pack(side="left")

        # Reframe toggle
        self.gaps_reframe_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bc, text="Reframe fisheye → perspective",
                        variable=self.gaps_reframe_var).pack(pady=3, padx=10, anchor="w")

        # Extract bridges button
        self.gaps_bridge_btn = ctk.CTkButton(
            bc, text="Extract Bridges", command=self.run_bridge_extraction,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=13, weight="bold"), height=40
        )
        self.gaps_bridge_btn.pack(pady=(10, 5), padx=10, fill="x")

        self.gaps_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_operation,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # --- Right: Results ---
        ctk.CTkLabel(right, text="Gap Analysis",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5), padx=15, anchor="w")

        ctk.CTkLabel(right, text="Detect spatial gaps and disconnected components in reconstructions",
                     text_color="gray").pack(pady=(0, 10), padx=15, anchor="w")

        # Summary
        self.gaps_summary_text = ctk.CTkTextbox(
            right, height=150, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a", state="disabled"
        )
        self.gaps_summary_text.pack(fill="x", padx=15, pady=(0, 10))

        # Gap list
        ctk.CTkLabel(right, text="Gap Details",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(5, 5), padx=15, anchor="w")

        self.gaps_detail_text = ctk.CTkTextbox(
            right, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1a1a1a", state="disabled"
        )
        self.gaps_detail_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Store last gap report for bridge extraction
        self._last_gap_report = None

    def _on_gaps_source_change(self):
        source = self.gaps_source_var.get()
        labels = {"colmap": "COLMAP:", "metashape": "XML:", "xmp": "XMP Dir:"}
        placeholders = {
            "colmap": "COLMAP output directory...",
            "metashape": "Metashape cameras.xml...",
            "xmp": "XMP sidecar directory...",
        }
        self.gaps_source_label.configure(text=labels[source])
        self.gaps_source_entry.delete(0, 'end')
        self.gaps_source_entry.configure(placeholder_text=placeholders[source])

    def _browse_gaps_source(self):
        source = self.gaps_source_var.get()
        if source == "metashape":
            path = filedialog.askopenfilename(
                title="Select Metashape cameras.xml",
                filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")]
            )
            if path:
                self.gaps_source_entry.delete(0, 'end')
                self.gaps_source_entry.insert(0, path)
        else:
            self._browse_folder_for(self.gaps_source_entry)

    def run_gap_analysis(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        source_path = self.gaps_source_entry.get()
        if not source_path:
            self.log("Error: Please select a camera source")
            return
        if not Path(source_path).exists():
            self.log(f"Error: Path not found: {source_path}")
            return

        self.gaps_analyze_btn.configure(state="disabled")
        threading.Thread(
            target=self._gap_analysis_worker, args=(source_path,),
            daemon=True).start()

    def _gap_analysis_worker(self, source_path):
        try:
            source_type = self.gaps_source_var.get()
            images_dir = self.gaps_images_entry.get() or None

            detector = GapDetector(
                cluster_eps=self.gaps_eps_var.get(),
                sparse_threshold=self.gaps_sparse_var.get(),
            )

            self.log(f"Analyzing gaps ({source_type}: {Path(source_path).name})...")

            kwargs = {}
            if source_type == "colmap":
                kwargs["colmap_dir"] = source_path
            elif source_type == "metashape":
                kwargs["metashape_xml"] = source_path
            elif source_type == "xmp":
                kwargs["xmp_dir"] = source_path

            if images_dir:
                kwargs["source_images_dir"] = images_dir

            report = detector.analyze(**kwargs)
            self._last_gap_report = report

            summary = report.summary()
            self.log(f"\n{summary}")

            # Build gap detail text
            detail_lines = []
            for i, gap in enumerate(report.gaps):
                detail_lines.append(f"Gap {i}: {gap.gap_type}")
                detail_lines.append(f"  Cameras: {', '.join(gap.camera_names[:5])}")
                if len(gap.camera_names) > 5:
                    detail_lines.append(f"  ... and {len(gap.camera_names) - 5} more")
                detail_lines.append(f"  Distance: {gap.distance:.2f}")
                detail_lines.append(f"  Severity: {gap.severity:.2f}")
                if gap.estimated_timestamps:
                    t_start, t_end = gap.estimated_timestamps
                    detail_lines.append(f"  Timestamps: {t_start:.1f}s - {t_end:.1f}s")
                detail_lines.append("")

            if not report.gaps:
                detail_lines.append("No spatial gaps detected — reconstruction looks well-connected.")

            detail_text = "\n".join(detail_lines)

            self.after(0, lambda: self._set_textbox(self.gaps_summary_text, summary))
            self.after(0, lambda: self._set_textbox(self.gaps_detail_text, detail_text))

            # Auto-expand bridge section if gaps found
            if report.gaps:
                self.after(0, lambda: self.gaps_bridge_section.expand())

        except Exception as e:
            self.log(f"Gap analysis error: {e}")
        finally:
            self.after(0, lambda: self.gaps_analyze_btn.configure(state="normal"))

    def run_bridge_extraction(self):
        if not HAS_PREP360:
            self.log("Error: prep360 core not available")
            return

        if self._last_gap_report is None:
            self.log("Error: Run gap analysis first")
            return

        video_path = self.gaps_video_entry.get()
        output_dir = self.gaps_bridge_output_entry.get()

        if not video_path:
            self.log("Error: Please select a source video")
            return
        if not Path(video_path).exists():
            self.log(f"Error: Video not found: {video_path}")
            return
        if not output_dir:
            self.log("Error: Please select an output folder")
            return

        self._start_operation(self.gaps_bridge_btn, self.gaps_stop_btn)
        threading.Thread(
            target=self._bridge_worker, args=(video_path, output_dir),
            daemon=True).start()

    def _bridge_worker(self, video_path, output_dir):
        try:
            extractor = BridgeExtractor()

            reframe = self.gaps_reframe_var.get()
            reframer = None
            view_config = None

            if reframe:
                try:
                    calib = self._get_fisheye_calibration()
                    reframer = FisheyeReframer(calib)
                    preset_name = self.fisheye_preset_var.get()
                    view_config = FISHEYE_PRESETS.get(preset_name)
                    self.log(f"Reframing enabled: {preset_name}")
                except Exception as e:
                    self.log(f"Warning: Cannot reframe ({e}), extracting raw frames")
                    reframe = False

            def progress(msg):
                if not self.cancel_flag.is_set():
                    self.log(f"  {msg}")

            result = extractor.extract(
                gap_report=self._last_gap_report,
                video_path=video_path,
                output_dir=output_dir,
                reframe=reframe,
                reframer=reframer,
                view_config=view_config,
                progress_callback=progress,
            )

            summary = result.summary()
            self.log(f"\n{summary}")
            self.after(0, lambda: self._set_textbox(self.gaps_detail_text, summary))

            # Save report
            report_path = str(Path(output_dir) / "bridge_result.json")
            result.save(report_path)
            self.log(f"Report saved: {report_path}")

        except Exception as e:
            self.log(f"Bridge extraction error: {e}")
        finally:
            self.after(0, lambda: self._stop_operation(
                self.gaps_bridge_btn, self.gaps_stop_btn))

    # =========================================================================
    # Shared Helpers
    # =========================================================================
    def _section_header(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(pady=(15, 5), padx=10, anchor="w")

    def _browse_video_for(self, entry):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")]
        )
        if path:
            entry.delete(0, 'end')
            entry.insert(0, path)

    def _browse_folder_for(self, entry, callback=None):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            entry.delete(0, 'end')
            entry.insert(0, folder)
            if callback:
                callback()

    def _start_operation(self, run_btn, stop_btn):
        self.is_running = True
        self.cancel_flag.clear()
        self.progress_bar.start()
        run_btn.configure(state="disabled")
        stop_btn.pack(pady=(0, 10), padx=10, fill="x")

    def _stop_operation(self, run_btn, stop_btn):
        self.is_running = False
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        run_btn.configure(state="normal")
        stop_btn.pack_forget()

    def stop_operation(self):
        if self.is_running:
            self.log("Stopping...")
            self.cancel_flag.set()

    def _setup_console_redirect(self):
        class QueueWriter:
            def __init__(self, q):
                self.q = q
            def write(self, text):
                if text.strip():
                    self.q.put(text.rstrip())
            def flush(self):
                pass
        sys.stdout = QueueWriter(self.log_queue)
        sys.stderr = QueueWriter(self.log_queue)

    def log(self, message):
        self.log_queue.put(message)

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_textbox.insert('end', msg + '\n')
                self.log_textbox.see('end')
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)


# =============================================================================
# SegmentationWindow - Modal for YOLO Segmentation
# =============================================================================

class SegmentationWindow(ctk.CTkToplevel):
    """Standalone window for YOLO segmentation with categorized class selection."""

    # COCO class categories
    CATEGORIES = {
        "People": [0],  # person
        "Vehicles": [1, 2, 3, 4, 5, 6, 7, 8],  # bicycle, car, motorcycle, airplane, bus, train, truck, boat
        "Animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird through giraffe
        "Sports": [32, 33, 34, 35, 36, 37, 38],  # sports ball through surfboard
        "Food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],  # banana through sandwich, etc.
        "Furniture": [56, 57, 58, 59, 60, 61, 62, 63],  # chair through bed
        "Electronics": [62, 63, 64, 65, 66, 67, 68, 69, 70, 71],  # tv through cell phone
        "Kitchen": [39, 40, 41, 42, 43, 44, 45, 71, 72, 73, 74, 75],  # bottle through cup, kitchen items
        "Other": [24, 25, 26, 27, 28, 29, 30, 31, 76, 77, 78, 79],  # backpack, umbrella, etc.
    }

    def __init__(self, parent, input_folder: str = ""):
        super().__init__(parent)
        self.parent = parent
        self.title("Mask Tool - YOLO Segmentation")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.input_folder = input_folder
        self.test_frame_paths: List[Path] = []
        self.selected_classes: set = {0}  # Default: person

        self._build_ui()

        # Focus on this window
        self.transient(parent)
        self.grab_set()

    def _build_ui(self):
        """Build the segmentation window UI."""
        # Main container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=15)

        # Left: Settings
        left = ctk.CTkFrame(container, width=350)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Right: Preview
        right = ctk.CTkFrame(container)
        right.pack(side="left", fill="both", expand=True)

        # === LEFT: Settings ===
        ctk.CTkLabel(left, text="Input / Output",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 5), padx=10, anchor="w")

        # Input folder
        in_frame = ctk.CTkFrame(left, fg_color="transparent")
        in_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(in_frame, text="Input:", width=50, anchor="w").pack(side="left")
        self.input_entry = ctk.CTkEntry(in_frame, placeholder_text="Images folder...")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        if self.input_folder:
            self.input_entry.insert(0, self.input_folder)
        ctk.CTkButton(in_frame, text="...", width=35,
                      command=self._browse_input).pack(side="left")

        # Output folder
        out_frame = ctk.CTkFrame(left, fg_color="transparent")
        out_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(out_frame, text="Output:", width=50, anchor="w").pack(side="left")
        self.output_entry = ctk.CTkEntry(out_frame, placeholder_text="Masks output...")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(out_frame, text="...", width=35,
                      command=self._browse_output).pack(side="left")

        # === Class Selection ===
        ctk.CTkLabel(left, text="Detection Classes",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(15, 5), padx=10, anchor="w")

        # Quick select buttons
        quick_frame = ctk.CTkFrame(left, fg_color="transparent")
        quick_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(quick_frame, text="Default", width=70, height=28,
                      command=self._select_default,
                      fg_color="#555555").pack(side="left", padx=(0, 5))
        ctk.CTkButton(quick_frame, text="All", width=50, height=28,
                      command=self._select_all,
                      fg_color="#555555").pack(side="left", padx=(0, 5))
        ctk.CTkButton(quick_frame, text="None", width=50, height=28,
                      command=self._select_none,
                      fg_color="#555555").pack(side="left")

        # Scrollable class selection
        class_scroll = ctk.CTkScrollableFrame(left, height=200)
        class_scroll.pack(fill="x", padx=10, pady=5)

        self.class_checkboxes = {}
        for category, class_ids in self.CATEGORIES.items():
            # Category header
            cat_frame = ctk.CTkFrame(class_scroll, fg_color="transparent")
            cat_frame.pack(fill="x", pady=(5, 2))

            ctk.CTkLabel(cat_frame, text=category,
                         font=ctk.CTkFont(size=11, weight="bold")).pack(side="left")

            # Class checkboxes in a grid
            classes_frame = ctk.CTkFrame(class_scroll, fg_color="transparent")
            classes_frame.pack(fill="x", padx=(15, 0))

            for i, cid in enumerate(class_ids):
                class_name = COCO_CLASSES.get(cid, f"class_{cid}") if HAS_PREP360 else f"class_{cid}"
                var = ctk.BooleanVar(value=(cid in self.selected_classes))
                cb = ctk.CTkCheckBox(classes_frame, text=class_name, variable=var,
                                     font=ctk.CTkFont(size=10),
                                     command=lambda cid=cid, v=var: self._on_class_toggle(cid, v))
                cb.grid(row=i // 3, column=i % 3, sticky="w", padx=2, pady=1)
                self.class_checkboxes[cid] = (var, cb)

        # === Model Settings ===
        ctk.CTkLabel(left, text="Model Settings",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(15, 5), padx=10, anchor="w")

        # Model selection
        model_frame = ctk.CTkFrame(left, fg_color="transparent")
        model_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(model_frame, text="Model:", width=70, anchor="w").pack(side="left")
        self.model_var = ctk.StringVar(value="yolo11n-seg.pt")
        ctk.CTkComboBox(model_frame, variable=self.model_var,
                        values=["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
                                "yolo11l-seg.pt", "yolo11x-seg.pt"],
                        state="readonly", width=150).pack(side="left", padx=(5, 0))

        # Confidence
        conf_frame = ctk.CTkFrame(left, fg_color="transparent")
        conf_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(conf_frame, text="Confidence:", width=70, anchor="w").pack(side="left")
        self.conf_var = ctk.DoubleVar(value=0.5)
        self.conf_label = ctk.CTkLabel(conf_frame, text="0.50", width=40)
        self.conf_label.pack(side="right")
        ctk.CTkSlider(conf_frame, from_=0.1, to=0.9, variable=self.conf_var,
                      command=lambda v: self.conf_label.configure(text=f"{float(v):.2f}")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Dilate
        dilate_frame = ctk.CTkFrame(left, fg_color="transparent")
        dilate_frame.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(dilate_frame, text="Dilate:", width=70, anchor="w").pack(side="left")
        self.dilate_var = ctk.IntVar(value=5)
        self.dilate_label = ctk.CTkLabel(dilate_frame, text="5 px", width=40)
        self.dilate_label.pack(side="right")
        ctk.CTkSlider(dilate_frame, from_=0, to=50, variable=self.dilate_var,
                      command=lambda v: self.dilate_label.configure(text=f"{int(v)} px")
                      ).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Invert option
        self.invert_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Invert mask (white background)",
                        variable=self.invert_var).pack(pady=5, padx=20, anchor="w")

        # === Buttons ===
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=15)

        self.run_btn = ctk.CTkButton(
            btn_frame, text="Generate Masks",
            fg_color="#E65100", hover_color="#BF360C",
            font=ctk.CTkFont(size=14, weight="bold"), height=45,
            command=self._run_segmentation
        )
        self.run_btn.pack(fill="x")

        # === RIGHT: Preview ===
        ctk.CTkLabel(right, text="Test Run",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5), padx=15, anchor="w")

        # Frame slider
        slider_frame = ctk.CTkFrame(right, fg_color="transparent")
        slider_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(slider_frame, text="Frame:", width=50, anchor="w").pack(side="left")
        self.test_frame_var = ctk.IntVar(value=0)
        self.test_frame_label = ctk.CTkLabel(slider_frame, text="0", width=40)
        self.test_frame_label.pack(side="right")
        self.test_slider = ctk.CTkSlider(
            slider_frame, from_=0, to=100, variable=self.test_frame_var,
            command=self._on_test_frame_change
        )
        self.test_slider.pack(side="left", fill="x", expand=True, padx=(5, 5))

        # Test run button
        ctk.CTkButton(right, text="Run Test", width=100,
                      command=self._run_test,
                      fg_color="#1976D2", hover_color="#1565C0"
                      ).pack(pady=5, padx=15, anchor="w")

        # Preview images (side by side)
        preview_frame = ctk.CTkFrame(right)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        # Original
        original_frame = ctk.CTkFrame(preview_frame, fg_color="#1a1a1a")
        original_frame.pack(side="left", fill="both", expand=True, padx=(5, 2), pady=5)
        ctk.CTkLabel(original_frame, text="Original",
                     font=ctk.CTkFont(size=11)).pack(pady=(5, 0))
        self.preview_original = ctk.CTkLabel(original_frame, text="Load images\nto preview",
                                              fg_color="transparent")
        self.preview_original.pack(fill="both", expand=True, padx=5, pady=5)

        # With mask overlay
        masked_frame = ctk.CTkFrame(preview_frame, fg_color="#1a1a1a")
        masked_frame.pack(side="left", fill="both", expand=True, padx=(2, 5), pady=5)
        ctk.CTkLabel(masked_frame, text="With Mask",
                     font=ctk.CTkFont(size=11)).pack(pady=(5, 0))
        self.preview_masked = ctk.CTkLabel(masked_frame, text="Run test\nto see mask",
                                            fg_color="transparent")
        self.preview_masked.pack(fill="both", expand=True, padx=5, pady=5)

        # Load test frames if input folder set
        if self.input_folder:
            self._load_test_frames()

    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_entry.delete(0, 'end')
            self.input_entry.insert(0, folder)
            self.input_folder = folder
            self._load_test_frames()

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_entry.delete(0, 'end')
            self.output_entry.insert(0, folder)

    def _load_test_frames(self):
        """Load frame list for test slider."""
        folder = Path(self.input_folder)
        if not folder.exists():
            return

        self.test_frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.test_frame_paths.extend(sorted(folder.glob(ext)))

        if self.test_frame_paths:
            self.test_slider.configure(to=len(self.test_frame_paths) - 1)
            self.test_frame_var.set(0)
            self._update_original_preview()

    def _on_class_toggle(self, class_id: int, var):
        """Handle class checkbox toggle."""
        if var.get():
            self.selected_classes.add(class_id)
        else:
            self.selected_classes.discard(class_id)

    def _select_default(self):
        """Select default classes (person, backpack, etc.)."""
        default_classes = {0, 24, 25, 26, 28}  # person, backpack, umbrella, handbag, suitcase
        self.selected_classes = default_classes.copy()
        for cid, (var, cb) in self.class_checkboxes.items():
            var.set(cid in default_classes)

    def _select_all(self):
        """Select all classes."""
        self.selected_classes = set(self.class_checkboxes.keys())
        for var, cb in self.class_checkboxes.values():
            var.set(True)

    def _select_none(self):
        """Deselect all classes."""
        self.selected_classes = set()
        for var, cb in self.class_checkboxes.values():
            var.set(False)

    def _on_test_frame_change(self, value):
        """Handle test frame slider change."""
        self.test_frame_label.configure(text=f"{int(value)}")
        self._update_original_preview()

    def _update_original_preview(self):
        """Update the original image preview."""
        if not HAS_PIL or not self.test_frame_paths:
            return

        idx = self.test_frame_var.get()
        if idx >= len(self.test_frame_paths):
            return

        try:
            img_path = self.test_frame_paths[idx]
            img = Image.open(img_path)
            # Resize for preview
            img.thumbnail((350, 250), Image.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_original.configure(image=ctk_img, text="")
            self.preview_original._ctk_image = ctk_img
        except Exception:
            pass

    def _run_test(self):
        """Run test segmentation on current frame."""
        if not HAS_PREP360 or not HAS_YOLO:
            return

        if not self.test_frame_paths:
            return

        idx = self.test_frame_var.get()
        if idx >= len(self.test_frame_paths):
            return

        threading.Thread(target=self._test_worker, args=(idx,), daemon=True).start()

    def _test_worker(self, frame_idx: int):
        """Worker thread for test segmentation."""
        try:
            import cv2
            import numpy as np

            img_path = self.test_frame_paths[frame_idx]

            config = SegmentConfig(
                model_name=self.model_var.get(),
                classes=list(self.selected_classes),
                confidence=self.conf_var.get(),
                dilate_pixels=self.dilate_var.get(),
                invert_mask=self.invert_var.get(),
            )

            segmenter = Segmenter(config)

            # Run segmentation
            img = cv2.imread(str(img_path))
            mask = segmenter.segment_image(img)

            # Create overlay visualization
            overlay = img.copy()
            if mask is not None:
                # Color mask regions
                mask_colored = np.zeros_like(img)
                mask_colored[mask > 0] = [0, 255, 0]  # Green for detected regions
                overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            # Convert to PIL
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)
            overlay_pil.thumbnail((350, 250), Image.LANCZOS)

            def update_gui():
                ctk_img = ctk.CTkImage(light_image=overlay_pil, dark_image=overlay_pil,
                                       size=overlay_pil.size)
                self.preview_masked.configure(image=ctk_img, text="")
                self.preview_masked._ctk_image = ctk_img

            self.after(0, update_gui)

        except Exception as e:
            self.after(0, lambda: self.preview_masked.configure(text=f"Error: {e}"))

    def _run_segmentation(self):
        """Run full segmentation on all images."""
        if not HAS_PREP360 or not HAS_YOLO:
            if self.parent:
                self.parent.log("Error: ultralytics not installed")
            return

        input_dir = self.input_entry.get()
        output_dir = self.output_entry.get()

        if not input_dir or not Path(input_dir).exists():
            if self.parent:
                self.parent.log("Error: Invalid input folder")
            return

        if not output_dir:
            if self.parent:
                self.parent.log("Error: Select output folder")
            return

        if not self.selected_classes:
            if self.parent:
                self.parent.log("Error: Select at least one class")
            return

        self.run_btn.configure(state="disabled")
        threading.Thread(target=self._segmentation_worker,
                         args=(input_dir, output_dir), daemon=True).start()

    def _segmentation_worker(self, input_dir: str, output_dir: str):
        """Worker thread for full segmentation."""
        try:
            config = SegmentConfig(
                model_name=self.model_var.get(),
                classes=list(self.selected_classes),
                confidence=self.conf_var.get(),
                dilate_pixels=self.dilate_var.get(),
                invert_mask=self.invert_var.get(),
            )

            segmenter = Segmenter(config)

            def progress(curr, total, name):
                if self.parent:
                    self.parent.log(f"[{curr}/{total}] {name}")

            if self.parent:
                self.parent.log(f"Segmenting: {input_dir}")
                self.parent.log(f"Classes: {list(self.selected_classes)}")

            result = segmenter.segment_batch(input_dir, output_dir, progress)

            if self.parent:
                self.parent.log(f"Processed {result.total_images} images")
                self.parent.log(f"Detections: {result.total_detections}")

        except Exception as e:
            if self.parent:
                self.parent.log(f"Error: {e}")
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal"))


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = PanoexApp()
    app.mainloop()

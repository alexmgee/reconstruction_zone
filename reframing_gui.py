#!/usr/bin/env python3
"""
Reframing GUI - 360 Video to Pinhole Perspectives Pipeline

Complete workflow:
1. Extract Frames - Video to equirectangular frames (ffmpeg)
2. Filter Blurry - Remove motion blur (Laplacian variance)
3. Reframe - Equirectangular to pinhole perspectives
"""

import customtkinter as ctk
import threading
import queue
import subprocess
import sys
from tkinter import filedialog
from pathlib import Path

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# View presets for reframing
VIEW_PRESETS = {
    "6-cubemap": {"views": 6, "fov_range": (90, 90), "spacing": 90, "desc": "Standard cubemap (no overlap)"},
    "8-hex": {"views": 8, "fov_range": (70, 80), "spacing": 60, "desc": "6 horizontal + top/bottom"},
    "10-octa": {"views": 10, "fov_range": (60, 70), "spacing": 45, "desc": "8 horizontal + top/bottom [RECOMMENDED]"},
    "12-full": {"views": 12, "fov_range": (55, 65), "spacing": 45, "desc": "8 horizontal + 2 elevated + top/bottom"},
    "14-dense": {"views": 14, "fov_range": (50, 60), "spacing": 45, "desc": "8 horizontal + 4 elevated + top/bottom"},
}

# Frame extraction presets
EXTRACT_PRESETS = {
    "photogrammetry": "2 fps, high quality JPEG",
    "photogrammetry-hq": "2 fps, PNG lossless",
    "3dgs": "Scene detection + 2fps cap",
    "scene": "Scene detection only",
    "sparse": "1 fps, for long/fast videos",
    "dense": "5 fps, detailed coverage",
    "timelapse": "1 frame every 5 seconds",
}


class ReframingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("360 Reframing Pipeline")
        self.geometry("1150x900")
        self.minsize(1000, 750)

        self.log_queue = queue.Queue()
        self.running_process = None
        self.is_running = False

        self._create_widgets()
        self.process_log_queue()
        self._setup_console_redirect()

        self.log("360 Reframing Pipeline initialized")
        self.log("Workflow: Extract Frames -> Filter Blurry -> Reframe to Perspectives")

    def _create_widgets(self):
        # Tab view at top
        self.tab_view = ctk.CTkTabview(master=self)
        self.tab_view.pack(fill="both", expand=True, padx=15, pady=(15, 5))

        self.tab_view.add("1. Extract Frames")
        self.tab_view.add("2. Filter Blurry")
        self.tab_view.add("3. Reframe")

        # Build each tab
        self._build_extract_tab()
        self._build_filter_tab()
        self._build_reframe_tab()

        # Bottom (progress + log)
        bottom_frame = ctk.CTkFrame(master=self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=15, pady=(5, 15))

        self.progress_bar = ctk.CTkProgressBar(master=bottom_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=(0, 10))

        self.log_textbox = ctk.CTkTextbox(master=bottom_frame, height=140)
        self.log_textbox.pack(fill="x")

    # =========================================================================
    # TAB 1: Extract Frames
    # =========================================================================
    def _build_extract_tab(self):
        tab = self.tab_view.tab("1. Extract Frames")

        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left pane - controls
        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        # Right pane - info
        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # === LEFT: Controls ===
        self._section_header(left, "Input Video")

        # Video file
        vid_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        vid_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(vid_frame, text="Video File:", width=100, anchor="w").pack(side="left")
        self.extract_video_entry = ctk.CTkEntry(vid_frame, placeholder_text="360 video file...")
        self.extract_video_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(vid_frame, text="...", width=40,
                      command=self._browse_video).pack(side="left")

        # Output folder
        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output Folder:", width=100, anchor="w").pack(side="left")
        self.extract_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Frames output...")
        self.extract_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.extract_output_entry)).pack(side="left")

        self._section_header(left, "Extraction Settings")

        # Preset
        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=100, anchor="w").pack(side="left")
        self.extract_preset_var = ctk.StringVar(value="photogrammetry")
        ctk.CTkComboBox(preset_frame, variable=self.extract_preset_var,
                        values=list(EXTRACT_PRESETS.keys()), state="readonly",
                        command=self._on_extract_preset_change).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.extract_preset_desc = ctk.CTkLabel(left, text="2 fps, high quality JPEG", text_color="gray")
        self.extract_preset_desc.pack(pady=(0, 5), padx=20, anchor="w")

        # Custom FPS (optional)
        fps_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        fps_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(fps_frame, text="Custom FPS:", width=100, anchor="w").pack(side="left")
        self.extract_fps_entry = ctk.CTkEntry(fps_frame, width=80, placeholder_text="(optional)")
        self.extract_fps_entry.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(fps_frame, text="overrides preset", text_color="gray").pack(side="left", padx=(10, 0))

        # Time range
        time_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        time_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(time_frame, text="Time Range:", width=100, anchor="w").pack(side="left")
        self.extract_start_entry = ctk.CTkEntry(time_frame, width=80, placeholder_text="Start")
        self.extract_start_entry.pack(side="left", padx=(10, 5))
        ctk.CTkLabel(time_frame, text="to").pack(side="left")
        self.extract_end_entry = ctk.CTkEntry(time_frame, width=80, placeholder_text="End")
        self.extract_end_entry.pack(side="left", padx=(5, 0))

        ctk.CTkLabel(left, text="Format: HH:MM:SS or seconds", text_color="gray").pack(pady=(0, 5), padx=20, anchor="w")

        # Quality
        qual_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        qual_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(qual_frame, text="JPEG Quality:", width=100, anchor="w").pack(side="left")
        self.extract_quality_var = ctk.StringVar(value="2")
        ctk.CTkComboBox(qual_frame, variable=self.extract_quality_var,
                        values=["1", "2", "3", "4", "5", "10", "15"],
                        width=80).pack(side="left", padx=(10, 0))
        ctk.CTkLabel(qual_frame, text="1=best, 31=worst", text_color="gray").pack(side="left", padx=(10, 0))

        # Dry run
        self.extract_dry_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Dry run (show command only)",
                        variable=self.extract_dry_var).pack(pady=5, padx=20, anchor="w")

        # Run button
        self.extract_run_btn = ctk.CTkButton(
            left, text="Extract Frames", command=self.run_extract,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.extract_run_btn.pack(pady=20, padx=10, fill="x")

        self.extract_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_process,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # === RIGHT: Info ===
        ctk.CTkLabel(right, text="Extraction Presets",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """photogrammetry: 2 fps, high quality JPEG
  - Good for walking-speed capture

photogrammetry-hq: 2 fps, PNG lossless
  - Maximum quality, larger files

3dgs: Scene detection + fps cap
  - For 3D Gaussian Splatting
  - Captures key changes

scene: Scene detection only
  - Captures key moments
  - Variable frame count

sparse: 1 fps
  - For long videos or fast motion

dense: 5 fps
  - For slow capture or detail

Requires: ffmpeg in PATH"""

        info_label = ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw")
        info_label.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _on_extract_preset_change(self, preset):
        if preset in EXTRACT_PRESETS:
            self.extract_preset_desc.configure(text=EXTRACT_PRESETS[preset])

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv"), ("All Files", "*.*")]
        )
        if path:
            self.extract_video_entry.delete(0, 'end')
            self.extract_video_entry.insert(0, path)
            self.log(f"Selected video: {Path(path).name}")

    def run_extract(self):
        video = self.extract_video_entry.get()
        output = self.extract_output_entry.get()

        if not video:
            self.log("Error: Please select a video file")
            return
        if not Path(video).exists():
            self.log(f"Error: Video not found: {video}")
            return
        if not output:
            self.log("Error: Please select an output folder")
            return

        script_path = Path(__file__).parent / "extract_frames.py"
        if not script_path.exists():
            self.log(f"Error: Script not found: {script_path}")
            return

        cmd = [
            sys.executable, str(script_path),
            video, output,
            "--preset", self.extract_preset_var.get(),
            "--quality", self.extract_quality_var.get(),
        ]

        # Custom FPS
        fps = self.extract_fps_entry.get().strip()
        if fps:
            cmd.extend(["--fps", fps])

        # Time range
        start = self.extract_start_entry.get().strip()
        end = self.extract_end_entry.get().strip()
        if start:
            cmd.extend(["--start", start])
        if end:
            cmd.extend(["--end", end])

        if self.extract_dry_var.get():
            cmd.append("--dry-run")

        self._run_command(cmd, self.extract_run_btn, self.extract_stop_btn)

    # =========================================================================
    # TAB 2: Filter Blurry
    # =========================================================================
    def _build_filter_tab(self):
        tab = self.tab_view.tab("2. Filter Blurry")

        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # === LEFT: Controls ===
        self._section_header(left, "Input / Output")

        # Input folder
        in_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        in_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(in_frame, text="Input Folder:", width=100, anchor="w").pack(side="left")
        self.filter_input_entry = ctk.CTkEntry(in_frame, placeholder_text="Frames to filter...")
        self.filter_input_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(in_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.filter_input_entry)).pack(side="left")

        # Output folder
        out_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        out_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(out_frame, text="Output Folder:", width=100, anchor="w").pack(side="left")
        self.filter_output_entry = ctk.CTkEntry(out_frame, placeholder_text="Sharp frames output...")
        self.filter_output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(out_frame, text="...", width=40,
                      command=lambda: self._browse_folder_for(self.filter_output_entry)).pack(side="left")

        self._section_header(left, "Filter Method")

        # Method selection
        self.filter_method_var = ctk.StringVar(value="percentile")

        ctk.CTkRadioButton(left, text="Keep top percentile", variable=self.filter_method_var,
                           value="percentile").pack(pady=5, padx=20, anchor="w")

        pct_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        pct_frame.pack(pady=2, padx=40, fill="x")
        self.filter_pct_var = ctk.DoubleVar(value=80)
        self.filter_pct_label = ctk.CTkLabel(pct_frame, text="80%", width=50)
        self.filter_pct_label.pack(side="right")
        ctk.CTkSlider(pct_frame, from_=50, to=99, variable=self.filter_pct_var,
                      command=lambda v: self.filter_pct_label.configure(text=f"{int(float(v))}%")
                      ).pack(side="left", fill="x", expand=True)

        ctk.CTkRadioButton(left, text="Keep top N images", variable=self.filter_method_var,
                           value="top_n").pack(pady=5, padx=20, anchor="w")

        topn_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        topn_frame.pack(pady=2, padx=40, fill="x")
        self.filter_topn_entry = ctk.CTkEntry(topn_frame, width=80, placeholder_text="100")
        self.filter_topn_entry.pack(side="left")
        ctk.CTkLabel(topn_frame, text="images", text_color="gray").pack(side="left", padx=(10, 0))

        ctk.CTkRadioButton(left, text="Score threshold", variable=self.filter_method_var,
                           value="threshold").pack(pady=5, padx=20, anchor="w")

        thresh_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        thresh_frame.pack(pady=2, padx=40, fill="x")
        self.filter_thresh_entry = ctk.CTkEntry(thresh_frame, width=80, placeholder_text="500")
        self.filter_thresh_entry.pack(side="left")
        ctk.CTkLabel(thresh_frame, text="Laplacian variance", text_color="gray").pack(side="left", padx=(10, 0))

        self._section_header(left, "Options")

        # Workers
        workers_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        workers_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(workers_frame, text="Workers:", width=100, anchor="w").pack(side="left")
        self.filter_workers_label = ctk.CTkLabel(workers_frame, text="4", width=50)
        self.filter_workers_label.pack(side="right")
        self.filter_workers_var = ctk.DoubleVar(value=4)
        ctk.CTkSlider(workers_frame, from_=1, to=16, variable=self.filter_workers_var,
                      command=lambda v: self.filter_workers_label.configure(text=f"{int(float(v))}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self.filter_dry_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Dry run (analyze only, don't copy)",
                        variable=self.filter_dry_var).pack(pady=5, padx=20, anchor="w")

        self.filter_verbose_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Verbose output",
                        variable=self.filter_verbose_var).pack(pady=5, padx=20, anchor="w")

        # Run button
        self.filter_run_btn = ctk.CTkButton(
            left, text="Filter Blurry Frames", command=self.run_filter,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.filter_run_btn.pack(pady=20, padx=10, fill="x")

        self.filter_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_process,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # === RIGHT: Info ===
        ctk.CTkLabel(right, text="Blur Detection",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        info_text = """Uses Laplacian variance to detect blur.

Higher score = sharper image
Lower score = blurrier image

Recommended settings:
• Percentile 80% - keeps sharpest 80%
• Top N - for fixed output count
• Threshold 500 - typical starting point
  (vary based on your images)

Workflow:
1. Run with "Dry run" first to see score range
2. Adjust threshold based on results
3. Run again to copy sharp frames

Tip: Use "Verbose" to see top/bottom 5 images
and score statistics."""

        ctk.CTkLabel(right, text=info_text, justify="left", anchor="nw"
                     ).pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def run_filter(self):
        input_folder = self.filter_input_entry.get()
        output_folder = self.filter_output_entry.get()

        if not input_folder:
            self.log("Error: Please select input folder")
            return
        if not Path(input_folder).exists():
            self.log(f"Error: Input folder not found: {input_folder}")
            return
        if not self.filter_dry_var.get() and not output_folder:
            self.log("Error: Please select output folder (or enable dry run)")
            return

        script_path = Path(__file__).parent / "filter_blurry.py"
        if not script_path.exists():
            self.log(f"Error: Script not found: {script_path}")
            return

        cmd = [
            sys.executable, str(script_path),
            input_folder,
        ]

        if output_folder:
            cmd.append(output_folder)

        # Filter method
        method = self.filter_method_var.get()
        if method == "percentile":
            cmd.extend(["--percentile", str(int(self.filter_pct_var.get()))])
        elif method == "top_n":
            topn = self.filter_topn_entry.get().strip()
            if topn:
                cmd.extend(["--top", topn])
        elif method == "threshold":
            thresh = self.filter_thresh_entry.get().strip()
            if thresh:
                cmd.extend(["--threshold", thresh])

        cmd.extend(["--workers", str(int(self.filter_workers_var.get()))])

        if self.filter_dry_var.get():
            cmd.append("--dry-run")
        if self.filter_verbose_var.get():
            cmd.append("--verbose")

        self._run_command(cmd, self.filter_run_btn, self.filter_stop_btn)

    # =========================================================================
    # TAB 3: Reframe
    # =========================================================================
    def _build_reframe_tab(self):
        tab = self.tab_view.tab("3. Reframe")

        container = ctk.CTkFrame(master=tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(master=container, fg_color="transparent", width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        right = ctk.CTkFrame(master=container)
        right.pack(side="left", fill="both", expand=True)

        # === LEFT: Controls ===
        self._section_header(left, "Input / Output")

        self.reframe_images_entry = self._folder_row_in(left, "Images Folder:", "Equirectangular images...")
        self.reframe_output_entry = self._folder_row_in(left, "Output Folder:", "COLMAP-ready output...")
        self.reframe_masks_entry = self._folder_row_in(left, "Masks Folder:", "(Optional) Binary masks...")
        self.reframe_xml_entry = self._file_row_in(left, "Metashape XML:", "(Optional) Camera poses...")

        self._section_header(left, "View Settings")

        # Preset
        preset_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        preset_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(preset_frame, text="Preset:", width=100, anchor="w").pack(side="left")
        self.reframe_preset_var = ctk.StringVar(value="10-octa")
        ctk.CTkComboBox(preset_frame, variable=self.reframe_preset_var,
                        values=list(VIEW_PRESETS.keys()), state="readonly",
                        command=self._on_reframe_preset_change).pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.reframe_preset_desc = ctk.CTkLabel(left, text="", text_color="gray")
        self.reframe_preset_desc.pack(pady=(0, 5), padx=20, anchor="w")

        # FOV
        fov_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        fov_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(fov_frame, text="Field of View:", width=100, anchor="w").pack(side="left")
        self.reframe_fov_label = ctk.CTkLabel(fov_frame, text="65.0°", width=50)
        self.reframe_fov_label.pack(side="right")
        self.reframe_fov_var = ctk.DoubleVar(value=65)
        ctk.CTkSlider(fov_frame, from_=50, to=90, variable=self.reframe_fov_var,
                      command=self._on_reframe_fov_change).pack(side="left", fill="x", expand=True, padx=(10, 10))

        self.reframe_fov_rec = ctk.CTkLabel(left, text="", text_color="gray")
        self.reframe_fov_rec.pack(pady=(0, 5), padx=20, anchor="w")

        # Crop size
        crop_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        crop_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(crop_frame, text="Crop Size:", width=100, anchor="w").pack(side="left")
        self.reframe_crop_var = ctk.StringVar(value="1920")
        ctk.CTkComboBox(crop_frame, variable=self.reframe_crop_var,
                        values=["512", "1024", "1280", "1536", "1920", "2048", "2560", "3840"],
                        state="readonly").pack(side="left", fill="x", expand=True, padx=(10, 0))

        self._section_header(left, "Processing Options")

        # Yaw offset
        yaw_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        yaw_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(yaw_frame, text="Yaw Offset:", width=100, anchor="w").pack(side="left")
        self.reframe_yaw_label = ctk.CTkLabel(yaw_frame, text="0°/frame", width=60)
        self.reframe_yaw_label.pack(side="right")
        self.reframe_yaw_var = ctk.DoubleVar(value=0)
        ctk.CTkSlider(yaw_frame, from_=0, to=30, variable=self.reframe_yaw_var,
                      command=lambda v: self.reframe_yaw_label.configure(text=f"{int(float(v))}°/frame")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Workers
        workers_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        workers_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(workers_frame, text="Workers:", width=100, anchor="w").pack(side="left")
        self.reframe_workers_label = ctk.CTkLabel(workers_frame, text="4", width=50)
        self.reframe_workers_label.pack(side="right")
        self.reframe_workers_var = ctk.DoubleVar(value=4)
        ctk.CTkSlider(workers_frame, from_=1, to=16, variable=self.reframe_workers_var,
                      command=lambda v: self.reframe_workers_label.configure(text=f"{int(float(v))}")
                      ).pack(side="left", fill="x", expand=True, padx=(10, 10))

        # Max images
        max_frame = ctk.CTkFrame(master=left, fg_color="transparent")
        max_frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(max_frame, text="Max Images:", width=100, anchor="w").pack(side="left")
        self.reframe_max_entry = ctk.CTkEntry(max_frame, width=80)
        self.reframe_max_entry.insert(0, "10000")
        self.reframe_max_entry.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(max_frame, text="(0 = all)", text_color="gray").pack(side="left", padx=(10, 0))

        self._section_header(left, "Output Options")

        self.reframe_skip_bottom_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Skip bottom view (tripod/operator)",
                        variable=self.reframe_skip_bottom_var).pack(pady=5, padx=20, anchor="w")

        self.reframe_rig_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Generate COLMAP rig config",
                        variable=self.reframe_rig_var).pack(pady=5, padx=20, anchor="w")

        # Run button
        self.reframe_run_btn = ctk.CTkButton(
            left, text="Run Reframing", command=self.run_reframe,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=14, weight="bold"), height=45
        )
        self.reframe_run_btn.pack(pady=20, padx=10, fill="x")

        self.reframe_stop_btn = ctk.CTkButton(
            left, text="Stop", command=self.stop_process,
            fg_color="#C62828", hover_color="#8B0000", height=35
        )

        # === RIGHT: Preview & Info ===
        ctk.CTkLabel(right, text="Preview & Info",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 10), padx=15, anchor="w")

        self.reframe_preview = ctk.CTkLabel(right, text="Select images folder\nto preview",
                                             fg_color="#1a1a1a", corner_radius=8)
        self.reframe_preview.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        info_frame = ctk.CTkFrame(master=right)
        info_frame.pack(fill="x", padx=15, pady=(0, 15))
        self.reframe_info = ctk.CTkLabel(info_frame, text="Overlap: --\nOutput images: --",
                                          justify="left", anchor="w")
        self.reframe_info.pack(pady=10, padx=15, anchor="w")

        # Initialize preset info
        self._on_reframe_preset_change("10-octa")

    def _folder_row_in(self, parent, label, placeholder):
        frame = ctk.CTkFrame(master=parent, fg_color="transparent")
        frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(frame, text=label, width=100, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(frame, placeholder_text=placeholder)
        entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(frame, text="...", width=40,
                      command=lambda e=entry: self._browse_folder_for(e, preview=(e == self.reframe_images_entry if hasattr(self, 'reframe_images_entry') else False))).pack(side="left")
        return entry

    def _file_row_in(self, parent, label, placeholder):
        frame = ctk.CTkFrame(master=parent, fg_color="transparent")
        frame.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(frame, text=label, width=100, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(frame, placeholder_text=placeholder)
        entry.pack(side="left", fill="x", expand=True, padx=(10, 5))
        ctk.CTkButton(frame, text="...", width=40,
                      command=lambda e=entry: self._browse_file_for(e)).pack(side="left")
        return entry

    def _on_reframe_preset_change(self, preset):
        if preset in VIEW_PRESETS:
            info = VIEW_PRESETS[preset]
            self.reframe_preset_desc.configure(text=info["desc"])
            fov_min, fov_max = info["fov_range"]
            self.reframe_fov_rec.configure(text=f"Recommended: {fov_min}-{fov_max}°")
        self._update_reframe_info()

    def _on_reframe_fov_change(self, val):
        self.reframe_fov_label.configure(text=f"{float(val):.1f}°")
        self._update_reframe_info()

    def _update_reframe_info(self):
        preset = self.reframe_preset_var.get()
        if preset not in VIEW_PRESETS:
            return
        info = VIEW_PRESETS[preset]
        fov = self.reframe_fov_var.get()
        spacing = info["spacing"]
        overlap_deg = max(0, fov - spacing)
        overlap_pct = (overlap_deg / fov * 100) if fov > 0 else 0

        num_views = info["views"]
        if self.reframe_skip_bottom_var.get():
            num_views -= 1

        images_folder = self.reframe_images_entry.get()
        input_count = 0
        if images_folder and Path(images_folder).exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                input_count += len(list(Path(images_folder).glob(ext)))

        output_count = input_count * num_views if input_count > 0 else "?"

        self.reframe_info.configure(
            text=f"Horizontal overlap: {overlap_deg:.1f}° ({overlap_pct:.0f}%)\n"
                 f"Views per image: {num_views}\n"
                 f"Input images: {input_count if input_count > 0 else '?'}\n"
                 f"Output images: {output_count}"
        )

    def run_reframe(self):
        images = self.reframe_images_entry.get()
        output = self.reframe_output_entry.get()

        if not images:
            self.log("Error: Please select images folder")
            return
        if not Path(images).exists():
            self.log(f"Error: Images folder not found: {images}")
            return
        if not output:
            self.log("Error: Please select output folder")
            return

        script_path = Path(__file__).parent / "equirect_to_perspectives.py"
        if not script_path.exists():
            self.log(f"Error: Script not found: {script_path}")
            return

        cmd = [
            sys.executable, str(script_path),
            "--images", images,
            "--output", output,
            "--preset", self.reframe_preset_var.get(),
            "--fov-deg", str(self.reframe_fov_var.get()),
            "--crop-size", self.reframe_crop_var.get(),
            "--yaw-offset", str(self.reframe_yaw_var.get()),
            "--num-workers", str(int(self.reframe_workers_var.get())),
        ]

        try:
            max_img = int(self.reframe_max_entry.get())
            if max_img > 0:
                cmd.extend(["--max-images", str(max_img)])
        except:
            pass

        masks = self.reframe_masks_entry.get()
        if masks and Path(masks).exists():
            cmd.extend(["--masks", masks])

        xml = self.reframe_xml_entry.get()
        if xml and Path(xml).exists():
            cmd.extend(["--xml", xml])

        if self.reframe_skip_bottom_var.get():
            cmd.append("--skip-bottom")
        if self.reframe_rig_var.get():
            cmd.append("--generate-rig-config")

        self._run_command(cmd, self.reframe_run_btn, self.reframe_stop_btn)

    # =========================================================================
    # Shared Helpers
    # =========================================================================
    def _section_header(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(pady=(15, 5), padx=10, anchor="w")

    def _browse_folder_for(self, entry, preview=False):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            entry.delete(0, 'end')
            entry.insert(0, folder)
            self.log(f"Selected: {folder}")
            if preview and hasattr(self, 'reframe_images_entry') and entry == self.reframe_images_entry:
                self._preview_reframe_sample(folder)
                self._update_reframe_info()

    def _browse_file_for(self, entry):
        path = filedialog.askopenfilename(title="Select File",
                                           filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")])
        if path:
            entry.delete(0, 'end')
            entry.insert(0, path)
            self.log(f"Selected: {path}")

    def _preview_reframe_sample(self, folder):
        if not HAS_PIL:
            return
        folder_path = Path(folder)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            files = list(folder_path.glob(ext))
            if files:
                try:
                    img = Image.open(files[0])
                    img.thumbnail((500, 350), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.reframe_preview.configure(image=photo, text="")
                    self.reframe_preview.image = photo
                    self.log(f"Preview: {files[0].name} ({len(files)} images)")
                    return
                except Exception as e:
                    self.log(f"Preview error: {e}")
                    return
        self.reframe_preview.configure(image=None, text="No images found")

    def _run_command(self, cmd, run_btn, stop_btn):
        """Run a command in a worker thread."""
        self.is_running = True
        run_btn.configure(state="disabled")
        stop_btn.pack(pady=(0, 10), padx=10, fill="x")
        threading.Thread(target=self._worker, args=(cmd, run_btn, stop_btn), daemon=True).start()

    def _worker(self, cmd, run_btn, stop_btn):
        self.progress_bar.start()
        self.log(f"Running: {' '.join(cmd)}")
        self.log("-" * 50)

        try:
            self.running_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            for line in self.running_process.stdout:
                if not self.is_running:
                    break
                self.log(line.rstrip())

            code = self.running_process.wait()
            self.log("-" * 50)
            if code == 0:
                self.log("Process completed successfully")
            elif self.is_running:
                self.log(f"Process failed with code {code}")
            else:
                self.log("Process stopped by user")

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.running_process = None
            self.is_running = False
            self.progress_bar.stop()
            self.progress_bar.set(1.0)
            self.after(0, lambda: self._on_done(run_btn, stop_btn))

    def _on_done(self, run_btn, stop_btn):
        run_btn.configure(state="normal")
        stop_btn.pack_forget()

    def stop_process(self):
        if self.running_process:
            self.log("Stopping process...")
            self.is_running = False
            try:
                self.running_process.terminate()
            except:
                pass

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


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ReframingApp()
    app.mainloop()

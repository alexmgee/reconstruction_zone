#!/usr/bin/env python3
"""
Reconstruction Zone
===================
Unified photogrammetry prep GUI. Double-click to launch.

Tabs:
  Projects — central registry for photogrammetry projects, lifecycle tracking
  Prepare  — video analysis, frame extraction (queue), reframe, fisheye, LUT, filters
  Mask     — multi-model masking pipeline (YOLO, SAM, shadow, ensemble)
  Review   — paginated thumbnail grid, large preview, OpenCV editor launch
  Gaps     — spatial gap detection, bridge frame extraction

No CLI arguments required. All configuration via the GUI.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, List, Dict
import threading
import queue
import sys
import os
import json
import logging
import shutil

# Ensure reconstruction_gui and prep360 are importable
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# When launched via pythonw.exe, sys.stderr/stdout are None — redirect to
# devnull so third-party libraries that write to stderr/stdout don't crash
if sys.stderr is None:
    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    sys.stdout = _devnull
    logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- lazy imports ----------

def _import_pipeline():
    from reconstruction_pipeline import (
        MaskingPipeline, MaskConfig, SegmentationModel,
        ImageGeometry, CLASS_PRESETS, COCO_CLASSES,
    )
    return MaskingPipeline, MaskConfig, SegmentationModel, ImageGeometry, CLASS_PRESETS, COCO_CLASSES


def _import_review():
    from review_gui import (
        load_overlay_thumbnail, compute_mask_area_percent,
        QUALITY_COLORS, STATUS_COLORS,
    )
    from review_status import ReviewStatusManager, MaskStatus
    return (load_overlay_thumbnail, compute_mask_area_percent,
            QUALITY_COLORS, STATUS_COLORS,
            ReviewStatusManager, MaskStatus)


# ---------- mask targets ----------
# label → COCO class ID (or None for prompt-only targets)
MASK_TARGETS = [
    ("person", 0),
    ("backpack", 24),
    ("car", 2),
    ("camera", None),
    ("selfie stick", None),
]


# ──────────────────────────────────────────────────────────────────────
# Shared widgets & infrastructure (extracted to separate modules)
# ──────────────────────────────────────────────────────────────────────

from widgets import Section as _Section, CollapsibleSection as _CollapsibleSection, slider_row, Tooltip
from app_infra import AppInfrastructure
from tabs.source_tab import build_source_tab
from tabs.gaps_tab import build_gaps_tab
from tabs.projects_tab import build_projects_tab


# ──────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────

class ReconstructionZone(AppInfrastructure, ctk.CTk):
    """Unified photogrammetry prep: prepare → mask → review → gaps."""

    _PREFS_FILE = _this_dir / ".studio_prefs.json"

    def __init__(self):
        super().__init__()
        self.title("Reconstruction Zone")
        self.geometry("2200x1200")
        self.minsize(1200, 800)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.log_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.cancel_flag = threading.Event()
        self._prefs = self._load_prefs()
        self._preview_mode = "process"  # "process" or "review"

        # Review state (lazy)
        self._review_loaded = False
        self._review_pairs: List[Dict] = []
        self._review_status_mgr = None
        self._selected_stem: Optional[str] = None
        self._filtered_pairs: List[Dict] = []
        self._thumb_cache: Dict[str, Any] = {}  # stem → PIL.Image
        self._thumb_widgets: Dict[str, ctk.CTkFrame] = {}  # stem → cell widget
        self._thumb_cell_count = 0
        self._nav_debounce_id = None  # after() ID for slider debounce
        # Persistent OpenCV editor state (subprocess-based)
        self._editor_proc = None
        self._editor_cmd_file = None
        self._editor_signal_file = None
        self._editor_poll_id = None

        self._build_ui()
        self._init_infrastructure()  # logging + console redirect (from AppInfrastructure)
        # CTkImage size is in logical pixels; winfo_width() returns physical pixels.
        # Divide by DPI scale so images fit the panel correctly on high-DPI displays.
        self._dpi_scale = ctk.ScalingTracker.get_widget_scaling(self)
        self._restore_prefs()
        self._check_external_tools()

        # Check for missing model weights on first launch
        from reconstruction_gui.model_downloader import check_missing_models, ModelDownloadDialog
        missing = check_missing_models()
        if missing:
            dialog = ModelDownloadDialog(self, missing)
            self.wait_window(dialog)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _check_external_tools(self):
        """Warn at startup if external CLI tools are missing from PATH."""
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found on PATH — video extraction will not work")
        if not shutil.which("exiftool"):
            logger.warning("exiftool not found on PATH — SRT geotagging will not work")

    # ── prefs ──
    # _load_prefs and _save_prefs inherited from AppInfrastructure

    def _restore_prefs(self):
        for key, entry in [("input_dir", self.input_entry), ("output_dir", self.output_entry),
                           ("masks_dir", self.masks_entry), ("images_dir", self.images_entry)]:
            val = self._prefs.get(key, "")
            if val:
                entry.delete(0, "end")
                entry.insert(0, val)
        # Restore prompt fields
        if self._prefs.get("remove_prompts"):
            self.remove_prompts_entry.delete(0, "end")
            self.remove_prompts_entry.insert(0, self._prefs["remove_prompts"])
        if self._prefs.get("keep_prompts"):
            self.keep_prompts_entry.delete(0, "end")
            self.keep_prompts_entry.insert(0, self._prefs["keep_prompts"])
        if self._prefs.get("shadow_detector"):
            self.shadow_detector_var.set(self._prefs["shadow_detector"])
        if self._prefs.get("shadow_verifier"):
            self.shadow_verifier_var.set(self._prefs["shadow_verifier"])
        if self._prefs.get("shadow_spatial"):
            self.shadow_spatial_var.set(self._prefs["shadow_spatial"])
        if "save_review_folder" in self._prefs:
            self.review_folder_var.set(bool(self._prefs["save_review_folder"]))
        if "save_reject_review_images" in self._prefs:
            self.review_rejects_var.set(bool(self._prefs["save_reject_review_images"]))
        # Restore preview state
        if self._prefs.get("preview_image"):
            self._preview_image_entry.delete(0, "end")
            self._preview_image_entry.insert(0, self._prefs["preview_image"])
        if self._prefs.get("preview_visible") is False:
            self._toggle_preview_panel()
        # Load image navigator if input is set
        if self._prefs.get("input_dir"):
            self.after(200, self._load_image_list)

    def _on_close(self):
        """Fast shutdown — kill subprocesses, clear caches, destroy widgets."""
        # Persist tracker store path
        if hasattr(self, '_project_store') and self._project_store:
            self._prefs["tracker_store_path"] = str(self._project_store.store_path)
            self._save_prefs()
        # Signal any running extraction/processing threads to stop
        self.cancel_flag.set()
        if self._editor_proc is not None and self._editor_proc.poll() is None:
            self._editor_proc.terminate()
        self._thumb_cache.clear()
        self._thumb_widgets.clear()
        self._preview_overlay_pil = None
        self._preview_mask_pil = None
        self.destroy()

    # log(), _poll_log_queue(), _setup_console_redirect() inherited from AppInfrastructure

    # ── UI construction ──

    def _build_ui(self):
        # Main 2-column grid: left tabview + right shared preview
        self._main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._main_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._main_frame.grid_columnconfigure(0, weight=35, uniform="split")
        self._main_frame.grid_columnconfigure(1, weight=65, uniform="split")
        self._main_frame.grid_rowconfigure(0, weight=1)

        # Left: tabview (settings tabs only)
        self.tabs = ctk.CTkTabview(self._main_frame, command=self._on_tab_change)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        self.tabs.add("Projects")
        self.tabs.add("Extract")
        self.tabs.add("Mask")
        self.tabs.add("Review")
        self.tabs.add("Coverage")

        # Right: shared preview panel (collapsible)
        self._preview_visible = True
        self._preview_panel = ctk.CTkFrame(self._main_frame)
        self._preview_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self._build_preview_panel()

        build_projects_tab(self, self.tabs.tab("Projects"))
        build_source_tab(self, self.tabs.tab("Extract"))
        self._build_process_tab()
        self._build_review_tab()
        build_gaps_tab(self, self.tabs.tab("Coverage"))

        # Projects is the default tab — swap preview for detail panel
        self.after(50, self._on_tab_change)

    # ── slider helper ──

    def _slider(self, parent, label, from_, to, default, steps,
                fmt=".2f", width=100, pad_left=0, tooltip=None):
        """Delegate to shared slider_row (from widgets.py)."""
        return slider_row(parent, label, from_, to, default,
                          steps=steps, fmt=fmt, width=width, pad_left=pad_left,
                          tooltip=tooltip)

    # ══════════════════════════════════════════════════════════════════
    # PROCESS TAB
    # ══════════════════════════════════════════════════════════════════

    def _build_process_tab(self):
        tab = self.tabs.tab("Mask")

        # Scrollable settings (preview panel is shared, outside tabs)
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # ==============================================================
        #  CORE PROCESS
        # ==============================================================
        core_sec = _CollapsibleSection(scroll, "Core Process", expanded=True)
        core_sec.pack(fill="x", pady=(0, 6), padx=4)
        core = core_sec.content

        # Input / Output
        row = ctk.CTkFrame(core, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(row, text="Input:", width=55, anchor="e").pack(side="left")
        self.input_entry = ctk.CTkEntry(row, placeholder_text="Folder, image, or video")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row, text="Folder", width=55, command=self._browse_input_folder).pack(side="left", padx=(0, 2))
        ctk.CTkButton(row, text="File", width=45, command=self._browse_input_file).pack(side="left")

        row2 = ctk.CTkFrame(core, fg_color="transparent")
        row2.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(row2, text="Output:", width=55, anchor="e").pack(side="left")
        self.output_entry = ctk.CTkEntry(row2, placeholder_text="Output folder for masks")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row2, text="Browse", width=65,
                      command=lambda: self._browse_dir_into(self.output_entry)).pack(side="left")

        row2b = ctk.CTkFrame(core, fg_color="transparent")
        row2b.pack(fill="x", padx=6, pady=(0, 3))
        ctk.CTkLabel(row2b, text="", width=55).pack(side="left")
        self.review_folder_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(
            row2b,
            text="Create review folder",
            variable=self.review_folder_var,
            width=150,
        )
        _w.pack(side="left", padx=(5, 8))
        Tooltip(_w, "Create separate masks/ and review/ subfolders inside output.\nOff = write mask files directly into the output folder.")
        self.review_rejects_var = ctk.BooleanVar(value=True)
        _w = ctk.CTkCheckBox(
            row2b,
            text="Include rejects",
            variable=self.review_rejects_var,
            width=130,
        )
        _w.pack(side="left", padx=(0, 8))
        Tooltip(_w, "Copy rejected masks to review/ folder for reference.\nOnly relevant when 'Create review folder' is enabled.")
        ctk.CTkLabel(
            row2b,
            text="Off: write masks directly into Output. On: create masks/ and review/ inside Output.",
            font=("Consolas", 10),
            text_color="#9ca3af",
        ).pack(side="left")

        # SAM3 mode: Hybrid / Unified Video  (above model row — mode determines what's relevant)
        sam3_mode_row = ctk.CTkFrame(core, fg_color="transparent")
        sam3_mode_row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(sam3_mode_row, text="SAM3 mode:").pack(side="left", padx=(0, 2))
        self.sam3_unified_var = ctk.BooleanVar(value=False)
        self._sam3_mode_btn = ctk.CTkSegmentedButton(
            sam3_mode_row, values=["Hybrid", "Unified Video"],
            command=self._on_sam3_mode_change,
        )
        self._sam3_mode_btn.set("Hybrid")
        self._sam3_mode_btn.pack(side="left", padx=4)
        ctk.CTkLabel(sam3_mode_row,
                     text="Hybrid = per-frame detect+segment; Unified = SAM 3.1 multiplex tracking",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Model / Geometry  (greyed out in Unified mode)
        r1 = ctk.CTkFrame(core, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=3)

        ctk.CTkLabel(r1, text="Model:").pack(side="left", padx=(0, 2))
        # Build model list based on distribution
        try:
            from prep360.distribution import is_gumroad as _is_gumroad_check
        except ImportError:
            def _is_gumroad_check(): return False
        _all_models = ["auto", "yolo26", "rfdetr", "sam3", "fastsam"]
        if _is_gumroad_check():
            _model_values = [m for m in _all_models if m not in ("yolo26", "fastsam")]
        else:
            _model_values = _all_models
        self.model_var = ctk.StringVar(value="auto")
        _w = ctk.CTkOptionMenu(r1, variable=self.model_var,
                               values=_model_values, width=95)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Detection model for object segmentation.\nauto = tries RF-DETR first, falls back to available models.")

        ctk.CTkLabel(r1, text="YOLO size:").pack(side="left", padx=(12, 2))
        self.yolo_size_var = ctk.StringVar(value="n")
        _w = ctk.CTkOptionMenu(r1, variable=self.yolo_size_var,
                               values=["n", "s", "m", "l", "x"], width=55)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "YOLO model size: n=fastest, x=most accurate.\nOnly matters when Model is set to yolo26.")

        ctk.CTkLabel(r1, text="Geometry:").pack(side="left", padx=(12, 2))
        self.geometry_var = ctk.StringVar(value="pinhole")
        _w = ctk.CTkOptionMenu(r1, variable=self.geometry_var,
                               values=["pinhole", "equirect", "fisheye", "cubemap"],
                               width=95)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Image projection type.\npinhole = standard photos/reframed views.\nequirect = 360° equirectangular (2:1 aspect).")

        self.conf_var = self._slider(r1, "Conf", 0, 1, 0.70, 20, width=80, pad_left=12,
                                     tooltip="Minimum detection confidence.\nLower = catches more objects but may include false positives.")

        # Prompts & Classes
        r2 = ctk.CTkFrame(core, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r2, text="Remove:", width=55, anchor="e").pack(side="left")
        self.remove_prompts_entry = ctk.CTkEntry(
            r2, placeholder_text="default: person, tripod, backpack, selfie stick")
        self.remove_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)
        Tooltip(self.remove_prompts_entry, "Objects to detect and mask out.\nComma-separated text prompts for SAM3/3.1, or COCO class names for YOLO.\nLeave empty for defaults: person, tripod, backpack, selfie stick.")

        r3 = ctk.CTkFrame(core, fg_color="transparent")
        r3.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r3, text="Keep:", width=55, anchor="e").pack(side="left")
        self.keep_prompts_entry = ctk.CTkEntry(
            r3, placeholder_text="(optional) objects to protect from masking")
        self.keep_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)
        Tooltip(self.keep_prompts_entry, "Objects to protect from masking.\nIf a detected region overlaps with a 'keep' object,\nit will be excluded from the mask.")

        # Multi-pass SAM3 prompts
        mp_wrapper = ctk.CTkFrame(core, fg_color="transparent")
        mp_wrapper.pack(fill="x", padx=6, pady=3)

        mp_toggle_row = ctk.CTkFrame(mp_wrapper, fg_color="transparent")
        mp_toggle_row.pack(fill="x")
        self.multi_pass_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(mp_toggle_row, text="Multi-pass SAM3",
                             variable=self.multi_pass_var,
                             command=self._toggle_multi_pass,
                             width=140)
        _w.pack(side="left")
        Tooltip(_w, "Run SAM3 multiple times with different prompts.\nEach pass generates masks that are merged (union).\nUseful for catching shadows or other secondary targets.\n(Uses SAM3 image model for per-frame passes.)")
        ctk.CTkLabel(mp_toggle_row, text="(union masks from multiple prompt passes)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)
        _w = ctk.CTkButton(mp_toggle_row, text="+ Add Pass", width=80,
                           command=self._add_multi_pass_row)
        _w.pack(side="right", padx=2)
        Tooltip(_w, "Add another prompt+confidence pair for multi-pass.")
        _w = ctk.CTkButton(mp_toggle_row, text="Remove", width=70,
                           command=self._remove_multi_pass_row)
        _w.pack(side="right", padx=2)
        Tooltip(_w, "Remove the last multi-pass prompt row.")

        self.multi_pass_entries = []
        self.multi_pass_list_frame = ctk.CTkFrame(mp_wrapper, fg_color="transparent")
        # Initially hidden — shown when checkbox is checked

        # Mask targets  (greyed out in Unified mode)
        targets_row = ctk.CTkFrame(core, fg_color="transparent")
        targets_row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(targets_row, text="Mask:", width=55, anchor="e").pack(side="left")
        self._target_vars: dict = {}
        for label, coco_id in MASK_TARGETS:
            var = ctk.BooleanVar(value=(label in ("person", "backpack", "selfie stick")))
            ctk.CTkCheckBox(targets_row, text=label, variable=var,
                            width=100).pack(side="left", padx=4)
            self._target_vars[label] = (var, coco_id)

        # Frames whose interactive children are disabled in Unified mode
        # targets_row stays enabled — checkboxes are the prompt source for Unified Video too
        self._unified_disable_frames = [r1]

        # File types, skip existing, review threshold
        opts_row = ctk.CTkFrame(core, fg_color="transparent")
        opts_row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(opts_row, text="File types:").pack(side="left", padx=(0, 2))
        self.pattern_var = ctk.StringVar(value="*.jpg *.png")
        _w = ctk.CTkEntry(opts_row, textvariable=self.pattern_var, width=80)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Glob patterns for input files.\nSpace-separated, e.g. '*.jpg *.png *.tif'")

        self.skip_existing_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(opts_row, text="Skip existing",
                             variable=self.skip_existing_var, width=110)
        _w.pack(side="left", padx=(12, 0))
        Tooltip(_w, "Skip images that already have a mask in the output folder.\nUseful for resuming interrupted batch runs.")

        self.review_thresh_var = self._slider(opts_row, "Review thresh", 0, 1, 0.85, 20,
                                              pad_left=12,
                                              tooltip="Quality score below this threshold flags masks for review.\nLower = fewer flags, higher = more masks flagged for manual check.")
        ctk.CTkLabel(opts_row, text="(below = flagged)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Output options: multi-label, inpaint
        out_row = ctk.CTkFrame(core, fg_color="transparent")
        out_row.pack(fill="x", padx=6, pady=3)
        self.multi_label_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(out_row, text="Multi-label output",
                             variable=self.multi_label_var, width=130)
        _w.pack(side="left")
        Tooltip(_w, "Output per-class segmentation maps instead of binary masks.\nEach detected class gets its own mask channel.")
        ctk.CTkLabel(out_row, text="(per-class segmaps)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(0, 12))
        self.inpaint_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(out_row, text="Inpaint masked",
                             variable=self.inpaint_var, width=120)
        _w.pack(side="left")
        Tooltip(_w, "Fill masked regions with plausible background.\nProduces clean images for 3DGS training alongside masks.")
        ctk.CTkLabel(out_row, text="(fill masked regions for 3DGS training)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # ==============================================================
        #  DETECTION & REFINEMENT
        # ==============================================================
        self._detect_sec = _CollapsibleSection(scroll, "Detection & Refinement", expanded=True)
        detect_sec = self._detect_sec
        detect_sec.pack(fill="x", pady=(0, 6), padx=4)
        detect = detect_sec.content

        # Shadow Detection
        shadow_section = _CollapsibleSection(detect, "Shadow Detection",
            subtitle="Detect shadows cast by masked objects")
        shadow_section.pack(fill="x", padx=2, pady=(0, 4))
        sc = shadow_section.content

        sr1 = ctk.CTkFrame(sc, fg_color="transparent")
        sr1.pack(fill="x", pady=2)
        self.shadow_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(sr1, text="Enable", variable=self.shadow_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Enable automatic shadow detection for masked objects.\nAdds detected shadow regions to the mask.")
        ctk.CTkLabel(sr1, text="Detector:").pack(side="left", padx=(12, 2))
        self.shadow_detector_var = ctk.StringVar(value="targeted_person")
        _w = ctk.CTkOptionMenu(sr1, variable=self.shadow_detector_var,
                               values=["targeted_person", "brightness",
                                       "c1c2c3", "hybrid"], width=140)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Shadow detection method.\ntargeted_person = optimized for human shadows.\nbrightness/c1c2c3/hybrid = general-purpose heuristic detectors.")
        ctk.CTkLabel(sr1, text="Verify:").pack(side="left", padx=(12, 2))
        self.shadow_verifier_var = ctk.StringVar(value="none")
        _w = ctk.CTkOptionMenu(sr1, variable=self.shadow_verifier_var,
                               values=["none", "c1c2c3", "hybrid", "brightness"],
                               width=100)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Optional second-pass verification of detected shadows.\nReduces false positives by cross-checking with a different method.")

        sr2 = ctk.CTkFrame(sc, fg_color="transparent")
        sr2.pack(fill="x", pady=2)
        ctk.CTkLabel(sr2, text="Spatial:").pack(side="left")
        self.shadow_spatial_var = ctk.StringVar(value="near_objects")
        _w = ctk.CTkSegmentedButton(sr2, values=["all", "near_objects", "connected"],
                                    variable=self.shadow_spatial_var)
        _w.pack(side="left", padx=4)
        self.shadow_dilation_var = self._slider(sr2, "Dilation", 0, 200, 50, 40,
                                               fmt=".0f", pad_left=8,
                                               tooltip="Expand the search area around detected objects (pixels).\nLarger values search further from the mask for shadows.")
        ctk.CTkLabel(sr2, text="px", font=("Consolas", 10),
                     text_color="#9ca3af").pack(side="left", padx=2)

        sr3 = ctk.CTkFrame(sc, fg_color="transparent")
        sr3.pack(fill="x", pady=2)
        self.shadow_conf_var = self._slider(sr3, "Confidence", 0, 1, 0.50, 20,
                                            tooltip="Minimum confidence for shadow detection.\nLower = more aggressive shadow catching.")
        self.shadow_darkness_var = self._slider(sr3, "Darkness", 0, 1, 0.70, 20, pad_left=8,
                                               tooltip="How dark a region must be to qualify as shadow.\nLower = catches lighter shadows. Higher = only deep shadows.")
        sr4 = ctk.CTkFrame(sc, fg_color="transparent")
        sr4.pack(fill="x", pady=2)
        self.shadow_chroma_var = self._slider(sr4, "Chromaticity", 0, 0.5, 0.15, 25,
                                             tooltip="Maximum color saturation for shadow regions.\nShadows are typically low-saturation. Higher = more permissive.")

        # SAM Mask Refinement
        sam_section = _CollapsibleSection(detect, "SAM Mask Refinement",
            subtitle="Tighten mask edges using SAM point prompts")
        sam_section.pack(fill="x", padx=2, pady=(0, 4))
        samc = sam_section.content

        samr1 = ctk.CTkFrame(samc, fg_color="transparent")
        samr1.pack(fill="x", pady=2)
        self.sam_refine_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(samr1, text="Enable", variable=self.sam_refine_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Refine mask boundaries using SAM point prompts.\nTightens edges where the initial detection was rough.")
        ctk.CTkLabel(samr1, text="SAM model:").pack(side="left", padx=(12, 2))
        self.sam_model_var = ctk.StringVar(value="vit_b")
        _w = ctk.CTkOptionMenu(samr1, variable=self.sam_model_var,
                               values=["vit_b", "vit_l", "vit_h"], width=90)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "SAM model size for refinement.\nvit_b = 375MB, fastest. vit_h = most accurate but slowest.")
        ctk.CTkLabel(samr1, text="(vit_b=375MB, fastest)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        samr2 = ctk.CTkFrame(samc, fg_color="transparent")
        samr2.pack(fill="x", pady=2)
        self.sam_margin_var = self._slider(samr2, "Box margin", 0, 0.5, 0.15, 10,
                                           tooltip="Padding around detected object for SAM refinement.\nLarger margin gives SAM more context but may include background.")
        self.sam_iou_var = self._slider(samr2, "IoU threshold", 0, 1, 0.5, 20, pad_left=8,
                                        tooltip="Intersection-over-Union threshold for refined masks.\nHigher = only keeps refinements that closely match the original.")

        # Alpha Matting
        mat_section = _CollapsibleSection(detect, "Alpha Matting (ViTMatte)",
            subtitle="Soft alpha edges for hair, fur, semi-transparent boundaries")
        mat_section.pack(fill="x", padx=2, pady=(0, 4))
        matc = mat_section.content

        matr1 = ctk.CTkFrame(matc, fg_color="transparent")
        matr1.pack(fill="x", pady=2)
        self.matting_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(matr1, text="Enable", variable=self.matting_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Generate soft alpha edges using ViTMatte.\nBest for hair, fur, and semi-transparent boundaries.")
        ctk.CTkLabel(matr1, text="Model:").pack(side="left", padx=(12, 2))
        self.matting_model_var = ctk.StringVar(value="small")
        _w = ctk.CTkOptionMenu(matr1, variable=self.matting_model_var,
                               values=["small", "base"], width=90)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "ViTMatte model size.\nsmall = ~100MB, good quality. base = larger, slightly better edges.")
        ctk.CTkLabel(matr1, text="(small=~100MB, recommended)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        matr2 = ctk.CTkFrame(matc, fg_color="transparent")
        matr2.pack(fill="x", pady=2)
        self.matting_erode_var = self._slider(matr2, "Erode", 0, 50, 10, 50, fmt=".0f",
                                             tooltip="Shrink trimap 'definite foreground' zone (pixels).\nLarger = more pixels treated as uncertain boundary.")
        self.matting_dilate_var = self._slider(matr2, "Dilate", 0, 50, 10, 50,
                                              fmt=".0f", pad_left=8,
                                              tooltip="Expand trimap 'definite background' zone (pixels).\nControls how far the soft-edge region extends.")
        ctk.CTkLabel(matr2, text="(trimap border)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Ensemble Detection
        ens_section = _CollapsibleSection(detect, "Ensemble Detection (WMF)",
            subtitle="Run multiple detectors and merge results")
        ens_section.pack(fill="x", padx=2, pady=(0, 4))
        ensc = ens_section.content

        ensr1 = ctk.CTkFrame(ensc, fg_color="transparent")
        ensr1.pack(fill="x", pady=2)
        self.ensemble_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(ensr1, text="Enable", variable=self.ensemble_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Run multiple detection models and merge results.\nWeighted Mask Fusion (WMF) combines outputs for higher accuracy.")
        ctk.CTkLabel(ensr1, text="RF-DETR size:").pack(side="left", padx=(12, 2))
        self.rfdetr_size_var = ctk.StringVar(value="small")
        _w = ctk.CTkOptionMenu(ensr1, variable=self.rfdetr_size_var,
                               values=["nano", "small", "medium", "large"], width=90)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "RF-DETR model variant for ensemble.\nnano=fastest, large=most accurate.")
        ctk.CTkLabel(ensr1, text="(runs YOLO26 + RF-DETR, fuses masks)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        ensr2 = ctk.CTkFrame(ensc, fg_color="transparent")
        ensr2.pack(fill="x", pady=2)
        ctk.CTkLabel(ensr2, text="Models:").pack(side="left")
        self.ens_models_var = ctk.StringVar(value="yolo26, rfdetr")
        _w = ctk.CTkEntry(ensr2, textvariable=self.ens_models_var, width=140)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Comma-separated model names to include in ensemble.\nDefault: yolo26, rfdetr")
        self.ens_iou_var = self._slider(ensr2, "IoU threshold", 0, 1, 0.5, 20, pad_left=8,
                                        tooltip="IoU threshold for merging overlapping detections.\nHigher = only merges very similar detections.")

        # Edge Injection
        edge_section = _CollapsibleSection(detect, "Edge Injection",
            subtitle="Canny edges for thin structures (wires, antennas)")
        edge_section.pack(fill="x", padx=2, pady=(0, 4))
        edgec = edge_section.content

        edge_row = ctk.CTkFrame(edgec, fg_color="transparent")
        edge_row.pack(fill="x", pady=2)
        self.edge_inject_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(edge_row, text="Enable",
                             variable=self.edge_inject_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Add Canny edge detection for thin structures.\nCatches wires, antennas, and poles that object detectors miss.")
        ctk.CTkLabel(edge_row,
                     text="Detects thin structures that object detectors miss",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=8)

        # COLMAP Geometric Validation
        col_section = _CollapsibleSection(detect, "COLMAP Geometric Validation",
            subtitle="Cross-check masks against 3D reconstruction")
        col_section.pack(fill="x", padx=2, pady=(0, 4))
        colc = col_section.content

        colr1 = ctk.CTkFrame(colc, fg_color="transparent")
        colr1.pack(fill="x", pady=2)
        self.colmap_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(colr1, text="Enable", variable=self.colmap_var, width=80)
        _w.pack(side="left")
        Tooltip(_w, "Validate masks against COLMAP 3D reconstruction.\nFlags masks where masked points appear in 3D space.")
        ctk.CTkLabel(colr1, text="Sparse dir:").pack(side="left", padx=(12, 2))
        self.colmap_dir_var = ctk.StringVar(value="")
        _w = ctk.CTkEntry(colr1, textvariable=self.colmap_dir_var, width=200)
        _w.pack(side="left", padx=2, fill="x", expand=True)
        Tooltip(_w, "Path to COLMAP sparse reconstruction folder.\nContains cameras.bin, images.bin, points3D.bin")
        ctk.CTkButton(colr1, text="Browse", width=60,
                      command=self._browse_colmap_dir).pack(side="left", padx=2)

        colr2 = ctk.CTkFrame(colc, fg_color="transparent")
        colr2.pack(fill="x", pady=2)
        self.colmap_agree_var = self._slider(colr2, "Agreement", 0, 1, 0.7, 20,
                                             tooltip="Minimum agreement ratio for a mask to pass validation.\nHigher = stricter geometric consistency requirement.")
        self.colmap_flag_var = self._slider(colr2, "Flag above", 0, 1, 0.15, 20, pad_left=8,
                                            tooltip="Threshold for flagging inconsistent masks.\nMasks with inconsistency above this are marked for review.")
        ctk.CTkLabel(colr2, text="(3D check)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # ==============================================================
        #  POST-PROCESSING
        # ==============================================================
        self._post_sec = _CollapsibleSection(scroll, "Post-Processing", expanded=True)
        post_sec = self._post_sec
        post_sec.pack(fill="x", pady=(0, 6), padx=4)
        post = post_sec.content

        # Mask dilate + Fill holes
        pp_row = ctk.CTkFrame(post, fg_color="transparent")
        pp_row.pack(fill="x", padx=6, pady=3)
        self.mask_dilate_var = self._slider(pp_row, "Mask dilate", 0, 50, 0, 50,
                                             fmt=".0f",
                                             tooltip="Expand the final mask by N pixels.\nHelps catch thin edges missed by detection.")
        ctk.CTkLabel(pp_row, text="px",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(0, 8))
        self.fill_holes_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(pp_row, text="Fill holes",
                             variable=self.fill_holes_var, width=90)
        _w.pack(side="left", padx=(8, 0))
        Tooltip(_w, "Fill enclosed holes inside the mask.\nFixes gaps from equipment, camera mounts, or tripod joints.")
        ctk.CTkLabel(pp_row, text="(camera mount, equipment gaps inside mask)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # 360 / Equirect
        eq_row = ctk.CTkFrame(post, fg_color="transparent")
        eq_row.pack(fill="x", padx=6, pady=3)
        self.nadir_mask_var = self._slider(eq_row, "Nadir mask", 0, 25, 0, 50,
                                           fmt=".0f",
                                           tooltip="Mask the bottom N% of equirectangular images.\nCovers the nadir (straight down) where the pole/mount is visible.")
        ctk.CTkLabel(eq_row, text="%",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(0, 8))
        self.pole_expand_var = self._slider(eq_row, "Pole expand", 0, 5, 1.2, 50,
                                            fmt=".1f",
                                            tooltip="Expansion factor for pole/body detection near nadir.\nHigher = larger masked area around the photographer's pole.")
        ctk.CTkLabel(eq_row, text="(360° photographer body / pole distortion)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Fisheye
        fe_row = ctk.CTkFrame(post, fg_color="transparent")
        fe_row.pack(fill="x", padx=6, pady=3)
        self.fisheye_circle_var = ctk.BooleanVar(value=True)
        _w = ctk.CTkCheckBox(fe_row, text="Fisheye circle mask",
                             variable=self.fisheye_circle_var, width=140)
        _w.pack(side="left")
        Tooltip(_w, "Mask the dark corners of fisheye images.\nRemoves the circular boundary where the lens has no coverage.")
        self.fisheye_margin_var = self._slider(fe_row, "Margin", 0, 20, 0, 50,
                                                fmt=".0f", pad_left=8,
                                                tooltip="Additional margin inside the fisheye circle (%).\nLarger = more aggressive corner masking.")
        ctk.CTkLabel(fe_row, text="% (mask corners + distorted periphery)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Performance
        perf_row = ctk.CTkFrame(post, fg_color="transparent")
        perf_row.pack(fill="x", padx=6, pady=3)
        self.torch_compile_var = ctk.BooleanVar(value=False)
        _w = ctk.CTkCheckBox(perf_row, text="torch.compile",
                             variable=self.torch_compile_var, width=120)
        _w.pack(side="left")
        Tooltip(_w, "Enable PyTorch compilation for faster inference.\nFirst run is slower (compilation), subsequent runs are faster.")
        self.cubemap_overlap_var = self._slider(perf_row, "Cubemap overlap", 0, 30, 0, 30,
                                                 fmt=".0f", pad_left=12,
                                                 tooltip="Overlap between cubemap faces (degrees).\nReduces seam artifacts when merging segmented cubemap faces.\nOnly applies to equirectangular geometry.")
        ctk.CTkLabel(perf_row, text="° (equirect only)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # ==============================================================
        #  RUN / CANCEL
        # ==============================================================
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(fill="x", pady=6)

        self._preview_run_btn = ctk.CTkButton(
            btn_row, text="Preview", width=90, height=36,
            fg_color="#2563eb", hover_color="#1d4ed8",
            command=self._on_preview,
        )
        self._preview_run_btn.pack(side="left", padx=(10, 5))
        Tooltip(self._preview_run_btn, "Process one image to preview the mask result.\nUses the currently displayed image from the navigator.")

        self.run_btn = ctk.CTkButton(
            btn_row, text="Run Masking", width=140, height=36,
            fg_color="#16a34a", hover_color="#15803d",
            command=self._on_run,
        )
        self.run_btn.pack(side="left", padx=5)
        Tooltip(self.run_btn, "Process all images in the input folder.\nGenerates masks using current settings.")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop", width=80, height=36,
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._on_stop,
        )
        Tooltip(self.stop_btn, "Stop the current masking operation.")

        self.progress_label = ctk.CTkLabel(btn_row, text="", font=("Consolas", 11))
        self.progress_label.pack(side="left", padx=12)

        # ==============================================================
        #  BATCH MASKING QUEUE
        # ==============================================================
        from masking_queue import MaskingQueue
        self.masking_queue = MaskingQueue()
        self.mq_item_widgets = {}
        self.mq_processing = False

        mq_sec = _CollapsibleSection(scroll, "Batch Queue", expanded=False)
        mq_sec.pack(fill="x", pady=(6, 0), padx=2)
        mqc = mq_sec.content

        # queue controls
        mq_ctrl = ctk.CTkFrame(mqc, fg_color="transparent")
        mq_ctrl.pack(fill="x", pady=(2, 4))
        _w = ctk.CTkButton(mq_ctrl, text="Add Current", width=85,
                           command=self._mq_add_current)
        _w.pack(side="left", padx=(0, 4))
        Tooltip(_w, "Add the current input/output folder pair to the batch queue.")
        _w = ctk.CTkButton(mq_ctrl, text="Add Subfolders", width=100,
                           command=self._mq_add_subfolders)
        _w.pack(side="left", padx=(0, 4))
        Tooltip(_w, "Scan input folder for subfolders and add each as a queue item.")
        _w = ctk.CTkButton(mq_ctrl, text="Remove", width=60, fg_color="#666",
                           command=self._mq_remove_selected)
        _w.pack(side="left", padx=(0, 4))
        Tooltip(_w, "Remove the selected item from the queue.")
        _w = ctk.CTkButton(mq_ctrl, text="Clear Done", width=72, fg_color="#666",
                           command=self._mq_clear_done)
        _w.pack(side="left")
        Tooltip(_w, "Remove completed items from the queue.")

        # queue list — scrollable, auto-resizes
        self.mq_scroll = ctk.CTkScrollableFrame(mqc, height=0, fg_color="transparent")
        self.mq_scroll.pack(fill="x", pady=(0, 4))

        self.mq_stats_label = ctk.CTkLabel(mqc, text="Queue: 0 pending, 0 done",
                                           text_color="gray", font=ctk.CTkFont(size=10))
        self.mq_stats_label.pack(anchor="w")

        # process queue button
        mq_btn_row = ctk.CTkFrame(mqc, fg_color="transparent")
        mq_btn_row.pack(fill="x", pady=(6, 4))
        self.mq_run_btn = ctk.CTkButton(
            mq_btn_row, text="Process Queue", command=self._run_masking_queue,
            fg_color="#2E7D32", hover_color="#1B5E20",
            font=ctk.CTkFont(size=13, weight="bold"), height=38,
        )
        self.mq_run_btn.pack(side="left", fill="x", expand=True)
        Tooltip(self.mq_run_btn, "Process all pending items in the batch queue sequentially.")

        self._mq_refresh()

        # ==============================================================
        #  PROGRESS BAR + STATUS
        # ==============================================================
        self.progress_bar = ctk.CTkProgressBar(scroll, height=8)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 4))
        self.progress_bar.set(0)

        self.stats_label = ctk.CTkLabel(
            scroll, text="Set input and output, then click Run Masking.",
            font=("Consolas", 11), anchor="w", justify="left",
        )
        self.stats_label.pack(fill="x", padx=10, pady=4)

    # ── SAM3 mode toggle ──

    def _on_sam3_mode_change(self, value):
        """Grey out widgets irrelevant to SAM3 Unified Video mode."""
        self.sam3_unified_var.set(value == "Unified Video")
        state = "disabled" if value == "Unified Video" else "normal"
        # Core Process: model/geometry row + mask target checkboxes
        for frame in self._unified_disable_frames:
            self._set_widgets_state(frame, state)
        # Entire Detection & Refinement + Post-Processing sections
        for sec in (self._detect_sec, self._post_sec):
            self._set_widgets_state(sec.content, state)
            if value == "Unified Video":
                sec.collapse()
            else:
                sec.expand()
        # Glow on Remove/Keep entries — these are the SAM3 text prompts
        glow = "#1a5276" if value == "Unified Video" else "transparent"
        for entry in (self.remove_prompts_entry, self.keep_prompts_entry):
            entry.master.configure(fg_color=glow)

    def _set_widgets_state(self, parent, state):
        """Recursively enable/disable all interactive widgets under *parent*."""
        _interactive = (ctk.CTkCheckBox, ctk.CTkOptionMenu, ctk.CTkSlider,
                        ctk.CTkEntry, ctk.CTkButton, ctk.CTkSegmentedButton)
        _dimmed = "#555555"
        for child in parent.winfo_children():
            if isinstance(child, _interactive):
                child.configure(state=state)
                # CTk disabled state doesn't visually dim fills
                if isinstance(child, ctk.CTkCheckBox):
                    if state == "disabled":
                        child._original_fg = child.cget("fg_color")
                        child.configure(fg_color=_dimmed)
                    elif hasattr(child, "_original_fg"):
                        child.configure(fg_color=child._original_fg)
                elif isinstance(child, ctk.CTkOptionMenu):
                    if state == "disabled":
                        child._original_fg = child.cget("fg_color")
                        child._original_btn = child.cget("button_color")
                        child.configure(fg_color=_dimmed, button_color=_dimmed)
                    elif hasattr(child, "_original_fg"):
                        child.configure(fg_color=child._original_fg,
                                        button_color=child._original_btn)
            self._set_widgets_state(child, state)

    # ── browse helpers ──
    # _browse_dir_into, _browse_folder_for, _browse_video_for inherited from AppInfrastructure

    def _browse_input_folder(self):
        self._browse_dir_into(self.input_entry, title="Select Input Folder")
        if self._preview_mode == "process":
            self._load_image_list()

    def _browse_input_file(self):
        self._browse_file_for(
            self.input_entry,
            title="Select Image or Video",
            filetypes=[
                ("Images & Videos", "*.jpg *.jpeg *.png *.tif *.mp4 *.mov *.avi *.mkv"),
                ("All Files", "*.*"),
            ],
        )
        if self._preview_mode == "process":
            self._load_image_list()

    def _browse_colmap_dir(self):
        path = filedialog.askdirectory(title="Select COLMAP Sparse Reconstruction Dir")
        if path:
            self.colmap_dir_var.set(path)

    # ── preview panel ──

    def _build_preview_panel(self):
        panel = self._preview_panel
        panel.grid_rowconfigure(1, weight=6)   # preview images get most space
        panel.grid_rowconfigure(3, weight=1)   # console gets minimal space
        panel.grid_columnconfigure(0, weight=1)

        # Row 0: Header with zoom + collapse toggle
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        ctk.CTkLabel(header, text="Preview",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(side="left")

        # Zoom control (100% = fit to panel width)
        self._zoom_var = ctk.DoubleVar(value=100)
        self._preview_overlay_pil = None  # cached PIL images for zoom
        self._preview_mask_pil = None
        ctk.CTkLabel(header, text="Zoom:", font=("Consolas", 10),
                     text_color="#9ca3af").pack(side="left", padx=(12, 2))
        self._zoom_label = ctk.CTkLabel(header, text="100%", width=35,
                                         font=("Consolas", 10))
        ctk.CTkSlider(header, from_=25, to=300, number_of_steps=55,
                      variable=self._zoom_var, width=90,
                      command=self._on_zoom_change,
        ).pack(side="left", padx=2)
        self._zoom_label.pack(side="left")

        # Mask view toggle (overlay only by default)
        self._show_mask_view = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(header, text="Mask", variable=self._show_mask_view,
                        width=55, command=self._on_view_toggle,
                        font=("Consolas", 10)).pack(side="left", padx=(10, 0))

        self._preview_collapse_btn = ctk.CTkButton(
            header, text="Hide >>", width=60, height=24,
            fg_color="transparent", hover_color=("gray75", "gray25"),
            text_color="#9ca3af", font=("Consolas", 10),
            command=self._toggle_preview_panel,
        )
        self._preview_collapse_btn.pack(side="right")

        # Row 1: Image display area (scrollable for tall equirect previews)
        self._preview_image_frame = ctk.CTkScrollableFrame(panel)
        self._preview_image_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        self._process_overlay_label = ctk.CTkLabel(self._preview_image_frame, text="")
        self._process_overlay_label.pack(padx=4, pady=4)

        # Mask view (hidden by default — toggled via checkbox)
        self._process_mask_label = ctk.CTkLabel(self._preview_image_frame, text="")

        # Hidden entry for internal path storage
        self._preview_image_entry = ctk.CTkEntry(panel)
        self._image_list: list = []
        self._nav_idx = ctk.IntVar(value=0)

        # Row 2: Navigator + stats + preview — all in one compact bar
        ctrl = ctk.CTkFrame(panel, fg_color="transparent")
        ctrl.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))

        ctk.CTkButton(ctrl, text="<", width=24, height=24,
                      command=self._nav_prev).pack(side="left")
        self._nav_slider = ctk.CTkSlider(
            ctrl, from_=0, to=1, number_of_steps=1,
            variable=self._nav_idx, command=self._on_nav_change,
        )
        self._nav_slider.pack(side="left", padx=2, fill="x", expand=True)
        ctk.CTkButton(ctrl, text=">", width=24, height=24,
                      command=self._nav_next).pack(side="left")
        self._nav_counter = ctk.CTkLabel(ctrl, text="0 / 0",
                                         font=("Consolas", 10), width=65)
        self._nav_counter.pack(side="left", padx=(4, 0))

        self._preview_stats = ctk.CTkLabel(
            ctrl, text="", font=("Consolas", 10),
            text_color="#9ca3af", anchor="w",
        )
        self._preview_stats.pack(side="left", padx=6)

        # Row 3: Console (compact)
        console_frame = ctk.CTkFrame(panel)
        console_frame.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
        console_frame.grid_rowconfigure(1, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(console_frame, text="Console", font=("Consolas", 10),
                     anchor="w").grid(row=0, column=0, sticky="w", padx=5, pady=(2, 0))
        self.log_textbox = ctk.CTkTextbox(console_frame, font=("Consolas", 10), height=80)
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 4))

    def _toggle_multi_pass(self):
        """Show/hide multi-pass list frame."""
        if self.multi_pass_var.get():
            self.multi_pass_list_frame.pack(fill="x", padx=5, pady=2)
            if not self.multi_pass_entries:
                self._add_multi_pass_row()
        else:
            self.multi_pass_list_frame.pack_forget()

    def _add_multi_pass_row(self):
        """Add a prompt + confidence row."""
        row = ctk.CTkFrame(self.multi_pass_list_frame, fg_color="transparent")
        prompt = ctk.CTkEntry(row, placeholder_text="e.g. human shadow", width=200)
        prompt.pack(side="left", padx=2)
        threshold = ctk.CTkEntry(row, placeholder_text="0.3", width=60)
        threshold.pack(side="left", padx=2)
        row.pack(fill="x", pady=1)
        self.multi_pass_entries.append((prompt, threshold, row))

    def _remove_multi_pass_row(self):
        """Remove the last prompt row."""
        if self.multi_pass_entries:
            _, _, row = self.multi_pass_entries.pop()
            row.destroy()

    def _toggle_preview_panel(self):
        if self._preview_visible:
            self._preview_panel.grid_forget()
            self._main_frame.grid_columnconfigure(1, weight=0, minsize=0)
            self._preview_visible = False
            self._preview_show_btn = ctk.CTkButton(
                self._main_frame, text="<<", width=24,
                fg_color="transparent", hover_color=("gray75", "gray25"),
                text_color="#9ca3af", font=("Consolas", 12),
                command=self._toggle_preview_panel,
            )
            self._preview_show_btn.grid(row=0, column=1, sticky="ns", padx=0)
        else:
            if hasattr(self, '_preview_show_btn'):
                self._preview_show_btn.destroy()
            self._main_frame.grid_columnconfigure(1, weight=65, uniform="split")
            self._preview_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
            self._preview_visible = True
        self._prefs["preview_visible"] = self._preview_visible
        self._save_prefs()

    def _on_view_toggle(self):
        """Toggle mask view visibility in preview panel."""
        if self._show_mask_view.get():
            self._process_mask_label.pack(padx=4, pady=4)
            if self._preview_mask_pil is not None:
                zoom = self._zoom_var.get() / 100.0
                ow, oh = self._preview_mask_pil.size
                zw, zh = max(1, int(ow * zoom)), max(1, int(oh * zoom))
                ctk_mask = ctk.CTkImage(light_image=self._preview_mask_pil, size=(zw, zh))
                self._process_mask_label.configure(image=ctk_mask, text="")
                self._process_mask_label._ctk_image = ctk_mask
        else:
            self._process_mask_label.pack_forget()

    def _on_tab_change(self):
        """Tab changed — switch navigator data source and swap right panel."""
        active = self.tabs.get()

        # -- Swap right-side panel: Projects detail vs. preview --
        if active == "Projects":
            # Hide preview, show project detail panel
            self._preview_panel.grid_forget()
            if hasattr(self, '_preview_show_btn'):
                self._preview_show_btn.destroy()
            self._proj_detail_panel.grid(
                row=0, column=1, sticky="nsew", padx=0, pady=0,
            )
            self._main_frame.grid_columnconfigure(1, weight=65, uniform="split")
        else:
            # Hide project detail panel, restore preview
            if hasattr(self, '_proj_detail_panel'):
                self._proj_detail_panel.grid_forget()
            if self._preview_visible:
                self._main_frame.grid_columnconfigure(1, weight=65, uniform="split")
                self._preview_panel.grid(
                    row=0, column=1, sticky="nsew", padx=0, pady=0,
                )
            else:
                self._main_frame.grid_columnconfigure(1, weight=0, minsize=0)

        if active == "Mask":
            self._preview_mode = "process"
            self._load_image_list()
        elif active == "Review":
            self._preview_mode = "review"
            self._load_review_nav_list()
        else:
            # Extract / Coverage / Projects — no special preview mode
            self._preview_mode = active.lower()
            if active == "Coverage":
                from tabs.gaps_tab import _refresh_bridge_info
                _refresh_bridge_info(self)

    def _browse_preview_image(self):
        path = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All", "*.*")],
        )
        if path:
            self._preview_image_entry.delete(0, "end")
            self._preview_image_entry.insert(0, path)

    # ── image navigator ──

    def _load_image_list(self):
        """Scan input folder and populate the navigator slider."""
        input_path = self.input_entry.get().strip()
        if not input_path:
            return
        inp = Path(input_path)
        if inp.is_file():
            self._image_list = [inp]
        elif inp.is_dir():
            pattern = self.pattern_var.get() or "*.jpg *.png"
            # Support space-separated multi-patterns (e.g. "*.jpg *.png")
            files = []
            for pat in pattern.split():
                files.extend(inp.glob(pat))
            self._image_list = sorted(set(files))
            # Auto-detect: if nothing matched, try common image extensions
            if not self._image_list:
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"):
                    self._image_list = sorted(inp.glob(ext))
                    if self._image_list:
                        break
        else:
            self._image_list = []

        n = len(self._image_list)
        if n == 0:
            self._nav_slider.configure(to=1, number_of_steps=1)
            self._nav_counter.configure(text="0 / 0")
            self._preview_stats.configure(text="")
            return

        # CTkSlider divides by step_size; to==from_ (n==1) gives step_size=0 → ZeroDivisionError
        slider_to = max(n - 1, 1)
        self._nav_slider.configure(to=slider_to, number_of_steps=slider_to)
        # If a previously saved image is in the list, jump to it
        saved = self._preview_image_entry.get().strip()
        if saved:
            try:
                idx = next(i for i, p in enumerate(self._image_list) if str(p) == saved)
                self._nav_idx.set(idx)
            except StopIteration:
                self._nav_idx.set(0)
        else:
            self._nav_idx.set(0)
        self._on_nav_change(self._nav_idx.get())

    def _on_nav_change(self, value):
        """Slider moved — update labels immediately, debounce heavy image load."""
        idx = int(float(value))
        n = len(self._image_list)
        if n == 0:
            return
        idx = max(0, min(idx, n - 1))
        self._nav_idx.set(idx)

        # Update lightweight labels immediately (keeps slider responsive)
        self._nav_counter.configure(text=f"{idx + 1} / {n}")

        if self._preview_mode == "review" and self._filtered_pairs:
            if idx < len(self._filtered_pairs):
                pair = self._filtered_pairs[idx]
                self._selected_stem = pair["stem"]
                self._preview_stats.configure(text=pair["image_path"].name)
                # Debounce the heavy overlay render so slider stays responsive
                if self._nav_debounce_id is not None:
                    self.after_cancel(self._nav_debounce_id)
                self._nav_debounce_id = self.after(
                    80, lambda p=pair: self._deferred_review_load(p))
            return

        img_path = self._image_list[idx]
        self._preview_stats.configure(text=img_path.name)
        self._preview_image_entry.delete(0, "end")
        self._preview_image_entry.insert(0, str(img_path))

    def _deferred_review_load(self, pair):
        """Load review overlay after debounce delay."""
        self._nav_debounce_id = None
        self._show_review_overlay(pair)
        self._update_review_info()

    def _nav_prev(self):
        idx = self._nav_idx.get()
        if idx > 0:
            self._nav_idx.set(idx - 1)
            self._on_nav_change(idx - 1)

    def _nav_next(self):
        idx = self._nav_idx.get()
        if idx < len(self._image_list) - 1:
            self._nav_idx.set(idx + 1)
            self._on_nav_change(idx + 1)

    def _load_review_nav_list(self):
        """Populate navigator from filtered review pairs."""
        if not self._review_loaded:
            self._image_list = []
            self._nav_slider.configure(to=1, number_of_steps=1)
            self._nav_counter.configure(text="0 / 0")
            self._preview_stats.configure(text="Load masks to begin.")
            return

        self._apply_filter()
        self._image_list = [pair["image_path"] for pair in self._filtered_pairs]
        n = len(self._image_list)
        if n == 0:
            self._nav_slider.configure(to=1, number_of_steps=1)
            self._nav_counter.configure(text="0 / 0")
            self._preview_stats.configure(text="No masks match filter.")
            return

        # CTkSlider divides by step_size; to==from_ (n==1) gives step_size=0 → ZeroDivisionError
        slider_to = max(n - 1, 1)
        self._nav_slider.configure(to=slider_to, number_of_steps=slider_to)
        if self._selected_stem:
            idx = next(
                (i for i, p in enumerate(self._filtered_pairs) if p["stem"] == self._selected_stem),
                0,
            )
            self._nav_idx.set(idx)
        else:
            self._nav_idx.set(0)
        self._on_nav_change(self._nav_idx.get())

    def _show_review_overlay(self, pair):
        """Render overlay for a review pair into the shared preview panel."""
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        img = cv2.imread(str(pair["image_path"]))
        mask = cv2.imread(str(pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return

        h, w = img.shape[:2]
        # winfo_width/height return physical pixels; CTkImage size is logical pixels
        # Use panel width from scrollable frame, but height from parent panel
        # (CTkScrollableFrame.winfo_height returns content height, not viewport)
        panel_w = self._preview_image_frame.winfo_width() / self._dpi_scale
        panel_total_h = self._preview_panel.winfo_height() / self._dpi_scale
        # Subtract header (~35), nav bar (~35), console (~15% of panel)
        panel_h = panel_total_h * 0.80 - 70
        target_w = max(200, panel_w - 20) if panel_w > 60 else 500
        target_h = max(200, panel_h) if panel_h > 60 else 500
        # Fit entire image within panel (both width and height)
        scale = min(target_w / w, target_h / h)
        pw, ph = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (pw, ph))
        mask_r = cv2.resize(mask, (pw, ph))

        # Red tint overlay on masked-out regions (black in mask = foreground to remove)
        overlay = img_r.copy()
        masked = mask_r < 128
        overlay[masked] = (overlay[masked].astype(np.float32) * 0.4 +
                           np.array([0, 0, 200], dtype=np.float32) * 0.6).astype(np.uint8)
        overlay_pil = PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        mask_pil = PILImage.fromarray(cv2.cvtColor(
            cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))

        self._preview_overlay_pil = overlay_pil
        self._preview_mask_pil = mask_pil

        # Apply current zoom (don't reset — preserves user's zoom across navigation)
        zoom = self._zoom_var.get() / 100.0
        zw, zh = max(1, int(pw * zoom)), max(1, int(ph * zoom))
        ctk_overlay = ctk.CTkImage(light_image=overlay_pil, size=(zw, zh))
        self._process_overlay_label.configure(image=ctk_overlay, text="")
        self._process_overlay_label._ctk_image = ctk_overlay

        if self._show_mask_view.get():
            ctk_mask = ctk.CTkImage(light_image=mask_pil, size=(zw, zh))
            self._process_mask_label.configure(image=ctk_mask, text="")
            self._process_mask_label._ctk_image = ctk_mask

    def _update_review_info(self):
        """Update the info panel in the Review tab for the current selection."""
        if not self._selected_stem or not self._review_status_mgr:
            return
        ms = self._review_status_mgr.get(self._selected_stem)
        self._review_info_label.configure(
            text=f"{self._selected_stem}\n"
                 f"Quality: {ms.quality or '?'}  |  Conf: {ms.confidence:.0%}  |  "
                 f"Area: {ms.area_percent:.1f}%  |  Status: {ms.status}"
        )
        # Update summary
        if self._review_status_mgr:
            summary = self._review_status_mgr.get_summary()
            done = summary.get("accepted", 0) + summary.get("edited", 0)
            self._review_summary_label.configure(
                text=f"{done} done | {summary.get('rejected', 0)} rej | "
                     f"{summary.get('pending', 0)} pending"
            )

    # ── zoom ──

    def _on_zoom_change(self, value):
        """Re-render cached preview images at new zoom level."""
        pct = int(float(value))
        self._zoom_label.configure(text=f"{pct}%")
        if self._preview_overlay_pil is None:
            return
        zoom = pct / 100.0
        ow, oh = self._preview_overlay_pil.size
        zw, zh = max(1, int(ow * zoom)), max(1, int(oh * zoom))

        ctk_overlay = ctk.CTkImage(light_image=self._preview_overlay_pil, size=(zw, zh))
        self._process_overlay_label.configure(image=ctk_overlay, text="")
        self._process_overlay_label._ctk_image = ctk_overlay

        if self._show_mask_view.get() and self._preview_mask_pil is not None:
            ctk_mask = ctk.CTkImage(light_image=self._preview_mask_pil, size=(zw, zh))
            self._process_mask_label.configure(image=ctk_mask, text="")
            self._process_mask_label._ctk_image = ctk_mask

    # ── preview execution ──

    def _resolve_preview_image(self):
        """Get the image to preview from navigator or input."""
        explicit = self._preview_image_entry.get().strip()
        if explicit and Path(explicit).is_file():
            return Path(explicit)
        input_path = self.input_entry.get().strip()
        if not input_path:
            return None
        inp = Path(input_path)
        if inp.is_file():
            return inp
        if inp.is_dir():
            pattern = self.pattern_var.get() or "*.jpg *.png"
            files = []
            for pat in pattern.split():
                files.extend(inp.glob(pat))
            files = sorted(set(files))
            if not files:
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"):
                    files = sorted(inp.glob(ext))
                    if files:
                        break
            if files:
                return files[0]
        return None

    def _on_preview(self):
        # Auto-load image list if not yet loaded
        if not self._image_list:
            self._load_image_list()
        img_path = self._resolve_preview_image()
        if img_path is None:
            self.log("Preview: no image found. Set an input path first.")
            return
        if self.is_running:
            self.log("Preview: batch is running, wait for it to finish.")
            return
        self._prefs["preview_image"] = str(img_path)
        self._save_prefs()
        self._preview_run_btn.configure(state="disabled", text="Running...")
        self._preview_stats.configure(text="Processing...")
        # Capture panel dimensions on main thread (winfo is unsafe from bg thread)
        # Convert physical pixels to logical pixels for CTkImage sizing
        # Use panel width from scrollable frame, but height from parent panel
        # (CTkScrollableFrame.winfo_height returns content height, not viewport)
        panel_w = self._preview_image_frame.winfo_width() / self._dpi_scale
        panel_total_h = self._preview_panel.winfo_height() / self._dpi_scale
        panel_h = panel_total_h * 0.80 - 70
        threading.Thread(target=self._preview_worker, args=(img_path, panel_w, panel_h), daemon=True).start()

    def _preview_worker(self, img_path: Path, panel_w: float = 0, panel_h: float = 0):
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage

            MaskingPipeline = _import_pipeline()[0]
            config, model_str, geometry = self._build_mask_config()
            pipeline = MaskingPipeline(config=config, auto_select_model=(model_str == "auto"))

            image = cv2.imread(str(img_path))
            if image is None:
                self.log(f"Preview: failed to load {img_path}")
                return

            self.log(f"Preview: processing {img_path.name}...")
            result = pipeline.process_image(image, geometry)

            # Build overlay — fit entire image within panel
            h, w = image.shape[:2]
            # panel_w/panel_h already in logical pixels (DPI-corrected on main thread)
            # panel_h is already adjusted (80% of panel minus header/nav)
            target_w = max(200, panel_w - 20) if panel_w > 60 else 500
            target_h = max(200, panel_h) if panel_h > 60 else 500
            scale = min(target_w / w, target_h / h)
            pw, ph = int(w * scale), int(h * scale)
            img_r = cv2.resize(image, (pw, ph))
            mask_uint8 = (result.mask * 255).astype(np.uint8) if result.mask.max() <= 1 else result.mask
            mask_r = cv2.resize(mask_uint8, (pw, ph), interpolation=cv2.INTER_NEAREST)

            # Red tint overlay
            overlay = img_r.copy()
            masked = mask_r > 127
            overlay[masked] = (overlay[masked].astype(np.float32) * 0.4 +
                               np.array([0, 0, 200], dtype=np.float32) * 0.6).astype(np.uint8)
            overlay_pil = PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            mask_pil = PILImage.fromarray(cv2.cvtColor(
                cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))

            area_pct = np.sum(mask_r > 127) / mask_r.size * 100

            def _update():
                # Cache PIL images for zoom re-render
                self._preview_overlay_pil = overlay_pil
                self._preview_mask_pil = mask_pil
                # Apply current zoom (preserves user setting)
                zoom = self._zoom_var.get() / 100.0
                zw = max(1, int(pw * zoom))
                zh = max(1, int(ph * zoom))

                ctk_overlay = ctk.CTkImage(light_image=overlay_pil, size=(zw, zh))
                self._process_overlay_label.configure(image=ctk_overlay, text="")
                self._process_overlay_label._ctk_image = ctk_overlay

                if self._show_mask_view.get():
                    ctk_mask = ctk.CTkImage(light_image=mask_pil, size=(zw, zh))
                    self._process_mask_label.configure(image=ctk_mask, text="")
                    self._process_mask_label._ctk_image = ctk_mask

                self._preview_stats.configure(
                    text=f"{img_path.name}  |  Quality: {result.quality.value}  |  "
                         f"Conf: {result.confidence:.0%}  |  Area: {area_pct:.1f}%"
                )

            self.after(0, _update)
            self.log(f"Preview: {img_path.name} -> {result.quality.value} ({result.confidence:.0%})")

        except Exception as e:
            self.log(f"Preview error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.after(0, lambda: self._preview_run_btn.configure(state="normal", text="Preview"))

    # ── run masking ──

    def _on_run(self):
        input_path = self.input_entry.get().strip()
        output_path = self.output_entry.get().strip()

        if not input_path:
            self.log("ERROR: No input path specified.")
            return
        if not output_path:
            self.log("ERROR: No output path specified.")
            return

        # Save prefs including prompts
        self._prefs["input_dir"] = input_path
        self._prefs["output_dir"] = output_path
        self._prefs["remove_prompts"] = self.remove_prompts_entry.get().strip()
        self._prefs["keep_prompts"] = self.keep_prompts_entry.get().strip()
        self._prefs["shadow_detector"] = self.shadow_detector_var.get()
        self._prefs["shadow_verifier"] = self.shadow_verifier_var.get()
        self._prefs["shadow_spatial"] = self.shadow_spatial_var.get()
        self._prefs["save_review_folder"] = self.review_folder_var.get()
        self._prefs["save_reject_review_images"] = self.review_rejects_var.get()
        self._save_prefs()

        if self.is_running or self.mq_processing:
            self.log("A process is already running.")
            return

        self.is_running = True
        self.cancel_flag.clear()
        self.run_btn.configure(state="disabled")
        self.mq_run_btn.configure(state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.progress_bar.set(0)
        self.stats_label.configure(text="Initializing...")

        threading.Thread(target=self._worker, daemon=True,
                         args=(input_path, output_path)).start()

    def _on_stop(self):
        if self.is_running:
            self.log("Stopping...")
            self.cancel_flag.set()

    def _build_mask_config(self):
        """Build MaskConfig from current GUI settings.

        Returns (config, model_str, geometry) tuple.
        """
        _, MaskConfig, SegmentationModel, ImageGeometry, _, _ = _import_pipeline()

        model_str = self.model_var.get()
        model_map = {
            "sam3": SegmentationModel.SAM3,
            "yolo26": SegmentationModel.YOLO26,
            "rfdetr": SegmentationModel.RFDETR,
            "fastsam": SegmentationModel.FASTSAM,
            "auto": None,
        }

        raw_remove = self.remove_prompts_entry.get().strip()
        remove_prompts = [p.strip() for p in raw_remove.split(",") if p.strip()] if raw_remove else []
        raw_keep = self.keep_prompts_entry.get().strip()
        keep_prompts = [p.strip() for p in raw_keep.split(",") if p.strip()] if raw_keep else []

        # Build YOLO classes + text prompts from target checkboxes.
        # Every checked label goes to remove_prompts (for SAM3/text models).
        # Labels with COCO IDs also go to classes (for YOLO models).
        classes = []
        for label, (var, coco_id) in self._target_vars.items():
            if var.get():
                if coco_id is not None:
                    classes.append(coco_id)
                if label not in remove_prompts:
                    remove_prompts.append(label)
        remove_prompts = remove_prompts or None

        geometry_map = {
            "pinhole": ImageGeometry.PINHOLE,
            "equirect": ImageGeometry.EQUIRECTANGULAR,
            "fisheye": ImageGeometry.FISHEYE,
            "cubemap": ImageGeometry.CUBEMAP,
        }
        geometry = geometry_map.get(self.geometry_var.get(), ImageGeometry.PINHOLE)

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        shadow_config = None
        shadow_enabled = self.shadow_var.get()
        if shadow_enabled:
            det = self.shadow_detector_var.get()
            verifier = self.shadow_verifier_var.get()
            shadow_config = {
                'primary_detector': det,
                'verification_detector': verifier if verifier != "none" else None,
                'spatial_mode': self.shadow_spatial_var.get(),
                'dilation_radius': int(self.shadow_dilation_var.get()),
                'confidence_threshold': float(self.shadow_conf_var.get()),
                'darkness_ratio_threshold': float(self.shadow_darkness_var.get()),
                'chromaticity_threshold': float(self.shadow_chroma_var.get()),
                'device': device,
            }

        sam_refine_config = None
        sam_refine_enabled = self.sam_refine_var.get()
        if sam_refine_enabled:
            sam_refine_config = {
                'sam_model_type': self.sam_model_var.get(),
                'box_margin_ratio': float(self.sam_margin_var.get()),
                'iou_threshold': float(self.sam_iou_var.get()),
                'device': device,
            }

        matting_config = None
        matting_enabled = self.matting_var.get()
        if matting_enabled:
            matting_config = {
                'model_size': self.matting_model_var.get(),
                'erode_kernel': int(self.matting_erode_var.get()),
                'dilate_kernel': int(self.matting_dilate_var.get()),
                'device': device,
            }

        ensemble_enabled = self.ensemble_var.get()
        ens_models_raw = self.ens_models_var.get().strip()
        ensemble_models = [m.strip() for m in ens_models_raw.split(",") if m.strip()] if ens_models_raw else ["yolo26", "rfdetr"]

        colmap_enabled = self.colmap_var.get()
        colmap_dir = self.colmap_dir_var.get().strip()
        colmap_config = None
        if colmap_enabled and colmap_dir:
            colmap_config = {
                'agreement_threshold': float(self.colmap_agree_var.get()),
                'inconsistency_threshold': float(self.colmap_flag_var.get()),
            }

        # Collect multi-pass prompts
        multi_pass = []
        if self.multi_pass_var.get():
            for prompt_entry, threshold_entry, _ in self.multi_pass_entries:
                text = prompt_entry.get().strip()
                if text:
                    try:
                        thresh = float(threshold_entry.get().strip() or "0.5")
                    except ValueError:
                        thresh = 0.5
                    multi_pass.append((text, thresh))

        config_kwargs = dict(
            model=model_map.get(model_str),
            device=device,
            yolo_classes=classes,
            detect_shadows=shadow_enabled,
            shadow_config=shadow_config,
            sam_refine=sam_refine_enabled,
            sam_refine_config=sam_refine_config,
            matting=matting_enabled,
            matting_config=matting_config,
            ensemble=ensemble_enabled,
            ensemble_models=ensemble_models,
            ensemble_iou_threshold=float(self.ens_iou_var.get()),
            rfdetr_model_size=self.rfdetr_size_var.get(),
            sam3_unified=self.sam3_unified_var.get(),
            multi_pass_prompts=multi_pass,
            colmap_validate=colmap_enabled and bool(colmap_dir),
            colmap_dir=colmap_dir,
            colmap_config=colmap_config,
            edge_injection=self.edge_inject_var.get(),
            multi_label=self.multi_label_var.get(),
            inpaint_masked=self.inpaint_var.get(),
            mask_dilate_px=int(self.mask_dilate_var.get()),
            fill_holes=self.fill_holes_var.get(),
            nadir_mask_percent=float(self.nadir_mask_var.get()),
            fisheye_circle_mask=self.fisheye_circle_var.get(),
            fisheye_margin_percent=float(self.fisheye_margin_var.get()),
            pole_mask_expand=float(self.pole_expand_var.get()),
            cubemap_overlap=float(self.cubemap_overlap_var.get()),
            torch_compile=self.torch_compile_var.get(),
            confidence_threshold=float(self.conf_var.get()),
            review_threshold=float(self.review_thresh_var.get()),
            yolo_model_size=self.yolo_size_var.get(),
            save_review_images=self.review_folder_var.get(),
            save_reject_review_images=self.review_rejects_var.get(),
            keep_prompts=keep_prompts,
        )
        if remove_prompts is not None:
            config_kwargs["remove_prompts"] = remove_prompts

        config = MaskConfig(**config_kwargs)
        return config, model_str, geometry

    def _worker(self, input_path: str, output_path: str):
        try:
            import numpy as np
            MaskingPipeline = _import_pipeline()[0]
            config, model_str, geometry = self._build_mask_config()

            inp = Path(input_path)
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            create_review_folder = self.review_folder_var.get()

            n_classes = len(config.yolo_classes) if config.yolo_classes else 0
            self.log(f"Model: {model_str} | Geometry: {self.geometry_var.get()} | "
                     f"Classes: {n_classes} selected | Device: {config.device}")
            if config.remove_prompts:
                self.log(f"Remove: {config.remove_prompts}")
            if config.keep_prompts:
                self.log(f"Keep: {config.keep_prompts}")

            pipeline = MaskingPipeline(config=config, auto_select_model=(model_str == "auto"))

            # Quality bridge: write per-image results to review_status.json
            if create_review_folder:
                mask_out_dir = out / "masks"
                review_dir = out / "review"
            else:
                mask_out_dir = out
                review_dir = None
            try:
                ReviewStatusManager = _import_review()[4]
                _status_mgr = ReviewStatusManager(mask_out_dir)
                def _on_result(stem, result):
                    area = float(np.sum(result.mask > 0) / result.mask.size * 100)
                    _status_mgr.set_quality_info(
                        stem, quality=result.quality.value,
                        confidence=result.confidence, area_percent=area)
            except Exception:
                _on_result = None

            if inp.is_dir():
                if create_review_folder:
                    self.log(f"Output: masks → {mask_out_dir}, review → {review_dir}")
                else:
                    self.log(f"Output: masks → {mask_out_dir}")
                if self.sam3_unified_var.get() and pipeline.sam3_video_pipeline is not None:
                    self.log(f"Processing directory with SAM 3.1 unified video: {inp}")
                    stats = pipeline.process_directory_sam3_unified(
                        input_dir=inp, output_dir=out,
                    )
                else:
                    self.log(f"Processing directory: {inp}")
                    stats = pipeline.process_directory(
                        input_dir=inp, output_dir=out,
                        geometry=geometry, pattern=self.pattern_var.get(),
                        result_callback=_on_result,
                        skip_existing=self.skip_existing_var.get(),
                        cancel_event=self.cancel_flag,
                        mask_dir=mask_out_dir,
                        review_dir=review_dir,
                    )
            elif inp.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                if create_review_folder:
                    self.log(f"Output: masks → {mask_out_dir}, review → {review_dir}")
                else:
                    self.log(f"Output: masks → {mask_out_dir}")
                if self.sam3_unified_var.get() and pipeline.sam3_video_pipeline is not None:
                    self.log(f"Processing video with SAM 3.1 unified: {inp}")
                    stats = pipeline.process_video_sam3_unified(
                        video_path=inp, output_dir=out,
                        mask_dir=mask_out_dir,
                    )
                else:
                    self.log(f"Processing video: {inp}")
                    stats = pipeline.process_video(
                        video_path=inp, output_dir=out,
                        geometry=geometry,
                        save_review=create_review_folder,
                        mask_dir=mask_out_dir,
                        review_dir=review_dir,
                    )
            else:
                self.log(f"Processing single image: {inp}")
                import cv2, shutil
                image = cv2.imread(str(inp))
                if image is None:
                    self.log(f"ERROR: Failed to load image: {inp}")
                    return
                result = pipeline.process_image(image, geometry)
                mask_dir = mask_out_dir
                mask_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(mask_dir / f"{inp.stem}.png"), result.mask * 255)
                if create_review_folder:
                    images_dir = out / "images"
                    images_dir.mkdir(exist_ok=True)
                    shutil.copy2(str(inp), str(images_dir / inp.name))
                    if result.should_save_review_image(self.review_rejects_var.get()):
                        review_dir.mkdir(exist_ok=True)
                        review_img = pipeline._create_review_image(image, result.mask)
                        cv2.imwrite(str(review_dir / f"review_{inp.stem}.jpg"), review_img)
                stats = {
                    "processed_images": 1,
                    "quality": result.quality.value,
                    "review_images": int(
                        create_review_folder and result.should_save_review_image(self.review_rejects_var.get())
                    ),
                }

            self.log(f"Done! {json.dumps(stats, indent=2)}")

            # Run COLMAP geometric validation if enabled
            colmap_dir = config.colmap_dir or ""
            if config.colmap_validate and colmap_dir:
                self.log("Running COLMAP geometric validation...")
                try:
                    mask_dir = out / "masks"
                    val_result = pipeline.validate_masks_colmap(
                        masks_dir=str(mask_dir),
                        images_dir=str(inp) if inp.is_dir() else None,
                        colmap_dir=colmap_dir,
                    )
                    if val_result:
                        self.log(f"COLMAP validation: {val_result['flagged_frames']}/{val_result['total_frames']} "
                                 f"frames flagged, avg consistency={val_result['avg_consistency']:.3f}")
                        if val_result.get('flagged_names'):
                            self.log(f"  Flagged: {', '.join(val_result['flagged_names'][:10])}")
                    else:
                        self.log("COLMAP validation: no results (check reconstruction path)")
                except Exception as e:
                    self.log(f"COLMAP validation error: {e}")

            self.after(0, lambda: self.stats_label.configure(
                text=f"Complete. {stats.get('processed_images', stats.get('processed_frames', '?'))} processed."
            ))
            self.after(0, lambda: self.progress_bar.set(1.0))

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.after(0, self._done)

    def _done(self):
        self.is_running = False
        self.run_btn.configure(state="normal")
        self.mq_run_btn.configure(state="normal")
        self.stop_btn.pack_forget()

    # ══════════════════════════════════════════════════════════════════
    # MASKING QUEUE — batch masking across multiple folders
    # ══════════════════════════════════════════════════════════════════

    _MQ_STATUS_COLORS = {
        "pending":    "#888888",
        "processing": "#FFA500",
        "done":       "#4CAF50",
        "error":      "#F44336",
        "cancelled":  "#9E9E9E",
    }

    def _mq_add_current(self):
        """Add the folder from the Input field to the masking queue."""
        folder = self.input_entry.get().strip()
        if not folder:
            self.log("Queue: no input folder set — enter a path in the Input field first.")
            return
        p = Path(folder)
        if not p.is_dir():
            self.log(f"Queue: {folder} is not a directory.")
            return
        item = self.masking_queue.add_folder(folder)
        if item:
            self.log(f"Queue: added {item.folder_name}")
        else:
            self.log(f"Queue: {p.name} already in queue")
        self._mq_refresh()

    def _mq_add_subfolders(self):
        """Pick a parent folder and add all its immediate subfolders to the queue."""
        parent = filedialog.askdirectory(title="Select parent — all subfolders will be added")
        if not parent:
            return
        subfolders = sorted(
            p for p in Path(parent).iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if not subfolders:
            self.log(f"Queue: no subfolders found in {Path(parent).name}")
            return
        count = self.masking_queue.add_folders([str(f) for f in subfolders])
        self.log(f"Queue: added {count} folder(s) from {Path(parent).name}")
        self._mq_refresh()

    def _mq_remove_selected(self):
        """Remove selected items from the masking queue."""
        to_remove = [iid for iid, w in self.mq_item_widgets.items() if w.get("selected")]
        for iid in to_remove:
            self.masking_queue.remove_item(iid)
        self._mq_refresh()

    def _mq_clear_done(self):
        """Clear completed/error/cancelled items from the masking queue."""
        self.masking_queue.clear_completed()
        self._mq_refresh()

    def _mq_refresh(self):
        """Rebuild the masking queue widget list."""
        for w in self.mq_item_widgets.values():
            if "frame" in w:
                w["frame"].destroy()
        self.mq_item_widgets = {}

        for item in self.masking_queue.items:
            self._mq_create_item(item)

        n = len(self.masking_queue.items)
        if n == 0:
            self.mq_scroll.pack_forget()
        else:
            row_h = 34
            target_h = min(n, 5) * row_h
            self.mq_scroll.configure(height=target_h)
            self.mq_scroll.pack(fill="x", pady=(0, 4))

        stats = self.masking_queue.get_stats()
        self.mq_stats_label.configure(
            text=(f"Queue: {stats['pending']} pending, {stats['processing']} processing, "
                  f"{stats['done']} done, {stats['error']} errors")
        )

    def _mq_create_item(self, item):
        """Create a widget row for a masking queue item."""
        frame = ctk.CTkFrame(self.mq_scroll, fg_color="#2b2b2b", corner_radius=5)
        frame.pack(fill="x", pady=2, padx=2)

        selected = False

        def toggle(event=None):
            nonlocal selected
            selected = not selected
            self.mq_item_widgets[item.id]["selected"] = selected
            frame.configure(fg_color="#3d5a80" if selected else "#2b2b2b")

        frame.bind("<Button-1>", toggle)

        color = self._MQ_STATUS_COLORS.get(item.status, "#888")
        ctk.CTkLabel(frame, text="\u25cf", text_color=color, width=18).pack(side="left", padx=(8, 4))

        display_name = item.folder_name
        if len(display_name) > 28:
            display_name = display_name[:12] + "\u2026" + display_name[-12:]
        name_lbl = ctk.CTkLabel(frame, text=display_name, anchor="w", width=180)
        name_lbl.pack(side="left", padx=(0, 6))
        name_lbl.bind("<Button-1>", toggle)

        status_text = item.status.capitalize()
        if item.status == "processing":
            status_text = f"Processing {item.progress}%"
        elif item.status == "done":
            status_text = f"Done ({item.processed_count} images)"
        elif item.status == "error":
            status_text = "Error"

        st_lbl = ctk.CTkLabel(frame, text=status_text, text_color=color, width=130)
        st_lbl.pack(side="left")

        progress_bar = None
        if item.status == "processing":
            progress_bar = ctk.CTkProgressBar(frame, width=80, height=8)
            progress_bar.set(item.progress / 100)
            progress_bar.pack(side="left", padx=(4, 0))

        self.mq_item_widgets[item.id] = {
            "frame": frame,
            "status_label": st_lbl,
            "progress": progress_bar,
            "selected": selected,
        }

    def _mq_update_item(self, item_id):
        """Update status/progress display for a masking queue item."""
        item = self.masking_queue.get_item(item_id)
        w = self.mq_item_widgets.get(item_id)
        if not item or not w:
            return
        color = self._MQ_STATUS_COLORS.get(item.status, "#888")
        status_text = item.status.capitalize()
        if item.status == "processing":
            status_text = f"Processing {item.progress}%"
        elif item.status == "done":
            status_text = f"Done ({item.processed_count} images)"
        w["status_label"].configure(text=status_text, text_color=color)
        if w["progress"] and item.status == "processing":
            w["progress"].set(item.progress / 100)

    def _run_masking_queue(self):
        """Start processing the masking queue."""
        if self.is_running or self.mq_processing:
            self.log("A process is already running.")
            return

        pending = self.masking_queue.get_pending_count()
        if pending == 0:
            self.log("Queue: no pending folders to process.")
            return

        self.mq_processing = True
        self.is_running = True
        self.cancel_flag.clear()
        self.run_btn.configure(state="disabled")
        self.mq_run_btn.configure(state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.progress_bar.set(0)
        self.stats_label.configure(text=f"Queue: loading model for {pending} folders...")

        threading.Thread(target=self._masking_queue_worker, daemon=True).start()

    def _masking_queue_worker(self):
        """Worker thread: process all pending folders in the masking queue."""
        try:
            import numpy as np
            MaskingPipeline = _import_pipeline()[0]
            config, model_str, geometry = self._build_mask_config()
            pattern = self.pattern_var.get()
            skip_existing = self.skip_existing_var.get()

            self.log(f"Queue: Model={model_str} | Geometry={self.geometry_var.get()} | "
                     f"Confidence={config.confidence_threshold}")
            if config.remove_prompts:
                self.log(f"Queue: Remove prompts: {config.remove_prompts}")

            # Load pipeline once — reuse across all folders
            pipeline = MaskingPipeline(config=config, auto_select_model=(model_str == "auto"))

            folder_idx = 0
            total_pending = self.masking_queue.get_pending_count()

            while not self.cancel_flag.is_set():
                item = self.masking_queue.get_next_pending()
                if not item:
                    break

                folder_idx += 1
                self.masking_queue.set_processing(item.id)
                self.after(0, self._mq_refresh)

                input_dir = Path(item.folder_path)
                masks_dir = input_dir.parent / f"{input_dir.name}_masks"
                review_dir = input_dir.parent / f"{input_dir.name}_review"

                self.log(f"Queue [{folder_idx}/{total_pending}]: {item.folder_name}")
                self.log(f"  Masks  → {masks_dir}")
                self.log(f"  Review → {review_dir}")

                # Count images for progress tracking
                image_files = []
                for pat in pattern.split():
                    image_files.extend(input_dir.glob(pat))
                image_files = sorted(set(image_files))
                if not image_files:
                    for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"):
                        image_files = sorted(input_dir.glob(ext))
                        if image_files:
                            break
                image_count = len(image_files)
                self.masking_queue.update_item(item.id, image_count=image_count)

                self.after(0, lambda f=folder_idx, t=total_pending, n=item.folder_name, c=image_count:
                    self.stats_label.configure(
                        text=f"Queue [{f}/{t}]: {n} — 0/{c} images"))

                # Progress callback via result_callback
                processed = [0]  # use list for closure mutability

                def on_result(stem, result, _iid=item.id, _count=image_count,
                              _fidx=folder_idx, _total=total_pending, _name=item.folder_name):
                    processed[0] += 1
                    pct = int(processed[0] / max(_count, 1) * 100)
                    self.masking_queue.set_progress(_iid, pct)
                    self.after(0, lambda: self._mq_update_item(_iid))
                    self.after(0, lambda p=processed[0], c=_count, f=_fidx, t=_total, n=_name:
                        self.stats_label.configure(
                            text=f"Queue [{f}/{t}]: {n} — {p}/{c} images"))

                # Review status bridge
                try:
                    ReviewStatusManager = _import_review()[4]
                    _status_mgr = ReviewStatusManager(masks_dir)
                    def on_result_with_review(stem, result, _iid=item.id, _count=image_count,
                                              _fidx=folder_idx, _total=total_pending,
                                              _name=item.folder_name):
                        area = float(np.sum(result.mask > 0) / result.mask.size * 100)
                        _status_mgr.set_quality_info(
                            stem, quality=result.quality.value,
                            confidence=result.confidence, area_percent=area)
                        on_result(stem, result, _iid, _count, _fidx, _total, _name)
                    result_cb = on_result_with_review
                except Exception:
                    result_cb = on_result

                try:
                    stats = pipeline.process_directory(
                        input_dir=input_dir,
                        output_dir=masks_dir,
                        mask_dir=masks_dir,
                        review_dir=review_dir,
                        geometry=geometry,
                        pattern=pattern,
                        result_callback=result_cb,
                        skip_existing=skip_existing,
                        cancel_event=self.cancel_flag,
                    )
                    done_count = stats.get('processed_images', 0)
                    self.masking_queue.set_done(item.id, processed_count=done_count)
                    self.log(f"  Done: {done_count} processed, "
                             f"{stats.get('review_images', 0)} for review, "
                             f"{stats.get('rejected_images', 0)} rejected")
                except Exception as e:
                    self.masking_queue.set_error(item.id, str(e))
                    self.log(f"  ERROR: {e}")
                    import traceback
                    self.log(traceback.format_exc())

                if self.cancel_flag.is_set():
                    # Mark current item as cancelled if it was still processing
                    current = self.masking_queue.get_item(item.id)
                    if current and current.status == "processing":
                        self.masking_queue.set_cancelled(item.id)
                    break

                self.after(0, self._mq_refresh)

            # Summary
            qs = self.masking_queue.get_stats()
            self.log(f"Queue complete: {qs['done']} done, {qs['error']} errors, "
                     f"{qs['cancelled']} cancelled, {qs['pending']} remaining")
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.stats_label.configure(
                text=f"Queue complete: {qs['done']} done, {qs['error']} errors"))

        except Exception as e:
            self.log(f"Queue ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.after(0, self._masking_queue_done)

    def _masking_queue_done(self):
        """Re-enable UI after masking queue finishes."""
        self.mq_processing = False
        self.is_running = False
        self.run_btn.configure(state="normal")
        self.mq_run_btn.configure(state="normal")
        self.stop_btn.pack_forget()
        self._mq_refresh()

    # ══════════════════════════════════════════════════════════════════
    # REVIEW TAB — compact controls, shared preview panel
    # ══════════════════════════════════════════════════════════════════

    def _build_review_tab(self):
        tab = self.tabs.tab("Review")
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=5, pady=5)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(2, weight=1)  # Thumbnails row fills remaining space

        # ── Data Source ──
        ds = _Section(container, "Data Source", subtitle="Masks + source images")
        ds.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        dsc = ds.content

        row = ctk.CTkFrame(dsc, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="Masks:", width=52, anchor="e").pack(side="left")
        self.masks_entry = ctk.CTkEntry(row, placeholder_text="Directory containing mask files")
        self.masks_entry.pack(side="left", fill="x", expand=True, padx=4)
        _w = ctk.CTkButton(row, text="Browse", width=60,
                           command=lambda: self._browse_dir_into(self.masks_entry))
        _w.pack(side="left")
        Tooltip(_w, "Select the folder containing mask PNG files.")

        row2 = ctk.CTkFrame(dsc, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkLabel(row2, text="Images:", width=52, anchor="e").pack(side="left")
        self.images_entry = ctk.CTkEntry(row2, placeholder_text="Directory containing source images")
        self.images_entry.pack(side="left", fill="x", expand=True, padx=4)
        _w = ctk.CTkButton(row2, text="Browse", width=60,
                           command=lambda: self._browse_dir_into(self.images_entry))
        _w.pack(side="left")
        Tooltip(_w, "Select the folder containing source images.\nUsed for overlay display in thumbnails.")

        btn_row = ctk.CTkFrame(dsc, fg_color="transparent")
        btn_row.pack(fill="x", pady=(2, 0))
        _w = ctk.CTkButton(btn_row, text="Load Masks", width=110, fg_color="#2563eb",
                           hover_color="#1d4ed8", command=self._load_review)
        _w.pack(side="left", padx=(0, 8))
        Tooltip(_w, "Load mask/image pairs from the specified directories.\nPopulates the thumbnail grid below.")
        _w = ctk.CTkButton(btn_row, text="Auto-detect from Output", width=170,
                           fg_color="#6b7280", hover_color="#4b5563",
                           command=self._auto_detect_review_paths)
        _w.pack(side="left")
        Tooltip(_w, "Auto-fill mask and image paths from the Mask tab output.\nLooks for masks/ and images/ subdirectories.")

        # ── Filter & Sort ──
        fs = _Section(container, "Filter & Sort")
        fs.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fsc = fs.content

        fs_row = ctk.CTkFrame(fsc, fg_color="transparent")
        fs_row.pack(fill="x", pady=2)
        ctk.CTkLabel(fs_row, text="Sort:").pack(side="left", padx=(0, 2))
        self.sort_var = ctk.StringVar(value="Filename")
        _w = ctk.CTkOptionMenu(fs_row, variable=self.sort_var,
                               values=["Filename", "Confidence", "Quality", "Area %"],
                               command=lambda _: self._on_filter_change(), width=110)
        _w.pack(side="left", padx=2)
        Tooltip(_w, "Sort order for the thumbnail grid.\nConfidence/Quality sort by mask quality score.\nArea % sorts by how much of the image is masked.")

        ctk.CTkLabel(fs_row, text="Filter:").pack(side="left", padx=(8, 2))
        self.filter_var = ctk.StringVar(value="All")
        _w = ctk.CTkSegmentedButton(
            fs_row, values=["All", "Needs Review", "Poor", "Unreviewed"],
            variable=self.filter_var, command=lambda _: self._on_filter_change()
        )
        _w.pack(side="left", padx=2)

        # ── Thumbnails (2-column grid, virtualized loading) ──
        ts = _Section(container, "Thumbnails")
        ts.grid(row=2, column=0, sticky="nsew", pady=(0, 6))
        # Override section content to fill vertically
        ts.content.pack_forget()
        ts.content.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._thumb_scroll = ctk.CTkScrollableFrame(ts.content)
        self._thumb_scroll.pack(fill="both", expand=True)
        self._current_thumb_row = None  # packed row frame for 2-column layout
        self._thumb_loading_label = ctk.CTkLabel(
            self._thumb_scroll, text="Load masks to see thumbnails.",
            font=("Consolas", 10), text_color="#9ca3af",
        )
        self._thumb_loading_label.pack(pady=4)

        # ── Current Mask ──
        cm = _Section(container, "Current Mask")
        cm.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        cmc = cm.content

        self._review_info_label = ctk.CTkLabel(
            cmc, text="Load masks to begin.",
            font=("Consolas", 11), anchor="w", justify="left",
        )
        self._review_info_label.pack(fill="x", padx=4, pady=2)

        action_row = ctk.CTkFrame(cmc, fg_color="transparent")
        action_row.pack(fill="x", pady=(2, 0))
        _w = ctk.CTkButton(action_row, text="Edit (OpenCV)", command=self._on_edit,
                           fg_color="#2563eb", width=110)
        _w.pack(side="left", padx=(0, 4))
        Tooltip(_w, "Open the selected mask in the OpenCV editor.\nFull brush, lasso, and flood-fill tools for manual refinement.")
        _tooltips = {
            "Accept": "Mark this mask as accepted (good quality).",
            "Reject": "Mark this mask as rejected (needs replacement or deletion).",
            "Skip": "Skip this mask without marking it.\nLeaves it in 'unreviewed' state.",
        }
        for text, cmd, color in [
            ("Accept", self._on_accept, "#16a34a"),
            ("Reject", self._on_reject, "#dc2626"),
            ("Skip", self._on_skip, "#6b7280"),
        ]:
            _w = ctk.CTkButton(action_row, text=text, command=cmd, fg_color=color,
                               width=75)
            _w.pack(side="left", padx=3)
            Tooltip(_w, _tooltips[text])

        # ── Batch Actions ──
        ba = _Section(container, "Batch")
        ba.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        bac = ba.content

        _w = ctk.CTkButton(bac, text="Accept All Good", command=self._on_accept_all_good,
                           fg_color="#065f46", width=140)
        _w.pack(side="left", padx=4, pady=2)
        Tooltip(_w, "Accept all masks with quality scores above the review threshold.\nBatch-approves masks that passed automated scoring.")
        _w = ctk.CTkButton(bac, text="Hide Done", command=self._on_hide_done,
                           fg_color="#374151", width=100)
        _w.pack(side="left", padx=4, pady=2)
        Tooltip(_w, "Hide accepted and rejected masks from the thumbnail grid.\nShows only masks still needing review.")

        self._review_summary_label = ctk.CTkLabel(
            bac, text="", font=("Consolas", 10), text_color="#9ca3af",
        )
        self._review_summary_label.pack(side="left", padx=8, pady=2)

    # ── review loading ──

    def _auto_detect_review_paths(self):
        out = self.output_entry.get().strip()
        if not out:
            self.log("Set an output directory on the Process tab first.")
            return
        out_path = Path(out)

        masks_dir = None
        for c in [out_path / "masks", out_path]:
            if c.exists() and any(c.glob("*.png")):
                masks_dir = c
                break
        images_dir = None
        for c in [out_path / "images", out_path / "frames",
                  out_path.parent / "frames", out_path.parent / "images"]:
            if c.exists():
                images_dir = c
                break
        if images_dir is None:
            inp = self.input_entry.get().strip()
            if inp and Path(inp).is_dir():
                images_dir = Path(inp)

        if masks_dir:
            self.masks_entry.delete(0, "end")
            self.masks_entry.insert(0, str(masks_dir))
        else:
            self.log("Could not auto-detect masks directory.")
        if images_dir:
            self.images_entry.delete(0, "end")
            self.images_entry.insert(0, str(images_dir))
        else:
            self.log("Could not auto-detect images directory.")
        if masks_dir and images_dir:
            self._load_review()

    def _load_review(self):
        masks_dir = self.masks_entry.get().strip()
        images_dir = self.images_entry.get().strip()

        if not masks_dir or not images_dir:
            self.log("Set both masks and images directories, then click Load.")
            return

        masks_path = Path(masks_dir)
        images_path = Path(images_dir)

        if not masks_path.exists():
            self.log(f"Masks dir not found: {masks_path}")
            return
        if not images_path.exists():
            self.log(f"Images dir not found: {images_path}")
            return

        self._prefs["masks_dir"] = masks_dir
        self._prefs["images_dir"] = images_dir
        self._save_prefs()

        try:
            (self._load_overlay_thumbnail, self._compute_mask_area_percent,
             self._quality_colors, self._status_colors,
             ReviewStatusManager, _) = _import_review()

            self._review_status_mgr = ReviewStatusManager(masks_path)
            self._review_pairs = []
            self._discover_pairs(masks_path, images_path)
            self._review_loaded = True
            self._selected_stem = None

            self.log(f"Loaded {len(self._review_pairs)} mask/image pairs")

            # Compute area % in background to avoid lockup on large sets
            threading.Thread(target=self._compute_areas_bg, daemon=True).start()

            # Load thumbnails in background (clear old cache + widgets)
            self._thumb_cache.clear()
            self._thumb_widgets.clear()
            threading.Thread(target=self._load_thumbnails_bg, daemon=True).start()

            # Set zoom to 100% = fit to panel width
            self._zoom_var.set(100)
            self._zoom_label.configure(text="100%")

            # If we're on the Review tab, refresh the navigator
            if self._preview_mode == "review":
                self._load_review_nav_list()

        except Exception as e:
            self.log(f"Failed to load review: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _compute_areas_bg(self):
        """Compute mask area percentages in background thread."""
        for pair in self._review_pairs:
            ms = self._review_status_mgr.get(pair["stem"])
            if ms.area_percent == 0.0:
                ms.area_percent = self._compute_mask_area_percent(pair["mask_path"])
        self._review_status_mgr.save()

    # ── thumbnails ──

    def _load_thumbnails_bg(self):
        """Load overlay thumbnails for the first visible batch, then on scroll."""
        THUMB_SZ = 300
        # Pre-load first ~40 thumbnails (visible + buffer) to fill screen
        pairs = list(self._filtered_pairs) if self._filtered_pairs else list(self._review_pairs)
        for i, pair in enumerate(pairs[:40]):
            try:
                pil = self._load_overlay_thumbnail(
                    pair["image_path"], pair["mask_path"], size=THUMB_SZ,
                    pad_to_square=False)
            except Exception:
                pil = None
            if pil is not None:
                self._thumb_cache[pair["stem"]] = pil  # PIL.Image
        self.after(0, self._rebuild_thumb_list)

    def _load_thumb_for_stem(self, stem):
        """Load a single thumbnail PIL image into cache (background thread)."""
        if stem in self._thumb_cache:
            return
        pair = next((p for p in self._review_pairs if p["stem"] == stem), None)
        if not pair:
            return
        try:
            pil = self._load_overlay_thumbnail(
                pair["image_path"], pair["mask_path"], size=300,
                pad_to_square=False)
        except Exception:
            pil = None
        if pil is not None:
            self._thumb_cache[stem] = pil  # PIL.Image

    def _create_thumb_cell(self, stem, grid_col):
        """Create a single thumbnail cell in a 2-column row."""
        # Create new row frame when starting column 0
        if grid_col == 0:
            self._current_thumb_row = ctk.CTkFrame(self._thumb_scroll, fg_color="transparent")
            self._current_thumb_row.pack(fill="x", pady=1)
            self._current_thumb_row.grid_columnconfigure(0, weight=1)
            self._current_thumb_row.grid_columnconfigure(1, weight=1)

        ms = self._review_status_mgr.get(stem) if self._review_status_mgr else None
        qc = getattr(self, '_quality_colors', {})
        sc = getattr(self, '_status_colors', {})

        border_color = "#6b7280"
        if ms:
            border_color = sc.get(ms.status, qc.get(ms.quality, "#6b7280"))

        cell = ctk.CTkFrame(self._current_thumb_row, border_width=2,
                            border_color=border_color, corner_radius=2)
        cell.grid(row=0, column=grid_col, sticky="new", padx=2, pady=1)

        pil_img = self._thumb_cache.get(stem)
        if pil_img is not None:
            from PIL import Image as PILImage, ImageTk
            # Compute display width from scroll frame (half minus padding)
            try:
                scroll_w = self._thumb_scroll.winfo_width()
                cell_w = max(80, scroll_w // 2 - 12) if scroll_w > 100 else 200
            except Exception:
                cell_w = 200
            aspect = pil_img.height / max(pil_img.width, 1)
            display_h = max(1, int(cell_w * aspect))
            # Resize PIL image to exact pixel dimensions — no DPI scaling
            thumb_resized = pil_img.resize((cell_w, display_h), PILImage.LANCZOS)
            photo = ImageTk.PhotoImage(thumb_resized)
            # Raw tkinter.Label has ZERO internal padding (unlike CTkLabel)
            img_lbl = tk.Label(cell, image=photo, bd=0, highlightthickness=0)
            img_lbl.pack(padx=0, pady=0)
            img_lbl._photo = photo  # prevent GC
            img_lbl.bind("<Button-1>", lambda e, s=stem: self._on_thumb_click(s))

        # Stem label (truncate long names)
        ctk.CTkLabel(cell, text=stem[:20], font=("Consolas", 9),
                     anchor="w", height=16).pack(fill="x", padx=3)

        # Status line — only show if meaningful data exists
        if ms:
            parts = []
            if ms.quality and ms.quality != "unknown":
                parts.append(ms.quality)
            if ms.confidence > 0:
                parts.append(f"{ms.confidence:.0%}")
            if ms.status and ms.status != "pending":
                parts.append(ms.status)
            if parts:
                ctk.CTkLabel(cell, text="  ".join(parts),
                             font=("Consolas", 8), text_color="#9ca3af",
                             anchor="w", height=14).pack(fill="x", padx=3)

        cell.bind("<Button-1>", lambda e, s=stem: self._on_thumb_click(s))
        self._thumb_widgets[stem] = cell
        return cell

    def _on_thumb_click(self, stem):
        """Jump navigator to the clicked thumbnail."""
        idx = next(
            (i for i, p in enumerate(self._filtered_pairs) if p["stem"] == stem),
            None,
        )
        if idx is not None:
            self._nav_idx.set(idx)
            self._on_nav_change(idx)
            # Highlight selected thumbnail
            for s, w in self._thumb_widgets.items():
                try:
                    if s == stem:
                        w.configure(border_color="#3b82f6")
                    else:
                        ms = self._review_status_mgr.get(s) if self._review_status_mgr else None
                        qc = getattr(self, '_quality_colors', {})
                        sc = getattr(self, '_status_colors', {})
                        bc = "#6b7280"
                        if ms:
                            bc = sc.get(ms.status, qc.get(ms.quality, "#6b7280"))
                        w.configure(border_color=bc)
                except Exception:
                    pass

    def _rebuild_thumb_list(self):
        """Rebuild visible thumbnail list from cache (no disk I/O)."""
        for w in self._thumb_scroll.winfo_children():
            w.destroy()
        self._thumb_widgets.clear()
        self._current_thumb_row = None
        if not self._filtered_pairs:
            ctk.CTkLabel(self._thumb_scroll, text="No masks match filter.",
                         font=("Consolas", 10), text_color="#9ca3af",
                         ).pack(pady=4)
            return

        # Only create cells for pairs that already have cached thumbnails
        self._thumb_cell_count = 0
        for pair in self._filtered_pairs:
            stem = pair["stem"]
            if stem in self._thumb_cache:
                col = self._thumb_cell_count % 2
                self._create_thumb_cell(stem, col)
                self._thumb_cell_count += 1

        # Load uncached thumbnails lazily in background batches
        uncached = [p["stem"] for p in self._filtered_pairs if p["stem"] not in self._thumb_cache]
        if uncached:
            self._thumb_loading_label = ctk.CTkLabel(
                self._thumb_scroll,
                text=f"{self._thumb_cell_count}/{len(self._filtered_pairs)} loaded...",
                font=("Consolas", 9), text_color="#9ca3af",
            )
            self._thumb_loading_label.pack(pady=2)
            threading.Thread(
                target=self._lazy_load_remaining, args=(uncached,), daemon=True
            ).start()

    def _lazy_load_remaining(self, stems):
        """Background thread: load remaining thumbnails in small batches."""
        BATCH = 10
        for i in range(0, len(stems), BATCH):
            batch = stems[i:i + BATCH]
            for stem in batch:
                self._load_thumb_for_stem(stem)
            loaded = min(i + BATCH, len(stems))
            total_cached = len(self._thumb_cache)
            total_pairs = len(self._filtered_pairs)
            # Update widgets on main thread
            batch_copy = list(batch)
            self.after(0, lambda b=batch_copy, tc=total_cached, tp=total_pairs:
                       self._update_thumb_batch(b, tc, tp))
        self.after(0, self._finish_thumb_loading)

    def _update_thumb_batch(self, stems, total_cached, total_pairs):
        """Main thread: add new thumb cells for just-loaded images."""
        filtered_set = {p["stem"] for p in self._filtered_pairs}
        for stem in stems:
            if stem not in filtered_set or stem in self._thumb_widgets:
                continue
            pil_img = self._thumb_cache.get(stem)
            if pil_img:
                col = self._thumb_cell_count % 2
                self._create_thumb_cell(stem, col)
                self._thumb_cell_count += 1
        if hasattr(self, '_thumb_loading_label') and self._thumb_loading_label.winfo_exists():
            self._thumb_loading_label.configure(
                text=f"{self._thumb_cell_count}/{total_pairs} loaded...")

    def _finish_thumb_loading(self):
        """Remove loading label when all thumbnails are loaded."""
        if hasattr(self, '_thumb_loading_label') and self._thumb_loading_label.winfo_exists():
            self._thumb_loading_label.destroy()

    def _discover_pairs(self, masks_dir: Path, images_dir: Path):
        """Find mask/image pairs (prefixed or same-stem)."""
        # Strategy 1: prefixed
        mask_files = sorted(
            list(masks_dir.glob("mask_*.png")) +
            list(masks_dir.glob("*_mask.png"))
        )
        for mf in mask_files:
            stem = mf.stem.replace("mask_", "").replace("_mask", "")
            img = self._find_image(images_dir, stem)
            if img:
                self._review_pairs.append({"stem": stem, "mask_path": mf, "image_path": img})

        # Strategy 2: same-stem
        if not self._review_pairs and masks_dir != images_dir:
            image_stems = {}
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                for f in images_dir.glob(ext):
                    image_stems[f.stem] = f

            seen = set()
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for mf in sorted(masks_dir.glob(ext)):
                    if mf.stem in image_stems and mf.stem not in seen:
                        seen.add(mf.stem)
                        self._review_pairs.append({
                            "stem": mf.stem,
                            "mask_path": mf,
                            "image_path": image_stems[mf.stem],
                        })

    def _find_image(self, images_dir: Path, stem: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    # ── filtering ──

    def _on_filter_change(self):
        if self._preview_mode == "review":
            self._load_review_nav_list()
            self._rebuild_thumb_list()

    def _apply_filter(self):
        """Apply sort/filter to _review_pairs → _filtered_pairs."""
        if not self._review_loaded:
            self._filtered_pairs = []
            return

        pairs = list(self._review_pairs)
        filt = self.filter_var.get()
        if filt == "Needs Review":
            pairs = [p for p in pairs if self._review_status_mgr.get(p["stem"]).quality in ("review", "poor")]
        elif filt == "Poor":
            pairs = [p for p in pairs if self._review_status_mgr.get(p["stem"]).quality == "poor"]
        elif filt == "Unreviewed":
            pairs = [p for p in pairs if self._review_status_mgr.get(p["stem"]).status == "pending"]

        sort_key = self.sort_var.get()
        if sort_key == "Confidence":
            pairs.sort(key=lambda p: self._review_status_mgr.get(p["stem"]).confidence)
        elif sort_key == "Quality":
            order = {"reject": 0, "poor": 1, "review": 2, "good": 3, "excellent": 4, "": -1}
            pairs.sort(key=lambda p: order.get(self._review_status_mgr.get(p["stem"]).quality, -1))
        elif sort_key == "Area %":
            pairs.sort(key=lambda p: self._review_status_mgr.get(p["stem"]).area_percent, reverse=True)
        else:
            pairs.sort(key=lambda p: p["stem"])

        self._filtered_pairs = pairs

    # ── actions ──

    def _on_edit(self):
        if not self._review_loaded or not self._filtered_pairs:
            return
        idx = self._nav_idx.get()
        if idx >= len(self._filtered_pairs):
            return
        pair = self._filtered_pairs[idx]

        # If editor subprocess already running, send new image via cmd file
        if self._editor_proc is not None and self._editor_proc.poll() is None:
            try:
                self._editor_cmd_file.write_text(
                    f"{pair['image_path']}|{pair['mask_path']}|{pair['stem']}\n",
                    encoding="utf-8")
            except Exception as e:
                self.log(f"Editor send failed: {e}")
            return

        # Launch editor as subprocess (OpenCV needs its own main thread)
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        self._editor_cmd_file = tmp_dir / "reconstruction_zone_editor_cmd.txt"
        self._editor_signal_file = tmp_dir / "reconstruction_zone_editor_signal.txt"
        self._editor_cmd_file.write_text("", encoding="utf-8")
        self._editor_signal_file.write_text("", encoding="utf-8")

        # Find review_masks.py
        script = Path(__file__).parent / "review_masks.py"
        if not script.exists():
            self.log("Could not find review_masks.py")
            return

        import subprocess
        self._editor_proc = subprocess.Popen(
            [sys.executable, str(script),
             str(pair["image_path"]), str(pair["mask_path"]),
             "--cmd-file", str(self._editor_cmd_file),
             "--signal-file", str(self._editor_signal_file),
             "--window-name", f"Edit: {pair['stem']}"],
        )
        self.log(f"Editor opened: {pair['stem']}")
        # Start polling for signals from editor
        self._editor_poll_id = self.after(200, self._poll_editor_signals)

    def _poll_editor_signals(self):
        """Poll the signal file for save/close events from the editor subprocess."""
        self._editor_poll_id = None

        # Check if process exited
        if self._editor_proc is not None and self._editor_proc.poll() is not None:
            # Process any remaining signals before cleanup
            self._read_editor_signals()
            self._on_editor_closed()
            return

        self._read_editor_signals()
        self._editor_poll_id = self.after(200, self._poll_editor_signals)

    def _read_editor_signals(self):
        """Read and process any signals written by the editor subprocess."""
        if self._editor_signal_file is None:
            return
        try:
            content = self._editor_signal_file.read_text(encoding="utf-8").strip()
            if not content:
                return
            self._editor_signal_file.write_text("", encoding="utf-8")
            for line in content.splitlines():
                if line.startswith("saved_modified|"):
                    mask_path = Path(line.split("|", 1)[1])
                    self._on_editor_save(mask_path, modified=True)
                elif line.startswith("saved_unchanged|"):
                    mask_path = Path(line.split("|", 1)[1])
                    self._on_editor_save(mask_path, modified=False)
                elif line.startswith("saved|"):
                    # Legacy fallback
                    mask_path = Path(line.split("|", 1)[1])
                    self._on_editor_save(mask_path, modified=True)
        except Exception:
            pass

    def _on_editor_save(self, mask_path, modified=True):
        """Called on main thread after editor saves a mask. Advances to next image."""
        mask_path = Path(mask_path)
        stem = next((p["stem"] for p in self._review_pairs
                     if p["mask_path"] == mask_path), None)
        if stem and self._review_status_mgr:
            status = "edited" if modified else "accepted"
            self._review_status_mgr.record_action(stem, status)
        # Invalidate thumb cache so it reloads on next rebuild
        if stem and stem in self._thumb_cache:
            del self._thumb_cache[stem]

        # Advance navigator to next image and push it to the editor
        idx = self._nav_idx.get()
        if idx + 1 < len(self._filtered_pairs):
            next_idx = idx + 1
            self._nav_idx.set(next_idx)
            self._on_nav_change(next_idx)
            next_pair = self._filtered_pairs[next_idx]
            # Push next image to editor subprocess
            if self._editor_cmd_file is not None:
                try:
                    self._editor_cmd_file.write_text(
                        f"{next_pair['image_path']}|{next_pair['mask_path']}|{next_pair['stem']}\n",
                        encoding="utf-8")
                except Exception:
                    pass

    def _on_editor_closed(self):
        """Called when the editor subprocess exits."""
        if self._editor_poll_id is not None:
            self.after_cancel(self._editor_poll_id)
            self._editor_poll_id = None
        self._editor_proc = None
        # Clean up temp files
        for f in (self._editor_cmd_file, self._editor_signal_file):
            if f is not None:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        self._editor_cmd_file = None
        self._editor_signal_file = None
        self._refresh_review()

    def _refresh_review(self):
        if self._preview_mode == "review":
            self._load_review_nav_list()
            self._rebuild_thumb_list()

    def _on_accept(self):
        if self._selected_stem and self._review_status_mgr:
            self._review_status_mgr.record_action(self._selected_stem, "accepted")
            self._advance()

    def _on_reject(self):
        if self._selected_stem and self._review_status_mgr:
            self._review_status_mgr.record_action(self._selected_stem, "rejected")
            self._advance()

    def _on_skip(self):
        if self._selected_stem and self._review_status_mgr:
            self._review_status_mgr.record_action(self._selected_stem, "skipped")
            self._advance()

    def _on_hide_done(self):
        """Filter out accepted and edited masks from the thumbnail grid."""
        if not self._review_loaded:
            return
        self._filtered_pairs = [
            p for p in self._filtered_pairs
            if self._review_status_mgr.get(p["stem"]).status not in ("accepted", "edited")
        ]
        self._rebuild_thumb_list()
        if self._filtered_pairs:
            self._nav_idx.set(0)
            self._on_nav_change(0)
        self._update_review_info()
        done = len(self._review_pairs) - len(self._filtered_pairs)
        self.log(f"Hidden {done} accepted/edited — {len(self._filtered_pairs)} remaining")

    def _on_accept_all_good(self):
        if not self._review_loaded:
            return
        count = 0
        for pair in self._review_pairs:
            ms = self._review_status_mgr.get(pair["stem"])
            if ms.status == "pending" and ms.quality in ("good", "excellent"):
                self._review_status_mgr.record_action(pair["stem"], "accepted")
                count += 1
        self.log(f"Accepted {count} good/excellent masks")
        self._refresh_review()

    def _advance(self):
        """Move navigator to next pending mask."""
        idx = self._nav_idx.get()
        for i in range(idx + 1, len(self._filtered_pairs)):
            if self._review_status_mgr.get(self._filtered_pairs[i]["stem"]).status == "pending":
                self._nav_idx.set(i)
                self._on_nav_change(i)
                self._update_review_info()
                return
        # No more pending — stay on current, just refresh info
        self._update_review_info()


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ReconstructionZone()
    app.mainloop()

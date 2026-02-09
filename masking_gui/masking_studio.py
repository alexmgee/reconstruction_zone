#!/usr/bin/env python3
"""
Masking Studio
==============
GUI application for batch mask generation and review. Double-click to launch.

Tabs:
  Process — all settings inline, run batch masking
  Review  — paginated thumbnail grid, large preview, OpenCV editor launch

No CLI arguments required. All configuration via the GUI.
"""

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, List, Dict
import threading
import queue
import sys
import os
import json
import logging

# Ensure masking_gui is importable
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_this_dir / "claude") not in sys.path:
    sys.path.insert(0, str(_this_dir / "claude"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- lazy imports ----------

def _import_pipeline():
    from masking_v2 import (
        MaskingPipeline, MaskConfig, SegmentationModel,
        ImageGeometry, CLASS_PRESETS, COCO_CLASSES,
    )
    return MaskingPipeline, MaskConfig, SegmentationModel, ImageGeometry, CLASS_PRESETS, COCO_CLASSES


def _import_review():
    from review_gui import (
        load_overlay_thumbnail, compute_mask_area_percent,
        ThumbnailWidget, QUALITY_COLORS, STATUS_COLORS, THUMB_SIZE,
    )
    from review_status import ReviewStatusManager, MaskStatus
    return (load_overlay_thumbnail, compute_mask_area_percent,
            ThumbnailWidget, QUALITY_COLORS, STATUS_COLORS, THUMB_SIZE,
            ReviewStatusManager, MaskStatus)


# ──────────────────────────────────────────────────────────────────────
# Widgets
# ──────────────────────────────────────────────────────────────────────

class _CollapsibleSection(ctk.CTkFrame):
    """Collapsible section with toggle button."""

    def __init__(self, master, title: str, expanded: bool = False, **kwargs):
        super().__init__(master, **kwargs)
        self._expanded = expanded
        self._title = title

        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", pady=(0, 3))
        self.toggle_btn = ctk.CTkButton(
            self.header,
            text=f"{'▼' if expanded else '▶'} {title}",
            anchor="w", fg_color="transparent",
            hover_color=("gray75", "gray25"),
            text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle, height=28,
        )
        self.toggle_btn.pack(fill="x")

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=(15, 0))

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.configure(
            text=f"{'▼' if self._expanded else '▶'} {self._title}"
        )
        if self._expanded:
            self.content.pack(fill="x", padx=(15, 0))
        else:
            self.content.pack_forget()


# ──────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────

ITEMS_PER_PAGE = 60  # thumbnails before pagination kicks in


class MaskingStudio(ctk.CTk):
    """Desktop GUI for the full masking workflow: process → review."""

    _PREFS_FILE = _this_dir / ".studio_prefs.json"

    def __init__(self):
        super().__init__()
        self.title("Masking Studio")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.log_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.cancel_flag = threading.Event()
        self._prefs = self._load_prefs()

        # Review state (lazy)
        self._review_loaded = False
        self._review_pairs: List[Dict] = []
        self._review_status_mgr = None
        self._selected_stem: Optional[str] = None
        self._page = 0
        self._filtered_pairs: List[Dict] = []

        self._build_ui()
        self._setup_console_redirect()
        self._poll_log_queue()
        self._restore_prefs()

    # ── prefs ──

    def _load_prefs(self) -> dict:
        try:
            return json.loads(self._PREFS_FILE.read_text())
        except Exception:
            return {}

    def _save_prefs(self):
        try:
            self._PREFS_FILE.write_text(json.dumps(self._prefs, indent=2))
        except Exception:
            pass

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

    # ── logging ──

    def _setup_console_redirect(self):
        q = self.log_queue

        class _QW:
            def __init__(self):
                self.q = q
            def write(self, text):
                if text.strip():
                    self.q.put(text.rstrip())
            def flush(self):
                pass

        sys.stdout = _QW()
        sys.stderr = _QW()

    def log(self, msg: str):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_textbox.insert("end", msg + "\n")
                self.log_textbox.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    # ── UI construction ──

    def _build_ui(self):
        # Bottom console (fixed height, packed first)
        bottom = ctk.CTkFrame(self, height=130)
        bottom.pack(fill="x", side="bottom", padx=8, pady=(4, 8))
        bottom.pack_propagate(False)

        ctk.CTkLabel(bottom, text="Console", font=("Consolas", 10),
                     anchor="w").pack(fill="x", padx=5, pady=(2, 0))
        self.log_textbox = ctk.CTkTextbox(bottom, font=("Consolas", 10), height=100)
        self.log_textbox.pack(fill="both", expand=True, padx=5, pady=(0, 4))

        # Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.tabs.add("Process")
        self.tabs.add("Review")

        self._build_process_tab()
        self._build_review_tab()

    # ══════════════════════════════════════════════════════════════════
    # PROCESS TAB
    # ══════════════════════════════════════════════════════════════════

    def _build_process_tab(self):
        tab = self.tabs.tab("Process")
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # ---- Input / Output ----
        io_frame = ctk.CTkFrame(scroll)
        io_frame.pack(fill="x", pady=(0, 6))

        row = ctk.CTkFrame(io_frame, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row, text="Input:", width=55, anchor="e").pack(side="left")
        self.input_entry = ctk.CTkEntry(row, placeholder_text="Folder, image, or video")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row, text="Browse", width=65, command=self._browse_input).pack(side="left")

        row2 = ctk.CTkFrame(io_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row2, text="Output:", width=55, anchor="e").pack(side="left")
        self.output_entry = ctk.CTkEntry(row2, placeholder_text="Output directory for masks")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row2, text="Browse", width=65,
                      command=lambda: self._browse_dir_into(self.output_entry)).pack(side="left")

        # ---- Model / Geometry ----
        settings_frame = ctk.CTkFrame(scroll)
        settings_frame.pack(fill="x", pady=(0, 6))

        r1 = ctk.CTkFrame(settings_frame, fg_color="transparent")
        r1.pack(fill="x", padx=10, pady=4)

        ctk.CTkLabel(r1, text="Model:").pack(side="left", padx=(0, 2))
        self.model_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(r1, variable=self.model_var,
                          values=["auto", "yolo26", "sam3", "fastsam"],
                          width=95).pack(side="left", padx=2)

        ctk.CTkLabel(r1, text="YOLO size:").pack(side="left", padx=(12, 2))
        self.yolo_size_var = ctk.StringVar(value="n")
        ctk.CTkOptionMenu(r1, variable=self.yolo_size_var,
                          values=["n", "s", "m", "l", "x"], width=55).pack(side="left", padx=2)

        ctk.CTkLabel(r1, text="Geometry:").pack(side="left", padx=(12, 2))
        self.geometry_var = ctk.StringVar(value="pinhole")
        ctk.CTkOptionMenu(r1, variable=self.geometry_var,
                          values=["pinhole", "equirect", "fisheye", "cubemap"],
                          width=95).pack(side="left", padx=2)

        ctk.CTkLabel(r1, text="Confidence:").pack(side="left", padx=(12, 2))
        self.conf_var = ctk.StringVar(value="0.70")
        ctk.CTkEntry(r1, textvariable=self.conf_var, width=50).pack(side="left", padx=2)

        # ---- Prompts & Classes ----
        prompts_frame = ctk.CTkFrame(scroll)
        prompts_frame.pack(fill="x", pady=(0, 6))

        r2 = ctk.CTkFrame(prompts_frame, fg_color="transparent")
        r2.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(r2, text="Remove:", width=55, anchor="e").pack(side="left")
        self.remove_prompts_entry = ctk.CTkEntry(
            r2, placeholder_text="tripod, camera operator, equipment, shadow of tripod, photographer")
        self.remove_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        r3 = ctk.CTkFrame(prompts_frame, fg_color="transparent")
        r3.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(r3, text="Keep:", width=55, anchor="e").pack(side="left")
        self.keep_prompts_entry = ctk.CTkEntry(
            r3, placeholder_text="(optional) objects to protect from masking")
        self.keep_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        r4 = ctk.CTkFrame(prompts_frame, fg_color="transparent")
        r4.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(r4, text="YOLO preset:").pack(side="left", padx=(0, 2))
        self.preset_var = ctk.StringVar(value="photographer")
        ctk.CTkOptionMenu(r4, variable=self.preset_var,
                          values=["photographer", "person", "equipment",
                                  "vehicles", "animals", "all_dynamic"],
                          width=130).pack(side="left", padx=2)
        ctk.CTkLabel(r4, text="(class filter for YOLO — SAM3 uses the text prompts above)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=8)

        # ---- Shadow Detection (collapsible) ----
        shadow_section = _CollapsibleSection(scroll, "Shadow Detection", expanded=False)
        shadow_section.pack(fill="x", padx=10, pady=(0, 6))
        sc = shadow_section.content

        # Enable + detector selection
        sr1 = ctk.CTkFrame(sc, fg_color="transparent")
        sr1.pack(fill="x", pady=2)
        self.shadow_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sr1, text="Enable", variable=self.shadow_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(sr1, text="Detector:").pack(side="left", padx=(12, 2))
        self.shadow_detector_var = ctk.StringVar(value="brightness")
        ctk.CTkOptionMenu(sr1, variable=self.shadow_detector_var,
                          values=["brightness", "c1c2c3", "hybrid",
                                  "sddnet", "careaga"],
                          width=110).pack(side="left", padx=2)
        ctk.CTkLabel(sr1, text="Verify:").pack(side="left", padx=(12, 2))
        self.shadow_verifier_var = ctk.StringVar(value="none")
        ctk.CTkOptionMenu(sr1, variable=self.shadow_verifier_var,
                          values=["none", "c1c2c3", "hybrid", "brightness"],
                          width=100).pack(side="left", padx=2)

        # Spatial filter + dilation
        sr2 = ctk.CTkFrame(sc, fg_color="transparent")
        sr2.pack(fill="x", pady=2)
        ctk.CTkLabel(sr2, text="Spatial:").pack(side="left")
        self.shadow_spatial_var = ctk.StringVar(value="near_objects")
        ctk.CTkSegmentedButton(sr2, values=["all", "near_objects", "connected"],
                               variable=self.shadow_spatial_var).pack(side="left", padx=4)
        ctk.CTkLabel(sr2, text="Dilation:").pack(side="left", padx=(8, 2))
        self.shadow_dilation_var = ctk.StringVar(value="50")
        ctk.CTkEntry(sr2, textvariable=self.shadow_dilation_var,
                     width=45).pack(side="left")
        ctk.CTkLabel(sr2, text="px", font=("Consolas", 10),
                     text_color="#9ca3af").pack(side="left", padx=2)

        # Thresholds
        sr3 = ctk.CTkFrame(sc, fg_color="transparent")
        sr3.pack(fill="x", pady=2)
        ctk.CTkLabel(sr3, text="Confidence:").pack(side="left")
        self.shadow_conf_var = ctk.StringVar(value="0.50")
        ctk.CTkEntry(sr3, textvariable=self.shadow_conf_var,
                     width=50).pack(side="left", padx=2)
        ctk.CTkLabel(sr3, text="Darkness ratio:").pack(side="left", padx=(12, 2))
        self.shadow_darkness_var = ctk.StringVar(value="0.70")
        ctk.CTkEntry(sr3, textvariable=self.shadow_darkness_var,
                     width=50).pack(side="left", padx=2)
        ctk.CTkLabel(sr3, text="Chromaticity:").pack(side="left", padx=(12, 2))
        self.shadow_chroma_var = ctk.StringVar(value="0.15")
        ctk.CTkEntry(sr3, textvariable=self.shadow_chroma_var,
                     width=50).pack(side="left", padx=2)

        # ---- Options row ----
        opts = ctk.CTkFrame(scroll, fg_color="transparent")
        opts.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(opts, text="Review thresh:").pack(side="left", padx=(0, 2))
        self.review_thresh_var = ctk.StringVar(value="0.85")
        ctk.CTkEntry(opts, textvariable=self.review_thresh_var, width=50).pack(side="left", padx=2)
        ctk.CTkLabel(opts, text="(below = flagged for review)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        ctk.CTkLabel(opts, text="Pattern:").pack(side="left", padx=(15, 2))
        self.pattern_var = ctk.StringVar(value="*.jpg")
        ctk.CTkEntry(opts, textvariable=self.pattern_var, width=70).pack(side="left", padx=2)

        # ---- Run / Cancel ----
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(fill="x", pady=6)

        self.run_btn = ctk.CTkButton(
            btn_row, text="Run Masking", width=140, height=36,
            fg_color="#16a34a", hover_color="#15803d",
            command=self._on_run,
        )
        self.run_btn.pack(side="left", padx=10)

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Cancel", width=80, height=36,
            fg_color="#dc2626", hover_color="#b91c1c",
            command=self._on_stop,
        )

        self.progress_label = ctk.CTkLabel(btn_row, text="", font=("Consolas", 11))
        self.progress_label.pack(side="left", padx=12)

        self.progress_bar = ctk.CTkProgressBar(scroll, height=8)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 4))
        self.progress_bar.set(0)

        self.stats_label = ctk.CTkLabel(
            scroll, text="Set input and output, then click Run Masking.",
            font=("Consolas", 11), anchor="w", justify="left",
        )
        self.stats_label.pack(fill="x", padx=10, pady=4)

    # ── browse helpers ──

    def _browse_input(self):
        """Single Browse button that works for both files and folders."""
        path = filedialog.askopenfilename(
            title="Select Image, Video, or Cancel to Pick Folder",
            filetypes=[
                ("Images & Videos", "*.jpg *.jpeg *.png *.tif *.mp4 *.mov *.avi *.mkv"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)
        else:
            # User cancelled file dialog — offer folder dialog
            path = filedialog.askdirectory(title="Select Input Folder")
            if path:
                self.input_entry.delete(0, "end")
                self.input_entry.insert(0, path)

    def _browse_dir_into(self, entry_widget):
        path = filedialog.askdirectory()
        if path:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, path)

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
        self._save_prefs()

        self.is_running = True
        self.cancel_flag.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.progress_bar.set(0)
        self.stats_label.configure(text="Initializing...")

        threading.Thread(target=self._worker, daemon=True,
                         args=(input_path, output_path)).start()

    def _on_stop(self):
        if self.is_running:
            self.log("Cancelling...")
            self.cancel_flag.set()

    def _worker(self, input_path: str, output_path: str):
        try:
            MaskingPipeline, MaskConfig, SegmentationModel, ImageGeometry, CLASS_PRESETS, _ = _import_pipeline()

            inp = Path(input_path)
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)

            model_str = self.model_var.get()
            model_map = {
                "sam3": SegmentationModel.SAM3,
                "yolo26": SegmentationModel.YOLO26,
                "fastsam": SegmentationModel.FASTSAM,
                "auto": None,
            }

            # Parse text prompts
            raw_remove = self.remove_prompts_entry.get().strip()
            remove_prompts = [p.strip() for p in raw_remove.split(",") if p.strip()] if raw_remove else None
            raw_keep = self.keep_prompts_entry.get().strip()
            keep_prompts = [p.strip() for p in raw_keep.split(",") if p.strip()] if raw_keep else []

            preset_name = self.preset_var.get()
            classes = CLASS_PRESETS.get(preset_name, CLASS_PRESETS["photographer"])

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

            # Build shadow config when enabled and not using plain brightness default
            shadow_config = None
            shadow_enabled = self.shadow_var.get()
            if shadow_enabled:
                det = self.shadow_detector_var.get()
                verifier = self.shadow_verifier_var.get()
                # Build full shadow config for any detector choice
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

            config_kwargs = dict(
                model=model_map.get(model_str),
                device=device,
                yolo_classes=classes,
                detect_shadows=shadow_enabled,
                shadow_config=shadow_config,
                confidence_threshold=float(self.conf_var.get()),
                review_threshold=float(self.review_thresh_var.get()),
                yolo_model_size=self.yolo_size_var.get(),
                save_review_images=True,
                keep_prompts=keep_prompts,
            )
            if remove_prompts is not None:
                config_kwargs["remove_prompts"] = remove_prompts

            config = MaskConfig(**config_kwargs)

            self.log(f"Model: {model_str} | Geometry: {self.geometry_var.get()} | "
                     f"Preset: {preset_name} | Device: {device}")
            if remove_prompts:
                self.log(f"Remove: {remove_prompts}")
            if keep_prompts:
                self.log(f"Keep: {keep_prompts}")

            pipeline = MaskingPipeline(config=config, auto_select_model=(model_str == "auto"))

            if inp.is_dir():
                self.log(f"Processing directory: {inp}")
                stats = pipeline.process_directory(
                    input_dir=inp, output_dir=out,
                    geometry=geometry, pattern=self.pattern_var.get(),
                )
            elif inp.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                self.log(f"Processing video: {inp}")
                stats = pipeline.process_video(
                    video_path=inp, output_dir=out,
                    geometry=geometry, save_review=True,
                )
            else:
                self.log(f"Processing single image: {inp}")
                import cv2, shutil
                image = cv2.imread(str(inp))
                if image is None:
                    self.log(f"ERROR: Failed to load image: {inp}")
                    return
                result = pipeline.process_image(image, geometry)
                mask_dir = out / "masks"
                mask_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(mask_dir / f"mask_{inp.stem}.png"), result.mask * 255)
                images_dir = out / "images"
                images_dir.mkdir(exist_ok=True)
                shutil.copy2(str(inp), str(images_dir / inp.name))
                stats = {"processed_images": 1, "quality": result.quality.value}

            self.log(f"Done! {json.dumps(stats, indent=2)}")
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
        self.stop_btn.pack_forget()

    # ══════════════════════════════════════════════════════════════════
    # REVIEW TAB — paginated grid + large preview
    # ══════════════════════════════════════════════════════════════════

    def _build_review_tab(self):
        tab = self.tabs.tab("Review")

        # ── top: path selectors ──
        top = ctk.CTkFrame(tab)
        top.pack(fill="x", padx=5, pady=(5, 0))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text="Masks:", width=52, anchor="e").pack(side="left")
        self.masks_entry = ctk.CTkEntry(row, placeholder_text="Directory containing mask files")
        self.masks_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row, text="Browse", width=60,
                      command=lambda: self._browse_dir_into(self.masks_entry)).pack(side="left")

        row2 = ctk.CTkFrame(top, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row2, text="Images:", width=52, anchor="e").pack(side="left")
        self.images_entry = ctk.CTkEntry(row2, placeholder_text="Directory containing source images")
        self.images_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row2, text="Browse", width=60,
                      command=lambda: self._browse_dir_into(self.images_entry)).pack(side="left")

        # Buttons on their own row, centered
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(3, 5))
        ctk.CTkButton(btn_row, text="Load Masks", width=110, fg_color="#2563eb",
                      hover_color="#1d4ed8", command=self._load_review).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Auto-detect from Output", width=170,
                      fg_color="#6b7280", hover_color="#4b5563",
                      command=self._auto_detect_review_paths).pack(side="left")

        # ── sort / filter / pagination bar ──
        filter_bar = ctk.CTkFrame(tab, height=36)
        filter_bar.pack(fill="x", padx=5, pady=(4, 0))
        filter_bar.pack_propagate(False)

        ctk.CTkLabel(filter_bar, text="Sort:").pack(side="left", padx=(8, 2))
        self.sort_var = ctk.StringVar(value="Filename")
        ctk.CTkOptionMenu(filter_bar, variable=self.sort_var,
                          values=["Filename", "Confidence", "Quality", "Area %"],
                          command=lambda _: self._on_filter_change(), width=110).pack(side="left", padx=2)

        ctk.CTkLabel(filter_bar, text="Filter:").pack(side="left", padx=(10, 2))
        self.filter_var = ctk.StringVar(value="All")
        ctk.CTkSegmentedButton(
            filter_bar, values=["All", "Needs Review", "Poor", "Unreviewed"],
            variable=self.filter_var, command=lambda _: self._on_filter_change()
        ).pack(side="left", padx=2)

        # Pagination controls (right side)
        self.page_label = ctk.CTkLabel(filter_bar, text="", font=("Consolas", 10))
        self.page_label.pack(side="right", padx=4)
        self.next_btn = ctk.CTkButton(filter_bar, text="Next >", width=60,
                                      command=self._next_page)
        self.next_btn.pack(side="right", padx=2)
        self.prev_btn = ctk.CTkButton(filter_bar, text="< Prev", width=60,
                                      command=self._prev_page)
        self.prev_btn.pack(side="right", padx=2)

        self.summary_label = ctk.CTkLabel(filter_bar, text="", font=("Consolas", 10))
        self.summary_label.pack(side="right", padx=8)

        # ── main: left grid + right preview ──
        content = ctk.CTkFrame(tab)
        content.pack(fill="both", expand=True, padx=5, pady=5)
        content.grid_columnconfigure(0, weight=2)
        content.grid_columnconfigure(1, weight=3)
        content.grid_rowconfigure(0, weight=1)

        # Left: thumbnail grid
        self.review_grid = ctk.CTkScrollableFrame(content, label_text="Masks")
        self.review_grid.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        for c in range(3):
            self.review_grid.columnconfigure(c, weight=1)

        # Right: preview + info + actions
        right = ctk.CTkFrame(content)
        right.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Preview: overlay + mask side by side
        preview_box = ctk.CTkFrame(right)
        preview_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        preview_box.grid_columnconfigure(0, weight=3)
        preview_box.grid_columnconfigure(1, weight=2)
        preview_box.grid_rowconfigure(0, weight=1)

        overlay_frame = ctk.CTkFrame(preview_box)
        overlay_frame.grid(row=0, column=0, sticky="nsew", padx=(2, 1), pady=2)
        ctk.CTkLabel(overlay_frame, text="Overlay (red = masked)", font=("Consolas", 10)).pack(pady=(2, 0))
        self.main_preview_label = ctk.CTkLabel(overlay_frame, text="")
        self.main_preview_label.pack(expand=True, fill="both", padx=4, pady=(0, 4))

        mask_frame = ctk.CTkFrame(preview_box)
        mask_frame.grid(row=0, column=1, sticky="nsew", padx=(1, 2), pady=2)
        ctk.CTkLabel(mask_frame, text="Mask", font=("Consolas", 10)).pack(pady=(2, 0))
        self.mask_preview_label = ctk.CTkLabel(mask_frame, text="")
        self.mask_preview_label.pack(expand=True, fill="both", padx=4, pady=(0, 4))

        # Info
        info_frame = ctk.CTkFrame(right)
        info_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self.info_text = ctk.CTkLabel(info_frame, text="Load masks to begin.",
                                      font=("Consolas", 11), anchor="w", justify="left")
        self.info_text.pack(fill="x", padx=8, pady=4)

        # Action buttons
        btn_frame = ctk.CTkFrame(right)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=(2, 4))

        ctk.CTkButton(btn_frame, text="Edit (OpenCV)", command=self._on_edit,
                      fg_color="#2563eb", width=110).pack(side="left", padx=4, pady=4)

        for text, cmd, color in [
            ("Accept", self._on_accept, "#16a34a"),
            ("Reject", self._on_reject, "#dc2626"),
            ("Skip", self._on_skip, "#6b7280"),
        ]:
            ctk.CTkButton(btn_frame, text=text, command=cmd, fg_color=color,
                          width=75).pack(side="left", padx=3, pady=4)

        ctk.CTkButton(btn_frame, text="Accept All Good", command=self._on_accept_all_good,
                      fg_color="#065f46", width=120).pack(side="right", padx=4, pady=4)

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
             self._ThumbnailWidget, _, _, self._THUMB_SIZE,
             ReviewStatusManager, _) = _import_review()

            self._review_status_mgr = ReviewStatusManager(masks_path)
            self._review_pairs = []
            self._discover_pairs(masks_path, images_path)
            self._review_loaded = True
            self._selected_stem = None
            self._page = 0

            self.log(f"Loaded {len(self._review_pairs)} mask/image pairs")

            # Compute area % in background to avoid lockup on large sets
            threading.Thread(target=self._compute_areas_bg, daemon=True).start()

            self._apply_filter_and_populate()

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

    # ── pagination + filtering ──

    def _on_filter_change(self):
        self._page = 0
        self._apply_filter_and_populate()

    def _next_page(self):
        max_page = max(0, (len(self._filtered_pairs) - 1) // ITEMS_PER_PAGE)
        if self._page < max_page:
            self._page += 1
            self._populate_review_grid()

    def _prev_page(self):
        if self._page > 0:
            self._page -= 1
            self._populate_review_grid()

    def _apply_filter_and_populate(self):
        """Apply sort/filter, then populate current page."""
        if not self._review_loaded:
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
        self._populate_review_grid()

    def _populate_review_grid(self):
        if not self._review_loaded:
            return

        for w in self.review_grid.winfo_children():
            w.destroy()

        total = len(self._filtered_pairs)
        start = self._page * ITEMS_PER_PAGE
        end = min(start + ITEMS_PER_PAGE, total)
        page_pairs = self._filtered_pairs[start:end]
        max_page = max(0, (total - 1) // ITEMS_PER_PAGE)

        for idx, pair in enumerate(page_pairs):
            ms = self._review_status_mgr.get(pair["stem"])
            thumb = self._load_overlay_thumbnail(pair["image_path"], pair["mask_path"])
            if thumb is None:
                continue
            tw = self._ThumbnailWidget(
                self.review_grid, pair["stem"], thumb, ms,
                on_click=self._on_thumb_click,
            )
            row, col = divmod(idx, 3)
            tw.grid(row=row, column=col, padx=4, pady=4, sticky="n")

        # Update page controls
        self.page_label.configure(text=f"Page {self._page + 1}/{max_page + 1} ({total} total)")
        self.prev_btn.configure(state="normal" if self._page > 0 else "disabled")
        self.next_btn.configure(state="normal" if self._page < max_page else "disabled")

        # Summary
        summary = self._review_status_mgr.get_summary()
        done = summary.get("accepted", 0) + summary.get("edited", 0)
        self.summary_label.configure(
            text=f"{done} done | {summary.get('rejected', 0)} rej | "
                 f"{summary.get('pending', 0)} pending"
        )

    # ── preview ──

    def _on_thumb_click(self, stem):
        self._selected_stem = stem
        self._update_preview()

    def _update_preview(self):
        if not self._selected_stem or not self._review_loaded:
            return
        pair = next((p for p in self._review_pairs if p["stem"] == self._selected_stem), None)
        if not pair:
            return
        ms = self._review_status_mgr.get(self._selected_stem)

        import cv2
        import numpy as np
        from PIL import Image as PILImage

        img = cv2.imread(str(pair["image_path"]))
        mask = cv2.imread(str(pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return

        h, w = img.shape[:2]

        # Overlay — fit to available width, respect aspect ratio
        target_w = 500
        scale = target_w / w
        pw, ph = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (pw, ph))
        mask_r = cv2.resize(mask, (pw, ph))

        # Red tint on MASKED pixels (where mask > 127 = areas being removed)
        overlay = img_r.copy()
        masked = mask_r > 127
        overlay[masked] = (overlay[masked].astype(np.float32) * 0.4 +
                           np.array([0, 0, 200], dtype=np.float32) * 0.6).astype(np.uint8)
        overlay_pil = PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        ctk_overlay = ctk.CTkImage(light_image=overlay_pil, size=(pw, ph))
        self.main_preview_label.configure(image=ctk_overlay, text="")
        self.main_preview_label._ctk_image = ctk_overlay

        # Mask view — same size
        mask_pil = PILImage.fromarray(cv2.cvtColor(mask_r, cv2.COLOR_GRAY2RGB))
        ctk_mask = ctk.CTkImage(light_image=mask_pil, size=(pw, ph))
        self.mask_preview_label.configure(image=ctk_mask, text="")
        self.mask_preview_label._ctk_image = ctk_mask

        self.info_text.configure(
            text=f"{self._selected_stem}  |  Quality: {ms.quality or '?'}  |  "
                 f"Conf: {ms.confidence:.0%}  |  Area: {ms.area_percent:.1f}%  |  Status: {ms.status}"
        )

    # ── actions ──

    def _on_edit(self):
        if not self._selected_stem or not self._review_loaded:
            return
        pair = next((p for p in self._review_pairs if p["stem"] == self._selected_stem), None)
        if not pair:
            return
        stem = self._selected_stem

        def _run():
            try:
                from review_masks import MaskReviewer
            except ImportError:
                from claude.review_masks import MaskReviewer
            reviewer = MaskReviewer()
            result = reviewer.review_mask(pair["image_path"], pair["mask_path"],
                                          window_name=f"Edit: {stem}")
            if result == "save":
                self._review_status_mgr.record_action(stem, "edited")
            self.after(0, self._refresh_review)

        threading.Thread(target=_run, daemon=True).start()

    def _refresh_review(self):
        self._apply_filter_and_populate()
        self._update_preview()

    def _on_accept(self):
        if self._selected_stem:
            self._review_status_mgr.record_action(self._selected_stem, "accepted")
            self._advance()

    def _on_reject(self):
        if self._selected_stem:
            self._review_status_mgr.record_action(self._selected_stem, "rejected")
            self._advance()

    def _on_skip(self):
        if self._selected_stem:
            self._review_status_mgr.record_action(self._selected_stem, "skipped")
            self._advance()

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
        self._apply_filter_and_populate()

    def _advance(self):
        """Move to next pending mask on current page."""
        idx = next((i for i, p in enumerate(self._filtered_pairs) if p["stem"] == self._selected_stem), -1)
        for i in range(idx + 1, len(self._filtered_pairs)):
            if self._review_status_mgr.get(self._filtered_pairs[i]["stem"]).status == "pending":
                self._selected_stem = self._filtered_pairs[i]["stem"]
                # Switch page if needed
                target_page = i // ITEMS_PER_PAGE
                if target_page != self._page:
                    self._page = target_page
                self._apply_filter_and_populate()
                self._update_preview()
                return
        self._apply_filter_and_populate()
        self._update_preview()


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MaskingStudio()
    app.mainloop()

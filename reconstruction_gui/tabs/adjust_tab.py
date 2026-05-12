"""
Adjust Tab — Image adjustment and colour correction.

Sits after Projects in the tab order. Accepts video files, RAW images, or
standard image directories. All adjustments applied via adjust_engine.

Usage::

    from tabs.adjust_tab import build_adjust_tab
    build_adjust_tab(app, tab_frame)
"""

import threading
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

import customtkinter as ctk
import numpy as np

from widgets import (
    Section, CollapsibleSection,
    COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
    COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
    COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
    COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
    COLOR_TEXT_MUTED,
    FONT_TEXT_SUBTITLE, FONT_TEXT_MONO_VALUE, FONT_TEXT_STATUS,
    FONT_TEXT_BTN_PRIMARY, FONT_TEXT_BTN_SECONDARY,
    LABEL_FIELD_WIDTH, BROWSE_BUTTON_WIDTH,
)

from adjust_engine import (
    AdjustmentState, RawDevelopState, AdjustDocument,
    UndoStack, apply_adjustments, load_raw, HAS_RAWPY,
    HAS_COLOUR_SCIENCE, HAS_MCC,
    detect_chart_mcc, get_reference_patches,
    fit_chart_correction, apply_chart_correction, match_histograms,
    DENOISE_METHODS,
    is_video_file, probe_video, extract_single_frame,
    normalize_exposures,
)

from prep360.core.adjustment_recipe import AdjustmentRecipe
from prep360.core.color_pipeline import apply_adjustment_recipe, load_image_float, write_image_float
from prep360.core.lut import LUTProcessor

try:
    from reconstruction_gui.adjust_workflow import (
        detect_adjust_input,
        export_adjusted_dataset,
        validate_adjust_export,
    )
except ImportError:
    from adjust_workflow import (
        detect_adjust_input,
        export_adjusted_dataset,
        validate_adjust_export,
    )

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".webm"}


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_PREVIEW_MAX_PIXELS = 2_000_000  # ~2MP preview buffer
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
_RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".dng", ".arw", ".raf", ".raw"}
_ALL_IMAGE_EXTENSIONS = _IMAGE_EXTENSIONS | _RAW_EXTENSIONS


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_adjust_tab(app, parent):
    """Populate the Adjust tab frame. All state stored on *app*."""

    # --- Adjust-specific state ---
    app._adjust_doc = AdjustDocument()
    app._adjust_undo = UndoStack()
    app._adjust_image_list: List[Path] = []
    app._adjust_nav_idx = 0
    app._adjust_source_img: Optional[np.ndarray] = None
    app._adjust_preview_buf: Optional[np.ndarray] = None
    app._adjust_debounce_id = None
    app._adjust_slider_vars = {}  # name → DoubleVar/StringVar
    app._adjust_ccm = None         # fitted colour correction matrix
    app._adjust_ccm_method = None  # method string used to fit
    app._adjust_ccm_degree = None  # degree used to fit
    app._adjust_ref_img = None     # reference image for histogram matching (float32 BGR)
    app._adjust_video_path = None  # Path if input is a video file
    app._adjust_video_info = None  # dict from probe_video
    app._adjust_dataset = None
    app._adjust_recipe = AdjustmentRecipe()
    app._adjust_lut_processor = LUTProcessor()

    # --- Scrollable left panel ---
    scroll = ctk.CTkScrollableFrame(parent)
    scroll.pack(fill="both", expand=True)

    _build_input_section(app, scroll)
    _build_raw_section(app, scroll)
    _build_lut_recipe_section(app, scroll)
    _build_tone_section(app, scroll)
    _build_colour_section(app, scroll)
    _build_detail_section(app, scroll)
    _build_corrections_section(app, scroll)
    _build_colour_correction_section(app, scroll)
    _build_export_section(app, scroll)


# ══════════════════════════════════════════════════════════════════════════════
# Slider helper
# ══════════════════════════════════════════════════════════════════════════════

def _add_slider(app, parent, label: str, var_name: str, from_: float, to: float,
                default: float, steps: int, fmt: str = ".1f", unit: str = ""):
    """Add a labelled slider row. Returns the DoubleVar."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=2, padx=6)

    ctk.CTkLabel(row, text=f"{label}:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")

    var = ctk.DoubleVar(value=default)
    app._adjust_slider_vars[var_name] = var

    slider = ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps,
                           variable=var,
                           command=lambda _val: _on_slider_change(app))
    slider.pack(side="left", fill="x", expand=True, padx=(6, 4))

    value_text = f"{default:{fmt}}{unit}"
    lbl = ctk.CTkLabel(row, text=value_text, width=60, font=FONT_TEXT_MONO_VALUE)
    lbl.pack(side="left")

    # Store label ref for updates
    var._adjust_label = lbl
    var._adjust_fmt = fmt
    var._adjust_unit = unit

    return var


# ══════════════════════════════════════════════════════════════════════════════
# Section builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_input_section(app, parent):
    sec = Section(parent, "Input")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Input row
    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Input:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.adjust_input_entry = ctk.CTkEntry(row, placeholder_text="Folder or image file...")
    app.adjust_input_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(row, text="Folder", width=55,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  font=ctk.CTkFont(size=12),
                  command=lambda: _browse_adjust_folder(app)
                  ).pack(side="left", padx=(0, 2))
    ctk.CTkButton(row, text="File", width=45,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  font=ctk.CTkFont(size=12),
                  command=lambda: _browse_adjust_file(app)
                  ).pack(side="left")

    # Output row
    row2 = ctk.CTkFrame(c, fg_color="transparent")
    row2.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row2, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.adjust_output_entry = ctk.CTkEntry(row2, placeholder_text="Output folder...")
    app.adjust_output_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(row2, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: app._browse_dir_into(app.adjust_output_entry)
                  ).pack(side="left")

    # In-place toggle
    inplace_row = ctk.CTkFrame(c, fg_color="transparent")
    inplace_row.pack(fill="x", pady=3, padx=6)
    app._adjust_inplace_var = ctk.BooleanVar(value=False)

    def _on_inplace_toggle():
        is_inplace = app._adjust_inplace_var.get()
        if is_inplace:
            app._adjust_output_saved = app.adjust_output_entry.get()
            app.adjust_output_entry.delete(0, "end")
            app.adjust_output_entry.configure(
                state="disabled",
                placeholder_text="Overwriting source frames",
            )
            app._adjust_refuse_non_empty_var.set(False)
        else:
            app.adjust_output_entry.configure(
                state="normal",
                placeholder_text="Output folder...",
            )
            saved = getattr(app, "_adjust_output_saved", "")
            if saved:
                app.adjust_output_entry.delete(0, "end")
                app.adjust_output_entry.insert(0, saved)
        if hasattr(app, "_adjust_run_btn"):
            app._adjust_run_btn.configure(
                text="Apply" if is_inplace else "Export Adjusted Dataset",
                fg_color=COLOR_ACTION_DANGER if is_inplace else COLOR_ACTION_PRIMARY,
                hover_color=COLOR_ACTION_DANGER_H if is_inplace else COLOR_ACTION_PRIMARY_H,
            )

    ctk.CTkCheckBox(inplace_row, text="Overwrite source frames",
                    variable=app._adjust_inplace_var,
                    command=_on_inplace_toggle,
                    font=ctk.CTkFont(size=12)).pack(
        side="left", padx=(LABEL_FIELD_WIDTH + 6, 0))

    # Info + navigation row
    nav_row = ctk.CTkFrame(c, fg_color="transparent")
    nav_row.pack(fill="x", pady=(4, 2), padx=6)

    ctk.CTkButton(nav_row, text="Load", width=60,
                  fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _load_adjust_images(app)
                  ).pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 6))

    ctk.CTkButton(nav_row, text="<", width=28, height=28,
                  fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                  command=lambda: _navigate(app, -1)
                  ).pack(side="left", padx=(0, 2))
    ctk.CTkButton(nav_row, text=">", width=28, height=28,
                  fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                  command=lambda: _navigate(app, +1)
                  ).pack(side="left", padx=(0, 6))

    # Paired lens selector — always exists, visibility toggled via grid
    app._adjust_paired_lens_var = ctk.StringVar(value="Front")
    app._adjust_paired_lens_menu = ctk.CTkOptionMenu(
        nav_row, variable=app._adjust_paired_lens_var,
        values=["Front", "Back"],
        width=80,
        command=lambda _v: _load_current_image(app))
    # Hidden by default — _show_paired_lens / _hide_paired_lens control visibility
    app._adjust_paired_lens_visible = False

    app._adjust_nav_label = ctk.CTkLabel(nav_row, text="0 / 0",
                                         font=FONT_TEXT_MONO_VALUE, width=70)
    app._adjust_nav_label.pack(side="left")

    app._adjust_info_label = ctk.CTkLabel(nav_row, text="",
                                          text_color=COLOR_TEXT_MUTED,
                                          font=FONT_TEXT_STATUS)
    app._adjust_info_label.pack(side="left", padx=(8, 0))

    # Reset button
    ctk.CTkButton(nav_row, text="Reset All", width=70,
                  fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
                  font=ctk.CTkFont(size=11),
                  command=lambda: _reset_all(app)
                  ).pack(side="right")


def _build_raw_section(app, parent):
    """RAW Development — shown only when RAW files are loaded."""
    sec = CollapsibleSection(parent, "RAW Development",
                             subtitle="rawpy" if HAS_RAWPY else "rawpy not installed")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content
    app._adjust_raw_section = sec

    if not HAS_RAWPY:
        ctk.CTkLabel(c, text="Install rawpy for RAW support: pip install rawpy",
                     text_color=COLOR_TEXT_MUTED, font=FONT_TEXT_STATUS
                     ).pack(fill="x", padx=12, pady=4)
        return

    # Demosaic algorithm
    row1 = ctk.CTkFrame(c, fg_color="transparent")
    row1.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row1, text="Demosaic:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_raw_demosaic_var = ctk.StringVar(value="AHD")
    ctk.CTkOptionMenu(row1, variable=app._adjust_raw_demosaic_var,
                      values=["AHD", "VNG", "PPG", "DCB", "LMMSE", "AMAZE", "DHT", "AAHD"],
                      width=100).pack(side="left", padx=(6, 12))

    ctk.CTkLabel(row1, text="Colour:", width=50, anchor="e").pack(side="left")
    app._adjust_raw_colorspace_var = ctk.StringVar(value="sRGB")
    ctk.CTkOptionMenu(row1, variable=app._adjust_raw_colorspace_var,
                      values=["sRGB", "Adobe", "ProPhoto", "ACES", "Rec2020"],
                      width=90).pack(side="left", padx=(6, 0))

    # White balance mode
    row2 = ctk.CTkFrame(c, fg_color="transparent")
    row2.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row2, text="WB:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_raw_wb_var = ctk.StringVar(value="Camera")
    ctk.CTkSegmentedButton(row2, values=["Camera", "Auto", "Daylight"],
                           variable=app._adjust_raw_wb_var,
                           ).pack(side="left", padx=(6, 0))

    # Exposure shift
    _add_slider(app, c, "Exp shift", "raw_exp_shift", 0.25, 4.0, 1.0, 75, ".2f", "x")

    # Highlight mode
    row3 = ctk.CTkFrame(c, fg_color="transparent")
    row3.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row3, text="Highlights:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_raw_highlight_var = ctk.StringVar(value="Clip")
    ctk.CTkOptionMenu(row3, variable=app._adjust_raw_highlight_var,
                      values=["Clip", "Unclip", "Blend", "Rebuild"],
                      width=100).pack(side="left", padx=(6, 0))

    # Re-develop button
    ctk.CTkButton(c, text="Re-develop RAW", width=120,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _redevelop_raw(app)
                  ).pack(padx=(LABEL_FIELD_WIDTH + 12, 0), pady=(4, 4), anchor="w")


def _build_lut_recipe_section(app, parent):
    sec = Section(parent, "LUT & Recipe")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Top row: Apply LUT checkbox + View original checkbox
    top_row = ctk.CTkFrame(c, fg_color="transparent")
    top_row.pack(fill="x", pady=3, padx=6)
    app._adjust_lut_enabled_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(
        top_row,
        text="Apply LUT",
        variable=app._adjust_lut_enabled_var,
        command=lambda: _on_recipe_change(app),
    ).pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 12))
    app._adjust_before_after_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(top_row, text="View original",
                    variable=app._adjust_before_after_var,
                    command=lambda: _refresh_preview(app),
                    font=ctk.CTkFont(size=12)).pack(side="left")

    # LUT dropdown (recent LUTs) + browse + reload
    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="LUT:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")

    # Load LUT history from prefs
    lut_history = list(app._prefs.get("lut_history", []))
    display_values = [Path(p).name for p in lut_history] if lut_history else ["(none)"]

    app._adjust_lut_history = list(lut_history)
    app._adjust_lut_dropdown = ctk.CTkOptionMenu(
        row,
        values=display_values,
        width=200,
        command=lambda name: _on_lut_dropdown_select(app, name),
    )
    app._adjust_lut_dropdown.pack(side="left", padx=(6, 4))
    if not lut_history:
        app._adjust_lut_dropdown.set("(none)")

    # Hidden entry for the actual path (used by recipe load/save and internal logic)
    app._adjust_lut_entry = ctk.CTkEntry(row)
    # Don't pack — this is a data holder, not visible
    # Pre-populate with the most recent LUT path so "Apply LUT" works immediately
    if lut_history:
        app._adjust_lut_entry.insert(0, lut_history[0])

    ctk.CTkButton(row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _browse_lut(app)).pack(side="left", padx=(0, 2))
    ctk.CTkButton(row, text="\u21bb", width=28,
                  fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                  font=ctk.CTkFont(size=14),
                  command=lambda: _reload_lut(app)).pack(side="left")

    str_row = ctk.CTkFrame(c, fg_color="transparent")
    str_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(str_row, text="Strength:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_lut_strength_var = ctk.DoubleVar(value=1.0)
    app._adjust_lut_strength_label = ctk.CTkLabel(str_row, text="100%", width=55,
                                                  font=FONT_TEXT_MONO_VALUE)
    app._adjust_lut_strength_label.pack(side="right")
    ctk.CTkSlider(
        str_row,
        from_=0,
        to=1,
        variable=app._adjust_lut_strength_var,
        command=lambda v: _on_lut_strength_change(app, v),
    ).pack(side="left", fill="x", expand=True, padx=(6, 4))

    status_row = ctk.CTkFrame(c, fg_color="transparent")
    status_row.pack(fill="x", pady=3, padx=6)
    app._adjust_lut_status = ctk.CTkLabel(status_row, text="",
                                          text_color=COLOR_TEXT_MUTED,
                                          font=FONT_TEXT_STATUS)
    app._adjust_lut_status.pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 0))

    recipe_row = ctk.CTkFrame(c, fg_color="transparent")
    recipe_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(recipe_row, text="Recipe:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    ctk.CTkButton(recipe_row, text="Load Recipe", width=95,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _load_recipe_dialog(app)).pack(side="left", padx=(6, 4))
    ctk.CTkButton(recipe_row, text="Save Recipe", width=95,
                  fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _save_recipe_dialog(app)).pack(side="left", padx=(0, 8))
    app._adjust_recipe_status = ctk.CTkLabel(recipe_row, text=app._adjust_recipe.name,
                                             text_color=COLOR_TEXT_MUTED,
                                             font=FONT_TEXT_STATUS)
    app._adjust_recipe_status.pack(side="left", fill="x", expand=True)


def _build_tone_section(app, parent):
    sec = Section(parent, "Tone")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    _add_slider(app, c, "Exposure", "exposure", -5.0, 5.0, 0.0, 400, "+.2f", " EV")
    _add_slider(app, c, "Contrast", "contrast", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Highlights", "highlights", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Shadows", "shadows", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Whites", "whites", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Blacks", "blacks", -100, 100, 0, 400, ".1f")


def _build_colour_section(app, parent):
    sec = Section(parent, "Colour")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    _add_slider(app, c, "Temperature", "temperature", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Tint", "tint", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Saturation", "saturation", -100, 100, 0, 400, ".1f")
    _add_slider(app, c, "Vibrance", "vibrance", -100, 100, 0, 400, ".1f")


def _build_detail_section(app, parent):
    sec = CollapsibleSection(parent, "Detail")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    _add_slider(app, c, "Sharpen", "sharpen_amount", 0, 100, 0, 200, ".1f")
    _add_slider(app, c, "Radius", "sharpen_radius", 0.5, 5.0, 1.0, 90, ".2f")
    _add_slider(app, c, "Threshold", "sharpen_threshold", 0, 100, 0, 200, ".1f")

    _add_slider(app, c, "Denoise", "denoise_strength", 0, 100, 0, 200, ".1f")

    # Denoise method dropdown
    method_row = ctk.CTkFrame(c, fg_color="transparent")
    method_row.pack(fill="x", pady=2, padx=6)
    ctk.CTkLabel(method_row, text="Method:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_denoise_method_var = ctk.StringVar(value="bilateral")
    ctk.CTkOptionMenu(method_row, variable=app._adjust_denoise_method_var,
                      values=DENOISE_METHODS,
                      width=110).pack(side="left", padx=(6, 0))


def _build_corrections_section(app, parent):
    sec = CollapsibleSection(parent, "Corrections")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    _add_slider(app, c, "CLAHE", "clahe_clip", 0, 4.0, 0, 40, ".1f")
    _add_slider(app, c, "Vignette fix", "vignette_strength", 0, 2.0, 0, 20, ".1f")


def _build_colour_correction_section(app, parent):
    sec = CollapsibleSection(parent, "Colour Correction",
                             subtitle="chart calibration & histogram matching")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Mode selector
    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Mode:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_cc_mode_var = ctk.StringVar(value="None")
    ctk.CTkSegmentedButton(
        row, values=["None", "Chart", "Histogram"],
        variable=app._adjust_cc_mode_var,
        command=lambda _: _on_cc_mode_change(app),
    ).pack(side="left", padx=(6, 0))

    # ── Chart sub-panel ──
    app._adjust_chart_frame = ctk.CTkFrame(c, fg_color="transparent")

    chart_row1 = ctk.CTkFrame(app._adjust_chart_frame, fg_color="transparent")
    chart_row1.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(chart_row1, text="Chart image:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_chart_entry = ctk.CTkEntry(chart_row1, placeholder_text="Image with colour chart...")
    app._adjust_chart_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(chart_row1, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _browse_chart_image(app)
                  ).pack(side="left")

    chart_row2 = ctk.CTkFrame(app._adjust_chart_frame, fg_color="transparent")
    chart_row2.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(chart_row2, text="Method:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_cc_method_var = ctk.StringVar(value="Finlayson 2015")
    ctk.CTkOptionMenu(chart_row2, variable=app._adjust_cc_method_var,
                      values=["Finlayson 2015", "Cheung 2004", "Vandermonde"],
                      width=120).pack(side="left", padx=(6, 12))

    ctk.CTkLabel(chart_row2, text="Degree:").pack(side="left")
    app._adjust_cc_degree_var = ctk.StringVar(value="2")
    ctk.CTkOptionMenu(chart_row2, variable=app._adjust_cc_degree_var,
                      values=["1", "2", "3"],
                      width=50).pack(side="left", padx=(4, 0))

    chart_row3 = ctk.CTkFrame(app._adjust_chart_frame, fg_color="transparent")
    chart_row3.pack(fill="x", pady=3, padx=6)

    detect_label = "Detect Chart"
    if HAS_MCC:
        detect_label += " (cv2.mcc)"
    ctk.CTkButton(chart_row3, text=detect_label, width=140,
                  fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _detect_and_fit_chart(app)
                  ).pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 6))

    app._adjust_cc_status = ctk.CTkLabel(chart_row3, text="No correction fitted",
                                         text_color=COLOR_TEXT_MUTED, font=FONT_TEXT_STATUS)
    app._adjust_cc_status.pack(side="left")

    if not HAS_COLOUR_SCIENCE:
        ctk.CTkLabel(app._adjust_chart_frame,
                     text="Install colour-science for chart correction: pip install colour-science",
                     text_color=COLOR_TEXT_MUTED, font=FONT_TEXT_STATUS
                     ).pack(fill="x", padx=12, pady=2)

    # ── Histogram match sub-panel ──
    app._adjust_hist_frame = ctk.CTkFrame(c, fg_color="transparent")

    hist_row = ctk.CTkFrame(app._adjust_hist_frame, fg_color="transparent")
    hist_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(hist_row, text="Reference:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_hist_ref_entry = ctk.CTkEntry(hist_row, placeholder_text="Reference image for matching...")
    app._adjust_hist_ref_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(hist_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=lambda: _browse_reference_image(app)
                  ).pack(side="left")

    hist_row2 = ctk.CTkFrame(app._adjust_hist_frame, fg_color="transparent")
    hist_row2.pack(fill="x", pady=3, padx=6)
    ctk.CTkButton(hist_row2, text="Load Reference", width=110,
                  fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                  font=FONT_TEXT_BTN_SECONDARY,
                  command=lambda: _load_reference_image(app)
                  ).pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 6))

    app._adjust_hist_status = ctk.CTkLabel(hist_row2, text="No reference loaded",
                                           text_color=COLOR_TEXT_MUTED, font=FONT_TEXT_STATUS)
    app._adjust_hist_status.pack(side="left")

    # Start with sub-panels hidden
    _on_cc_mode_change(app)


def _build_export_section(app, parent):
    sec = Section(parent, "Export")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Format + quality + bit depth row
    fmt_row = ctk.CTkFrame(c, fg_color="transparent")
    fmt_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(fmt_row, text="Format:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app._adjust_export_fmt_var = ctk.StringVar(value="jpg")
    ctk.CTkOptionMenu(fmt_row, variable=app._adjust_export_fmt_var,
                      values=["jpg", "png", "tiff"],
                      width=80).pack(side="left", padx=(6, 8))
    ctk.CTkLabel(fmt_row, text="Quality:").pack(side="left")
    app._adjust_export_quality_var = ctk.StringVar(value="99")
    ctk.CTkEntry(fmt_row, textvariable=app._adjust_export_quality_var,
                 width=40).pack(side="left", padx=(4, 8))
    ctk.CTkLabel(fmt_row, text="Depth:").pack(side="left")
    app._adjust_export_depth_var = ctk.StringVar(value="8-bit")
    ctk.CTkOptionMenu(fmt_row, variable=app._adjust_export_depth_var,
                      values=["8-bit", "16-bit"],
                      width=70).pack(side="left", padx=(4, 0))

    # Metadata options
    meta_row = ctk.CTkFrame(c, fg_color="transparent")
    meta_row.pack(fill="x", pady=3, padx=6)
    app._adjust_copy_exif_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(meta_row, text="Copy EXIF metadata",
                    variable=app._adjust_copy_exif_var,
                    font=ctk.CTkFont(size=12)).pack(
        side="left", padx=(LABEL_FIELD_WIDTH + 6, 12))
    app._adjust_embed_icc_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(meta_row, text="Embed sRGB ICC profile",
                    variable=app._adjust_embed_icc_var,
                    font=ctk.CTkFont(size=12)).pack(side="left")

    # Batch options
    batch_row = ctk.CTkFrame(c, fg_color="transparent")
    batch_row.pack(fill="x", pady=3, padx=6)
    app._adjust_normalize_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(batch_row, text="Normalize exposures across batch",
                    variable=app._adjust_normalize_var,
                    font=ctk.CTkFont(size=12)).pack(
        side="left", padx=(LABEL_FIELD_WIDTH + 6, 12))
    app._adjust_refuse_non_empty_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(batch_row, text="Refuse non-empty output",
                    variable=app._adjust_refuse_non_empty_var,
                    font=ctk.CTkFont(size=12)).pack(side="left")

    # Buttons row
    btn_row = ctk.CTkFrame(c, fg_color="transparent")
    btn_row.pack(fill="x", pady=4, padx=6)

    app._adjust_run_btn = ctk.CTkButton(
        btn_row, text="Export Adjusted Dataset", width=160,
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=FONT_TEXT_BTN_PRIMARY,
        command=lambda: _start_export(app),
    )
    app._adjust_run_btn.pack(side="left", padx=(LABEL_FIELD_WIDTH + 6, 6))

    app._adjust_stop_btn = ctk.CTkButton(
        btn_row, text="Stop", width=60,
        fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
        font=FONT_TEXT_BTN_SECONDARY,
        command=lambda: app.stop_operation(),
    )
    # Hidden initially — shown by _start_operation pattern

    app._adjust_progress = ctk.CTkProgressBar(c, height=8)
    app._adjust_progress.pack(fill="x", padx=10, pady=(0, 4))
    app._adjust_progress.set(0)

    app._adjust_export_status = ctk.CTkLabel(
        c, text="", text_color=COLOR_TEXT_MUTED, font=FONT_TEXT_STATUS)
    app._adjust_export_status.pack(fill="x", padx=12, pady=(0, 4))


# ══════════════════════════════════════════════════════════════════════════════
# LUT / recipe helpers
# ══════════════════════════════════════════════════════════════════════════════

def _browse_lut(app):
    path = filedialog.askopenfilename(
        title="Select .cube LUT",
        filetypes=[("CUBE LUT", "*.cube"), ("All files", "*.*")],
    )
    if path:
        _set_lut_path(app, path)
        app._adjust_lut_enabled_var.set(True)
        _reload_lut(app)


def _on_lut_dropdown_select(app, display_name):
    """Handle selection from the LUT history dropdown."""
    if display_name == "(none)":
        return
    history = getattr(app, "_adjust_lut_history", [])
    for full_path in history:
        if Path(full_path).name == display_name:
            _set_lut_path(app, full_path)
            app._adjust_lut_enabled_var.set(True)
            _reload_lut(app)
            return


def _set_lut_path(app, path: str):
    """Set the LUT path in the hidden entry and add to persistent history."""
    app._adjust_lut_entry.delete(0, "end")
    app._adjust_lut_entry.insert(0, path)
    _add_lut_to_history(app, path)


def _add_lut_to_history(app, path: str):
    """Add a LUT path to the persistent dropdown history."""
    resolved = str(Path(path).resolve())
    history = getattr(app, "_adjust_lut_history", [])
    # Remove if already present (will re-add at front)
    history = [p for p in history if str(Path(p).resolve()) != resolved]
    history.insert(0, str(path))
    # Keep last 10
    history = history[:10]
    app._adjust_lut_history = history
    app._prefs["lut_history"] = history
    app._save_prefs()
    # Update dropdown values
    display_values = [Path(p).name for p in history]
    app._adjust_lut_dropdown.configure(values=display_values)
    app._adjust_lut_dropdown.set(Path(path).name)


def _reload_lut(app):
    lut_path = app._adjust_lut_entry.get().strip()
    if not lut_path:
        app._adjust_lut_status.configure(text="No LUT selected")
        return
    try:
        _, info = app._adjust_lut_processor.load_cube(lut_path)
        app._adjust_lut_status.configure(text=f"{Path(lut_path).name}, {info.size}x{info.size}x{info.size}")
        app.log(f"[Adjust] LUT loaded: {info.name} ({info.size}^3)")
    except Exception as e:
        app._adjust_lut_status.configure(text=f"Invalid LUT: {e}")
        app.log(f"[Adjust] Invalid LUT: {e}")
    _on_recipe_change(app)


def _on_lut_strength_change(app, value):
    app._adjust_lut_strength_label.configure(text=f"{int(float(value) * 100)}%")
    _on_recipe_change(app)


def _on_recipe_change(app):
    app._adjust_recipe = _recipe_from_ui(app)
    _refresh_preview(app)


def _load_recipe_dialog(app):
    path = filedialog.askopenfilename(
        title="Load Adjust Recipe",
        filetypes=[("JSON recipe", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return
    try:
        recipe = AdjustmentRecipe.load(path)
        errors = recipe.validate()
        if errors:
            app.log(f"[Adjust] Recipe invalid: {'; '.join(errors)}")
            return
        _apply_recipe_to_ui(app, recipe)
        app._adjust_recipe_status.configure(text=f"Loaded: {Path(path).name}")
        app.log(f"[Adjust] Recipe loaded: {path}")
    except Exception as e:
        app.log(f"[Adjust] Failed to load recipe: {e}")


def _save_recipe_dialog(app):
    recipe = _recipe_from_ui(app)
    path = filedialog.asksaveasfilename(
        title="Save Adjust Recipe",
        defaultextension=".json",
        filetypes=[("JSON recipe", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return
    try:
        recipe.save(path)
        app._adjust_recipe_status.configure(text=f"Saved: {Path(path).name}")
        app.log(f"[Adjust] Recipe saved: {path}")
    except Exception as e:
        app.log(f"[Adjust] Failed to save recipe: {e}")


def _recipe_from_ui(app) -> AdjustmentRecipe:
    recipe = getattr(app, "_adjust_recipe", AdjustmentRecipe())
    recipe.input_lut.enabled = app._adjust_lut_enabled_var.get() if hasattr(app, "_adjust_lut_enabled_var") else False
    recipe.input_lut.path = app._adjust_lut_entry.get().strip() if hasattr(app, "_adjust_lut_entry") else ""
    recipe.input_lut.strength = app._adjust_lut_strength_var.get() if hasattr(app, "_adjust_lut_strength_var") else 1.0

    if hasattr(app, "_adjust_slider_vars") and app._adjust_slider_vars:
        v = app._adjust_slider_vars
        recipe.tone.exposure = v["exposure"].get()
        recipe.tone.contrast = v["contrast"].get()
        recipe.tone.highlights = v["highlights"].get()
        recipe.tone.shadows = v["shadows"].get()
        recipe.tone.whites = v["whites"].get()
        recipe.tone.blacks = v["blacks"].get()
        recipe.white_balance.temperature = v["temperature"].get()
        recipe.white_balance.tint = v["tint"].get()
        recipe.color.saturation = v["saturation"].get()
        recipe.color.vibrance = v["vibrance"].get()
        recipe.detail.sharpen_amount = v["sharpen_amount"].get()
        recipe.detail.sharpen_radius = v["sharpen_radius"].get()
        recipe.detail.sharpen_threshold = v["sharpen_threshold"].get()
        recipe.detail.denoise_strength = v["denoise_strength"].get()
        recipe.corrections.clahe_clip = v["clahe_clip"].get()
        recipe.corrections.vignette_strength = v["vignette_strength"].get()
    if hasattr(app, "_adjust_denoise_method_var"):
        recipe.detail.denoise_method = app._adjust_denoise_method_var.get()
    if hasattr(app, "_adjust_export_fmt_var"):
        recipe.output.format = app._adjust_export_fmt_var.get()
        try:
            recipe.output.quality = int(app._adjust_export_quality_var.get())
        except ValueError:
            recipe.output.quality = 99
        recipe.output.bit_depth = app._adjust_export_depth_var.get()
        recipe.output.copy_exif = app._adjust_copy_exif_var.get()
        recipe.output.embed_icc = app._adjust_embed_icc_var.get()
    return recipe


def _apply_recipe_to_ui(app, recipe: AdjustmentRecipe):
    app._adjust_recipe = recipe
    app._adjust_lut_enabled_var.set(recipe.input_lut.enabled)
    app._adjust_lut_entry.delete(0, "end")
    app._adjust_lut_entry.insert(0, recipe.input_lut.path)
    app._adjust_lut_strength_var.set(recipe.input_lut.strength)
    app._adjust_lut_strength_label.configure(text=f"{int(recipe.input_lut.strength * 100)}%")

    mapping = {
        "exposure": recipe.tone.exposure,
        "contrast": recipe.tone.contrast,
        "highlights": recipe.tone.highlights,
        "shadows": recipe.tone.shadows,
        "whites": recipe.tone.whites,
        "blacks": recipe.tone.blacks,
        "temperature": recipe.white_balance.temperature,
        "tint": recipe.white_balance.tint,
        "saturation": recipe.color.saturation,
        "vibrance": recipe.color.vibrance,
        "sharpen_amount": recipe.detail.sharpen_amount,
        "sharpen_radius": recipe.detail.sharpen_radius,
        "sharpen_threshold": recipe.detail.sharpen_threshold,
        "denoise_strength": recipe.detail.denoise_strength,
        "clahe_clip": recipe.corrections.clahe_clip,
        "vignette_strength": recipe.corrections.vignette_strength,
    }
    for name, value in mapping.items():
        if name in app._adjust_slider_vars:
            app._adjust_slider_vars[name].set(value)
    if hasattr(app, "_adjust_denoise_method_var"):
        app._adjust_denoise_method_var.set(recipe.detail.denoise_method)
    if hasattr(app, "_adjust_export_fmt_var"):
        app._adjust_export_fmt_var.set(recipe.output.format)
        app._adjust_export_quality_var.set(str(recipe.output.quality))
        app._adjust_export_depth_var.set(recipe.output.bit_depth)
        app._adjust_copy_exif_var.set(recipe.output.copy_exif)
        app._adjust_embed_icc_var.set(recipe.output.embed_icc)
    _reload_lut(app) if recipe.input_lut.enabled and recipe.input_lut.path else _refresh_preview(app)


# ══════════════════════════════════════════════════════════════════════════════
# Browse helpers
# ══════════════════════════════════════════════════════════════════════════════

def _browse_adjust_folder(app):
    folder = filedialog.askdirectory(
        title="Select image folder",
        initialdir=app._entry_initialdir(app.adjust_input_entry),
    )
    if folder:
        app.adjust_input_entry.delete(0, "end")
        app.adjust_input_entry.insert(0, folder)


def _browse_adjust_file(app):
    exts = " ".join(f"*{e}" for e in sorted(_ALL_IMAGE_EXTENSIONS))
    path = filedialog.askopenfilename(
        title="Select image file",
        initialdir=app._entry_initialdir(app.adjust_input_entry),
        filetypes=[("Images", exts), ("All files", "*.*")],
    )
    if path:
        app.adjust_input_entry.delete(0, "end")
        app.adjust_input_entry.insert(0, path)


# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════

def _show_paired_lens(app):
    if not getattr(app, "_adjust_paired_lens_visible", False):
        app._adjust_paired_lens_menu.pack(side="left", padx=(0, 6),
                                          before=app._adjust_nav_label)
        app._adjust_paired_lens_visible = True


def _hide_paired_lens(app):
    if getattr(app, "_adjust_paired_lens_visible", False):
        app._adjust_paired_lens_menu.pack_forget()
        app._adjust_paired_lens_var.set("Front")
        app._adjust_paired_lens_visible = False


def _navigate(app, delta: int):
    """Move to next/previous image, or step through video frames."""
    # Video mode — step by 1 second
    if app._adjust_video_path and app._adjust_video_info:
        dur = app._adjust_video_info["duration"]
        fps = app._adjust_video_info["fps"]
        step = 1.0  # 1 second per step
        current_time = app._adjust_nav_idx * step
        new_time = current_time + delta * step
        new_time = max(0, min(new_time, dur))
        app._adjust_nav_idx = max(0, int(new_time / step))
        app._adjust_nav_label.configure(text=f"{new_time:.1f}s / {dur:.1f}s")

        frame = extract_single_frame(str(app._adjust_video_path), new_time)
        if frame is not None:
            import cv2
            app._adjust_source_img = frame
            h, w = frame.shape[:2]
            total = h * w
            if total > _PREVIEW_MAX_PIXELS:
                scale = np.sqrt(_PREVIEW_MAX_PIXELS / total)
                app._adjust_preview_buf = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA)
            else:
                app._adjust_preview_buf = frame.copy()
            _refresh_preview(app)
        return

    # Image mode
    if not app._adjust_image_list:
        return
    new_idx = app._adjust_nav_idx + delta
    if new_idx < 0 or new_idx >= len(app._adjust_image_list):
        return
    app._adjust_nav_idx = new_idx
    _update_nav_label(app)
    # Keep shared navigator in sync
    app._nav_idx.set(new_idx)
    total = len(app._adjust_image_list)
    app._nav_counter.configure(text=f"{new_idx + 1} / {total}")
    _load_current_image(app)


def _update_nav_label(app):
    total = len(app._adjust_image_list)
    if total == 0 and not app._adjust_video_path:
        app._adjust_nav_label.configure(text="0 / 0")
    elif app._adjust_video_path:
        pass  # video nav label handled in _navigate
    elif getattr(app, "_adjust_dataset", None) and app._adjust_dataset.kind == "paired_extraction":
        app._adjust_nav_label.configure(text=f"Pair {app._adjust_nav_idx + 1} / {total}")
    else:
        app._adjust_nav_label.configure(text=f"{app._adjust_nav_idx + 1} / {total}")


def _sync_shared_navigator(app):
    """Sync the shared preview navigator slider to the Adjust image list."""
    n = len(app._adjust_image_list)
    if n == 0:
        app._nav_slider.configure(to=1, number_of_steps=1)
        app._nav_counter.configure(text="0 / 0")
        return
    app._nav_slider.configure(to=n - 1, number_of_steps=max(n - 1, 1))
    app._nav_idx.set(app._adjust_nav_idx)
    app._nav_counter.configure(text=f"{app._adjust_nav_idx + 1} / {n}")


# ══════════════════════════════════════════════════════════════════════════════
# Image loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_adjust_images(app):
    """Scan input path and populate image list. Also handles paired roots and videos."""
    input_path = app.adjust_input_entry.get().strip()
    if not input_path:
        app.log("[Adjust] No input path specified")
        return

    p = Path(input_path)

    try:
        dataset = detect_adjust_input(p)
    except Exception as e:
        app.log(f"[Adjust] Path not found or not recognized: {e}")
        return

    app._adjust_video_path = None
    app._adjust_video_info = None
    app._adjust_dataset = dataset

    if dataset.kind == "video_sample":
        _load_video(app, dataset.video_path)
        return

    if dataset.kind == "paired_extraction":
        app._adjust_image_list = list(dataset.front_images)
        count = min(len(dataset.front_images), len(dataset.back_images))
        mismatch = "" if len(dataset.front_images) == len(dataset.back_images) else " (count mismatch)"
        app._adjust_info_label.configure(
            text=f"Paired: {len(dataset.front_images)}F / {len(dataset.back_images)}B{mismatch}"
        )
        app.log(
            f"[Adjust] Detected paired extraction: front={len(dataset.front_images)}, "
            f"back={len(dataset.back_images)}, root={dataset.root}"
        )
        if dataset.source_manifest:
            app.log(f"[Adjust] Manifest: {dataset.source_manifest.name}")
        _show_paired_lens(app)
    else:
        app._adjust_image_list = list(dataset.images)
        count = len(dataset.images)
        app._adjust_info_label.configure(text=f"{count} images loaded ({dataset.kind})")
        app.log(f"[Adjust] Loaded {count} images from {input_path} ({dataset.kind})")
        _hide_paired_lens(app)

    app._adjust_nav_idx = 0

    if hasattr(app, "adjust_output_entry") and not app._adjust_inplace_var.get():
        if not app.adjust_output_entry.get().strip():
            default_out = dataset.root.with_name(f"{dataset.root.name}_adjusted")
            app.adjust_output_entry.delete(0, "end")
            app.adjust_output_entry.insert(0, str(default_out))

    _update_nav_label(app)
    _sync_shared_navigator(app)

    if app._adjust_image_list:
        _load_current_image(app)


def _load_video(app, video_path: Path):
    """Load a video file for frame-by-frame adjustment."""
    info = probe_video(str(video_path))
    if info is None:
        app.log(f"[Adjust] Failed to probe video (ffprobe not found or failed): {video_path.name}")
        return

    app._adjust_video_path = video_path
    app._adjust_video_info = info
    app._adjust_image_list = []  # no image list for video mode
    app._adjust_nav_idx = 0

    w, h = info["width"], info["height"]
    dur = info["duration"]
    fps = info["fps"]
    fc = info["frame_count"]
    codec = info["codec"]

    label = f"Video: {w}x{h} {fps:.1f}fps {dur:.1f}s ({fc} frames) [{codec}]"
    app._adjust_info_label.configure(text=label)
    app._adjust_nav_label.configure(text=f"0.0s / {dur:.1f}s")
    app.log(f"[Adjust] Loaded video: {video_path.name} — {label}")

    # Load first frame for preview
    frame = extract_single_frame(str(video_path), 0.0)
    if frame is not None:
        app._adjust_source_img = frame
        import cv2
        h_px, w_px = frame.shape[:2]
        total = h_px * w_px
        if total > _PREVIEW_MAX_PIXELS:
            scale = np.sqrt(_PREVIEW_MAX_PIXELS / total)
            app._adjust_preview_buf = cv2.resize(
                frame, (int(w_px * scale), int(h_px * scale)),
                interpolation=cv2.INTER_AREA)
        else:
            app._adjust_preview_buf = frame.copy()
        _refresh_preview(app)
    else:
        app.log("[Adjust] Failed to extract first frame")


def _current_adjust_image_path(app) -> Optional[Path]:
    dataset = getattr(app, "_adjust_dataset", None)
    if dataset is not None and dataset.kind == "paired_extraction":
        images = dataset.back_images if app._adjust_paired_lens_var.get() == "Back" else dataset.front_images
        if 0 <= app._adjust_nav_idx < len(images):
            return images[app._adjust_nav_idx]
        return None
    if app._adjust_image_list and 0 <= app._adjust_nav_idx < len(app._adjust_image_list):
        return app._adjust_image_list[app._adjust_nav_idx]
    return None


def _load_current_image(app):
    """Load image at current nav index into preview buffer."""
    if not app._adjust_image_list:
        return

    path = _current_adjust_image_path(app)
    if path is None:
        return
    is_raw = path.suffix.lower() in _RAW_EXTENSIONS

    if is_raw and HAS_RAWPY:
        # Develop RAW via rawpy
        try:
            raw_state = _read_raw_state(app)
            app._adjust_source_img = load_raw(str(path), raw_state)
            app.log(f"[Adjust] RAW developed: {path.name}")
        except Exception as e:
            app.log(f"[Adjust] RAW load failed: {path.name} — {e}")
            return
    elif is_raw and not HAS_RAWPY:
        app.log(f"[Adjust] Cannot load RAW (rawpy not installed): {path.name}")
        return
    else:
        try:
            app._adjust_source_img = load_image_float(path).image
        except Exception as e:
            app.log(f"[Adjust] Failed to load: {path.name}")
            app.log(f"[Adjust] {e}")
            return

    # Downsample for preview
    import cv2
    h, w = app._adjust_source_img.shape[:2]
    total_pixels = h * w
    if total_pixels > _PREVIEW_MAX_PIXELS:
        scale = np.sqrt(_PREVIEW_MAX_PIXELS / total_pixels)
        new_w = int(w * scale)
        new_h = int(h * scale)
        app._adjust_preview_buf = cv2.resize(
            app._adjust_source_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        app._adjust_preview_buf = app._adjust_source_img.copy()

    _refresh_preview(app)


def _read_raw_state(app) -> RawDevelopState:
    """Read RAW develop UI widgets into a RawDevelopState."""
    state = app._adjust_doc.raw

    if not HAS_RAWPY:
        return state

    if hasattr(app, '_adjust_raw_demosaic_var'):
        state.demosaic_algorithm = app._adjust_raw_demosaic_var.get()
    if hasattr(app, '_adjust_raw_colorspace_var'):
        state.output_color = app._adjust_raw_colorspace_var.get()
    if hasattr(app, '_adjust_raw_wb_var'):
        wb = app._adjust_raw_wb_var.get()
        state.use_camera_wb = (wb == "Camera")
        state.use_auto_wb = (wb == "Auto")
    if "raw_exp_shift" in app._adjust_slider_vars:
        state.exp_shift = app._adjust_slider_vars["raw_exp_shift"].get()
    if hasattr(app, '_adjust_raw_highlight_var'):
        hl_map = {"Clip": 0, "Unclip": 1, "Blend": 2, "Rebuild": 3}
        state.highlight_mode = hl_map.get(app._adjust_raw_highlight_var.get(), 0)

    return state


def _redevelop_raw(app):
    """Re-develop the current image with updated RAW settings."""
    if not app._adjust_image_list:
        return
    path = app._adjust_image_list[app._adjust_nav_idx]
    if path.suffix.lower() not in _RAW_EXTENSIONS:
        app.log("[Adjust] Current image is not RAW")
        return
    _load_current_image(app)


# ══════════════════════════════════════════════════════════════════════════════
# Preview rendering
# ══════════════════════════════════════════════════════════════════════════════

def _read_state_from_sliders(app) -> AdjustmentState:
    """Read all slider values into an AdjustmentState."""
    v = app._adjust_slider_vars
    state = app._adjust_doc.adjustments
    state.exposure = v["exposure"].get()
    state.contrast = v["contrast"].get()
    state.highlights = v["highlights"].get()
    state.shadows = v["shadows"].get()
    state.whites = v["whites"].get()
    state.blacks = v["blacks"].get()
    state.temperature = v["temperature"].get()
    state.tint = v["tint"].get()
    state.saturation = v["saturation"].get()
    state.vibrance = v["vibrance"].get()
    state.sharpen_amount = v["sharpen_amount"].get()
    state.sharpen_radius = v["sharpen_radius"].get()
    state.sharpen_threshold = v["sharpen_threshold"].get()
    state.denoise_strength = v["denoise_strength"].get()
    if hasattr(app, '_adjust_denoise_method_var'):
        state.denoise_method = app._adjust_denoise_method_var.get()
    state.clahe_clip = v["clahe_clip"].get()
    state.vignette_strength = v["vignette_strength"].get()
    return state


def _on_slider_change(app):
    """Debounced handler for any slider movement."""
    # Update value labels
    for name, var in app._adjust_slider_vars.items():
        lbl = getattr(var, '_adjust_label', None)
        if lbl:
            fmt = var._adjust_fmt
            unit = var._adjust_unit
            lbl.configure(text=f"{var.get():{fmt}}{unit}")

    # Debounce preview refresh
    if app._adjust_debounce_id is not None:
        app.after_cancel(app._adjust_debounce_id)
    app._adjust_debounce_id = app.after(16, lambda: _refresh_preview(app))


def _apply_preview_pipeline(app) -> np.ndarray:
    """Apply LUT + adjustments to the preview buffer, using GPU when available."""
    result = app._adjust_preview_buf.copy()
    recipe = _recipe_from_ui(app)

    # 1. LUT (GPU-accelerated via lut.py)
    if recipe.input_lut.enabled and recipe.input_lut.path:
        proc = app._adjust_lut_processor
        lut_3d, _ = proc.load_cube(recipe.input_lut.path)
        result = proc.apply_float(result, lut_3d, recipe.input_lut.strength)

    # 2. Tone + colour adjustments
    state = _read_state_from_sliders(app)
    _has_gpu = False
    try:
        from adjust_engine import _has_torch_cuda, apply_adjustments_gpu
        _has_gpu = _has_torch_cuda()
    except ImportError:
        pass
    if not _has_gpu:
        try:
            from reconstruction_gui.adjust_engine import _has_torch_cuda, apply_adjustments_gpu
            _has_gpu = _has_torch_cuda()
        except ImportError:
            pass

    if _has_gpu:
        # GPU path: tone + colour in ~10ms
        result = apply_adjustments_gpu(result, state)
        # Spatial ops on CPU only if active (these are slower but less common during preview)
        if state.sharpen_amount > 0:
            from adjust_engine import adjust_sharpen
            result = adjust_sharpen(result, state.sharpen_amount,
                                    state.sharpen_radius, state.sharpen_threshold)
        if state.denoise_strength > 0:
            from adjust_engine import denoise_image
            result = denoise_image(result, state.denoise_strength, state.denoise_method)
        if state.clahe_clip > 0:
            from adjust_engine import adjust_clahe
            result = adjust_clahe(result, state.clahe_clip)
        if state.vignette_strength > 0:
            from adjust_engine import adjust_vignette
            result = adjust_vignette(result, state.vignette_strength)
    else:
        # CPU fallback: full pipeline
        from adjust_engine import apply_adjustments
        result = apply_adjustments(result, state)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _refresh_preview(app):
    """Apply adjustments to preview buffer and render.

    Uses the GPU pipeline for tone/colour (~10ms) when CUDA is available,
    falling back to CPU. Spatial ops (sharpen, denoise, CLAHE, vignette)
    are always CPU — they're skipped during interactive preview and only
    applied on export.
    """
    app._adjust_debounce_id = None

    if app._adjust_preview_buf is None:
        return

    if hasattr(app, "_adjust_before_after_var") and app._adjust_before_after_var.get():
        adjusted = app._adjust_preview_buf
    else:
        try:
            adjusted = _apply_preview_pipeline(app)
        except Exception as e:
            if hasattr(app, "_adjust_lut_status"):
                app._adjust_lut_status.configure(text=f"Preview error: {e}")
            adjusted = app._adjust_preview_buf

    # Apply colour correction (chart/histogram, step 8 in pipeline)
    adjusted = _apply_colour_correction(app, adjusted)

    from PIL import Image as PILImage

    rgb = (np.clip(adjusted, 0, 1) * 255).astype(np.uint8)
    rgb = rgb[:, :, ::-1]  # BGR → RGB
    pil_img = PILImage.fromarray(rgb)

    _render_to_preview(app, pil_img)


def _render_to_preview(app, pil_img):
    """Push a PIL image into the shared preview panel.

    Also sets _preview_overlay_pil so the shared zoom/scroll/pan handlers
    can re-render at different zoom levels without calling back into the
    Adjust tab.
    """
    if not hasattr(app, '_process_overlay_label'):
        return

    # Store the PIL image so the shared zoom/scroll system can re-render it
    app._preview_overlay_pil = pil_img

    zoom = app._zoom_var.get() / 100.0
    ow, oh = pil_img.size
    zw = max(1, int(ow * zoom))
    zh = max(1, int(oh * zoom))

    ctk_img = ctk.CTkImage(light_image=pil_img, size=(zw, zh))
    app._process_overlay_label.configure(image=ctk_img, text="")
    app._process_overlay_label._ctk_image = ctk_img  # standard name for GC prevention


# ══════════════════════════════════════════════════════════════════════════════
# Colour correction — shared apply logic
# ══════════════════════════════════════════════════════════════════════════════

def _apply_colour_correction(app, img: np.ndarray) -> np.ndarray:
    """Apply chart correction or histogram matching based on current mode.

    Called from both preview refresh and export pipeline.
    """
    mode = app._adjust_cc_mode_var.get() if hasattr(app, '_adjust_cc_mode_var') else "None"

    if mode == "Chart" and app._adjust_ccm is not None:
        try:
            return apply_chart_correction(
                img, app._adjust_ccm,
                method=app._adjust_ccm_method,
                degree=app._adjust_ccm_degree,
            )
        except Exception:
            return img

    if mode == "Histogram" and app._adjust_ref_img is not None:
        try:
            return match_histograms(img, app._adjust_ref_img)
        except Exception:
            return img

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Colour correction handlers
# ══════════════════════════════════════════════════════════════════════════════

def _on_cc_mode_change(app):
    """Show/hide chart or histogram sub-panels based on mode."""
    mode = app._adjust_cc_mode_var.get()
    if mode == "Chart":
        app._adjust_chart_frame.pack(fill="x", pady=(2, 0))
        app._adjust_hist_frame.pack_forget()
    elif mode == "Histogram":
        app._adjust_chart_frame.pack_forget()
        app._adjust_hist_frame.pack(fill="x", pady=(2, 0))
    else:
        app._adjust_chart_frame.pack_forget()
        app._adjust_hist_frame.pack_forget()


def _browse_chart_image(app):
    exts = " ".join(f"*{e}" for e in sorted(_IMAGE_EXTENSIONS))
    path = filedialog.askopenfilename(
        title="Select image with colour chart",
        filetypes=[("Images", exts), ("All files", "*.*")],
    )
    if path:
        app._adjust_chart_entry.delete(0, "end")
        app._adjust_chart_entry.insert(0, path)


def _browse_reference_image(app):
    exts = " ".join(f"*{e}" for e in sorted(_IMAGE_EXTENSIONS))
    path = filedialog.askopenfilename(
        title="Select reference image for histogram matching",
        filetypes=[("Images", exts), ("All files", "*.*")],
    )
    if path:
        app._adjust_hist_ref_entry.delete(0, "end")
        app._adjust_hist_ref_entry.insert(0, path)


def _detect_and_fit_chart(app):
    """Detect chart in the selected image, fit a correction matrix."""
    chart_path = app._adjust_chart_entry.get().strip()
    if not chart_path:
        app.log("[Adjust] No chart image specified")
        return

    import cv2

    chart_img = cv2.imread(chart_path, cv2.IMREAD_COLOR)
    if chart_img is None:
        app.log(f"[Adjust] Failed to load chart image: {chart_path}")
        return

    # Try cv2.mcc detection
    measured = detect_chart_mcc(chart_img)
    if measured is not None:
        app.log(f"[Adjust] Chart detected via cv2.mcc — 24 patches extracted")
    else:
        if HAS_MCC:
            app.log("[Adjust] cv2.mcc detection failed — chart not found in image")
        else:
            app.log("[Adjust] cv2.mcc not available (need opencv-contrib-python)")
        app._adjust_cc_status.configure(text="Detection failed")
        return

    # Get reference patches
    reference = get_reference_patches()
    if reference is None:
        app.log("[Adjust] colour-science not available — cannot compute correction")
        app._adjust_cc_status.configure(text="colour-science not installed")
        return

    # Fit correction matrix
    method = app._adjust_cc_method_var.get()
    degree = int(app._adjust_cc_degree_var.get())
    try:
        ccm = fit_chart_correction(measured, reference, method=method, degree=degree)
    except Exception as e:
        app.log(f"[Adjust] Chart fitting failed: {e}")
        app._adjust_cc_status.configure(text=f"Fitting failed: {e}")
        return

    app._adjust_ccm = ccm
    app._adjust_ccm_method = method
    app._adjust_ccm_degree = degree

    shape_str = f"{ccm.shape[0]}x{ccm.shape[1]}"
    app._adjust_cc_status.configure(text=f"Fitted — {method} deg={degree} ({shape_str} matrix)")
    app.log(f"[Adjust] Colour correction fitted: {method} degree={degree} matrix={shape_str}")

    _refresh_preview(app)


def _load_reference_image(app):
    """Load a reference image for histogram matching."""
    ref_path = app._adjust_hist_ref_entry.get().strip()
    if not ref_path:
        app.log("[Adjust] No reference image specified")
        return

    import cv2

    ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref is None:
        app.log(f"[Adjust] Failed to load reference: {ref_path}")
        return

    app._adjust_ref_img = ref.astype(np.float32) / 255.0
    name = Path(ref_path).name
    app._adjust_hist_status.configure(text=f"Loaded: {name}")
    app.log(f"[Adjust] Reference loaded: {name}")

    _refresh_preview(app)


# ══════════════════════════════════════════════════════════════════════════════
# Reset
# ══════════════════════════════════════════════════════════════════════════════

def _reset_all(app):
    """Reset all sliders to defaults."""
    defaults = AdjustmentState()
    mapping = {
        "exposure": defaults.exposure,
        "contrast": defaults.contrast,
        "highlights": defaults.highlights,
        "shadows": defaults.shadows,
        "whites": defaults.whites,
        "blacks": defaults.blacks,
        "temperature": defaults.temperature,
        "tint": defaults.tint,
        "saturation": defaults.saturation,
        "vibrance": defaults.vibrance,
        "sharpen_amount": defaults.sharpen_amount,
        "sharpen_radius": defaults.sharpen_radius,
        "sharpen_threshold": defaults.sharpen_threshold,
        "denoise_strength": defaults.denoise_strength,
        "clahe_clip": defaults.clahe_clip,
        "vignette_strength": defaults.vignette_strength,
    }
    for name, val in mapping.items():
        if name in app._adjust_slider_vars:
            app._adjust_slider_vars[name].set(val)
    _on_slider_change(app)


# ══════════════════════════════════════════════════════════════════════════════
# Export (threaded with progress + cancel)
# ══════════════════════════════════════════════════════════════════════════════

def _start_export(app):
    """Begin threaded batch export or in-place apply."""
    inplace = app._adjust_inplace_var.get()

    if not inplace:
        output_path = app.adjust_output_entry.get().strip()
        if not output_path:
            app.log("[Adjust] No output folder specified")
            return

    if not app._adjust_image_list and not app._adjust_video_path:
        app.log("[Adjust] No images or video loaded")
        return
    if app.is_running:
        app.log("[Adjust] Already running")
        return

    recipe = _recipe_from_ui(app)
    dataset = getattr(app, "_adjust_dataset", None)
    if dataset is None and not app._adjust_video_path:
        try:
            dataset = detect_adjust_input(app.adjust_input_entry.get().strip())
            app._adjust_dataset = dataset
        except Exception as e:
            app.log(f"[Adjust] Could not resolve input dataset: {e}")
            return

    if inplace:
        _start_inplace_apply(app, dataset, recipe)
        return

    if dataset is not None and dataset.kind != "video_sample":
        errors = validate_adjust_export(dataset, recipe)
        if errors:
            app.log(f"[Adjust] Export blocked: {'; '.join(errors)}")
            app._adjust_export_status.configure(text="Export blocked")
            return

    app._start_operation(app._adjust_run_btn, app._adjust_stop_btn)
    app._adjust_progress.set(0)
    app._adjust_export_status.configure(text="Exporting...")

    video_path = app._adjust_video_path
    video_info = app._adjust_video_info
    out_dir = Path(output_path)
    out_fmt = recipe.output.format

    def _worker():
        try:
            if video_path and video_info:
                _export_video_sample(app, video_path, video_info, out_dir, recipe)
                return

            def progress(curr, total, _name):
                if total > 0:
                    app.after(0, lambda p=min(1.0, curr / total): app._adjust_progress.set(p))

            result = export_adjusted_dataset(
                dataset,
                out_dir,
                recipe,
                refuse_non_empty=app._adjust_refuse_non_empty_var.get(),
                keep_partials=False,
                normalize_exposure=app._adjust_normalize_var.get(),
                cancel_check=lambda: app.cancel_flag.is_set(),
                progress_callback=progress,
                lut_processor=app._adjust_lut_processor,
            )
            app.after(
                0,
                lambda r=result: _finish_export(
                    app,
                    r.count,
                    out_dir,
                    r.cancelled,
                    out_fmt,
                    manifest_path=r.manifest_path,
                    input_kind=dataset.kind if dataset else "unknown",
                ),
            )
        except Exception as e:
            app.after(0, lambda err=e: _finish_export_error(app, out_dir, err))
            return

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _start_inplace_apply(app, dataset, recipe):
    """Apply adjustments to source images in-place (overwrite originals)."""
    import cv2

    if dataset is None:
        app.log("[Adjust] No dataset loaded for in-place apply")
        return

    # Collect all source images
    if dataset.is_paired:
        all_images = list(dataset.front_images) + list(dataset.back_images)
    else:
        all_images = list(dataset.images)

    if not all_images:
        app.log("[Adjust] No images to process")
        return

    errors = recipe.validate()
    if errors:
        app.log(f"[Adjust] Recipe invalid: {'; '.join(errors)}")
        return

    app._start_operation(app._adjust_run_btn, app._adjust_stop_btn)
    app._adjust_progress.set(0)
    app._adjust_export_status.configure(text="Applying in-place...")
    app.log(f"[Adjust] Applying in-place to {len(all_images)} images...")

    processor = app._adjust_lut_processor
    total = len(all_images)

    def _worker():
        count = 0
        try:
            for idx, img_path in enumerate(all_images, start=1):
                if app.cancel_flag.is_set():
                    app.after(0, lambda c=count: _finish_inplace(app, c, total, cancelled=True))
                    return
                loaded = load_image_float(img_path)
                adjusted = apply_adjustment_recipe(loaded.image, recipe, processor)
                # Write back in the source format and quality
                ext = img_path.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    out_img = np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)
                    cv2.imwrite(str(img_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, recipe.output.quality])
                elif ext == ".png":
                    if loaded.source_dtype == "uint16":
                        out_img = np.clip(np.rint(adjusted * 65535.0), 0, 65535).astype(np.uint16)
                    else:
                        out_img = np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)
                    cv2.imwrite(str(img_path), out_img)
                else:
                    out_img = np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)
                    cv2.imwrite(str(img_path), out_img)
                count += 1
                if total > 0:
                    app.after(0, lambda p=min(1.0, idx / total): app._adjust_progress.set(p))
            app.after(0, lambda c=count: _finish_inplace(app, c, total, cancelled=False))
        except Exception as e:
            app.after(0, lambda err=e, c=count: _finish_inplace_error(app, c, err))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _finish_inplace(app, count, total, cancelled):
    app._stop_operation(app._adjust_run_btn, app._adjust_stop_btn)
    if cancelled:
        app.log(f"[Adjust] In-place apply cancelled after {count}/{total} images")
        app._adjust_export_status.configure(text=f"Cancelled ({count}/{total})")
    else:
        app.log(f"[Adjust] In-place apply complete: {count} images modified")
        app._adjust_export_status.configure(text=f"Applied to {count} images")
    # Reload current preview to show updated image
    _load_current_image(app)


def _finish_inplace_error(app, count, err):
    app._stop_operation(app._adjust_run_btn, app._adjust_stop_btn)
    app.log(f"[Adjust] In-place apply failed after {count} images: {err}")
    app._adjust_export_status.configure(text=f"Error after {count} images")


def _export_video_sample(app, video_path: Path, video_info: dict, out_dir: Path, recipe: AdjustmentRecipe):
    """Export one adjusted sample frame per second from a video input."""
    from prep360.core.color_pipeline import output_extension

    if out_dir.exists() and app._adjust_refuse_non_empty_var.get() and any(out_dir.iterdir()):
        raise ValueError(f"Output folder is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = video_info["duration"]
    interval = 1.0
    total = max(1, int(duration / interval))
    stem = video_path.stem
    ext = output_extension(recipe.output)
    count = 0
    written = []
    try:
        for i in range(total):
            if app.cancel_flag.is_set():
                break
            t = i * interval
            frame = extract_single_frame(str(video_path), t)
            if frame is None:
                continue
            adjusted = apply_adjustment_recipe(frame, recipe, app._adjust_lut_processor)
            adjusted = _apply_colour_correction(app, adjusted)
            out_file = out_dir / f"{stem}_{i:05d}{ext}"
            write_image_float(out_file, adjusted, recipe.output)
            written.append(out_file)
            count += 1
            app.after(0, lambda p=(i + 1) / total: app._adjust_progress.set(p))

        cancelled = app.cancel_flag.is_set()
        if cancelled:
            for path in reversed(written):
                try:
                    path.unlink()
                except OSError:
                    pass
        app.after(0, lambda: _finish_export(app, count, out_dir, cancelled, recipe.output.format,
                                            input_kind="video_sample"))
    except Exception:
        for path in reversed(written):
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _finish_export(app, count: int, out_dir: Path, cancelled: bool,
                   out_fmt: str = "jpg", manifest_path: Optional[Path] = None,
                   input_kind: str = ""):
    """Called on main thread after export completes."""
    app._stop_operation(app._adjust_run_btn, app._adjust_stop_btn)

    if cancelled:
        msg = f"Cancelled — partial {out_fmt} outputs removed"
    else:
        msg = f"Exported {count} images as {out_fmt}"
    app._adjust_export_status.configure(text=msg)
    app.log(f"[Adjust] {msg} → {out_dir}")
    if manifest_path:
        app.log(f"[Adjust] Manifest: {manifest_path}")

    if count > 0 and not cancelled:
        dataset = getattr(app, "_adjust_dataset", None)
        if dataset and dataset.kind == "paired_extraction":
            input_path = str(dataset.root)
        elif dataset and dataset.images:
            input_path = str(dataset.root)
        elif app._adjust_video_path:
            input_path = str(app._adjust_video_path)
        else:
            input_path = str(app._adjust_image_list[0].parent) if app._adjust_image_list else ""
        app.record_activity(
            operation="adjust_export",
            input_path=input_path,
            output_path=str(out_dir),
            status="completed",
            details={"count": count, "format": out_fmt, "input_kind": input_kind},
        )


def _finish_export_error(app, out_dir: Path, error: Exception):
    app._stop_operation(app._adjust_run_btn, app._adjust_stop_btn)
    app._adjust_export_status.configure(text=f"Export failed: {error}")
    app.log(f"[Adjust] Export failed → {out_dir}: {error}")

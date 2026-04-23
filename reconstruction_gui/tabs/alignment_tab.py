"""
Alignment Tab - sparse COLMAP / SphereSfM alignment UI.

This tab is intentionally treated as an advanced surface. It exposes a broader
set of reconstruction controls plus raw per-stage CLI passthrough fields, while
keeping the right-side column focused on alignment-specific status, summary, and
log output instead of the shared Mask/Review preview shell.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import customtkinter as ctk

try:
    from reconstruction_gui.alignment_profiles import (
        build_colmap_pinhole_profile,
        build_spheresfm_erp_profile,
    )
    from reconstruction_gui.widgets import (
        Section, CollapsibleSection, Tooltip,
        COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
        COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
        COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
        COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
        COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
        FONT_TEXT_MONO_VALUE, FONT_TEXT_CONSOLE,
        LABEL_FIELD_WIDTH, BROWSE_BUTTON_WIDTH,
    )
except ImportError:
    from alignment_profiles import build_colmap_pinhole_profile, build_spheresfm_erp_profile
    from widgets import (
        Section, CollapsibleSection, Tooltip,
        COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
        COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
        COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
        COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
        COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
        FONT_TEXT_MONO_VALUE, FONT_TEXT_CONSOLE,
        LABEL_FIELD_WIDTH, BROWSE_BUTTON_WIDTH,
    )


ALIGNMENT_STAGES: List[Tuple[str, str]] = [
    ("feature_extraction", "Feature Extraction"),
    ("matching", "Feature Matching"),
    ("reconstruction", "Reconstruction"),
]

STAGE_COLORS = {
    "pending": "#94A3B8",
    "running": "#F59E0B",
    "complete": "#22C55E",
    "warning": "#F97316",
    "failed": "#EF4444",
    "cancelled": "#F97316",
}


def build_alignment_tab(app, parent):
    """Populate the Alignment tab and create the right-side detail panel.

    Left column: single scrollable frame with all sections in order:
      Inputs, General, Matching, Reconstruction, Advanced, Actions,
      Run Status & Model, Alignment Log.

    Right column (detail panel): header, stats strip, viewer, controls, summary.
    """
    scroll = ctk.CTkScrollableFrame(parent)
    scroll.pack(fill="both", expand=True)

    app._alignment_runner = None
    app._alignment_thread = None
    app._alignment_cancel_event = threading.Event()
    app._alignment_snapshot_signature = ""
    app._alignment_stage_states = {stage: "pending" for stage, _ in ALIGNMENT_STAGES}
    app._alignment_stage_results = {}
    app._alignment_last_auto_binary = ""
    app._alignment_last_auto_workspace = ""
    app._alignment_last_auto_camera_model = ""
    app._alignment_last_selected_model_dir = ""
    app._alignment_quality_info = {}
    app._alignment_diag_info = {}

    _build_alignment_paths_section(app, scroll)
    _build_alignment_engine_section(app, scroll)
    _build_alignment_engine_panel(app, scroll)
    _build_alignment_rig_section(app, scroll)
    _build_alignment_extract_section(app, scroll)
    _build_alignment_match_section(app, scroll)
    _build_alignment_reconstruct_section(app, scroll)
    _build_alignment_advanced_section(app, scroll)
    _build_alignment_actions_section(app, scroll)

    _build_alignment_left_diagnostics(app, scroll)

    # Right column detail panel
    _build_alignment_detail_panel(app)

    _reset_alignment_session(app, clear_summary=True)
    app.after(0, lambda: _on_engine_change(app, force=False))


def _build_alignment_paths_section(app, parent):
    sec = Section(parent, "Alignment Inputs")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Images:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_images_entry = ctk.CTkEntry(
        row,
        placeholder_text="Directory containing alignment images",
    )
    app.alignment_images_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.alignment_images_entry.bind("<FocusOut>", lambda _e: _maybe_autofill_alignment_workspace(app))
    ctk.CTkButton(
        row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_folder_for(app.alignment_images_entry),
    ).pack(side="left")
    Tooltip(app.alignment_images_entry, "Directory of images to reconstruct.")

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_masks_entry = ctk.CTkEntry(row, placeholder_text="Optional mask directory")
    app.alignment_masks_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_folder_for(app.alignment_masks_entry),
    ).pack(side="left")
    Tooltip(app.alignment_masks_entry, "Optional image-mask directory.")

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Workspace:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_workspace_entry = ctk.CTkEntry(row, placeholder_text="Run workspace root")
    app.alignment_workspace_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_folder_for(app.alignment_workspace_entry),
    ).pack(side="left")
    Tooltip(
        app.alignment_workspace_entry,
        "Root folder for per-run alignment outputs. Each run gets its own subdirectory.",
    )


def _build_alignment_engine_section(app, parent):
    """Engine radio toggle: COLMAP or SphereSfM."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=(0, 6), padx=4)

    ctk.CTkLabel(row, text="Engine:", width=LABEL_FIELD_WIDTH, anchor="e",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")

    app.alignment_engine_var = ctk.StringVar(value="colmap")

    ctk.CTkRadioButton(
        row, text="COLMAP", variable=app.alignment_engine_var, value="colmap",
        command=lambda: _on_engine_change(app),
    ).pack(side="left", padx=(6, 12))

    ctk.CTkRadioButton(
        row, text="SphereSfM", variable=app.alignment_engine_var, value="spheresfm",
        command=lambda: _on_engine_change(app),
    ).pack(side="left")


def _build_alignment_engine_panel(app, parent):
    """Engine-specific settings panel. Content swaps on engine change."""
    sec = Section(parent, "Engine Settings")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # -- COLMAP frame --
    app._alignment_colmap_frame = ctk.CTkFrame(c, fg_color="transparent")

    row = ctk.CTkFrame(app._alignment_colmap_frame, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Camera model:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_camera_model_var = ctk.StringVar(value="PINHOLE")
    app.alignment_camera_model_menu = ctk.CTkOptionMenu(
        row,
        values=["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "SIMPLE_RADIAL", "RADIAL",
                "SIMPLE_PINHOLE", "FULL_OPENCV"],
        variable=app.alignment_camera_model_var,
        width=180,
    )
    app.alignment_camera_model_menu.pack(side="left", padx=(6, 12))

    app.alignment_single_camera_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        row, text="Single camera", variable=app.alignment_single_camera_var,
    ).pack(side="left", padx=(10, 0))

    # -- SphereSfM frame --
    app._alignment_spheresfm_frame = ctk.CTkFrame(c, fg_color="transparent")

    row = ctk.CTkFrame(app._alignment_spheresfm_frame, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Camera model:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    ctk.CTkLabel(row, text="SPHERE (fixed)", anchor="w",
                 text_color="#94A3B8").pack(side="left", padx=(6, 12))
    ctk.CTkLabel(row, text="Pose path is in Advanced > Camera And Reader", anchor="w",
                 text_color="#64748b", font=ctk.CTkFont(size=10)).pack(side="left", padx=(10, 0))

    # Show COLMAP frame by default
    app._alignment_colmap_frame.pack(fill="x")


def _build_alignment_rig_section(app, parent):
    """Rig Configuration — collapsible, greyed out for SphereSfM."""
    sec = CollapsibleSection(parent, "Rig Configuration", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    app._alignment_rig_section = sec
    c = sec.content

    # Preset row
    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Preset:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")

    try:
        from reconstruction_gui.rig_presets import PRESET_DISPLAY_NAMES
    except ImportError:
        from rig_presets import PRESET_DISPLAY_NAMES

    app.alignment_rig_preset_var = ctk.StringVar(value="None")
    app.alignment_rig_preset_menu = ctk.CTkOptionMenu(
        row,
        values=PRESET_DISPLAY_NAMES,
        variable=app.alignment_rig_preset_var,
        width=220,
        command=lambda _: _on_rig_preset_change(app),
    )
    app.alignment_rig_preset_menu.pack(side="left", padx=(6, 4))

    # File row
    row2 = ctk.CTkFrame(c, fg_color="transparent")
    row2.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row2, text="File:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_rig_file_entry = ctk.CTkEntry(
        row2, placeholder_text="Custom rig_config.json",
    )
    app.alignment_rig_file_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app._alignment_rig_file_browse_btn = ctk.CTkButton(
        row2, text="...", width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: _browse_rig_config_file(app),
    )
    app._alignment_rig_file_browse_btn.pack(side="left")

    # Read-only summary
    app.alignment_rig_summary_text = ctk.CTkTextbox(
        c, height=80,
        font=ctk.CTkFont(family="Consolas", size=10),
        fg_color="#141414",
        state="disabled",
    )
    app.alignment_rig_summary_text.pack(fill="x", padx=6, pady=(3, 6))

    _update_rig_file_entry_state(app)


def _build_alignment_extract_section(app, parent):
    """Extract settings — non-collapsible, always expanded."""
    sec = Section(parent, "Extract")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Max features:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_max_features_entry = ctk.CTkEntry(row, width=120)
    app.alignment_max_features_entry.insert(0, "8192")
    app.alignment_max_features_entry.pack(side="left", padx=(6, 12))

    ctk.CTkLabel(row, text="Max image size:", width=96, anchor="e").pack(side="left", padx=(10, 0))
    app.alignment_max_image_size_entry = ctk.CTkEntry(row, width=120)
    app.alignment_max_image_size_entry.insert(0, "2048")
    app.alignment_max_image_size_entry.pack(side="left", padx=(6, 0))


def _build_alignment_tool_section(app, parent):
    sec = CollapsibleSection(parent, "Tool Setup", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    app._alignment_binary_override_visible = False

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Binary:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_binary_summary_label = ctk.CTkLabel(
        row,
        text="Detecting preferred binary...",
        anchor="w",
        justify="left",
    )
    app.alignment_binary_summary_label.pack(side="left", fill="x", expand=True, padx=(6, 4))

    app.alignment_binary_change_button = ctk.CTkButton(
        row,
        text="Change...",
        width=96,
        command=lambda: _toggle_alignment_binary_override(app),
    )
    app.alignment_binary_change_button.pack(side="left", padx=(0, 4))
    app.alignment_binary_preferred_button = ctk.CTkButton(
        row,
        text="Use Preferred",
        width=120,
        command=lambda: _on_engine_change(app, force=True),
    )
    app.alignment_binary_preferred_button.pack(side="left")

    path_row = ctk.CTkFrame(c, fg_color="transparent")
    path_row.pack(fill="x", pady=(0, 2), padx=6)
    app.alignment_binary_path_label = ctk.CTkLabel(
        path_row,
        text="",
        anchor="w",
        justify="left",
        text_color="#CBD5E1",
        wraplength=900,
    )
    app.alignment_binary_path_label.pack(side="left", fill="x", expand=True, padx=(102, 0))

    hint_row = ctk.CTkFrame(c, fg_color="transparent")
    hint_row.pack(fill="x", pady=(0, 3), padx=6)
    app.alignment_binary_hint_label = ctk.CTkLabel(
        hint_row,
        text="",
        anchor="w",
        justify="left",
        text_color="#94A3B8",
        wraplength=900,
    )
    app.alignment_binary_hint_label.pack(side="left", fill="x", expand=True, padx=(102, 0))

    app.alignment_binary_override_frame = ctk.CTkFrame(c, fg_color="transparent")

    override_row = ctk.CTkFrame(app.alignment_binary_override_frame, fg_color="transparent")
    override_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(override_row, text="Override:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_binary_entry = ctk.CTkEntry(
        override_row,
        placeholder_text="COLMAP / SphereSfM executable",
    )
    app.alignment_binary_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    app.alignment_binary_entry.bind("<FocusOut>", lambda _e: _update_alignment_binary_hint(app))
    ctk.CTkButton(
        override_row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: _browse_alignment_binary_override(app),
    ).pack(side="left")
    Tooltip(
        app.alignment_binary_entry,
        "Optional manual override. Most users should keep the preferred binary and leave this unchanged.",
    )


def _build_alignment_camera_section(app, parent):
    sec = CollapsibleSection(parent, "Camera And Reader", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Camera params:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_camera_params_entry = ctk.CTkEntry(
        row,
        placeholder_text="Optional camera params string",
    )
    app.alignment_camera_params_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Pose path:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_pose_path_entry = ctk.CTkEntry(
        row,
        placeholder_text="Optional pose path (SphereSfM / pose-aware workflows)",
    )
    app.alignment_pose_path_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_file_for(
            app.alignment_pose_path_entry,
            title="Select Pose File",
            filetypes=[("All Files", "*.*")],
        ),
    ).pack(side="left")

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Camera mask:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_camera_mask_path_entry = ctk.CTkEntry(
        row,
        placeholder_text="Optional camera mask file (e.g. fisheye / ERP masking)",
    )
    app.alignment_camera_mask_path_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        row,
        text="...",
        width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_file_for(
            app.alignment_camera_mask_path_entry,
            title="Select Camera Mask File",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All Files", "*.*")],
        ),
    ).pack(side="left")


def _build_alignment_match_section(app, parent):
    """Match settings — non-collapsible, always expanded."""
    sec = Section(parent, "Match")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Strategy:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_strategy_var = ctk.StringVar(value="sequential")
    ctk.CTkOptionMenu(
        row,
        values=["exhaustive", "sequential", "spatial", "vocab_tree"],
        variable=app.alignment_strategy_var,
        width=160,
        command=lambda _: _on_strategy_change(app),
    ).pack(side="left", padx=(6, 12))

    ctk.CTkLabel(row, text="Max matches:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left", padx=(10, 0))
    app.alignment_max_num_matches_entry = ctk.CTkEntry(row, width=120)
    app.alignment_max_num_matches_entry.insert(0, "32768")
    app.alignment_max_num_matches_entry.pack(side="left", padx=(6, 4))

    app.alignment_guided_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(row, text="Guided matching", variable=app.alignment_guided_var).pack(
        side="left", padx=(10, 0))

    # Conditional: spatial fields (hidden by default)
    app._alignment_spatial_frame = ctk.CTkFrame(c, fg_color="transparent")
    s_row = ctk.CTkFrame(app._alignment_spatial_frame, fg_color="transparent")
    s_row.pack(fill="x", pady=3, padx=6)
    app.alignment_spatial_is_gps_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(s_row, text="Spatial matcher uses GPS",
                    variable=app.alignment_spatial_is_gps_var).pack(side="left")
    ctk.CTkLabel(s_row, text="Max distance:", width=LABEL_FIELD_WIDTH, anchor="e").pack(
        side="left", padx=(10, 0))
    app.alignment_spatial_max_distance_entry = ctk.CTkEntry(
        s_row, width=120, placeholder_text="Optional",
    )
    app.alignment_spatial_max_distance_entry.pack(side="left", padx=(6, 0))

    # Conditional: vocab tree fields (hidden by default)
    app._alignment_vocab_frame = ctk.CTkFrame(c, fg_color="transparent")
    v_row = ctk.CTkFrame(app._alignment_vocab_frame, fg_color="transparent")
    v_row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(v_row, text="Vocab tree:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_vocab_tree_entry = ctk.CTkEntry(
        v_row, placeholder_text="Required for vocab_tree matcher",
    )
    app.alignment_vocab_tree_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        v_row, text="...", width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_file_for(
            app.alignment_vocab_tree_entry,
            title="Select Vocabulary Tree",
            filetypes=[("All Files", "*.*")],
        ),
    ).pack(side="left")


def _build_alignment_reconstruct_section(app, parent):
    """Reconstruct settings — non-collapsible, always expanded."""
    sec = Section(parent, "Reconstruct")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    row = ctk.CTkFrame(c, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text="Mapper:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_mapper_var = ctk.StringVar(value="incremental")
    ctk.CTkOptionMenu(
        row, values=["incremental", "global"],
        variable=app.alignment_mapper_var, width=160,
    ).pack(side="left", padx=(6, 12))

    ctk.CTkLabel(row, text="Min inliers:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left", padx=(10, 0))
    app.alignment_min_num_inliers_entry = ctk.CTkEntry(row, width=120)
    app.alignment_min_num_inliers_entry.insert(0, "15")
    app.alignment_min_num_inliers_entry.pack(side="left", padx=(6, 0))

    row2 = ctk.CTkFrame(c, fg_color="transparent")
    row2.pack(fill="x", pady=3, padx=6)

    app.alignment_ba_refine_focal_var = ctk.BooleanVar(value=True)
    app._alignment_ba_refine_focal_cb = ctk.CTkCheckBox(
        row2, text="Refine focal", variable=app.alignment_ba_refine_focal_var,
    )
    app._alignment_ba_refine_focal_cb.pack(side="left", padx=(0, 10))

    app.alignment_ba_refine_principal_var = ctk.BooleanVar(value=False)
    app._alignment_ba_refine_principal_cb = ctk.CTkCheckBox(
        row2, text="Refine principal", variable=app.alignment_ba_refine_principal_var,
    )
    app._alignment_ba_refine_principal_cb.pack(side="left", padx=(0, 10))

    app.alignment_ba_refine_extra_var = ctk.BooleanVar(value=True)
    app._alignment_ba_refine_extra_cb = ctk.CTkCheckBox(
        row2, text="Refine extra", variable=app.alignment_ba_refine_extra_var,
    )
    app._alignment_ba_refine_extra_cb.pack(side="left")

    row3 = ctk.CTkFrame(c, fg_color="transparent")
    row3.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row3, text="Snapshots:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
    app.alignment_snapshot_path_entry = ctk.CTkEntry(
        row3, placeholder_text="Optional (blank = auto when freq > 0)",
    )
    app.alignment_snapshot_path_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    ctk.CTkButton(
        row3, text="...", width=BROWSE_BUTTON_WIDTH,
        fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: app._browse_folder_for(app.alignment_snapshot_path_entry),
    ).pack(side="left", padx=(0, 4))
    ctk.CTkLabel(row3, text="Freq:", width=40, anchor="e").pack(side="left", padx=(10, 0))
    app.alignment_snapshot_freq_entry = ctk.CTkEntry(row3, width=80, placeholder_text="0")
    app.alignment_snapshot_freq_entry.pack(side="left", padx=(6, 0))


def _build_alignment_advanced_section(app, parent):
    """Advanced section — collapsed by default, nests Tool Setup, Camera, CLI Args."""
    outer = CollapsibleSection(parent, "Advanced", expanded=False)
    outer.pack(fill="x", pady=(0, 6), padx=4)
    inner = outer.content

    _build_alignment_tool_section(app, inner)
    _build_alignment_camera_section(app, inner)
    _build_alignment_cli_passthrough_section(app, inner)


def _build_alignment_cli_passthrough_section(app, parent):
    sec = CollapsibleSection(parent, "Per-Stage CLI Args", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    ctk.CTkLabel(
        c,
        text="One key=value pair per line. Blank lines and lines starting with # are ignored.",
        text_color="#9CA3AF",
        anchor="w",
        justify="left",
    ).pack(fill="x", padx=8, pady=(2, 6))

    _build_cli_args_textbox(app, c, "Extract args", "alignment_extract_args_text")
    _build_cli_args_textbox(app, c, "Match args", "alignment_match_args_text")
    _build_cli_args_textbox(app, c, "Reconstruct args", "alignment_reconstruct_args_text")


def _build_cli_args_textbox(app, parent, label: str, attr_name: str):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(row, text=label, width=96, anchor="nw").pack(side="left", pady=(4, 0))
    textbox = ctk.CTkTextbox(
        row,
        height=72,
        font=ctk.CTkFont(family="Consolas", size=11),
        fg_color="#1A1A1A",
    )
    textbox.pack(side="left", fill="x", expand=True, padx=(6, 0))
    setattr(app, attr_name, textbox)


def _build_alignment_actions_section(app, parent):
    sec = Section(parent, "Actions")
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    btn_row = ctk.CTkFrame(c, fg_color="transparent")
    btn_row.pack(fill="x", pady=4, padx=6)

    app.alignment_run_btn = ctk.CTkButton(
        btn_row,
        text="Align",
        command=lambda: _start_alignment(app, mode="full"),
        fg_color=COLOR_ACTION_PRIMARY,
        hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=13, weight="bold"),
        height=38,
    )
    app.alignment_run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

    app.alignment_next_btn = ctk.CTkButton(
        btn_row,
        text="Next Step",
        command=lambda: _start_alignment(app, mode="step"),
        fg_color=COLOR_ACTION_SECONDARY,
        hover_color=COLOR_ACTION_SECONDARY_H,
        font=ctk.CTkFont(size=12),
        height=38,
        width=110,
    )
    app.alignment_next_btn.pack(side="left", padx=(0, 4))

    app.alignment_cancel_btn = ctk.CTkButton(
        btn_row,
        text="Cancel",
        command=lambda: _cancel_alignment(app),
        fg_color=COLOR_ACTION_DANGER,
        hover_color=COLOR_ACTION_DANGER_H,
        font=ctk.CTkFont(size=12),
        height=38,
        width=90,
        state="disabled",
    )
    app.alignment_cancel_btn.pack(side="left", padx=(0, 4))

    ctk.CTkButton(
        btn_row, text="Load Model", width=100, height=38,
        font=ctk.CTkFont(size=12), fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        command=lambda: _on_viewer_load_model(app),
    ).pack(side="left")


def _build_alignment_left_diagnostics(app, parent):
    """Build Run Status & Model and Alignment Log below the scrollable settings.

    Run Status & Model is a collapsible section.
    Alignment Log fills all remaining vertical space when expanded (min 300px).
    """
    # Run Status & Model
    status_sec = CollapsibleSection(parent, "Run Status & Model", expanded=True)
    status_sec.pack(fill="x", padx=4, pady=(2, 0))
    c = status_sec.content

    run_dir_row = ctk.CTkFrame(c, fg_color="transparent", height=18)
    run_dir_row.pack(fill="x", pady=0, padx=6)
    ctk.CTkLabel(run_dir_row, text="Run Dir:", width=LABEL_FIELD_WIDTH, anchor="e", height=18,
                 font=ctk.CTkFont(size=10)).pack(side="left")
    app.alignment_run_dir_label = ctk.CTkLabel(
        run_dir_row, text="Not started", anchor="w", justify="left", height=18,
        font=ctk.CTkFont(size=10), text_color="#94a3b8", wraplength=600,
    )
    app.alignment_run_dir_label.pack(side="left", fill="x", expand=True, padx=(4, 0))

    app.alignment_mask_status_label = _detail_value_row(c, "Masks:", font_size=10)
    app.alignment_snapshot_status_label = _detail_value_row(c, "Snapshots:", font_size=10)
    app.alignment_last_progress_label = _detail_value_row(c, "Progress:", font_size=10)
    app.alignment_selected_model_notes_label = _detail_value_row(c, "Notes:", font_size=10)

    # Alignment Log — fills remaining vertical space
    log_frame = ctk.CTkFrame(parent, fg_color="transparent")
    log_frame.pack(fill="both", expand=True, padx=4, pady=(2, 4))

    log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
    log_header.pack(fill="x", pady=(2, 0))
    app._alignment_log_expanded = True
    app._alignment_log_frame = log_frame

    def _toggle_log():
        if app._alignment_log_expanded:
            app.alignment_log_text.pack_forget()
            app._alignment_log_toggle_btn.configure(text="\u25B6")
            app._alignment_log_expanded = False
        else:
            app.alignment_log_text.pack(fill="both", expand=True, padx=4, pady=(2, 2))
            app._alignment_log_toggle_btn.configure(text="\u25BC")
            app._alignment_log_expanded = True

    app._alignment_log_toggle_btn = ctk.CTkButton(
        log_header, text="\u25BC", width=20, height=20,
        fg_color="transparent", hover_color=("gray75", "gray25"),
        text_color=("#888888", "#666666"),
        command=_toggle_log,
    )
    app._alignment_log_toggle_btn.pack(side="left", padx=(6, 2))
    ctk.CTkLabel(
        log_header, text="Alignment Log",
        font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
    ).pack(side="left")

    app.alignment_log_text = ctk.CTkTextbox(
        log_frame, height=300,
        font=ctk.CTkFont(family="Consolas", size=10),
        fg_color="#141414",
    )
    app.alignment_log_text.pack(fill="both", expand=True, padx=4, pady=(2, 2))
    _make_readonly_selectable(app.alignment_log_text)
    _capture_scroll(app.alignment_log_text)


def _build_alignment_detail_panel(app):
    """Build the right-column detail panel with viewer-dominant layout.

    Layout (top to bottom):
      1. Header bar: "Alignment Session" + status badge + stage status
      2. Summary strip: registered, points, reproj error, quality
      3. Viewer area: placeholder tk.Frame for PointCloudViewer
      4. Viewer controls: color mode, point size, toggles, reset/save
      5. Summary: collapsible text summary
    """
    import tkinter as tk

    app._alignment_detail_panel = ctk.CTkFrame(app._main_frame, corner_radius=0)
    panel = app._alignment_detail_panel

    # Rows: header, viewer, controls, summary
    panel.grid_rowconfigure(0, weight=0)  # header
    panel.grid_rowconfigure(1, weight=6)  # viewer (dominant)
    panel.grid_rowconfigure(2, weight=0)  # viewer controls
    panel.grid_rowconfigure(3, weight=1)  # summary
    panel.grid_columnconfigure(0, weight=1)

    # ── Row 0: Header bar (title + stages + stats in one row) ──
    header = ctk.CTkFrame(panel, fg_color=("gray90", "gray20"), height=36, corner_radius=0)
    header.grid(row=0, column=0, sticky="ew")
    header.grid_propagate(False)

    ctk.CTkLabel(
        header, text="Alignment Session",
        font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
    ).pack(side="left", padx=(12, 8))

    app._alignment_status_badge = ctk.CTkLabel(
        header, text="Ready", font=ctk.CTkFont(size=10, weight="bold"),
        fg_color="#374151", corner_radius=4, text_color="#94a3b8",
        width=70, height=22,
    )
    app._alignment_status_badge.pack(side="left", padx=(0, 12))

    # Stage labels
    app._alignment_stage_labels = {}
    for stage_key, label in ALIGNMENT_STAGES:
        stage_label = ctk.CTkLabel(
            header, text=f"{label}: pending",
            text_color=STAGE_COLORS["pending"],
            font=ctk.CTkFont(size=10), anchor="w",
        )
        stage_label.pack(side="left", padx=(0, 8))
        app._alignment_stage_labels[stage_key] = stage_label

    # Quality badge (after stages, before stats)
    app.alignment_selected_model_quality_label = ctk.CTkLabel(
        header, text="", font=ctk.CTkFont(size=11, weight="bold"),
        text_color="#94a3b8", anchor="w",
    )
    app.alignment_selected_model_quality_label.pack(side="left", padx=(8, 12))

    # Stats: pts / images / reproj (inline in header)
    _stat_font = ctk.CTkFont(size=11, weight="bold")
    _unit_font = ctk.CTkFont(size=10)

    ctk.CTkLabel(header, text="pts", font=_unit_font,
                 text_color="#64748b").pack(side="left", padx=(0, 2))
    app.alignment_selected_model_points_label = ctk.CTkLabel(
        header, text="—", font=_stat_font, text_color="#e2e8f0",
    )
    app.alignment_selected_model_points_label.pack(side="left", padx=(0, 10))

    ctk.CTkLabel(header, text="images", font=_unit_font,
                 text_color="#64748b").pack(side="left", padx=(0, 2))
    app.alignment_selected_model_registered_label = ctk.CTkLabel(
        header, text="—", font=_stat_font, text_color="#e2e8f0",
    )
    app.alignment_selected_model_registered_label.pack(side="left", padx=(0, 10))

    ctk.CTkLabel(header, text="reproj", font=_unit_font,
                 text_color="#64748b").pack(side="left", padx=(0, 2))
    app.alignment_selected_model_reproj_label = ctk.CTkLabel(
        header, text="—", font=_stat_font, text_color="#e2e8f0",
    )
    app.alignment_selected_model_reproj_label.pack(side="left", padx=(0, 10))

    # Path (right-aligned)
    app.alignment_selected_model_path_label = ctk.CTkLabel(
        header, text="", font=ctk.CTkFont(size=10),
        text_color="#64748b", anchor="e",
    )
    app.alignment_selected_model_path_label.pack(side="right", padx=(0, 12))

    # ── Row 1: Viewer area (placeholder) ──
    # Raw tk.Frame — PointCloudViewer will embed VTK here via HWND
    app._alignment_viewer_frame = tk.Frame(panel, bg="#141414")
    app._alignment_viewer_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)

    # Placeholder label shown when no model is loaded
    app._alignment_viewer_placeholder = ctk.CTkLabel(
        app._alignment_viewer_frame,
        text="Run alignment to see sparse reconstruction",
        font=ctk.CTkFont(size=13), text_color="#334155",
    )
    app._alignment_viewer_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    # ── Row 2: Viewer controls bar ──
    ctrl_bar = ctk.CTkFrame(panel, fg_color=("gray90", "gray20"), height=56, corner_radius=0)
    ctrl_bar.grid(row=2, column=0, sticky="ew", pady=(4, 4))
    ctrl_bar.grid_propagate(False)
    app._alignment_viewer_controls = ctrl_bar

    # Reset View first (leftmost)
    ctk.CTkButton(
        ctrl_bar, text="Reset View", width=80, height=28,
        font=ctk.CTkFont(size=12), fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        command=lambda: _on_viewer_reset(app),
    ).pack(side="left", padx=(10, 4))

    ctk.CTkButton(
        ctrl_bar, text="Set Upright", width=90, height=28,
        font=ctk.CTkFont(size=12), fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
        command=lambda: _on_set_upright(app),
    ).pack(side="left", padx=(0, 10))

    ctk.CTkLabel(ctrl_bar, text="Color:", font=ctk.CTkFont(size=12),
                 text_color="#94a3b8").pack(side="left", padx=(0, 3))
    app._alignment_color_mode_var = ctk.StringVar(value="rgb")
    ctk.CTkOptionMenu(
        ctrl_bar, variable=app._alignment_color_mode_var,
        values=["rgb", "reproj_error", "track_length", "depth", "elevation"],
        width=130, height=28, font=ctk.CTkFont(size=12),
        command=lambda _: _on_viewer_color_change(app),
    ).pack(side="left", padx=(0, 10))

    ctk.CTkLabel(ctrl_bar, text="Size:", font=ctk.CTkFont(size=12),
                 text_color="#94a3b8").pack(side="left", padx=(0, 3))
    app._alignment_point_size_var = ctk.DoubleVar(value=3.0)
    ctk.CTkSlider(
        ctrl_bar, from_=1, to=10, variable=app._alignment_point_size_var,
        width=160, height=16,
        command=lambda _: _on_viewer_point_size_change(app),
    ).pack(side="left", padx=(0, 10))

    app._alignment_show_cameras_var = ctk.BooleanVar(value=True)
    ctk.CTkCheckBox(
        ctrl_bar, text="Cameras", variable=app._alignment_show_cameras_var,
        font=ctk.CTkFont(size=12), width=0, height=24, checkbox_width=16, checkbox_height=16,
        command=lambda: _on_viewer_camera_toggle(app),
    ).pack(side="left", padx=(0, 10))

    # Separator before roll slider
    ctk.CTkFrame(ctrl_bar, width=1, height=24, fg_color="#4b5563").pack(
        side="left", padx=(2, 8))

    # CW/CCW roll slider
    ctk.CTkLabel(ctrl_bar, text="Roll", font=ctk.CTkFont(size=12),
                 text_color="#94a3b8").pack(side="left", padx=(0, 3))
    app._alignment_roll_var = ctk.DoubleVar(value=0.0)
    ctk.CTkSlider(
        ctrl_bar, from_=-180, to=180, variable=app._alignment_roll_var,
        width=400, height=14,
        command=lambda _: _on_roll_slider(app),
    ).pack(side="left", padx=(0, 6))

    # Right side: session actions (disabled until model is loaded)
    app.alignment_reset_btn = ctk.CTkButton(
        ctrl_bar, text="Reset Session", width=110, height=28,
        font=ctk.CTkFont(size=12), fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        state="disabled",
        command=lambda: _reset_alignment_session(app, clear_summary=True),
    )
    app.alignment_reset_btn.pack(side="right", padx=(0, 10))

    app.alignment_send_to_coverage_btn = ctk.CTkButton(
        ctrl_bar, text="Send To Coverage", width=130, height=28,
        font=ctk.CTkFont(size=12), fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        state="disabled",
        command=lambda: _send_alignment_to_coverage(app, switch_tab=True),
    )
    app.alignment_send_to_coverage_btn.pack(side="right", padx=(0, 6))

    # ── Row 3: Summary ──
    # Use a plain frame (not CollapsibleSection) so it fills grid weight properly
    summary_frame = ctk.CTkFrame(panel, corner_radius=0)
    summary_frame.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 4))
    summary_frame.grid_rowconfigure(1, weight=1)
    summary_frame.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        summary_frame, text="\u25BC  Summary",
        font=ctk.CTkFont(size=11, weight="bold"), anchor="w",
        text_color=("#1a1a1a", "#e0e0e0"),
    ).grid(row=0, column=0, sticky="w", padx=8, pady=(4, 0))

    app.alignment_summary_text = ctk.CTkTextbox(
        summary_frame,
        font=ctk.CTkFont(family="Consolas", size=10),
        fg_color="#1A1A1A",
    )
    app.alignment_summary_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
    _make_readonly_selectable(app.alignment_summary_text)


# ── Viewer control callbacks (placeholders until viewer is wired) ────

def _on_viewer_color_change(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.set_color_mode(app._alignment_color_mode_var.get())


def _on_viewer_point_size_change(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.set_point_size(app._alignment_point_size_var.get())


def _on_viewer_camera_toggle(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.toggle_cameras(app._alignment_show_cameras_var.get())


def _on_viewer_reset(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.reset_upright()
        viewer.reset_camera()
        # Delete the saved upright file so it doesn't reload next time
        model_dir = getattr(app, "_alignment_loaded_model_dir", None)
        if model_dir:
            from pathlib import Path
            upright_file = Path(model_dir) / "upright_transform.json"
            if upright_file.exists():
                upright_file.unlink()
    roll_var = getattr(app, "_alignment_roll_var", None)
    if roll_var:
        roll_var.set(0.0)


def _on_set_upright(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.set_upright()
        # Auto-save to the model directory
        model_dir = getattr(app, "_alignment_loaded_model_dir", None)
        if model_dir:
            viewer.save_upright(model_dir)
            app.log("Upright orientation saved")
    roll_var = getattr(app, "_alignment_roll_var", None)
    if roll_var:
        roll_var.set(0.0)


def _on_roll_slider(app):
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.set_roll(app._alignment_roll_var.get())


def _on_viewer_screenshot(app):
    from tkinter import filedialog
    viewer = getattr(app, "_alignment_viewer", None)
    if not viewer:
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("All Files", "*.*")],
        title="Save Viewer Screenshot",
    )
    if path:
        viewer.screenshot(path)
        app.log(f"Screenshot saved: {path}")


def _on_viewer_load_model(app):
    """Browse for a sparse model folder and load it into the viewer."""
    from tkinter import filedialog
    from pathlib import Path

    folder = filedialog.askdirectory(
        title="Select Sparse Model Directory (contains cameras.txt/bin, images.txt/bin, points3D.txt/bin)",
    )
    if not folder:
        return

    model_dir = Path(folder)

    # Check for valid COLMAP model files (text or binary)
    has_text = all((model_dir / f"{s}.txt").exists() for s in ("cameras", "images", "points3D"))
    has_bin = all((model_dir / f"{s}.bin").exists() for s in ("cameras", "images", "points3D"))

    if not has_text and not has_bin:
        app.log(f"Not a valid sparse model: {folder}")
        app.log("Expected cameras.txt/bin, images.txt/bin, points3D.txt/bin")
        return

    app.log(f"Loading sparse model from: {folder}")
    _load_viewer_model(app, str(model_dir))


# ── Viewer lifecycle ─────────────────────────────────────────────────

def _ensure_viewer(app):
    """Create the PointCloudViewer if available and not yet created."""
    if getattr(app, "_alignment_viewer", None) is not None:
        return True

    try:
        from pointcloud_viewer import PointCloudViewer
    except ImportError:
        try:
            from reconstruction_gui.pointcloud_viewer import PointCloudViewer
        except ImportError:
            return False

    if not PointCloudViewer.available():
        return False

    viewer_frame = getattr(app, "_alignment_viewer_frame", None)
    if viewer_frame is None:
        return False

    # Hide placeholder
    placeholder = getattr(app, "_alignment_viewer_placeholder", None)
    if placeholder:
        placeholder.place_forget()

    app._alignment_viewer = PointCloudViewer(viewer_frame)
    return True


def _load_viewer_model(app, model_dir: str):
    """Parse a sparse model and load it into the viewer."""
    if not model_dir:
        return

    from pathlib import Path
    if not Path(model_dir).is_dir():
        app.log(f"Model directory not found: {model_dir}")
        return

    if not _ensure_viewer(app):
        app.log("3D viewer not available (pyvista/vtk not installed)")
        return

    try:
        model_path = Path(model_dir)

        # If binary files exist but text files don't, convert first
        has_text = all((model_path / f"{s}.txt").exists() for s in ("cameras", "images", "points3D"))
        has_bin = all((model_path / f"{s}.bin").exists() for s in ("cameras", "images", "points3D"))

        if has_bin and not has_text:
            # Need to convert — use ColmapRunner if available, else try colmap CLI
            runner = getattr(app, "_alignment_runner", None)
            if runner:
                runner._convert_model_to_text(model_path)
            else:
                # Try to find colmap binary for conversion
                colmap_binary = app.get_alignment_binary_path("colmap") if hasattr(app, "get_alignment_binary_path") else ""
                if colmap_binary:
                    import subprocess
                    cmd = [colmap_binary, "model_converter",
                           "--input_path", str(model_path),
                           "--output_path", str(model_path),
                           "--output_type", "TXT"]
                    app.log(f"Converting binary model to text: {model_path.name}")
                    subprocess.run(cmd, capture_output=True, timeout=30)
                else:
                    app.log("Cannot load binary model — no COLMAP binary available for conversion")
                    return

        # Parse text files
        try:
            from colmap_validation import parse_cameras_txt, parse_images_txt, parse_points3d_txt
        except ImportError:
            from reconstruction_gui.colmap_validation import parse_cameras_txt, parse_images_txt, parse_points3d_txt

        cameras = parse_cameras_txt(model_path / "cameras.txt")
        images = parse_images_txt(model_path / "images.txt")
        points3D = parse_points3d_txt(model_path / "points3D.txt")

        num_registered = len(images)
        num_points = len(points3D)
        mean_error = (sum(p.error for p in points3D.values()) / num_points) if num_points > 0 else 0.0

        model_data = {
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "stats": {
                "num_registered": num_registered,
                "num_points": num_points,
                "mean_reproj_error": mean_error,
            },
        }

        app._alignment_viewer.load_model(model_data)
        app._alignment_loaded_model_dir = str(model_path)
        app.log(f"Loaded model: {num_registered} images, {num_points} points, reproj error {mean_error:.3f}")

        # Auto-load saved upright transform if one exists
        if app._alignment_viewer.load_upright(str(model_path)):
            app.log("Restored saved upright orientation")

    except Exception as e:
        app.log(f"Failed to load model: {e}")
        logger.warning("Failed to load model into viewer: %s", e)


def alignment_viewer_pause(app):
    """Pause the viewer event pump (call on tab switch away from Align)."""
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.pause_pump()


def alignment_viewer_resume(app):
    """Resume the viewer event pump (call on tab switch back to Align)."""
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.resume_pump()


def alignment_viewer_destroy(app):
    """Destroy the viewer (call on app close)."""
    viewer = getattr(app, "_alignment_viewer", None)
    if viewer:
        viewer.destroy()
        app._alignment_viewer = None


def _detail_value_row(parent, label_text: str, font_size: int = 12):
    row = ctk.CTkFrame(parent, fg_color="transparent", height=18)
    row.pack(fill="x", pady=0, padx=6)
    ctk.CTkLabel(row, text=label_text, width=LABEL_FIELD_WIDTH, anchor="e", height=18,
                 font=ctk.CTkFont(size=font_size)).pack(side="left")
    value = ctk.CTkLabel(row, text="n/a", anchor="w", justify="left", height=18,
                         wraplength=600, font=ctk.CTkFont(size=font_size),
                         text_color="#94a3b8")
    value.pack(side="left", fill="x", expand=True, padx=(4, 0))
    return value


def send_extract_output_to_alignment(app):
    """Populate Align inputs from the Extract tab and switch tabs."""
    output_dir = getattr(app, "extract_output_entry", None)
    if output_dir is None:
        app.log("Align handoff is unavailable: extract output control not found.")
        return

    images_dir = output_dir.get().strip()
    if not images_dir:
        app.log("Set an Extract output folder before sending to Align.")
        return

    _set_entry_text(app.alignment_images_entry, images_dir)
    _maybe_autofill_alignment_workspace(app, force=True)
    app.tabs.set("Align")


def _set_entry_text(entry, value: str):
    entry.delete(0, "end")
    entry.insert(0, value)


def _capture_scroll(textbox):
    """Capture mousewheel events so nested textbox scrolls instead of parent."""
    inner = textbox._textbox if hasattr(textbox, "_textbox") else textbox

    def _on_mousewheel(event):
        inner.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    inner.bind("<MouseWheel>", _on_mousewheel)


def _make_readonly_selectable(textbox):
    """Allow selecting and copying text but block edits."""
    def _on_key(e):
        # Allow Ctrl+C, Ctrl+A, arrow keys, Home, End, Page Up/Down
        if e.state & 0x4 and e.keysym.lower() in ("c", "a"):
            return  # allow through
        if e.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                        "Prior", "Next", "Shift_L", "Shift_R",
                        "Control_L", "Control_R"):
            return  # allow navigation
        return "break"  # block everything else
    inner = textbox._textbox if hasattr(textbox, "_textbox") else textbox
    inner.bind("<Key>", _on_key)


def _set_textbox_text(textbox, text: str):
    textbox.delete("1.0", "end")
    textbox.insert("1.0", text)


def _append_textbox_text(textbox, text: str):
    textbox.insert("end", text)
    textbox.see("end")


def _browse_alignment_binary_override(app):
    app._browse_file_for(
        app.alignment_binary_entry,
        title="Select Alignment Binary",
        filetypes=[("Executables", "*.exe"), ("All Files", "*.*")],
    )
    _update_alignment_binary_hint(app)


def _toggle_alignment_binary_override(app):
    visible = not bool(getattr(app, "_alignment_binary_override_visible", False))
    app._alignment_binary_override_visible = visible
    frame = getattr(app, "alignment_binary_override_frame", None)
    button = getattr(app, "alignment_binary_change_button", None)
    if frame is not None:
        if visible:
            frame.pack(fill="x", pady=(2, 0))
        else:
            frame.pack_forget()
    if button is not None:
        button.configure(text="Hide" if visible else "Change...")
    _update_alignment_binary_hint(app)


def _on_engine_change(app, force: bool = False):
    engine_name = app.alignment_engine_var.get().strip().lower()

    # Auto-fill binary
    detected = app.get_alignment_binary_path(engine_name)
    current_binary = app.alignment_binary_entry.get().strip()
    last_auto_binary = getattr(app, "_alignment_last_auto_binary", "")
    if detected and (force or not current_binary or current_binary == last_auto_binary):
        _set_entry_text(app.alignment_binary_entry, detected)
        app._alignment_last_auto_binary = detected

    # Auto-fill camera model (dropdown for COLMAP, fixed for SphereSfM)
    if hasattr(app, "alignment_camera_model_var"):
        auto_camera_model = "PINHOLE" if engine_name != "spheresfm" else "PINHOLE"
        current_camera_model = app.alignment_camera_model_var.get().strip()
        last_auto_camera_model = getattr(app, "_alignment_last_auto_camera_model", "")
        if force or not current_camera_model or current_camera_model == last_auto_camera_model:
            app.alignment_camera_model_var.set(auto_camera_model)
            app._alignment_last_auto_camera_model = auto_camera_model

    # Swap engine panel frames
    if hasattr(app, "_alignment_colmap_frame"):
        if engine_name == "spheresfm":
            app._alignment_colmap_frame.pack_forget()
            app._alignment_spheresfm_frame.pack(fill="x")
        else:
            app._alignment_spheresfm_frame.pack_forget()
            app._alignment_colmap_frame.pack(fill="x")

    # Rig section: expanded + interactive for COLMAP, collapsed + greyed for SphereSfM
    if hasattr(app, "_alignment_rig_section"):
        _update_rig_section_state(app)

    # Lock BA refine checkboxes for SphereSfM
    if hasattr(app, "_alignment_ba_refine_focal_cb"):
        _update_ba_refine_lock(app)

    _update_alignment_binary_hint(app)


def _on_strategy_change(app):
    """Show/hide conditional fields based on matching strategy."""
    strategy = app.alignment_strategy_var.get().strip().lower()
    if strategy == "spatial":
        app._alignment_vocab_frame.pack_forget()
        app._alignment_spatial_frame.pack(fill="x")
    elif strategy == "vocab_tree":
        app._alignment_spatial_frame.pack_forget()
        app._alignment_vocab_frame.pack(fill="x")
    else:
        app._alignment_spatial_frame.pack_forget()
        app._alignment_vocab_frame.pack_forget()


def _set_widget_tree_state(widget, state: str):
    """Recursively enable/disable all interactive widgets in a tree."""
    for child in widget.winfo_children():
        try:
            child.configure(state=state)
        except Exception:
            pass
        _set_widget_tree_state(child, state)


def _is_rig_active(app) -> bool:
    """Derive rig mode from the rig section state: preset != None or custom file provided."""
    preset = getattr(app, "alignment_rig_preset_var", None)
    if preset is None:
        return False
    preset_name = preset.get()
    if preset_name not in ("None", ""):
        return True
    file_entry = getattr(app, "alignment_rig_file_entry", None)
    if file_entry and file_entry.get().strip():
        return True
    return False


def _update_rig_section_state(app):
    """Update rig section interactivity based on engine selection."""
    sec = getattr(app, "_alignment_rig_section", None)
    if sec is None:
        return

    engine_name = app.alignment_engine_var.get().strip().lower()

    if engine_name == "spheresfm":
        if sec._expanded:
            sec._toggle()
        _set_widget_tree_state(sec, "disabled")
    else:
        _set_widget_tree_state(sec, "normal")
        _update_rig_file_entry_state(app)


def _update_ba_refine_lock(app):
    """Lock/unlock BA refine checkboxes based on engine selection."""
    engine_name = app.alignment_engine_var.get().strip().lower()
    is_sphere = engine_name == "spheresfm"

    for var, cb in [
        (app.alignment_ba_refine_focal_var, getattr(app, "_alignment_ba_refine_focal_cb", None)),
        (app.alignment_ba_refine_principal_var, getattr(app, "_alignment_ba_refine_principal_cb", None)),
        (app.alignment_ba_refine_extra_var, getattr(app, "_alignment_ba_refine_extra_cb", None)),
    ]:
        if cb is None:
            continue
        if is_sphere:
            var.set(False)
            cb.configure(state="disabled")
        else:
            cb.configure(state="normal")


def _update_rig_file_entry_state(app):
    """Enable/disable the file entry based on preset selection."""
    preset_name = app.alignment_rig_preset_var.get()
    is_custom = preset_name == "Custom..."
    state = "normal" if is_custom else "disabled"
    app.alignment_rig_file_entry.configure(state=state)
    app._alignment_rig_file_browse_btn.configure(state=state)


def _on_rig_preset_change(app):
    """Called when the rig preset dropdown changes."""
    try:
        from reconstruction_gui.rig_presets import get_preset_by_display_name, format_rig_summary, load_rig_config
    except ImportError:
        from rig_presets import get_preset_by_display_name, format_rig_summary, load_rig_config

    _update_rig_file_entry_state(app)

    preset_name = app.alignment_rig_preset_var.get()
    preset = get_preset_by_display_name(preset_name)

    if preset:
        summary = format_rig_summary(preset.config, preset)
    elif preset_name == "Custom...":
        file_path = app.alignment_rig_file_entry.get().strip()
        if file_path and Path(file_path).is_file():
            try:
                config = load_rig_config(Path(file_path))
                summary = format_rig_summary(config)
            except Exception as exc:
                summary = f"(error reading file: {exc})"
        else:
            summary = "(select a rig config file)"
    else:
        summary = "(no rig preset selected)"

    _set_rig_summary(app, summary)


def _set_rig_summary(app, text: str):
    """Update the read-only rig summary textbox."""
    tb = app.alignment_rig_summary_text
    tb.configure(state="normal")
    tb.delete("1.0", "end")
    tb.insert("1.0", text)
    tb.configure(state="disabled")


def _browse_rig_config_file(app):
    """Browse for a custom rig_config.json file."""
    from tkinter import filedialog
    path = filedialog.askopenfilename(
        title="Select Rig Config JSON",
        filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
    )
    if path:
        _set_entry_text(app.alignment_rig_file_entry, path)
        _on_rig_preset_change(app)


def _update_alignment_binary_hint(app):
    label = getattr(app, "alignment_binary_hint_label", None)
    summary_label = getattr(app, "alignment_binary_summary_label", None)
    path_label = getattr(app, "alignment_binary_path_label", None)
    if label is None or summary_label is None or path_label is None:
        return

    engine_name = app.alignment_engine_var.get().strip().lower()
    current_binary = app.alignment_binary_entry.get().strip()
    info = app.get_alignment_binary_info(engine_name)
    preferred_binary = str(info.get("resolved_binary", "") or "")
    source = str(info.get("source", "") or "")
    status = str(info.get("status", "") or "")

    def _same_path(a: str, b: str) -> bool:
        if not a or not b:
            return False
        try:
            return str(Path(a).expanduser().resolve()).lower() == str(Path(b).expanduser().resolve()).lower()
        except Exception:
            return a.strip().lower() == b.strip().lower()

    if engine_name == "spheresfm":
        if status != "ready":
            summary = "SphereSfM binary not configured"
            shown_path = current_binary or preferred_binary
            text = "Recommended: official prebuilt SphereSfM binary. Configure it once, then leave it unchanged."
        elif source == "official-prebuilt-v1.2":
            if _same_path(current_binary, preferred_binary):
                summary = "Official prebuilt SphereSfM V1.2"
                shown_path = current_binary or preferred_binary
                text = "Using the preferred SphereSfM binary: official prebuilt V1.2."
            elif current_binary:
                summary = "Manual SphereSfM override"
                shown_path = current_binary
                text = "Preferred SphereSfM binary is the official prebuilt V1.2. Use Preferred to switch back."
            else:
                summary = "Official prebuilt SphereSfM V1.2"
                shown_path = preferred_binary
                text = "Preferred SphereSfM binary is the official prebuilt V1.2."
        else:
            summary = "SphereSfM binary configured"
            shown_path = current_binary or preferred_binary
            text = "SphereSfM usually only needs to be set once. The official prebuilt V1.2 binary is preferred."
    else:
        if status == "ready" and _same_path(current_binary, preferred_binary):
            summary = "Detected COLMAP binary"
            shown_path = current_binary or preferred_binary
            text = "Binary path is a one-time advanced setting. Use Preferred to restore the detected COLMAP binary."
        elif current_binary:
            summary = "Manual COLMAP override"
            shown_path = current_binary
        else:
            summary = "COLMAP binary"
            shown_path = preferred_binary
            text = "Binary path is a one-time advanced setting. Most users set it once and leave it alone."

    summary_label.configure(text=summary)
    path_label.configure(text=shown_path or "(no binary path selected)")
    label.configure(text=text)


def _maybe_autofill_alignment_workspace(app, force: bool = False):
    images_dir = app.alignment_images_entry.get().strip()
    if not images_dir:
        return

    new_value = str(Path(images_dir).parent / "alignment")
    current = app.alignment_workspace_entry.get().strip()
    last_auto = getattr(app, "_alignment_last_auto_workspace", "")
    if force or not current or current == last_auto:
        _set_entry_text(app.alignment_workspace_entry, new_value)
        app._alignment_last_auto_workspace = new_value


def _parse_int(value: str, label: str, minimum: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return parsed


def _parse_optional_float(value: str, label: str) -> Optional[float]:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"{label} must be a number") from exc


def _parse_optional_int(value: str, label: str, minimum: int = 0) -> Optional[int]:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except Exception as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return parsed


def _coerce_cli_value(raw_value: str):
    text = raw_value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        if "." not in text and "e" not in lowered:
            return int(text)
    except Exception:
        pass

    try:
        return float(text)
    except Exception:
        return text


def _parse_cli_args_text(text: str, label: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{label}: line {idx} must be key=value")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label}: line {idx} is missing a key")
        parsed[key] = _coerce_cli_value(raw_value)
    return parsed


def _detect_first_image_size(images_dir: str) -> Tuple[int, int]:
    from PIL import Image

    image_root = Path(images_dir)
    supported_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    for candidate in sorted(image_root.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in supported_exts:
            with Image.open(candidate) as img:
                return img.size
    raise ValueError(f"No readable images found in {image_root}")


def _normalized_mapper(mapper_value: str) -> str:
    mapper_key = (mapper_value or "").strip().lower()
    if mapper_key in {"global", "global_mapper"}:
        return "global_mapper"
    return "incremental"


def _snapshot_alignment_settings(app) -> Dict[str, object]:
    images_dir = app.alignment_images_entry.get().strip()
    masks_dir = app.alignment_masks_entry.get().strip()
    workspace_root = app.alignment_workspace_entry.get().strip()
    binary_path = app.alignment_binary_entry.get().strip()
    engine_name = app.alignment_engine_var.get().strip().lower()
    strategy = app.alignment_strategy_var.get().strip().lower()
    mapper = _normalized_mapper(app.alignment_mapper_var.get())
    camera_model = app.alignment_camera_model_var.get().strip().upper() if engine_name != "spheresfm" else "SPHERE"
    feature_type = "SIFT"
    camera_params = app.alignment_camera_params_entry.get().strip()
    pose_path = app.alignment_pose_path_entry.get().strip()
    camera_mask_path = app.alignment_camera_mask_path_entry.get().strip()
    snapshot_path = app.alignment_snapshot_path_entry.get().strip()
    vocab_tree_path = app.alignment_vocab_tree_entry.get().strip()

    if not images_dir:
        raise ValueError("Images directory is required")
    if not Path(images_dir).is_dir():
        raise ValueError(f"Images directory not found: {images_dir}")
    if masks_dir and not Path(masks_dir).is_dir():
        raise ValueError(f"Masks directory not found: {masks_dir}")
    if not workspace_root:
        raise ValueError("Workspace root is required")
    if not binary_path:
        raise ValueError("Alignment binary is required")
    if not camera_model:
        raise ValueError("Camera model is required")
    if strategy == "vocab_tree" and not vocab_tree_path:
        raise ValueError("Vocab tree path is required for vocab_tree matching")
    if pose_path and not Path(pose_path).exists():
        raise ValueError(f"Pose path not found: {pose_path}")
    if camera_mask_path and not Path(camera_mask_path).exists():
        raise ValueError(f"Camera mask path not found: {camera_mask_path}")
    if vocab_tree_path and not Path(vocab_tree_path).exists():
        raise ValueError(f"Vocab tree path not found: {vocab_tree_path}")

    max_features = _parse_int(app.alignment_max_features_entry.get(), "Max features", minimum=1)
    max_image_size = _parse_int(app.alignment_max_image_size_entry.get(), "Max image size", minimum=0)
    max_num_matches = _parse_int(
        app.alignment_max_num_matches_entry.get(),
        "Max matches",
        minimum=1,
    )
    min_num_inliers = _parse_int(
        app.alignment_min_num_inliers_entry.get(),
        "Mapper min",
        minimum=1,
    )
    guided_matching = bool(app.alignment_guided_var.get())
    single_camera = bool(app.alignment_single_camera_var.get())
    spatial_is_gps = bool(app.alignment_spatial_is_gps_var.get())
    spatial_max_distance = _parse_optional_float(
        app.alignment_spatial_max_distance_entry.get(),
        "Spatial max distance",
    )
    snapshot_images_freq = _parse_optional_int(
        app.alignment_snapshot_freq_entry.get(),
        "Snapshot frequency",
        minimum=0,
    )
    ba_refine_focal = bool(app.alignment_ba_refine_focal_var.get())
    ba_refine_principal = bool(app.alignment_ba_refine_principal_var.get())
    ba_refine_extra = bool(app.alignment_ba_refine_extra_var.get())

    extract_args_raw = app.alignment_extract_args_text.get("1.0", "end").strip()
    match_args_raw = app.alignment_match_args_text.get("1.0", "end").strip()
    reconstruct_args_raw = app.alignment_reconstruct_args_text.get("1.0", "end").strip()

    extract_cli_args = _parse_cli_args_text(extract_args_raw, "Extract args")
    match_cli_args = _parse_cli_args_text(match_args_raw, "Match args")
    reconstruct_cli_args = _parse_cli_args_text(reconstruct_args_raw, "Reconstruct args")

    base_mapper = "global" if mapper == "global_mapper" else "incremental"
    rig_mode = False
    rig_preset_name = "None"
    rig_file_path = ""

    if engine_name == "spheresfm":
        image_width, image_height = _detect_first_image_size(images_dir)
        profile = build_spheresfm_erp_profile(
            image_width=image_width,
            image_height=image_height,
            pose_path=pose_path or None,
            camera_mask_path=camera_mask_path or None,
            matching_strategy=strategy,
            spatial_is_gps=int(spatial_is_gps),
            spatial_max_distance=spatial_max_distance,
            guided_matching=guided_matching,
            vocab_tree_path=vocab_tree_path,
            extract_extra_args=extract_cli_args,
            match_extra_args=match_cli_args,
            reconstruct_extra_args=reconstruct_cli_args,
        )
    else:
        rig_mode = _is_rig_active(app)
        rig_preset_name = app.alignment_rig_preset_var.get() if rig_mode else "None"
        rig_file_path = app.alignment_rig_file_entry.get().strip() if rig_mode else ""

        if rig_mode and rig_preset_name not in ("None", ""):
            try:
                from reconstruction_gui.alignment_profiles import build_colmap_rig_profile
            except ImportError:
                from alignment_profiles import build_colmap_rig_profile
            profile = build_colmap_rig_profile(
                camera_model=camera_model,
                matching_strategy=strategy,
                guided_matching=guided_matching,
                mapper=base_mapper,
                extract_extra_args=extract_cli_args,
                match_extra_args=match_cli_args,
                reconstruct_extra_args=reconstruct_cli_args,
            )
        else:
            profile = build_colmap_pinhole_profile(
                camera_model=camera_model,
                single_camera=single_camera,
                camera_params=camera_params or None,
                matching_strategy=strategy,
                guided_matching=guided_matching,
                mapper=base_mapper,
                extract_extra_args=extract_cli_args,
                match_extra_args=match_cli_args,
                reconstruct_extra_args=reconstruct_cli_args,
            )

    # Resolve rig config path
    rig_config_path = ""
    if engine_name == "colmap" and rig_mode:
        if rig_preset_name == "Custom..." and rig_file_path:
            if not Path(rig_file_path).is_file():
                raise ValueError(f"Rig config file not found: {rig_file_path}")
            rig_config_path = rig_file_path
        elif rig_preset_name not in ("None", "Custom...", ""):
            try:
                from reconstruction_gui.rig_presets import get_preset_by_display_name, write_rig_config
            except ImportError:
                from rig_presets import get_preset_by_display_name, write_rig_config
            preset = get_preset_by_display_name(rig_preset_name)
            if preset is None:
                raise ValueError(f"Unknown rig preset: {rig_preset_name}")
            import tempfile
            preset_file = Path(tempfile.gettempdir()) / "rz_rig_config.json"
            write_rig_config(preset, preset_file)
            rig_config_path = str(preset_file)

    extract_args = dict(profile.extract_extra_args)
    match_args = dict(profile.match_extra_args)
    reconstruct_args = dict(profile.reconstruct_extra_args)

    extract_args["ImageReader.single_camera"] = single_camera
    if camera_params:
        extract_args["ImageReader.camera_params"] = camera_params
    if pose_path:
        extract_args["ImageReader.pose_path"] = pose_path
    if camera_mask_path:
        extract_args["ImageReader.camera_mask_path"] = camera_mask_path

    if strategy == "spatial":
        match_args["SpatialMatching.is_gps"] = int(spatial_is_gps)
        if spatial_max_distance is not None:
            match_args["SpatialMatching.max_distance"] = spatial_max_distance

    reconstruct_args["Mapper.ba_refine_focal_length"] = ba_refine_focal
    reconstruct_args["Mapper.ba_refine_principal_point"] = ba_refine_principal
    reconstruct_args["Mapper.ba_refine_extra_params"] = ba_refine_extra
    if snapshot_path:
        reconstruct_args["Mapper.snapshot_path"] = snapshot_path
    if snapshot_images_freq:
        reconstruct_args["Mapper.snapshot_frames_freq"] = snapshot_images_freq

    profile = replace(
        profile,
        camera_model=camera_model,
        feature_type=feature_type,
        mapper=base_mapper,
        vocab_tree_path=vocab_tree_path,
        extract_extra_args=extract_args,
        match_extra_args=match_args,
        reconstruct_extra_args=reconstruct_args,
    )

    prefs = app._prefs
    prefs["alignment_images_dir"] = images_dir
    prefs["alignment_masks_dir"] = masks_dir
    prefs["alignment_workspace_root"] = workspace_root
    prefs["alignment_engine"] = engine_name
    prefs["alignment_strategy"] = strategy
    prefs["alignment_mapper"] = base_mapper
    prefs["alignment_binary_path"] = binary_path
    prefs["alignment_camera_model"] = camera_model
    prefs["alignment_feature_type"] = feature_type
    prefs["alignment_single_camera"] = single_camera
    prefs["alignment_camera_params"] = camera_params
    prefs["alignment_pose_path"] = pose_path
    prefs["alignment_camera_mask_path"] = camera_mask_path
    prefs["alignment_snapshot_path"] = snapshot_path
    prefs["alignment_snapshot_images_freq"] = "" if snapshot_images_freq is None else snapshot_images_freq
    prefs["alignment_guided_matching"] = guided_matching
    prefs["alignment_vocab_tree_path"] = vocab_tree_path
    prefs["alignment_max_features"] = max_features
    prefs["alignment_max_image_size"] = max_image_size
    prefs["alignment_max_num_matches"] = max_num_matches
    prefs["alignment_min_num_inliers"] = min_num_inliers
    prefs["alignment_spatial_is_gps"] = spatial_is_gps
    prefs["alignment_spatial_max_distance"] = (
        "" if spatial_max_distance is None else spatial_max_distance
    )
    prefs["alignment_ba_refine_focal_length"] = ba_refine_focal
    prefs["alignment_ba_refine_principal_point"] = ba_refine_principal
    prefs["alignment_ba_refine_extra_params"] = ba_refine_extra
    prefs["alignment_extract_args"] = extract_args_raw
    prefs["alignment_match_args"] = match_args_raw
    prefs["alignment_reconstruct_args"] = reconstruct_args_raw
    prefs["alignment_rig_mode"] = bool(rig_mode) if engine_name == "colmap" else False
    prefs["alignment_rig_preset"] = rig_preset_name if engine_name == "colmap" else "None"
    prefs["alignment_rig_file"] = rig_file_path if engine_name == "colmap" else ""
    prefs[app._alignment_binary_pref_key(engine_name)] = binary_path
    app._save_prefs()

    signature_payload = {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "workspace_root": workspace_root,
        "engine_name": engine_name,
        "binary_path": binary_path,
        "strategy": strategy,
        "mapper": base_mapper,
        "camera_model": camera_model,
        "feature_type": feature_type,
        "single_camera": single_camera,
        "camera_params": camera_params,
        "pose_path": pose_path,
        "camera_mask_path": camera_mask_path,
        "snapshot_path": snapshot_path,
        "snapshot_images_freq": snapshot_images_freq,
        "guided_matching": guided_matching,
        "vocab_tree_path": vocab_tree_path,
        "max_features": max_features,
        "max_image_size": max_image_size,
        "max_num_matches": max_num_matches,
        "min_num_inliers": min_num_inliers,
        "spatial_is_gps": spatial_is_gps,
        "spatial_max_distance": spatial_max_distance,
        "ba_refine_focal_length": ba_refine_focal,
        "ba_refine_principal_point": ba_refine_principal,
        "ba_refine_extra_params": ba_refine_extra,
        "extract_extra_args": profile.extract_extra_args,
        "match_extra_args": profile.match_extra_args,
        "reconstruct_extra_args": profile.reconstruct_extra_args,
        "rig_config_path": rig_config_path,
    }

    return {
        "images_dir": images_dir,
        "masks_dir": masks_dir or "",
        "workspace_root": workspace_root,
        "engine_name": engine_name,
        "binary_path": binary_path,
        "profile": profile,
        "max_features": max_features,
        "max_image_size": max_image_size,
        "max_num_matches": max_num_matches,
        "min_num_inliers": min_num_inliers,
        "rig_config_path": rig_config_path,
        "signature": json.dumps(signature_payload, sort_keys=True, default=str),
    }


def _start_alignment(app, mode: str):
    if getattr(app, "is_running", False) or getattr(app, "mq_processing", False) or getattr(app, "extract_queue_processing", False):
        app.log("Another task is already running. Finish or cancel it before starting Alignment.")
        return

    try:
        snapshot = _snapshot_alignment_settings(app)
    except Exception as exc:
        app.log(f"Alignment settings error: {exc}")
        return

    same_signature = (
        snapshot["signature"] == getattr(app, "_alignment_snapshot_signature", "")
        and getattr(app, "_alignment_runner", None) is not None
        and getattr(app._alignment_runner, "current_run_dir", None) is not None
    )

    if mode == "full":
        stages_to_run = [stage for stage, _label in ALIGNMENT_STAGES]
        start_new_run = True
    else:
        start_new_run = not same_signature
        if start_new_run:
            _reset_alignment_session(app, clear_summary=True)
            app._alignment_snapshot_signature = snapshot["signature"]
        next_stage = _next_pending_stage(app)
        if next_stage is None:
            app.log("All alignment stages are already complete for the current session.")
            return
        stages_to_run = [next_stage]

    if mode == "full":
        _reset_alignment_session(app, clear_summary=True)
        app._alignment_snapshot_signature = snapshot["signature"]
    elif not same_signature:
        app._alignment_snapshot_signature = snapshot["signature"]

    if getattr(app, "_alignment_cancel_event", None) is None:
        app._alignment_cancel_event = threading.Event()
    app._alignment_cancel_event.clear()

    _set_alignment_busy(app, True)
    app._alignment_thread = threading.Thread(
        target=_alignment_worker,
        args=(app, snapshot, stages_to_run, start_new_run),
        daemon=True,
    )
    app._alignment_thread.start()


def _next_pending_stage(app) -> Optional[str]:
    stage_states = getattr(app, "_alignment_stage_states", {})
    for stage_key, _label in ALIGNMENT_STAGES:
        if stage_states.get(stage_key) != "complete":
            return stage_key
    return None


def _alignment_worker(app, snapshot: Dict[str, object], stages_to_run: List[str], start_new_run: bool):
    try:
        runner = getattr(app, "_alignment_runner", None)
        if start_new_run or runner is None:
            runner = app.create_alignment_runner(
                engine_name=str(snapshot["engine_name"]),
                workspace_root=str(snapshot["workspace_root"]),
                camera_model=str(snapshot["profile"].camera_model),
                binary_path=str(snapshot["binary_path"]),
            )
            runner.start_run(
                images_dir=str(snapshot["images_dir"]),
                masks_dir=str(snapshot["masks_dir"]) or None,
            )
            app._alignment_runner = runner
            run_dir = str(runner.current_run_dir) if runner.current_run_dir else ""
            if run_dir:
                app.after(
                    0,
                    lambda path=run_dir: app.alignment_run_dir_label.configure(
                        text=path
                    ),
                )

        profile = snapshot["profile"]
        progress = lambda message: _alignment_log(app, message)

        for stage_key in stages_to_run:
            if app._alignment_cancel_event.is_set():
                app.after(0, lambda s=stage_key: _set_stage_state(app, s, "cancelled"))
                break

            app.after(0, lambda s=stage_key: _set_stage_state(app, s, "running"))

            if stage_key == "feature_extraction":
                result = runner.extract_features(
                    images_dir=str(snapshot["images_dir"]),
                    masks_dir=str(snapshot["masks_dir"]) or None,
                    max_features=int(snapshot["max_features"]),
                    max_image_size=int(snapshot["max_image_size"]),
                    feature_type=str(profile.feature_type),
                    progress_callback=progress,
                    cancel_event=app._alignment_cancel_event,
                    extra_args=dict(profile.extract_extra_args),
                )
                # Apply rig config after extraction, before matching
                rig_config_path = str(snapshot.get("rig_config_path", ""))
                if result.success and rig_config_path:
                    progress("Applying rig configuration...")
                    rig_result = runner.apply_rig_config(
                        rig_config_path=rig_config_path,
                        progress_callback=progress,
                        cancel_event=app._alignment_cancel_event,
                    )
                    app.after(0, lambda r=rig_result: _store_rig_result(app, r))
                    if not rig_result.success:
                        result = rig_result
            elif stage_key == "matching":
                result = runner.match_features(
                    strategy=str(profile.matching_strategy),
                    guided=bool(profile.guided_matching),
                    max_num_matches=int(snapshot["max_num_matches"]),
                    vocab_tree_path=str(profile.vocab_tree_path or "") or None,
                    progress_callback=progress,
                    cancel_event=app._alignment_cancel_event,
                    extra_args=dict(profile.match_extra_args),
                )
            elif stage_key == "reconstruction":
                result = runner.reconstruct(
                    mapper=str(profile.mapper),
                    min_num_inliers=int(snapshot["min_num_inliers"]),
                    images_dir=str(snapshot["images_dir"]),
                    progress_callback=progress,
                    cancel_event=app._alignment_cancel_event,
                    extra_args=dict(profile.reconstruct_extra_args),
                )
            else:
                raise RuntimeError(f"Unknown alignment stage: {stage_key}")

            app.after(0, lambda s=stage_key, r=result: _apply_stage_result(app, s, r))

            if not result.success:
                break
    except Exception as exc:
        _alignment_log(app, f"Alignment failed: {exc}")
    finally:
        app.after(0, lambda: _set_alignment_busy(app, False))


def _set_alignment_busy(app, busy: bool):
    app.is_running = bool(busy)
    if hasattr(app, "alignment_run_btn"):
        app.alignment_run_btn.configure(state="disabled" if busy else "normal")
    if hasattr(app, "alignment_next_btn"):
        app.alignment_next_btn.configure(state="disabled" if busy else "normal")
    if hasattr(app, "alignment_cancel_btn"):
        app.alignment_cancel_btn.configure(state="normal" if busy else "disabled")
    if hasattr(app, "alignment_reset_btn"):
        app.alignment_reset_btn.configure(state="disabled" if busy else "normal")
    if hasattr(app, "alignment_send_to_coverage_btn"):
        has_model = bool(getattr(app, "_alignment_last_selected_model_dir", ""))
        app.alignment_send_to_coverage_btn.configure(
            state="disabled" if busy or not has_model else "normal"
        )


def _cancel_alignment(app):
    if getattr(app, "_alignment_cancel_event", None) is None:
        return
    if getattr(app, "is_running", False):
        app._alignment_cancel_event.set()
        _alignment_log(app, "Alignment cancellation requested.")


def _set_stage_state(app, stage_key: str, status: str):
    app._alignment_stage_states[stage_key] = status
    label_text = dict(ALIGNMENT_STAGES).get(stage_key, stage_key)
    label = getattr(app, "_alignment_stage_labels", {}).get(stage_key)
    if label is not None:
        label.configure(
            text=f"{label_text}: {status}",
            text_color=STAGE_COLORS.get(status, "#E5E7EB"),
        )


def _count_alignment_images(app) -> Optional[int]:
    images_dir = app.alignment_images_entry.get().strip() if hasattr(app, "alignment_images_entry") else ""
    if not images_dir or not Path(images_dir).is_dir():
        return None
    supported_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    return sum(1 for path in Path(images_dir).iterdir() if path.is_file() and path.suffix.lower() in supported_exts)


def _format_mask_diagnostic(stage_stats: Dict[str, object]) -> str:
    masks_dir = str(stage_stats.get("masks_dir", "") or "").strip()
    if not masks_dir:
        return "No masks requested"

    mode = str(stage_stats.get("mask_resolution_mode", "") or "").strip()
    direct = int(stage_stats.get("mask_matches_direct", 0) or 0)
    stem = int(stage_stats.get("mask_matches_stem", 0) or 0)
    missing = int(stage_stats.get("mask_missing", 0) or 0)

    if mode == "native":
        summary = f"Native names: {direct} matched"
    elif mode:
        summary = f"Remapped masks: {stem} stem-name matches"
        if direct:
            summary += f", {direct} native-name matches"
    else:
        summary = "Mask resolution pending"

    if missing:
        summary += f", {missing} missing"
    return summary


def _format_snapshot_diagnostic(stage_stats: Dict[str, object]) -> str:
    freq = int(stage_stats.get("snapshot_images_freq", 0) or 0)
    path = str(stage_stats.get("snapshot_path", "") or "").strip()
    if freq <= 0:
        return "Off"
    if path:
        return f"Every {freq} images -> {path}"
    return f"Every {freq} images"


def _assess_alignment_quality(app, selected_stats: Dict[str, object]) -> Dict[str, object]:
    total_images = _count_alignment_images(app)
    num_registered = int(selected_stats.get("num_registered", 0) or 0)
    num_points = int(selected_stats.get("num_points", 0) or 0)
    mean_reproj = selected_stats.get("mean_reproj_error")
    try:
        mean_reproj_value = float(mean_reproj)
    except Exception:
        mean_reproj_value = None

    findings: List[str] = []
    severity = 0

    if total_images:
        ratio = num_registered / total_images if total_images else 0.0
        if ratio < 0.35:
            findings.append(f"Very low registration rate: {num_registered}/{total_images} images ({ratio:.0%}).")
            severity = max(severity, 3)
        elif ratio < 0.60:
            findings.append(f"Low registration rate: {num_registered}/{total_images} images ({ratio:.0%}).")
            severity = max(severity, 2)
        elif ratio < 0.80:
            findings.append(f"Moderate registration rate: {num_registered}/{total_images} images ({ratio:.0%}).")
            severity = max(severity, 1)
    else:
        ratio = None

    if num_points < 100:
        findings.append(f"Very sparse model: only {num_points} points.")
        severity = max(severity, 3)
    elif num_points < 500:
        findings.append(f"Sparse model: only {num_points} points.")
        severity = max(severity, 2)
    elif num_points < 2000:
        findings.append(f"Limited sparse detail: {num_points} points.")
        severity = max(severity, 1)

    if mean_reproj_value is not None and mean_reproj_value > 2.0:
        findings.append(f"High mean reprojection error: {mean_reproj_value:.3f}px.")
        severity = max(severity, 2)

    if severity >= 3:
        summary = "Poor"
        ui_state = "warning"
    elif severity >= 1:
        summary = "Weak"
        ui_state = "warning"
    else:
        summary = "Good"
        ui_state = "complete"

    return {
        "summary": summary,
        "ui_state": ui_state,
        "notes": findings,
        "total_images": total_images,
        "registered_ratio": ratio,
    }


def _windows_access_violation(returncode: Optional[int]) -> bool:
    if returncode is None:
        return False
    return (int(returncode) & 0xFFFFFFFF) == 0xC0000005


def _reconstruction_failure_quality_info(app, result) -> Dict[str, object]:
    notes: List[str] = []

    if _windows_access_violation(result.returncode):
        notes.append(
            "Mapper crashed with Windows access violation (0xC0000005). This is a native SphereSfM/COLMAP crash, not a normal reconstruction failure."
        )
    elif result.error:
        notes.append(result.error)
    else:
        notes.append("Reconstruction failed before a usable sparse model was selected.")

    if result.stats.get("partial_model_available"):
        if result.stats.get("recovered_from_snapshot"):
            notes.append("A recoverable snapshot model was selected after the crash.")
        else:
            notes.append("A partial sparse model was recovered from disk for inspection.")
        summary = "Crash + Partial"
    else:
        notes.append("No recoverable sparse model was written before the crash/failure.")
        summary = "Crash"

    progress_summary = str(result.stats.get("progress_summary", "") or "").strip()
    if progress_summary:
        notes.append(progress_summary)

    return {
        "summary": summary,
        "ui_state": "failed",
        "notes": notes,
    }


def _reset_alignment_session(app, clear_summary: bool):
    app._alignment_runner = None
    app._alignment_thread = None
    app._alignment_snapshot_signature = ""
    app._alignment_stage_results = {}
    app._alignment_rig_result = None
    app._alignment_stage_states = {stage: "pending" for stage, _ in ALIGNMENT_STAGES}
    app._alignment_last_selected_model_dir = ""
    app._alignment_quality_info = {}
    app._alignment_diag_info = {}

    for stage_key, _label in ALIGNMENT_STAGES:
        _set_stage_state(app, stage_key, "pending")

    if hasattr(app, "alignment_run_dir_label"):
        app.alignment_run_dir_label.configure(text="Not started")
    if hasattr(app, "alignment_mask_status_label"):
        app.alignment_mask_status_label.configure(text="n/a")
    if hasattr(app, "alignment_snapshot_status_label"):
        app.alignment_snapshot_status_label.configure(text="n/a")
    if hasattr(app, "alignment_last_progress_label"):
        app.alignment_last_progress_label.configure(text="n/a")
    if hasattr(app, "alignment_selected_model_path_label"):
        app.alignment_selected_model_path_label.configure(text="n/a")
    if hasattr(app, "alignment_selected_model_registered_label"):
        app.alignment_selected_model_registered_label.configure(text="\u2014")
    if hasattr(app, "alignment_selected_model_points_label"):
        app.alignment_selected_model_points_label.configure(text="\u2014")
    if hasattr(app, "alignment_selected_model_reproj_label"):
        app.alignment_selected_model_reproj_label.configure(text="\u2014")
    if hasattr(app, "alignment_selected_model_quality_label"):
        app.alignment_selected_model_quality_label.configure(text="\u2014")
    if hasattr(app, "alignment_selected_model_notes_label"):
        app.alignment_selected_model_notes_label.configure(text="n/a")

    if clear_summary and hasattr(app, "alignment_summary_text"):
        _set_textbox_text(app.alignment_summary_text, "Ready.\n")
    if clear_summary and hasattr(app, "alignment_log_text"):
        _set_textbox_text(
            app.alignment_log_text,
            "Ready.\nUse Align to run all stages or Next Step to advance one stage at a time.\n",
        )

    _set_alignment_busy(app, False)
    _refresh_alignment_summary(app)


def _refresh_alignment_summary(app):
    if not hasattr(app, "alignment_summary_text"):
        return

    lines: List[str] = []
    images_dir = app.alignment_images_entry.get().strip() if hasattr(app, "alignment_images_entry") else ""
    workspace_root = app.alignment_workspace_entry.get().strip() if hasattr(app, "alignment_workspace_entry") else ""
    if images_dir:
        lines.append(f"Images: {images_dir}")
    if workspace_root:
        lines.append(f"Workspace: {workspace_root}")
    if getattr(app, "_alignment_runner", None) and getattr(app._alignment_runner, "current_run_dir", None):
        lines.append(f"Run Dir: {app._alignment_runner.current_run_dir}")

    diag_info = getattr(app, "_alignment_diag_info", {})
    mask_status = str(diag_info.get("mask_status", "") or "").strip()
    snapshot_status = str(diag_info.get("snapshot_status", "") or "").strip()
    progress_status = str(diag_info.get("progress_status", "") or "").strip()
    if mask_status:
        lines.append(f"Masks: {mask_status}")
    if snapshot_status:
        lines.append(f"Snapshots: {snapshot_status}")
    if progress_status:
        lines.append(f"Progress: {progress_status}")

    if lines:
        lines.append("")

    for stage_key, label in ALIGNMENT_STAGES:
        status = app._alignment_stage_states.get(stage_key, "pending")
        result = app._alignment_stage_results.get(stage_key)
        if result is None:
            lines.append(f"{label}: {status}")
        else:
            suffix = f" ({result.duration_s:.2f}s)" if result.duration_s else ""
            lines.append(f"{label}: {status}{suffix}")
            if result.error:
                lines.append(f"  Error: {result.error}")
            if result.stats:
                for key, value in result.stats.items():
                    if key == "selected_model_stats":
                        continue
                    lines.append(f"  {key}: {value}")

        # Insert rig configurator result after feature extraction
        if stage_key == "feature_extraction":
            rig_result = getattr(app, "_alignment_rig_result", None)
            if rig_result is not None:
                rig_suffix = f" ({rig_result.duration_s:.2f}s)" if rig_result.duration_s else ""
                lines.append(f"Rig Configuration: {rig_result.status}{rig_suffix}")
                if rig_result.error:
                    lines.append(f"  Error: {rig_result.error}")
                if rig_result.stats:
                    for key, value in rig_result.stats.items():
                        lines.append(f"  {key}: {value}")

    selected_model_dir = getattr(app, "_alignment_last_selected_model_dir", "")
    quality_info = getattr(app, "_alignment_quality_info", {})
    if selected_model_dir:
        lines.extend(
            [
                "",
                f"Selected model: {selected_model_dir}",
                f"Registered: {app.alignment_selected_model_registered_label.cget('text')}",
                f"Points: {app.alignment_selected_model_points_label.cget('text')}",
                f"Mean reproj: {app.alignment_selected_model_reproj_label.cget('text')}",
            ]
        )
        if quality_info:
            lines.append(f"Quality: {quality_info.get('summary', 'n/a')}")
            for note in quality_info.get("notes", []):
                lines.append(f"  Note: {note}")
    elif quality_info:
        lines.extend(["", f"Quality: {quality_info.get('summary', 'n/a')}"])
        for note in quality_info.get("notes", []):
            lines.append(f"  Note: {note}")

    summary_text = "\n".join(lines).strip()
    _set_textbox_text(app.alignment_summary_text, summary_text + ("\n" if summary_text else "Ready.\n"))


def _alignment_log(app, message: str):
    line = str(message).rstrip()
    if not line:
        return
    app.log(line)
    if hasattr(app, "alignment_log_text"):
        app.after(0, lambda text=f"{line}\n": _append_textbox_text(app.alignment_log_text, text))


def _send_alignment_to_coverage(app, switch_tab: bool = False):
    selected_model_dir = getattr(app, "_alignment_last_selected_model_dir", "") or app._prefs.get(
        "alignment_last_selected_model_dir",
        "",
    )
    if not selected_model_dir:
        app.log("No alignment model is available to send to Coverage yet.")
        return
    if not Path(selected_model_dir).exists():
        app.log(f"Selected alignment model no longer exists: {selected_model_dir}")
        return

    try:
        from reconstruction_gui.tabs.gaps_tab import _on_source_change
    except ImportError:
        from tabs.gaps_tab import _on_source_change

    app.gaps_source_var.set("colmap")
    _on_source_change(app)
    _set_entry_text(app.gaps_source_entry, selected_model_dir)

    images_dir = app.alignment_images_entry.get().strip()
    if images_dir and hasattr(app, "gaps_images_entry") and not app.gaps_images_entry.get().strip():
        _set_entry_text(app.gaps_images_entry, images_dir)

    if switch_tab:
        app.tabs.set("Coverage")

    app.log(f"Coverage source updated to alignment model: {selected_model_dir}")


def _store_rig_result(app, result):
    """Store the rig configurator result for the summary display."""
    app._alignment_rig_result = result
    _refresh_alignment_summary(app)


def _apply_stage_result(app, stage_key: str, result):
    app._alignment_stage_results[stage_key] = result

    if result.status == "success":
        _set_stage_state(app, stage_key, "complete")
    elif result.status == "cancelled":
        _set_stage_state(app, stage_key, "cancelled")
    else:
        _set_stage_state(app, stage_key, "failed")

    if result.run_dir:
        app.alignment_run_dir_label.configure(text=result.run_dir)

    if result.error:
        _alignment_log(app, f"{dict(ALIGNMENT_STAGES).get(stage_key, stage_key)} error: {result.error}")

    if stage_key == "feature_extraction":
        mask_status = _format_mask_diagnostic(result.stats)
        app._alignment_diag_info["mask_status"] = mask_status
        if hasattr(app, "alignment_mask_status_label"):
            app.alignment_mask_status_label.configure(text=mask_status)
        _alignment_log(app, f"Mask status: {mask_status}")

    if stage_key == "reconstruction":
        snapshot_status = _format_snapshot_diagnostic(result.stats)
        progress_status = str(result.stats.get("progress_summary", "") or "").strip()
        app._alignment_diag_info["snapshot_status"] = snapshot_status
        app._alignment_diag_info["progress_status"] = progress_status or "n/a"
        if hasattr(app, "alignment_snapshot_status_label"):
            app.alignment_snapshot_status_label.configure(text=snapshot_status)
        if hasattr(app, "alignment_last_progress_label"):
            app.alignment_last_progress_label.configure(text=progress_status or "n/a")

        selected_model_dir = result.selected_model_dir or result.stats.get("selected_model_dir", "")
        selected_stats = result.stats.get("selected_model_stats", {})
        if selected_model_dir:
            app._alignment_last_selected_model_dir = selected_model_dir
            app._prefs["alignment_last_selected_model_dir"] = selected_model_dir
            app._save_prefs()
            app.alignment_selected_model_path_label.configure(text=selected_model_dir)
            app.alignment_selected_model_registered_label.configure(
                text=str(selected_stats.get("num_registered", "n/a"))
            )
            app.alignment_selected_model_points_label.configure(
                text=str(selected_stats.get("num_points", "n/a"))
            )
            reproj = selected_stats.get("mean_reproj_error", "n/a")
            if isinstance(reproj, float):
                reproj = f"{reproj:.4f}"
            app.alignment_selected_model_reproj_label.configure(text=str(reproj))
            quality_info = _assess_alignment_quality(app, selected_stats)
            if not result.success:
                failure_quality = _reconstruction_failure_quality_info(app, result)
                quality_info = {
                    "summary": failure_quality["summary"],
                    "ui_state": failure_quality["ui_state"],
                    "notes": failure_quality["notes"] + quality_info.get("notes", []),
                }
            app._alignment_quality_info = quality_info
            app.alignment_selected_model_quality_label.configure(
                text=str(quality_info.get("summary", "n/a"))
            )
            notes = quality_info.get("notes", [])
            app.alignment_selected_model_notes_label.configure(
                text="; ".join(notes) if notes else "No obvious quality warnings."
            )
            if result.success and quality_info.get("ui_state") == "warning":
                _set_stage_state(app, stage_key, "warning")
                _alignment_log(
                    app,
                    "Reconstruction completed, but the result quality looks weak: "
                    + "; ".join(notes),
                )
            elif result.success:
                _alignment_log(app, "Reconstruction completed with no obvious quality warnings.")
            if progress_status:
                _alignment_log(app, f"Reconstruction progress summary: {progress_status}")

            app.alignment_send_to_coverage_btn.configure(state="normal")
            _load_viewer_model(app, selected_model_dir)
            if result.success:
                _send_alignment_to_coverage(app, switch_tab=False)
            elif result.stats.get("partial_model_available"):
                if result.stats.get("recovered_from_snapshot"):
                    _alignment_log(
                        app,
                        "Mapper failed, but the best snapshot model was recovered for inspection.",
                    )
                else:
                    _alignment_log(
                        app,
                        "Mapper failed, but a partial sparse model was recovered for inspection.",
                    )
        else:
            quality_info = _reconstruction_failure_quality_info(app, result)
            app._alignment_quality_info = quality_info
            app.alignment_selected_model_path_label.configure(text="No recoverable model written")
            app.alignment_selected_model_registered_label.configure(text="\u2014")
            app.alignment_selected_model_points_label.configure(text="\u2014")
            app.alignment_selected_model_reproj_label.configure(text="\u2014")
            app.alignment_selected_model_quality_label.configure(
                text=str(quality_info.get("summary", "n/a"))
            )
            app.alignment_selected_model_notes_label.configure(
                text="; ".join(quality_info.get("notes", []))
            )
            _alignment_log(app, "Reconstruction failed before a recoverable sparse model was written.")
            if progress_status:
                _alignment_log(app, f"Last mapper progress before failure: {progress_status}")

    _refresh_alignment_summary(app)

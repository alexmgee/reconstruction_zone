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
from typing import Any, Optional, List, Dict
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

from _version import __version__
from widgets import (
    Section as _Section, CollapsibleSection as _CollapsibleSection, slider_row, Tooltip,
    COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
    COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
    COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
    COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
    COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    FONT_TEXT_SUBTITLE, FONT_TEXT_MONO_VALUE, FONT_TEXT_CONSOLE, FONT_TEXT_STATUS,
    LABEL_FIELD_WIDTH, BROWSE_BUTTON_WIDTH,
)
from app_infra import AppInfrastructure
from tabs.alignment_tab import build_alignment_tab
from tabs.source_tab import build_source_tab
from tabs.gaps_tab import build_gaps_tab
from tabs.projects_tab import build_projects_tab


# ──────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────

class ReconstructionZone(AppInfrastructure, ctk.CTk):
    """Unified photogrammetry prep: prepare → mask → review → gaps."""

    _PREFS_FILE = _this_dir / ".studio_prefs.json"
    _SPHERESFM_PREBUILT_CANDIDATES = (
        _project_root / ".tmp" / "SphereSfM-2025-8-18" / "SphereSfM-2024-12-14" / "colmap.exe",
    )
    _SPHERESFM_LEGACY_CUSTOM_DEFAULT = Path(r"D:\Tools\SphereSfM\bin\colmap.exe")
    _ALIGNMENT_BINARY_PREF_KEYS = {
        "colmap": "alignment_colmap_binary",
        "spheresfm": "alignment_spheresfm_binary",
    }
    _ALIGNMENT_BINARY_ENV_KEYS = {
        "colmap": "COLMAP_BINARY",
        "spheresfm": "SPHERESFM_BINARY",
    }

    def __init__(self):
        super().__init__()
        self.title(f"Reconstruction Zone v{__version__}")
        self.geometry("2200x1200")
        self.minsize(1200, 800)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Window icon (also controls taskbar icon while running)
        _ico = Path(__file__).resolve().parent.parent / "reconstruction-zone.ico"
        if _ico.exists():
            self.after(200, lambda: self.iconbitmap(str(_ico)))

        # State
        self.log_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.cancel_flag = threading.Event()
        self._prefs = self._load_prefs()
        self._activity_store = None  # Lazy — set after project store in build_projects_tab
        self._preview_mode = "process"  # "process" or "review"
        self._alignment_binary_paths: Dict[str, str] = {}
        self._alignment_binary_status: Dict[str, Dict[str, object]] = {}

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

    def record_activity(self, operation: str, input_path: str, output_path: str,
                        status: str = "completed",
                        details: Optional[Dict[str, Any]] = None):
        """Record a completed operation for the Recent Activity view.

        Called by tab workers after an operation finishes. Thread-safe
        because ActivityStore.save() is atomic (write + replace).
        """
        if self._activity_store is None:
            return
        try:
            self._activity_store.record(
                operation=operation,
                input_path=input_path,
                output_path=output_path,
                status=status,
                details=details,
            )
        except Exception as e:
            logger.warning(f"Failed to record activity: {e}")

    def _check_external_tools(self):
        """Warn at startup if external CLI tools are missing from PATH."""
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found on PATH — video extraction will not work")
        if not shutil.which("exiftool"):
            logger.warning("exiftool not found on PATH — SRT geotagging will not work")
        self._refresh_alignment_binary_cache()

    def _normalize_alignment_engine_name(self, engine_name: str) -> str:
        normalized = (engine_name or "").strip().lower()
        if normalized in {"colmap", "stock", "stock-colmap"}:
            return "colmap"
        if normalized in {"sphere", "spheresfm", "sphere-sfm"}:
            return "spheresfm"
        raise ValueError(f"Unknown alignment engine: {engine_name}")

    def _alignment_binary_pref_key(self, engine_name: str) -> str:
        normalized = self._normalize_alignment_engine_name(engine_name)
        return self._ALIGNMENT_BINARY_PREF_KEYS[normalized]

    def _iter_spheresfm_prebuilt_candidates(self):
        seen = set()
        for candidate in self._SPHERESFM_PREBUILT_CANDIDATES:
            path = Path(candidate).expanduser()
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            yield path

    def _maybe_promote_prebuilt_spheresfm_pref(self) -> bool:
        pref_key = self._alignment_binary_pref_key("spheresfm")
        configured_value = str(self._prefs.get(pref_key, "") or "").strip()
        if not configured_value:
            return False

        try:
            configured_path = Path(configured_value).expanduser().resolve()
        except Exception:
            configured_path = Path(configured_value).expanduser()

        legacy_path = self._SPHERESFM_LEGACY_CUSTOM_DEFAULT
        try:
            legacy_path = legacy_path.resolve()
        except Exception:
            pass

        if str(configured_path).lower() != str(legacy_path).lower():
            return False

        for candidate in self._iter_spheresfm_prebuilt_candidates():
            if candidate.is_file():
                preferred = str(candidate.resolve())
                self._prefs[pref_key] = preferred
                logger.info(
                    "Promoting official prebuilt SphereSfM binary over legacy custom default: %s",
                    preferred,
                )
                return True
        return False

    def _iter_alignment_binary_candidates(self, engine_name: str):
        normalized = self._normalize_alignment_engine_name(engine_name)
        pref_key = self._alignment_binary_pref_key(normalized)
        env_key = self._ALIGNMENT_BINARY_ENV_KEYS[normalized]
        seen = set()

        def add_candidate(raw_path: Optional[str], source: str):
            candidate = (raw_path or "").strip()
            if not candidate:
                return
            path = Path(candidate).expanduser()
            key = str(path).lower()
            if key in seen:
                return
            seen.add(key)
            yield path, source

        yield from add_candidate(self._prefs.get(pref_key, ""), "prefs")
        yield from add_candidate(os.environ.get(env_key, ""), "env")

        if normalized == "colmap":
            yield from add_candidate(shutil.which("colmap"), "path")
            home_colmap_root = Path.home() / "COLMAP"
            if home_colmap_root.is_dir():
                for path in sorted(home_colmap_root.glob("*/bin/colmap.exe"), reverse=True):
                    yield from add_candidate(str(path), "home-colmap")
                yield from add_candidate(str(home_colmap_root / "bin" / "colmap.exe"), "home-colmap")
        else:
            for path in self._iter_spheresfm_prebuilt_candidates():
                yield from add_candidate(str(path), "official-prebuilt-v1.2")
            yield from add_candidate(r"D:\Tools\SphereSfM\bin\colmap.exe", "legacy-custom-spheresfm")
            yield from add_candidate(str(Path.home() / "SphereSfM" / "bin" / "colmap.exe"), "home-spheresfm")

    def _validate_alignment_binary_candidate(self, engine_name: str, binary_path: Path) -> Dict[str, object]:
        from reconstruction_gui.colmap_runner import ColmapRunner

        normalized = self._normalize_alignment_engine_name(engine_name)
        probe_camera_model = "SPHERE" if normalized == "spheresfm" else "PINHOLE"
        probe_root = _project_root / ".tmp" / "alignment_binary_probe" / normalized
        runner = ColmapRunner(
            binary_path=str(binary_path),
            camera_model=probe_camera_model,
            workspace_root=str(probe_root),
            engine_name=normalized,
        )
        validation = runner.validate_binary()
        return {
            "success": validation.success,
            "path": str(binary_path),
            "resolved_binary": validation.resolved_binary,
            "binary_flavor": validation.binary_flavor,
            "supports_sphere_workflow": validation.supports_sphere_workflow,
            "detected_commands": list(validation.detected_commands),
            "error": validation.error,
        }

    def _refresh_alignment_binary_cache(self):
        updated_prefs = self._maybe_promote_prebuilt_spheresfm_pref()

        for engine_name in ("colmap", "spheresfm"):
            pref_key = self._alignment_binary_pref_key(engine_name)
            configured_value = str(self._prefs.get(pref_key, "") or "").strip()
            errors: List[str] = []
            selected: Optional[Dict[str, object]] = None

            for candidate, source in self._iter_alignment_binary_candidates(engine_name):
                try:
                    info = self._validate_alignment_binary_candidate(engine_name, candidate)
                except Exception as exc:
                    errors.append(f"{candidate}: {exc}")
                    continue

                if not info["success"]:
                    errors.append(f"{candidate}: {info['error']}")
                    continue

                if engine_name == "spheresfm" and not info["supports_sphere_workflow"]:
                    errors.append(
                        f"{candidate}: binary validated but does not expose sphere workflow commands"
                    )
                    continue

                if (
                    engine_name == "colmap"
                    and info["binary_flavor"] == "spheresfm-like"
                    and source not in {"prefs", "env"}
                ):
                    errors.append(
                        f"{candidate}: skipping sphere-focused binary for stock COLMAP auto-discovery"
                    )
                    continue

                selected = {
                    **info,
                    "engine_name": engine_name,
                    "source": source,
                    "status": "ready",
                }
                break

            if selected is None:
                message = "; ".join(errors) if errors else "No candidate binary found"
                self._alignment_binary_paths[engine_name] = ""
                self._alignment_binary_status[engine_name] = {
                    "engine_name": engine_name,
                    "path": configured_value,
                    "resolved_binary": "",
                    "binary_flavor": "",
                    "supports_sphere_workflow": False,
                    "detected_commands": [],
                    "status": "missing",
                    "source": "",
                    "error": message,
                }
                if configured_value:
                    logger.warning(
                        "%s binary configured but unavailable: %s",
                        engine_name,
                        message,
                    )
                else:
                    logger.info(
                        "%s binary not discovered yet — alignment support will stay unavailable until one is configured",
                        engine_name,
                    )
                continue

            resolved_binary = str(selected["resolved_binary"])
            self._alignment_binary_paths[engine_name] = resolved_binary
            self._alignment_binary_status[engine_name] = selected

            if configured_value != resolved_binary:
                self._prefs[pref_key] = resolved_binary
                updated_prefs = True

            logger.info(
                "Detected %s binary: %s (%s)",
                engine_name,
                resolved_binary,
                selected["binary_flavor"],
            )

        if updated_prefs:
            self._save_prefs()

    def get_alignment_binary_info(self, engine_name: str) -> Dict[str, object]:
        normalized = self._normalize_alignment_engine_name(engine_name)
        if normalized not in self._alignment_binary_status:
            self._refresh_alignment_binary_cache()
        return dict(self._alignment_binary_status.get(normalized, {}))

    def get_alignment_binary_path(self, engine_name: str) -> str:
        info = self.get_alignment_binary_info(engine_name)
        return str(info.get("resolved_binary", "") or "")

    def create_alignment_runner(
        self,
        engine_name: str,
        workspace_root: str,
        camera_model: str,
        binary_path: Optional[str] = None,
    ):
        from reconstruction_gui.colmap_runner import ColmapRunner

        normalized = self._normalize_alignment_engine_name(engine_name)
        resolved_binary = (binary_path or "").strip() or self.get_alignment_binary_path(normalized)
        if not resolved_binary:
            status = self.get_alignment_binary_info(normalized)
            detail = status.get("error", "No binary configured")
            raise RuntimeError(f"{normalized} binary is unavailable: {detail}")

        return ColmapRunner(
            binary_path=resolved_binary,
            camera_model=camera_model,
            workspace_root=workspace_root,
            engine_name=normalized,
        )

    # ── prefs ──
    # _load_prefs and _save_prefs inherited from AppInfrastructure

    def _restore_prefs(self):
        for key, entry in [("input_dir", self.input_entry), ("output_dir", self.output_entry),
                           ("masks_dir", self.masks_entry), ("images_dir", self.images_entry)]:
            val = self._prefs.get(key, "")
            if val:
                entry.delete(0, "end")
                entry.insert(0, val)
        alignment_entry_map = [
            ("alignment_images_dir", "alignment_images_entry"),
            ("alignment_masks_dir", "alignment_masks_entry"),
            ("alignment_workspace_root", "alignment_workspace_entry"),
            ("alignment_vocab_tree_path", "alignment_vocab_tree_entry"),
            ("alignment_camera_params", "alignment_camera_params_entry"),
            ("alignment_pose_path", "alignment_pose_path_entry"),
            ("alignment_camera_mask_path", "alignment_camera_mask_path_entry"),
            ("alignment_snapshot_path", "alignment_snapshot_path_entry"),
            ("alignment_spatial_max_distance", "alignment_spatial_max_distance_entry"),
        ]
        for pref_key, attr_name in alignment_entry_map:
            entry = getattr(self, attr_name, None)
            val = self._prefs.get(pref_key, "")
            if entry is not None and val:
                entry.delete(0, "end")
                entry.insert(0, val)
        if hasattr(self, "alignment_engine_var") and self._prefs.get("alignment_engine"):
            self.alignment_engine_var.set(self._prefs["alignment_engine"])
        if hasattr(self, "alignment_strategy_var") and self._prefs.get("alignment_strategy"):
            self.alignment_strategy_var.set(self._prefs["alignment_strategy"])
        if hasattr(self, "alignment_mapper_var") and self._prefs.get("alignment_mapper"):
            mapper_value = str(self._prefs["alignment_mapper"])
            if mapper_value == "global_mapper":
                mapper_value = "global"
            self.alignment_mapper_var.set(mapper_value)
        if hasattr(self, "alignment_guided_var") and "alignment_guided_matching" in self._prefs:
            self.alignment_guided_var.set(bool(self._prefs["alignment_guided_matching"]))
        if hasattr(self, "alignment_single_camera_var") and "alignment_single_camera" in self._prefs:
            self.alignment_single_camera_var.set(bool(self._prefs["alignment_single_camera"]))
        if hasattr(self, "alignment_spatial_is_gps_var") and "alignment_spatial_is_gps" in self._prefs:
            self.alignment_spatial_is_gps_var.set(bool(self._prefs["alignment_spatial_is_gps"]))
        if hasattr(self, "alignment_ba_refine_focal_var") and "alignment_ba_refine_focal_length" in self._prefs:
            self.alignment_ba_refine_focal_var.set(bool(self._prefs["alignment_ba_refine_focal_length"]))
        if hasattr(self, "alignment_ba_refine_principal_var") and "alignment_ba_refine_principal_point" in self._prefs:
            self.alignment_ba_refine_principal_var.set(bool(self._prefs["alignment_ba_refine_principal_point"]))
        if hasattr(self, "alignment_ba_refine_extra_var") and "alignment_ba_refine_extra_params" in self._prefs:
            self.alignment_ba_refine_extra_var.set(bool(self._prefs["alignment_ba_refine_extra_params"]))
        if hasattr(self, "alignment_max_features_entry") and self._prefs.get("alignment_max_features"):
            self.alignment_max_features_entry.delete(0, "end")
            self.alignment_max_features_entry.insert(0, str(self._prefs["alignment_max_features"]))
        if hasattr(self, "alignment_max_image_size_entry") and "alignment_max_image_size" in self._prefs:
            self.alignment_max_image_size_entry.delete(0, "end")
            self.alignment_max_image_size_entry.insert(0, str(self._prefs["alignment_max_image_size"]))
        if hasattr(self, "alignment_max_num_matches_entry") and "alignment_max_num_matches" in self._prefs:
            self.alignment_max_num_matches_entry.delete(0, "end")
            self.alignment_max_num_matches_entry.insert(0, str(self._prefs["alignment_max_num_matches"]))
        if hasattr(self, "alignment_min_num_inliers_entry") and "alignment_min_num_inliers" in self._prefs:
            self.alignment_min_num_inliers_entry.delete(0, "end")
            self.alignment_min_num_inliers_entry.insert(0, str(self._prefs["alignment_min_num_inliers"]))
        if hasattr(self, "alignment_snapshot_freq_entry") and "alignment_snapshot_images_freq" in self._prefs:
            self.alignment_snapshot_freq_entry.delete(0, "end")
            snapshot_freq = self._prefs["alignment_snapshot_images_freq"]
            if snapshot_freq not in ("", None):
                self.alignment_snapshot_freq_entry.insert(0, str(snapshot_freq))
        if hasattr(self, "alignment_binary_entry"):
            engine_name = self._prefs.get("alignment_engine", "colmap")
            pref_key = self._ALIGNMENT_BINARY_PREF_KEYS.get(engine_name, self._ALIGNMENT_BINARY_PREF_KEYS["colmap"])
            binary_val = self._prefs.get(pref_key, "")
            if binary_val:
                self.alignment_binary_entry.delete(0, "end")
                self.alignment_binary_entry.insert(0, binary_val)
                detected_binary = self.get_alignment_binary_path(engine_name)
                self._alignment_last_auto_binary = binary_val if binary_val == detected_binary else ""
        if hasattr(self, "alignment_camera_model_var") and self._prefs.get("alignment_camera_model"):
            engine_name = self._prefs.get("alignment_engine", "colmap")
            restored_camera_model = self._prefs.get("alignment_camera_model", "")
            if engine_name != "spheresfm" and restored_camera_model:
                self.alignment_camera_model_var.set(restored_camera_model)
            default_camera_model = "SPHERE" if engine_name == "spheresfm" else "PINHOLE"
            self._alignment_last_auto_camera_model = (
                restored_camera_model if restored_camera_model == default_camera_model else ""
            )
        # Rig mode, preset, file
        # Rig mode is derived from preset selection, no checkbox to restore
        if hasattr(self, "alignment_rig_preset_var") and self._prefs.get("alignment_rig_preset"):
            self.alignment_rig_preset_var.set(self._prefs["alignment_rig_preset"])
        if hasattr(self, "alignment_rig_file_entry") and self._prefs.get("alignment_rig_file"):
            self.alignment_rig_file_entry.delete(0, "end")
            self.alignment_rig_file_entry.insert(0, self._prefs["alignment_rig_file"])
        for pref_key, attr_name in [
            ("alignment_extract_args", "alignment_extract_args_text"),
            ("alignment_match_args", "alignment_match_args_text"),
            ("alignment_reconstruct_args", "alignment_reconstruct_args_text"),
        ]:
            textbox = getattr(self, attr_name, None)
            text_val = self._prefs.get(pref_key, "")
            if textbox is not None and text_val:
                textbox.configure(state="normal")
                textbox.delete("1.0", "end")
                textbox.insert("1.0", str(text_val))
                textbox.configure(state="disabled")
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
        # Restore static mask layers if output dir has them
        output_dir = self._prefs.get("output_dir", "")
        if output_dir and Path(output_dir).is_dir():
            sm_dir = Path(output_dir) / "static_masks"
            if sm_dir.exists():
                self._init_static_mask_manager(output_dir)

    def _on_close(self):
        """Fast shutdown — kill subprocesses, clear caches, destroy widgets."""
        # Persist tracker store path
        if hasattr(self, '_project_store') and self._project_store:
            self._prefs["tracker_store_path"] = str(self._project_store.store_path)
            self._save_prefs()
        # Signal any running extraction/processing threads to stop
        self.cancel_flag.set()
        if hasattr(self, "_alignment_cancel_event") and self._alignment_cancel_event is not None:
            self._alignment_cancel_event.set()
        # Destroy VTK viewer before window is destroyed
        from tabs.alignment_tab import alignment_viewer_destroy
        alignment_viewer_destroy(self)
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
        # Left column: fixed width with a minimum floor.
        # Right column: absorbs all width changes.
        # Derive minimum from screen width so it scales across monitor sizes.
        screen_w = self.winfo_screenwidth()  # logical pixels (DPI-adjusted)
        self._LEFT_COL_MIN = max(400, min(600, int(screen_w * 0.28)))
        self._main_frame.grid_columnconfigure(0, weight=0, minsize=self._LEFT_COL_MIN)
        self._main_frame.grid_columnconfigure(1, weight=1)
        self._main_frame.grid_rowconfigure(0, weight=1)
        self._col_layout_initialized = False
        self._main_frame.bind("<Configure>", self._on_main_frame_configure)

        # Left: tabview (settings tabs only)
        self.tabs = ctk.CTkTabview(self._main_frame, command=self._on_tab_change)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        self.tabs.add("Projects")
        self.tabs.add("Extract")
        self.tabs.add("Mask")
        self.tabs.add("Review")
        self.tabs.add("Align")
        self.tabs.add("Coverage")

        # Right: shared preview panel (collapsible)
        self._preview_visible = True
        self._preview_panel = ctk.CTkFrame(self._main_frame, corner_radius=0)
        self._preview_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=(17, 0))
        self._build_preview_panel()

        build_projects_tab(self, self.tabs.tab("Projects"))
        build_source_tab(self, self.tabs.tab("Extract"))
        self._build_process_tab()
        self._build_review_tab()
        build_alignment_tab(self, self.tabs.tab("Align"))
        build_gaps_tab(self, self.tabs.tab("Coverage"))

        # Projects is the default tab — swap preview for detail panel
        self.after(50, self._on_tab_change)

    # ── responsive column layout ──

    def _on_main_frame_configure(self, event=None):
        """Responsive two-column layout.

        On first render, captures 35% of the frame width as the left
        column's fixed size (floored at _LEFT_COL_MIN).  The right
        column absorbs all subsequent width changes.
        """
        if self._col_layout_initialized:
            return  # nothing to do after init

        frame_w = event.width if event else self._main_frame.winfo_width()
        if frame_w < 400:
            return  # not rendered yet

        left_w = max(int(frame_w * 0.35), self._LEFT_COL_MIN)
        self._main_frame.grid_columnconfigure(0, weight=0, minsize=left_w)
        self._main_frame.grid_columnconfigure(1, weight=1)
        self._col_layout_initialized = True

    # ── slider helper ──

    def _slider(self, parent, label, from_, to, default, steps,
                fmt=".2f", width=100, pad_left=0):
        """Delegate to shared slider_row (from widgets.py)."""
        return slider_row(parent, label, from_, to, default,
                          steps=steps, fmt=fmt, width=width, pad_left=pad_left)

    # ══════════════════════════════════════════════════════════════════
    # PROCESS TAB
    # ══════════════════════════════════════════════════════════════════

    def _build_process_tab(self):
        tab = self.tabs.tab("Mask")

        # Scrollable settings (preview panel is shared, outside tabs)
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)

        # ==============================================================
        #  DETECTION
        # ==============================================================
        core_sec = _CollapsibleSection(scroll, "Detection", expanded=True)
        core_sec.pack(fill="x", pady=(0, 6), padx=4)
        core = core_sec.content

        # ── I/O ──
        row = ctk.CTkFrame(core, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(row, text="Input:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.input_entry = ctk.CTkEntry(row, placeholder_text="Folder, image, or video")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row, text="Folder", width=55,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12),
                      command=self._browse_input_folder).pack(side="left", padx=(0, 2))
        ctk.CTkButton(row, text="File", width=45,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12),
                      command=self._browse_input_file).pack(side="left")

        row2 = ctk.CTkFrame(core, fg_color="transparent")
        row2.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(row2, text="Output:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.output_entry = ctk.CTkEntry(row2, placeholder_text="Output folder for masks")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(row2, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=self._browse_output_folder).pack(side="left")

        row2b = ctk.CTkFrame(core, fg_color="transparent")
        row2b.pack(fill="x", padx=6, pady=(0, 3))
        ctk.CTkLabel(row2b, text="", width=LABEL_FIELD_WIDTH).pack(side="left")
        self.review_folder_var = ctk.BooleanVar(value=False)
        _rf = ctk.CTkCheckBox(row2b, text="Create review folder",
                              variable=self.review_folder_var, width=0)
        _rf.pack(side="left", padx=(5, 0))
        Tooltip(_rf, "Creates masks/ and review/ subdirectories inside Output")
        self.review_rejects_var = ctk.BooleanVar(value=True)
        _rr = ctk.CTkCheckBox(row2b, text="Include rejects",
                              variable=self.review_rejects_var, width=0)
        _rr.pack(side="left", padx=(16, 0))
        Tooltip(_rr, "Creates a mask for every image regardless of confidence score")

        self.pattern_var = ctk.StringVar(value="*.jpg *.png")
        ctk.CTkEntry(row2b, textvariable=self.pattern_var, width=80).pack(side="right", padx=(2, 70))
        ctk.CTkLabel(row2b, text="File types:").pack(side="right", padx=(0, 2))

        # ── Model ──
        sam3_mode_row = ctk.CTkFrame(core, fg_color="transparent")
        sam3_mode_row.pack(fill="x", padx=6, pady=(8, 3))
        ctk.CTkLabel(sam3_mode_row, text="SAM3 mode:").pack(side="left", padx=(0, 2))
        self.sam3_unified_var = ctk.BooleanVar(value=False)
        self._sam3_mode_btn = ctk.CTkSegmentedButton(
            sam3_mode_row, values=["Hybrid", "Unified Video"],
            command=self._on_sam3_mode_change,
        )
        self._sam3_mode_btn.set("Hybrid")
        self._sam3_mode_btn.pack(side="left", padx=4)
        ctk.CTkLabel(sam3_mode_row,
                     text="Hybrid = per-frame detect+segment; Unified = SAM3 video tracking",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(6, 0))

        r1 = ctk.CTkFrame(core, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r1, text="Model:").pack(side="left", padx=(0, 2))
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
        ctk.CTkOptionMenu(r1, variable=self.model_var,
                          values=_model_values,
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

        r1b = ctk.CTkFrame(core, fg_color="transparent")
        r1b.pack(fill="x", padx=6, pady=3)
        self.conf_var = self._slider(r1b, "Conf", 0, 1, 0.70, 20, width=100)
        self.review_thresh_var = self._slider(r1b, "Review thresh", 0, 1, 0.85, 20,
                                              pad_left=40)
        self.torch_compile_var = ctk.BooleanVar(value=False)
        _tc = ctk.CTkCheckBox(r1b, text="torch.compile",
                              variable=self.torch_compile_var, width=0)
        _tc.pack(side="left", padx=(40, 0))
        Tooltip(_tc, "Compile models with torch.compile() for ~20-50% faster\n"
                     "GPU inference. First run is slow (warmup).\n"
                     "Requires PyTorch 2.0+ and CUDA.")

        # ── Prompts & Targets ──
        r2 = ctk.CTkFrame(core, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=(8, 3))
        ctk.CTkLabel(r2, text="Remove:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.remove_prompts_entry = ctk.CTkEntry(
            r2, placeholder_text="default: person, tripod, backpack, selfie stick")
        self.remove_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        r3 = ctk.CTkFrame(core, fg_color="transparent")
        r3.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r3, text="Keep:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.keep_prompts_entry = ctk.CTkEntry(
            r3, placeholder_text="(optional) objects to protect from masking")
        self.keep_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        self._target_vars: dict = {}

        mp_wrapper = ctk.CTkFrame(core, fg_color="transparent")
        mp_wrapper.pack(fill="x", padx=6, pady=3)
        mp_toggle_row = ctk.CTkFrame(mp_wrapper, fg_color="transparent")
        mp_toggle_row.pack(fill="x")
        self.multi_pass_var = ctk.BooleanVar(value=False)
        _mp = ctk.CTkCheckBox(mp_toggle_row, text="Multi-pass SAM3",
                              variable=self.multi_pass_var,
                              command=self._toggle_multi_pass, width=0)
        _mp.pack(side="left")
        Tooltip(_mp, "Run SAM3 once per prompt with individual confidence\n"
                     "thresholds, then union all masks together")
        ctk.CTkButton(mp_toggle_row, text="+ Add Pass", width=80,
                      command=self._add_multi_pass_row).pack(side="right", padx=2)
        ctk.CTkButton(mp_toggle_row, text="Remove", width=70,
                      command=self._remove_multi_pass_row).pack(side="right", padx=2)

        self.multi_pass_entries = []
        self.multi_pass_list_frame = ctk.CTkFrame(mp_wrapper, fg_color="transparent")
        # Initially hidden — shown when checkbox is checked

        # Frames whose interactive children are disabled in Unified mode
        self._unified_disable_frames = [r1, r1b]

        # ── Output ──
        out_row = ctk.CTkFrame(core, fg_color="transparent")
        out_row.pack(fill="x", padx=6, pady=(8, 3))
        self.skip_existing_var = ctk.BooleanVar(value=False)
        _se = ctk.CTkCheckBox(out_row, text="Skip existing",
                              variable=self.skip_existing_var, width=0)
        _se.pack(side="left")
        Tooltip(_se, "Skip images that already have a mask file in the output directory")
        self.multi_label_var = ctk.BooleanVar(value=False)
        _ml = ctk.CTkCheckBox(out_row, text="Multi-label output",
                              variable=self.multi_label_var, width=0)
        _ml.pack(side="left")
        Tooltip(_ml, "Export per-class segmentation maps (pixel values = class IDs)\n"
                     "for Gaussian Grouping and semantic scene decomposition")
        self.inpaint_var = ctk.BooleanVar(value=False)
        _ip = ctk.CTkCheckBox(out_row, text="Inpaint masked",
                              variable=self.inpaint_var, width=0)
        _ip.pack(side="left", padx=(40, 0))
        Tooltip(_ip, "Fill masked regions with plausible background texture\n"
                     "so 3DGS gets gradient signal instead of black void")

        # ==============================================================
        #  STATIC MASKS (user-authored persistent overlays)
        # ==============================================================
        sm_sec = _CollapsibleSection(scroll, "Static Masks",
                                     subtitle="persistent overlays applied to every frame",
                                     expanded=False)
        sm_sec.pack(fill="x", pady=(0, 6), padx=4)
        sm = sm_sec.content

        # Layer list — plain frame, grows with content (no empty scroll area)
        self._static_mask_list = ctk.CTkFrame(sm, fg_color="transparent")
        self._static_mask_list.pack(fill="x", padx=6, pady=(2, 4))

        # Placeholder label shown when no layers exist
        self._sm_empty_label = ctk.CTkLabel(
            self._static_mask_list, text="No static masks defined",
            font=("Consolas", 10), text_color="#6b7280",
        )
        self._sm_empty_label.pack(pady=4)

        # Buttons row
        sm_btns = ctk.CTkFrame(sm, fg_color="transparent")
        sm_btns.pack(fill="x", padx=6, pady=(0, 4))

        ctk.CTkButton(sm_btns, text="New", width=70, height=28,
                       fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_new_static_mask).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_btns, text="Edit", width=70, height=28,
                       fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_edit_static_mask).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_btns, text="Delete", width=70, height=28,
                       fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_delete_static_mask).pack(side="left")

        self._static_mask_manager = None  # Initialized when input dir is set
        self._sm_layer_widgets = []       # List of (frame, checkbox_var, name_label) per layer
        self._sm_selected_index = None    # Currently selected layer for edit/delete

        # ==============================================================
        #  DETECTION & REFINEMENT
        # ==============================================================
        self._detect_sec = _CollapsibleSection(scroll, "Detection & Refinement", expanded=True)
        detect_sec = self._detect_sec
        detect_sec.pack(fill="x", pady=(0, 6), padx=4)
        detect = detect_sec.content

        # Shadow Detection
        shadow_section = _CollapsibleSection(detect, "Shadow Detection",
            subtitle="detect shadows cast by masked objects")
        shadow_section.pack(fill="x", padx=2, pady=(0, 4))
        sc = shadow_section.content

        sr1 = ctk.CTkFrame(sc, fg_color="transparent")
        sr1.pack(fill="x", pady=2)
        self.shadow_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sr1, text="Enable", variable=self.shadow_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(sr1, text="Detector:").pack(side="left", padx=(12, 2))
        self.shadow_detector_var = ctk.StringVar(value="targeted_person")
        ctk.CTkOptionMenu(sr1, variable=self.shadow_detector_var,
                          values=["targeted_person", "brightness",
                                  "c1c2c3", "hybrid"],
                          width=140).pack(side="left", padx=2)
        ctk.CTkLabel(sr1, text="Verify:").pack(side="left", padx=(12, 2))
        self.shadow_verifier_var = ctk.StringVar(value="none")
        ctk.CTkOptionMenu(sr1, variable=self.shadow_verifier_var,
                          values=["none", "c1c2c3", "hybrid", "brightness"],
                          width=100).pack(side="left", padx=2)

        sr2 = ctk.CTkFrame(sc, fg_color="transparent")
        sr2.pack(fill="x", pady=2)
        ctk.CTkLabel(sr2, text="Spatial:").pack(side="left")
        self.shadow_spatial_var = ctk.StringVar(value="near_objects")
        ctk.CTkSegmentedButton(sr2, values=["all", "near_objects", "connected"],
                               variable=self.shadow_spatial_var).pack(side="left", padx=4)
        self.shadow_dilation_var = self._slider(sr2, "Dilation", 0, 200, 50, 40,
                                               fmt=".0f", pad_left=8)
        ctk.CTkLabel(sr2, text="px", font=("Consolas", 10),
                     text_color="#9ca3af").pack(side="left", padx=2)

        sr3 = ctk.CTkFrame(sc, fg_color="transparent")
        sr3.pack(fill="x", pady=2)
        self.shadow_conf_var = self._slider(sr3, "Confidence", 0, 1, 0.50, 20)
        self.shadow_darkness_var = self._slider(sr3, "Darkness", 0, 1, 0.70, 20, pad_left=8)
        sr4 = ctk.CTkFrame(sc, fg_color="transparent")
        sr4.pack(fill="x", pady=2)
        self.shadow_chroma_var = self._slider(sr4, "Chromaticity", 0, 0.5, 0.15, 25)

        # SAM Mask Refinement
        sam_section = _CollapsibleSection(detect, "SAM Mask Refinement",
            subtitle="tighten mask edges using SAM point prompts")
        sam_section.pack(fill="x", padx=2, pady=(0, 4))
        samc = sam_section.content

        samr1 = ctk.CTkFrame(samc, fg_color="transparent")
        samr1.pack(fill="x", pady=2)
        self.sam_refine_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(samr1, text="Enable", variable=self.sam_refine_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(samr1, text="SAM model:").pack(side="left", padx=(12, 2))
        self.sam_model_var = ctk.StringVar(value="vit_b")
        ctk.CTkOptionMenu(samr1, variable=self.sam_model_var,
                          values=["vit_b", "vit_l", "vit_h"],
                          width=90).pack(side="left", padx=2)
        ctk.CTkLabel(samr1, text="(vit_b=375MB, fastest)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        samr2 = ctk.CTkFrame(samc, fg_color="transparent")
        samr2.pack(fill="x", pady=2)
        self.sam_margin_var = self._slider(samr2, "Box margin", 0, 0.5, 0.15, 10)
        self.sam_iou_var = self._slider(samr2, "IoU threshold", 0, 1, 0.5, 20, pad_left=8)

        # Alpha Matting
        mat_section = _CollapsibleSection(detect, "Alpha Matting (ViTMatte)",
            subtitle="soft alpha edges for hair, fur, semi-transparent boundaries")
        mat_section.pack(fill="x", padx=2, pady=(0, 4))
        matc = mat_section.content

        matr1 = ctk.CTkFrame(matc, fg_color="transparent")
        matr1.pack(fill="x", pady=2)
        self.matting_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(matr1, text="Enable", variable=self.matting_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(matr1, text="Model:").pack(side="left", padx=(12, 2))
        self.matting_model_var = ctk.StringVar(value="small")
        ctk.CTkOptionMenu(matr1, variable=self.matting_model_var,
                          values=["small", "base"],
                          width=90).pack(side="left", padx=2)
        ctk.CTkLabel(matr1, text="(small=~100MB, recommended)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        matr2 = ctk.CTkFrame(matc, fg_color="transparent")
        matr2.pack(fill="x", pady=2)
        self.matting_erode_var = self._slider(matr2, "Erode", 0, 50, 10, 50, fmt=".0f")
        self.matting_dilate_var = self._slider(matr2, "Dilate", 0, 50, 10, 50,
                                              fmt=".0f", pad_left=8)
        ctk.CTkLabel(matr2, text="(trimap border)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        # Ensemble Detection
        ens_section = _CollapsibleSection(detect, "Ensemble Detection (WMF)",
            subtitle="run multiple detectors and merge results")
        ens_section.pack(fill="x", padx=2, pady=(0, 4))
        ensc = ens_section.content

        ensr1 = ctk.CTkFrame(ensc, fg_color="transparent")
        ensr1.pack(fill="x", pady=2)
        self.ensemble_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(ensr1, text="Enable", variable=self.ensemble_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(ensr1, text="RF-DETR size:").pack(side="left", padx=(12, 2))
        self.rfdetr_size_var = ctk.StringVar(value="small")
        ctk.CTkOptionMenu(ensr1, variable=self.rfdetr_size_var,
                          values=["nano", "small", "medium", "large"],
                          width=90).pack(side="left", padx=2)
        ctk.CTkLabel(ensr1, text="(runs YOLO26 + RF-DETR, fuses masks)",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=4)

        ensr2 = ctk.CTkFrame(ensc, fg_color="transparent")
        ensr2.pack(fill="x", pady=2)
        ctk.CTkLabel(ensr2, text="Models:").pack(side="left")
        self.ens_models_var = ctk.StringVar(value="yolo26, rfdetr")
        ctk.CTkEntry(ensr2, textvariable=self.ens_models_var,
                     width=140).pack(side="left", padx=2)
        self.ens_iou_var = self._slider(ensr2, "IoU threshold", 0, 1, 0.5, 20, pad_left=8)

        # Edge Injection
        edge_section = _CollapsibleSection(detect, "Edge Injection",
            subtitle="canny edges for thin structures like wires and antennas")
        edge_section.pack(fill="x", padx=2, pady=(0, 4))
        edgec = edge_section.content

        edge_row = ctk.CTkFrame(edgec, fg_color="transparent")
        edge_row.pack(fill="x", pady=2)
        self.edge_inject_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(edge_row, text="Enable",
                        variable=self.edge_inject_var, width=80).pack(side="left")
        ctk.CTkLabel(edge_row,
                     text="Detects thin structures that object detectors miss",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=8)

        # COLMAP Geometric Validation
        col_section = _CollapsibleSection(detect, "COLMAP Geometric Validation",
            subtitle="cross-check masks against 3D reconstruction")
        col_section.pack(fill="x", padx=2, pady=(0, 4))
        colc = col_section.content

        colr1 = ctk.CTkFrame(colc, fg_color="transparent")
        colr1.pack(fill="x", pady=2)
        self.colmap_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(colr1, text="Enable", variable=self.colmap_var,
                        width=80).pack(side="left")
        ctk.CTkLabel(colr1, text="Sparse dir:").pack(side="left", padx=(12, 2))
        self.colmap_dir_var = ctk.StringVar(value="")
        ctk.CTkEntry(colr1, textvariable=self.colmap_dir_var,
                     width=200).pack(side="left", padx=2, fill="x", expand=True)
        ctk.CTkButton(colr1, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=self._browse_colmap_dir).pack(side="left", padx=2)

        colr2 = ctk.CTkFrame(colc, fg_color="transparent")
        colr2.pack(fill="x", pady=2)
        self.colmap_agree_var = self._slider(colr2, "Agreement", 0, 1, 0.7, 20)
        self.colmap_flag_var = self._slider(colr2, "Flag above", 0, 1, 0.15, 20, pad_left=8)
        ctk.CTkLabel(colr2, text="3D check",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(6, 0))

        # ==============================================================
        #  POST-PROCESSING
        # ==============================================================
        self._post_sec = _CollapsibleSection(scroll, "Post-Processing", expanded=True)
        post_sec = self._post_sec
        post_sec.pack(fill="x", pady=(0, 6), padx=4)
        post = post_sec.content

        # General (all image types)
        gen_row = ctk.CTkFrame(post, fg_color="transparent")
        gen_row.pack(fill="x", padx=6, pady=3)
        self.mask_dilate_var = self._slider(gen_row, "Mask dilate", 0, 50, 0, 50,
                                             fmt=".0f")
        ctk.CTkLabel(gen_row, text="px",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(2, 0))
        self.fill_holes_var = ctk.BooleanVar(value=False)
        _fh = ctk.CTkCheckBox(gen_row, text="Fill holes",
                               variable=self.fill_holes_var, width=0)
        _fh.pack(side="left", padx=(40, 0))
        Tooltip(_fh, "Fill enclosed holes inside the mask\n"
                     "(camera mount, equipment gaps)")

        # Equirectangular
        erp_sec = _CollapsibleSection(post, "Equirectangular",
                                      subtitle="nadir masking, pole expansion, cubemap seam overlap",
                                      expanded=True)
        erp_sec.pack(fill="x", pady=(0, 2), padx=2)
        erp = erp_sec.content

        erp_row1 = ctk.CTkFrame(erp, fg_color="transparent")
        erp_row1.pack(fill="x", padx=6, pady=3)
        self.nadir_mask_var = self._slider(erp_row1, "Nadir mask", 0, 25, 0, 50,
                                           fmt=".0f")
        _nm_unit = ctk.CTkLabel(erp_row1, text="%",
                                font=("Consolas", 10), text_color="#9ca3af")
        _nm_unit.pack(side="left", padx=(2, 0))
        self.pole_expand_var = self._slider(erp_row1, "Pole expand", 0, 5, 1.2, 50,
                                            fmt=".1f", pad_left=40)
        Tooltip(_nm_unit, "Auto-mask the bottom N% of equirectangular images\n"
                          "(hides nadir tripod/photographer)")

        erp_row2 = ctk.CTkFrame(erp, fg_color="transparent")
        erp_row2.pack(fill="x", padx=6, pady=3)
        self.cubemap_overlap_var = self._slider(erp_row2, "Cubemap overlap", 0, 30, 0, 30,
                                                 fmt=".0f")
        ctk.CTkLabel(erp_row2, text="°",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(2, 0))

        # Fisheye
        fe_sec = _CollapsibleSection(post, "Fisheye",
                                     subtitle="circle masking and periphery trimming",
                                     expanded=False)
        fe_sec.pack(fill="x", pady=(0, 2), padx=2)
        fe = fe_sec.content

        fe_row = ctk.CTkFrame(fe, fg_color="transparent")
        fe_row.pack(fill="x", padx=6, pady=3)
        self.fisheye_circle_var = ctk.BooleanVar(value=True)
        _fc = ctk.CTkCheckBox(fe_row, text="Fisheye circle mask",
                               variable=self.fisheye_circle_var, width=0)
        _fc.pack(side="left")
        Tooltip(_fc, "Mask corners and periphery outside the fisheye circle")
        self.fisheye_margin_var = self._slider(fe_row, "Margin", 0, 20, 0, 50,
                                                fmt=".0f", pad_left=40)
        ctk.CTkLabel(fe_row, text="%",
                     font=("Consolas", 10), text_color="#9ca3af").pack(side="left", padx=(2, 0))

        # ==============================================================
        #  RUN / CANCEL
        # ==============================================================
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(fill="x", pady=6)

        self._preview_run_btn = ctk.CTkButton(
            btn_row, text="Preview", width=90, height=38,
            fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
            font=ctk.CTkFont(size=12),
            command=self._on_preview,
        )
        self._preview_run_btn.pack(side="left", padx=(10, 5))

        self.run_btn = ctk.CTkButton(
            btn_row, text="Run Masking", height=38,
            fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_run,
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=5)

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Stop", width=70, height=38,
            fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
            font=ctk.CTkFont(size=12),
            command=self._on_stop,
        )

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
        ctk.CTkButton(mq_ctrl, text="Add Current", width=85,
                      command=self._mq_add_current).pack(side="left", padx=(0, 4))
        ctk.CTkButton(mq_ctrl, text="Add Subfolders", width=100,
                      command=self._mq_add_subfolders).pack(side="left", padx=(0, 4))
        ctk.CTkButton(mq_ctrl, text="Remove", width=60,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12),
                      command=self._mq_remove_selected).pack(side="left", padx=(0, 4))
        ctk.CTkButton(mq_ctrl, text="Clear Done", width=72,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12),
                      command=self._mq_clear_done).pack(side="left")

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
            fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
            font=ctk.CTkFont(size=13, weight="bold"), height=38,
        )
        self.mq_run_btn.pack(side="left", fill="x", expand=True)

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

    def _browse_output_folder(self):
        self._browse_dir_into(self.output_entry, title="Select Output Folder")
        self._reset_static_mask_manager()

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
        header.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
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
            self._main_frame.grid_columnconfigure(1, weight=1, uniform="")
            self._preview_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=(17, 0))
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

        # Swap right-side panel: Projects info vs. Align detail vs. preview
        if active == "Projects":
            self._preview_panel.grid_forget()
            if hasattr(self, '_preview_show_btn'):
                self._preview_show_btn.destroy()
            if hasattr(self, '_alignment_detail_panel'):
                self._alignment_detail_panel.grid_forget()
            if hasattr(self, '_proj_right_panel'):
                self._proj_right_panel.grid(
                    row=0, column=1, sticky="nsew", padx=0, pady=(17, 0),
                )
            self._main_frame.grid_columnconfigure(1, weight=1, uniform="")
            # Refresh recent activity when returning to Projects tab
            if hasattr(self, '_proj_list_mode_var') and self._proj_list_mode_var.get() == "Recent Activity":
                from tabs.projects_tab import _refresh_recent_activity
                _refresh_recent_activity(self)
        elif active == "Align":
            self._preview_panel.grid_forget()
            if hasattr(self, '_preview_show_btn'):
                self._preview_show_btn.destroy()
            if hasattr(self, '_proj_right_panel'):
                self._proj_right_panel.grid_forget()
            if hasattr(self, '_alignment_detail_panel'):
                self._alignment_detail_panel.grid(
                    row=0, column=1, sticky="nsew", padx=0, pady=(17, 0),
                )
            self._main_frame.grid_columnconfigure(1, weight=1, uniform="")
            from tabs.alignment_tab import alignment_viewer_resume
            alignment_viewer_resume(self)
        else:
            if hasattr(self, '_proj_right_panel'):
                self._proj_right_panel.grid_forget()
            if hasattr(self, '_alignment_detail_panel'):
                self._alignment_detail_panel.grid_forget()
            from tabs.alignment_tab import alignment_viewer_pause
            alignment_viewer_pause(self)
            if self._preview_visible:
                self._main_frame.grid_columnconfigure(1, weight=1, uniform="")
                self._preview_panel.grid(
                    row=0, column=1, sticky="nsew", padx=0, pady=(17, 0),
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

        self._nav_slider.configure(to=n - 1, number_of_steps=max(n - 1, 1))
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

        self._nav_slider.configure(to=n - 1, number_of_steps=max(n - 1, 1))
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

    # ------------------------------------------------------------------
    #  Static mask handlers
    # ------------------------------------------------------------------

    def _init_static_mask_manager(self, dataset_dir: str):
        """Initialize the static mask manager for the current dataset."""
        from static_masks import StaticMaskManager
        self._static_mask_manager = StaticMaskManager(Path(dataset_dir))
        self._refresh_static_mask_list()

    def _reset_static_mask_manager(self):
        """Reset the static mask manager and reload from the current output dir."""
        self._static_mask_manager = None
        self._refresh_static_mask_list()
        # Auto-discover if the new output dir already has static masks
        output_dir = self.output_entry.get().strip()
        if output_dir and Path(output_dir).is_dir():
            sm_dir = Path(output_dir) / "static_masks"
            if sm_dir.exists():
                self._init_static_mask_manager(output_dir)

    def _refresh_static_mask_list(self):
        """Rebuild the layer list UI from the manager's state."""
        for w in self._sm_layer_widgets:
            w["frame"].destroy()
        self._sm_layer_widgets = []

        mgr = self._static_mask_manager
        if mgr is None or not mgr.layers:
            self._sm_empty_label.pack(pady=4)
            return

        self._sm_empty_label.pack_forget()

        for i, layer in enumerate(mgr.layers):
            row = ctk.CTkFrame(self._static_mask_list, fg_color="#2b2b2b",
                                corner_radius=5)
            row.pack(fill="x", pady=2, padx=2)

            # Click-to-select toggle (batch queue pattern)
            selected = False
            def toggle(event=None, idx=i):
                nonlocal selected
                selected = not selected
                self._sm_layer_widgets[idx]["selected"] = selected
                self._sm_layer_widgets[idx]["frame"].configure(
                    fg_color="#3d5a80" if selected else "#2b2b2b")
            row.bind("<Button-1>", toggle)

            var = ctk.BooleanVar(value=layer.enabled)
            cb = ctk.CTkCheckBox(row, text="", variable=var, width=24,
                                  command=lambda idx=i, v=var: self._on_toggle_static_mask(idx, v))
            cb.pack(side="left", padx=(6, 0))

            name_label = ctk.CTkLabel(row, text=layer.name,
                                       font=("Consolas", 11), text_color="#e5e7eb")
            name_label.pack(side="left", padx=(4, 0))
            name_label.bind("<Button-1>", toggle)

            self._sm_layer_widgets.append({
                "frame": row,
                "enabled_var": var,
                "name_label": name_label,
                "selected": False,
                "index": i,
            })

    def _on_toggle_static_mask(self, index, var):
        """Toggle a layer's enabled state."""
        if self._static_mask_manager:
            self._static_mask_manager.set_enabled(index, var.get())

    def _on_new_static_mask(self):
        """Open the OpenCV editor to paint a new static mask."""
        input_dir = self.input_entry.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            self.log("Set an input directory first")
            return
        output_dir = self.output_entry.get().strip()
        if not output_dir:
            self.log("Set an output directory first")
            return

        # Re-init if manager is missing or points to a different output dir
        expected_dir = Path(output_dir) / "static_masks"
        if (self._static_mask_manager is None
                or self._static_mask_manager.static_dir != expected_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self._init_static_mask_manager(output_dir)

        reference = self._get_current_reference_image()
        if reference is None:
            self.log("No images found in input directory")
            return
        self.log(f"Opening static mask editor on: {reference.name}")

        import threading

        def _paint():
            from review_masks import MaskReviewer
            reviewer = MaskReviewer()
            result = reviewer.paint_static_mask(
                reference, image_list=self._image_list,
            )
            if result is not None:
                self.after(0, lambda: self._prompt_static_mask_name(result))
            else:
                self.after(0, lambda: self.log("Static mask cancelled"))

        threading.Thread(target=_paint, daemon=True).start()

    def _prompt_static_mask_name(self, mask):
        """Show a small dialog to name the new static mask layer."""
        dialog = ctk.CTkInputDialog(
            text="Name this static mask layer:",
            title="Save Static Mask",
        )
        name = dialog.get_input()
        if name and name.strip():
            self._static_mask_manager.add_layer(name.strip(), mask)
            self._refresh_static_mask_list()
            self.log(f"Static mask saved: {name.strip()}")
            # Persist to active project if one is selected
            if (hasattr(self, '_project_store') and self._project_store
                    and hasattr(self, '_selected_project_id') and self._selected_project_id):
                proj = self._project_store.get(self._selected_project_id)
                if proj:
                    proj.static_masks_dir = str(self._static_mask_manager.static_dir)
                    self._project_store.save()
        else:
            self.log("Static mask not saved (no name given)")

    def _get_selected_static_mask_indices(self):
        """Return indices of selected (highlighted) static mask layers."""
        return [w["index"] for w in self._sm_layer_widgets if w["selected"]]

    def _get_current_reference_image(self):
        """Get the current preview image path, or first image in input dir."""
        # Use the image currently shown in the preview panel
        preview = self._preview_image_entry.get().strip()
        if preview and Path(preview).is_file():
            return Path(preview)
        # Fallback to first image in input dir
        input_dir = self.input_entry.get().strip()
        if input_dir and Path(input_dir).is_dir():
            img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            images = sorted(
                p for p in Path(input_dir).iterdir()
                if p.suffix.lower() in img_exts
            )
            if images:
                return images[0]
        return None

    def _on_edit_static_mask(self):
        """Re-open the editor for the selected static mask layer."""
        selected = self._get_selected_static_mask_indices()
        if not selected:
            self.log("Click a static mask row to select it, then click Edit")
            return
        if len(selected) > 1:
            self.log("Select one layer to edit (multiple selected)")
            return
        mgr = self._static_mask_manager
        if mgr is None:
            return

        import cv2 as _cv2
        import numpy as _np

        idx = selected[0]
        layer = mgr.layers[idx]
        mask_path = mgr.static_dir / layer.filename
        existing = _cv2.imread(str(mask_path), _cv2.IMREAD_GRAYSCALE)
        if existing is None:
            self.log(f"Could not load mask: {mask_path}")
            return
        existing = (existing > 127).astype(_np.uint8)

        reference = self._get_current_reference_image()
        if reference is None:
            self.log("No images found in input directory")
            return
        self.log(f"Editing static mask '{layer.name}' on: {reference.name}")

        import threading

        def _edit():
            from review_masks import MaskReviewer
            reviewer = MaskReviewer()
            result = reviewer.paint_static_mask(
                reference, existing_mask=existing,
                image_list=self._image_list,
            )
            if result is not None:
                mgr.update_layer(idx, result)
                self.after(0, lambda: self.log(f"Static mask updated: {layer.name}"))
            else:
                self.after(0, lambda: self.log("Edit cancelled"))

        threading.Thread(target=_edit, daemon=True).start()

    def _on_delete_static_mask(self):
        """Delete all selected static mask layers."""
        selected = self._get_selected_static_mask_indices()
        if not selected:
            self.log("Click a static mask row to select it, then click Delete")
            return
        mgr = self._static_mask_manager
        if mgr is None:
            return
        # Remove in reverse order so indices don't shift
        names = [mgr.layers[i].name for i in selected]
        for i in sorted(selected, reverse=True):
            mgr.remove_layer(i)
        self._refresh_static_mask_list()
        self.log(f"Deleted static mask(s): {', '.join(names)}")

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

        # Initialize static mask manager for this dataset
        expected_dir = Path(output_path) / "static_masks"
        if (self._static_mask_manager is None
                or self._static_mask_manager.static_dir != expected_dir):
            if Path(output_path).is_dir():
                self._init_static_mask_manager(output_path)

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

        # Static mask overlay paths
        if self._static_mask_manager:
            config_kwargs["static_mask_paths"] = self._static_mask_manager.get_enabled_paths()

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
                    self.log(f"Processing directory with SAM3 unified video: {inp}")
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

            # Record activity for Recent Activity view
            self.record_activity(
                operation="mask",
                input_path=input_path,
                output_path=output_path,
                details={
                    "image_count": stats.get("processed_images", 0),
                    "model": model_str,
                    "geometry": self.geometry_var.get(),
                },
            )

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

                    # Record activity for Recent Activity view
                    self.record_activity(
                        operation="mask",
                        input_path=item.folder_path,
                        output_path=str(masks_dir),
                        details={
                            "image_count": done_count,
                            "model": model_str,
                        },
                    )
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
        ctk.CTkLabel(row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.masks_entry = ctk.CTkEntry(row, placeholder_text="Directory containing mask files")
        self.masks_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=lambda: self._browse_dir_into(self.masks_entry)).pack(side="left")

        row2 = ctk.CTkFrame(dsc, fg_color="transparent")
        row2.pack(fill="x", pady=2)
        ctk.CTkLabel(row2, text="Images:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.images_entry = ctk.CTkEntry(row2, placeholder_text="Directory containing source images")
        self.images_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(row2, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=lambda: self._browse_dir_into(self.images_entry)).pack(side="left")

        btn_row = ctk.CTkFrame(dsc, fg_color="transparent")
        btn_row.pack(fill="x", pady=(2, 0))
        ctk.CTkButton(btn_row, text="Load Masks", width=110,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12),
                      command=self._load_review).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Auto-detect from Output", width=170,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12),
                      command=self._auto_detect_review_paths).pack(side="left")

        # ── Filter & Sort ──
        fs = _Section(container, "Filter & Sort")
        fs.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fsc = fs.content

        fs_row = ctk.CTkFrame(fsc, fg_color="transparent")
        fs_row.pack(fill="x", pady=2)
        ctk.CTkLabel(fs_row, text="Sort:").pack(side="left", padx=(0, 2))
        self.sort_var = ctk.StringVar(value="Filename")
        ctk.CTkOptionMenu(fs_row, variable=self.sort_var,
                          values=["Filename", "Confidence", "Quality", "Area %"],
                          command=lambda _: self._on_filter_change(), width=110).pack(side="left", padx=2)

        ctk.CTkLabel(fs_row, text="Filter:").pack(side="left", padx=(8, 2))
        self.filter_var = ctk.StringVar(value="All")
        ctk.CTkSegmentedButton(
            fs_row, values=["All", "Needs Review", "Poor", "Unreviewed"],
            variable=self.filter_var, command=lambda _: self._on_filter_change()
        ).pack(side="left", padx=2)

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
        ctk.CTkButton(action_row, text="Edit (OpenCV)", command=self._on_edit,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12), width=110).pack(side="left", padx=(0, 4))
        for text, cmd, color in [
            ("Accept", self._on_accept, COLOR_ACTION_PRIMARY),
            ("Reject", self._on_reject, COLOR_ACTION_DANGER),
            ("Skip", self._on_skip, COLOR_ACTION_MUTED),
        ]:
            ctk.CTkButton(action_row, text=text, command=cmd, fg_color=color,
                          font=ctk.CTkFont(size=12),
                          width=75).pack(side="left", padx=3)

        # ── Batch Actions ──
        ba = _Section(container, "Batch")
        ba.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        bac = ba.content

        ctk.CTkButton(bac, text="Accept All Good", command=self._on_accept_all_good,
                      fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                      font=ctk.CTkFont(size=12), width=140).pack(side="left", padx=4, pady=2)
        ctk.CTkButton(bac, text="Hide Done", command=self._on_hide_done,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12), width=100).pack(side="left", padx=4, pady=2)

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

def _show_crash_dialog(exc_type, exc_value, exc_tb):
    """Show a tkinter error dialog when the app crashes fatally.

    Works even under pythonw.exe where there is no console.  Falls back
    silently if even tkinter is broken.
    """
    import traceback as _tb
    log_path = Path.home() / ".reconstruction_zone" / "crash.log"
    detail = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        root = _tk.Tk()
        root.withdraw()
        _mb.showerror(
            "Reconstruction Zone — Fatal Error",
            f"The application encountered an unexpected error and needs to close.\n\n"
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"A detailed crash log has been written to:\n{log_path}",
        )
        root.destroy()
    except Exception:
        pass  # if tkinter itself is broken, nothing we can do


def _check_environment():
    """Run lightweight startup checks and return a list of warnings."""
    warnings = []

    # CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA is not available — GPU acceleration is disabled. Processing will be significantly slower.")
    except ImportError:
        warnings.append("PyTorch is not installed. Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")

    # ffmpeg / ffprobe
    if not shutil.which("ffmpeg"):
        warnings.append("ffmpeg not found on PATH — video extraction features will not work.")
    if not shutil.which("ffprobe"):
        warnings.append("ffprobe not found on PATH — video analysis features will not work.")

    return warnings


def main():
    try:
        app = ReconstructionZone()

        # Show startup warnings in the log console (non-blocking)
        env_warnings = _check_environment()
        for w in env_warnings:
            app.log(f"WARNING: {w}")

        app.mainloop()
    except Exception:
        # Write to crash log (app_infra's excepthook may have already done this,
        # but ensure it happens even if the app failed before infra init)
        import traceback
        log_dir = Path.home() / ".reconstruction_zone"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "crash.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nFATAL CRASH: {__import__('datetime').datetime.now().isoformat()}\n{'='*60}\n")
            traceback.print_exc(file=f)

        _show_crash_dialog(*sys.exc_info())
        raise


if __name__ == "__main__":
    main()

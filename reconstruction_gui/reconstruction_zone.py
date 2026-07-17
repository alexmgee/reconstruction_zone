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
import copy
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
        COCO_CLASSES,
        COCO_NAME_TO_ID,
        ImageGeometry,
        MaskConfig,
        MaskingPipeline,
        SegmentationModel,
    )
    return MaskingPipeline, MaskConfig, SegmentationModel, ImageGeometry, COCO_CLASSES, COCO_NAME_TO_ID


def _import_review():
    from review_gui import (
        load_overlay_thumbnail, compute_mask_area_percent, REVIEW_COLORS,
    )
    from review_status import ReviewStatusManager, MaskStatus
    return (load_overlay_thumbnail, compute_mask_area_percent,
            REVIEW_COLORS, ReviewStatusManager, MaskStatus)



# ──────────────────────────────────────────────────────────────────────
# Shared widgets & infrastructure (extracted to separate modules)
# ──────────────────────────────────────────────────────────────────────

from _version import __version__  # noqa: E402
from widgets import (  # noqa: E402
    Section as _Section, CollapsibleSection as _CollapsibleSection, slider_row, Tooltip,
    COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
    COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
    COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
    COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
    COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    LABEL_FIELD_WIDTH, BROWSE_BUTTON_WIDTH,
)
from app_infra import AppInfrastructure  # noqa: E402
from tabs.adjust_tab import build_adjust_tab  # noqa: E402
from tabs.alignment_tab import build_alignment_tab  # noqa: E402
from tabs.source_tab import build_source_tab  # noqa: E402
from tabs.gaps_tab import build_gaps_tab  # noqa: E402
from tabs.projects_tab import build_projects_tab  # noqa: E402


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
        self._zoom_debounce_id = None  # after() ID for scroll-zoom debounce
        # Layered review state (lazy, populated when Layered mode is loaded)
        self._layer_dirs: Dict[str, Path] = {}                  # name → directory
        self._layer_colors: Dict[str, tuple] = {}               # name → (b_delta, g_delta, r_delta)
        self._layer_visible: Dict[str, ctk.BooleanVar] = {}     # name → visibility toggle
        self._layer_status_mgrs: Dict[str, Any] = {}            # name → ReviewStatusManager
        self._active_layer_var = ctk.StringVar(value="")
        # Per-layer manual-delete state (two-click confirm)
        self._layer_delete_btns: Dict[str, Any] = {}            # name → × button widget
        self._delete_pending_layer: Optional[str] = None        # name in confirm-window, or None
        self._delete_pending_after_id = None                    # after() id for the 3s revert timer
        # Auto-delete checkbox (above the Merge button). Default OFF preserves
        # the re-editable design; users opt in for consume-after-merge.
        self._auto_delete_merged_var = ctk.BooleanVar(value=False)
        # Persistent OpenCV editor state (subprocess-based)
        self._editor_proc = None
        self._editor_cmd_file = None
        self._editor_signal_file = None
        self._editor_poll_id = None
        # Shared MaskingPipeline cache. All call sites go through
        # ``_get_pipeline`` so the SAM3 image model is loaded once and reused.
        # ``_pipeline_lock`` guards both construction (get-or-build) and
        # inference (process_image / predict_inst) to serialize GPU access.
        self._pipeline_cached = None
        self._pipeline_cached_key = None
        self._pipeline_lock = threading.Lock()

        # SAM3 video click session state
        self._click_session = None           # SAM3VideoClickSession or None
        self._click_state = "idle"           # idle | loading | ready | predicting | propagating | error
        self._click_current_frame_idx = 0    # currently displayed frame index
        self._click_current_frame_pil = None # fit-scaled display PIL of current frame
        self._click_current_display_size = None  # (H, W) of display PIL
        self._click_preview_mask = None      # uint8 0/1 mask from last add_prompt

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
        backend_preference: Optional[str] = None,
    ):
        from reconstruction_gui.colmap_runner import ColmapRunner

        normalized = self._normalize_alignment_engine_name(engine_name)
        resolved_binary = (binary_path or "").strip() or self.get_alignment_binary_path(normalized)
        if not resolved_binary:
            status = self.get_alignment_binary_info(normalized)
            detail = status.get("error", "No binary configured")
            raise RuntimeError(f"{normalized} binary is unavailable: {detail}")

        resolved_backend = (
            (backend_preference or "").strip().lower()
            or str(self._prefs.get("alignment_colmap_backend", "auto")).strip().lower()
            or "auto"
        )
        return ColmapRunner(
            binary_path=resolved_binary,
            camera_model=camera_model,
            workspace_root=workspace_root,
            engine_name=normalized,
            backend_preference=resolved_backend,
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
        if hasattr(self, "alignment_feature_type_var") and self._prefs.get("alignment_feature_type"):
            self.alignment_feature_type_var.set(self._prefs["alignment_feature_type"])
        if hasattr(self, "alignment_matcher_type_var") and self._prefs.get("alignment_matcher_type"):
            self.alignment_matcher_type_var.set(self._prefs["alignment_matcher_type"])
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
        for attr, pref_key in [
            ("alignment_prior_std_x_entry", "alignment_prior_std_x"),
            ("alignment_prior_std_y_entry", "alignment_prior_std_y"),
            ("alignment_prior_std_z_entry", "alignment_prior_std_z"),
        ]:
            if hasattr(self, attr) and pref_key in self._prefs:
                val = self._prefs[pref_key]
                if val not in ("", None):
                    getattr(self, attr).delete(0, "end")
                    getattr(self, attr).insert(0, str(val))
        if hasattr(self, "alignment_prior_overwrite_var") and "alignment_prior_overwrite" in self._prefs:
            self.alignment_prior_overwrite_var.set(bool(self._prefs["alignment_prior_overwrite"]))
        if hasattr(self, "alignment_deterministic_var") and "alignment_deterministic" in self._prefs:
            self.alignment_deterministic_var.set(bool(self._prefs["alignment_deterministic"]))
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
        # Load shared static mask library (fixed location next to the GUI
        # module — see static_masks.LIBRARY_DIR).  Independent of which
        # dataset is open.
        self._init_static_mask_manager()

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
        # Release the cached pipeline and any click-PVS session (frees GPU mem).
        self._release_pipeline()
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
        self.tabs.add("Adjust")
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
        build_adjust_tab(self, self.tabs.tab("Adjust"))
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

        ctk.CTkLabel(row2b, text="Input file types:").pack(side="left", padx=(5, 2))
        self.pattern_var = ctk.StringVar(value="*.jpg *.png")
        ctk.CTkEntry(row2b, textvariable=self.pattern_var, width=80).pack(side="left", padx=(0, 12))

        self.output_format_var = ctk.StringVar(value="png")
        ctk.CTkLabel(row2b, text="Output format:").pack(side="left", padx=(0, 2))
        _fmt = ctk.CTkOptionMenu(row2b, variable=self.output_format_var,
                                 values=["png", "jpg", "npy"], width=65)
        _fmt.pack(side="left", padx=(0, 0))
        Tooltip(_fmt, "Mask output format:\n"
                      "  png — lossless, standard (recommended)\n"
                      "  jpg — lossy, smaller files\n"
                      "  npy — numpy array for programmatic use")

        # ── Model ──
        sam3_mode_row = ctk.CTkFrame(core, fg_color="transparent")
        sam3_mode_row.pack(fill="x", padx=6, pady=(8, 3))
        ctk.CTkLabel(sam3_mode_row, text="Mode:").pack(side="left", padx=(0, 2))
        self.temporal_tracking_var = ctk.BooleanVar(value=False)
        self._sam3_mode_btn = ctk.CTkSegmentedButton(
            sam3_mode_row, values=["Per-frame", "Tracked"],
            command=self._on_sam3_mode_change,
        )
        self._sam3_mode_btn.set("Per-frame")
        self._sam3_mode_btn.pack(side="left", padx=4)
        _mode_hint = ctk.CTkLabel(sam3_mode_row,
                     text="Per-frame = detect each image; Tracked = SAM3 video tracking + full pipeline",
                     font=("Consolas", 10), text_color="#9ca3af")
        _mode_hint.pack(side="left", padx=(6, 0))
        Tooltip(_mode_hint,
                "Per-frame -- detect objects independently on each image. All models.\n"
                "Tracked -- SAM3 detects on one keyframe, tracks across all frames,\n"
                "  then runs full post-processing (shadow, refine, matting). Requires SAM3.")

        r1 = ctk.CTkFrame(core, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r1, text="Model:").pack(side="left", padx=(0, 2))
        try:
            from prep360.distribution import is_gumroad as _is_gumroad_check
        except ImportError:
            def _is_gumroad_check(): return False
        _all_models = ["auto", "sam3", "rfdetr", "yolo26", "fastsam"]
        if _is_gumroad_check():
            _model_values = [m for m in _all_models if m not in ("yolo26", "fastsam")]
        else:
            _model_values = _all_models
        self.model_var = ctk.StringVar(value="auto")
        _model_menu = ctk.CTkOptionMenu(r1, variable=self.model_var,
                          values=_model_values,
                          command=self._on_model_change,
                          width=95)
        _model_menu.pack(side="left", padx=2)
        Tooltip(_model_menu,
                "auto — picks best available (SAM3 > RF-DETR > YOLO)\n"
                "sam3 — text-prompted (any object). 848M params, slowest.\n"
                "rfdetr — transformer, 80 COCO classes. Good ensemble partner.\n"
                "yolo26 — CNN, 80 COCO classes. Fastest for known objects.\n"
                "fastsam — legacy, segments everything unfiltered.")

        self._yolo_size_label = ctk.CTkLabel(r1, text="YOLO size:")
        self._yolo_size_label.pack(side="left", padx=(12, 2))
        self.yolo_size_var = ctk.StringVar(value="n")
        self._yolo_size_menu = ctk.CTkOptionMenu(r1, variable=self.yolo_size_var,
                          values=["n", "s", "m", "l", "x"], width=55)
        self._yolo_size_menu.pack(side="left", padx=2)

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
            r2, placeholder_text="(optional) objects to mask out, e.g. person, tripod")
        self.remove_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        r3 = ctk.CTkFrame(core, fg_color="transparent")
        r3.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(r3, text="Keep:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.keep_prompts_entry = ctk.CTkEntry(
            r3, placeholder_text="(optional) objects to protect from masking")
        self.keep_prompts_entry.pack(side="left", fill="x", expand=True, padx=5)

        self._mp_wrapper = mp_wrapper = ctk.CTkFrame(core, fg_color="transparent")
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
        self.merge_existing_var = ctk.BooleanVar(value=False)
        _me = ctk.CTkCheckBox(out_row, text="Merge with existing",
                              variable=self.merge_existing_var, width=0)
        _me.pack(side="left", padx=(16, 0))
        Tooltip(_me, "Union new detections into existing mask files\n"
                     "instead of overwriting. Use for additive passes.")

        # Mutual exclusion: Skip and Merge can't both be on
        def _on_skip_toggle(*_):
            if self.skip_existing_var.get() and self.merge_existing_var.get():
                self.merge_existing_var.set(False)
        def _on_merge_toggle(*_):
            if self.merge_existing_var.get() and self.skip_existing_var.get():
                self.skip_existing_var.set(False)
        self.skip_existing_var.trace_add("write", _on_skip_toggle)
        self.merge_existing_var.trace_add("write", _on_merge_toggle)

        self.review_folder_var = ctk.BooleanVar(value=False)
        _rf = ctk.CTkCheckBox(out_row, text="Create review folder",
                              variable=self.review_folder_var, width=0)
        _rf.pack(side="left", padx=(16, 0))
        Tooltip(_rf, "Also write review preview JPGs to a review/ subfolder\n"
                     "alongside the masks. Output remains the masks folder.")
        self.review_rejects_var = ctk.BooleanVar(value=True)
        _rr = ctk.CTkCheckBox(out_row, text="Include rejects",
                              variable=self.review_rejects_var, width=0)
        _rr.pack(side="left", padx=(16, 0))
        Tooltip(_rr, "Creates a mask for every image regardless of confidence score")
        self.inpaint_var = ctk.BooleanVar(value=False)
        _ip = ctk.CTkCheckBox(out_row, text="Inpaint masked",
                              variable=self.inpaint_var, width=0)
        _ip.pack(side="left", padx=(16, 0))
        Tooltip(_ip, "Fill masked regions with plausible background texture\n"
                     "so 3DGS gets gradient signal instead of black void")

        # Layer name row — separates this masking pass into a named layer
        layer_row = ctk.CTkFrame(core, fg_color="transparent")
        layer_row.pack(fill="x", padx=6, pady=(0, 3))
        ctk.CTkLabel(layer_row, text="Layer:", width=LABEL_FIELD_WIDTH,
                     anchor="e").pack(side="left")
        self.layer_name_var = ctk.StringVar(value="")
        _le = ctk.CTkEntry(layer_row, textvariable=self.layer_name_var,
                           placeholder_text="(optional) people, shadows, tripods")
        _le.pack(side="left", fill="x", expand=True, padx=4)
        Tooltip(_le, "Name this masking pass. Saves to layers/{name}/ inside the masks folder.\n"
                     "Leave empty for standard mask output.")

        # ==============================================================
        #  STATIC MASKS (user-authored persistent overlays)
        # ==============================================================
        sm_sec = _CollapsibleSection(scroll, "Static Masks",
                                     subtitle="persistent overlays applied to every frame",
                                     expanded=False)
        sm_sec.pack(fill="x", pady=(0, 6), padx=4)
        sm = sm_sec.content

        # Layer list — created but NOT packed initially. CTkFrame defaults to
        # height=200 when empty, which would leave a tall gap above the
        # buttons. _refresh_static_mask_list() packs it (before sm_btns) only
        # when there are actual layers, and pack_forget()s it when the list
        # empties back out.
        self._static_mask_list = ctk.CTkFrame(sm, fg_color="transparent")

        # Buttons row
        sm_btns = ctk.CTkFrame(sm, fg_color="transparent")
        sm_btns.pack(fill="x", padx=6, pady=(0, 4))
        self._sm_btns_row = sm_btns  # ref for pack(before=...) on refresh

        ctk.CTkButton(sm_btns, text="New", width=70, height=28,
                       fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_new_static_mask).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_btns, text="Import", width=70, height=28,
                       fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_import_static_mask).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_btns, text="Edit", width=70, height=28,
                       fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_edit_static_mask).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_btns, text="Delete", width=70, height=28,
                       fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_delete_static_mask).pack(side="left")

        # Save mode toggle: OFF (default) writes new masks into the shared
        # library; ON pops a Save As dialog so the PNG can live anywhere on
        # disk while still being indexed in the library.
        sm_save_row = ctk.CTkFrame(sm, fg_color="transparent")
        sm_save_row.pack(fill="x", padx=6, pady=(2, 4))
        self.sm_save_elsewhere_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(sm_save_row, text="Save new mask to a chosen location",
                        variable=self.sm_save_elsewhere_var,
                        font=ctk.CTkFont(size=12)).pack(side="left")

        # Stamp row: pick a target dir, then OR the enabled layers onto every
        # mask file in it. Lives inline (no popup) so it matches the rest of
        # the Mask tab — the entry is the confirm.
        sm_stamp = ctk.CTkFrame(sm, fg_color="transparent")
        sm_stamp.pack(fill="x", padx=6, pady=(0, 4))
        ctk.CTkLabel(sm_stamp, text="Output Dir:", anchor="w",
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 4))
        self.sm_stamp_entry = ctk.CTkEntry(
            sm_stamp,
            placeholder_text="folder of mask files to stamp (defaults to main Output)",
            font=ctk.CTkFont(size=12),
        )
        self.sm_stamp_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(sm_stamp, text="...", width=BROWSE_BUTTON_WIDTH, height=28,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12),
                      command=self._browse_stamp_dir).pack(side="left", padx=(0, 4))
        ctk.CTkButton(sm_stamp, text="Stamp to Output", width=130, height=28,
                       fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
                       font=ctk.CTkFont(size=12),
                       command=self._on_stamp_static_mask).pack(side="left")

        self._static_mask_manager = None  # Initialized when input dir is set
        self._sm_layer_widgets = []       # List of (frame, checkbox_var, name_label) per layer
        self._sm_selected_index = None    # Currently selected layer for edit/delete

        # ==============================================================
        #  DETECTION & REFINEMENT
        # ==============================================================
        self._detect_sec = _CollapsibleSection(scroll, "Detection & Refinement", expanded=False)
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
        #  INTERACTIVE CLICK MODE (PVS)
        # ==============================================================
        click_sec = _CollapsibleSection(scroll, "Interactive Click Mode",
                                        subtitle="left = include, right = exclude",
                                        expanded=False)
        click_sec.pack(fill="x", pady=(0, 6), padx=4)
        clk = click_sec.content

        # Row 1: activate checkbox
        clk_row1 = ctk.CTkFrame(clk, fg_color="transparent")
        clk_row1.pack(fill="x", padx=6, pady=3)
        self.click_mode_var = ctk.BooleanVar(value=False)
        _click_cb = ctk.CTkCheckBox(clk_row1, text="Activate video click mode",
                                    variable=self.click_mode_var,
                                    command=self._on_click_mode_toggle, width=0)
        _click_cb.pack(side="left")
        Tooltip(_click_cb,
                "Start a SAM3 video session on the input directory.\n"
                "Click on any frame to segment, then propagate\n"
                "masks across all frames. Requires SAM3 model.")
        ctk.CTkLabel(clk_row1, text="requires SAM3",
                     font=("Consolas", 10), text_color="#888888").pack(
            side="left", padx=(10, 0))

        # Row 2: status label
        clk_row2 = ctk.CTkFrame(clk, fg_color="transparent")
        clk_row2.pack(fill="x", padx=6, pady=3)
        self._click_status_label = ctk.CTkLabel(
            clk_row2, text="idle. Activate to start a video session.",
            anchor="w", font=("Consolas", 11), text_color=COLOR_TEXT_MUTED,
        )
        self._click_status_label.pack(side="left", fill="x", expand=True)

        # Row 3: frame navigation
        clk_nav = ctk.CTkFrame(clk, fg_color="transparent")
        clk_nav.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(clk_nav, text="Frame:", font=("Consolas", 11),
                     width=45, anchor="e").pack(side="left")
        self._click_frame_prev_btn = ctk.CTkButton(
            clk_nav, text="\u25C0", width=28, height=26,
            fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
            font=ctk.CTkFont(size=11),
            command=lambda: self._click_navigate_delta(-1), state="disabled",
        )
        self._click_frame_prev_btn.pack(side="left", padx=(4, 2))
        self._click_frame_next_btn = ctk.CTkButton(
            clk_nav, text="\u25B6", width=28, height=26,
            fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
            font=ctk.CTkFont(size=11),
            command=lambda: self._click_navigate_delta(1), state="disabled",
        )
        self._click_frame_next_btn.pack(side="left", padx=(0, 4))
        self._click_frame_idx_var = ctk.StringVar(value="0")
        self._click_frame_idx_entry = ctk.CTkEntry(
            clk_nav, textvariable=self._click_frame_idx_var,
            width=50, height=26, font=("Consolas", 11), justify="center",
            state="disabled",
        )
        self._click_frame_idx_entry.pack(side="left")
        self._click_frame_idx_entry.bind(
            "<Return>", lambda e: self._click_navigate_to_entry())
        self._click_frame_total_label = ctk.CTkLabel(
            clk_nav, text="/0", font=("Consolas", 11), text_color=COLOR_TEXT_DIM,
        )
        self._click_frame_total_label.pack(side="left", padx=(2, 6))
        self._click_frame_slider = ctk.CTkSlider(
            clk_nav, from_=0, to=1, number_of_steps=1,
            command=self._on_click_slider, state="disabled",
            height=16, width=120,
        )
        self._click_frame_slider.pack(side="left", fill="x", expand=True)

        # Row 4: clear buttons + click count
        clk_row4 = ctk.CTkFrame(clk, fg_color="transparent")
        clk_row4.pack(fill="x", padx=6, pady=(3, 3))
        self._click_clear_frame_btn = ctk.CTkButton(
            clk_row4, text="Clear frame", width=85,
            fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
            font=ctk.CTkFont(size=12),
            command=self._on_click_clear_frame, state="disabled",
        )
        self._click_clear_frame_btn.pack(side="left")
        self._click_clear_all_btn = ctk.CTkButton(
            clk_row4, text="Clear all", width=75,
            fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
            font=ctk.CTkFont(size=12),
            command=self._on_click_clear_all, state="disabled",
        )
        self._click_clear_all_btn.pack(side="left", padx=(6, 0))
        self._click_count_label = ctk.CTkLabel(
            clk_row4, text="+0  -0",
            font=("Consolas", 11, "bold"), text_color=COLOR_TEXT_DIM,
        )
        self._click_count_label.pack(side="right")

        # Row 5: prompted frames summary
        clk_row5 = ctk.CTkFrame(clk, fg_color="transparent")
        clk_row5.pack(fill="x", padx=6, pady=(0, 3))
        self._click_prompted_label = ctk.CTkLabel(
            clk_row5, text="No prompted frames",
            anchor="w", font=("Consolas", 10), text_color=COLOR_TEXT_DIM,
        )
        self._click_prompted_label.pack(side="left", fill="x", expand=True)

        # Row 6: propagate + cancel + progress
        clk_row6 = ctk.CTkFrame(clk, fg_color="transparent")
        clk_row6.pack(fill="x", padx=6, pady=(0, 3))
        self._click_propagate_btn = ctk.CTkButton(
            clk_row6, text="Propagate All \u25B6", width=120,
            fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
            font=ctk.CTkFont(size=12),
            command=self._on_click_propagate, state="disabled",
        )
        self._click_propagate_btn.pack(side="left")
        self._click_cancel_btn = ctk.CTkButton(
            clk_row6, text="Cancel", width=70,
            fg_color=COLOR_ACTION_DANGER, hover_color="#c0392b",
            font=ctk.CTkFont(size=12),
            command=self._on_click_cancel_propagation,
        )
        # Cancel button hidden by default, shown during propagation

        clk_progress_row = ctk.CTkFrame(clk, fg_color="transparent")
        clk_progress_row.pack(fill="x", padx=6, pady=(0, 6))
        self._click_progress_bar = ctk.CTkProgressBar(
            clk_progress_row, height=10, width=200,
        )
        self._click_progress_bar.pack(side="left", fill="x", expand=True)
        self._click_progress_bar.set(0)
        self._click_progress_label = ctk.CTkLabel(
            clk_progress_row, text="",
            font=("Consolas", 10), text_color=COLOR_TEXT_DIM,
        )
        self._click_progress_label.pack(side="left", padx=(6, 0))

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

        self.num_workers_var = ctk.StringVar(value="4")
        _nw = ctk.CTkOptionMenu(mq_ctrl, variable=self.num_workers_var,
                                values=["1", "2", "4", "8"], width=55)
        _nw.pack(side="right", padx=(0, 0))
        ctk.CTkLabel(mq_ctrl, text="Workers:", font=ctk.CTkFont(size=11)).pack(
            side="right", padx=(8, 2))
        Tooltip(_nw, "Number of parallel threads for batch processing.\n"
                     "Higher = faster but uses more GPU memory.\n"
                     "Set to 1 for sequential processing.")

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
        """Update state for Per-frame / Tracked mode selector."""
        self.temporal_tracking_var.set(value == "Tracked")

        if value == "Per-frame":
            for frame in self._unified_disable_frames:
                self._set_widgets_state(frame, "normal")
            for sec in (self._detect_sec, self._post_sec):
                self._set_widgets_state(sec.content, "normal")
            for entry in (self.remove_prompts_entry, self.keep_prompts_entry):
                entry.master.configure(fg_color="transparent")

        elif value == "Tracked":
            # All controls enabled (tracking feeds into full pipeline),
            # glow on prompts since they drive the tracker
            for frame in self._unified_disable_frames:
                self._set_widgets_state(frame, "normal")
            for sec in (self._detect_sec, self._post_sec):
                self._set_widgets_state(sec.content, "normal")
            for entry in (self.remove_prompts_entry, self.keep_prompts_entry):
                entry.master.configure(fg_color="#1a5276")

    def _on_model_change(self, value):
        """Grey out controls irrelevant to the selected model.

        Rules:
        - YOLO size: only relevant for yolo26
        - Multi-pass SAM3: only relevant for sam3
        - auto: everything enabled (actual model unknown until runtime)
        """
        yolo_state = "normal" if value in ("auto", "yolo26") else "disabled"
        for w in (self._yolo_size_label, self._yolo_size_menu):
            w.configure(state=yolo_state)
            if isinstance(w, ctk.CTkOptionMenu):
                if yolo_state == "disabled":
                    w._original_fg = w.cget("fg_color")
                    w._original_btn = w.cget("button_color")
                    w.configure(fg_color="#555555", button_color="#555555")
                elif hasattr(w, "_original_fg"):
                    w.configure(fg_color=w._original_fg, button_color=w._original_btn)

        mp_state = "normal" if value in ("auto", "sam3") else "disabled"
        self._set_widgets_state(self._mp_wrapper, mp_state)

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
        # Pan offsets applied via place() on the overlay/mask labels. Reset
        # on image change; updated by mouse-wheel zoom-to-cursor and middle-
        # mouse drag.
        self._preview_pan_x = 0
        self._preview_pan_y = 0
        self._pan_drag_anchor = None
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

        # Cached-pipeline indicator + free button. Only visible when a model
        # is loaded in GPU memory — doubles as a status display ("sam3 cached
        # on cuda") and a one-click way to free that memory if the user wants
        # to launch something else GPU-heavy.
        self._preview_cache_btn = ctk.CTkButton(
            header, text="", width=110, height=24,
            fg_color="transparent", hover_color=("gray75", "gray25"),
            text_color="#9ca3af", font=("Consolas", 10),
            command=self._clear_preview_pipeline,
        )
        # Not packed initially — _refresh_preview_cache_indicator controls visibility.
        Tooltip(self._preview_cache_btn,
                "Click to free GPU memory.\nNext Preview will reload the model.")

        # Row 1: Image display area. Regular frame (not scrollable) — labels
        # are placed via place() at center+pan offsets so zoom keeps the
        # image visually centered, and middle-mouse drag pans when zoomed in.
        self._preview_image_frame = ctk.CTkFrame(panel, fg_color="#1e1e1e")
        self._preview_image_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        self._process_overlay_label = ctk.CTkLabel(self._preview_image_frame, text="")
        self._process_overlay_label.place(
            relx=0.5, rely=0.5, anchor="center",
            x=self._preview_pan_x, y=self._preview_pan_y,
        )
        # SAM3 click-PVS bindings (Phase 2). Handlers no-op unless click mode
        # is "ready" AND the Mask tab is current; Review tab is unaffected.
        self._process_overlay_label.bind(
            "<Button-1>", lambda e: self._on_click_overlay(e, positive=True))
        self._process_overlay_label.bind(
            "<Button-3>", lambda e: self._on_click_overlay(e, positive=False))
        # Mouse wheel = zoom-to-cursor. Middle-mouse drag = pan. Bind on the
        # label AND the surrounding frame so empty space around the image
        # also reacts.
        for _w in (self._process_overlay_label, self._preview_image_frame):
            _w.bind("<MouseWheel>", self._on_preview_wheel_zoom)
            _w.bind("<ButtonPress-2>", self._on_preview_pan_start)
            _w.bind("<B2-Motion>", self._on_preview_pan_motion)
            _w.bind("<ButtonRelease-2>", self._on_preview_pan_end)

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

        # Console header row: label on left, controls on right
        console_header = ctk.CTkFrame(console_frame, fg_color="transparent")
        console_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(2, 0))
        ctk.CTkLabel(console_header, text="Console", font=("Consolas", 10),
                     anchor="w").pack(side="left")
        ctk.CTkButton(
            console_header, text="Clear", width=50, height=20,
            fg_color="transparent", hover_color=("gray75", "gray25"),
            text_color="#9ca3af", font=("Consolas", 9),
            command=self._clear_console,
        ).pack(side="right", padx=(0, 4))
        ctk.CTkButton(
            console_header, text="↓ End", width=50, height=20,
            fg_color="transparent", hover_color=("gray75", "gray25"),
            text_color="#9ca3af", font=("Consolas", 9),
            command=self._console_scroll_to_end,
        ).pack(side="right", padx=(0, 4))

        self.log_textbox = ctk.CTkTextbox(console_frame, font=("Consolas", 10), height=80)
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 4))

    def _clear_console(self):
        """Empty the console textbox."""
        if hasattr(self, "log_textbox"):
            try:
                self.log_textbox.delete("1.0", "end")
            except Exception:
                pass

    def _console_scroll_to_end(self):
        """Force the console textbox to scroll to the latest entry."""
        if hasattr(self, "log_textbox"):
            try:
                self.log_textbox.see("end")
                self.log_textbox.update_idletasks()
            except Exception:
                pass

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
            self._process_mask_label.place(
                relx=0.5, rely=0.5, anchor="center",
                x=self._preview_pan_x, y=self._preview_pan_y,
            )
            if self._preview_mask_pil is not None:
                zoom = self._zoom_var.get() / 100.0
                ow, oh = self._preview_mask_pil.size
                zw, zh = max(1, int(ow * zoom)), max(1, int(oh * zoom))
                ctk_mask = ctk.CTkImage(light_image=self._preview_mask_pil, size=(zw, zh))
                self._process_mask_label.configure(image=ctk_mask, text="")
                self._process_mask_label._ctk_image = ctk_mask
        else:
            self._process_mask_label.place_forget()

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
        elif active == "Adjust":
            self._preview_mode = "adjust"
            # Sync the shared navigator to the Adjust image list
            from tabs.adjust_tab import _sync_shared_navigator
            _sync_shared_navigator(self)
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

        # Adjust mode: delegate to adjust tab's navigation
        if self._preview_mode == "adjust":
            img_list = getattr(self, '_adjust_image_list', [])
            n = len(img_list)
            if n == 0:
                return
            idx = max(0, min(idx, n - 1))
            self._nav_idx.set(idx)
            self._nav_counter.configure(text=f"{idx + 1} / {n}")
            if idx != self._adjust_nav_idx:
                self._adjust_nav_idx = idx
                from tabs.adjust_tab import _load_current_image
                _load_current_image(self)
            return

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

        # New image → reset pan so the next render is centered.
        self._reset_preview_pan()

        # Click mode: if active, navigate to the image the user selected
        # in the navigator (find its index in the video session frame list).
        if self._click_state == "ready" and self._click_session is not None:
            fname = img_path.name
            try:
                idx = self._click_session.frame_names.index(fname)
                self._click_navigate_to(idx)
            except ValueError:
                pass  # image not in the click session's frame list

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
        """Render the preview pane: composite of all visible layers in their colors.

        Mirrors the thumbnail composite so the big preview matches what the
        user sees in the grid. The "Mask" toggle (top-right) shows the active
        layer's mask in white-on-black.
        """
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        img = cv2.imread(str(pair["image_path"]))
        if img is None:
            return

        h, w = img.shape[:2]
        # winfo_width/height return physical pixels; CTkImage size is logical pixels
        panel_w = self._preview_image_frame.winfo_width() / self._dpi_scale
        panel_total_h = self._preview_panel.winfo_height() / self._dpi_scale
        # Subtract header (~35), nav bar (~35), console (~15% of panel)
        panel_h = panel_total_h * 0.80 - 70
        target_w = max(200, panel_w - 20) if panel_w > 60 else 500
        target_h = max(200, panel_h) if panel_h > 60 else 500
        scale = min(target_w / w, target_h / h)
        pw, ph = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (pw, ph))

        # Composite every visible layer using additive BGR-delta blending — the
        # exact same math as load_multi_layer_thumbnail, so the big preview
        # matches the grid.
        overlay = img_r.copy()
        layer_masks = pair.get("layer_masks", {})
        for name in self._layer_dirs:
            if not (name in self._layer_visible and self._layer_visible[name].get()):
                continue
            mask_path = layer_masks.get(name)
            if mask_path is None or not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_r = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
            masked = mask_r < 128
            b, g, r = self._layer_colors[name]
            overlay[masked, 0] = np.clip(overlay[masked, 0].astype(int) + b, 0, 255).astype(np.uint8)
            overlay[masked, 1] = np.clip(overlay[masked, 1].astype(int) + g, 0, 255).astype(np.uint8)
            overlay[masked, 2] = np.clip(overlay[masked, 2].astype(int) + r, 0, 255).astype(np.uint8)
        overlay_pil = PILImage.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # "Mask" toggle view: render the ACTIVE layer's mask only, white on black
        active_mask_path = pair.get("mask_path")
        mask_pil = None
        if active_mask_path is not None and active_mask_path.exists():
            mask = cv2.imread(str(active_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_r = cv2.resize(mask, (pw, ph))
                mask_pil = PILImage.fromarray(cv2.cvtColor(
                    cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
        if mask_pil is None:
            # Fallback: empty black canvas so the Mask toggle doesn't show stale data
            mask_pil = PILImage.new("RGB", (pw, ph), (0, 0, 0))

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
        mod_str = ""
        if ms.last_edit_modified is True:
            mod_str = "  |  edited"
        elif ms.last_edit_modified is False:
            mod_str = "  |  unchanged"
        # When >1 layer is loaded, surface the active layer so the user knows
        # which one Save / Delete / Skip / Edit will act on.
        active_str = ""
        if len(self._layer_dirs) > 1:
            active = self._active_layer_var.get()
            if active:
                active_str = f"  ·  active: {active}"
        self._review_info_label.configure(
            text=f"{self._selected_stem}{active_str}\n"
                 f"{ms.status}  |  Conf: {ms.confidence:.0%}  |  "
                 f"Area: {ms.area_percent:.1f}%{mod_str}"
        )
        # Update summary
        if self._review_status_mgr:
            summary = self._review_status_mgr.get_summary()
            reviewed = summary.get("reviewed", 0)
            unreviewed = summary.get("unreviewed", 0)
            self._review_summary_label.configure(
                text=f"{reviewed} reviewed | {unreviewed} unreviewed"
            )

    # ── zoom ──

    def _on_zoom_change(self, value):
        """Re-render cached preview images at new zoom level."""
        pct = int(float(value))
        self._zoom_label.configure(text=f"{pct}%")
        self._zoom_render(pct)

    def _zoom_render(self, pct):
        """Create CTkImage at the given zoom percent and push to label."""
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

    def _on_preview_wheel_zoom(self, event):
        """Mouse wheel on the preview area -> zoom-to-cursor with debounce."""
        if event.delta == 0:
            return "break"
        step = 5  # slider granularity (25..300 / 55 steps)
        direction = 1 if event.delta > 0 else -1
        cur = self._zoom_var.get()
        new = max(25, min(300, cur + direction * step))
        if new == cur:
            return "break"

        # Cursor position relative to the preview frame, in logical pixels.
        frame = self._preview_image_frame
        cx = (event.x_root - frame.winfo_rootx()) / self._dpi_scale
        cy = (event.y_root - frame.winfo_rooty()) / self._dpi_scale
        fw = frame.winfo_width() / self._dpi_scale
        fh = frame.winfo_height() / self._dpi_scale

        # Adjust pan so the image-pixel under the cursor stays under the
        # cursor across the zoom change. Scale factor f = new/old.
        f = new / cur
        self._preview_pan_x = (cx - fw / 2) * (1 - f) + f * self._preview_pan_x
        self._preview_pan_y = (cy - fh / 2) * (1 - f) + f * self._preview_pan_y

        self._zoom_var.set(new)
        self._zoom_label.configure(text=f"{new}%")
        # Debounce the expensive CTkImage creation. Cancel any pending render
        # and schedule a new one 30ms out. Pan updates immediately so the
        # position tracks the cursor without waiting for the image rebuild.
        if hasattr(self, "_zoom_debounce_id") and self._zoom_debounce_id is not None:
            self.after_cancel(self._zoom_debounce_id)
        self._zoom_debounce_id = self.after(30, lambda: self._zoom_render(new))
        self._apply_preview_pan()
        return "break"

    def _on_preview_pan_start(self, event):
        """Middle-mouse press: anchor pan drag."""
        self._pan_drag_anchor = (
            event.x_root, event.y_root,
            self._preview_pan_x, self._preview_pan_y,
        )

    def _on_preview_pan_motion(self, event):
        """Middle-mouse drag: shift pan offsets."""
        if self._pan_drag_anchor is None:
            return
        rx, ry, px0, py0 = self._pan_drag_anchor
        # event.x_root is in physical pixels; convert via dpi.
        dx = (event.x_root - rx) / self._dpi_scale
        dy = (event.y_root - ry) / self._dpi_scale
        self._preview_pan_x = px0 + dx
        self._preview_pan_y = py0 + dy
        self._apply_preview_pan()

    def _on_preview_pan_end(self, event):
        """Middle-mouse release."""
        self._pan_drag_anchor = None

    def _apply_preview_pan(self):
        """Push current pan offsets onto the placed labels."""
        try:
            self._process_overlay_label.place_configure(
                x=self._preview_pan_x, y=self._preview_pan_y)
            if self._show_mask_view.get():
                self._process_mask_label.place_configure(
                    x=self._preview_pan_x, y=self._preview_pan_y)
        except Exception:
            pass

    def _reset_preview_pan(self):
        """Reset pan offsets to centered. Called when the image changes."""
        self._preview_pan_x = 0
        self._preview_pan_y = 0
        self._apply_preview_pan()

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

    def _init_static_mask_manager(self):
        """Initialize the static mask manager against the shared library.

        Idempotent: the library is fixed (``static_mask_library/`` next to
        the GUI module), so re-calling this just rebuilds the layer list.
        """
        from static_masks import StaticMaskManager
        if self._static_mask_manager is None:
            self._static_mask_manager = StaticMaskManager()
        self._refresh_static_mask_list()

    def _refresh_static_mask_list(self):
        """Rebuild the layer list UI from the manager's state.

        Empty state collapses to zero height by un-packing the list frame —
        otherwise CTkFrame's default height (~200px) would leave a gap above
        the action buttons.
        """
        for w in self._sm_layer_widgets:
            w["frame"].destroy()
        self._sm_layer_widgets = []

        mgr = self._static_mask_manager
        if mgr is None or not mgr.layers:
            self._static_mask_list.pack_forget()
            return

        # Insert the list above the action buttons row.
        self._static_mask_list.pack(fill="x", padx=6, pady=(2, 0),
                                    before=self._sm_btns_row)

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

        # Library manager is fixed; just make sure it's initialized.
        if self._static_mask_manager is None:
            self._init_static_mask_manager()

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
        """Persist a freshly-painted mask.

        Routing depends on the 'Save new mask to a chosen location' checkbox:
        - Unchecked → write into the shared library (just prompts for a name).
        - Checked  → Save As dialog for the PNG path, then prompt for a
                     display name.  The layer entry stores the absolute path.
        """
        save_elsewhere = bool(getattr(self, "sm_save_elsewhere_var", None)
                              and self.sm_save_elsewhere_var.get())

        if save_elsewhere:
            dest = filedialog.asksaveasfilename(
                title="Save static mask PNG to...",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if not dest:
                self.log("Static mask save cancelled")
                return
            name_dialog = ctk.CTkInputDialog(
                text="Name this static mask layer:",
                title="Static Mask Name",
            )
            name = name_dialog.get_input()
            if not (name and name.strip()):
                self.log("Static mask not saved (no name given)")
                return
            self._static_mask_manager.add_layer_at_path(name.strip(), mask, Path(dest))
            self._refresh_static_mask_list()
            self.log(f"Static mask saved to {dest}")
            return

        dialog = ctk.CTkInputDialog(
            text="Name this static mask layer:",
            title="Save Static Mask",
        )
        name = dialog.get_input()
        if not (name and name.strip()):
            self.log("Static mask not saved (no name given)")
            return
        self._static_mask_manager.add_layer(name.strip(), mask)
        self._refresh_static_mask_list()
        self.log(f"Static mask saved to library: {name.strip()}")

    def _on_import_static_mask(self):
        """Add an existing PNG anywhere on disk to the library index.

        The file is NOT copied — only an absolute-path reference is stored.
        Useful for re-using a mask painted in another session without
        duplicating data on disk.
        """
        if self._static_mask_manager is None:
            self._init_static_mask_manager()
        source = filedialog.askopenfilename(
            title="Import static mask PNG",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not source:
            return

        # Suggest a name from the filename stem; let the user override.
        suggested = Path(source).stem
        name_dialog = ctk.CTkInputDialog(
            text=f"Name this imported layer (default: {suggested}):",
            title="Import Static Mask",
        )
        name = name_dialog.get_input()
        if name is None:
            self.log("Import cancelled")
            return
        name = name.strip() or suggested
        layer = self._static_mask_manager.import_layer(name, Path(source))
        if layer is None:
            self.log(f"Import failed: not a readable mask: {source}")
            return
        self._refresh_static_mask_list()
        self.log(f"Imported static mask '{name}' from {source}")

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
        # Resolve absolute or library-relative filenames uniformly.
        mask_path = mgr._resolve(layer)
        existing = _cv2.imread(str(mask_path), _cv2.IMREAD_GRAYSCALE)
        if existing is None:
            self.log(f"Could not load mask: {mask_path}")
            return
        # Disk convention: black (0) = masked. Internal: 1 = masked.
        existing = (existing < 128).astype(_np.uint8)

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

    def _browse_stamp_dir(self):
        """Browse helper for the Static Masks 'Output Dir' entry."""
        current = self.sm_stamp_entry.get().strip()
        initial = current if current and Path(current).is_dir() else self.output_entry.get().strip()
        path = filedialog.askdirectory(
            title="Select folder of mask files to stamp",
            initialdir=initial if initial and Path(initial).is_dir() else "",
        )
        if path:
            self.sm_stamp_entry.delete(0, "end")
            self.sm_stamp_entry.insert(0, path)

    def _on_stamp_static_mask(self):
        """OR the enabled static-mask layers onto every mask file in the
        Static Masks 'Output Dir' field. Falls back to the main Output dir
        when the field is empty. Skips the segmentation pipeline entirely —
        purely a file I/O pass.
        """
        mgr = self._static_mask_manager
        if mgr is None or not any(l.enabled for l in mgr.layers):  # noqa: E741
            self.log("Stamp: no enabled static mask layers")
            return

        target = self.sm_stamp_entry.get().strip()
        if not target:
            target = self.output_entry.get().strip()
        if not target:
            self.log("Stamp: set Static Masks → Output Dir (or the main Output) first")
            return

        target_path = Path(target)
        if not target_path.is_dir():
            self.log(f"Stamp: not a directory: {target_path}")
            return

        n_files = mgr.count_stampable_targets(target_path)
        if n_files == 0:
            self.log(f"Stamp: no .png/.npy mask files found in {target_path}")
            return

        n_layers = sum(1 for l in mgr.layers if l.enabled)  # noqa: E741
        self.log(f"Stamping {n_layers} layer(s) onto {n_files} file(s) in {target_path}...")

        import threading

        def _worker():
            try:
                stats = mgr.stamp_onto_directory(target_path)
                msg = (
                    f"Stamped {stats['modified']}/{stats['total_examined']} mask(s)"
                )
                if stats["skipped_unreadable"]:
                    msg += f"  skipped {stats['skipped_unreadable']} unreadable"
                if stats["skipped_resize_failed"]:
                    msg += f"  skipped {stats['skipped_resize_failed']} resize-failed"
                self.after(0, lambda: self.log(msg))
            except Exception as e:
                self.after(0, lambda: self.log(f"Stamp failed: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_preview(self):
        # Invalidate cached image list + explicit preview image if the Input
        # field has changed since the last scan. Without this, switching
        # between sibling folders (e.g. front/frames → back/frames) keeps the
        # navigator pointing at stale entries and _resolve_preview_image()
        # returns the old explicit path because the file still exists.
        current_input = self.input_entry.get().strip()
        if current_input != getattr(self, "_image_list_input_path", None):
            self._image_list = []
            self._preview_image_entry.delete(0, "end")
        # Auto-load image list if not yet loaded
        if not self._image_list:
            self._load_image_list()
            self._image_list_input_path = current_input
        img_path = self._resolve_preview_image()
        if img_path is None:
            self.log("Preview: no image found. Set an input path first.")
            return
        if self.is_running:
            self.log("Preview: batch is running, wait for it to finish.")
            return
        # Immediate click feedback — the worker thread will log "loading model..."
        # next, but model init can take several seconds. Without this line the
        # user sees nothing happen for that whole wait.
        self.log(f"Preview: starting on {img_path.name}...")
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

    def _refresh_preview_cache_indicator(self):
        """Show/hide the 'cached pipeline' button in the Preview header."""
        if not hasattr(self, "_preview_cache_btn"):
            return
        if self._pipeline_cached is None:
            self._preview_cache_btn.pack_forget()
            return
        model_str = "model"
        device = ""
        try:
            model_str = self._pipeline_cached.config.model.value
            device = self._pipeline_cached.config.device
        except Exception:
            pass
        label = f"\u2715 {model_str} on {device}" if device else f"\u2715 {model_str}"
        self._preview_cache_btn.configure(text=label)
        self._preview_cache_btn.pack(
            side="right", padx=(0, 4), before=self._preview_collapse_btn)

    def _clear_preview_pipeline(self):
        """Free the cached pipeline (and its GPU memory)."""
        if self._pipeline_cached is None:
            return
        model_label = ""
        try:
            model_label = self._pipeline_cached.config.model.value
        except Exception:
            pass
        self._release_pipeline()
        msg = f"Cleared cached {model_label} pipeline" if model_label else "Cleared cached pipeline"
        self.log(msg)
        self._refresh_preview_cache_indicator()

    def _preview_worker(self, img_path: Path, panel_w: float = 0, panel_h: float = 0):
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage

            config, model_str, geometry = self._build_mask_config()
            auto_select = (model_str == "auto")
            pipeline = self._get_pipeline(config, auto_select=auto_select)

            image = cv2.imread(str(img_path))
            if image is None:
                self.log(f"Preview: failed to load {img_path}")
                return

            self.log(f"Preview: processing {img_path.name}...")
            with self._pipeline_lock:
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

    # ──────────────────────────────────────────────────────────────────
    # SAM3 interactive video click mode
    # ──────────────────────────────────────────────────────────────────

    def _click_set_state(self, state: str, text: str):
        """Update click-mode state, status line, and widget enable/disable."""
        self._click_state = state
        if not hasattr(self, "_click_status_label"):
            return
        color_map = {
            "idle": COLOR_TEXT_MUTED,
            "loading": "#1976D2",
            "ready": "#16a34a",
            "predicting": "#9333ea",
            "propagating": "#1976D2",
            "error": COLOR_ACTION_DANGER,
        }
        self._click_status_label.configure(
            text=text, text_color=color_map.get(state, COLOR_TEXT_MUTED))

        is_ready = state == "ready"
        has_clicks = (self._click_session is not None
                      and bool(self._click_session.prompted_frames))
        nav_state = "normal" if is_ready else "disabled"
        for w in (self._click_frame_prev_btn, self._click_frame_next_btn,
                  self._click_frame_slider):
            w.configure(state=nav_state)
        self._click_frame_idx_entry.configure(state="normal" if is_ready else "disabled")
        self._click_clear_frame_btn.configure(state="normal" if is_ready else "disabled")
        self._click_clear_all_btn.configure(state="normal" if is_ready else "disabled")
        self._click_propagate_btn.configure(
            state="normal" if is_ready and has_clicks else "disabled")

    def _on_click_mode_toggle(self):
        """Checkbox handler: activate or deactivate video click mode."""
        if not self.click_mode_var.get():
            self._click_teardown()
            return

        input_dir = self.input_entry.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            self.click_mode_var.set(False)
            self._click_set_state("idle", "no input directory. Set Input first.")
            return

        # Release batch pipeline to free VRAM for the video predictor
        self._release_pipeline()

        self._click_set_state("loading", "loading video session ...")
        self.log(f"Click mode: starting video session on {input_dir}")
        threading.Thread(
            target=self._click_session_worker,
            args=(Path(input_dir),),
            daemon=True,
        ).start()

    def _click_session_worker(self, input_dir: Path):
        """Worker thread: build video predictor and start session."""
        try:
            from sam3_click_session import SAM3VideoClickSession
            session = SAM3VideoClickSession()
            session.start(input_dir, device="cuda")

            def _on_main():
                self._click_session = session
                n = session.num_frames
                # Configure slider range
                self._click_frame_slider.configure(
                    to=max(0, n - 1),
                    number_of_steps=max(1, n - 1),
                )
                self._click_frame_total_label.configure(text=f"/{n}")
                self._click_navigate_to(0)
                self._click_set_state(
                    "ready", f"ready \u00b7 frame 1/{n}")
                self.log(f"Click mode: video session ready ({n} frames)")

            self.after(0, _on_main)
        except Exception as exc:
            err = str(exc)
            self.log(f"Click mode session error: {err}")
            import traceback
            self.log(traceback.format_exc())
            self.after(0, lambda: (
                self._click_set_state("error", f"session error: {err}"),
                self.click_mode_var.set(False),
            ))

    # ── Frame navigation ──

    def _click_navigate_to(self, frame_idx: int):
        """Load and display a specific frame. Main thread."""
        if self._click_session is None or not self._click_session.is_ready:
            return
        n = self._click_session.num_frames
        frame_idx = max(0, min(frame_idx, n - 1))
        self._click_current_frame_idx = frame_idx

        # Read frame from disk
        import cv2
        from PIL import Image as PILImage
        frame_name = self._click_session.frame_names[frame_idx]
        frame_path = Path(self.input_entry.get().strip()) / frame_name
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            self.log(f"Click nav: failed to read {frame_path}")
            return

        # Fit-scale for display panel
        h, w = bgr.shape[:2]
        panel_w = self._preview_image_frame.winfo_width() / self._dpi_scale
        panel_total_h = self._preview_panel.winfo_height() / self._dpi_scale
        panel_h = panel_total_h * 0.80 - 70
        target_w = max(200, panel_w - 20) if panel_w > 60 else 500
        target_h = max(200, panel_h) if panel_h > 60 else 500
        scale = min(target_w / w, target_h / h)
        pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
        img_r = cv2.resize(bgr, (pw, ph))
        display_pil = PILImage.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))

        self._click_current_frame_pil = display_pil
        self._click_current_display_size = (ph, pw)

        # Update nav widgets
        self._click_frame_idx_var.set(str(frame_idx + 1))
        self._click_frame_slider.set(frame_idx)

        # Render overlay with any existing mask/clicks for this frame
        mask = self._click_session.get_frame_mask(frame_idx)
        self._click_preview_mask = mask
        self._render_click_overlay(mask)
        self._update_click_count()
        self._update_prompted_frames_label()

    def _click_navigate_delta(self, delta: int):
        """Move forward/backward by delta frames."""
        self._click_navigate_to(self._click_current_frame_idx + delta)

    def _click_navigate_to_entry(self):
        """Jump to the frame number typed in the entry field."""
        try:
            idx = int(self._click_frame_idx_var.get()) - 1  # 1-based display
            self._click_navigate_to(idx)
        except ValueError:
            pass

    def _on_click_slider(self, value):
        """Slider moved — navigate to that frame."""
        self._click_navigate_to(int(float(value)))

    # ── Click handler ──

    def _on_click_overlay(self, event, positive: bool):
        """Click handler on the preview label. Main thread."""
        if self._click_state != "ready":
            return
        if self.tabs.get() != "Mask":
            return
        if self._click_session is None or self._click_current_display_size is None:
            return
        zoom = self._zoom_var.get() / 100.0
        if zoom <= 0:
            return
        dh, dw = self._click_current_display_size
        if dh <= 0 or dw <= 0:
            return

        # Map display coords → original pixel coords
        disp_x = (event.x / self._dpi_scale) / zoom
        disp_y = (event.y / self._dpi_scale) / zoom
        if not (0 <= disp_x < dw and 0 <= disp_y < dh):
            return

        oh, ow = self._click_session._orig_hw
        ox = int(disp_x * ow / dw)
        oy = int(disp_y * oh / dh)
        ox = max(0, min(ox, ow - 1))
        oy = max(0, min(oy, oh - 1))

        frame_idx = self._click_current_frame_idx
        self._click_set_state("predicting", "decoding ...")
        threading.Thread(
            target=self._click_decode_worker,
            args=(frame_idx, ox, oy, positive),
            daemon=True,
        ).start()

    def _click_decode_worker(self, frame_idx, ox, oy, positive):
        """Worker thread: add click and decode mask via video session."""
        try:
            with self._pipeline_lock:
                mask, score = self._click_session.add_click(
                    frame_idx, ox, oy, positive)

            def _on_main():
                self._click_preview_mask = mask
                self._render_click_overlay(mask)
                self._update_click_count()
                self._update_prompted_frames_label()
                n = self._click_session.num_clicks_on_frame(frame_idx)
                self._click_set_state(
                    "ready",
                    f"frame {frame_idx + 1} \u00b7 {n} click{'s' if n != 1 else ''}")

            self.after(0, _on_main)
        except Exception as exc:
            err = str(exc)
            self.log(f"Click decode error: {err}")
            self.after(0, lambda: self._click_set_state("error", f"decode error: {err}"))

    # ── Overlay rendering ──

    def _get_click_fisheye_circle(self, width, height):
        """Return the fisheye circle mask if enabled and geometry is fisheye, else None."""
        if not self.fisheye_circle_var.get():
            return None
        if self.geometry_var.get() != "fisheye":
            return None
        try:
            from prep360.core.fisheye_reframer import generate_fisheye_circle_mask
            margin = float(self.fisheye_margin_var.get())
            return generate_fisheye_circle_mask(width, height, margin_percent=margin)
        except Exception:
            return None

    def _render_click_overlay(self, mask_orig):
        """Composite mask + click dots into the preview. Main thread."""
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        if self._click_current_frame_pil is None or self._click_current_display_size is None:
            return
        dh, dw = self._click_current_display_size

        display_rgb = np.asarray(self._click_current_frame_pil).copy()
        display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)

        # Composite: click mask + fisheye circle mask (if applicable)
        composite = mask_orig
        if self._click_session is not None and self._click_session._orig_hw is not None:
            oh, ow = self._click_session._orig_hw
            circle = self._get_click_fisheye_circle(ow, oh)
            if circle is not None:
                if composite is not None:
                    composite = np.maximum(composite, circle)
                else:
                    composite = circle

        if composite is not None:
            mask_disp = cv2.resize(
                composite, (dw, dh), interpolation=cv2.INTER_NEAREST)
            selected = mask_disp > 0
            if np.any(selected):
                display_bgr[selected] = (
                    display_bgr[selected].astype(np.float32) * 0.5
                    + np.array([0, 0, 200], dtype=np.float32) * 0.5
                ).astype(np.uint8)

        # Draw click dots for current frame
        if self._click_session is not None and self._click_session._orig_hw is not None:
            oh, ow = self._click_session._orig_hw
            for pt in self._click_session.get_frame_clicks(self._click_current_frame_idx):
                dx = int(pt["x"] * dw / ow)
                dy = int(pt["y"] * dh / oh)
                color = (0, 200, 0) if pt["label"] == 1 else (0, 0, 200)
                cv2.circle(display_bgr, (dx, dy), 4, color, -1)

        overlay_pil = PILImage.fromarray(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB))
        self._preview_overlay_pil = overlay_pil
        self._on_zoom_change(self._zoom_var.get())

    def _update_click_count(self):
        """Update the +N -N click count label for the current frame."""
        if not hasattr(self, "_click_count_label"):
            return
        if self._click_session is None:
            self._click_count_label.configure(text="+0  -0")
            return
        clicks = self._click_session.get_frame_clicks(self._click_current_frame_idx)
        pos = sum(1 for c in clicks if c["label"] == 1)
        neg = len(clicks) - pos
        self._click_count_label.configure(text=f"+{pos}  -{neg}")

    def _update_prompted_frames_label(self):
        """Update the summary of which frames have clicks."""
        if not hasattr(self, "_click_prompted_label"):
            return
        if self._click_session is None:
            self._click_prompted_label.configure(text="No prompted frames")
            return
        prompted = self._click_session.prompted_frames
        if not prompted:
            self._click_prompted_label.configure(text="No prompted frames")
            return
        indices = sorted(prompted.keys())
        if len(indices) <= 8:
            frame_list = ", ".join(str(i + 1) for i in indices)
        else:
            frame_list = ", ".join(str(i + 1) for i in indices[:6]) + f" ... +{len(indices) - 6} more"
        self._click_prompted_label.configure(
            text=f"Prompted: {frame_list}  ({len(indices)} frame{'s' if len(indices) != 1 else ''})")

    # ── Clear ──

    def _on_click_clear_frame(self):
        """Clear clicks on the current frame only."""
        if self._click_session is None:
            return
        self._click_session.reset_frame(self._click_current_frame_idx)
        self._click_preview_mask = None
        self._render_click_overlay(None)
        self._update_click_count()
        self._update_prompted_frames_label()
        self._click_set_state("ready", f"frame {self._click_current_frame_idx + 1} cleared")

    def _on_click_clear_all(self):
        """Clear all clicks across all frames."""
        if self._click_session is None:
            return
        self._click_session.reset_all()
        self._click_preview_mask = None
        self._render_click_overlay(None)
        self._update_click_count()
        self._update_prompted_frames_label()
        self._click_set_state("ready", "all clicks cleared")

    # ── Propagation ──

    def _on_click_propagate(self):
        """Start propagation across all frames. Main thread."""
        if self._click_session is None:
            return
        output_dir = self.output_entry.get().strip()
        if not output_dir:
            self.log("Click propagate: no output folder set.")
            return

        self.cancel_flag.clear()
        self._click_set_state("propagating", "propagating ...")
        self._click_progress_bar.set(0)
        self._click_progress_label.configure(text="0/0")
        self._click_cancel_btn.pack(side="left", padx=(6, 0))

        def fisheye_fn(w, h):
            return self._get_click_fisheye_circle(w, h)

        threading.Thread(
            target=self._click_propagate_worker,
            args=(Path(output_dir), fisheye_fn),
            daemon=True,
        ).start()

    def _click_propagate_worker(self, output_dir, fisheye_fn):
        """Worker thread: run propagation and write masks."""
        try:
            def on_progress(cur, total, msg):
                self.after(0, lambda c=cur, t=total: self._click_update_progress(c, t))

            with self._pipeline_lock:
                written = self._click_session.propagate(
                    output_dir=output_dir,
                    fisheye_circle_fn=fisheye_fn,
                    progress_callback=on_progress,
                    cancel_event=self.cancel_flag,
                )

            def _on_main():
                self._click_cancel_btn.pack_forget()
                self._click_set_state(
                    "ready", f"done \u00b7 {written} masks \u2192 {output_dir}")
                self.log(f"Click propagation: {written} masks \u2192 {output_dir}")

            self.after(0, _on_main)
        except Exception as exc:
            err = str(exc)
            self.log(f"Click propagation error: {err}")
            import traceback
            self.log(traceback.format_exc())
            self.after(0, lambda: (
                self._click_cancel_btn.pack_forget(),
                self._click_set_state("error", f"propagation error: {err}"),
            ))

    def _click_update_progress(self, current, total):
        """Update propagation progress bar and label. Main thread."""
        if total > 0:
            self._click_progress_bar.set(current / total)
        self._click_progress_label.configure(text=f"{current}/{total}")

    def _on_click_cancel_propagation(self):
        """Cancel an in-progress propagation (session stays open)."""
        self.cancel_flag.set()

    # ── Teardown ──

    def _click_teardown(self):
        """Drop the session and clear all click-mode state. Main thread."""
        if self._click_session is not None:
            try:
                self._click_session.close()
            except Exception:
                pass
        self._click_session = None
        self._click_current_frame_idx = 0
        self._click_current_frame_pil = None
        self._click_current_display_size = None
        self._click_preview_mask = None
        self._preview_overlay_pil = None
        self._preview_mask_pil = None
        # Don't call configure(image=None) on overlay labels (pyimage bug).
        if hasattr(self, "click_mode_var") and self.click_mode_var.get():
            self.click_mode_var.set(False)
        if hasattr(self, "_click_cancel_btn"):
            self._click_cancel_btn.pack_forget()
        self._update_click_count()
        self._update_prompted_frames_label()
        self._click_set_state("idle", "idle. Activate to start a video session.")
        self._click_progress_bar.set(0)
        self._click_progress_label.configure(text="")

    # ── run masking ──

    def _on_run(self):
        # Click mode holds the video predictor on the GPU — cannot run batch
        if self._click_session is not None:
            self.log("ERROR: Deactivate interactive click mode before running batch masking.")
            return

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

        # Ensure shared library manager is loaded (fixed location, not
        # per-dataset). Cheap if already initialized.
        if self._static_mask_manager is None:
            self._init_static_mask_manager()

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

    # ── shared MaskingPipeline cache ──
    # See docs/planning/2026-04-30-pipeline-cache.md.
    # Concurrency policy A: ``_pipeline_lock`` covers BOTH construction
    # (get-or-build) and inference. This serializes click-mode decode against
    # an in-flight batch run; tradeoff is a click can wait one batch image's
    # duration. Lock is held by ``_get_pipeline`` and by every inference call
    # site. Encoder-only granularity is deferred until SAM3 reentrancy is
    # verified.

    def _pipeline_cache_key(self, config):
        """Build a hashable cache key from constructor-affecting config bits.

        Excludes per-call inference fields (confidence_threshold, prompts,
        yolo_classes, multi_pass_prompts, geometry, postprocess knobs, etc.).
        Includes everything ``MaskingPipeline.__init__`` reads to load weights.

        Per-call fields are patched onto the cached pipeline in
        ``_get_pipeline`` on cache hit — any new per-call field must be
        added there too.
        """
        def _hash_subdict(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return tuple(sorted((k, _hash_subdict(v)) for k, v in d.items()))
            if isinstance(d, (list, tuple)):
                return tuple(_hash_subdict(v) for v in d)
            return d

        return (
            config.model,
            config.device,
            config.ensemble,
            tuple(config.ensemble_models or ()),
            config.vos_propagation,
            _hash_subdict(getattr(config, "vos_config", None)),
            config.detect_shadows,
            _hash_subdict(getattr(config, "shadow_config", None)),
            config.sam_refine,
            _hash_subdict(getattr(config, "sam_refine_config", None)),
            config.matting,
            _hash_subdict(getattr(config, "matting_config", None)),
        )

    def _normalize_pipeline_config(self, config, auto_select):
        """Deep-copy ``config`` and resolve auto-select to a concrete model.

        Returns ``(MaskConfig copy, key tuple)``. The copy isolates the GUI's
        live config from ``MaskingPipeline.__init__``'s internal mutation, and
        normalizes ``("auto", True)`` and ``("SAM3", False)`` to the same key
        when SAM3 is auto-resolvable.
        """
        from reconstruction_pipeline import resolve_segmentation_model
        normalized = copy.deepcopy(config)
        if auto_select:
            normalized.model = resolve_segmentation_model()
        key = self._pipeline_cache_key(normalized)
        return normalized, key

    def _get_pipeline(self, config, auto_select):
        """Get-or-build the cached ``MaskingPipeline``. Holds ``_pipeline_lock``.

        Caller passes the current GUI ``MaskConfig`` and whether the user
        selected "auto" in the model dropdown. Returns a pipeline whose
        constructor inputs hash to the same key for the current config; if
        the cached key matches, returns the existing instance, otherwise
        evicts and rebuilds.
        """
        MaskingPipeline = _import_pipeline()[0]
        with self._pipeline_lock:
            normalized, key = self._normalize_pipeline_config(config, auto_select)
            if self._pipeline_cached is not None and self._pipeline_cached_key == key:
                # Patch per-call config fields that are excluded from the
                # cache key (prompts, class filters, thresholds, geometry
                # knobs, postprocess, output).  These don't require model
                # reload but must reflect the current GUI state.
                cached_cfg = self._pipeline_cached.config
                cached_cfg.remove_prompts = normalized.remove_prompts
                cached_cfg.keep_prompts = normalized.keep_prompts
                cached_cfg.keep_classes = normalized.keep_classes
                cached_cfg.multi_pass_prompts = normalized.multi_pass_prompts
                cached_cfg.yolo_classes = normalized.yolo_classes
                cached_cfg.confidence_threshold = normalized.confidence_threshold
                cached_cfg.review_threshold = normalized.review_threshold
                cached_cfg.pole_mask_expand = normalized.pole_mask_expand
                cached_cfg.nadir_mask_percent = normalized.nadir_mask_percent
                cached_cfg.fisheye_circle_mask = normalized.fisheye_circle_mask
                cached_cfg.fisheye_margin_percent = normalized.fisheye_margin_percent
                cached_cfg.cubemap_overlap = normalized.cubemap_overlap
                cached_cfg.mask_dilate_px = normalized.mask_dilate_px
                cached_cfg.fill_holes = normalized.fill_holes
                cached_cfg.edge_injection = normalized.edge_injection
                cached_cfg.multi_label = normalized.multi_label
                cached_cfg.inpaint_masked = normalized.inpaint_masked
                cached_cfg.output_format = normalized.output_format
                cached_cfg.num_workers = normalized.num_workers
                cached_cfg.colmap_validate = normalized.colmap_validate
                cached_cfg.colmap_dir = normalized.colmap_dir
                cached_cfg.colmap_config = normalized.colmap_config
                cached_cfg.temporal_tracking = normalized.temporal_tracking
                cached_cfg.save_review_images = normalized.save_review_images
                cached_cfg.save_reject_review_images = normalized.save_reject_review_images
                cached_cfg.torch_compile = normalized.torch_compile
                cached_cfg.yolo_model_size = normalized.yolo_model_size
                cached_cfg.rfdetr_model_size = normalized.rfdetr_model_size
                cached_cfg.static_mask_paths = normalized.static_mask_paths
                # The segmenter's _static_composite is built in __init__ from
                # the original paths; on cache hit we must refresh it or stale
                # overlays (often None from an empty initial run) get reused.
                rebuild = getattr(self._pipeline_cached.segmenter,
                                  "_rebuild_static_composite", None)
                if callable(rebuild):
                    rebuild()
                return self._pipeline_cached
            # Mismatch (or cold): evict before building so GPU memory is freed
            # before the new pipeline allocates.
            self._release_pipeline_locked()
            pipeline = MaskingPipeline(config=normalized, auto_select_model=False)
            self._pipeline_cached = pipeline
            self._pipeline_cached_key = key
            return pipeline

    def _release_pipeline_locked(self):
        """Drop the cached pipeline and any click session that depends on it.

        MUST be called with ``_pipeline_lock`` held. The click session holds
        an inference_state tied to the current model, so close it before the
        pipeline reference is dropped. UI cleanup is routed to the main
        thread via ``after()`` because ``_get_pipeline`` runs on workers.
        """
        if self._click_session is not None:
            try:
                self._click_session.close()
            except Exception:
                pass
            self._click_session = None
            # Schedule UI/state cleanup on the main thread.
            try:
                self.after(0, self._click_teardown)
            except Exception:
                pass
        self._pipeline_cached = None
        self._pipeline_cached_key = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_pipeline(self):
        """Public release: acquires the lock, then delegates."""
        with self._pipeline_lock:
            self._release_pipeline_locked()

    def _build_mask_config(self):
        """Build MaskConfig from current GUI settings.

        Returns (config, model_str, geometry) tuple.
        """
        _, MaskConfig, SegmentationModel, ImageGeometry, _, COCO_NAME_TO_ID = _import_pipeline()

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

        # Auto-derive YOLO class IDs from Remove text via COCO name lookup.
        # SAM3 uses the text directly; YOLO/RF-DETR need numeric class IDs.
        classes = []
        if remove_prompts:
            for prompt in remove_prompts:
                coco_id = COCO_NAME_TO_ID.get(prompt.lower())
                if coco_id is not None:
                    classes.append(coco_id)
                else:
                    # Try common aliases (e.g. "phone" → "cell phone")
                    for coco_name, cid in COCO_NAME_TO_ID.items():
                        if prompt.lower() in coco_name or coco_name in prompt.lower():
                            classes.append(cid)
                            break
        # Auto-derive keep class IDs from Keep text the same way.
        keep_classes = []
        if keep_prompts:
            for prompt in keep_prompts:
                coco_id = COCO_NAME_TO_ID.get(prompt.lower())
                if coco_id is not None:
                    keep_classes.append(coco_id)
                else:
                    for coco_name, cid in COCO_NAME_TO_ID.items():
                        if prompt.lower() in coco_name or coco_name in prompt.lower():
                            keep_classes.append(cid)
                            break

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
            temporal_tracking=self.temporal_tracking_var.get(),
            multi_pass_prompts=multi_pass,
            colmap_validate=colmap_enabled and bool(colmap_dir),
            colmap_dir=colmap_dir,
            colmap_config=colmap_config,
            edge_injection=self.edge_inject_var.get(),
            multi_label=self.multi_label_var.get(),
            inpaint_masked=self.inpaint_var.get(),
            output_format=self.output_format_var.get(),
            num_workers=int(self.num_workers_var.get()),
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
            keep_classes=keep_classes,
            remove_prompts=remove_prompts,
        )

        # Static mask overlay paths
        if self._static_mask_manager:
            config_kwargs["static_mask_paths"] = self._static_mask_manager.get_enabled_paths()

        config = MaskConfig(**config_kwargs)
        return config, model_str, geometry

    def _worker(self, input_path: str, output_path: str):
        try:
            import numpy as np
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

            pipeline = self._get_pipeline(config, auto_select=(model_str == "auto"))

            # The Output field IS the masks folder. "Create review folder"
            # purely controls whether review previews are also written alongside.
            mask_out_dir = out
            review_dir = (out / "review") if create_review_folder else None

            # Layered output: nest into layers/<name>/ inside the masks folder
            layer_name = self.layer_name_var.get().strip()
            if layer_name:
                mask_out_dir = mask_out_dir / "layers" / layer_name
                self.log(f"Saving to layer: {layer_name}")

            try:
                ReviewStatusManager = _import_review()[3]
                _status_mgr = ReviewStatusManager(mask_out_dir)
                def _on_result(stem, result):
                    area = float(np.sum(result.mask > 0) / result.mask.size * 100)
                    _status_mgr.set_quality_info(
                        stem, quality=result.quality.value,
                        confidence=result.confidence, area_percent=area)
            except Exception:
                _on_result = None

            paired_frame_dirs = None
            if inp.is_dir() and (inp / "front" / "frames").is_dir() and (inp / "back" / "frames").is_dir():
                paired_frame_dirs = {
                    "front": inp / "front" / "frames",
                    "back": inp / "back" / "frames",
                }

            if paired_frame_dirs:
                manifest = inp / "adjustment_manifest.json"
                self.log(f"Detected paired dataset: {inp}")
                if manifest.exists():
                    self.log(f"Adjustment manifest: {manifest}")
                total_stats = {"processed_images": 0, "review_images": 0, "paired_dataset": True}
                for lens, lens_input in paired_frame_dirs.items():
                    if self.cancel_flag.is_set():
                        break
                    lens_root = out / lens
                    lens_mask_dir = lens_root / "masks"
                    if layer_name:
                        lens_mask_dir = lens_mask_dir / "layers" / layer_name
                    lens_review_dir = (lens_root / "review") if create_review_folder else None
                    if create_review_folder:
                        self.log(f"Output {lens}: masks → {lens_mask_dir}, review → {lens_review_dir}")
                    else:
                        self.log(f"Output {lens}: masks → {lens_mask_dir}")
                    self.log(f"Processing paired {lens} frames: {lens_input}")
                    try:
                        ReviewStatusManager = _import_review()[3]
                        lens_status_mgr = ReviewStatusManager(lens_mask_dir)

                        def lens_result_cb(stem, result, _mgr=lens_status_mgr):
                            area = float(np.sum(result.mask > 0) / result.mask.size * 100)
                            _mgr.set_quality_info(
                                stem,
                                quality=result.quality.value,
                                confidence=result.confidence,
                                area_percent=area,
                            )
                    except Exception:
                        lens_result_cb = None
                    with self._pipeline_lock:
                        lens_stats = pipeline.process_directory(
                            input_dir=lens_input,
                            output_dir=lens_root,
                            geometry=geometry,
                            pattern=self.pattern_var.get(),
                            result_callback=lens_result_cb,
                            skip_existing=self.skip_existing_var.get(),
                            merge_existing=self.merge_existing_var.get(),
                            cancel_event=self.cancel_flag,
                            mask_dir=lens_mask_dir,
                            review_dir=lens_review_dir,
                        )
                    total_stats["processed_images"] += int(lens_stats.get("processed_images", 0) or 0)
                    total_stats["review_images"] += int(lens_stats.get("review_images", 0) or 0)
                    total_stats[f"{lens}_stats"] = lens_stats
                stats = total_stats
            elif inp.is_dir():
                if create_review_folder:
                    self.log(f"Output: masks → {mask_out_dir}, review → {review_dir}")
                else:
                    self.log(f"Output: masks → {mask_out_dir}")
                if self.temporal_tracking_var.get():
                    self.log(f"Processing directory with SAM3 temporal tracking: {inp}")
                else:
                    self.log(f"Processing directory: {inp}")
                with self._pipeline_lock:
                    stats = pipeline.process_directory(
                        input_dir=inp, output_dir=out,
                        geometry=geometry, pattern=self.pattern_var.get(),
                        result_callback=_on_result,
                        skip_existing=self.skip_existing_var.get(),
                        merge_existing=self.merge_existing_var.get(),
                        cancel_event=self.cancel_flag,
                        mask_dir=mask_out_dir,
                        review_dir=review_dir,
                    )
            elif inp.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                if create_review_folder:
                    self.log(f"Output: masks → {mask_out_dir}, review → {review_dir}")
                else:
                    self.log(f"Output: masks → {mask_out_dir}")
                if self.temporal_tracking_var.get():
                    self.log(f"Processing video with SAM3 temporal tracking: {inp}")
                else:
                    self.log(f"Processing video: {inp}")
                with self._pipeline_lock:
                    stats = pipeline.process_video(
                            video_path=inp, output_dir=out,
                            geometry=geometry,
                            save_review=create_review_folder,
                            mask_dir=mask_out_dir,
                            review_dir=review_dir,
                        )
            else:
                self.log(f"Processing single image: {inp}")
                import cv2
                import shutil
                image = cv2.imread(str(inp))
                if image is None:
                    self.log(f"ERROR: Failed to load image: {inp}")
                    return
                with self._pipeline_lock:
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
        if self._click_session is not None:
            self.log("ERROR: Deactivate interactive click mode before running batch masking.")
            return
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
            config, model_str, geometry = self._build_mask_config()
            pattern = self.pattern_var.get()
            skip_existing = self.skip_existing_var.get()
            merge_existing = self.merge_existing_var.get()
            layer_name = self.layer_name_var.get().strip()

            self.log(f"Queue: Model={model_str} | Geometry={self.geometry_var.get()} | "
                     f"Confidence={config.confidence_threshold}")
            if config.remove_prompts:
                self.log(f"Queue: Remove prompts: {config.remove_prompts}")
            if layer_name:
                self.log(f"Queue: Saving to layer: {layer_name}")

            # Load pipeline once — reuse across all folders
            pipeline = self._get_pipeline(config, auto_select=(model_str == "auto"))

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

                # Layered output: nest into layers/<name>/ inside the masks folder
                if layer_name:
                    masks_dir = masks_dir / "layers" / layer_name

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
                    ReviewStatusManager = _import_review()[3]
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
                    with self._pipeline_lock:
                        stats = pipeline.process_directory(
                            input_dir=input_dir,
                            output_dir=masks_dir,
                            mask_dir=masks_dir,
                            review_dir=review_dir,
                            geometry=geometry,
                            pattern=pattern,
                            result_callback=result_cb,
                            skip_existing=skip_existing,
                            merge_existing=merge_existing,
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
        ds = _Section(container, "Data Source",
                      subtitle="Masks folder (optionally containing layers/) + source images")
        ds.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        dsc = ds.content

        # Masks folder row — root may contain bare *.png and/or layers/<name>/*.png
        masks_row = ctk.CTkFrame(dsc, fg_color="transparent")
        masks_row.pack(fill="x", pady=2)
        ctk.CTkLabel(masks_row, text="Masks:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.masks_entry = ctk.CTkEntry(
            masks_row, placeholder_text="Masks folder (bare *.png and/or layers/ subdir)")
        self.masks_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(masks_row, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=lambda: self._browse_dir_into(self.masks_entry)).pack(side="left")

        images_row = ctk.CTkFrame(dsc, fg_color="transparent")
        images_row.pack(fill="x", pady=2)
        ctk.CTkLabel(images_row, text="Images:", width=LABEL_FIELD_WIDTH, anchor="e").pack(side="left")
        self.images_entry = ctk.CTkEntry(images_row, placeholder_text="Directory containing source images")
        self.images_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(images_row, text="...", width=BROWSE_BUTTON_WIDTH,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      command=lambda: self._browse_dir_into(self.images_entry)).pack(side="left")

        btn_row = ctk.CTkFrame(dsc, fg_color="transparent")
        btn_row.pack(fill="x", pady=(2, 0))
        ctk.CTkButton(btn_row, text="Load", width=110,
                      fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                      font=ctk.CTkFont(size=12),
                      command=self._load_review).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Auto-detect from Output", width=170,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12),
                      command=self._auto_detect_review_paths).pack(side="left")

        # Layer list — packed only when there are >1 layers (i.e. base + named).
        # When only the bare-root layer exists, this frame stays hidden so the
        # UI looks like the old single-layer review.
        self._layer_list_frame = ctk.CTkFrame(dsc, fg_color="transparent")

        # ── Filter & Sort ──
        fs = _Section(container, "Filter & Sort")
        fs.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fsc = fs.content

        fs_row = ctk.CTkFrame(fsc, fg_color="transparent")
        fs_row.pack(fill="x", pady=2)
        ctk.CTkLabel(fs_row, text="Sort:").pack(side="left", padx=(0, 2))
        self.sort_var = ctk.StringVar(value="Filename")
        ctk.CTkOptionMenu(fs_row, variable=self.sort_var,
                          values=["Filename", "Confidence", "Quality", "Area %", "Modified"],
                          command=lambda _: self._on_filter_change(), width=110).pack(side="left", padx=2)

        ctk.CTkLabel(fs_row, text="Filter:").pack(side="left", padx=(8, 2))
        self.filter_var = ctk.StringVar(value="All")
        ctk.CTkSegmentedButton(
            fs_row, values=["All", "Unreviewed", "Reviewed"],
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

        # Two-row 2-column grid: stats label / Edit button on the left,
        # checkbox / Merge button on the right. Right column appears only when
        # >1 layer is loaded. _load_review toggles column 1's weight so
        # single-layer mode gives column 0 the full row instead of half.
        self._stats_row = ctk.CTkFrame(cmc, fg_color="transparent")
        self._stats_row.pack(fill="x", padx=4, pady=2)
        self._stats_row.grid_columnconfigure(0, weight=1)
        # Column 1 weight starts at 0 (collapses); _load_review sets to 1 in multi-layer.

        self._review_info_label = ctk.CTkLabel(
            self._stats_row, text="Load masks to begin.",
            font=("Consolas", 11), anchor="w", justify="left",
        )
        self._review_info_label.grid(row=0, column=0, sticky="ew")
        self._merge_checkbox = ctk.CTkCheckBox(
            self._stats_row, text="Delete source layers after merge",
            variable=self._auto_delete_merged_var, font=("Consolas", 11),
        )
        Tooltip(self._merge_checkbox,
                "After merge succeeds, delete the visible named layer\n"
                "directories. Base is kept (it holds the composite).")
        # Not gridded initially — _load_review controls visibility.

        self._action_row = ctk.CTkFrame(cmc, fg_color="transparent")
        self._action_row.pack(fill="x", pady=(2, 0))
        self._action_row.grid_columnconfigure(0, weight=1)
        # Column 1 weight starts at 0; _load_review sets to 1 in multi-layer.

        ctk.CTkButton(
            self._action_row, text="Edit (OpenCV)", command=self._on_edit,
            fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
            font=ctk.CTkFont(size=12), height=30,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._merge_btn = ctk.CTkButton(
            self._action_row, text="Merge Visible Layers",
            fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
            font=ctk.CTkFont(size=12, weight="bold"), height=30,
            command=self._merge_layers,
        )
        # Not gridded initially — _load_review controls visibility.

        # ── Batch Actions ──
        ba = _Section(container, "Batch")
        ba.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        bac = ba.content

        ctk.CTkButton(bac, text="Hide Reviewed", command=self._on_hide_reviewed,
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      font=ctk.CTkFont(size=12), width=120).pack(side="left", padx=4, pady=2)

        self._review_summary_label = ctk.CTkLabel(
            bac, text="", font=("Consolas", 10), text_color="#9ca3af",
        )
        self._review_summary_label.pack(side="left", padx=8, pady=2)

    # ── layered review helpers ──

    def _discover_layers(self, masks_path: Path):
        """Populate self._layer_dirs from a masks folder root.

        Layer ordering: ``base`` (bare root *.png) first if present, then named
        subdirs from ``masks_path/layers/`` in sorted order. Empty subdirs are
        skipped. Returns the dict for callers that want to introspect.
        """
        self._layer_dirs.clear()
        # Base layer: bare PNGs at the masks folder root. Always first when present.
        if any(masks_path.glob("*.png")):
            self._layer_dirs["base"] = masks_path
        # Named layers: <masks_path>/layers/<name>/*.png
        layers_root = masks_path / "layers"
        if layers_root.exists():
            for d in sorted(layers_root.iterdir()):
                if d.is_dir() and any(d.glob("*.png")):
                    self._layer_dirs[d.name] = d
        return self._layer_dirs

    def _discover_layer_pairs(self, masks_path: Path, images_dir: Path):
        """Build _review_pairs as the union of stems across all known layers."""
        all_stems = set()
        for layer_dir in self._layer_dirs.values():
            for f in layer_dir.glob("*.png"):
                all_stems.add(f.stem)

        # Index image dir by stem once
        image_by_stem: Dict[str, Path] = {}
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"):
            for f in images_dir.glob(ext):
                image_by_stem.setdefault(f.stem, f)

        self._review_pairs = []
        for stem in sorted(all_stems):
            img = image_by_stem.get(stem)
            if img is None:
                continue
            layer_masks = {
                name: (d / f"{stem}.png")
                for name, d in self._layer_dirs.items()
                if (d / f"{stem}.png").exists()
            }
            self._review_pairs.append({
                "stem": stem,
                "image_path": img,
                "mask_path": None,    # filled by _sync_active_layer_paths
                "layer_masks": layer_masks,
            })

    def _sync_active_layer_paths(self):
        """Point each pair's mask_path at the active layer's mask file (or None)."""
        active = self._active_layer_var.get()
        if not active:
            return
        for pair in self._review_pairs:
            pair["mask_path"] = pair.get("layer_masks", {}).get(active)

    def _rebuild_layer_list(self):
        """Populate the layer list with a header row + one row per discovered layer.

        Column layout (mirrored between header and rows):
            [radio][swatch][name fills →][visible cb][count][× delete]
        """
        # Cancel any pending delete-confirm — its button is about to be destroyed
        if self._delete_pending_after_id is not None:
            try:
                self.after_cancel(self._delete_pending_after_id)
            except Exception:
                pass
            self._delete_pending_after_id = None
        self._delete_pending_layer = None
        self._layer_delete_btns.clear()

        for w in self._layer_list_frame.winfo_children():
            w.destroy()
        if not self._layer_dirs:
            return

        # Header: column labels align to the row below.
        # "active" sits above the radio column (which layer the action buttons
        # — Edit / Save / Delete / Skip — operate on, and which drives the
        # filter bar). "visible" sits above the visibility checkbox column.
        header = ctk.CTkFrame(self._layer_list_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(header, text="active", font=("Consolas", 10),
                     text_color="#9ca3af", width=20).pack(side="left", padx=(2, 4))   # radio col
        ctk.CTkLabel(header, text="", width=14).pack(side="left", padx=(0, 6))         # swatch col
        ctk.CTkLabel(header, text="", anchor="w").pack(side="left", fill="x", expand=True)  # name col
        ctk.CTkLabel(header, text="visible", font=("Consolas", 10),
                     text_color="#9ca3af").pack(side="left", padx=(4, 8))              # checkbox col
        ctk.CTkLabel(header, text="", font=("Consolas", 10), width=70).pack(side="left", padx=(0, 4))  # count col
        ctk.CTkLabel(header, text="", width=24).pack(side="left", padx=(2, 4))         # delete col

        for name, layer_dir in self._layer_dirs.items():
            count = sum(1 for _ in layer_dir.glob("*.png"))
            bgr = self._layer_colors[name]
            swatch_hex = self._layer_color_to_hex(bgr)
            row = ctk.CTkFrame(self._layer_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            ctk.CTkRadioButton(
                row, text="", value=name, variable=self._active_layer_var,
                command=self._on_active_layer_change, width=20,
            ).pack(side="left", padx=(2, 4))
            swatch = ctk.CTkFrame(row, fg_color=swatch_hex, width=14, height=14, corner_radius=2)
            swatch.pack(side="left", padx=(0, 6))
            swatch.pack_propagate(False)
            # Click swatch → open color picker (Phase 4b)
            swatch.bind("<Button-1>", lambda e, n=name: self._on_swatch_click(n))
            ctk.CTkLabel(row, text=name, font=("Consolas", 11), anchor="w").pack(
                side="left", fill="x", expand=True)
            ctk.CTkCheckBox(
                row, text="", variable=self._layer_visible[name], width=20,
            ).pack(side="left", padx=(4, 8))
            ctk.CTkLabel(row, text=f"{count} masks", font=("Consolas", 10),
                         text_color="#9ca3af", width=70, anchor="w").pack(side="left", padx=(0, 4))
            # Per-layer delete button (two-click confirm pattern)
            del_btn = ctk.CTkButton(
                row, text="×", width=24, height=22,
                fg_color="transparent", hover_color="#7f1d1d",
                text_color="#9ca3af", font=("Consolas", 14),
                command=lambda n=name: self._on_layer_delete_click(n),
            )
            del_btn.pack(side="left", padx=(2, 4))
            self._layer_delete_btns[name] = del_btn

        # Note: the merge controls (checkbox + Merge button) live in the
        # Current Mask section, not here. _load_review handles their
        # show/hide based on layer count.

    @staticmethod
    def _layer_color_to_hex(bgr_delta) -> str:
        """Convert a BGR-delta tuple into a representative hex color for the swatch."""
        b, g, r = bgr_delta
        # Treat the delta as the saturated overlay color on a mid-gray base
        return f"#{min(255, r + 80):02x}{min(255, g + 80):02x}{min(255, b + 80):02x}"

    @staticmethod
    def _bgr_delta_to_rgb_hex(bgr_delta) -> str:
        """Convert a BGR-delta to a fully-saturated RGB hex (for the color picker)."""
        b, g, r = bgr_delta
        max_ch = max(b, g, r) or 1
        rr = int(r * 255 / max_ch)
        gg = int(g * 255 / max_ch)
        bb = int(b * 255 / max_ch)
        return f"#{rr:02x}{gg:02x}{bb:02x}"

    @staticmethod
    def _rgb_to_bgr_delta(rgb_tuple, target_intensity: int = 100):
        """Convert a picker RGB triple to a BGR-delta scaled to target_intensity max."""
        r, g, b = (int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))
        max_ch = max(r, g, b) or 1
        scale = target_intensity / max_ch
        return (int(b * scale), int(g * scale), int(r * scale))

    def _on_swatch_click(self, layer_name: str):
        """Open a color picker for the given layer and persist the choice."""
        from tkinter import colorchooser
        if layer_name not in self._layer_colors:
            return
        initial = self._bgr_delta_to_rgb_hex(self._layer_colors[layer_name])
        try:
            result = colorchooser.askcolor(color=initial, title=f"Color for layer '{layer_name}'")
        except Exception as e:
            self.log(f"Color picker failed: {e}")
            return
        if result is None or result[0] is None:
            return  # cancelled
        self._layer_colors[layer_name] = self._rgb_to_bgr_delta(result[0])
        self._save_layer_colors()
        # Cascade: layer list (swatch), thumbnails, preview, editor IPC
        self._rebuild_layer_list()
        self._thumb_cache.clear()
        self._rebuild_thumb_list()
        if self._selected_stem:
            pair = next((p for p in self._review_pairs if p["stem"] == self._selected_stem), None)
            if pair is not None:
                self._show_review_overlay(pair)
        # Push refresh to a running editor so it picks up the new color
        if (self._editor_proc is not None and self._editor_proc.poll() is None
                and self._editor_cmd_file is not None):
            idx = self._nav_idx.get()
            if 0 <= idx < len(self._filtered_pairs):
                try:
                    self._editor_cmd_file.write_text(
                        self._build_editor_ipc_line(self._filtered_pairs[idx]),
                        encoding="utf-8")
                except Exception:
                    pass

    def _save_layer_colors(self):
        """Persist the current layer colors to <masks>/.layer_colors.json."""
        import json
        masks_dir_str = self.masks_entry.get().strip()
        if not masks_dir_str:
            return
        masks_dir = Path(masks_dir_str)
        if not masks_dir.exists():
            return
        config_path = masks_dir / ".layer_colors.json"
        data = {name: list(self._layer_colors[name])
                for name in self._layer_dirs if name in self._layer_colors}
        try:
            config_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self.log(f"Failed to save layer colors: {e}")

    def _load_layer_colors_file(self, masks_dir) -> Dict[str, tuple]:
        """Load persisted layer colors from <masks>/.layer_colors.json."""
        import json
        config_path = Path(masks_dir) / ".layer_colors.json"
        if not config_path.exists():
            return {}
        try:
            raw = json.loads(config_path.read_text())
            return {k: tuple(v) for k, v in raw.items()
                    if isinstance(v, list) and len(v) == 3}
        except Exception:
            return {}

    def _on_active_layer_change(self, *_):
        """Active layer changed: re-bind status manager, refresh paths and view."""
        active = self._active_layer_var.get()
        if not active or active not in self._layer_dirs:
            return
        self._review_status_mgr = self._layer_status_mgrs[active]
        self._sync_active_layer_paths()
        self._apply_filter()
        self._rebuild_thumb_list()
        if self._selected_stem:
            self._update_review_info()
            # Refresh preview so the "Mask" toggle reflects the new active layer
            pair = next((p for p in self._review_pairs if p["stem"] == self._selected_stem), None)
            if pair is not None:
                self._show_review_overlay(pair)

    def _on_layer_visibility_change(self, *_):
        """Visibility toggled: invalidate cache, refresh preview, lazy-rerender thumbs."""
        if not self._review_loaded:
            return
        self._thumb_cache.clear()
        self._rebuild_thumb_list()
        # Refresh the big preview to match the new visible-layer set
        if self._selected_stem:
            pair = next((p for p in self._review_pairs if p["stem"] == self._selected_stem), None)
            if pair is not None:
                self._show_review_overlay(pair)

    # ── per-layer delete (two-click confirm) ──

    def _on_layer_delete_click(self, name: str):
        """Two-click confirm: first click arms, second click within 3s deletes."""
        if name not in self._layer_dirs:
            return
        if self._delete_pending_layer == name:
            # Confirmed → execute
            self._cancel_delete_pending()
            self._delete_layer(name)
            return
        # First click — arm the pending state, cancel any other pending
        self._cancel_delete_pending()
        self._delete_pending_layer = name
        btn = self._layer_delete_btns.get(name)
        if btn is not None:
            try:
                btn.configure(text="confirm?", fg_color=COLOR_ACTION_DANGER,
                              text_color="#ffffff", width=70)
            except Exception:
                pass
        self._delete_pending_after_id = self.after(3000, self._cancel_delete_pending)

    def _cancel_delete_pending(self):
        """Revert the armed delete button (timer expiry, click on different layer, etc.)."""
        if self._delete_pending_after_id is not None:
            try:
                self.after_cancel(self._delete_pending_after_id)
            except Exception:
                pass
            self._delete_pending_after_id = None
        if self._delete_pending_layer:
            btn = self._layer_delete_btns.get(self._delete_pending_layer)
            if btn is not None:
                try:
                    btn.configure(text="×", fg_color="transparent",
                                  text_color="#9ca3af", width=24)
                except Exception:
                    pass
        self._delete_pending_layer = None

    def _delete_layer(self, name: str):
        """Delete a layer's files from disk, then reload to reflect the change.

        - Named layer: remove ``<masks>/layers/<name>/`` recursively.
        - Base layer:  remove only ``<masks>/*.png`` (the bare-root masks) and
          the root ``review_status.json``. Subdirs (``layers/``, ``review/``,
          ``images/``) are preserved. If named layers exist, log a warning
          since the user may be wiping their composite output.
        """
        import shutil

        if name not in self._layer_dirs:
            return
        masks_path_str = self.masks_entry.get().strip()
        if not masks_path_str:
            self.log("Cannot delete: masks folder is unset.")
            return
        masks_path = Path(masks_path_str)

        if name == "base":
            bare_files = list(masks_path.glob("*.png"))
            named_count = sum(1 for n in self._layer_dirs if n != "base")
            if named_count > 0:
                self.log(f"Warning: deleting base ({len(bare_files)} masks) while "
                         f"{named_count} named layer(s) still exist. The composite "
                         f"output will be gone — re-merge to regenerate.")
            failures = 0
            for f in bare_files:
                try:
                    f.unlink()
                except Exception as e:
                    failures += 1
                    self.log(f"  failed to delete {f.name}: {e}")
            # Drop the root review_status.json since base is gone
            status_file = masks_path / "review_status.json"
            if status_file.exists():
                try:
                    status_file.unlink()
                except Exception:
                    pass
            removed = len(bare_files) - failures
            self.log(f"Deleted layer 'base' — {removed} mask file(s) removed.")
        else:
            layer_dir = self._layer_dirs[name]
            mask_count = sum(1 for _ in layer_dir.glob("*.png"))
            try:
                shutil.rmtree(str(layer_dir))
            except Exception as e:
                self.log(f"Failed to delete layer '{name}': {e}")
                return
            self.log(f"Deleted layer '{name}' — {mask_count} mask file(s) removed.")

        # Reload to refresh the layer list. Preserve user state where possible.
        saved_active = self._active_layer_var.get()
        saved_visibility = {n: v.get() for n, v in self._layer_visible.items()}
        self._load_review()
        for n, was_visible in saved_visibility.items():
            if n in self._layer_visible and not was_visible:
                self._layer_visible[n].set(False)
        if saved_active in self._layer_dirs and saved_active != name:
            self._active_layer_var.set(saved_active)
            self._on_active_layer_change()

    def _merge_layers(self):
        """Kick off a background-threaded merge so the UI stays responsive.

        The merge writes the OR composite of every visible layer to the masks
        folder root (overwriting ``base`` if visible). Per-layer directories
        under ``layers/<name>/`` are untouched — they remain editable so you
        can re-edit a layer and re-merge.
        """
        if not self._review_loaded or not self._layer_dirs:
            self.log("Load a project before merging layers.")
            return

        masks_dir_str = self.masks_entry.get().strip()
        if not masks_dir_str:
            self.log("Masks folder is unset.")
            return
        masks_dir = Path(masks_dir_str)

        visible = {n: d for n, d in self._layer_dirs.items()
                   if self._layer_visible.get(n) and self._layer_visible[n].get()}
        if not visible:
            self.log("No visible layers to merge.")
            return

        # Disable the button + flip text so the user knows it's running. A
        # background thread does the I/O so the console keeps polling.
        try:
            self._merge_btn.configure(state="disabled", text="Merging...")
        except Exception:
            pass
        # Snapshot UI state for restoration after reload (we're going to call
        # _load_review which resets visibility/active).
        saved_active = self._active_layer_var.get()
        saved_visibility = {n: v.get() for n, v in self._layer_visible.items()}
        threading.Thread(
            target=self._merge_layers_worker,
            args=(masks_dir, visible, saved_active, saved_visibility),
            daemon=True,
        ).start()

    def _merge_layers_worker(self, masks_dir: Path, visible: dict,
                             saved_active: str, saved_visibility: dict):
        """Worker thread: do the merge I/O and post results back to main thread."""
        import cv2
        import numpy as np

        # Union of stems across visible layers
        all_stems = set()
        for layer_dir in visible.values():
            for f in layer_dir.glob("*.png"):
                all_stems.add(f.stem)
        if not all_stems:
            self.log("No mask files in the visible layers.")
            self.after(0, self._merge_finish, 0, 0, masks_dir, saved_active, saved_visibility)
            return

        # Notify (no popout dialogs): the bare-root *.png files will be overwritten.
        existing_files = list(masks_dir.glob("*.png"))
        layer_summary = ", ".join(visible)
        if existing_files:
            self.log(f"Merging [{layer_summary}] into {masks_dir} — "
                     f"{len(existing_files)} bare-root mask(s) will be overwritten "
                     f"(this is the canonical composite output).")
        else:
            self.log(f"Merging [{layer_summary}] into {masks_dir}...")

        masks_dir.mkdir(parents=True, exist_ok=True)
        merged_count = 0
        skipped_count = 0
        for stem in sorted(all_stems):
            merged = None
            for name, layer_dir in visible.items():
                mask_path = layer_dir / f"{stem}.png"
                if not mask_path.exists():
                    continue
                raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    continue
                layer_internal = (raw < 128).astype(np.uint8)
                if merged is None:
                    merged = layer_internal
                else:
                    if layer_internal.shape == merged.shape:
                        merged = np.maximum(merged, layer_internal)
                    else:
                        self.log(f"  size mismatch on {stem}, skipping {name}")
            if merged is not None:
                cv2.imwrite(str(masks_dir / f"{stem}.png"), (1 - merged) * 255)
                merged_count += 1
            else:
                skipped_count += 1

        msg = f"Merged {merged_count} composite mask(s) → {masks_dir}"
        if skipped_count:
            msg += f"  ({skipped_count} skipped)"
        self.log(msg)

        # Auto-delete: consume the visible NAMED source layers if requested.
        # Base is never auto-deleted (it's the merge target — holds the composite).
        # Invisible layers stay (they weren't part of the merge).
        if self._auto_delete_merged_var.get():
            import shutil
            deleted = []
            for name, layer_dir in visible.items():
                if name == "base":
                    continue
                try:
                    shutil.rmtree(str(layer_dir))
                    deleted.append(name)
                except Exception as e:
                    self.log(f"Auto-delete: failed to remove '{name}': {e}")
            if deleted:
                self.log(f"Auto-deleted source layer(s): {', '.join(deleted)}")
                # Drop deleted layers from the snapshot so _merge_finish doesn't
                # try to restore visibility/active for them.
                for n in deleted:
                    saved_visibility.pop(n, None)
                if saved_active in deleted:
                    saved_active = ""

        self.after(0, self._merge_finish, merged_count, skipped_count,
                   masks_dir, saved_active, saved_visibility)

    def _merge_finish(self, merged_count: int, skipped_count: int,
                      masks_dir: Path, saved_active: str, saved_visibility: dict):
        """Main-thread cleanup after merge: reload, restore state, flash success."""
        # Reload to reflect the new bare-root state
        self._load_review()
        # Restore visibility — _load_review defaults all layers to True
        for n, was_visible in saved_visibility.items():
            if n in self._layer_visible and not was_visible:
                self._layer_visible[n].set(False)
        if saved_active in self._layer_dirs:
            self._active_layer_var.set(saved_active)
            self._on_active_layer_change()

        # Visible button feedback that survives the console scroll
        try:
            success_text = f"✓ Merged {merged_count} mask(s)"
            self._merge_btn.configure(state="normal", text=success_text,
                                      fg_color=COLOR_ACTION_PRIMARY_H)
            # Revert to original label after 4 seconds — long enough to read,
            # short enough not to obscure the button's normal purpose.
            def _revert():
                try:
                    self._merge_btn.configure(text="Merge Visible Layers",
                                              fg_color=COLOR_ACTION_PRIMARY)
                except Exception:
                    pass
            self.after(4000, _revert)
        except Exception:
            pass

    # ── review loading ──

    def _auto_detect_review_paths(self):
        """Point the Masks field at the resolved mask output directory and load.

        The Output field on the Mask tab IS the masks folder, so the masks
        directory equals out_path. Bare PNGs at the root and any layers/
        subdirs are discovered together by the unified loader.
        """
        out = self.output_entry.get().strip()
        if not out:
            self.log("Set an output directory on the Process tab first.")
            return
        self.log(f"Auto-detect: scanning {out}...")
        out_path = Path(out)

        # Resolve images directory
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

        # Populate fields
        self.masks_entry.delete(0, "end")
        self.masks_entry.insert(0, str(out_path))
        if images_dir:
            self.images_entry.delete(0, "end")
            self.images_entry.insert(0, str(images_dir))
        else:
            self.log("Could not auto-detect images directory.")

        # Load if both paths resolved — _load_review handles base + named layers
        if images_dir:
            self._load_review()

    def _load_review(self):
        """Unified loader: masks_path is the masks folder root.

        Discovers two kinds of layers in one pass:
          - Bare ``masks_path/*.png`` → ``base`` layer (legacy/single workflow)
          - ``masks_path/layers/<name>/*.png`` → named layers

        Only the base case (single layer) collapses to old single-mask UX:
        the layer list stays hidden, the active layer is ``base``, and editing
        opens the bare mask file. With multiple layers, the layer list
        appears and overlays composite together.
        """
        from review_gui import LAYER_COLORS

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

        self.log(f"Loading from {masks_path}...")
        self._prefs["masks_dir"] = masks_dir
        self._prefs["images_dir"] = images_dir
        self._save_prefs()

        try:
            (self._load_overlay_thumbnail, self._compute_mask_area_percent,
             self._review_colors, ReviewStatusManager, _) = _import_review()

            # Discover layers (base + named)
            self._discover_layers(masks_path)
            if not self._layer_dirs:
                self.log(f"No masks found in {masks_path}.")
                return

            # Reset per-layer state and instantiate one ReviewStatusManager per layer
            self._layer_colors.clear()
            self._layer_visible.clear()
            self._layer_status_mgrs.clear()
            saved_colors = self._load_layer_colors_file(masks_path)
            for i, name in enumerate(self._layer_dirs):
                # Persisted custom color overrides the palette default
                self._layer_colors[name] = saved_colors.get(
                    name, LAYER_COLORS[i % len(LAYER_COLORS)])
                self._layer_visible[name] = ctk.BooleanVar(value=True)
                self._layer_visible[name].trace_add("write", self._on_layer_visibility_change)
                self._layer_status_mgrs[name] = ReviewStatusManager(self._layer_dirs[name])

            # Show layer list + merge controls only when there are multiple
            # layers — keeps the legacy single-mask layout clean for users
            # with no named layers.
            multi = len(self._layer_dirs) > 1
            if multi:
                self._layer_list_frame.pack(fill="x", pady=(4, 0))
                self._rebuild_layer_list()
            else:
                self._layer_list_frame.pack_forget()
            # Merge controls live in the Current Mask section: checkbox
            # right-aligned in stats row, Merge button right-aligned in action
            # row, both filling column 1. Toggling column 1's weight is what
            # collapses the right side cleanly when single-layer.
            if hasattr(self, "_stats_row") and hasattr(self, "_merge_checkbox"):
                if multi:
                    self._stats_row.grid_columnconfigure(1, weight=1)
                    self._merge_checkbox.grid(row=0, column=1, sticky="e", padx=(8, 0))
                else:
                    self._merge_checkbox.grid_forget()
                    self._stats_row.grid_columnconfigure(1, weight=0)
            if hasattr(self, "_action_row") and hasattr(self, "_merge_btn"):
                if multi:
                    self._action_row.grid_columnconfigure(1, weight=1)
                    self._merge_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
                else:
                    self._merge_btn.grid_forget()
                    self._action_row.grid_columnconfigure(1, weight=0)

            # Build pairs as union of stems across layers, then bind to active layer
            self._review_pairs = []
            self._discover_layer_pairs(masks_path, images_path)
            first_layer = next(iter(self._layer_dirs))
            self._active_layer_var.set(first_layer)
            self._review_status_mgr = self._layer_status_mgrs[first_layer]
            self._sync_active_layer_paths()

            self._review_loaded = True
            self._selected_stem = None
            self._thumb_cache.clear()
            self._thumb_widgets.clear()
            self._apply_filter()
            self._rebuild_thumb_list()

            # Compute areas + load thumbnails in background
            threading.Thread(target=self._compute_areas_bg, daemon=True).start()
            threading.Thread(target=self._load_thumbnails_bg, daemon=True).start()

            # Set zoom to 100% = fit to panel width
            self._zoom_var.set(100)
            self._zoom_label.configure(text="100%")

            # If we're on the Review tab, refresh the navigator
            if self._preview_mode == "review":
                self._load_review_nav_list()

            layer_summary = ", ".join(self._layer_dirs.keys())
            self.log(f"Loaded {len(self._layer_dirs)} layer(s) [{layer_summary}], "
                     f"{len(self._review_pairs)} image pairs")

        except Exception as e:
            self.log(f"Failed to load review: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _compute_areas_bg(self):
        """Compute mask area percentages for the active layer (background thread)."""
        if self._review_status_mgr is None:
            return
        for pair in self._review_pairs:
            mask_path = pair.get("mask_path")
            if mask_path is None:
                continue
            ms = self._review_status_mgr.get(pair["stem"])
            if ms.area_percent == 0.0:
                ms.area_percent = self._compute_mask_area_percent(mask_path)
        self._review_status_mgr.save()

    # ── thumbnails ──

    def _compose_layer_thumb(self, pair, size: int = 300):
        """Composite all visible layers of a pair into a single PIL thumbnail."""
        try:
            from review_gui import load_multi_layer_thumbnail
        except ImportError:
            return None
        layer_masks = [
            (pair["layer_masks"][name], self._layer_colors[name])
            for name in self._layer_dirs
            if self._layer_visible.get(name) and self._layer_visible[name].get()
            and name in pair.get("layer_masks", {})
        ]
        if not layer_masks:
            return None
        return load_multi_layer_thumbnail(
            pair["image_path"], layer_masks, size=size, pad_to_square=False)

    def _load_thumbnails_bg(self):
        """Load thumbnails for the first visible batch (layer-composite-aware)."""
        pairs = list(self._filtered_pairs) if self._filtered_pairs else list(self._review_pairs)
        for pair in pairs[:40]:
            try:
                pil = self._compose_layer_thumb(pair)
            except Exception:
                pil = None
            if pil is not None:
                self._thumb_cache[pair["stem"]] = pil
        self.after(0, self._rebuild_thumb_list)

    def _load_thumb_for_stem(self, stem):
        """Load a single thumbnail (background thread, layer-composite-aware)."""
        if stem in self._thumb_cache:
            return
        pair = next((p for p in self._review_pairs if p["stem"] == stem), None)
        if not pair:
            return
        try:
            pil = self._compose_layer_thumb(pair)
        except Exception:
            pil = None
        if pil is not None:
            self._thumb_cache[stem] = pil

    def _create_thumb_cell(self, stem, grid_col):
        """Create a single thumbnail cell in a 2-column row."""
        # Create new row frame when starting column 0
        if grid_col == 0:
            self._current_thumb_row = ctk.CTkFrame(self._thumb_scroll, fg_color="transparent")
            self._current_thumb_row.pack(fill="x", pady=1)
            self._current_thumb_row.grid_columnconfigure(0, weight=1)
            self._current_thumb_row.grid_columnconfigure(1, weight=1)

        ms = self._review_status_mgr.get(stem) if self._review_status_mgr else None
        rc = getattr(self, '_review_colors', {})

        border_color = "#6b7280"
        if ms:
            border_color = rc.get(ms.status, "#6b7280")

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
            if ms.status == "reviewed":
                parts.append("✓")
            if ms.confidence > 0:
                parts.append(f"{ms.confidence:.0%}")
            if ms.last_edit_modified is True:
                parts.append("edited")
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
                        rc = getattr(self, '_review_colors', {})
                        bc = "#6b7280"
                        if ms:
                            bc = rc.get(ms.status, "#6b7280")
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
        if filt == "Unreviewed":
            pairs = [p for p in pairs if self._review_status_mgr.get(p["stem"]).status == "unreviewed"]
        elif filt == "Reviewed":
            pairs = [p for p in pairs if self._review_status_mgr.get(p["stem"]).status == "reviewed"]

        sort_key = self.sort_var.get()
        if sort_key == "Confidence":
            pairs.sort(key=lambda p: self._review_status_mgr.get(p["stem"]).confidence)
        elif sort_key == "Quality":
            order = {"reject": 0, "poor": 1, "review": 2, "good": 3, "excellent": 4, "": -1}
            pairs.sort(key=lambda p: order.get(self._review_status_mgr.get(p["stem"]).quality, -1))
        elif sort_key == "Area %":
            pairs.sort(key=lambda p: self._review_status_mgr.get(p["stem"]).area_percent, reverse=True)
        elif sort_key == "Modified":
            order = {True: 0, False: 1, None: 2}
            pairs.sort(key=lambda p: order.get(self._review_status_mgr.get(p["stem"]).last_edit_modified, 2))
        else:
            pairs.sort(key=lambda p: p["stem"])

        self._filtered_pairs = pairs

    # ── actions ──

    def _build_editor_ipc_line(self, pair) -> str:
        """Build the pipe-delimited IPC line for the editor subprocess.

        Single mode: ``image|mask|stem``
        Layered mode: ``image|mask|stem|active_color=B,G,R|bg=path:B,G,R|...``

        The bg path:color separator is the LAST ``:`` to coexist with Windows
        drive letters in absolute paths.
        """
        # Layered IPC format:
        #   image_path|stem|active_idx=N|layer=name^path^B,G,R|layer=...
        # ``^`` is the inner separator (illegal in Windows filenames, so it
        # never collides with paths). Includes ALL layers known for this pair
        # so the editor can switch between them via number keys.
        active = self._active_layer_var.get()
        layer_masks = pair.get("layer_masks", {})
        # Build ordered list matching self._layer_dirs ordering, including only
        # layers that have a mask file for this image (others are skipped —
        # the editor's number-key indices align with this filtered list).
        layers_for_pair = [(name, layer_masks[name]) for name in self._layer_dirs
                           if name in layer_masks]
        active_idx = 0
        for i, (name, _) in enumerate(layers_for_pair):
            if name == active:
                active_idx = i
                break

        parts = [str(pair['image_path']), pair['stem'], f"active_idx={active_idx}"]
        for name, layer_path in layers_for_pair:
            b, g, r = self._layer_colors[name]
            parts.append(f"layer={name}^{layer_path}^{b},{g},{r}")
        return "|".join(parts) + "\n"

    def _on_edit(self):
        if not self._review_loaded or not self._filtered_pairs:
            return
        idx = self._nav_idx.get()
        if idx >= len(self._filtered_pairs):
            return
        pair = self._filtered_pairs[idx]
        if pair.get("mask_path") is None:
            self.log(f"No mask file for {pair['stem']} in active layer.")
            return

        ipc_line = self._build_editor_ipc_line(pair)

        # If editor subprocess already running, send new image via cmd file
        if self._editor_proc is not None and self._editor_proc.poll() is None:
            try:
                self._editor_cmd_file.write_text(ipc_line, encoding="utf-8")
            except Exception as e:
                self.log(f"Editor send failed: {e}")
            return

        # Launch editor as subprocess (OpenCV needs its own main thread)
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        self._editor_cmd_file = tmp_dir / "reconstruction_zone_editor_cmd.txt"
        self._editor_signal_file = tmp_dir / "reconstruction_zone_editor_signal.txt"
        # Pre-load the IPC line so the editor sees layered specs on its first poll.
        # The signal file is cleared.
        self._editor_cmd_file.write_text(ipc_line, encoding="utf-8")
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
                elif line.startswith("layer_saved|"):
                    # Auto-save during in-editor layer switch — update status
                    # for the saved layer but do NOT advance to next image.
                    mask_path = Path(line.split("|", 1)[1])
                    self._on_layer_save(mask_path)
                elif line.startswith("saved|"):
                    # Legacy fallback
                    mask_path = Path(line.split("|", 1)[1])
                    self._on_editor_save(mask_path, modified=True)
        except Exception:
            pass

    def _layer_for_mask_path(self, mask_path: Path) -> Optional[str]:
        """Return the layer name a mask file belongs to (e.g. 'base', 'shadows')."""
        for name, layer_dir in self._layer_dirs.items():
            try:
                mask_path.relative_to(layer_dir)
                return name
            except ValueError:
                continue
        return None

    def _on_layer_save(self, mask_path):
        """Editor auto-saved a layer mid-switch. Update status without advancing."""
        mask_path = Path(mask_path)
        layer_name = self._layer_for_mask_path(mask_path)
        if layer_name is None:
            return
        stem = mask_path.stem
        mgr = self._layer_status_mgrs.get(layer_name)
        if mgr is not None:
            mgr.record_action(stem, "reviewed", modified=True)
        # Refresh thumbnail for this stem (pixels on disk changed)
        self._thumb_cache.pop(stem, None)
        self._rebuild_thumb_list()
        # Refresh preview if this is the currently selected pair
        if stem == self._selected_stem:
            pair = next((p for p in self._review_pairs if p["stem"] == stem), None)
            if pair is not None:
                self._show_review_overlay(pair)
            self._update_review_info()

    def _on_editor_save(self, mask_path, modified=True):
        """Editor saved active layer (s key). Advance to next image."""
        mask_path = Path(mask_path)
        # Derive layer + stem from the path so we can update the right status manager
        # even if the editor saved a non-active layer (shouldn't happen for `s`, but safe).
        layer_name = self._layer_for_mask_path(mask_path)
        stem = mask_path.stem
        mgr = self._layer_status_mgrs.get(layer_name) if layer_name else self._review_status_mgr
        if stem and mgr is not None:
            mgr.record_action(stem, "reviewed", modified=modified)
        # Invalidate thumb cache so it reloads on next rebuild
        self._thumb_cache.pop(stem, None)

        # Advance navigator to next image and push it to the editor
        idx = self._nav_idx.get()
        if idx + 1 < len(self._filtered_pairs):
            next_idx = idx + 1
            self._nav_idx.set(next_idx)
            self._on_nav_change(next_idx)
            next_pair = self._filtered_pairs[next_idx]
            # Push next image to editor subprocess (with layered specs)
            if self._editor_cmd_file is not None and next_pair.get("mask_path"):
                try:
                    self._editor_cmd_file.write_text(
                        self._build_editor_ipc_line(next_pair),
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

    def _on_hide_reviewed(self):
        """Switch to Unreviewed filter to hide reviewed masks."""
        if not self._review_loaded:
            return
        self.filter_var.set("Unreviewed")
        self._on_filter_change()

    def _advance(self):
        """Move navigator to next unreviewed mask."""
        idx = self._nav_idx.get()
        for i in range(idx + 1, len(self._filtered_pairs)):
            if self._review_status_mgr.get(self._filtered_pairs[i]["stem"]).status == "unreviewed":
                self._nav_idx.set(i)
                self._on_nav_change(i)
                self._update_review_info()
                return
        # No more unreviewed — stay on current, just refresh info
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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(detail, encoding="utf-8")
    except Exception:
        pass
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

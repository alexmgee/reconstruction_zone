"""
Reframe Engine — equirectangular to pinhole perspective views.

Geometry reimplemented from documented ERP pinhole spec (see docs/ERP_REFRAME.md).
Provenance: independent implementation; behavioral reference only from external plugin reports.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PATH_UNSAFE = re.compile(r'[<>:"|?*\\/]')


class OutputLayout(str, Enum):
  RIG = "rig"
  STATION = "station"
  FLAT = "flat"


@dataclass
class Ring:
  pitch: float
  count: int
  fov: float = 65.0
  start_yaw: float = 0.0
  flip_vertical: bool = False

  def get_yaw_positions(self) -> List[float]:
    if self.count == 0:
      return []
    step = 360.0 / self.count
    return [self.start_yaw + i * step for i in range(self.count)]


@dataclass
class FreeView:
  name: str
  yaw: float
  pitch: float
  fov: float = 90.0


class ViewSpec(NamedTuple):
  yaw: float
  pitch: float
  fov: float
  name: str
  flip_vertical: bool


@dataclass
class ViewConfig:
  rings: List[Ring] = field(default_factory=list)
  views: List[FreeView] = field(default_factory=list)
  include_zenith: bool = True
  include_nadir: bool = False
  zenith_fov: float = 65.0
  output_size: int = 1920
  jpeg_quality: int = 95

  def total_views(self) -> int:
    count = sum(ring.count for ring in self.rings)
    count += len(self.views)
    if self.include_zenith:
      count += 1
    if self.include_nadir:
      count += 1
    return count

  def get_all_views(self) -> List[ViewSpec]:
    result: List[ViewSpec] = []
    for ring_idx, ring in enumerate(self.rings):
      for view_idx, yaw in enumerate(ring.get_yaw_positions()):
        name = f"{ring_idx:02d}_{view_idx:02d}"
        result.append(ViewSpec(yaw, ring.pitch, ring.fov, name, ring.flip_vertical))
    for fv in self.views:
      result.append(ViewSpec(fv.yaw, fv.pitch, fv.fov, fv.name, False))
    if self.include_zenith:
      result.append(ViewSpec(0, 90, self.zenith_fov, "ZN_00", False))
    if self.include_nadir:
      result.append(ViewSpec(0, -90, self.zenith_fov, "ND_00", False))
    return result

  def to_dict(self) -> dict:
    d: dict = {
      "rings": [
        {
          "pitch": r.pitch,
          "count": r.count,
          "fov": r.fov,
          "start_yaw": r.start_yaw,
          "flip_vertical": r.flip_vertical,
        }
        for r in self.rings
      ],
      "include_zenith": self.include_zenith,
      "include_nadir": self.include_nadir,
      "zenith_fov": self.zenith_fov,
      "output_size": self.output_size,
      "jpeg_quality": self.jpeg_quality,
    }
    if self.views:
      d["views"] = [
        {"name": v.name, "yaw": v.yaw, "pitch": v.pitch, "fov": v.fov}
        for v in self.views
      ]
    return d

  @classmethod
  def from_dict(cls, data: dict) -> "ViewConfig":
    rings = [
      Ring(
        pitch=r["pitch"],
        count=r["count"],
        fov=r.get("fov", 65.0),
        start_yaw=r.get("start_yaw", 0.0),
        flip_vertical=r.get("flip_vertical", False),
      )
      for r in data.get("rings", [])
    ]
    views = [
      FreeView(
        name=v["name"],
        yaw=v["yaw"],
        pitch=v["pitch"],
        fov=v.get("fov", 90.0),
      )
      for v in data.get("views", [])
    ]
    return cls(
      rings=rings,
      views=views,
      include_zenith=data.get("include_zenith", True),
      include_nadir=data.get("include_nadir", False),
      zenith_fov=data.get("zenith_fov", 65.0),
      output_size=data.get("output_size", 1920),
      jpeg_quality=data.get("jpeg_quality", 95),
    )


def copy_view_config(config: ViewConfig) -> ViewConfig:
  return ViewConfig.from_dict(config.to_dict())


def validate_view_config(config: ViewConfig) -> None:
  if config.output_size <= 0:
    raise ValueError(f"output_size must be > 0, got {config.output_size}")
  if not 1 <= config.jpeg_quality <= 100:
    raise ValueError(f"jpeg_quality must be 1..100, got {config.jpeg_quality}")

  views = config.get_all_views()
  if not views:
    raise ValueError("ViewConfig produces zero views")

  seen: set[str] = set()
  for view in views:
    if view.fov <= 0 or view.fov >= 180:
      raise ValueError(f"Invalid fov {view.fov} for view {view.name}")
    if view.pitch < -90 or view.pitch > 90:
      raise ValueError(f"Invalid pitch {view.pitch} for view {view.name}")
    if not view.name or _PATH_UNSAFE.search(view.name):
      raise ValueError(f"Invalid or path-unsafe view name: {view.name!r}")
    if view.name in seen:
      raise ValueError(f"Duplicate view name: {view.name}")
    seen.add(view.name)


def resolve_preset_name(preset_name: str) -> str:
  if preset_name in VIEW_PRESETS:
    return preset_name
  if preset_name in LEGACY_PRESET_ALIASES:
    return LEGACY_PRESET_ALIASES[preset_name]
  raise KeyError(f"Unknown preset: {preset_name}")


def get_view_preset(preset_name: str) -> ViewConfig:
  key = resolve_preset_name(preset_name)
  return copy_view_config(VIEW_PRESETS[key])


# --- Built-in ERP presets (immutable templates; copy before mutating) ---

ERP_SCAFFOLD_PRESET = "erp_scaffold_8"
DEFAULT_PRESET = "medium"

VIEW_PRESETS: dict[str, ViewConfig] = {
  ERP_SCAFFOLD_PRESET: ViewConfig(
    rings=[
      Ring(pitch=-35.0, count=4, fov=90.0, start_yaw=0.0),
      Ring(pitch=35.0, count=4, fov=90.0, start_yaw=45.0),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  "cubemap": ViewConfig(
    rings=[
      Ring(pitch=0, count=4, fov=90),
      Ring(pitch=-90, count=1, fov=90, flip_vertical=True),
      Ring(pitch=90, count=1, fov=90, flip_vertical=True),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  "low": ViewConfig(
    views=[
      FreeView("00_00", 0.0, 90.0, 90),
      FreeView("00_01", 137.51, 54.9, 90),
      FreeView("00_02", -84.98, 39.52, 90),
      FreeView("00_03", 52.52, 27.04, 90),
      FreeView("00_04", -169.97, 15.83, 90),
      FreeView("00_05", -32.46, 5.22, 90),
      FreeView("00_06", 105.05, -5.22, 90),
      FreeView("00_07", -117.45, -15.83, 90),
      FreeView("00_08", 20.06, -27.04, 90),
      FreeView("00_09", 157.57, -39.52, 90),
      FreeView("00_10", -64.92, -54.9, 90),
      FreeView("00_11", 0.0, -90.0, 90),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  "medium": ViewConfig(
    views=[
      FreeView("00_00", 0.0, -35.0, 90),
      FreeView("00_01", 45.0, -35.0, 90),
      FreeView("00_02", 90.0, -35.0, 90),
      FreeView("00_03", 135.0, -35.0, 90),
      FreeView("00_04", 180.0, -35.0, 90),
      FreeView("00_05", -135.0, -35.0, 90),
      FreeView("00_06", -90.0, -35.0, 90),
      FreeView("00_07", -45.0, -35.0, 90),
      FreeView("01_00", 22.5, 35.0, 90),
      FreeView("01_01", 67.5, 35.0, 90),
      FreeView("01_02", 112.5, 35.0, 90),
      FreeView("01_03", 157.5, 35.0, 90),
      FreeView("01_04", -157.5, 35.0, 90),
      FreeView("01_05", -112.5, 35.0, 90),
      FreeView("01_06", -67.5, 35.0, 90),
      FreeView("01_07", -22.5, 35.0, 90),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  "high": ViewConfig(
    views=[
      FreeView("00_00", 0.0, 90.0, 90),
      FreeView("00_01", 137.51, 63.47, 90),
      FreeView("00_02", -84.98, 52.14, 90),
      FreeView("00_03", 52.52, 43.17, 90),
      FreeView("00_04", -169.97, 35.38, 90),
      FreeView("00_05", -32.46, 28.27, 90),
      FreeView("00_06", 105.05, 21.62, 90),
      FreeView("00_07", -117.45, 15.26, 90),
      FreeView("00_08", 20.06, 9.08, 90),
      FreeView("00_09", 157.57, 3.02, 90),
      FreeView("00_10", -64.92, -3.02, 90),
      FreeView("00_11", 72.59, -9.08, 90),
      FreeView("00_12", -149.91, -15.26, 90),
      FreeView("00_13", -12.4, -21.62, 90),
      FreeView("00_14", 125.11, -28.27, 90),
      FreeView("00_15", -97.38, -35.38, 90),
      FreeView("00_16", 40.12, -43.17, 90),
      FreeView("00_17", 177.63, -52.14, 90),
      FreeView("00_18", -44.86, -63.47, 90),
      FreeView("00_19", 180.0, -90.0, 90),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  "ultra": ViewConfig(
    views=[
      FreeView("00_00", 0.0, 90.0, 90),
      FreeView("00_01", 137.51, 65.93, 90),
      FreeView("00_02", -84.98, 55.7, 90),
      FreeView("00_03", 52.52, 47.66, 90),
      FreeView("00_04", -169.97, 40.71, 90),
      FreeView("00_05", -32.46, 34.42, 90),
      FreeView("00_06", 105.05, 28.57, 90),
      FreeView("00_07", -117.45, 23.04, 90),
      FreeView("00_08", 20.06, 17.72, 90),
      FreeView("00_09", 157.57, 12.56, 90),
      FreeView("00_10", -64.92, 7.49, 90),
      FreeView("00_11", 72.59, 2.49, 90),
      FreeView("00_12", -149.91, -2.49, 90),
      FreeView("00_13", -12.4, -7.49, 90),
      FreeView("00_14", 125.11, -12.56, 90),
      FreeView("00_15", -97.38, -17.72, 90),
      FreeView("00_16", 40.12, -23.04, 90),
      FreeView("00_17", 177.63, -28.57, 90),
      FreeView("00_18", -44.86, -34.42, 90),
      FreeView("00_19", 92.65, -40.71, 90),
      FreeView("00_20", -129.84, -47.66, 90),
      FreeView("00_21", 7.66, -55.7, 90),
      FreeView("00_22", 145.17, -65.93, 90),
      FreeView("00_23", 0.0, -90.0, 90),
    ],
    include_zenith=False,
    include_nadir=False,
  ),
  # Legacy ring-only presets kept for compatibility
  "prep360_default_legacy": ViewConfig(
    rings=[
      Ring(pitch=0, count=8, fov=65),
      Ring(pitch=-20, count=4, fov=65),
    ],
    include_zenith=True,
    include_nadir=False,
  ),
  "dense_coverage": ViewConfig(
    rings=[
      Ring(pitch=0, count=8, fov=60),
      Ring(pitch=30, count=4, fov=60),
      Ring(pitch=-30, count=4, fov=60),
    ],
    include_zenith=True,
    include_nadir=True,
  ),
  "lightweight": ViewConfig(
    rings=[Ring(pitch=0, count=6, fov=75)],
    include_zenith=True,
    include_nadir=False,
  ),
  "octa_horizon": ViewConfig(
    rings=[Ring(pitch=0, count=8, fov=65)],
    include_zenith=True,
    include_nadir=False,
  ),
  "full_sphere": ViewConfig(
    rings=[
      Ring(pitch=0, count=8, fov=60),
      Ring(pitch=45, count=4, fov=60),
      Ring(pitch=-45, count=4, fov=60),
    ],
    include_zenith=True,
    include_nadir=True,
  ),
}

LEGACY_PRESET_ALIASES = {
  "prep360_default": "medium",
  "dense": "dense_coverage",
}


def create_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
  yaw = np.radians(yaw_deg)
  pitch = np.radians(pitch_deg)
  fwd = np.array([
    np.cos(pitch) * np.sin(yaw),
    np.sin(pitch),
    np.cos(pitch) * np.cos(yaw),
  ])
  r = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
  rl = np.linalg.norm(r)
  if rl < 1e-6:
    r = np.array([1.0, 0.0, 0.0])
  else:
    r = r / rl
  u = np.cross(r, fwd)
  return np.array([r, u, -fwd])


def _create_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0) -> np.ndarray:
  """Compatibility alias for colmap_export and legacy callers (roll ignored)."""
  if roll_deg != 0:
    logger.debug("roll_deg=%s ignored; ERP reframe uses yaw/pitch only", roll_deg)
  return create_rotation_matrix(yaw_deg, pitch_deg)


def reframe_view(
  equirect: np.ndarray,
  fov_deg: float,
  yaw_deg: float,
  pitch_deg: float,
  out_size: int,
  mode: str = "bilinear",
) -> np.ndarray:
  h_eq, w_eq = equirect.shape[:2]
  fov_rad = np.radians(fov_deg)
  f = (out_size / 2) / np.tan(fov_rad / 2)
  cxx = out_size / 2
  cy = out_size / 2

  R = create_rotation_matrix(yaw_deg, pitch_deg)
  r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
  r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
  r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

  px_arr = np.arange(out_size, dtype=np.float64)
  px_grid, py_grid = np.meshgrid(px_arr, px_arr)
  crx = (px_grid - cxx) / f
  cry = -(py_grid - cy) / f

  wx = r00 * crx + r10 * cry - r20
  wy = r01 * crx + r11 * cry - r21
  wz = r02 * crx + r12 * cry - r22
  length = np.sqrt(wx * wx + wy * wy + wz * wz)
  wx, wy, wz = wx / length, wy / length, wz / length

  theta = np.arctan2(wx, wz)
  phi = np.arcsin(np.clip(wy, -1, 1))
  u_eq = ((theta / np.pi + 1) / 2) * w_eq
  v_eq = (0.5 - phi / np.pi) * h_eq

  if mode == "nearest":
    u_idx = np.round(u_eq).astype(int) % w_eq
    v_idx = np.clip(np.round(v_eq).astype(int), 0, h_eq - 1)
    out = equirect[v_idx, u_idx]
    return np.fliplr(out)

  map_x = u_eq.astype(np.float32) % w_eq
  map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)
  result = cv2.remap(
    equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP,
  )
  return np.fliplr(result)


def _build_reframe_remap(
  fov_deg: float,
  yaw_deg: float,
  pitch_deg: float,
  out_size: int,
  erp_w: int,
  erp_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
  fov_rad = np.radians(fov_deg)
  f = (out_size / 2) / np.tan(fov_rad / 2)
  cxx = out_size / 2
  cy = out_size / 2
  R = create_rotation_matrix(yaw_deg, pitch_deg)
  r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
  r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
  r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

  px_arr = np.arange(out_size, dtype=np.float64)
  px_grid, py_grid = np.meshgrid(px_arr, px_arr)
  crx = (px_grid - cxx) / f
  cry = -(py_grid - cy) / f

  wx = r00 * crx + r10 * cry - r20
  wy = r01 * crx + r11 * cry - r21
  wz = r02 * crx + r12 * cry - r22
  length = np.sqrt(wx * wx + wy * wy + wz * wz)
  wx, wy, wz = wx / length, wy / length, wz / length

  theta = np.arctan2(wx, wz)
  phi = np.arcsin(np.clip(wy, -1, 1))
  u_eq = ((theta / np.pi + 1) / 2) * erp_w
  v_eq = (0.5 - phi / np.pi) * erp_h

  map_x = u_eq.astype(np.float32) % erp_w
  map_y = np.clip(v_eq.astype(np.float32), 0, erp_h - 1)
  map_x = np.ascontiguousarray(np.fliplr(map_x))
  map_y = np.ascontiguousarray(np.fliplr(map_y))
  return map_x, map_y


def _apply_reframe_remap(
  image: np.ndarray,
  map_x: np.ndarray,
  map_y: np.ndarray,
  mode: str = "bilinear",
) -> np.ndarray:
  interp = cv2.INTER_NEAREST if mode == "nearest" else cv2.INTER_LINEAR
  return cv2.remap(image, map_x, map_y, interp, borderMode=cv2.BORDER_WRAP)


def compute_pinhole_intrinsics(fov_deg: float, crop_size: int) -> dict:
  f = crop_size / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
  cx = crop_size / 2.0
  cy = crop_size / 2.0
  return {
    "model": "PINHOLE",
    "width": crop_size,
    "height": crop_size,
    "fx": round(f, 4),
    "fy": round(f, 4),
    "cx": cx,
    "cy": cy,
  }


@dataclass
class ReframePaths:
  image_path: Path
  mask_path: Optional[Path]
  relative_image: str


def resolve_output_paths(
  output_root: Path,
  layout: OutputLayout,
  view_name: str,
  stem: str,
  has_masks: bool,
) -> ReframePaths:
  images_root = output_root / "images"
  masks_root = output_root / "masks"

  if layout == OutputLayout.RIG:
    image_path = images_root / view_name / f"{stem}.jpg"
    relative = f"images/{view_name}/{stem}.jpg"
    mask_path = masks_root / view_name / f"{stem}.png" if has_masks else None
  elif layout == OutputLayout.STATION:
    fname = f"{stem}_{view_name}.jpg"
    image_path = images_root / stem / fname
    relative = f"images/{stem}/{fname}"
    mask_path = masks_root / stem / f"{stem}_{view_name}.png" if has_masks else None
  else:
    fname = f"{stem}_{view_name}.jpg"
    image_path = images_root / fname
    relative = f"images/{fname}"
    mask_path = masks_root / f"{stem}_{view_name}.png" if has_masks else None

  return ReframePaths(image_path=image_path, mask_path=mask_path, relative_image=relative)


def _layout_from_legacy(
  station_dirs: Optional[bool],
  layout: Optional[OutputLayout],
) -> OutputLayout:
  if layout is not None:
    return layout
  if station_dirs is True:
    return OutputLayout.STATION
  if station_dirs is False:
    return OutputLayout.FLAT
  return OutputLayout.RIG


def _process_mask_crop(mask_persp: np.ndarray) -> np.ndarray:
  if mask_persp.ndim == 3:
    mask_persp = mask_persp[:, :, 0]
  out = (mask_persp > 0).astype(np.uint8) * 255
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  return cv2.erode(out, kernel, iterations=1)


def _collect_image_files(directory: Path) -> List[Path]:
  extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
  seen: set[str] = set()
  files: List[Path] = []
  for ext in extensions:
    for path in directory.glob(ext):
      norm = os.path.normcase(str(path.resolve()))
      if norm in seen:
        continue
      seen.add(norm)
      files.append(path)
  return sorted(files, key=lambda p: os.path.normcase(str(p)))


def _normalize_mask_stem(stem: str) -> str:
  normalized = re.sub(r"(?i)^mask[_-]", "", stem)
  normalized = re.sub(r"(?i)[_-]mask$", "", normalized)
  return os.path.normcase(normalized)


def _discover_mask_scan_dir(mask_dir: Path) -> tuple[Path, str]:
  """Choose where ERP masks live: flat root or Masking Studio ``layers/<name>/``."""
  mask_dir = mask_dir.resolve()
  if _collect_image_files(mask_dir):
    return mask_dir, "mask root"

  layers_root = mask_dir / "layers"
  if not layers_root.is_dir():
    return mask_dir, "mask root"

  populated_layers: List[tuple[Path, int]] = []
  for layer_dir in sorted(layers_root.iterdir(), key=lambda p: os.path.normcase(p.name)):
    if not layer_dir.is_dir():
      continue
    count = len(_collect_image_files(layer_dir))
    if count:
      populated_layers.append((layer_dir, count))

  if not populated_layers:
    return mask_dir, "mask root"

  if len(populated_layers) > 1:
    names = ", ".join(path.name for path, _ in populated_layers[:5])
    logger.warning(
      "Multiple mask layers found under %s; using %s (%d files). Others: %s",
      layers_root,
      populated_layers[0][0].name,
      populated_layers[0][1],
      names,
    )

  layer_dir, _ = populated_layers[0]
  return layer_dir, f"layers/{layer_dir.name}"


def _build_mask_map(
  mask_dir: Path,
  log: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, str], List[str], str, Path]:
  """Map normalized frame stems to ERP mask file paths."""
  scan_dir, source_label = _discover_mask_scan_dir(mask_dir)
  mask_files = _collect_image_files(scan_dir)
  mask_map: dict[str, str] = {}
  duplicate_mask_stems: List[str] = []

  for mask_path in mask_files:
    keys = {os.path.normcase(mask_path.stem), _normalize_mask_stem(mask_path.stem)}
    if any(key in mask_map for key in keys):
      duplicate_mask_stems.append(mask_path.stem)
      continue
    path_str = str(mask_path)
    for key in keys:
      mask_map[key] = path_str

  if log:
    log(f"Mask scan: {len(mask_files)} file(s) in {source_label}")
    if mask_files and not mask_map:
      log("Warning: mask files were found but none produced usable stem keys")

  return mask_map, duplicate_mask_stems, source_label, scan_dir


def _lookup_mask_path(
  frame_stem: str,
  mask_map: dict[str, str],
  scan_dir: Optional[Path] = None,
) -> Optional[str]:
  frame_key = os.path.normcase(frame_stem)
  if frame_key in mask_map:
    return mask_map[frame_key]

  if scan_dir is None:
    return None

  for pattern in (f"mask_{frame_stem}.*", f"{frame_stem}_mask.*"):
    matches = sorted(scan_dir.glob(pattern), key=lambda p: os.path.normcase(p.name))
    if matches:
      return str(matches[0])

  return None


class _RemapCache:
  """Lazy per-view remap tables keyed by ERP dimensions."""

  def __init__(self, config: ViewConfig):
    self._config = config
    self._key: Optional[Tuple[int, int]] = None
    self._maps: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

  def get_maps(self, erp_w: int, erp_h: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    key = (erp_w, erp_h)
    if self._maps is not None and self._key == key:
      return self._maps
    maps = []
    for view in self._config.get_all_views():
      maps.append(_build_reframe_remap(
        view.fov, view.yaw, view.pitch, self._config.output_size, erp_w, erp_h,
      ))
    self._key = key
    self._maps = maps
    logger.debug("Built remap cache: %d views for ERP %dx%d", len(maps), erp_w, erp_h)
    return maps


def _process_single_image(
  image_path: str,
  output_root: Path,
  config: ViewConfig,
  mask_path: Optional[str],
  output_layout: OutputLayout,
  remap_cache: _RemapCache,
) -> Tuple[List[str], Optional[str]]:
  equirect = cv2.imread(image_path)
  if equirect is None:
    return [], f"Failed to load {image_path}"

  mask = None
  if mask_path:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
      return [], f"Failed to load mask {mask_path}"
    if mask.shape[:2] != equirect.shape[:2]:
      return [], (
        f"Mask dimensions {mask.shape[:2]} don't match image "
        f"{equirect.shape[:2]} for {image_path}"
      )

  stem = Path(image_path).stem
  h_eq, w_eq = equirect.shape[:2]
  maps = remap_cache.get_maps(w_eq, h_eq)
  views = config.get_all_views()
  output_files: List[str] = []

  for vi, view in enumerate(views):
    map_x, map_y = maps[vi]
    persp = _apply_reframe_remap(equirect, map_x, map_y, mode="bilinear")
    if view.flip_vertical:
      persp = np.flipud(persp)

    paths = resolve_output_paths(
      output_root, output_layout, view.name, stem, mask is not None,
    )
    paths.image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
      str(paths.image_path), persp,
      [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
    )
    output_files.append(paths.relative_image)

    if mask is not None and paths.mask_path is not None:
      mask_persp = _apply_reframe_remap(mask, map_x, map_y, mode="nearest")
      if view.flip_vertical:
        mask_persp = np.flipud(mask_persp)
      mask_out = _process_mask_crop(mask_persp)
      paths.mask_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(paths.mask_path), mask_out)

  return output_files, None


@dataclass
class ReframeResult:
  success: bool
  input_count: int
  output_count: int
  output_dir: str
  errors: List[str] = field(default_factory=list)


class Reframer:
  def __init__(self, config: Optional[ViewConfig] = None, preset_name: str = DEFAULT_PRESET):
    if config is None:
      config = get_view_preset(preset_name)
    else:
      config = copy_view_config(config)
    validate_view_config(config)
    self.config = config
    self.preset_name = preset_name

  def reframe_single(
    self,
    image_path: str,
    output_dir: str,
    mask_path: Optional[str] = None,
    output_layout: Optional[OutputLayout] = None,
    station_dirs: Optional[bool] = None,
  ) -> Tuple[List[str], Optional[str]]:
    layout = _layout_from_legacy(station_dirs, output_layout)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = _RemapCache(self.config)
    return _process_single_image(
      image_path, output_root, self.config, mask_path, layout, cache,
    )

  def reframe_batch(
    self,
    input_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    num_workers: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    output_layout: Optional[OutputLayout] = None,
    station_dirs: Optional[bool] = None,
    log: Optional[Callable[[str], None]] = None,
    preset_name: Optional[str] = None,
  ) -> ReframeResult:
    from .reframe_metadata import log_pinhole_intrinsics, write_reframe_metadata
    from .rig_config import write_rig_config

    layout = _layout_from_legacy(station_dirs, output_layout)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)

    input_path = Path(input_dir)
    images = _collect_image_files(input_path)

    def _log(msg: str) -> None:
      if log:
        log(msg)

    if not images:
      return ReframeResult(
        success=False, input_count=0, output_count=0,
        output_dir=str(output_root), errors=["No images found in input directory"],
      )

    mask_map: dict[str, str] = {}
    duplicate_mask_stems: List[str] = []
    mask_source_label = ""
    mask_scan_dir: Optional[Path] = None
    if mask_dir:
      mask_map, duplicate_mask_stems, mask_source_label, mask_scan_dir = _build_mask_map(
        Path(mask_dir), _log,
      )
      if duplicate_mask_stems:
        _log(
          "Warning: duplicate mask stems ignored: "
          + ", ".join(sorted(set(duplicate_mask_stems))[:5])
        )
      if not mask_map:
        _log(
          f"Warning: no ERP mask files found under {mask_dir}"
          + (f" (checked {mask_source_label})" if mask_source_label else "")
        )
      matched = sum(
        1 for img in images
        if _lookup_mask_path(img.stem, mask_map, mask_scan_dir) is not None
      )
      unmatched = len(images) - matched
      if unmatched > 0:
        _log(f"Warning: {unmatched} of {len(images)} frames have no matching mask")
        if matched == 0 and images:
          frame_sample = ", ".join(img.stem for img in images[:3])
          mask_sample = ", ".join(Path(p).stem for p in list(mask_map.values())[:3])
          if mask_sample:
            _log(f"  Example frame stems: {frame_sample}")
            _log(f"  Example mask stems: {mask_sample}")
      _log(f"Masks: {matched}/{len(images)} matched")

    _log(
      f"Reframer: {len(images)} images, {self.config.total_views()} views/frame, "
      f"layout={layout.value}"
    )
    _log(f"  Output root: {output_root}")
    _log(f"  Crop: {self.config.output_size}px, JPEG q{self.config.jpeg_quality}")
    _log("  Remap strategy: lazy per-view cache (in-process)")

    if num_workers > 1:
      _log("  Note: num_workers>1 falls back to sequential (remap cache not pickled)")
      num_workers = 1

    cache = _RemapCache(self.config)
    errors: List[str] = []
    total_outputs = 0
    t0 = time.perf_counter()

    for i, img in enumerate(images):
      outputs, error = _process_single_image(
        str(img), output_root, self.config,
        _lookup_mask_path(img.stem, mask_map, mask_scan_dir) if mask_dir else None,
        layout, cache,
      )
      if error:
        errors.append(error)
      elif outputs:
        total_outputs += len(outputs)
      if progress_callback:
        progress_callback(i + 1, len(images), img.name)

    elapsed = time.perf_counter() - t0
    _log(
      f"Reframing complete: {len(images)} frames → {total_outputs} crops "
      f"({elapsed:.1f}s, {elapsed / max(len(images), 1):.2f}s/frame)"
    )
    if errors:
      _log(f"  Errors: {len(errors)}")
      for err in errors[:3]:
        _log(f"    {err}")

    meta_preset = preset_name or self.preset_name
    if not errors:
      write_reframe_metadata(
        output_root, self.config, layout, meta_preset, images,
      )
      if layout == OutputLayout.RIG:
        write_rig_config(self.config, str(output_root / "rig_config.json"))
      log_pinhole_intrinsics(self.config, layout, log)

    return ReframeResult(
      success=len(errors) == 0,
      input_count=len(images),
      output_count=total_outputs,
      output_dir=str(output_root),
      errors=errors,
    )

  def preview_view_positions(self) -> str:
    lines = [f"Total views: {self.config.total_views()}"]
    for i, ring in enumerate(self.config.rings):
      spacing = 360.0 / ring.count if ring.count > 0 else 0
      lines.append(
        f"Ring {i}: pitch={ring.pitch:+.0f}° count={ring.count} "
        f"spacing={spacing:.0f}° fov={ring.fov}°"
      )
    for fv in self.config.views:
      lines.append(f"FreeView {fv.name}: yaw={fv.yaw:+.1f}° pitch={fv.pitch:+.1f}° fov={fv.fov}°")
    if self.config.include_zenith:
      lines.append(f"Zenith: pitch=+90° fov={self.config.zenith_fov}°")
    if self.config.include_nadir:
      lines.append(f"Nadir: pitch=-90° fov={self.config.zenith_fov}°")
    return "\n".join(lines)


def _run_reframe_cli(argv: Optional[List[str]] = None) -> int:
  import argparse
  import sys
  import warnings

  parser = argparse.ArgumentParser(description="Reframe equirectangular images to perspective views")
  parser.add_argument("input", nargs="?", help="Input directory with equirectangular images")
  parser.add_argument("output", nargs="?", help="Output dataset root")
  parser.add_argument("--preset", "-p", default=DEFAULT_PRESET, help="View preset")
  parser.add_argument("--size", "-s", type=int, default=1920, help="Output size")
  parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality")
  parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
  parser.add_argument("--layout", choices=[m.value for m in OutputLayout], default=None)
  parser.add_argument("--mask-dir", help="ERP masks directory")
  parser.add_argument("--stations", action="store_true", help="Deprecated: use --layout station")
  parser.add_argument("--info", action="store_true", help="Show preset info")
  parser.add_argument("--list-presets", action="store_true", help="List presets")

  args = parser.parse_args(argv)

  if args.list_presets:
    for name in sorted(VIEW_PRESETS.keys()):
      if name == "prep360_default":
        continue
      print(f"  {name}: {VIEW_PRESETS[name].total_views()} views")
    return 0

  layout = OutputLayout(args.layout or OutputLayout.RIG.value)
  if args.stations:
    if args.layout is not None and args.layout != OutputLayout.STATION.value:
      print(
        "Error: --stations conflicts with --layout; use --layout station",
        file=sys.stderr,
      )
      return 2
    warnings.warn("--stations is deprecated; use --layout station", DeprecationWarning, stacklevel=2)
    layout = OutputLayout.STATION

  try:
    preset_key = resolve_preset_name(args.preset)
    config = get_view_preset(preset_key)
  except KeyError:
    print(f"Unknown preset: {args.preset}", file=sys.stderr)
    return 1
  config.output_size = args.size
  config.jpeg_quality = args.quality
  validate_view_config(config)

  reframer = Reframer(config, preset_name=preset_key)

  if args.info:
    print(f"Preset: {args.preset} (-> {preset_key})")
    print(reframer.preview_view_positions())
    return 0

  if not args.input or not args.output:
    parser.error("input and output directories are required")

  def progress(current: int, total: int, name: str) -> None:
    print(f"[{current}/{total}] {name}")

  result = reframer.reframe_batch(
    args.input,
    args.output,
    mask_dir=args.mask_dir,
    num_workers=args.workers,
    progress_callback=progress,
    output_layout=layout,
    preset_name=preset_key,
  )

  if result.success:
    print(f"Reframed {result.input_count} images -> {result.output_count} views ({layout.value} layout)")
    return 0
  print(f"Errors: {len(result.errors)}", file=__import__("sys").stderr)
  for err in result.errors[:5]:
    print(f"  {err}", file=__import__("sys").stderr)
  return 1


def main() -> int:
  return _run_reframe_cli()


if __name__ == "__main__":
  raise SystemExit(main())

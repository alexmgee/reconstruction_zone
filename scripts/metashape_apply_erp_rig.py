"""
Apply ERP rig calibration from reframe_metadata.json inside Metashape Pro.

Tools > Run Script > this file. One script, no mode switching:

  - No alignment yet  →  configures rig (intrinsics + slave rotations)
  - Already aligned   →  prints tie-point / per-sensor stats

Do NOT paste exec(open(...).read()) from chat — use Run Script instead.
Do NOT type verify_alignment_status() in the console (stale copies cause errors).

Metashape 2.3 multi-camera rig (ERP cubemap):
  - Workspace shows ONE row per rig pose (one frustum); six sensors are bundled inside.
  - A line of frustums along the path = camera movement over time (expected).
  - Tie points come from TEMPORAL matching within each sensor folder, not from
    matching orthogonal cubemap faces at the same timestamp (they do not overlap).
  - One enabled pose rarely produces a full 360 tie cloud — use many poses / full dataset.

Virtual zero-baseline cubemap uses zero translation and fixed slave rotations.

Rotations use reframe_config yaw/pitch (same world_from_cam matrices as prep360
reframer). Metashape slave offset is slave→master (forum #16123):

    P_master = R @ P_slave

which is the inverse of COLMAP cam_from_rig (R_i @ R_ref.T). Do not use the
cam_from_rig quaternions in reframe_metadata.json directly — they are COLMAP-facing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


# Tight priors: Metashape accuracy = std dev; small values = effectively fixed rig.
_LOC_ACCURACY_M = 1e-6
_ROT_ACCURACY_DEG = 1e-4


def _create_rotation_matrix(yaw_deg: float, pitch_deg: float):
    """Match prep360.core.reframer.create_rotation_matrix (rows = camera basis in world)."""
    import numpy as np

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fwd = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ])
    r = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    rl = np.linalg.norm(r)
    if rl < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / rl
    u = np.cross(r, fwd)
    return np.array([r, u, -fwd])


def _world_from_cam_matrix(yaw_deg: float, pitch_deg: float):
    """
    Same as prep360 rig_config / COLMAP cam_from_rig — yaw/pitch only.

    flip_vertical affects saved JPEGs (flipud) but is NOT folded into rig
    relative rotations (see lichtfeld-360-plugin rig_config.py).
    """
    return _create_rotation_matrix(yaw_deg, pitch_deg)


def _metashape_slave_to_master(ref_view: dict, slave_view: dict):
    """
    Metashape slave offset: P_master = R @ P_slave (forum #16123).

    Inverse of COLMAP cam_from_rig (R_slave @ R_ref.T) from prep360 rig_config.
    """
    R_ref = _world_from_cam_matrix(ref_view["yaw"], ref_view["pitch"])
    R_slave = _world_from_cam_matrix(slave_view["yaw"], slave_view["pitch"])
    return R_ref @ R_slave.T


def _colmap_cam_from_rig(ref_view: dict, slave_view: dict):
    """COLMAP cam_from_rig_rotation matrix (prep360 rig_config convention)."""
    R_ref = _world_from_cam_matrix(ref_view["yaw"], ref_view["pitch"])
    R_slave = _world_from_cam_matrix(slave_view["yaw"], slave_view["pitch"])
    return R_slave @ R_ref.T


def _folder_from_path(photo_path: str) -> Optional[str]:
    parts = Path(photo_path.replace("\\", "/")).parts
    if "images" not in parts:
        return None
    idx = parts.index("images")
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return None


def _build_folder_to_sensor_map(chunk) -> dict[str, object]:
    folder_to_sensor: dict[str, object] = {}
    for camera in chunk.cameras:
        if not camera or not camera.photo or not camera.photo.path:
            continue
        folder = _folder_from_path(camera.photo.path)
        if folder and camera.sensor:
            folder_to_sensor[folder] = camera.sensor
    return folder_to_sensor


def _debug_sensor_paths(chunk, limit: int = 6) -> None:
    seen: set[int] = set()
    for camera in chunk.cameras:
        sensor = camera.sensor
        if sensor is None or id(sensor) in seen:
            continue
        seen.add(id(sensor))
        sample = camera.photo.path if camera.photo else "(no path)"
        print(f"  sensor key={sensor.key} label={sensor.label!r} sample={sample}")
        if len(seen) >= limit:
            break


def _reset_rig_alignment(chunk) -> None:
    """Clear stale alignment so Slave Offset Adjusted column does not show import defaults."""
    chunk.point_cloud = None
    for camera in chunk.cameras:
        camera.transform = None


def _apply_slave_offset(Metashape, sensor, opk_vec, zero, *, hard_lock: bool) -> None:
    """
    Apply slave offset (forum #15021 / #16123).

    ref.enabled must be True or Metashape ignores the offset values.
    """
    ref = sensor.reference
    rot_mat = Metashape.utils.opk2mat(opk_vec)

    ref.enabled = True
    ref.location = zero
    ref.rotation = opk_vec
    ref.location_accuracy = Metashape.Vector([_LOC_ACCURACY_M] * 3)
    ref.rotation_accuracy = Metashape.Vector([_ROT_ACCURACY_DEG] * 3)

    for attr in ("location_enabled", "rotation_enabled"):
        if hasattr(ref, attr):
            setattr(ref, attr, True)

    sensor.location = zero
    sensor.rotation = rot_mat

    if hard_lock:
        sensor.fixed_location = True
        sensor.fixed_rotation = True
    else:
        sensor.fixed_location = False
        sensor.fixed_rotation = False


def verify_erp_rig(chunk=None) -> None:
    """Dump chunk.sensors slave offsets — run after apply_erp_rig to compare with GUI."""
    import Metashape

    if chunk is None:
        chunk = Metashape.app.document.chunk
    if chunk is None:
        raise RuntimeError("No active chunk")

    print("verify_erp_rig() — chunk.sensors readback:")
    for sensor in chunk.sensors:
        master = sensor.master
        if master is not None and sensor.key != master.key:
            opk = Metashape.utils.mat2opk(sensor.rotation)
            ref = sensor.reference
            ref_opk = ref.rotation if ref.enabled else "(ref disabled)"
            print(
                f"  {sensor.label!r} key={sensor.key}"
                f"  fixed=({sensor.fixed_location},{sensor.fixed_rotation})"
                f"  ref.enabled={ref.enabled}"
                f"  mat2opk=({opk.x:.3f},{opk.y:.3f},{opk.z:.3f})"
                f"  ref.rotation={ref_opk}"
            )


def _frame_stem_from_path(photo_path: str) -> Optional[str]:
    return Path(photo_path.replace("\\", "/")).stem


def _normalize_pose_label(label: str) -> str:
    """Workspace labels look like 'frame_000001, NA'."""
    return label.split(",")[0].strip()


def _cameras_in_pose(camera) -> list:
    """Multi-camera import: one Workspace row may contain sensor sub-cameras."""
    frames = getattr(camera, "frames", None)
    if frames:
        return list(frames)
    return [camera]


def _pose_stem(camera) -> Optional[str]:
    if camera.label:
        stem = _normalize_pose_label(camera.label)
        if stem:
            return stem
    if camera.photo and camera.photo.path:
        return _frame_stem_from_path(camera.photo.path)
    return None


def enable_single_rig_pose(chunk=None, frame_stem: str | None = None) -> str:
    """
    Enable one rig instant for smoke testing.

    Multi-camera import shows ONE Workspace row per frame (e.g. frame_000001).
    That single row contains all 6 sensor images (00_00 … 02_00) — you do NOT
    enable six separate rows.

    GUI: Workspace > Cameras — enable one frame row, disable the rest (red minus).
    """
    import Metashape

    if chunk is None:
        chunk = Metashape.app.document.chunk
    if chunk is None:
        raise RuntimeError("No active chunk")

    stems: set[str] = set()
    for camera in chunk.cameras:
        stem = _pose_stem(camera)
        if stem:
            stems.add(stem)

    if not stems:
        raise RuntimeError("No cameras with labels or photo paths in chunk")

    if frame_stem is None:
        frame_stem = sorted(stems)[0]
    else:
        frame_stem = _normalize_pose_label(frame_stem)
        if frame_stem not in stems:
            sample = ", ".join(sorted(stems)[:5])
            raise RuntimeError(f"Frame {frame_stem!r} not found. Examples: {sample}")

    enabled_poses = 0
    enabled_sensor_cams = 0
    disabled_count = 0
    sensor_paths: list[str] = []

    for camera in chunk.cameras:
        stem = _pose_stem(camera)
        enable_pose = stem == frame_stem
        if enable_pose:
            enabled_poses += 1
        for cam in _cameras_in_pose(camera):
            cam.enabled = enable_pose
            if enable_pose:
                enabled_sensor_cams += 1
                if cam.photo and cam.photo.path:
                    folder = _folder_from_path(cam.photo.path) or "?"
                    sensor_paths.append(f"{folder}/{Path(cam.photo.path).name}")
            else:
                disabled_count += 1

    print(f"enable_single_rig_pose({frame_stem!r})")
    print(f"  Rig poses enabled: {enabled_poses} (Workspace rows — expect 1)")
    print(f"  Sensor cameras enabled: {enabled_sensor_cams} (expect 6 for cubemap)")
    print(f"  Disabled camera entries: {disabled_count}")
    if sensor_paths:
        print("  Enabled images:")
        for line in sorted(sensor_paths):
            print(f"    {line}")
    if enabled_poses != 1:
        print(f"  NOTE: {enabled_poses} pose rows enabled (GUI usually shows one row per pose).")
    if enabled_sensor_cams not in (1, 6) and enabled_poses == 1:
        print(f"  NOTE: {enabled_sensor_cams} sensor cameras — OK if Metashape bundles sensors internally.")
    return frame_stem


def enable_all_rig_poses(chunk=None) -> None:
    """Re-enable every camera / rig pose in the chunk."""
    import Metashape

    if chunk is None:
        chunk = Metashape.app.document.chunk
    if chunk is None:
        raise RuntimeError("No active chunk")

    n = 0
    for camera in chunk.cameras:
        for cam in _cameras_in_pose(camera):
            cam.enabled = True
            n += 1
    print(f"enable_all_rig_poses(): enabled {n} camera entries")


def _iter_all_cameras(chunk):
    for camera in chunk.cameras:
        yield from _cameras_in_pose(camera)


def _tie_point_count(tp) -> int:
    try:
        return len(tp.points)
    except TypeError:
        return sum(1 for _ in tp.points)


def _count_valid_projections(chunk, tp) -> tuple[dict[str, int], dict[str, int]]:
    """
    Valid tie-point projections per sensor folder (matches Reference pane counts).
    Returns (valid_counts, total_projection_counts) keyed by sensor folder label.
    """
    points = tp.points
    npoints = _tie_point_count(tp)
    projections = tp.projections

    point_ids = [-1] * len(tp.tracks)
    for point_id in range(npoints):
        point_ids[points[point_id].track_id] = point_id

    valid: dict[str, int] = {}
    total: dict[str, int] = {}

    for camera in _iter_all_cameras(chunk):
        if not camera.transform:
            continue
        folder = "?"
        if camera.photo and camera.photo.path:
            folder = _folder_from_path(camera.photo.path) or "?"
        elif camera.sensor:
            folder = camera.sensor.label

        try:
            cam_projs = projections[camera]
        except (KeyError, TypeError):
            continue

        total[folder] = total.get(folder, 0) + len(cam_projs)
        n_valid = 0
        for proj in cam_projs:
            point_id = point_ids[proj.track_id]
            if point_id < 0:
                continue
            if points[point_id].valid:
                n_valid += 1
        valid[folder] = valid.get(folder, 0) + n_valid

    return valid, total


def verify_alignment_status(chunk=None) -> None:
    """
    Summarize alignment after matchPhotos/alignCameras.

    API note: chunk.cameras has one entry per image (482 frames x 6 sensors = 2892).
    The Workspace UI groups them into 482 rig-pose rows.
    """
    import Metashape

    if chunk is None:
        chunk = Metashape.app.document.chunk
    if chunk is None:
        raise RuntimeError("No active chunk")

    all_cams = list(_iter_all_cameras(chunk))
    if not all_cams:
        all_cams = list(chunk.cameras)

    enabled = [c for c in all_cams if c.enabled]
    aligned = [c for c in all_cams if c.transform is not None]

    # Unique frame stems among enabled cameras
    enabled_stems = {_pose_stem(c) for c in enabled if _pose_stem(c)}

    print("verify_alignment_status()")
    print(
        f"  Camera entries: {len(all_cams)} total, {len(enabled)} enabled, "
        f"{len(aligned)} aligned"
    )
    print(f"  Enabled rig poses (unique frame names): {len(enabled_stems)}")

    if not aligned:
        print("  No aligned cameras — run Workflow > Align Photos first.")
        return

    tp = chunk.tie_points
    if tp is None:
        print("  Tie points: none (alignment failed or not run)")
        return

    n_tp = _tie_point_count(tp)
    print(f"  Tie points: {n_tp}")

    try:
        valid, total = _count_valid_projections(chunk, tp)
    except Exception as exc:
        print(f"  Per-sensor projection stats failed: {exc}")
        return
    if valid:
        print("  Valid tie-point projections by sensor folder:")
        sensors = sorted(set(valid) | set(total))
        for folder in sensors:
            v = valid.get(folder, 0)
            t = total.get(folder, 0)
            print(f"    {folder}: {v} valid ({t} raw projections)")
        active = sum(1 for f in sensors if valid.get(f, 0) > 0)
        print(f"  Sensors contributing tie points: {active}/{len(sensors)}")
        if active < len(sensors):
            print("  Missing sensors did not match temporally — check slave rotations or enable more frames.")
    else:
        print("  Could not compute per-sensor projection counts.")


def verify_aligned_rig(chunk=None) -> None:
    """Alias for verify_alignment_status."""
    verify_alignment_status(chunk)


def apply_erp_rig(
    metadata_path: str,
    chunk=None,
    *,
    reset_alignment: bool = True,
    lock_rig: bool = False,
) -> None:
    import Metashape

    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if meta.get("output_layout") != "rig":
        raise ValueError("reframe_metadata.json output_layout is not 'rig'")

    doc = Metashape.app.document
    if chunk is None:
        chunk = doc.chunk
    if chunk is None:
        raise RuntimeError("No active chunk")

    chunk.euler_angles = Metashape.EulerAngles.EulerAnglesOPK

    if reset_alignment:
        print("  Clearing stale alignment (point cloud + camera transforms)...")
        _reset_rig_alignment(chunk)

    intr = meta["default_pinhole_intrinsics"]
    by_fov = meta.get("pinhole_intrinsics_by_fov", {})
    views_by_name = {v["name"]: v for v in meta["reframe_config"]["views"]}
    ref_view = meta["reframe_config"]["views"][0]
    ref_folder = ref_view["name"]

    folder_to_sensor = _build_folder_to_sensor_map(chunk)
    missing = sorted(set(views_by_name) - set(folder_to_sensor))
    if missing:
        print("Debug: first cameras per sensor:")
        _debug_sensor_paths(chunk)
        raise RuntimeError(
            "Could not map metadata folders to chunk sensors: "
            + ", ".join(missing)
            + ". Re-import via Workflow > Add Folder on the images/ parent."
        )

    ref_sensor = folder_to_sensor[ref_folder]
    ref_sensor.makeMaster()

    print(f"Applying ERP rig from {metadata_path}")
    print(f"  Master: {ref_folder}/")
    mode = "reference priors (adjust ON)" if not lock_rig else "hard lock (fixed location+rotation)"
    print(f"  Slave offsets: {mode}")

    identity = Metashape.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    zero = Metashape.Vector([0, 0, 0])

    f_px = float(intr["fx"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    for folder, sensor in sorted(folder_to_sensor.items()):
        view = views_by_name[folder]
        fov_key = f"{float(view['fov']):g}"
        f_px = float(intr["fx"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        width = int(intr["width"])
        height = int(intr["height"])
        if fov_key in by_fov:
            si = by_fov[fov_key]
            f_px = float(si["fx"])
            cx = float(si["cx"])
            cy = float(si["cy"])
            width = int(si["width"])
            height = int(si["height"])

        calib = sensor.user_calib
        if calib is None:
            calib = Metashape.Calibration()
        calib.f = f_px
        calib.cx = cx
        calib.cy = cy
        calib.width = width
        calib.height = height
        calib.k1 = calib.k2 = calib.k3 = calib.k4 = 0.0
        calib.p1 = calib.p2 = 0.0
        calib.b1 = calib.b2 = 0.0
        sensor.user_calib = calib
        sensor.type = Metashape.Sensor.Frame
        sensor.fixed_calibration = True
        sensor.label = folder

        if folder == ref_folder:
            ref = sensor.reference
            ref.enabled = False
            ref.location = zero
            ref.rotation = Metashape.Vector([0.0, 0.0, 0.0])
            sensor.location = zero
            sensor.rotation = identity
            sensor.fixed_location = True
            sensor.fixed_rotation = True
            print(f"    {folder}: master")
            continue

        R_ms = _metashape_slave_to_master(ref_view, view)
        opk = Metashape.utils.mat2opk(Metashape.Matrix(R_ms.tolist()))
        opk_vec = Metashape.Vector([opk.x, opk.y, opk.z])
        _apply_slave_offset(Metashape, sensor, opk_vec, zero, hard_lock=lock_rig)

        ref_obj = sensor.reference
        print(
            f"    {folder}: ref=({ref_obj.rotation.x:.3f},{ref_obj.rotation.y:.3f},{ref_obj.rotation.z:.3f})"
            f"  mat2opk=({opk.x:.3f},{opk.y:.3f},{opk.z:.3f})"
            f"  ref.enabled={ref_obj.enabled}"
            f"  (yaw={view['yaw']}, pitch={view['pitch']})"
        )

    print(f"  Sensors: {len(folder_to_sensor)}  Intrinsics: f={f_px}, cx={cx}, cy={cy} (fixed)")
    print()
    if lock_rig:
        print("  GUI: Adjust location=OFF, Adjust rotation=OFF (fixed_location/rotation=True).")
    else:
        print("  GUI: Adjust location=ON, Adjust rotation=ON — Metashape may refine slave offsets.")
    print("  ref.enabled=True on slaves (required — forum #15021).")
    print()
    print("  Enable many frames, align, Run Script again for per-sensor stats.")


def _has_alignment(chunk) -> bool:
    """True when alignCameras has run — cameras have transforms and tie points exist."""
    if chunk is None:
        return False
    if chunk.tie_points is None:
        return False
    return any(c.transform for c in chunk.cameras)


METADATA_JSON = r"D:\Edgeworks\deliverables\grand_island-ermas_desire\input\osmo360\reframes\pinhole_rig\cubemap_thru482\reframe_metadata.json"

if __name__ == "__main__":
    import Metashape

    chunk = Metashape.app.document.chunk
    if _has_alignment(chunk):
        print("Alignment found — printing stats.\n")
        verify_alignment_status(chunk)
    else:
        apply_erp_rig(METADATA_JSON, chunk)
elif __name__ == "__console__":
    pass  # exec() from console; functions stay in console namespace

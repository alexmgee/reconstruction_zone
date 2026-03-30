"""
Spatial Gap Detection in COLMAP Reconstructions

Analyze COLMAP output to find:
    1. Disconnected components (cameras that cluster separately)
    2. Sparse regions (weak geometry between cameras)
    3. Failed images (not in the reconstruction at all)

Maps gaps to video timestamps when possible, enabling automated
bridge frame extraction (Phase 6).

Reads standard COLMAP text format — works with output from:
    - multicam-extract.py
    - Option B decoupled export
    - Any COLMAP-compatible reconstruction

Usage:
    detector = GapDetector()
    report = detector.analyze(colmap_dir)
    print(report.summary())

    # With timestamp mapping
    report = detector.analyze(colmap_dir, source_images_dir="./frames/")
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraPosition:
    """Camera position extracted from COLMAP images.txt."""
    image_id: int
    name: str
    position: np.ndarray        # 3D world position
    qvec: np.ndarray            # quaternion (qw, qx, qy, qz)
    tvec: np.ndarray            # translation
    camera_id: int
    timestamp: Optional[float] = None  # seconds, if extractable from filename

    def distance_to(self, other: "CameraPosition") -> float:
        return float(np.linalg.norm(self.position - other.position))


@dataclass
class SpatialGap:
    """A detected gap in camera coverage."""
    gap_type: str               # "disconnected" or "sparse"
    center: np.ndarray          # midpoint of the gap
    extent: float               # approximate gap size (distance)
    component_a: int            # component index on one side
    component_b: int            # component index on other side
    cameras_before: List[str]   # nearest cameras on side A
    cameras_after: List[str]    # nearest cameras on side B
    estimated_timestamps: Optional[Tuple[float, float]] = None
    severity: float = 0.0       # higher = worse gap

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type,
            "center": self.center.tolist(),
            "extent": self.extent,
            "component_a": self.component_a,
            "component_b": self.component_b,
            "cameras_before": self.cameras_before,
            "cameras_after": self.cameras_after,
            "estimated_timestamps": list(self.estimated_timestamps) if self.estimated_timestamps else None,
            "severity": self.severity,
        }


@dataclass
class GapReport:
    """Complete gap analysis of a COLMAP reconstruction."""
    num_components: int
    num_cameras_aligned: int
    num_cameras_total: int      # aligned + failed
    gaps: List[SpatialGap]
    failed_images: List[str]
    camera_positions: List[CameraPosition]
    component_labels: np.ndarray  # cluster label per camera

    def summary(self) -> str:
        lines = [
            f"Cameras: {self.num_cameras_aligned} aligned"
            f" / {self.num_cameras_total} total"
            f" ({self.num_cameras_total - self.num_cameras_aligned} failed)",
            f"Components: {self.num_components}",
            f"Gaps: {len(self.gaps)}",
        ]
        for i, gap in enumerate(self.gaps):
            ts = ""
            if gap.estimated_timestamps:
                ts = f" t=[{gap.estimated_timestamps[0]:.1f}s, {gap.estimated_timestamps[1]:.1f}s]"
            lines.append(
                f"  Gap {i}: {gap.gap_type} "
                f"extent={gap.extent:.2f} "
                f"severity={gap.severity:.1f}{ts}"
            )
        if self.failed_images:
            lines.append(f"Failed images ({len(self.failed_images)}):")
            for name in self.failed_images[:10]:
                lines.append(f"  - {name}")
            if len(self.failed_images) > 10:
                lines.append(f"  ... and {len(self.failed_images) - 10} more")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "num_components": self.num_components,
            "num_cameras_aligned": self.num_cameras_aligned,
            "num_cameras_total": self.num_cameras_total,
            "failed_images": self.failed_images,
            "gaps": [g.to_dict() for g in self.gaps],
            "cameras": [
                {
                    "image_id": c.image_id,
                    "name": c.name,
                    "position": c.position.tolist(),
                    "timestamp": c.timestamp,
                }
                for c in self.camera_positions
            ],
        }

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "GapReport":
        """Load a gap report from JSON (for bridge extraction)."""
        data = json.loads(Path(path).read_text())
        cameras = [
            CameraPosition(
                image_id=c["image_id"],
                name=c["name"],
                position=np.array(c["position"]),
                qvec=np.zeros(4),  # not stored in report
                tvec=np.zeros(3),
                camera_id=0,
                timestamp=c.get("timestamp"),
            )
            for c in data.get("cameras", [])
        ]
        gaps = [
            SpatialGap(
                gap_type=g["gap_type"],
                center=np.array(g["center"]),
                extent=g["extent"],
                component_a=g["component_a"],
                component_b=g["component_b"],
                cameras_before=g["cameras_before"],
                cameras_after=g["cameras_after"],
                estimated_timestamps=tuple(g["estimated_timestamps"]) if g.get("estimated_timestamps") else None,
                severity=g.get("severity", 0.0),
            )
            for g in data.get("gaps", [])
        ]
        return cls(
            num_components=data["num_components"],
            num_cameras_aligned=data["num_cameras_aligned"],
            num_cameras_total=data["num_cameras_total"],
            gaps=gaps,
            failed_images=data.get("failed_images", []),
            camera_positions=cameras,
            component_labels=np.zeros(len(cameras)),  # not stored
        )


class GapDetector:
    """Analyze COLMAP reconstructions for spatial gaps."""

    def __init__(
        self,
        cluster_eps: float = 5.0,
        min_cluster_size: int = 3,
        sparse_threshold: float = 2.0,
        nearest_k: int = 3,
    ):
        """Initialize gap detector.

        Args:
            cluster_eps: DBSCAN epsilon — maximum distance between
                         neighboring cameras in the same cluster.
                         Units match COLMAP world coordinates.
            min_cluster_size: Minimum cameras to form a component.
            sparse_threshold: Distance threshold for detecting sparse
                              regions (gaps within a connected component).
            nearest_k: Number of nearest cameras to report per gap side.
        """
        self.cluster_eps = cluster_eps
        self.min_cluster_size = min_cluster_size
        self.sparse_threshold = sparse_threshold
        self.nearest_k = nearest_k

    def parse_colmap_images(self, images_txt: str) -> List[CameraPosition]:
        """Parse camera positions from COLMAP images.txt.

        COLMAP stores world-to-camera transforms (R_w2c, t_w2c).
        Camera center in world coords: center = -R_w2c.T @ t_w2c
        """
        cameras = []
        path = Path(images_txt)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {images_txt}")

        lines = path.read_text().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Image lines have: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                camera_id = int(parts[8])
                name = parts[9]
            except (ValueError, IndexError):
                continue

            # Skip the POINTS2D line that follows each image line
            if i < len(lines):
                i += 1

            # Quaternion to rotation matrix
            qvec = np.array([qw, qx, qy, qz])
            R = _qvec2rotmat(qvec)
            tvec = np.array([tx, ty, tz])

            # Camera center in world coordinates
            center = -R.T @ tvec

            # Try to extract timestamp from filename
            timestamp = _extract_timestamp(name)

            cameras.append(CameraPosition(
                image_id=image_id,
                name=name,
                position=center,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                timestamp=timestamp,
            ))

        return cameras

    def parse_metashape_cameras(self, xml_path: str) -> List[CameraPosition]:
        """Parse camera positions from Metashape cameras.xml export.

        Reads the XML directly — no need to run the full COLMAP export
        just to check for gaps.

        Metashape camera transform is a 4x4 camera-to-chunk matrix.
        Camera center in chunk coords = transform[:3, 3].
        If a chunk transform exists, applies it to get world coords.
        """
        from .colmap_export import parse_metashape_xml, get_chunk_rotation_and_scale

        project = parse_metashape_xml(xml_path)
        R_cw, cw_scale = get_chunk_rotation_and_scale(
            project.chunk_transform, flip_yz=True,
        )

        cameras = []
        for cam in project.cameras:
            if cam.transform is None or not cam.enabled:
                continue

            R_parent = cam.transform[:3, :3].copy()
            t_parent = cam.transform[:3, 3].copy()

            # Camera center in world coordinates
            center = cw_scale * (R_cw @ t_parent)

            timestamp = _extract_timestamp(cam.label)

            cameras.append(CameraPosition(
                image_id=cam.camera_id,
                name=cam.label,
                position=center,
                qvec=np.zeros(4),
                tvec=np.zeros(3),
                camera_id=cam.sensor_id,
                timestamp=timestamp,
            ))

        return cameras

    def parse_xmp_cameras(self, xmp_dir: str) -> List[CameraPosition]:
        """Parse camera positions from XMP sidecar files.

        Reads RealityScan/RealityCapture XMP sidecars (xcr namespace).
        Each .xmp file contains:
            xcr:Rotation — R_w2c as 9 floats (row-major 3x3)
            xcr:Position — camera center in world coordinates (3 floats)

        Scans for *.xmp files in the directory, skipping _common.xmp.
        """
        xmp_path = Path(xmp_dir)
        xcr_ns = "http://www.capturingreality.com/ns/xcr/1.1#"

        cameras = []
        xmp_files = sorted(
            f for f in xmp_path.iterdir()
            if f.suffix.lower() == ".xmp" and f.name != "_common.xmp"
        )

        for i, xmp_file in enumerate(xmp_files):
            try:
                tree = ET.parse(str(xmp_file))
                root = tree.getroot()

                # Find the Description element with xcr attributes
                desc = None
                for elem in root.iter():
                    if elem.get(f"{{{xcr_ns}}}Version") is not None:
                        desc = elem
                        break
                    # Also check without namespace (some XMP uses plain attributes)
                    if elem.get("xcr:Version") is not None:
                        desc = elem
                        break

                if desc is None:
                    continue

                # Parse position
                pos_elem = desc.find(f"{{{xcr_ns}}}Position")
                if pos_elem is None:
                    # Try without namespace prefix
                    for child in desc:
                        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if tag == "Position":
                            pos_elem = child
                            break

                if pos_elem is None or not pos_elem.text:
                    continue

                pos_vals = [float(v) for v in pos_elem.text.strip().split()]
                if len(pos_vals) != 3:
                    continue

                position = np.array(pos_vals, dtype=np.float64)

                # Derive image name from XMP filename
                # photo.jpg.xmp → photo.jpg, photo.xmp → photo.jpg
                image_name = xmp_file.stem
                if not Path(image_name).suffix:
                    image_name = image_name + ".jpg"

                timestamp = _extract_timestamp(image_name)

                cameras.append(CameraPosition(
                    image_id=i + 1,
                    name=image_name,
                    position=position,
                    qvec=np.zeros(4),
                    tvec=np.zeros(3),
                    camera_id=1,
                    timestamp=timestamp,
                ))

            except (ET.ParseError, ValueError):
                continue

        return cameras

    def detect_components(
        self, cameras: List[CameraPosition],
    ) -> Tuple[List[List[CameraPosition]], np.ndarray]:
        """Cluster cameras into spatial components using DBSCAN.

        Returns:
            (list_of_components, labels_array)
            Each component is a list of CameraPosition.
            labels_array has one entry per camera (-1 = noise/outlier).
        """
        if len(cameras) < 2:
            return [cameras], np.zeros(len(cameras), dtype=int)

        positions = np.array([c.position for c in cameras])

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.min_cluster_size,
            ).fit(positions)
            labels = clustering.labels_
        except ImportError:
            # Fallback: simple distance-based connected components
            labels = self._fallback_clustering(positions)

        # Group cameras by label
        components = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # noise
            if label not in components:
                components[label] = []
            components[label].append(cameras[i])

        # Sort by size descending
        sorted_components = sorted(
            components.values(), key=len, reverse=True,
        )

        return sorted_components, labels

    def find_gaps(
        self,
        components: List[List[CameraPosition]],
    ) -> List[SpatialGap]:
        """Find gaps between disconnected components.

        For each pair of components, computes the minimum inter-component
        distance and identifies the nearest cameras on each side.
        """
        if len(components) <= 1:
            return []

        gaps = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                gap = self._gap_between_components(
                    components[i], components[j], i, j,
                )
                if gap is not None:
                    gaps.append(gap)

        # Sort by severity (extent / number of cameras)
        gaps.sort(key=lambda g: g.severity, reverse=True)
        return gaps

    def find_sparse_regions(
        self, cameras: List[CameraPosition],
    ) -> List[SpatialGap]:
        """Find sparse regions within a single component.

        Looks for pairs of consecutive cameras (sorted by trajectory)
        with unusually large gaps.
        """
        if len(cameras) < 3:
            return []

        # Sort by timestamp if available, else by position along principal axis
        sorted_cams = self._sort_by_trajectory(cameras)

        # Compute consecutive distances
        distances = []
        for i in range(len(sorted_cams) - 1):
            d = sorted_cams[i].distance_to(sorted_cams[i + 1])
            distances.append(d)

        if not distances:
            return []

        # Detect outlier gaps (> median + sparse_threshold * MAD)
        median_d = np.median(distances)
        mad = np.median(np.abs(np.array(distances) - median_d))
        threshold = median_d + self.sparse_threshold * max(mad, median_d * 0.1)

        gaps = []
        for i, d in enumerate(distances):
            if d > threshold:
                cam_a = sorted_cams[i]
                cam_b = sorted_cams[i + 1]
                center = (cam_a.position + cam_b.position) / 2.0

                # Estimate timestamps
                ts = None
                if cam_a.timestamp is not None and cam_b.timestamp is not None:
                    ts = (cam_a.timestamp, cam_b.timestamp)

                gaps.append(SpatialGap(
                    gap_type="sparse",
                    center=center,
                    extent=d,
                    component_a=0,
                    component_b=0,
                    cameras_before=[cam_a.name],
                    cameras_after=[cam_b.name],
                    estimated_timestamps=ts,
                    severity=d / max(median_d, 0.01),
                ))

        return gaps

    def analyze(
        self,
        colmap_dir: Optional[str] = None,
        metashape_xml: Optional[str] = None,
        xmp_dir: Optional[str] = None,
        source_images_dir: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> GapReport:
        """Complete gap analysis on a reconstruction.

        Accepts camera poses from any of these sources (exactly one required):

        Args:
            colmap_dir: Directory containing COLMAP text files
                        (images.txt, cameras.txt, points3D.txt).
            metashape_xml: Path to Metashape cameras.xml export.
            xmp_dir: Directory containing XMP sidecar files
                     (RealityScan/RealityCapture export).
            source_images_dir: Optional directory of source images.
                               Used to detect failed images (present
                               in source but not in reconstruction).
        """
        def _log(msg):
            if log:
                log(msg)

        # Parse cameras from the provided source
        if metashape_xml is not None:
            cameras = self.parse_metashape_cameras(metashape_xml)
        elif xmp_dir is not None:
            cameras = self.parse_xmp_cameras(xmp_dir)
        elif colmap_dir is not None:
            colmap_path = Path(colmap_dir)

            # Find images.txt (might be in sparse/ subdirectory)
            images_txt = colmap_path / "images.txt"
            if not images_txt.exists():
                images_txt = colmap_path / "sparse" / "images.txt"
            if not images_txt.exists():
                sparse_dirs = list(colmap_path.glob("sparse/*/images.txt"))
                if sparse_dirs:
                    images_txt = sparse_dirs[0]
            if not images_txt.exists():
                raise FileNotFoundError(
                    f"Cannot find images.txt in {colmap_dir}"
                )

            cameras = self.parse_colmap_images(str(images_txt))
        else:
            raise ValueError(
                "Provide one of: colmap_dir, metashape_xml, or xmp_dir"
            )

        _log(f"Gap analysis: {len(cameras)} cameras aligned")

        aligned_names = {c.name for c in cameras}

        # Detect failed images
        failed_images = []
        total_images = len(cameras)
        if source_images_dir:
            source_path = Path(source_images_dir)
            exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            all_source = sorted(
                f.name for f in source_path.iterdir()
                if f.suffix in exts
            )
            failed_images = [n for n in all_source if n not in aligned_names]
            total_images = len(all_source)
            if failed_images:
                _log(f"  Failed images: {len(failed_images)} of {total_images} not in reconstruction")

        # Cluster into components
        components, labels = self.detect_components(cameras)
        comp_sizes = ', '.join(str(len(c)) for c in components)
        _log(f"  Components: {len(components)} (sizes: {comp_sizes})")
        _log(f"  Clustering: eps={self.cluster_eps:.1f}, sparse_threshold={self.sparse_threshold:.1f}")

        # Find inter-component gaps
        gaps = self.find_gaps(components)

        # Find sparse regions within the largest component
        if components:
            sparse_gaps = self.find_sparse_regions(components[0])
            disc_count = len(gaps)  # before adding sparse
            gaps.extend(sparse_gaps)
        else:
            sparse_gaps = []
            disc_count = len(gaps)
        _log(f"  Gaps found: {disc_count} disconnected, {len(sparse_gaps)} sparse")

        # Sort all gaps by severity
        gaps.sort(key=lambda g: g.severity, reverse=True)

        for i, gap in enumerate(gaps):
            ts_str = ""
            if gap.estimated_timestamps:
                ts_str = f", time={gap.estimated_timestamps[0]:.1f}\u2013{gap.estimated_timestamps[1]:.1f}s"
            _log(f"  Gap {i}: {gap.gap_type}, severity={gap.severity:.2f}, extent={gap.extent:.2f}{ts_str}")

        return GapReport(
            num_components=len(components),
            num_cameras_aligned=len(cameras),
            num_cameras_total=total_images,
            gaps=gaps,
            failed_images=failed_images,
            camera_positions=cameras,
            component_labels=labels,
        )

    def _gap_between_components(
        self,
        comp_a: List[CameraPosition],
        comp_b: List[CameraPosition],
        idx_a: int,
        idx_b: int,
    ) -> Optional[SpatialGap]:
        """Find the gap between two components."""
        positions_a = np.array([c.position for c in comp_a])
        positions_b = np.array([c.position for c in comp_b])

        # Compute pairwise distances (brute force for small sets)
        min_dist = float('inf')
        best_i, best_j = 0, 0

        for i in range(len(positions_a)):
            dists = np.linalg.norm(positions_b - positions_a[i], axis=1)
            j = np.argmin(dists)
            if dists[j] < min_dist:
                min_dist = dists[j]
                best_i = i
                best_j = j

        # Nearest cameras on each side
        dists_from_gap_a = np.linalg.norm(
            positions_a - (positions_a[best_i] + positions_b[best_j]) / 2, axis=1
        )
        dists_from_gap_b = np.linalg.norm(
            positions_b - (positions_a[best_i] + positions_b[best_j]) / 2, axis=1
        )

        k = min(self.nearest_k, len(comp_a), len(comp_b))
        nearest_a = [comp_a[i].name for i in np.argsort(dists_from_gap_a)[:k]]
        nearest_b = [comp_b[i].name for i in np.argsort(dists_from_gap_b)[:k]]

        center = (positions_a[best_i] + positions_b[best_j]) / 2.0

        # Estimate timestamps from nearest cameras
        ts = None
        ts_a = [c.timestamp for c in comp_a if c.timestamp is not None]
        ts_b = [c.timestamp for c in comp_b if c.timestamp is not None]
        if ts_a and ts_b:
            ts = (max(ts_a), min(ts_b))
            # Ensure ts[0] < ts[1]
            if ts[0] > ts[1]:
                ts = (ts[1], ts[0])

        return SpatialGap(
            gap_type="disconnected",
            center=center,
            extent=min_dist,
            component_a=idx_a,
            component_b=idx_b,
            cameras_before=nearest_a,
            cameras_after=nearest_b,
            estimated_timestamps=ts,
            severity=min_dist * (len(comp_a) + len(comp_b)),
        )

    def _sort_by_trajectory(
        self, cameras: List[CameraPosition],
    ) -> List[CameraPosition]:
        """Sort cameras by trajectory order.

        Uses timestamps if available, otherwise PCA + projection
        onto the principal axis.
        """
        # Try timestamps first
        if all(c.timestamp is not None for c in cameras):
            return sorted(cameras, key=lambda c: c.timestamp)

        # Fallback: PCA-based ordering
        positions = np.array([c.position for c in cameras])
        centroid = positions.mean(axis=0)
        centered = positions - centroid

        # Principal axis via SVD
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal = Vt[0]

        # Project onto principal axis
        projections = centered @ principal
        order = np.argsort(projections)

        return [cameras[i] for i in order]

    def _fallback_clustering(self, positions: np.ndarray) -> np.ndarray:
        """Simple connected-components clustering when sklearn is unavailable.

        Uses a distance threshold to build adjacency, then BFS to find
        connected components.
        """
        n = len(positions)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        current_label = 0

        for start in range(n):
            if visited[start]:
                continue

            # BFS from this node
            queue = [start]
            component = []
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)

                # Find neighbors within eps
                dists = np.linalg.norm(positions - positions[node], axis=1)
                neighbors = np.where(
                    (dists < self.cluster_eps) & (~visited)
                )[0]
                queue.extend(neighbors.tolist())

            if len(component) >= self.min_cluster_size:
                for idx in component:
                    labels[idx] = current_label
                current_label += 1

        return labels


# --- Utility functions ---

def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def _extract_timestamp(filename: str) -> Optional[float]:
    """Try to extract a timestamp (seconds) from a frame filename.

    Handles common naming patterns:
    - frame_00123_t45.67s.jpg  (from motion_selector)
    - 00001.jpg  (sequential index)
    - front_00001.jpg / back_00001.jpg  (from osv.py)
    - DJI_20250101_120000_00001.jpg  (DJI naming)
    """
    # Pattern: t{seconds}s
    m = re.search(r't(\d+\.?\d*)s', filename)
    if m:
        return float(m.group(1))

    # Pattern: bare number (interpret as frame index, assume 2s intervals)
    m = re.search(r'(\d{4,6})', filename)
    if m:
        return float(int(m.group(1))) * 2.0  # rough estimate

    return None


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect spatial gaps in reconstruction alignments"
    )
    # Input source — one required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--colmap", dest="colmap_dir",
                             help="COLMAP output directory (images.txt)")
    input_group.add_argument("--metashape", dest="metashape_xml",
                             help="Metashape cameras.xml export")
    input_group.add_argument("--xmp", dest="xmp_dir",
                             help="Directory of XMP sidecar files (RealityScan)")

    parser.add_argument("--source-images", help="Source images directory (for failed detection)")
    parser.add_argument("--eps", type=float, default=5.0,
                        help="DBSCAN epsilon (default: 5.0)")
    parser.add_argument("--min-cluster", type=int, default=3,
                        help="Minimum cluster size (default: 3)")
    parser.add_argument("--sparse-threshold", type=float, default=2.0,
                        help="Sparse gap threshold multiplier (default: 2.0)")
    parser.add_argument("-o", "--output", help="Save report JSON")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    try:
        detector = GapDetector(
            cluster_eps=args.eps,
            min_cluster_size=args.min_cluster,
            sparse_threshold=args.sparse_threshold,
        )

        report = detector.analyze(
            colmap_dir=args.colmap_dir,
            metashape_xml=args.metashape_xml,
            xmp_dir=args.xmp_dir,
            source_images_dir=args.source_images,
        )

        print(report.summary())

        if args.verbose and report.camera_positions:
            print(f"\nCamera positions ({len(report.camera_positions)}):")
            for cam in report.camera_positions[:20]:
                pos = cam.position
                ts = f" t={cam.timestamp:.1f}s" if cam.timestamp else ""
                print(f"  {cam.name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}){ts}")
            if len(report.camera_positions) > 20:
                print(f"  ... and {len(report.camera_positions) - 20} more")

        if args.output:
            report.save(args.output)
            print(f"\nReport saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

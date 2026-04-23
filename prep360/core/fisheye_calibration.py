"""
Fisheye Camera Calibration

Calibrate fisheye lenses using ChArUco boards. Produces intrinsics
(camera matrix K + distortion coefficients D) for OpenCV's fisheye module.

Usage:
    # One-time calibration from images
    calibrator = FisheyeCalibrator()
    calib = calibrator.calibrate(images, image_size)
    calib.save("osmo360_front.json")

    # Load and reuse
    calib = FisheyeCalibration.load("osmo360_front.json")

    # Dual-camera (DJI Osmo 360)
    dual = DualFisheyeCalibration(front=front_calib, back=back_calib)
    dual.save("osmo360.json")
"""

import json
import datetime
import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class FisheyeCalibration:
    """Calibration result for a single fisheye lens."""
    camera_matrix: np.ndarray       # 3x3 K matrix
    dist_coeffs: np.ndarray         # 4x1 (k1, k2, k3, k4)
    image_size: Tuple[int, int]     # (width, height)
    rms_error: float                # reprojection error
    num_images_used: int
    fov_degrees: float              # estimated diagonal FOV

    def save(self, path: str):
        """Save calibration to JSON."""
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "image_size": list(self.image_size),
            "rms_error": self.rms_error,
            "num_images_used": self.num_images_used,
            "fov_degrees": self.fov_degrees,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "FisheyeCalibration":
        """Load calibration from JSON."""
        data = json.loads(Path(path).read_text())

        # Handle both standalone and nested (dual) format
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(4, 1),
            image_size=tuple(data["image_size"]),
            rms_error=data["rms_error"],
            num_images_used=data["num_images_used"],
            fov_degrees=data["fov_degrees"],
        )

    def summary(self) -> str:
        K = self.camera_matrix
        D = self.dist_coeffs.flatten()
        return (
            f"Image: {self.image_size[0]}x{self.image_size[1]}\n"
            f"Focal: fx={K[0,0]:.1f} fy={K[1,1]:.1f}\n"
            f"Center: cx={K[0,2]:.1f} cy={K[1,2]:.1f}\n"
            f"Distortion: [{D[0]:.6f}, {D[1]:.6f}, {D[2]:.6f}, {D[3]:.6f}]\n"
            f"FOV: {self.fov_degrees:.1f}°\n"
            f"RMS: {self.rms_error:.4f} ({self.num_images_used} images)"
        )


@dataclass
class DualFisheyeCalibration:
    """Calibration for a dual-fisheye camera (e.g., DJI Osmo 360).

    The front and back lenses are independent optical systems pointing
    in opposite directions. back_rotation_deg=180 means the back lens
    is rotated 180° in yaw relative to the front.

    Rig geometry (added 2026-04-21, from empirical calibration):
        baseline_m:  Distance between optical centers in meters.
        baseline_axis: Unit vector from front to back optical center
                       in the front camera's coordinate frame.
    """
    front: FisheyeCalibration
    back: FisheyeCalibration
    front_rotation_deg: float = 0.0
    back_rotation_deg: float = 180.0
    camera_model: str = "DJI Osmo 360"
    calibration_date: str = ""
    baseline_m: float = 0.0
    baseline_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def save(self, path: str):
        """Save dual calibration to single JSON file."""
        data = {
            "camera_model": self.camera_model,
            "calibration_date": self.calibration_date or datetime.date.today().isoformat(),
            "front_rotation_deg": self.front_rotation_deg,
            "back_rotation_deg": self.back_rotation_deg,
            "baseline_m": self.baseline_m,
            "baseline_axis": list(self.baseline_axis),
            "front": {
                "camera_matrix": self.front.camera_matrix.tolist(),
                "dist_coeffs": self.front.dist_coeffs.flatten().tolist(),
                "image_size": list(self.front.image_size),
                "rms_error": self.front.rms_error,
                "num_images_used": self.front.num_images_used,
                "fov_degrees": self.front.fov_degrees,
            },
            "back": {
                "camera_matrix": self.back.camera_matrix.tolist(),
                "dist_coeffs": self.back.dist_coeffs.flatten().tolist(),
                "image_size": list(self.back.image_size),
                "rms_error": self.back.rms_error,
                "num_images_used": self.back.num_images_used,
                "fov_degrees": self.back.fov_degrees,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "DualFisheyeCalibration":
        """Load dual calibration from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            front=FisheyeCalibration(
                camera_matrix=np.array(data["front"]["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["front"]["dist_coeffs"], dtype=np.float64).reshape(4, 1),
                image_size=tuple(data["front"]["image_size"]),
                rms_error=data["front"]["rms_error"],
                num_images_used=data["front"]["num_images_used"],
                fov_degrees=data["front"]["fov_degrees"],
            ),
            back=FisheyeCalibration(
                camera_matrix=np.array(data["back"]["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(data["back"]["dist_coeffs"], dtype=np.float64).reshape(4, 1),
                image_size=tuple(data["back"]["image_size"]),
                rms_error=data["back"]["rms_error"],
                num_images_used=data["back"]["num_images_used"],
                fov_degrees=data["back"]["fov_degrees"],
            ),
            front_rotation_deg=data.get("front_rotation_deg", 0.0),
            back_rotation_deg=data.get("back_rotation_deg", 180.0),
            camera_model=data.get("camera_model", "Unknown"),
            calibration_date=data.get("calibration_date", ""),
            baseline_m=data.get("baseline_m", 0.0),
            baseline_axis=tuple(data.get("baseline_axis", (0.0, 0.0, 1.0))),
        )

    def summary(self) -> str:
        lines = [
            f"Camera: {self.camera_model}",
            f"Date: {self.calibration_date}",
            f"Rotations: front={self.front_rotation_deg}° back={self.back_rotation_deg}°",
            f"Baseline: {self.baseline_m * 1000:.1f} mm along {self.baseline_axis}",
            "",
            "--- Front Lens ---",
            self.front.summary(),
            "",
            "--- Back Lens ---",
            self.back.summary(),
        ]
        return "\n".join(lines)


class FisheyeCalibrator:
    """Calibrate fisheye cameras using ChArUco boards."""

    def __init__(
        self,
        board_squares_x: int = 7,
        board_squares_y: int = 5,
        square_length_mm: float = 40.0,
        marker_length_mm: float = 30.0,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_100,
    ):
        """Initialize with board parameters.

        Args:
            board_squares_x: Number of chessboard squares in X.
            board_squares_y: Number of chessboard squares in Y.
            square_length_mm: Size of chessboard squares in mm.
            marker_length_mm: Size of ArUco markers in mm.
            aruco_dict_id: ArUco dictionary ID (default DICT_4X4_100).
        """
        self.board_squares_x = board_squares_x
        self.board_squares_y = board_squares_y
        self.square_length_mm = square_length_mm
        self.marker_length_mm = marker_length_mm

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (board_squares_x, board_squares_y),
            square_length_mm / 1000.0,   # convert to meters
            marker_length_mm / 1000.0,
            self.aruco_dict,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def generate_board_image(
        self, output_path: str, pixels_per_mm: float = 10.0
    ) -> str:
        """Generate a printable ChArUco board image.

        Args:
            output_path: Where to save the PNG.
            pixels_per_mm: Resolution (10 = 254 DPI).

        Returns:
            Path to saved image.
        """
        w = int(self.board_squares_x * self.square_length_mm * pixels_per_mm)
        h = int(self.board_squares_y * self.square_length_mm * pixels_per_mm)
        board_img = self.board.generateImage((w, h))
        cv2.imwrite(output_path, board_img)
        return output_path

    def detect_board(
        self, image: np.ndarray, min_corners: int = 6
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect ChArUco corners in an image.

        Args:
            image: BGR or grayscale image.
            min_corners: Minimum corners for a valid detection.

        Returns:
            (charuco_corners, charuco_ids) or None if detection fails.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self.detector.detectBoard(gray)

        if charuco_corners is None or len(charuco_corners) < min_corners:
            return None

        return charuco_corners, charuco_ids

    def visualize_detection(
        self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray
    ) -> np.ndarray:
        """Draw detected corners on image for verification."""
        vis = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
        # Add corner count text
        cv2.putText(
            vis, f"{len(corners)} corners",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        return vis

    def calibrate(
        self,
        images: List[np.ndarray],
        image_size: Tuple[int, int],
        min_corners: int = 6,
    ) -> FisheyeCalibration:
        """Calibrate fisheye lens from ChArUco detections.

        Args:
            images: List of BGR images containing the ChArUco board.
            image_size: (width, height) of the images.
            min_corners: Minimum corners per image to include.

        Returns:
            FisheyeCalibration with intrinsics.

        Raises:
            ValueError: If too few usable images.
        """
        all_obj_points = []
        all_img_points = []

        for img in images:
            result = self.detect_board(img, min_corners)
            if result is None:
                continue
            corners, ids = result
            obj_pts = self.board.getChessboardCorners()[ids.flatten()]
            all_obj_points.append(
                obj_pts.reshape(1, -1, 3).astype(np.float64)
            )
            all_img_points.append(
                corners.reshape(1, -1, 2).astype(np.float64)
            )

        if len(all_obj_points) < 5:
            raise ValueError(
                f"Only {len(all_obj_points)} usable images (need >= 5). "
                f"Ensure the board is visible and well-lit."
            )

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            all_obj_points,
            all_img_points,
            image_size,
            K, D,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-6,
            ),
        )

        # Estimate diagonal FOV
        # For equidistant fisheye: r = f * theta
        # At image edge: theta_max = r_max / f
        diag_px = np.sqrt(image_size[0]**2 + image_size[1]**2) / 2.0
        f_avg = (K[0, 0] + K[1, 1]) / 2.0
        fov_est = 2.0 * np.degrees(diag_px / f_avg) if f_avg > 0 else 0

        return FisheyeCalibration(
            camera_matrix=K,
            dist_coeffs=D,
            image_size=image_size,
            rms_error=rms,
            num_images_used=len(all_obj_points),
            fov_degrees=fov_est,
        )

    def calibrate_from_paths(
        self,
        image_paths: List[str],
        min_corners: int = 6,
    ) -> FisheyeCalibration:
        """Calibrate from a list of image file paths.

        Automatically determines image size from the first image.
        """
        images = []
        image_size = None

        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)
            images.append(img)

        if not images:
            raise ValueError("No valid images found")

        return self.calibrate(images, image_size, min_corners)

    def calibrate_from_video(
        self,
        video_path: str,
        max_frames: int = 50,
        sample_interval: int = 15,
        min_corners: int = 6,
    ) -> FisheyeCalibration:
        """Calibrate from a video of the ChArUco board.

        Samples frames at regular intervals, auto-detects the board,
        and runs calibration on the best detections.

        Args:
            video_path: Path to calibration video.
            max_frames: Maximum frames to use.
            sample_interval: Sample every Nth frame.
            min_corners: Minimum corners per detection.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_size = (w, h)

        images = []
        frame_idx = 0

        while len(images) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Only keep frames where the board is detected
                result = self.detect_board(frame, min_corners)
                if result is not None:
                    images.append(frame)

            frame_idx += 1

        cap.release()

        if len(images) < 5:
            raise ValueError(
                f"Only {len(images)} frames with board detected "
                f"(sampled {frame_idx} frames). Need at least 5."
            )

        return self.calibrate(images, image_size, min_corners)


# --- CLI interface ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fisheye camera calibration using ChArUco boards"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Generate board
    gen_p = sub.add_parser("board", help="Generate printable ChArUco board")
    gen_p.add_argument("output", help="Output PNG path")
    gen_p.add_argument("--squares-x", type=int, default=7)
    gen_p.add_argument("--squares-y", type=int, default=5)
    gen_p.add_argument("--square-mm", type=float, default=40.0)
    gen_p.add_argument("--marker-mm", type=float, default=30.0)
    gen_p.add_argument("--dpi", type=float, default=254.0,
                        help="Print resolution (default: 254 DPI)")

    # Calibrate from images
    cal_p = sub.add_parser("calibrate", help="Calibrate from images")
    cal_p.add_argument("images", nargs="+", help="Image files or glob pattern")
    cal_p.add_argument("-o", "--output", required=True, help="Output JSON path")
    cal_p.add_argument("--squares-x", type=int, default=7)
    cal_p.add_argument("--squares-y", type=int, default=5)
    cal_p.add_argument("--square-mm", type=float, default=40.0)
    cal_p.add_argument("--marker-mm", type=float, default=30.0)

    # Calibrate from video
    vid_p = sub.add_parser("calibrate-video", help="Calibrate from video")
    vid_p.add_argument("video", help="Calibration video")
    vid_p.add_argument("-o", "--output", required=True, help="Output JSON path")
    vid_p.add_argument("--squares-x", type=int, default=7)
    vid_p.add_argument("--squares-y", type=int, default=5)
    vid_p.add_argument("--square-mm", type=float, default=40.0)
    vid_p.add_argument("--marker-mm", type=float, default=30.0)
    vid_p.add_argument("--max-frames", type=int, default=50)
    vid_p.add_argument("--sample-interval", type=int, default=15)

    # Show calibration
    show_p = sub.add_parser("show", help="Display calibration details")
    show_p.add_argument("calibration", help="Calibration JSON file")

    args = parser.parse_args()

    try:
        if args.command == "board":
            ppm = args.dpi / 25.4  # DPI to pixels per mm
            cal = FisheyeCalibrator(
                args.squares_x, args.squares_y,
                args.square_mm, args.marker_mm,
            )
            path = cal.generate_board_image(args.output, ppm)
            print(f"Board saved to {path}")
            print(f"  {args.squares_x}x{args.squares_y} squares, "
                  f"{args.square_mm}mm squares, {args.marker_mm}mm markers")

        elif args.command == "calibrate":
            cal = FisheyeCalibrator(
                args.squares_x, args.squares_y,
                args.square_mm, args.marker_mm,
            )
            # Expand glob patterns
            import glob
            paths = []
            for p in args.images:
                expanded = glob.glob(p)
                paths.extend(expanded if expanded else [p])

            print(f"Calibrating from {len(paths)} images...")
            result = cal.calibrate_from_paths(paths)
            result.save(args.output)
            print(f"Saved to {args.output}")
            print(result.summary())

        elif args.command == "calibrate-video":
            cal = FisheyeCalibrator(
                args.squares_x, args.squares_y,
                args.square_mm, args.marker_mm,
            )
            print(f"Calibrating from video {args.video}...")
            result = cal.calibrate_from_video(
                args.video,
                max_frames=args.max_frames,
                sample_interval=args.sample_interval,
            )
            result.save(args.output)
            print(f"Saved to {args.output}")
            print(result.summary())

        elif args.command == "show":
            path = Path(args.calibration)
            data = json.loads(path.read_text())
            if "front" in data and "back" in data:
                dual = DualFisheyeCalibration.load(args.calibration)
                print(dual.summary())
            else:
                single = FisheyeCalibration.load(args.calibration)
                print(single.summary())

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

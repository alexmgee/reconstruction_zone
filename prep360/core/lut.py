"""
LUT Processor Module

Apply color lookup tables to convert log footage to standard color space.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import shutil

import cv2
import numpy as np


def _has_torch_cuda() -> bool:
    """Check once whether PyTorch CUDA is available, then cache."""
    global _TORCH_CUDA_AVAILABLE
    try:
        return _TORCH_CUDA_AVAILABLE
    except NameError:
        pass
    try:
        import torch
        _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        _TORCH_CUDA_AVAILABLE = False
    return _TORCH_CUDA_AVAILABLE


@dataclass
class LUTInfo:
    """Information about a loaded LUT."""
    path: str
    name: str
    size: int       # LUT size (e.g., 33 for 33x33x33)
    format: str     # "cube" or "3dl"


class LUTProcessor:
    """Load and apply LUTs to images."""

    def __init__(self):
        self._lut_cache: dict = {}

    def load_cube(self, cube_path: str) -> Tuple[np.ndarray, LUTInfo]:
        """
        Load a .cube LUT file.

        Returns:
            (lut_3d array, LUTInfo)
        """
        path = Path(cube_path)
        cache_key = str(path.resolve()) if path.exists() else str(path)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"LUT not found: {cube_path}")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            raise OSError(f"Could not read LUT '{cube_path}': {e}") from e

        size = None
        lut_data = []
        title = path.stem

        for line in lines:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            upper = line.upper()

            if upper.startswith("TITLE"):
                parts = line.split('"')
                if len(parts) >= 3:
                    title = parts[1]
                else:
                    tokens = line.split(maxsplit=1)
                    if len(tokens) == 2:
                        title = tokens[1].strip()
                continue

            if upper.startswith("LUT_1D_SIZE"):
                raise ValueError(f"Unsupported 1D LUT in {cube_path}; expected a 3D .cube LUT")

            if upper.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Malformed LUT_3D_SIZE line in {cube_path}: {line}")
                try:
                    size = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"Invalid LUT_3D_SIZE in {cube_path}: {parts[1]}") from e
                if size < 2:
                    raise ValueError(f"Invalid LUT size in {cube_path}: {size}")
                continue

            if upper.startswith("DOMAIN_MIN") or upper.startswith("DOMAIN_MAX"):
                continue

            # Parse RGB values
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    lut_data.append(values)
                elif values:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Malformed LUT data line in {cube_path}: {line}")

        if size is None:
            # Try to infer from data length
            data_len = len(lut_data)
            size = int(round(data_len ** (1/3)))
            if size < 2 or size ** 3 != data_len:
                raise ValueError(f"Missing LUT_3D_SIZE and cannot infer cube size from {data_len} rows")

        expected_len = size ** 3
        if len(lut_data) != expected_len:
            raise ValueError(
                f"LUT data mismatch in {cube_path}: expected {expected_len} RGB rows for "
                f"{size}x{size}x{size}, got {len(lut_data)}"
            )

        # Reshape to 3D LUT
        lut_3d = np.array(lut_data, dtype=np.float32).reshape(size, size, size, 3)
        if not np.all(np.isfinite(lut_3d)):
            raise ValueError(f"LUT contains non-finite values: {cube_path}")

        info = LUTInfo(
            path=str(path.absolute()),
            name=title,
            size=size,
            format="cube"
        )

        # Cache
        self._lut_cache[cache_key] = (lut_3d, info)

        return lut_3d, info

    def apply_float(
        self,
        image_bgr_float: np.ndarray,
        lut_3d: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply 3D LUT to image using trilinear interpolation.

        Uses torch.nn.functional.grid_sample on CUDA when available (~67ms
        for a 12MP image on an RTX 3090 Ti). Falls back to NumPy (~8.7s).

        Args:
            image_bgr_float: Input BGR image (float32/float64 in [0, 1])
            lut_3d: 3D LUT array (size, size, size, 3)
            strength: Blend strength (0-1, 1=full LUT)

        Returns:
            Color-corrected BGR image (float32 in [0, 1])
        """
        if image_bgr_float.ndim != 3 or image_bgr_float.shape[2] != 3:
            raise ValueError("LUT input must be a 3-channel BGR image")
        if lut_3d.ndim != 4 or lut_3d.shape[3] != 3:
            raise ValueError("LUT array must have shape (size, size, size, 3)")

        strength = float(strength)
        if not np.isfinite(strength):
            raise ValueError("LUT strength must be finite")
        strength = max(0.0, min(1.0, strength))
        if strength == 0.0:
            return np.clip(image_bgr_float, 0.0, 1.0).astype(np.float32, copy=True)

        size = lut_3d.shape[0]
        if lut_3d.shape[:3] != (size, size, size) or size < 2:
            raise ValueError("LUT array must be cubic and at least 2x2x2")

        if _has_torch_cuda():
            return self._apply_float_cuda(image_bgr_float, lut_3d, strength)
        return self._apply_float_numpy(image_bgr_float, lut_3d, strength)

    def _apply_float_cuda(
        self,
        image_bgr_float: np.ndarray,
        lut_3d: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """GPU path using torch.nn.functional.grid_sample."""
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda")
        h, w = image_bgr_float.shape[:2]
        img = np.clip(image_bgr_float, 0.0, 1.0).astype(np.float32)

        # Cache LUT tensor on GPU (keyed by id to avoid re-uploading)
        lut_id = id(lut_3d)
        if not hasattr(self, "_cuda_lut_cache"):
            self._cuda_lut_cache = {}
        if lut_id not in self._cuda_lut_cache:
            # LUT shape: (B, G, R, 3) -> grid_sample wants (N, C, D, H, W)
            lut_t = torch.from_numpy(lut_3d).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            self._cuda_lut_cache = {lut_id: lut_t}  # only keep one
        lut_t = self._cuda_lut_cache[lut_id]

        img_t = torch.from_numpy(img).to(device)

        # Build sample grid: pixel BGR [0,1] -> grid coords [-1,1]
        # grid_sample grid (x,y,z) maps to LUT axes (R, G, B)
        r = img_t[:, :, 2] * 2.0 - 1.0
        g = img_t[:, :, 1] * 2.0 - 1.0
        b = img_t[:, :, 0] * 2.0 - 1.0
        grid = torch.stack([r, g, b], dim=-1).unsqueeze(0).unsqueeze(1)  # (1,1,H,W,3)

        result = F.grid_sample(
            lut_t, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, 3, 1, H, W)

        # result channels are RGB; reshape to (H, W, 3)
        result = result.squeeze(0).squeeze(1).permute(1, 2, 0)  # (H, W, 3) RGB

        if strength < 1.0:
            original_rgb = torch.stack(
                [img_t[:, :, 2], img_t[:, :, 1], img_t[:, :, 0]], dim=-1
            )
            result = original_rgb + (result - original_rgb) * strength

        result = torch.clamp(result, 0.0, 1.0)
        result_bgr = result.flip(-1)  # RGB -> BGR
        return result_bgr.cpu().numpy()

    def _apply_float_numpy(
        self,
        image_bgr_float: np.ndarray,
        lut_3d: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """CPU fallback using NumPy trilinear interpolation."""
        img_float = np.clip(image_bgr_float.astype(np.float32), 0.0, 1.0)
        size = lut_3d.shape[0]

        b, g, r = cv2.split(img_float)

        r_idx = r * (size - 1)
        g_idx = g * (size - 1)
        b_idx = b * (size - 1)

        r0 = np.floor(r_idx).astype(np.int32)
        g0 = np.floor(g_idx).astype(np.int32)
        b0 = np.floor(b_idx).astype(np.int32)

        r1 = np.minimum(r0 + 1, size - 1)
        g1 = np.minimum(g0 + 1, size - 1)
        b1 = np.minimum(b0 + 1, size - 1)

        rf = r_idx - r0
        gf = g_idx - g0
        bf = b_idx - b0

        r0 = np.clip(r0, 0, size - 1)
        g0 = np.clip(g0, 0, size - 1)
        b0 = np.clip(b0, 0, size - 1)

        # .cube format: B outermost (axis 0), G middle (axis 1), R innermost (axis 2)
        c000 = lut_3d[b0, g0, r0]
        c001 = lut_3d[b1, g0, r0]
        c010 = lut_3d[b0, g1, r0]
        c011 = lut_3d[b1, g1, r0]
        c100 = lut_3d[b0, g0, r1]
        c101 = lut_3d[b1, g0, r1]
        c110 = lut_3d[b0, g1, r1]
        c111 = lut_3d[b1, g1, r1]

        rf = rf[..., np.newaxis]
        gf = gf[..., np.newaxis]
        bf = bf[..., np.newaxis]

        c00 = c000 * (1 - rf) + c100 * rf
        c01 = c001 * (1 - rf) + c101 * rf
        c10 = c010 * (1 - rf) + c110 * rf
        c11 = c011 * (1 - rf) + c111 * rf

        c0 = c00 * (1 - gf) + c10 * gf
        c1 = c01 * (1 - gf) + c11 * gf

        result = c0 * (1 - bf) + c1 * bf

        if strength < 1.0:
            original_rgb = np.stack([r, g, b], axis=-1)
            result = original_rgb * (1 - strength) + result * strength

        result = np.clip(result, 0, 1)
        result_bgr = result[..., ::-1]  # RGB to BGR
        return result_bgr.astype(np.float32)

    def apply_uint8(
        self,
        image_bgr: np.ndarray,
        lut_3d: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """Apply a 3D LUT to a uint8 BGR image."""
        if image_bgr.dtype != np.uint8:
            raise ValueError("apply_uint8 expects a uint8 BGR image")
        result = self.apply_float(image_bgr.astype(np.float32) / 255.0, lut_3d, strength)
        return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)

    def apply(self, image: np.ndarray, lut_3d: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Backward-compatible alias for applying a LUT to a uint8 image."""
        return self.apply_uint8(image, lut_3d, strength)

    def apply_lut_uint8(
        self,
        image_bgr: np.ndarray,
        lut_path: str,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Load a .cube file and apply it to a uint8 BGR image."""
        lut_3d, _ = self.load_cube(lut_path)
        return self.apply_uint8(image_bgr, lut_3d, strength)

    def apply_lut_float(
        self,
        image_bgr_float: np.ndarray,
        lut_path: str,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Load a .cube file and apply it to a float BGR image in [0, 1]."""
        lut_3d, _ = self.load_cube(lut_path)
        return self.apply_float(image_bgr_float, lut_3d, strength)

    def apply_lut(
        self,
        image_bgr: np.ndarray,
        lut_path: str,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Compatibility wrapper for older GUI/CLI callers."""
        return self.apply_lut_uint8(image_bgr, lut_path, strength)

    def process_image(
        self,
        image_path: str,
        output_path: str,
        lut_path: str,
        strength: float = 1.0
    ) -> bool:
        """
        Apply LUT to a single image and save.

        Returns:
            True on success
        """
        try:
            image = cv2.imread(image_path)

            if image is None:
                return False

            result = self.apply_lut_uint8(image, lut_path, strength)
            cv2.imwrite(output_path, result)

            return True

        except Exception:
            return False

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        lut_path: str,
        strength: float = 1.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, int]:
        """
        Apply LUT to all images in a directory.

        Returns:
            (success_count, error_count)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load LUT once
        lut_3d, info = self.load_cube(lut_path)

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

        success = 0
        errors = 0

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images), img_path.name)

            image = cv2.imread(str(img_path))
            if image is None:
                errors += 1
                continue

            result = self.apply_uint8(image, lut_3d, strength)

            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), result)
            success += 1

        return success, errors


def list_luts(luts_dir: str) -> List[LUTInfo]:
    """List available LUT files in a directory."""
    path = Path(luts_dir)
    if not path.exists():
        return []

    processor = LUTProcessor()
    luts = []

    for cube_file in path.glob("*.cube"):
        try:
            _, info = processor.load_cube(str(cube_file))
            luts.append(info)
        except Exception:
            pass

    return luts


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply LUT to images")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output image or directory")
    parser.add_argument("--lut", "-l", required=True, help="Path to .cube LUT file")
    parser.add_argument("--strength", "-s", type=float, default=1.0,
                        help="LUT strength 0-1 (default: 1.0)")
    parser.add_argument("--info", action="store_true",
                        help="Show LUT info and exit")

    args = parser.parse_args()

    processor = LUTProcessor()

    if args.info:
        try:
            _, info = processor.load_cube(args.lut)
            print(f"LUT: {info.name}")
            print(f"Size: {info.size}x{info.size}x{info.size}")
            print(f"Format: {info.format}")
            print(f"Path: {info.path}")
        except Exception as e:
            print(f"Error loading LUT: {e}")
            return 1
        return 0

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Batch mode
        def progress(curr, total, name):
            print(f"[{curr}/{total}] {name}")

        success, errors = processor.process_batch(
            str(input_path),
            str(output_path),
            args.lut,
            args.strength,
            progress
        )
        print(f"Processed {success} images, {errors} errors")

    else:
        # Single image
        if processor.process_image(str(input_path), str(output_path), args.lut, args.strength):
            print(f"Saved: {output_path}")
        else:
            print("Error processing image")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())

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
        # Check cache
        if cube_path in self._lut_cache:
            return self._lut_cache[cube_path]

        path = Path(cube_path)
        if not path.exists():
            raise FileNotFoundError(f"LUT not found: {cube_path}")

        with open(path, 'r') as f:
            lines = f.readlines()

        size = None
        lut_data = []
        title = path.stem

        for line in lines:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('TITLE'):
                title = line.split('"')[1] if '"' in line else line.split()[1]
                continue

            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
                continue

            if line.startswith('DOMAIN_MIN') or line.startswith('DOMAIN_MAX'):
                continue

            # Parse RGB values
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    lut_data.append(values)
            except ValueError:
                continue

        if size is None:
            # Try to infer from data length
            data_len = len(lut_data)
            size = int(round(data_len ** (1/3)))

        expected_len = size ** 3
        if len(lut_data) != expected_len:
            raise ValueError(f"LUT data mismatch: expected {expected_len}, got {len(lut_data)}")

        # Reshape to 3D LUT
        lut_3d = np.array(lut_data, dtype=np.float32).reshape(size, size, size, 3)

        info = LUTInfo(
            path=str(path.absolute()),
            name=title,
            size=size,
            format="cube"
        )

        # Cache
        self._lut_cache[cube_path] = (lut_3d, info)

        return lut_3d, info

    def apply(
        self,
        image: np.ndarray,
        lut_3d: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply 3D LUT to image using trilinear interpolation.

        Args:
            image: Input BGR image (uint8)
            lut_3d: 3D LUT array (size, size, size, 3)
            strength: Blend strength (0-1, 1=full LUT)

        Returns:
            Color-corrected image (uint8)
        """
        # Normalize to 0-1
        img_float = image.astype(np.float32) / 255.0

        size = lut_3d.shape[0]

        # Scale to LUT indices
        # Note: OpenCV uses BGR, LUT might expect RGB
        # Swap channels if needed
        b, g, r = cv2.split(img_float)

        # Scale to LUT coordinates
        r_idx = r * (size - 1)
        g_idx = g * (size - 1)
        b_idx = b * (size - 1)

        # Get integer and fractional parts
        r0 = np.floor(r_idx).astype(np.int32)
        g0 = np.floor(g_idx).astype(np.int32)
        b0 = np.floor(b_idx).astype(np.int32)

        r1 = np.minimum(r0 + 1, size - 1)
        g1 = np.minimum(g0 + 1, size - 1)
        b1 = np.minimum(b0 + 1, size - 1)

        rf = r_idx - r0
        gf = g_idx - g0
        bf = b_idx - b0

        # Clamp indices
        r0 = np.clip(r0, 0, size - 1)
        g0 = np.clip(g0, 0, size - 1)
        b0 = np.clip(b0, 0, size - 1)

        # Trilinear interpolation
        # 8 corners of the cube
        c000 = lut_3d[r0, g0, b0]
        c001 = lut_3d[r0, g0, b1]
        c010 = lut_3d[r0, g1, b0]
        c011 = lut_3d[r0, g1, b1]
        c100 = lut_3d[r1, g0, b0]
        c101 = lut_3d[r1, g0, b1]
        c110 = lut_3d[r1, g1, b0]
        c111 = lut_3d[r1, g1, b1]

        # Expand fractional parts for broadcasting
        rf = rf[..., np.newaxis]
        gf = gf[..., np.newaxis]
        bf = bf[..., np.newaxis]

        # Interpolate
        c00 = c000 * (1 - rf) + c100 * rf
        c01 = c001 * (1 - rf) + c101 * rf
        c10 = c010 * (1 - rf) + c110 * rf
        c11 = c011 * (1 - rf) + c111 * rf

        c0 = c00 * (1 - gf) + c10 * gf
        c1 = c01 * (1 - gf) + c11 * gf

        result = c0 * (1 - bf) + c1 * bf

        # Blend with original if strength < 1
        if strength < 1.0:
            # Convert original to same RGB order
            original_rgb = np.stack([r, g, b], axis=-1)
            result = original_rgb * (1 - strength) + result * strength

        # Convert back to BGR and uint8
        result = np.clip(result, 0, 1)
        result_bgr = result[..., ::-1]  # RGB to BGR

        return (result_bgr * 255).astype(np.uint8)

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
            lut_3d, _ = self.load_cube(lut_path)
            image = cv2.imread(image_path)

            if image is None:
                return False

            result = self.apply(image, lut_3d, strength)
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

            result = self.apply(image, lut_3d, strength)

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

"""
Image Adjustments Module

Shadow and highlight adjustments using OpenCV curve manipulation.
"""

import cv2
import numpy as np
from typing import Tuple


def create_shadow_highlight_lut(shadow: int = 50, highlight: int = 50) -> np.ndarray:
    """
    Create a lookup table for shadow/highlight adjustments.

    Args:
        shadow: Shadow adjustment 0-100 (50=neutral, >50=lift shadows, <50=crush shadows)
        highlight: Highlight adjustment 0-100 (50=neutral, >50=compress highlights, <50=boost highlights)

    Returns:
        LUT array of shape (256,) for cv2.LUT
    """
    # Create base linear LUT
    lut = np.arange(256, dtype=np.float32)

    # Shadow adjustment (affects values below midpoint ~128)
    # shadow > 50: lift shadows (raise dark values)
    # shadow < 50: crush shadows (lower dark values)
    if shadow != 50:
        # Convert 0-100 to adjustment factor
        # 50 = no change, 0 = maximum crush, 100 = maximum lift
        shadow_factor = (shadow - 50) / 50.0  # -1 to +1

        for i in range(256):
            if i < 128:
                # Apply stronger effect to darker values
                # Use a curve that's strongest at 0 and fades to 0 at 128
                strength = (128 - i) / 128.0  # 1 at i=0, 0 at i=128
                adjustment = shadow_factor * strength * 64  # Max adjustment of 64 levels
                lut[i] = np.clip(lut[i] + adjustment, 0, 255)
            else:
                # Gradual fade in the transition zone (128-160)
                if i < 160:
                    transition = (160 - i) / 32.0  # 1 at 128, 0 at 160
                    adjustment = shadow_factor * transition * 16
                    lut[i] = np.clip(lut[i] + adjustment, 0, 255)

    # Highlight adjustment (affects values above midpoint ~128)
    # highlight > 50: compress highlights (lower bright values)
    # highlight < 50: boost highlights (raise bright values)
    if highlight != 50:
        # Convert 0-100 to adjustment factor
        # 50 = no change, 0 = maximum boost, 100 = maximum compress
        highlight_factor = (highlight - 50) / 50.0  # -1 to +1

        for i in range(256):
            if i > 128:
                # Apply stronger effect to brighter values
                # Use a curve that's 0 at 128 and strongest at 255
                strength = (i - 128) / 127.0  # 0 at i=128, 1 at i=255
                adjustment = -highlight_factor * strength * 64  # Negative because we compress
                lut[i] = np.clip(lut[i] + adjustment, 0, 255)
            else:
                # Gradual fade in the transition zone (96-128)
                if i > 96:
                    transition = (i - 96) / 32.0  # 0 at 96, 1 at 128
                    adjustment = -highlight_factor * transition * 16
                    lut[i] = np.clip(lut[i] + adjustment, 0, 255)

    return lut.astype(np.uint8)


def apply_shadow_highlight(
    image: np.ndarray,
    shadow: int = 50,
    highlight: int = 50
) -> np.ndarray:
    """
    Apply shadow and highlight adjustments to an image.

    Args:
        image: Input BGR image (uint8)
        shadow: Shadow adjustment 0-100 (50=neutral)
        highlight: Highlight adjustment 0-100 (50=neutral)

    Returns:
        Adjusted image (uint8)
    """
    if shadow == 50 and highlight == 50:
        return image  # No adjustment needed

    # Create LUT
    lut = create_shadow_highlight_lut(shadow, highlight)

    # Apply LUT to each channel
    # Convert to LAB for better perceptual results (adjust L channel only)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply LUT to luminance channel
    l_adjusted = cv2.LUT(l, lut)

    # Merge and convert back
    lab_adjusted = cv2.merge([l_adjusted, a, b])
    result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

    return result


def apply_shadow_highlight_rgb(
    image: np.ndarray,
    shadow: int = 50,
    highlight: int = 50
) -> np.ndarray:
    """
    Apply shadow and highlight adjustments to RGB channels independently.
    Alternative to LAB-based adjustment.

    Args:
        image: Input BGR image (uint8)
        shadow: Shadow adjustment 0-100 (50=neutral)
        highlight: Highlight adjustment 0-100 (50=neutral)

    Returns:
        Adjusted image (uint8)
    """
    if shadow == 50 and highlight == 50:
        return image

    lut = create_shadow_highlight_lut(shadow, highlight)

    # Apply same LUT to all channels
    b, g, r = cv2.split(image)
    b = cv2.LUT(b, lut)
    g = cv2.LUT(g, lut)
    r = cv2.LUT(r, lut)

    return cv2.merge([b, g, r])


def process_image_with_adjustments(
    image_path: str,
    output_path: str,
    shadow: int = 50,
    highlight: int = 50,
    quality: int = 95
) -> bool:
    """
    Load an image, apply adjustments, and save.

    Returns:
        True on success
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False

        adjusted = apply_shadow_highlight(image, shadow, highlight)

        # Determine output format from extension
        ext = output_path.lower().split('.')[-1]
        if ext in ('jpg', 'jpeg'):
            cv2.imwrite(output_path, adjusted, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == 'png':
            cv2.imwrite(output_path, adjusted, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(output_path, adjusted)

        return True
    except Exception:
        return False


def batch_adjust_images(
    input_dir: str,
    output_dir: str,
    shadow: int = 50,
    highlight: int = 50,
    quality: int = 95,
    progress_callback=None
) -> Tuple[int, int]:
    """
    Apply shadow/highlight adjustments to all images in a directory.

    Returns:
        (success_count, error_count)
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

        out_file = output_path / img_path.name
        if process_image_with_adjustments(
            str(img_path), str(out_file), shadow, highlight, quality
        ):
            success += 1
        else:
            errors += 1

    return success, errors


# CLI interface
def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Apply shadow/highlight adjustments")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output image or directory")
    parser.add_argument("--shadow", "-s", type=int, default=50,
                        help="Shadow adjustment 0-100 (default: 50)")
    parser.add_argument("--highlight", "-l", type=int, default=50,
                        help="Highlight adjustment 0-100 (default: 50)")
    parser.add_argument("--quality", "-q", type=int, default=95,
                        help="JPEG quality (default: 95)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        def progress(curr, total, name):
            print(f"[{curr}/{total}] {name}")

        success, errors = batch_adjust_images(
            args.input, args.output,
            args.shadow, args.highlight, args.quality,
            progress
        )
        print(f"Processed {success} images, {errors} errors")
    else:
        if process_image_with_adjustments(
            args.input, args.output,
            args.shadow, args.highlight, args.quality
        ):
            print(f"Saved: {args.output}")
        else:
            print("Error processing image")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())

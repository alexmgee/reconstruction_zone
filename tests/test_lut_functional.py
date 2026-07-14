"""
Functional tests for LUTProcessor — identity, inversion, strength blending,
and error handling for both apply_uint8 and apply_lut compatibility wrapper.

These verify the lut.py additions from commit 2741e55 are correct.
"""

import numpy as np
import pytest

from prep360.core.lut import LUTProcessor


@pytest.fixture
def processor():
    return LUTProcessor()


@pytest.fixture
def identity_cube(tmp_path):
    """Write a minimal identity .cube LUT (size 2).

    .cube format: R varies fastest, then G, then B.
    Each line is an RGB output triplet.
    Identity means output = input for every corner.
    """
    path = tmp_path / "identity.cube"
    lines = [
        "TITLE \"Identity\"",
        "LUT_3D_SIZE 2",
        # 8 entries: R fastest, G middle, B slowest
        "0.0 0.0 0.0",  # R=0 G=0 B=0 → black
        "1.0 0.0 0.0",  # R=1 G=0 B=0 → red
        "0.0 1.0 0.0",  # R=0 G=1 B=0 → green
        "1.0 1.0 0.0",  # R=1 G=1 B=0 → yellow
        "0.0 0.0 1.0",  # R=0 G=0 B=1 → blue
        "1.0 0.0 1.0",  # R=1 G=0 B=1 → magenta
        "0.0 1.0 1.0",  # R=0 G=1 B=1 → cyan
        "1.0 1.0 1.0",  # R=1 G=1 B=1 → white
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def invert_cube(tmp_path):
    """Write a .cube LUT that inverts colors (size 2).

    .cube format: R varies fastest, then G, then B.
    Invert means output = 1 - input for each channel.
    """
    path = tmp_path / "invert.cube"
    lines = [
        "TITLE \"Invert\"",
        "LUT_3D_SIZE 2",
        # R fastest, G middle, B slowest — each output = 1-input
        "1.0 1.0 1.0",  # R=0 G=0 B=0 → white
        "0.0 1.0 1.0",  # R=1 G=0 B=0 → cyan (1-R, 1-G, 1-B)
        "1.0 0.0 1.0",  # R=0 G=1 B=0 → magenta
        "0.0 0.0 1.0",  # R=1 G=1 B=0 → blue
        "1.0 1.0 0.0",  # R=0 G=0 B=1 → yellow
        "0.0 1.0 0.0",  # R=1 G=0 B=1 → green
        "1.0 0.0 0.0",  # R=0 G=1 B=1 → red
        "0.0 0.0 0.0",  # R=1 G=1 B=1 → black
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def test_image_uint8():
    """A small test image with known BGR values."""
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [0, 0, 0]        # black
    img[0, 1] = [255, 0, 0]      # blue (BGR)
    img[1, 0] = [0, 255, 0]      # green
    img[1, 1] = [255, 255, 255]  # white
    return img


class TestIdentityLUT:
    def test_identity_preserves_pixels(self, processor, identity_cube, test_image_uint8):
        lut_3d, _ = processor.load_cube(identity_cube)
        result = processor.apply_uint8(test_image_uint8, lut_3d, strength=1.0)
        np.testing.assert_array_equal(result, test_image_uint8)

    def test_identity_via_apply_lut(self, processor, identity_cube, test_image_uint8):
        result = processor.apply_lut(test_image_uint8, identity_cube, strength=1.0)
        np.testing.assert_array_equal(result, test_image_uint8)


class TestInvertLUT:
    def test_invert_corners(self, processor, invert_cube, test_image_uint8):
        lut_3d, _ = processor.load_cube(invert_cube)
        result = processor.apply_uint8(test_image_uint8, lut_3d, strength=1.0)
        # Black → white
        np.testing.assert_array_equal(result[0, 0], [255, 255, 255])
        # White → black
        np.testing.assert_array_equal(result[1, 1], [0, 0, 0])

    def test_invert_via_apply_lut(self, processor, invert_cube, test_image_uint8):
        result = processor.apply_lut(test_image_uint8, invert_cube, strength=1.0)
        np.testing.assert_array_equal(result[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(result[1, 1], [0, 0, 0])


class TestStrengthBlending:
    def test_strength_zero_preserves_input(self, processor, invert_cube, test_image_uint8):
        lut_3d, _ = processor.load_cube(invert_cube)
        result = processor.apply_uint8(test_image_uint8, lut_3d, strength=0.0)
        np.testing.assert_array_equal(result, test_image_uint8)

    def test_strength_half_blends(self, processor, invert_cube):
        # Pure black with invert at 50% should give mid-gray
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        lut_3d, _ = processor.load_cube(invert_cube)
        result = processor.apply_uint8(img, lut_3d, strength=0.5)
        # 0 * 0.5 + 255 * 0.5 = 127.5 → 128 (rounded)
        expected = 128
        assert abs(int(result[0, 0, 0]) - expected) <= 1
        assert abs(int(result[0, 0, 1]) - expected) <= 1
        assert abs(int(result[0, 0, 2]) - expected) <= 1


class TestErrorHandling:
    def test_malformed_cube_raises(self, processor, tmp_path):
        bad = tmp_path / "bad.cube"
        bad.write_text("this is not a valid cube file\n")
        with pytest.raises(Exception):
            processor.load_cube(str(bad))

    def test_nonexistent_cube_raises(self, processor):
        with pytest.raises(Exception):
            processor.load_cube("/nonexistent/path/to.cube")

    def test_wrong_dtype_raises(self, processor, identity_cube):
        lut_3d, _ = processor.load_cube(identity_cube)
        float_img = np.zeros((2, 2, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            processor.apply_uint8(float_img, lut_3d)

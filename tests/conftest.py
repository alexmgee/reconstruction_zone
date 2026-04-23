"""Shared fixtures for Reconstruction Zone tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that auto-cleans."""
    return tmp_path


@pytest.fixture
def sample_image_rgb():
    """A small 64x64 RGB test image as numpy array (uint8)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_equirect():
    """A 256x128 equirectangular-shaped test image (2:1 aspect ratio)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (128, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask_binary():
    """A 64x64 binary mask with a circle in the center (0/1 uint8)."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    cy, cx, r = 32, 32, 15
    yy, xx = np.ogrid[:64, :64]
    mask[((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2] = 1
    return mask

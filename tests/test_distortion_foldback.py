"""Tests for distortion polynomial fold-back detection.

Ported from upstream Fisheye-to-Cubemap tests/test_distortion_foldback.py
(commit d7dc6fd back-port; see docs/2026-07-15-foldback-backport-plan.md).
The two upstream single-pinhole tests are excluded (gui.adaptive_undistort
is not vendored). Three tests added per the adversarial plan review:
corrections-path wiring capture, dark-input failure, and the partial-mask
no-op equivalence with its premise asserted in-test.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from prep360.core.cubeface_engine import (
    monotonic_radius_limit,
    monotonic_distorted_radius_limit,
    compute_monotonic_mask,
)


# --- monotonic_radius_limit (undistorted normalized space) ---

def test_no_distortion_returns_none():
    """Zero distortion coefficients — model is monotonic everywhere."""
    assert monotonic_radius_limit(0.0, 0.0, 0.0, 0.0) is None


def test_positive_coefficients_returns_none():
    """All-positive coefficients — D is monotonically increasing, no fold."""
    assert monotonic_radius_limit(0.1, 0.05, 0.01, 0.0) is None


def test_oa4_calibration_finds_limit():
    """DJI Osmo Action 4 — known fold-back from the original bug."""
    k1, k2, k3 = 0.195, 0.177, -0.090
    limit = monotonic_radius_limit(k1, k2, k3, 0.0)
    assert limit is not None
    assert limit > 0
    # r_u_limit ≈ 1.473 (normalized, not pixel)
    assert 1.4 < limit < 1.6


def test_k4_only_foldback():
    """k4 negative enough to cause fold-back even with k1-k3 zero."""
    limit = monotonic_radius_limit(0.0, 0.0, 0.0, -0.001)
    assert limit is not None
    assert limit > 0


def test_k4_extends_monotonic_region():
    """Positive k4 should push the fold boundary further out (or eliminate it)."""
    k1, k2, k3 = 0.195, 0.177, -0.090
    limit_no_k4 = monotonic_radius_limit(k1, k2, k3, 0.0)
    limit_pos_k4 = monotonic_radius_limit(k1, k2, k3, 0.01)
    assert limit_no_k4 is not None
    if limit_pos_k4 is not None:
        assert limit_pos_k4 > limit_no_k4
    # else: k4 eliminated the fold entirely, also valid


def test_k4_contracts_monotonic_region():
    """Negative k4 should pull the fold boundary inward."""
    k1, k2, k3 = 0.195, 0.177, -0.090
    limit_no_k4 = monotonic_radius_limit(k1, k2, k3, 0.0)
    limit_neg_k4 = monotonic_radius_limit(k1, k2, k3, -0.01)
    assert limit_no_k4 is not None
    assert limit_neg_k4 is not None
    assert limit_neg_k4 < limit_no_k4


def test_x5_calibration():
    """Insta360 X5 front lens — the dataset that exposed the X artifact."""
    k1, k2, k3 = 0.01195, 0.08137, -0.02200
    limit = monotonic_radius_limit(k1, k2, k3, 0.0)
    assert limit is not None


def test_isreal_tolerance(monkeypatch):
    """Roots with tiny imaginary parts from float arithmetic should be found."""
    original_roots = np.roots

    def roots_with_residue(coeffs):
        result = original_roots(coeffs)
        return result + 1e-13j  # simulate float noise in companion matrix eigenvalues

    monkeypatch.setattr(np, "roots", roots_with_residue)
    # OA4 coefficients — known to have a real root
    limit = monotonic_radius_limit(0.195, 0.177, -0.090, 0.0)
    assert limit is not None


# --- monotonic_distorted_radius_limit ---

def test_distorted_limit_exceeds_undistorted_for_oa4():
    """For OA4, D(r_u²) > 1 at fold boundary, so r_d_limit > r_u_limit."""
    k1, k2, k3 = 0.195, 0.177, -0.090
    r_u = monotonic_radius_limit(k1, k2, k3, 0.0)
    r_d = monotonic_distorted_radius_limit(k1, k2, k3, 0.0)
    assert r_u is not None and r_d is not None
    assert r_d > r_u
    # OA4: r_d ≈ 1.97, r_u ≈ 1.47
    assert 1.9 < r_d < 2.1


def test_distorted_limit_none_when_monotonic():
    """No fold → both functions return None."""
    assert monotonic_distorted_radius_limit(0.1, 0.05, 0.01, 0.0) is None


# --- compute_monotonic_mask ---

def test_monotonic_mask_shape():
    """Mask has correct shape and dtype."""
    mask = compute_monotonic_mask(64, 64, 32.0, 0.0, 0.0, 0.195, 0.177, -0.090, 0.0)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


def test_monotonic_mask_all_true_when_no_foldback():
    """No fold-back: every pixel is valid."""
    mask = compute_monotonic_mask(64, 64, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert mask.all()


def test_monotonic_mask_excludes_corners():
    """With fold-back, corners (largest radius) should be excluded first."""
    mask = compute_monotonic_mask(
        100, 100, 30.0, 0.0, 0.0, 0.195, 0.177, -0.090, 0.0
    )
    assert mask[50, 50]  # center valid
    assert not mask.all()  # some excluded
    if not mask[0, 50]:  # if edge excluded...
        assert not mask[0, 0]  # ...corner must also be excluded


def test_monotonic_mask_respects_optical_center_offset():
    """Mask circle should be centered on optical center, not pixel center."""
    mask_centered = compute_monotonic_mask(
        100, 100, 30.0, 0.0, 0.0, 0.195, 0.177, -0.090, 0.0
    )
    mask_offset = compute_monotonic_mask(
        100, 100, 30.0, 10.0, 0.0, 0.195, 0.177, -0.090, 0.0
    )
    assert not np.array_equal(mask_centered, mask_offset)


def test_monotonic_mask_uses_distorted_not_undistorted_radius():
    """The mask must compare in distorted space — valid pixels exist between
    f*r_u_limit and f*r_d_limit that would be wrongly excluded in undistorted space."""
    f = 1261.88
    k1, k2, k3, k4 = 0.195, 0.177, -0.090, 0.0
    r_u = monotonic_radius_limit(k1, k2, k3, k4)
    r_d = monotonic_distorted_radius_limit(k1, k2, k3, k4)
    assert r_u is not None and r_d is not None

    # A pixel at distorted radius between r_u and r_d should be VALID.
    # In a sensor large enough to contain this radius:
    mid_radius_px = f * (r_u + r_d) / 2.0  # between the two limits
    sensor_size = int(mid_radius_px * 2.5)  # large enough sensor
    mask = compute_monotonic_mask(sensor_size, sensor_size, f, 0.0, 0.0, k1, k2, k3, k4)

    # Pixel at mid_radius from center should be valid
    center = sensor_size // 2
    test_x = center + int(mid_radius_px)
    if test_x < sensor_size:
        assert mask[center, test_x], (
            f"Pixel at distorted radius {mid_radius_px:.0f} px "
            f"(between f*r_u={f*r_u:.0f} and f*r_d={f*r_d:.0f}) "
            f"should be valid but was excluded"
        )


def test_monotonic_mask_respects_b1_b2():
    """B1/B2 affinity terms should affect the mask geometry."""
    mask_no_b = compute_monotonic_mask(
        100, 100, 30.0, 0.0, 0.0, 0.195, 0.177, -0.090, 0.0, b1=0.0, b2=0.0
    )
    mask_with_b = compute_monotonic_mask(
        100, 100, 30.0, 0.0, 0.0, 0.195, 0.177, -0.090, 0.0, b1=5.0, b2=3.0
    )
    # With large B1/B2, the normalized coordinates change, so the mask differs
    assert not np.array_equal(mask_no_b, mask_with_b)


# --- Integration: multi-pinhole path ---

def _oa4_params():
    """OA4-like params tuple: (f, cx, cy, k1, k2, k3, k4, p1, p2, b1, b2)."""
    return (1261.88, 7.63, 6.65, 0.195, 0.177, -0.090, 0.0, 0.0002, 0.001, 0.0, 0.0)


def test_multipinhole_useful_mask_excludes_foldback():
    """The useful-pixel mask from the multi-pinhole path must exclude fold-back pixels."""
    from prep360.core.cubeface_engine import compute_metashape_rays_usefulpixmap

    scale = 128 / 3840
    f_orig, cx, cy, k1, k2, k3, k4, p1, p2, b1, b2 = _oa4_params()
    f_scaled = f_orig * scale
    params = (f_scaled, cx * scale, cy * scale, k1, k2, k3, k4, p1, p2, b1, b2)

    with tempfile.TemporaryDirectory() as tmpdir:
        rays, mask, maxangle = compute_metashape_rays_usefulpixmap(
            128, 128, params, 180.0, tmpdir, model="equidistant",
        )

    r_d_limit = monotonic_distorted_radius_limit(k1, k2, k3, k4)
    if r_d_limit is not None:
        # Corner (0, 0) distorted normalized radius
        up = 0 - (64 + cx * scale)
        vp = 0 - (64 + cy * scale)
        yp = vp / f_scaled
        xp = up / f_scaled
        corner_r_d = np.sqrt(xp**2 + yp**2)
        if corner_r_d > r_d_limit:
            assert mask[0, 0] == 0
            assert mask[127, 127] == 0


# --- Integration: corrected-rays path ---

def test_corrected_rays_mask_excludes_foldback():
    """derive_useful_pixel_mask must exclude fold-back pixels when given monotonic_mask."""
    from prep360.core.corrected_rays import compute_rays_with_corrections, derive_useful_pixel_mask

    scale = 128 / 3840
    f_orig = 1261.88
    params = (
        f_orig * scale,   # f
        7.63 * scale,     # cx
        6.65 * scale,     # cy
        0.195, 0.177, -0.090, 0.0,  # k1-k4
        0.0002, 0.001,    # p1, p2
        0.0, 0.0,         # b1, b2
    )
    rays, _ = compute_rays_with_corrections(128, 128, params, "equidistant")
    mono_mask = compute_monotonic_mask(
        128, 128, params[0], params[1], params[2],
        params[3], params[4], params[5], params[6],
    )
    mask, omega, maxangle = derive_useful_pixel_mask(
        rays, monotonic_mask=mono_mask,
    )

    r_d_limit = monotonic_distorted_radius_limit(params[3], params[4], params[5], params[6])
    if r_d_limit is not None:
        # Corner (0,0) distorted normalized radius
        up = 0 - (64 + params[1])
        vp = 0 - (64 + params[2])
        yp = vp / params[0]
        xp = up / params[0]
        corner_r_d = np.sqrt(xp**2 + yp**2)
        if corner_r_d > r_d_limit:
            assert mask[0, 0] == 0
            assert mask[127, 127] == 0


# --- Back-port additions (adversarial plan review findings) ---

def _write_calibration_xml(path: Path, projection: str, width: int, height: int,
                           f: float, ks=(0.0, 0.0, 0.0, 0.0), corrections: bool = False):
    """Minimal standalone Metashape <calibration> export."""
    corr = ""
    if corrections:
        coeffs = " ".join(["0"] * 96)
        corr = (
            f'<corrections type="fourier"><coeffs>{coeffs}</coeffs>'
            f"<extent><min>0 0</min><max>{width} {height}</max></extent></corrections>"
        )
    k1, k2, k3, k4 = ks
    path.write_text(
        f"<calibration><projection>{projection}</projection>"
        f"<width>{width}</width><height>{height}</height>"
        f"<f>{f}</f><cx>0</cx><cy>0</cy>"
        f"<k1>{k1}</k1><k2>{k2}</k2><k3>{k3}</k3><k4>{k4}</k4>"
        f"<p1>0</p1><p2>0</p2><b1>0</b1><b2>0</b2>"
        f"<date>test</date>{corr}</calibration>",
        encoding="utf-8",
    )


def _write_disc_image(path: Path, size: int, radius: int, value: int = 200):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (value, value, value), -1)
    cv2.imwrite(str(path), img)


def test_corrections_path_receives_monotonic_mask(tmp_path, monkeypatch):
    """process_cubeface_sensor must pass a real monotonic_mask into
    derive_useful_pixel_mask on the corrections path (plan-review finding #3:
    omitting the wiring would pass every other test)."""
    import prep360.core.cubeface_processing as cp

    size = 64
    xml = tmp_path / "calibration.xml"
    _write_calibration_xml(xml, "equidistant_fisheye", size, size, f=20.0,
                           ks=(0.195, 0.177, -0.090, 0.0), corrections=True)
    images = tmp_path / "images"
    images.mkdir()
    _write_disc_image(images / "img_001.png", size, radius=24)

    captured = {}
    original = cp.derive_useful_pixel_mask

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cp, "derive_useful_pixel_mask", capture)

    cp.process_cubeface_sensor(
        calibration_xml=xml,
        image_dirs=[images],
        output_dir=tmp_path / "out",
        face_width=16,
    )

    mono = captured.get("monotonic_mask")
    assert mono is not None, "corrections path did not pass monotonic_mask"
    expected = compute_monotonic_mask(size, size, 20.0, 0.0, 0.0,
                                      0.195, 0.177, -0.090, 0.0, b1=0.0, b2=0.0)
    assert np.array_equal(mono, expected)


def test_dark_images_no_masks_raises(tmp_path):
    """No masks + all-dark images: image-derived lens support must fail loudly,
    not silently remap fold-back garbage (plan-review finding #4)."""
    from prep360.core.cubeface_processing import process_cubeface_sensor

    size = 64
    xml = tmp_path / "calibration.xml"
    _write_calibration_xml(xml, "equidistant_fisheye", size, size, f=20.0)
    images = tmp_path / "images"
    images.mkdir()
    dark = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(images / "img_001.png"), dark)

    with pytest.raises(RuntimeError, match="no bright pixels"):
        process_cubeface_sensor(
            calibration_xml=xml,
            image_dirs=[images],
            output_dir=tmp_path / "out",
            face_width=16,
        )


def test_partial_mask_noop_when_support_inside_monotonic_region(tmp_path):
    """For a lens whose pre-fix useful mask lies entirely inside mono_mask
    (e.g. Osmo360), the guard must be a strict no-op: identical maxangle and
    identical useful_pixel_mask. The premise is ASSERTED, not assumed
    (plan-review finding #2)."""
    from prep360.core.cubeface_engine import (
        compute_metashape_rays_usefulpixmap,
        compute_rays,
        filter_center_component,
    )

    size = 128
    scale = size / 3840
    # Osmo360 front lens (equisolid), scaled
    params = (1047.898 * scale, -2.403 * scale, -0.124 * scale,
              0.0559, 0.0114, -0.0095, 0.0005, 0.0, 0.0, 0.0, 0.0)

    # Partial sensor-sized mask: disc well inside the lens circle. Single mask,
    # never jointly full-frame, so the full-frame fallback cannot trigger.
    mask = np.zeros((size, size), dtype=np.uint16)
    cv2.circle(mask, (size // 2, size // 2), 40, 1, -1)
    assert not np.all(mask > 0)

    # Pre-fix derivation, replicated: support = mask > 0, no mono anywhere.
    rays, _phi = compute_rays(size, size, params, "equisolid")
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rz = np.clip(rays[..., 2] / norms[..., 0], -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(rz))
    maxangle_old = float(theta_deg[mask > 0].max())
    upm_old = filter_center_component(theta_deg, maxangle_old)

    # PREMISE (asserted): every pre-fix useful pixel is inside mono_mask.
    mono = compute_monotonic_mask(size, size, *params[:7],
                                  b1=params[9], b2=params[10])
    assert mono[upm_old > 0].all(), (
        "fixture invalid: pre-fix useful mask extends into the fold-back zone"
    )

    # Post-fix path with the same mask-derived support.
    with tempfile.TemporaryDirectory() as tmpdir:
        _rays, upm_new, maxangle_new = compute_metashape_rays_usefulpixmap(
            size, size, params, 180.0, tmpdir, model="equisolid",
            maskpixelcount=mask,
        )

    assert maxangle_new == pytest.approx(maxangle_old, abs=1e-9)
    assert np.array_equal(upm_new, upm_old)

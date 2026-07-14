"""Tests for ERP reframe output layouts."""

import cv2
import numpy as np
import pytest

from prep360.core.reframer import (
    FreeView,
    OutputLayout,
    Reframer,
    ViewConfig,
    copy_view_config,
    _process_mask_crop,
    resolve_output_paths,
)


def _tiny_erp() -> np.ndarray:
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:, :64] = (255, 0, 0)
    img[:, 64:] = (0, 255, 0)
    return img


def _two_view_config() -> ViewConfig:
    return ViewConfig(
        views=[
            FreeView("00_00", 0.0, 0.0, 90),
            FreeView("00_01", 90.0, 0.0, 90),
        ],
        include_zenith=False,
        include_nadir=False,
        output_size=32,
        jpeg_quality=95,
    )


@pytest.fixture
def erp_frame(tmp_path):
    path = tmp_path / "input" / "frame_a.jpg"
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), _tiny_erp())
    return tmp_path / "input"


@pytest.fixture
def erp_mask(tmp_path, erp_frame):
    mask_path = tmp_path / "masks_erp" / "frame_a.png"
    mask_path.parent.mkdir(parents=True)
    m = np.ones((64, 128), dtype=np.uint8) * 255
    cv2.imwrite(str(mask_path), m)
    return tmp_path / "masks_erp"


class TestResolveOutputPaths:
    def test_rig_paths(self, tmp_path):
        p = resolve_output_paths(tmp_path, OutputLayout.RIG, "00_00", "frame_a", True)
        assert p.image_path == tmp_path / "images" / "00_00" / "frame_a.jpg"
        assert p.mask_path == tmp_path / "masks" / "00_00" / "frame_a.png"

    def test_station_paths(self, tmp_path):
        p = resolve_output_paths(tmp_path, OutputLayout.STATION, "00_00", "frame_a", True)
        assert p.image_path == tmp_path / "images" / "frame_a" / "frame_a_00_00.jpg"
        assert p.mask_path == tmp_path / "masks" / "frame_a" / "frame_a_00_00.png"

    def test_flat_paths(self, tmp_path):
        p = resolve_output_paths(tmp_path, OutputLayout.FLAT, "00_00", "frame_a", True)
        assert p.image_path == tmp_path / "images" / "frame_a_00_00.jpg"
        assert p.mask_path == tmp_path / "masks" / "frame_a_00_00.png"


class TestReframeLayouts:
    def test_rig_layout(self, erp_frame, tmp_path):
        out = tmp_path / "out_rig"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(str(erp_frame), str(out), output_layout=OutputLayout.RIG)
        assert (out / "images" / "00_00" / "frame_a.jpg").is_file()
        assert (out / "images" / "00_01" / "frame_a.jpg").is_file()
        assert (out / "reframe_metadata.json").is_file()
        assert (out / "rig_config.json").is_file()

    def test_station_layout(self, erp_frame, tmp_path):
        out = tmp_path / "out_station"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(str(erp_frame), str(out), output_layout=OutputLayout.STATION)
        assert (out / "images" / "frame_a" / "frame_a_00_00.jpg").is_file()
        assert (out / "reframe_metadata.json").is_file()
        assert not (out / "rig_config.json").exists()

    def test_flat_layout(self, erp_frame, tmp_path):
        out = tmp_path / "out_flat"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(str(erp_frame), str(out), output_layout=OutputLayout.FLAT)
        assert (out / "images" / "frame_a_00_00.jpg").is_file()

    def test_legacy_station_dirs_true_maps_to_station(self, erp_frame, tmp_path):
        out = tmp_path / "out_legacy_station"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(str(erp_frame), str(out), station_dirs=True)
        assert (out / "images" / "frame_a" / "frame_a_00_00.jpg").is_file()
        assert not (out / "images" / "00_00" / "frame_a.jpg").exists()

    def test_legacy_station_dirs_false_maps_to_flat(self, erp_frame, tmp_path):
        out = tmp_path / "out_legacy_flat"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(str(erp_frame), str(out), station_dirs=False)
        assert (out / "images" / "frame_a_00_00.jpg").is_file()
        assert not (out / "images" / "00_00" / "frame_a.jpg").exists()

    def test_mask_parallel_tree_rig(self, erp_frame, erp_mask, tmp_path):
        out = tmp_path / "out_mask"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(
            str(erp_frame), str(out), mask_dir=str(erp_mask), output_layout=OutputLayout.RIG,
        )
        mask = cv2.imread(str(out / "masks" / "00_00" / "frame_a.png"), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert set(np.unique(mask)).issubset({0, 255})

    def test_mask_erosion_shrinks_foreground(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        eroded = _process_mask_crop(mask)
        assert set(np.unique(eroded)).issubset({0, 255})
        assert np.count_nonzero(eroded) < np.count_nonzero(mask)

    def test_mask_station_layout(self, erp_frame, erp_mask, tmp_path):
        out = tmp_path / "out_mask_st"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(
            str(erp_frame), str(out), mask_dir=str(erp_mask), output_layout=OutputLayout.STATION,
        )
        assert (out / "masks" / "frame_a" / "frame_a_00_00.png").is_file()

    def test_duplicate_mask_stems_warn_and_first_wins(self, erp_frame, tmp_path):
        mask_dir = tmp_path / "masks_dup"
        mask_dir.mkdir()
        mask = np.ones((64, 128), dtype=np.uint8) * 255
        cv2.imwrite(str(mask_dir / "frame_a.jpg"), mask)
        cv2.imwrite(str(mask_dir / "frame_a.png"), mask)

        out = tmp_path / "out_mask_dup"
        logs = []
        cfg = copy_view_config(_two_view_config())
        result = Reframer(cfg).reframe_batch(
            str(erp_frame), str(out), mask_dir=str(mask_dir),
            output_layout=OutputLayout.RIG, log=logs.append,
        )
        assert result.success
        assert any("duplicate mask stems ignored" in msg for msg in logs)

    def test_masks_in_layers_subdirectory(self, erp_frame, tmp_path):
        layer_dir = tmp_path / "maskset" / "layers" / "photographer"
        layer_dir.mkdir(parents=True)
        mask = np.ones((64, 128), dtype=np.uint8) * 255
        cv2.imwrite(str(layer_dir / "frame_a.png"), mask)

        out = tmp_path / "out_mask_layers"
        logs = []
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(
            str(erp_frame), str(out), mask_dir=str(tmp_path / "maskset"),
            output_layout=OutputLayout.RIG, log=logs.append,
        )
        assert (out / "masks" / "00_00" / "frame_a.png").is_file()
        assert any("layers/photographer" in msg for msg in logs)

    def test_mask_prefix_stem_matching(self, erp_frame, tmp_path):
        mask_dir = tmp_path / "masks_prefixed"
        mask_dir.mkdir()
        mask = np.ones((64, 128), dtype=np.uint8) * 255
        cv2.imwrite(str(mask_dir / "mask_frame_a.png"), mask)

        out = tmp_path / "out_mask_prefixed"
        cfg = copy_view_config(_two_view_config())
        Reframer(cfg).reframe_batch(
            str(erp_frame), str(out), mask_dir=str(mask_dir),
            output_layout=OutputLayout.RIG,
        )
        assert (out / "masks" / "00_00" / "frame_a.png").is_file()

"""Tests for prep360 Ring/ViewConfig geometry."""


import pytest

from prep360.core.reframer import (
    FreeView,
    Ring,
    ViewConfig,
    VIEW_PRESETS,
    copy_view_config,
    get_view_preset,
    resolve_preset_name,
    validate_view_config,
)


class TestRing:
    def test_yaw_positions_count_8(self):
        ring = Ring(pitch=0, count=8, fov=65)
        yaws = ring.get_yaw_positions()
        assert len(yaws) == 8
        assert yaws[0] == 0.0
        assert yaws[1] == 45.0
        assert yaws[-1] == 315.0

    def test_yaw_positions_with_start_offset(self):
        ring = Ring(pitch=0, count=4, fov=65, start_yaw=10.0)
        yaws = ring.get_yaw_positions()
        assert yaws == [10.0, 100.0, 190.0, 280.0]

    def test_yaw_positions_count_zero(self):
        ring = Ring(pitch=0, count=0)
        assert ring.get_yaw_positions() == []

    def test_single_view_ring(self):
        ring = Ring(pitch=90, count=1)
        assert ring.get_yaw_positions() == [0.0]


class TestViewConfig:
    def test_total_views_default(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=8), Ring(pitch=-20, count=4)],
            include_zenith=True,
            include_nadir=False,
        )
        assert cfg.total_views() == 13

    def test_total_views_with_nadir(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=8)],
            include_zenith=True,
            include_nadir=True,
        )
        assert cfg.total_views() == 10

    def test_total_views_no_poles(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=6)],
            include_zenith=False,
            include_nadir=False,
        )
        assert cfg.total_views() == 6

    def test_get_all_views_names(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=2)],
            include_zenith=True,
            include_nadir=False,
        )
        views = cfg.get_all_views()
        names = [v.name for v in views]
        assert names == ["00_00", "00_01", "ZN_00"]

    def test_freeview_order(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=1)],
            views=[FreeView("FV_01", 10, 5, 80)],
            include_zenith=False,
            include_nadir=False,
        )
        names = [v.name for v in cfg.get_all_views()]
        assert names == ["00_00", "FV_01"]

    def test_flip_vertical_on_ring(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=90, count=1, flip_vertical=True)],
            include_zenith=False,
            include_nadir=False,
        )
        assert cfg.get_all_views()[0].flip_vertical is True

    def test_get_all_views_zenith_pitch(self):
        cfg = ViewConfig(
            rings=[],
            include_zenith=True,
            include_nadir=True,
        )
        views = cfg.get_all_views()
        assert len(views) == 2
        assert views[0].pitch == 90
        assert views[1].pitch == -90

    def test_empty_config_validation(self):
        cfg = ViewConfig(rings=[], include_zenith=False, include_nadir=False)
        assert cfg.total_views() == 0
        with pytest.raises(ValueError):
            validate_view_config(cfg)

    def test_serialization_roundtrip(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=-35, count=4, fov=90, flip_vertical=True)],
            views=[FreeView("01_00", 22.5, 35, 90)],
            include_zenith=False,
            include_nadir=False,
            output_size=1600,
            jpeg_quality=92,
        )
        cfg2 = ViewConfig.from_dict(cfg.to_dict())
        assert cfg2.to_dict() == cfg.to_dict()

    def test_copy_view_config_independent(self):
        cfg = get_view_preset("medium")
        copy = copy_view_config(cfg)
        copy.output_size = 512
        assert VIEW_PRESETS["medium"].output_size != 512

    def test_unknown_preset_rejected(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            resolve_preset_name("__missing__")

    def test_validate_duplicate_names(self):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_00", 45, 0, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        with pytest.raises(ValueError, match="Duplicate"):
            validate_view_config(cfg)


class TestMediumPresetSnapshot:
    """Preset order frozen for rig reference sensor contract."""

    EXPECTED = [
        ("00_00", 0.0, -35.0, 90.0),
        ("00_01", 45.0, -35.0, 90.0),
        ("00_02", 90.0, -35.0, 90.0),
        ("00_03", 135.0, -35.0, 90.0),
        ("00_04", 180.0, -35.0, 90.0),
        ("00_05", -135.0, -35.0, 90.0),
        ("00_06", -90.0, -35.0, 90.0),
        ("00_07", -45.0, -35.0, 90.0),
        ("01_00", 22.5, 35.0, 90.0),
        ("01_01", 67.5, 35.0, 90.0),
        ("01_02", 112.5, 35.0, 90.0),
        ("01_03", 157.5, 35.0, 90.0),
        ("01_04", -157.5, 35.0, 90.0),
        ("01_05", -112.5, 35.0, 90.0),
        ("01_06", -67.5, 35.0, 90.0),
        ("01_07", -22.5, 35.0, 90.0),
    ]

    def test_medium_view_order(self):
        views = VIEW_PRESETS["medium"].get_all_views()
        assert len(views) == 16
        for got, exp in zip(views, self.EXPECTED):
            assert got.name == exp[0]
            assert got.yaw == exp[1]
            assert got.pitch == exp[2]
            assert got.fov == exp[3]
            assert got.flip_vertical is False

    def test_medium_total_views(self):
        assert VIEW_PRESETS["medium"].total_views() == 16

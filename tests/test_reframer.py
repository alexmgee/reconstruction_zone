"""Tests for prep360 Ring/ViewConfig geometry."""

import math

from prep360.core.reframer import Ring, ViewConfig


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
        assert cfg.total_views() == 13  # 8 + 4 + 1 zenith

    def test_total_views_with_nadir(self):
        cfg = ViewConfig(
            rings=[Ring(pitch=0, count=8)],
            include_zenith=True,
            include_nadir=True,
        )
        assert cfg.total_views() == 10  # 8 + zenith + nadir

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
        names = [v[3] for v in views]
        assert names == ["00_00", "00_01", "ZN_00"]

    def test_get_all_views_zenith_pitch(self):
        cfg = ViewConfig(
            rings=[],
            include_zenith=True,
            include_nadir=True,
        )
        views = cfg.get_all_views()
        assert len(views) == 2
        assert views[0][1] == 90   # zenith pitch
        assert views[1][1] == -90  # nadir pitch

    def test_empty_config(self):
        cfg = ViewConfig(rings=[], include_zenith=False, include_nadir=False)
        assert cfg.total_views() == 0
        assert cfg.get_all_views() == []

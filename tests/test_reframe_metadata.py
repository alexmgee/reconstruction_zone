"""Tests for reframe_metadata.json writer."""

import json

import pytest

from prep360.core.reframer import FreeView, OutputLayout, ViewConfig
from prep360.core.reframe_metadata import build_reframe_metadata, write_reframe_metadata


class TestReframeMetadata:
    def test_rig_sensors(self, tmp_path):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, -35, 90), FreeView("00_01", 45, -35, 90)],
            include_zenith=False,
            include_nadir=False,
            output_size=1600,
        )
        images = [tmp_path / "frame_0001.jpg"]
        meta = build_reframe_metadata(cfg, OutputLayout.RIG, "medium", images)
        assert meta["output_layout"] == "rig"
        assert len(meta["metashape"]["sensors"]) == 2
        assert meta["metashape"]["sensors"][0]["ref_sensor"] is True
        assert "default_pinhole_intrinsics" in meta
        assert meta["frames"][0]["views"]["00_00"] == "images/00_00/frame_0001.jpg"

    def test_station_stations_list(self, tmp_path):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90)],
            include_zenith=False,
            include_nadir=False,
            output_size=512,
        )
        images = [tmp_path / "frame_0001.jpg"]
        meta = build_reframe_metadata(cfg, OutputLayout.STATION, "medium", images)
        assert len(meta["stations"]) == 1
        assert meta["stations"][0]["image_directory"] == "images/frame_0001/"

    def test_flat_metadata(self, tmp_path):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_01", 90, 0, 110)],
            include_zenith=False,
            include_nadir=False,
            output_size=512,
        )
        meta = build_reframe_metadata(cfg, OutputLayout.FLAT, "low")
        assert meta["output_layout"] == "flat"
        assert isinstance(meta["pinhole_intrinsics_by_fov"], dict)
        assert "90" in meta["pinhole_intrinsics_by_fov"]
        assert "110" in meta["pinhole_intrinsics_by_fov"]
        assert meta["default_pinhole_intrinsics"]["model"] == "PINHOLE"
        assert "90" not in meta["default_pinhole_intrinsics"]
        assert "110" not in meta["default_pinhole_intrinsics"]
        assert "default_pinhole_intrinsics or pinhole_intrinsics_by_fov" in (
            meta["metashape"]["pinhole_setup"]["note"]
        )

    def test_flat_metadata_frames_list(self, tmp_path):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_01", 90, 0, 90)],
            include_zenith=False,
            include_nadir=False,
            output_size=512,
        )
        images = [tmp_path / "frame_0001.jpg"]
        meta = build_reframe_metadata(cfg, OutputLayout.FLAT, "low", images)
        assert meta["frames"][0]["source_image"] == "frame_0001.jpg"
        assert meta["frames"][0]["views"]["00_01"] == "images/frame_0001_00_01.jpg"

    def test_write_file(self, tmp_path):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        path = write_reframe_metadata(tmp_path, cfg, OutputLayout.RIG, "medium")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 2

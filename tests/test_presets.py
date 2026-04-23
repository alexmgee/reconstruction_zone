"""Tests for prep360 preset load/save/validate."""

import json

from prep360.core.presets import Preset, PresetManager, BUILTIN_PRESETS
from prep360.core.reframer import Ring, ViewConfig


class TestPreset:
    def test_roundtrip_dict(self):
        p = Preset(name="test", description="A test preset")
        d = p.to_dict()
        p2 = Preset.from_dict(d)
        assert p2.name == "test"
        assert p2.description == "A test preset"

    def test_from_dict_missing_fields_uses_defaults(self):
        p = Preset.from_dict({"name": "minimal"})
        assert p.extraction_interval == 2.0
        assert p.reframe_zenith is True
        assert p.mask_confidence == 0.5

    def test_from_dict_empty_name(self):
        p = Preset.from_dict({})
        assert p.name == "Unnamed"

    def test_get_view_config(self):
        p = Preset(
            name="test",
            reframe_rings=[
                {"pitch": 0, "count": 8, "fov": 65},
                {"pitch": -20, "count": 4, "fov": 65},
            ],
            reframe_zenith=True,
            reframe_nadir=False,
            reframe_output_size=1920,
        )
        vc = p.get_view_config()
        assert isinstance(vc, ViewConfig)
        assert len(vc.rings) == 2
        assert vc.rings[0].count == 8
        assert vc.total_views() == 13

    def test_get_view_config_empty_rings(self):
        p = Preset(name="empty", reframe_rings=[])
        vc = p.get_view_config()
        assert vc.total_views() == 1  # zenith only (default)


class TestBuiltinPresets:
    def test_all_builtins_exist(self):
        expected = {"prep360_default", "dji_osmo_360", "insta360_x3", "gopro_max", "lightweight", "dense"}
        assert set(BUILTIN_PRESETS.keys()) == expected

    def test_all_builtins_have_rings(self):
        for name, preset in BUILTIN_PRESETS.items():
            assert len(preset.reframe_rings) > 0, f"{name} has no rings"

    def test_all_builtins_produce_valid_view_configs(self):
        for name, preset in BUILTIN_PRESETS.items():
            vc = preset.get_view_config()
            assert vc.total_views() > 0, f"{name} produces zero views"

    def test_default_preset_view_count(self):
        vc = BUILTIN_PRESETS["prep360_default"].get_view_config()
        assert vc.total_views() == 13  # 8 + 4 + zenith


class TestPresetManager:
    def test_save_and_load(self, tmp_dir):
        mgr = PresetManager(presets_dir=str(tmp_dir))
        p = Preset(name="my_preset", description="Test save")
        mgr.save(p)

        loaded = mgr.load("my_preset")
        assert loaded.name == "my_preset"
        assert loaded.description == "Test save"

    def test_load_builtin(self, tmp_dir):
        mgr = PresetManager(presets_dir=str(tmp_dir))
        p = mgr.load("prep360_default")
        assert p.name == "prep360_default"

    def test_load_nonexistent_raises(self, tmp_dir):
        mgr = PresetManager(presets_dir=str(tmp_dir))
        try:
            mgr.load("does_not_exist")
            assert False, "Should have raised"
        except (KeyError, FileNotFoundError, ValueError):
            pass

    def test_save_creates_json_file(self, tmp_dir):
        mgr = PresetManager(presets_dir=str(tmp_dir))
        mgr.save(Preset(name="checkfile"))
        assert (tmp_dir / "checkfile.json").exists()

        with open(tmp_dir / "checkfile.json") as f:
            data = json.load(f)
        assert data["name"] == "checkfile"

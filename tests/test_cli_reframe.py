"""Tests for prep360 reframe CLI behavior."""

from types import SimpleNamespace

from prep360.cli import cmd_reframe


def _args(**overrides):
    values = {
        "list_presets": False,
        "preset": "medium",
        "size": 32,
        "quality": 95,
        "layout": None,
        "stations": False,
        "info": True,
        "input": None,
        "output": None,
        "mask_dir": None,
        "workers": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestReframeCli:
    def test_unknown_preset_fails(self, capsys):
        code = cmd_reframe(_args(preset="__missing__"))
        captured = capsys.readouterr()
        assert code == 1
        assert "Unknown preset: __missing__" in captured.err

    def test_stations_conflicts_with_explicit_rig_layout(self, capsys):
        code = cmd_reframe(_args(stations=True, layout="rig"))
        captured = capsys.readouterr()
        assert code == 2
        assert "--stations conflicts with --layout" in captured.err

    def test_stations_allowed_when_layout_omitted(self):
        code = cmd_reframe(_args(stations=True, layout=None))
        assert code == 0

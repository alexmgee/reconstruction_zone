"""Phase D: backend qualification, selector preference, ERP capability refusal."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reconstruction_gui"))

from reconstruction_gui import colmap_runner as cr
from reconstruction_gui.colmap_runner import (
    ColmapRunner,
    cli_supports_camera_model,
    qualify_pycolmap_backend,
)


@pytest.fixture(autouse=True)
def _clean_qualification_caches():
    saved_py = dict(cr._PYCOLMAP_QUALIFICATION_CACHE)
    saved_cli = dict(cr._CLI_CAMERA_MODEL_CACHE)
    cr._PYCOLMAP_QUALIFICATION_CACHE.clear()
    cr._CLI_CAMERA_MODEL_CACHE.clear()
    yield
    cr._PYCOLMAP_QUALIFICATION_CACHE.clear()
    cr._PYCOLMAP_QUALIFICATION_CACHE.update(saved_py)
    cr._CLI_CAMERA_MODEL_CACHE.clear()
    cr._CLI_CAMERA_MODEL_CACHE.update(saved_cli)


def _make_runner(tmp_path: Path, **kwargs) -> ColmapRunner:
    fake_binary = tmp_path / "colmap.exe"
    if not fake_binary.exists():
        fake_binary.write_bytes(b"")
    defaults = dict(
        binary_path=str(fake_binary),
        camera_model="PINHOLE",
        workspace_root=str(tmp_path),
        engine_name="colmap",
    )
    defaults.update(kwargs)
    return ColmapRunner(**defaults)


def _failed_qualification(**overrides):
    result = {
        "importable": False, "version": "", "has_cuda": False,
        "caspar": False, "camera_models": [], "error": "forced by test",
    }
    result.update(overrides)
    return result


# ── qualification probe ──────────────────────────────────────────────

def test_qualification_probe_matches_in_process_pycolmap():
    pycolmap = pytest.importorskip("pycolmap")
    result = qualify_pycolmap_backend(force=True)
    assert result["importable"] is True
    assert result["version"] == str(pycolmap.__version__)


def test_qualification_excludes_broken_wheel_via_shim():
    # pycolmap lives in the user site-packages; PYTHONNOUSERSITE=1 makes the
    # fresh probe process unable to import it — the plan's forced-failure case.
    result = qualify_pycolmap_backend(
        extra_env={"PYTHONNOUSERSITE": "1"}, force=True,
    )
    assert result["importable"] is False


def test_qualification_is_cached_per_session():
    sentinel = _failed_qualification(error="cached sentinel")
    key = sys.executable + "|" + repr(sorted({}.items()))
    cr._PYCOLMAP_QUALIFICATION_CACHE[key] = sentinel
    assert qualify_pycolmap_backend() is sentinel  # no re-probe, no recovery


# ── selector preference ──────────────────────────────────────────────

def test_preference_cli_forces_subprocess(tmp_path):
    runner = _make_runner(tmp_path, backend_preference="cli")
    assert runner._use_pycolmap() is False


def test_preference_pycolmap_raises_when_unqualified(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "qualify_pycolmap_backend", lambda *a, **k: _failed_qualification())
    runner = _make_runner(tmp_path, backend_preference="pycolmap")
    with pytest.raises(RuntimeError, match="failed qualification"):
        runner._use_pycolmap()


def test_preference_auto_falls_back_to_cli_when_unqualified(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "qualify_pycolmap_backend", lambda *a, **k: _failed_qualification())
    runner = _make_runner(tmp_path, backend_preference="auto")
    assert runner._use_pycolmap() is False  # excluded, run proceeds via CLI


def test_preference_auto_uses_pycolmap_when_qualified(tmp_path):
    pytest.importorskip("pycolmap")
    runner = _make_runner(tmp_path, backend_preference="auto")
    assert runner._use_pycolmap() is True


def test_unknown_preference_rejected(tmp_path):
    with pytest.raises(ValueError, match="backend preference"):
        _make_runner(tmp_path, backend_preference="magic")


def test_spheresfm_always_subprocess_regardless_of_preference(tmp_path):
    runner = _make_runner(
        tmp_path, engine_name="spheresfm", camera_model="SPHERE",
        backend_preference="pycolmap",
    )
    assert runner._use_pycolmap() is False


def test_cpu_only_wheel_excluded_under_auto(tmp_path, monkeypatch):
    # Final-audit finding 1: a generic pip upgrade would install PyPI's
    # CPU-only Windows wheel — importable, but a silent GPU regression.
    # Provenance enforcement excludes it; auto falls back to the CLI.
    monkeypatch.setattr(
        cr, "qualify_pycolmap_backend",
        lambda *a, **k: _failed_qualification(importable=True, has_cuda=False, version="4.1.0"),
    )
    runner = _make_runner(tmp_path, backend_preference="auto")
    assert runner._use_pycolmap() is False


def test_cpu_only_wheel_raises_under_explicit_pycolmap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cr, "qualify_pycolmap_backend",
        lambda *a, **k: _failed_qualification(importable=True, has_cuda=False, version="4.1.0"),
    )
    runner = _make_runner(tmp_path, backend_preference="pycolmap")
    with pytest.raises(RuntimeError, match="failed qualification"):
        runner._use_pycolmap()


# ── ERP/EUCM capability refusal ──────────────────────────────────────

def test_erp_run_refuses_on_incapable_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "cli_supports_camera_model", lambda *a, **k: False)
    runner = _make_runner(
        tmp_path, camera_model="EQUIRECTANGULAR", backend_preference="cli",
    )
    with pytest.raises(RuntimeError, match="not supported by the configured binary"):
        runner.start_run(images_dir=str(tmp_path))
    assert not (tmp_path / ColmapRunner.RUNS_DIRNAME).exists()  # refused BEFORE creating a run


def test_erp_run_refuses_on_pre41_wheel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cr, "qualify_pycolmap_backend",
        lambda *a, **k: _failed_qualification(
            importable=True, has_cuda=True, version="4.0.2",
            camera_models=["PINHOLE", "OPENCV"],
        ),
    )
    runner = _make_runner(
        tmp_path, camera_model="EQUIRECTANGULAR", backend_preference="pycolmap",
    )
    with pytest.raises(RuntimeError, match="requires pycolmap >= 4.1.0"):
        runner.start_run(images_dir=str(tmp_path))


def test_erp_run_proceeds_on_capable_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "cli_supports_camera_model", lambda *a, **k: True)
    runner = _make_runner(
        tmp_path, camera_model="EQUIRECTANGULAR", backend_preference="cli",
    )
    images = tmp_path / "imgs"
    images.mkdir()
    run_dir = runner.start_run(images_dir=str(images))
    assert run_dir.is_dir()


def test_pinhole_needs_no_capability_probe(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("probe must not run for pinhole")
    monkeypatch.setattr(cr, "cli_supports_camera_model", _boom)
    runner = _make_runner(tmp_path, camera_model="PINHOLE", backend_preference="cli")
    images = tmp_path / "imgs"
    images.mkdir()
    assert runner.start_run(images_dir=str(images)).is_dir()


# ── real CLI probe (integration, gated on a colmap on PATH) ──────────

@pytest.mark.skipif(shutil.which("colmap") is None, reason="no colmap on PATH")
def test_cli_probe_accepts_pinhole_on_real_binary(tmp_path):
    assert cli_supports_camera_model(shutil.which("colmap"), "PINHOLE", tmp_path) is True

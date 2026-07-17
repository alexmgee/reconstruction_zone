"""
COLMAP / SphereSfM runner for sparse alignment work.

This module is intentionally backend-only:

- supports COLMAP-compatible CLIs via subprocess
- supports both COLMAP and SphereSfM binaries through one interface
- keeps per-run outputs isolated under a run directory
- selects the best sparse model explicitly instead of assuming sparse/0

ERP-specific projection and viewer semantics are intentionally out of scope
here. This runner focuses on process execution, output management, and model
parsing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Backend qualification (Phase D) ─────────────────────────────────
# Both backends are qualified in FRESH subprocesses before use so a broken
# wheel (crashing DLL, missing enum) can never take the GUI process down or
# silently degrade a run. Results are cached for the session — a backend
# that fails qualification stays excluded until app restart (imports cache
# in sys.modules; there is no hot-reload).

_PYCOLMAP_QUALIFICATION_CACHE: Dict[str, Dict[str, Any]] = {}
_CLI_CAMERA_MODEL_CACHE: Dict[Tuple[str, str], bool] = {}
_PYCOLMAP_PYD_SHA256_CACHE: Optional[str] = None

_PYCOLMAP_PROBE_SCRIPT = (
    "import json\n"
    "result = {'importable': False, 'version': '', 'has_cuda': False,"
    " 'caspar': False, 'camera_models': [], 'error': ''}\n"
    "try:\n"
    "    import pycolmap\n"
    "    result['importable'] = True\n"
    "    result['version'] = str(getattr(pycolmap, '__version__', ''))\n"
    "    result['has_cuda'] = bool(getattr(pycolmap, 'has_cuda', False))\n"
    "    bab = getattr(pycolmap, 'BundleAdjustmentBackend', None)\n"
    "    result['caspar'] = bool(bab is not None and hasattr(bab, 'CASPAR'))\n"
    "    cmi = getattr(pycolmap, 'CameraModelId', None)\n"
    "    if cmi is not None:\n"
    "        result['camera_models'] = ["
    "m for m in dir(cmi) if m.isupper() and m != 'INVALID']\n"
    "except Exception as exc:\n"
    "    result['error'] = repr(exc)\n"
    "print(json.dumps(result))\n"
)


def qualify_pycolmap_backend(
    python_exe: Optional[str] = None,
    *,
    extra_env: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Probe pycolmap in a fresh subprocess; cached per interpreter+env."""
    exe = python_exe or sys.executable
    cache_key = exe + "|" + repr(sorted((extra_env or {}).items()))
    if not force and cache_key in _PYCOLMAP_QUALIFICATION_CACHE:
        return _PYCOLMAP_QUALIFICATION_CACHE[cache_key]

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    result: Dict[str, Any] = {
        "importable": False, "version": "", "has_cuda": False,
        "caspar": False, "camera_models": [], "error": "",
    }
    try:
        proc = subprocess.run(
            [exe, "-c", _PYCOLMAP_PROBE_SCRIPT],
            capture_output=True, text=True, timeout=120, env=env,
        )
        line = (proc.stdout or "").strip().splitlines()
        if proc.returncode == 0 and line:
            result = json.loads(line[-1])
        else:
            result["error"] = (proc.stderr or "").strip()[-500:] or f"exit {proc.returncode}"
    except Exception as exc:
        result["error"] = repr(exc)

    logger.info(
        "pycolmap qualification (%s): importable=%s version=%s cuda=%s caspar=%s%s",
        exe, result.get("importable"), result.get("version"),
        result.get("has_cuda"), result.get("caspar"),
        f" error={result.get('error')}" if result.get("error") else "",
    )
    _PYCOLMAP_QUALIFICATION_CACHE[cache_key] = result
    return result


def cli_supports_camera_model(
    binary_path: str,
    camera_model: str,
    probe_root: str | Path,
) -> bool:
    """Probe whether a COLMAP CLI accepts a camera model; cached per session.

    Runs feature_extractor against an empty image dir: an unsupported model
    fails the ExistsCameraModelWithName check before any real work happens.
    """
    key = (str(binary_path), camera_model.upper())
    if key in _CLI_CAMERA_MODEL_CACHE:
        return _CLI_CAMERA_MODEL_CACHE[key]

    probe_dir = Path(probe_root) / "camera_model_probe"
    images_dir = probe_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    db_path = probe_dir / f"probe_{camera_model.lower()}.db"
    supported = False
    try:
        proc = subprocess.run(
            [
                str(binary_path), "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(images_dir),
                "--ImageReader.camera_model", camera_model.upper(),
            ],
            capture_output=True, text=True, timeout=120,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        supported = "ExistsCameraModelWithName" not in output
    except Exception as exc:
        logger.warning(
            "CLI camera-model probe failed for %s / %s: %s",
            binary_path, camera_model, exc,
        )
        supported = False
    finally:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass

    logger.info(
        "CLI camera-model probe: %s %s -> %s",
        Path(binary_path).name, camera_model.upper(),
        "supported" if supported else "NOT supported",
    )
    _CLI_CAMERA_MODEL_CACHE[key] = supported
    return supported


def _import_colmap_parsers():
    try:
        from colmap_validation import (
            parse_cameras_txt,
            parse_images_txt,
            parse_points3d_txt,
        )
    except ImportError:
        from reconstruction_gui.colmap_validation import (
            parse_cameras_txt,
            parse_images_txt,
            parse_points3d_txt,
        )
    return parse_cameras_txt, parse_images_txt, parse_points3d_txt


@dataclass
class BinaryValidationResult:
    """Result of validating a COLMAP-compatible executable."""

    success: bool
    resolved_binary: str
    command: List[str] = field(default_factory=list)
    output: str = ""
    error: str = ""
    detected_commands: List[str] = field(default_factory=list)
    detected_capabilities: Dict[str, bool] = field(default_factory=dict)
    binary_flavor: str = "unknown"
    version: str = ""

    @property
    def supports_sphere_workflow(self) -> bool:
        return bool(self.detected_capabilities.get("sphere_workflow"))


@dataclass
class StageResult:
    """Result from a single alignment stage."""

    stage: str
    status: str  # success, failed, cancelled
    duration_s: float
    stats: Dict[str, Any] = field(default_factory=dict)
    log: str = ""
    returncode: Optional[int] = None
    error: str = ""
    command: List[str] = field(default_factory=list)
    run_dir: str = ""
    selected_model_dir: str = ""

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


@dataclass
class _CommandResult:
    """Internal result for a low-level CLI invocation."""

    status: str
    duration_s: float
    returncode: Optional[int]
    command: List[str]
    stdout: str
    stderr: str
    log: str
    error: str = ""


class ColmapRunner:
    """Run COLMAP-compatible sparse alignment stages via subprocess."""

    MATCHER_COMMANDS = {
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
        "spatial": "spatial_matcher",
        "vocab_tree": "vocab_tree_matcher",
    }
    RUNS_DIRNAME = "runs"
    LOGS_DIRNAME = "logs"
    SPARSE_DIRNAME = "sparse"
    RUN_INFO_FILENAME = "run_info.json"
    _CORE_COMMANDS = ("feature_extractor", "mapper", "model_converter")
    _KNOWN_COMMANDS = (
        "automatic_reconstructor",
        "database_creator",
        "exhaustive_matcher",
        "feature_extractor",
        "global_mapper",
        "gui",
        "mapper",
        "model_converter",
        "patch_match_stereo",
        "pose_prior_mapper",
        "rig_configurator",
        "sequential_matcher",
        "spatial_matcher",
        "sphere_cubic_reprojecer",
        "stereo_fusion",
        "vocab_tree_matcher",
    )

    def __init__(
        self,
        binary_path: str,
        camera_model: str = "PINHOLE",
        workspace_root: Optional[str] = None,
        engine_name: Optional[str] = None,
        backend_preference: str = "auto",
    ):
        normalized_backend = (backend_preference or "auto").strip().lower()
        if normalized_backend not in ("auto", "pycolmap", "cli"):
            raise ValueError(
                f"Unknown backend preference: {backend_preference!r} "
                f"(expected auto | pycolmap | cli)"
            )
        self.backend_preference = normalized_backend
        if not binary_path or not str(binary_path).strip():
            raise ValueError("binary_path is required")
        if workspace_root is None or not str(workspace_root).strip():
            raise ValueError("workspace_root is required")

        self.binary_path = self._resolve_binary_path(binary_path)
        self.camera_model = camera_model
        self.workspace_root = Path(workspace_root).expanduser()
        self.engine_name = engine_name or self.binary_path.stem

        self.active_process: Optional[subprocess.Popen] = None
        self._binary_validation: Optional[BinaryValidationResult] = None

        self.current_run_id: Optional[str] = None
        self.current_run_dir: Optional[Path] = None
        self.current_images_dir: Optional[Path] = None
        self.current_masks_dir: Optional[Path] = None
        self.current_effective_masks_dir: Optional[Path] = None
        self.current_run_info: Dict[str, Any] = {}
        self._binary_sha256_cache: Optional[str] = None

    @staticmethod
    def _resolve_binary_path(binary_path: str) -> Path:
        candidate = Path(binary_path).expanduser()
        if candidate.exists():
            if not candidate.is_file():
                raise FileNotFoundError(f"Binary path is not a file: {candidate}")
            return candidate.resolve()

        resolved = shutil.which(binary_path)
        if resolved:
            return Path(resolved).resolve()

        raise FileNotFoundError(f"COLMAP-compatible binary not found: {binary_path}")

    @staticmethod
    def _parse_version(output: str) -> str:
        """Extract version string from COLMAP help output (e.g. '4.0.3')."""
        for line in output.splitlines()[:3]:
            match = re.match(r"COLMAP\s+([\d]+\.[\d]+[\.\d]*\S*)", line.strip())
            if match:
                return match.group(1)
        return ""

    @classmethod
    def _parse_available_commands(cls, output: str) -> List[str]:
        commands: List[str] = []
        in_commands = False

        for raw_line in output.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                continue
            if stripped == "Available commands:":
                in_commands = True
                continue
            if not in_commands:
                continue
            if not raw_line.startswith("  "):
                break

            token = stripped.split()[0]
            if token and token.replace("_", "").isalnum():
                commands.append(token)

        if commands:
            return sorted(dict.fromkeys(commands))

        normalized = output.lower()
        return [
            command
            for command in cls._KNOWN_COMMANDS
            if command in normalized
        ]

    @classmethod
    def _detect_capabilities(cls, commands: Iterable[str]) -> Dict[str, bool]:
        command_set = set(commands)
        return {
            "core_cli": all(command in command_set for command in cls._CORE_COMMANDS),
            "feature_extractor": "feature_extractor" in command_set,
            "mapper": "mapper" in command_set,
            "model_converter": "model_converter" in command_set,
            "global_mapper": "global_mapper" in command_set,
            "spatial_matcher": "spatial_matcher" in command_set,
            "vocab_tree_matcher": "vocab_tree_matcher" in command_set,
            "sphere_workflow": "sphere_cubic_reprojecer" in command_set,
        }

    def _binary_requires_sphere_support(self) -> bool:
        if self.camera_model.upper() == "SPHERE":
            return True
        return "sphere" in self.engine_name.lower()

    @staticmethod
    def _classify_binary_flavor(capabilities: Dict[str, bool]) -> str:
        if capabilities.get("sphere_workflow"):
            return "spheresfm-like"
        if capabilities.get("core_cli"):
            return "colmap-like"
        return "unknown"

    def _ensure_command_supported(self, command: str) -> None:
        validation = self._binary_validation
        if not validation or not validation.success:
            self._ensure_binary_validated()
            validation = self._binary_validation
        if not validation:
            raise RuntimeError("Binary validation state is unavailable")

        commands = set(validation.detected_commands)
        if command not in commands:
            raise RuntimeError(
                f"{self.binary_path.name} does not advertise the '{command}' command"
            )

    def _is_spheresfm_like(self) -> bool:
        validation = self._binary_validation
        if not validation:
            # No binary validated yet (e.g. parsing a model without running
            # alignment) — fall back to the explicit engine/camera-model
            # configuration so namespace and backend choices stay correct.
            return self._binary_requires_sphere_support()
        return validation.binary_flavor == "spheresfm-like"

    def _matching_option_key(self, stock_key: str, spheresfm_key: str) -> str:
        """Return the correct matcher option name for the validated binary."""
        return spheresfm_key if self._is_spheresfm_like() else stock_key

    def _mapper_option_key(self, stock_key: str, spheresfm_key: Optional[str] = None) -> str:
        """Return the correct mapper option name for the validated binary."""
        if self._is_spheresfm_like():
            return spheresfm_key or stock_key
        return stock_key

    @staticmethod
    def _format_returncode_error(command: str, returncode: Optional[int]) -> str:
        if returncode is None:
            return f"{command} failed"

        if os.name == "nt":
            normalized = int(returncode) & 0xFFFFFFFF
            if normalized == 0xC0000005:
                return (
                    f"{command} crashed with Windows access violation "
                    f"(0xC0000005 / {normalized}). This is a native crash inside the "
                    "alignment binary, not a normal reconstruction error."
                )
            if normalized == 0xC0000409:
                return (
                    f"{command} crashed with Windows stack buffer overrun "
                    f"(0xC0000409 / {normalized}). This is a native crash inside the "
                    "alignment binary."
                )

        return f"{command} failed with exit code {returncode}"

    @staticmethod
    def _subprocess_window_kwargs() -> Dict[str, Any]:
        """Hide spawned console windows on Windows while still capturing output."""
        if os.name != "nt":
            return {}

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
            "startupinfo": startupinfo,
        }

    @staticmethod
    def _extract_reconstruction_progress(log_text: str) -> Dict[str, Any]:
        registration_pattern = re.compile(r"Registering image #(\d+) \((\d+)\)")
        heading_pattern = re.compile(
            r"^(Registering image #\d+ \(\d+\)|Retriangulation|Global bundle adjustment|Finding good initial image pair)$"
        )

        cleaned_lines: List[str] = []
        for raw_line in log_text.splitlines():
            line = raw_line.strip()
            if line.startswith("[stdout]"):
                line = line[len("[stdout]") :].strip()
            elif line.startswith("[stderr]"):
                line = line[len("[stderr]") :].strip()
            cleaned_lines.append(line)

        last_registration: Optional[Dict[str, int]] = None
        last_registration_line = ""
        registration_line_index = -1
        last_heading = ""
        last_heading_after_registration = ""

        for idx, line in enumerate(cleaned_lines):
            if not line:
                continue
            if heading_pattern.match(line):
                last_heading = line
                if registration_line_index >= 0 and idx > registration_line_index:
                    last_heading_after_registration = line
            match = registration_pattern.search(line)
            if match:
                last_registration = {
                    "image_id": int(match.group(1)),
                    "registered_count": int(match.group(2)),
                }
                last_registration_line = line
                registration_line_index = idx
                last_heading_after_registration = ""

        stats: Dict[str, Any] = {}
        if last_registration:
            stats["last_registered_image_id"] = last_registration["image_id"]
            stats["last_registered_count"] = last_registration["registered_count"]
            stats["last_registration_line"] = last_registration_line
        if last_heading:
            stats["last_mapper_heading"] = last_heading
        if last_heading_after_registration:
            stats["last_phase_after_registration"] = last_heading_after_registration

        summary_parts: List[str] = []
        if last_registration:
            summary_parts.append(
                f"Reached image #{last_registration['image_id']} ({last_registration['registered_count']} registered)"
            )
        if last_heading_after_registration:
            summary_parts.append(f"last phase: {last_heading_after_registration}")
        elif last_heading and not last_registration:
            summary_parts.append(f"last phase: {last_heading}")
        if summary_parts:
            stats["progress_summary"] = "; ".join(summary_parts)

        return stats

    def validate_binary(self, timeout_s: float = 10.0) -> BinaryValidationResult:
        """Verify that the configured executable looks like a COLMAP-compatible CLI."""
        attempts: List[Tuple[List[str], str, str, Optional[int]]] = []

        for suffix in (["help"], ["-h"]):
            cmd = [str(self.binary_path), *suffix]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    **self._subprocess_window_kwargs(),
                )
            except Exception as exc:
                attempts.append((cmd, "", str(exc), None))
                continue

            output = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
            attempts.append((cmd, output, "", proc.returncode))
            commands = self._parse_available_commands(output)
            capabilities = self._detect_capabilities(commands)
            flavor = self._classify_binary_flavor(capabilities)

            if proc.returncode == 0 and capabilities["core_cli"]:
                if self._binary_requires_sphere_support() and not capabilities["sphere_workflow"]:
                    attempts.append(
                        (
                            cmd,
                            output,
                            "Configured for SphereSfM/SPHERE, but the binary does not expose sphere workflow commands.",
                            proc.returncode,
                        )
                    )
                    continue
                result = BinaryValidationResult(
                    success=True,
                    resolved_binary=str(self.binary_path),
                    command=cmd,
                    output=output,
                    detected_commands=commands,
                    detected_capabilities=capabilities,
                    binary_flavor=flavor,
                    version=self._parse_version(output),
                )
                self._binary_validation = result
                self._record_binary_validation(result)
                return result

        details = []
        specific_errors: List[str] = []
        for cmd, output, error, returncode in attempts:
            if error:
                specific_errors.append(error)
            details.append(
                {
                    "command": cmd,
                    "returncode": returncode,
                    "error": error,
                    "output_preview": output[:500],
                }
            )

        result = BinaryValidationResult(
            success=False,
            resolved_binary=str(self.binary_path),
            error=(
                specific_errors[-1]
                if specific_errors
                else (
                    "Binary validation failed. The executable did not respond like a "
                    "COLMAP-compatible CLI."
                )
            ),
            output=json.dumps(details, indent=2),
        )
        self._binary_validation = result
        self._record_binary_validation(result)
        return result

    def _ensure_binary_validated(self) -> None:
        if self._binary_validation and self._binary_validation.success:
            return
        validation = self.validate_binary()
        if not validation.success:
            raise RuntimeError(validation.error or "Binary validation failed")

    def ensure_camera_model_supported(self) -> None:
        """Refuse to start when the selected backend cannot handle the camera model.

        EQUIRECTANGULAR and EUCM exist only in COLMAP >= 4.1.0; older wheels
        and binaries reject (or worse, lack) them. Refusal names the incapable
        backend so the fix is obvious.
        """
        model = self.camera_model.upper()
        if model not in ("EQUIRECTANGULAR", "EUCM"):
            return
        if self._is_spheresfm_like():
            raise RuntimeError(
                f"Camera model {model} is not available in SphereSfM mode — "
                f"use the COLMAP engine (SPHERE is the SphereSfM native model)."
            )
        if self._use_pycolmap():
            qual = qualify_pycolmap_backend()
            if model not in qual.get("camera_models", []):
                raise RuntimeError(
                    f"Camera model {model} requires pycolmap >= 4.1.0; the "
                    f"qualified wheel is version "
                    f"'{qual.get('version') or 'unknown'}' and does not "
                    f"expose it. Upgrade the wheel or switch backend to cli "
                    f"with a COLMAP 4.1.0+ binary."
                )
        else:
            if not cli_supports_camera_model(
                str(self.binary_path), model,
                self.workspace_root / ".backend_probe",
            ):
                raise RuntimeError(
                    f"Camera model {model} is not supported by the configured "
                    f"binary {self.binary_path} — it predates COLMAP 4.1.0. "
                    f"Point the Align tab at a COLMAP 4.1.0+ build."
                )

    def start_run(self, images_dir: str, masks_dir: Optional[str] = None) -> Path:
        """Create a fresh run directory and initialize lightweight run metadata."""
        self.ensure_camera_model_supported()
        images_path = self._validate_existing_dir(images_dir, label="Images directory")
        masks_path = None
        if masks_dir:
            masks_path = self._validate_existing_dir(masks_dir, label="Masks directory")

        run_id = self._new_run_id()
        run_dir = self.workspace_root / self.RUNS_DIRNAME / run_id
        logs_dir = run_dir / self.LOGS_DIRNAME
        sparse_dir = run_dir / self.SPARSE_DIRNAME

        logs_dir.mkdir(parents=True, exist_ok=False)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_id = run_id
        self.current_run_dir = run_dir
        self.current_images_dir = images_path
        self.current_masks_dir = masks_path
        self.current_effective_masks_dir = None
        self.current_run_info = {
            "schema_version": 2,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": self.engine_name,
            "engine_name": self.engine_name,
            "camera_model": self.camera_model,
            "binary_path": str(self.binary_path),
            "binary_sha256": self._binary_sha256(),
            "backend": "",
            "pycolmap_version": self._pycolmap_version(),
            "pycolmap_pyd_sha256": self._pycolmap_pyd_sha256(),
            "camera_model_namespace": (
                "spheresfm" if self._is_spheresfm_like() else "colmap"
            ),
            "ba_backend_resolved": "" if self._is_spheresfm_like() else "CERES",
            "workspace_root": str(self.workspace_root.resolve()),
            "run_dir": str(run_dir.resolve()),
            "images_dir": str(images_path.resolve()),
            "masks_dir": str(masks_path.resolve()) if masks_path else "",
            "stages": {},
            "selected_model_dir": "",
            "adjustment_manifest_path": "",
        }
        if self._binary_validation is not None:
            self._record_binary_validation(self._binary_validation)
        self._write_run_info()
        logger.info("Started alignment run %s in %s", run_id, run_dir)
        return run_dir

    def _binary_sha256(self) -> str:
        if self._binary_sha256_cache is not None:
            return self._binary_sha256_cache
        try:
            digest = hashlib.sha256()
            with self.binary_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._binary_sha256_cache = digest.hexdigest()
        except Exception as exc:
            logger.warning("Could not hash alignment binary %s: %s", self.binary_path, exc)
            self._binary_sha256_cache = ""
        return self._binary_sha256_cache

    @staticmethod
    def _pycolmap_version() -> str:
        try:
            import pycolmap
            return str(getattr(pycolmap, "__version__", ""))
        except Exception:
            return ""

    @staticmethod
    def _pycolmap_pyd_sha256() -> str:
        """Hash of the pycolmap native module — provenance for run_info v2."""
        global _PYCOLMAP_PYD_SHA256_CACHE
        if _PYCOLMAP_PYD_SHA256_CACHE is not None:
            return _PYCOLMAP_PYD_SHA256_CACHE
        digest = ""
        try:
            from pycolmap import _core
            pyd = Path(_core.__file__)
            h = hashlib.sha256()
            with pyd.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except Exception as exc:
            logger.debug("Could not hash pycolmap _core: %s", exc)
        _PYCOLMAP_PYD_SHA256_CACHE = digest
        return digest

    def _resolve_effective_masks_dir(self) -> Tuple[Optional[Path], Dict[str, Any]]:
        if self.current_masks_dir is None:
            return None, {}
        if self.current_images_dir is None or self.current_run_dir is None:
            raise RuntimeError("Mask resolution requires an active run with images")
        if self.current_effective_masks_dir is not None:
            stats = {
                "masks_dir": str(self.current_masks_dir),
                "resolved_masks_dir": str(self.current_effective_masks_dir),
            }
            return self.current_effective_masks_dir, stats

        direct_matches = 0
        stem_matches = 0
        missing_masks = 0
        mappings: List[Tuple[Path, Path]] = []
        supported_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

        for image_path in sorted(self.current_images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in supported_exts:
                continue
            rel_path = image_path.relative_to(self.current_images_dir)
            direct_mask = self.current_masks_dir / Path(f"{rel_path}.png")
            stem_mask = self.current_masks_dir / rel_path.with_suffix(".png")

            if direct_mask.is_file():
                direct_matches += 1
                mappings.append((direct_mask, rel_path))
            elif stem_mask.is_file():
                stem_matches += 1
                mappings.append((stem_mask, rel_path))
            else:
                missing_masks += 1

        if not mappings:
            raise ValueError(
                "No masks matched the selected images. SphereSfM/COLMAP expects "
                "either '<image_name>.png' (for example 'frame.jpg.png') or a "
                "stem-only fallback that the app can remap automatically."
            )

        stats = {
            "requested_masks_dir": str(self.current_masks_dir),
            "mask_matches_direct": direct_matches,
            "mask_matches_stem": stem_matches,
            "mask_missing": missing_masks,
        }

        if stem_matches == 0:
            self.current_effective_masks_dir = self.current_masks_dir
            stats["resolved_masks_dir"] = str(self.current_masks_dir)
            stats["mask_resolution_mode"] = "native"
            return self.current_effective_masks_dir, stats

        resolved_masks_dir = self.current_run_dir / "resolved_masks"
        if resolved_masks_dir.exists():
            shutil.rmtree(resolved_masks_dir)

        for src_mask, rel_path in mappings:
            dest_mask = resolved_masks_dir / Path(f"{rel_path}.png")
            dest_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_mask, dest_mask)

        self.current_effective_masks_dir = resolved_masks_dir
        stats["resolved_masks_dir"] = str(resolved_masks_dir)
        stats["mask_resolution_mode"] = (
            "remapped-stem" if direct_matches == 0 else "mixed-remapped"
        )
        return self.current_effective_masks_dir, stats

    def _new_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{timestamp}_{suffix}"

    def _validate_existing_dir(self, raw_path: str, label: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path.resolve()

    def _write_run_info(self) -> None:
        if not self.current_run_dir:
            return
        path = self.current_run_dir / self.RUN_INFO_FILENAME
        path.write_text(json.dumps(self.current_run_info, indent=2), encoding="utf-8")

    def _record_stage_result(self, result: StageResult) -> None:
        if not self.current_run_info:
            return
        self.current_run_info.setdefault("stages", {})[result.stage] = {
            "status": result.status,
            "duration_s": round(result.duration_s, 3),
            "returncode": result.returncode,
            "error": result.error,
            "stats": result.stats,
        }
        self._write_run_info()

    def _record_selected_model(self, model_dir: Path) -> None:
        if not self.current_run_info:
            return
        self.current_run_info["selected_model_dir"] = str(model_dir.resolve())
        self._write_run_info()

    def _record_binary_validation(self, result: BinaryValidationResult) -> None:
        if not self.current_run_info:
            return
        self.current_run_info["binary_validation"] = {
            "success": result.success,
            "resolved_binary": result.resolved_binary,
            "binary_flavor": result.binary_flavor,
            "version": result.version,
            "detected_commands": result.detected_commands,
            "detected_capabilities": result.detected_capabilities,
            "error": result.error,
        }
        self._write_run_info()

    def _write_stage_log(self, stage: str, text: str) -> None:
        if not self.current_run_dir:
            return
        log_dir = self.current_run_dir / self.LOGS_DIRNAME
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{stage}.log").write_text(text or "", encoding="utf-8")

    def _ensure_active_run(
        self,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
    ) -> None:
        if self.current_run_dir is not None:
            return
        if not images_dir:
            raise RuntimeError("No active run. Call start_run() or extract_features() first.")
        self.start_run(images_dir=images_dir, masks_dir=masks_dir)

    def _build_command(self, command: str, args: Dict[str, Any]) -> List[str]:
        cmd = [str(self.binary_path), command]
        for key, value in args.items():
            if value is None:
                continue
            cmd.extend([f"--{key}", self._normalize_cli_value(value)])
        return cmd

    @staticmethod
    def _normalize_cli_value(value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, Path):
            return str(value)
        return str(value)

    def _run_cli_command(
        self,
        command: str,
        args: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        poll_interval_s: float = 0.05,
        terminate_grace_s: float = 2.0,
    ) -> _CommandResult:
        cmd = self._build_command(command, args)
        logger.info("Running command: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **self._subprocess_window_kwargs(),
        )
        self.active_process = proc

        line_queue: queue.Queue = queue.Queue()
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        combined_lines: List[str] = []

        def _reader(stream, label: str, sink: List[str]) -> None:
            try:
                for line in iter(stream.readline, ""):
                    if not line:
                        break
                    clean = line.rstrip("\r\n")
                    sink.append(clean)
                    line_queue.put((label, clean))
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        threads = [
            threading.Thread(target=_reader, args=(proc.stdout, "stdout", stdout_lines), daemon=True),
            threading.Thread(target=_reader, args=(proc.stderr, "stderr", stderr_lines), daemon=True),
        ]
        for thread in threads:
            thread.start()

        start = time.monotonic()
        cancel_requested = False
        cancel_started_at = 0.0

        try:
            while True:
                while True:
                    try:
                        label, line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    combined_lines.append(f"[{label}] {line}")
                    if progress_callback and line:
                        progress_callback(line)

                returncode = proc.poll()
                if returncode is not None:
                    break

                if cancel_event and cancel_event.is_set() and not cancel_requested:
                    cancel_requested = True
                    cancel_started_at = time.monotonic()
                    logger.info("Cancellation requested for %s", command)
                    proc.terminate()

                if cancel_requested and proc.poll() is None:
                    if (time.monotonic() - cancel_started_at) > terminate_grace_s:
                        logger.warning("Escalating to kill() for %s", command)
                        proc.kill()

                time.sleep(poll_interval_s)
        finally:
            for thread in threads:
                thread.join(timeout=0.5)
            while True:
                try:
                    label, line = line_queue.get_nowait()
                except queue.Empty:
                    break
                combined_lines.append(f"[{label}] {line}")
                if progress_callback and line:
                    progress_callback(line)
            self.active_process = None

        duration_s = time.monotonic() - start
        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)
        log_text = "\n".join(combined_lines)

        if cancel_requested:
            return _CommandResult(
                status="cancelled",
                duration_s=duration_s,
                returncode=proc.returncode,
                command=cmd,
                stdout=stdout_text,
                stderr=stderr_text,
                log=log_text,
                error=f"Cancelled during {command}",
            )
        if proc.returncode != 0:
            return _CommandResult(
                status="failed",
                duration_s=duration_s,
                returncode=proc.returncode,
                command=cmd,
                stdout=stdout_text,
                stderr=stderr_text,
                log=log_text,
                error=self._format_returncode_error(command, proc.returncode),
            )
        return _CommandResult(
            status="success",
            duration_s=duration_s,
            returncode=proc.returncode,
            command=cmd,
            stdout=stdout_text,
            stderr=stderr_text,
            log=log_text,
        )

    def extract_features(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        max_features: int = 8192,
        max_image_size: int = 0,
        feature_type: str = "SIFT",
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        stage = "feature_extraction"
        try:
            requested_images = self._validate_existing_dir(images_dir, label="Images directory")
            requested_masks = None
            if masks_dir:
                requested_masks = self._validate_existing_dir(masks_dir, label="Masks directory")

            if (
                self.current_run_dir is None
                or self.current_images_dir != requested_images
                or self.current_masks_dir != requested_masks
            ):
                self.start_run(images_dir=str(requested_images), masks_dir=str(requested_masks) if requested_masks else None)

            mask_stats: Dict[str, Any] = {}
            effective_masks_dir = None
            if self.current_masks_dir is not None:
                effective_masks_dir, mask_stats = self._resolve_effective_masks_dir()

            start = time.monotonic()

            if self._use_pycolmap():
                self._extract_features_pycolmap(
                    feature_type=feature_type,
                    max_features=max_features,
                    max_image_size=max_image_size,
                    effective_masks_dir=effective_masks_dir,
                    extra_args=extra_args,
                    progress_callback=progress_callback,
                )
                log_text = "pycolmap.extract_features completed"
            else:
                self._ensure_binary_validated()
                if self.camera_model.upper() == "SPHERE":
                    validation = self._binary_validation
                    if not validation or not validation.supports_sphere_workflow:
                        raise RuntimeError(
                            "SPHERE camera model requires a SphereSfM-like binary "
                            "that exposes sphere workflow commands."
                        )
                args: Dict[str, Any] = {
                    "database_path": self.current_run_dir / "database.db",
                    "image_path": self.current_images_dir,
                    "ImageReader.camera_model": self.camera_model,
                }
                normalized_feature_type = (feature_type or "").strip().upper()
                is_aliked = normalized_feature_type.startswith("ALIKED")
                if is_aliked:
                    args["FeatureExtraction.type"] = normalized_feature_type
                    args["AlikedExtraction.max_num_features"] = max_features
                else:
                    args["SiftExtraction.max_num_features"] = max_features
                    if normalized_feature_type and normalized_feature_type != "SIFT":
                        args["FeatureExtraction.type"] = feature_type
                if effective_masks_dir is not None:
                    args["ImageReader.mask_path"] = effective_masks_dir
                if max_image_size > 0:
                    max_image_size_key = (
                        "SiftExtraction.max_image_size"
                        if self._is_spheresfm_like()
                        else "FeatureExtraction.max_image_size"
                    )
                    args[max_image_size_key] = max_image_size
                if extra_args:
                    args.update({k: v for k, v in extra_args.items() if v is not None})

                cmd_result = self._run_cli_command(
                    "feature_extractor",
                    args,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
                if cmd_result.status != "success":
                    raise RuntimeError(cmd_result.error or f"feature_extractor failed (exit {cmd_result.returncode})")
                log_text = cmd_result.log

            duration_s = time.monotonic() - start
            stats = {
                "camera_model": self.camera_model,
                "feature_type": feature_type,
                "max_features": max_features,
                "max_image_size": max_image_size,
                "images_dir": str(self.current_images_dir),
                "masks_dir": str(self.current_masks_dir) if self.current_masks_dir else "",
                "backend": "pycolmap" if self._use_pycolmap() else "subprocess",
            }
            stats.update(mask_stats)
            result = StageResult(
                stage=stage,
                status="success",
                duration_s=duration_s,
                stats=stats,
                log=log_text,
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )
        except Exception as exc:
            result = StageResult(
                stage=stage,
                status="failed",
                duration_s=0.0,
                error=str(exc),
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )

        self._write_stage_log(stage, result.log or result.error)
        self._record_stage_result(result)
        return result

    def _extract_features_pycolmap(
        self,
        feature_type: str,
        max_features: int,
        max_image_size: int,
        effective_masks_dir: Optional[Path],
        extra_args: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[str], None]],
    ) -> None:
        import pycolmap

        db_path = str(self.current_run_dir / "database.db")
        image_dir = str(self.current_images_dir)

        reader_options = pycolmap.ImageReaderOptions(
            camera_model=self.camera_model,
        )
        if effective_masks_dir is not None:
            reader_options.mask_path = str(effective_masks_dir)

        # Camera params from extra_args
        if extra_args:
            if extra_args.get("ImageReader.single_camera"):
                pass  # handled via camera_mode below
            if extra_args.get("ImageReader.camera_params"):
                reader_options.camera_params = str(extra_args["ImageReader.camera_params"])

        camera_mode = pycolmap.CameraMode.AUTO
        if extra_args and extra_args.get("ImageReader.single_camera"):
            camera_mode = pycolmap.CameraMode.SINGLE
        if extra_args and extra_args.get("ImageReader.single_camera_per_folder"):
            camera_mode = pycolmap.CameraMode.PER_FOLDER

        normalized_type = (feature_type or "").strip().upper()

        if progress_callback:
            progress_callback(f"Extracting {normalized_type} features via pycolmap...")

        kwargs: Dict[str, Any] = {
            "camera_mode": camera_mode,
            "reader_options": reader_options,
        }

        opts = pycolmap.FeatureExtractionOptions()
        if max_image_size > 0:
            opts.max_image_size = max_image_size
        if max_features > 0:
            opts.sift.max_num_features = max_features

        if normalized_type.startswith("ALIKED"):
            extractor_name = (
                "ALIKED_N32" if normalized_type == "ALIKED_N32" else "ALIKED_N16ROT"
            )
            if (
                hasattr(opts, "type")
                and hasattr(pycolmap, "FeatureExtractorType")
                and hasattr(pycolmap.FeatureExtractorType, extractor_name)
            ):
                opts.type = getattr(pycolmap.FeatureExtractorType, extractor_name)
                if max_features > 0 and hasattr(opts, "aliked"):
                    if hasattr(opts.aliked, "max_num_features"):
                        opts.aliked.max_num_features = max_features
                    else:
                        logger.info("ALIKED feature extraction has no feature-cap option")
                        if progress_callback:
                            progress_callback(
                                "ALIKED feature extraction has no feature-cap option"
                            )
            else:
                logger.warning(
                    "pycolmap %s lacks ALIKED extraction options; falling back to SIFT",
                    getattr(pycolmap, "__version__", "unknown"),
                )
        # SIFT is the default — no extra kwarg needed

        kwargs["extraction_options"] = opts

        pycolmap.extract_features(db_path, image_dir, **kwargs)

        if progress_callback:
            progress_callback("Feature extraction complete")

    def apply_rig_config(
        self,
        rig_config_path: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> StageResult:
        """Apply a rig config to the database via COLMAP's rig_configurator CLI."""
        stage = "rig_configuration"
        try:
            self._ensure_binary_validated()
            self._ensure_active_run()
            self._ensure_command_supported("rig_configurator")

            src_path = Path(rig_config_path)
            if not src_path.is_file():
                raise FileNotFoundError(f"Rig config not found: {rig_config_path}")

            run_rig_path = self.current_run_dir / "rig_config.json"
            shutil.copy2(src_path, run_rig_path)

            args: Dict[str, Any] = {
                "database_path": self.current_run_dir / "database.db",
                "rig_config_path": run_rig_path,
            }

            cmd_result = self._run_cli_command(
                "rig_configurator",
                args,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            stats = {
                "rig_config_source": str(src_path),
                "rig_config_run_copy": str(run_rig_path),
            }
            result = StageResult(
                stage=stage,
                status=cmd_result.status,
                duration_s=cmd_result.duration_s,
                stats=stats,
                log=cmd_result.log,
                returncode=cmd_result.returncode,
                error=cmd_result.error,
                command=cmd_result.command,
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )
        except Exception as exc:
            result = StageResult(
                stage=stage,
                status="failed",
                duration_s=0.0,
                error=str(exc),
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )

        self._write_stage_log(stage, result.log or result.error)
        self._record_stage_result(result)
        return result

    def match_features(
        self,
        strategy: str = "exhaustive",
        guided: bool = False,
        max_num_matches: int = 32768,
        vocab_tree_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        stage = "matching"
        try:
            self._ensure_active_run()

            if strategy not in self.MATCHER_COMMANDS:
                raise ValueError(f"Unknown matching strategy: {strategy}")
            if strategy == "vocab_tree" and not vocab_tree_path:
                raise ValueError("vocab_tree strategy requires vocab_tree_path to be set")

            start = time.monotonic()

            if self._use_pycolmap():
                self._match_features_pycolmap(
                    strategy=strategy,
                    guided=guided,
                    max_num_matches=max_num_matches,
                    vocab_tree_path=vocab_tree_path,
                    extra_args=extra_args,
                    progress_callback=progress_callback,
                )
                log_text = f"pycolmap.match_{strategy} completed"
            else:
                self._ensure_binary_validated()
                command = self.MATCHER_COMMANDS[strategy]
                self._ensure_command_supported(command)
                args: Dict[str, Any] = {
                    "database_path": self.current_run_dir / "database.db",
                    self._matching_option_key(
                        "FeatureMatching.max_num_matches",
                        "SiftMatching.max_num_matches",
                    ): max_num_matches,
                }
                if guided:
                    args[
                        self._matching_option_key(
                            "FeatureMatching.guided_matching",
                            "SiftMatching.guided_matching",
                        )
                    ] = True
                if strategy == "vocab_tree":
                    args["VocabTreeMatching.vocab_tree_path"] = vocab_tree_path
                if extra_args:
                    args.update({k: v for k, v in extra_args.items() if v is not None})

                cmd_result = self._run_cli_command(
                    command,
                    args,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
                if cmd_result.status != "success":
                    raise RuntimeError(cmd_result.error or f"{command} failed (exit {cmd_result.returncode})")
                log_text = cmd_result.log

            duration_s = time.monotonic() - start
            stats = {
                "strategy": strategy,
                "guided": guided,
                "max_num_matches": max_num_matches,
                "vocab_tree_path": vocab_tree_path or "",
                "backend": "pycolmap" if self._use_pycolmap() else "subprocess",
            }
            result = StageResult(
                stage=stage,
                status="success",
                duration_s=duration_s,
                stats=stats,
                log=log_text,
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )
        except Exception as exc:
            result = StageResult(
                stage=stage,
                status="failed",
                duration_s=0.0,
                error=str(exc),
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )

        self._write_stage_log(stage, result.log or result.error)
        self._record_stage_result(result)
        return result

    def _match_features_pycolmap(
        self,
        strategy: str,
        guided: bool,
        max_num_matches: int,
        vocab_tree_path: Optional[str],
        extra_args: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[str], None]],
    ) -> None:
        import pycolmap

        db_path = str(self.current_run_dir / "database.db")

        matching_options = pycolmap.FeatureMatchingOptions(
            max_num_matches=max_num_matches,
            guided_matching=guided,
        )

        # Set matching type from extra_args (e.g. SIFT_LIGHTGLUE, ALIKED_LIGHTGLUE)
        if extra_args:
            matching_type = extra_args.get("FeatureMatching.type")
            if matching_type:
                matching_options.type = getattr(
                    pycolmap.FeatureMatcherType, matching_type, matching_options.type
                )

        if progress_callback:
            progress_callback(f"Matching features ({strategy}) via pycolmap...")

        if strategy == "exhaustive":
            pycolmap.match_exhaustive(db_path, matching_options=matching_options)
        elif strategy == "sequential":
            pycolmap.match_sequential(db_path, matching_options=matching_options)
        elif strategy == "spatial":
            pycolmap.match_spatial(db_path, matching_options=matching_options)
        elif strategy == "vocab_tree":
            pycolmap.match_vocabtree(db_path, matching_options=matching_options)

        if progress_callback:
            progress_callback("Feature matching complete")

    def _reconstruct_pycolmap(
        self,
        mapper_key: str,
        sparse_root: Path,
        min_num_inliers: int,
        extra_args: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[str], None]],
        ba_backend: str = "AUTO",
        ba_use_gpu: bool = True,
        ba_gpu_index: int = -1,
    ) -> StageResult:
        """Run reconstruction via pycolmap (incremental or global)."""
        import pycolmap

        stage = "reconstruction"
        start = time.monotonic()

        db_path = str(self.current_run_dir / "database.db")
        image_dir = str(self.current_images_dir)

        # Deterministic seed
        seed = 0
        if extra_args and extra_args.get("Mapper.random_seed") is not None:
            seed = int(extra_args["Mapper.random_seed"])
        pycolmap.set_random_seed(seed)

        if progress_callback:
            mode_label = "global" if mapper_key in ("global", "global_mapper") else "incremental"
            progress_callback(f"Running {mode_label} mapping via pycolmap...")

        caspar_surface = (
            hasattr(pycolmap.BundleAdjustmentBackend, "CASPAR")
            and hasattr(pycolmap.IncrementalPipelineOptions(), "ba_local_backend")
            and hasattr(pycolmap.IncrementalPipelineOptions(), "ba_global_backend")
        )
        missing_reason = ""
        if not caspar_surface:
            missing_reason = f"pycolmap {pycolmap.__version__} lacks CASPAR"
        resolved_backend = self._resolve_ba_backend(
            ba_backend=ba_backend,
            mapper_key=mapper_key,
            caspar_surface_exists=caspar_surface,
            missing_surface_reason=missing_reason,
            progress_callback=progress_callback,
        )
        self._record_reconstruct_provenance("pycolmap", resolved_backend)

        if mapper_key in ("global", "global_mapper"):
            recs = pycolmap.global_mapping(db_path, image_dir, str(sparse_root))
        else:
            opts = pycolmap.IncrementalPipelineOptions(
                min_num_matches=min_num_inliers,
            )
            opts.ba_use_gpu = ba_use_gpu
            opts.ba_gpu_index = str(ba_gpu_index)
            if resolved_backend == "CASPAR":
                # Hybrid measured 2026-07-17 on 430-image garage set (identical
                # quality, mapping wall): both-Caspar 341s · Ceres-GPU 209s ·
                # local=Ceres/global=Caspar 159s. Caspar's GPU launch overhead
                # loses on the many small local BAs and wins on large global BAs.
                opts.ba_local_backend = pycolmap.BundleAdjustmentBackend.CERES
                opts.ba_global_backend = pycolmap.BundleAdjustmentBackend.CASPAR
            if extra_args:
                for attr, key in [
                    ("ba_refine_focal_length", "Mapper.ba_refine_focal_length"),
                    ("ba_refine_principal_point", "Mapper.ba_refine_principal_point"),
                    ("ba_refine_extra_params", "Mapper.ba_refine_extra_params"),
                ]:
                    if key in extra_args:
                        setattr(opts, attr, bool(extra_args[key]))
            recs = pycolmap.incremental_mapping(db_path, image_dir, str(sparse_root), opts)

        duration_s = time.monotonic() - start

        if not recs:
            return StageResult(
                stage=stage, status="failed", duration_s=duration_s,
                error="Reconstruction produced no models",
                run_dir=str(self.current_run_dir),
            )

        # Find best reconstruction
        best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
        best_rec = recs[best_idx]
        model_dir = sparse_root / str(best_idx)

        # Write model to disk (binary format — pycolmap model reader handles it)
        model_dir.mkdir(parents=True, exist_ok=True)
        best_rec.write(str(model_dir))
        self._record_selected_model(model_dir)

        # Build stats from the Reconstruction object directly
        num_registered = best_rec.num_reg_images()
        num_points = len(best_rec.points3D)
        track_lengths = [pt.track.length() for pt in best_rec.points3D.values()]
        mean_error = best_rec.compute_mean_reprojection_error() if num_points > 0 else 0.0
        mean_track = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
        if track_lengths:
            sorted_tracks = sorted(track_lengths)
            mid = len(sorted_tracks) // 2
            median_track = float(
                sorted_tracks[mid] if len(sorted_tracks) % 2
                else (sorted_tracks[mid - 1] + sorted_tracks[mid]) / 2.0
            )
        else:
            median_track = 0.0

        selected_stats = {
            "num_registered": num_registered,
            "num_points": num_points,
            "mean_reproj_error": mean_error,
            "mean_track_length": mean_track,
            "median_track_length": median_track,
        }

        mapper_label = "global_mapper" if mapper_key in ("global", "global_mapper") else "mapper"
        stats: Dict[str, Any] = {
            "mapper": mapper_label,
            "num_models": len(recs),
            "selected_model_dir": str(model_dir),
            "selected_model_stats": selected_stats,
            "backend": "pycolmap",
        }

        log_text = best_rec.summary()
        if progress_callback:
            progress_callback(f"Reconstruction complete: {num_registered} images, {num_points} points")

        return StageResult(
            stage=stage,
            status="success",
            duration_s=duration_s,
            stats=stats,
            log=log_text,
            run_dir=str(self.current_run_dir),
            selected_model_dir=str(model_dir),
        )

    def _camera_model_ids(self) -> set[int]:
        database_path = self.current_run_dir / "database.db"
        if not database_path.is_file():
            return set()
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute("SELECT DISTINCT model FROM cameras").fetchall()
        return {int(row[0]) for row in rows}

    def _resolve_ba_backend(
        self,
        ba_backend: str,
        mapper_key: str,
        caspar_surface_exists: bool,
        missing_surface_reason: str = "",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        requested = (ba_backend or "AUTO").strip().upper()
        if requested not in {"AUTO", "CERES", "CASPAR"}:
            raise ValueError(f"Unknown BA backend: {ba_backend}")

        camera_models = self._camera_model_ids()
        observed = ", ".join(str(model) for model in sorted(camera_models)) or "none"
        logger.info("Observed camera model IDs for BA backend selection: {%s}", observed)
        if progress_callback:
            progress_callback(f"Observed camera model IDs: {{{observed}}}")

        is_incremental = mapper_key in ("incremental", "mapper")
        is_spheresfm = self._is_spheresfm_like()
        offending = sorted(camera_models - {1, 2})

        if requested == "CASPAR":
            if offending:
                raise RuntimeError(
                    f"CASPAR does not support camera model IDs {offending}; "
                    "only PINHOLE (1) and SIMPLE_RADIAL (2) are eligible"
                )
            if not is_incremental:
                raise RuntimeError(f"CASPAR requires incremental mapping, not {mapper_key}")
            if is_spheresfm:
                raise RuntimeError("CASPAR is unavailable for SphereSfM mode")
            if not caspar_surface_exists:
                raise RuntimeError(missing_surface_reason or "CASPAR backend surface is unavailable")
            resolved = "CASPAR"
        elif requested == "AUTO":
            resolved = (
                "CASPAR"
                if not offending
                and is_incremental
                and not is_spheresfm
                and caspar_surface_exists
                else "CERES"
            )
            if resolved == "CERES" and missing_surface_reason and not caspar_surface_exists:
                logger.info("AUTO selected CERES: %s", missing_surface_reason)
                if progress_callback:
                    progress_callback(f"AUTO selected CERES: {missing_surface_reason}")
        else:
            resolved = "CERES"

        logger.info("Selected BA backend: %s", resolved)
        if progress_callback:
            progress_callback(f"Selected BA backend: {resolved}")
        return resolved

    def _record_reconstruct_provenance(self, backend: str, ba_backend: str) -> None:
        if not self.current_run_info:
            return
        self.current_run_info["backend"] = backend
        self.current_run_info["ba_backend_resolved"] = (
            "" if self._is_spheresfm_like() else ba_backend
        )
        self._write_run_info()

    def reconstruct(
        self,
        mapper: str = "incremental",
        min_num_inliers: int = 15,
        images_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        ba_backend: str = "AUTO",
        ba_use_gpu: bool = True,
        ba_gpu_index: int = -1,
    ) -> StageResult:
        stage = "reconstruction"
        try:
            self._ensure_active_run(images_dir=images_dir)

            if images_dir:
                self.current_images_dir = self._validate_existing_dir(
                    images_dir, label="Images directory"
                )

            sparse_root = self.current_run_dir / self.SPARSE_DIRNAME
            sparse_root.mkdir(parents=True, exist_ok=True)

            mapper_key = mapper.lower().strip()

            # pycolmap path for stock COLMAP (not SphereSfM, not pose_prior)
            if self._use_pycolmap() and mapper_key not in ("pose_prior", "pose_prior_mapper"):
                result = self._reconstruct_pycolmap(
                    mapper_key=mapper_key,
                    sparse_root=sparse_root,
                    min_num_inliers=min_num_inliers,
                    extra_args=extra_args,
                    progress_callback=progress_callback,
                    ba_backend=ba_backend,
                    ba_use_gpu=ba_use_gpu,
                    ba_gpu_index=ba_gpu_index,
                )
                self._write_stage_log(stage, result.log or result.error)
                self._record_stage_result(result)
                return result

            # Subprocess path (SphereSfM + pose_prior + fallback)
            if not self._use_pycolmap():
                self._ensure_binary_validated()

            if mapper_key in ("incremental", "mapper"):
                command = "mapper"
            elif mapper_key in ("global", "global_mapper"):
                command = "global_mapper"
            elif mapper_key in ("pose_prior", "pose_prior_mapper"):
                command = "pose_prior_mapper"
            else:
                raise ValueError(f"Unknown mapper mode: {mapper}")
            self._ensure_command_supported(command)

            resolved_backend = self._resolve_ba_backend(
                ba_backend=ba_backend,
                mapper_key=mapper_key,
                caspar_surface_exists=not self._is_spheresfm_like(),
                progress_callback=progress_callback,
            )
            self._record_reconstruct_provenance("subprocess", resolved_backend)

            args: Dict[str, Any] = {
                "database_path": self.current_run_dir / "database.db",
                "image_path": self.current_images_dir,
                "output_path": sparse_root,
            }
            if command in ("mapper", "pose_prior_mapper"):
                args[self._mapper_option_key("Mapper.min_num_matches")] = min_num_inliers
            if command == "mapper" and not self._is_spheresfm_like():
                args["Mapper.ba_use_gpu"] = ba_use_gpu
                if ba_gpu_index >= 0:
                    args["Mapper.ba_gpu_index"] = ba_gpu_index
                if resolved_backend == "CASPAR":
                    # Hybrid (see _reconstruct_pycolmap): local stays CERES.
                    args["Mapper.ba_local_backend"] = "CERES"
                    args["Mapper.ba_global_backend"] = "CASPAR"
            if extra_args:
                args.update({k: v for k, v in extra_args.items() if v is not None})

            snapshot_freq_raw = args.get("Mapper.snapshot_images_freq")
            try:
                snapshot_freq = int(snapshot_freq_raw) if snapshot_freq_raw is not None else 0
            except Exception:
                snapshot_freq = 0
            snapshot_path = args.get("Mapper.snapshot_path")
            if snapshot_path:
                Path(snapshot_path).mkdir(parents=True, exist_ok=True)
            if snapshot_freq > 0 and not snapshot_path:
                snapshot_path = self.current_run_dir / "snapshots"
                Path(snapshot_path).mkdir(parents=True, exist_ok=True)
                args["Mapper.snapshot_path"] = snapshot_path

            cmd_result = self._run_cli_command(
                command,
                args,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            progress_stats = self._extract_reconstruction_progress(cmd_result.log)
            snapshot_stats = {
                "snapshot_images_freq": snapshot_freq,
                "snapshot_path": str(Path(snapshot_path).resolve()) if snapshot_path else "",
            }

            if cmd_result.status != "success":
                stats: Dict[str, Any] = {"mapper": command}
                stats.update(snapshot_stats)
                stats.update(progress_stats)
                selected_model_dir = ""
                if cmd_result.status == "failed":
                    try:
                        resolved_snapshot_path = (
                            Path(snapshot_path) if snapshot_path else None
                        )
                        model_dirs, recovery_info = self._recover_model_dirs(
                            sparse_root=sparse_root,
                            snapshot_path=resolved_snapshot_path,
                        )
                        stats.update(recovery_info)
                        if model_dirs:
                            for model_dir in model_dirs:
                                self._convert_model_to_text(model_dir)
                            selected_model, selected_stats = self.pick_best_model(model_dirs=model_dirs)
                            recovered_from_snapshot = (
                                resolved_snapshot_path is not None
                                and resolved_snapshot_path in selected_model.parents
                            )
                            stats.update(
                                {
                                    "num_models": len(model_dirs),
                                    "selected_model_dir": str(selected_model),
                                    "selected_model_stats": selected_stats,
                                    "partial_model_available": True,
                                    "recovered_model_source": (
                                        "snapshot" if recovered_from_snapshot else "sparse"
                                    ),
                                    "recovered_from_snapshot": recovered_from_snapshot,
                                }
                            )
                            selected_model_dir = str(selected_model)
                            self._record_selected_model(selected_model)
                    except Exception as exc:
                        stats["partial_model_parse_error"] = str(exc)
                result = StageResult(
                    stage=stage,
                    status=cmd_result.status,
                    duration_s=cmd_result.duration_s,
                    stats=stats,
                    log=cmd_result.log,
                    returncode=cmd_result.returncode,
                    error=cmd_result.error,
                    command=cmd_result.command,
                    run_dir=str(self.current_run_dir),
                    selected_model_dir=selected_model_dir,
                )
            else:
                model_dirs = self._list_model_dirs(sparse_root)
                if not model_dirs:
                    raise RuntimeError(f"No sparse models found under {sparse_root}")

                for model_dir in model_dirs:
                    self._convert_model_to_text(model_dir)

                selected_model, selected_stats = self.pick_best_model(model_dirs=model_dirs)
                self._record_selected_model(selected_model)
                stats = {
                    "mapper": command,
                    "snapshot_images_freq": snapshot_freq,
                    "snapshot_path": str(Path(snapshot_path).resolve()) if snapshot_path else "",
                    **progress_stats,
                    "num_models": len(model_dirs),
                    "selected_model_dir": str(selected_model),
                    "selected_model_stats": selected_stats,
                }
                result = StageResult(
                    stage=stage,
                    status="success",
                    duration_s=cmd_result.duration_s,
                    stats=stats,
                    log=cmd_result.log,
                    returncode=cmd_result.returncode,
                    command=cmd_result.command,
                    run_dir=str(self.current_run_dir),
                    selected_model_dir=str(selected_model),
                )
        except Exception as exc:
            result = StageResult(
                stage=stage,
                status="failed",
                duration_s=0.0,
                error=str(exc),
                run_dir=str(self.current_run_dir) if self.current_run_dir else "",
            )

        self._write_stage_log(stage, result.log or result.error)
        self._record_stage_result(result)
        return result

    def _is_model_dir(self, model_dir: Path) -> bool:
        required_stems = ("cameras", "images", "points3D")
        return all(
            (model_dir / f"{stem}.bin").exists() or (model_dir / f"{stem}.txt").exists()
            for stem in required_stems
        )

    def _list_model_dirs(self, sparse_root: Path) -> List[Path]:
        if not sparse_root.exists():
            return []

        candidates: List[Path] = []
        if self._is_model_dir(sparse_root):
            candidates.append(sparse_root)
        for child in sorted(sparse_root.iterdir()):
            if child.is_dir() and self._is_model_dir(child):
                candidates.append(child)
        return candidates

    def _recover_model_dirs(
        self,
        sparse_root: Path,
        snapshot_path: Optional[Path] = None,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        candidates: List[Path] = []
        recovery_info: Dict[str, Any] = {
            "sparse_model_count": 0,
            "snapshot_model_count": 0,
        }

        sparse_dirs = self._list_model_dirs(sparse_root)
        candidates.extend(sparse_dirs)
        recovery_info["sparse_model_count"] = len(sparse_dirs)

        if snapshot_path and Path(snapshot_path).exists():
            snapshot_root = Path(snapshot_path)
            snapshot_dirs = [
                child
                for child in sorted(snapshot_root.iterdir())
                if child.is_dir() and self._is_model_dir(child)
            ]
            recovery_info["snapshot_model_count"] = len(snapshot_dirs)
            for snapshot_dir in snapshot_dirs:
                if snapshot_dir not in candidates:
                    candidates.append(snapshot_dir)

        return candidates, recovery_info

    def _convert_model_to_text(self, model_dir: Path) -> None:
        txt_files = [model_dir / f"{stem}.txt" for stem in ("cameras", "images", "points3D")]
        if all(path.exists() for path in txt_files):
            return

        bin_files = [model_dir / f"{stem}.bin" for stem in ("cameras", "images", "points3D")]
        if not all(path.exists() for path in bin_files):
            raise FileNotFoundError(
                f"Model directory does not contain a complete COLMAP model: {model_dir}"
            )

        cmd_result = self._run_cli_command(
            "model_converter",
            {
                "input_path": model_dir,
                "output_path": model_dir,
                "output_type": "TXT",
            },
        )
        if cmd_result.status != "success":
            raise RuntimeError(
                f"model_converter failed for {model_dir}: {cmd_result.error or cmd_result.log}"
            )

    def _model_stats_from_parsed(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        images = parsed["images"]
        points3d = parsed["points3D"]

        num_registered = len(images)
        num_points = len(points3d)
        mean_error = (
            float(sum(float(p.error) for p in points3d.values()) / num_points)
            if num_points > 0
            else 0.0
        )
        track_lengths = [len(p.track) for p in points3d.values()]
        mean_track = (
            float(sum(track_lengths) / len(track_lengths))
            if track_lengths
            else 0.0
        )
        if track_lengths:
            sorted_tracks = sorted(track_lengths)
            mid = len(sorted_tracks) // 2
            median_track = float(
                sorted_tracks[mid]
                if len(sorted_tracks) % 2
                else (sorted_tracks[mid - 1] + sorted_tracks[mid]) / 2.0
            )
        else:
            median_track = 0.0
        return {
            "num_registered": num_registered,
            "num_points": num_points,
            "mean_reproj_error": mean_error,
            "mean_track_length": mean_track,
            "median_track_length": median_track,
        }

    def pick_best_model(
        self,
        model_dirs: Optional[Iterable[Path]] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Choose the strongest sparse model explicitly instead of assuming sparse/0."""
        if model_dirs is None:
            self._ensure_active_run()
            model_dirs = self._list_model_dirs(self.current_run_dir / self.SPARSE_DIRNAME)

        scored: List[Tuple[Tuple[int, int, float, str], Path, Dict[str, Any]]] = []
        errors: List[str] = []
        for model_dir in model_dirs:
            try:
                parsed = self.parse_model(model_dir=model_dir)
                stats = parsed["stats"]
                key = (
                    int(stats["num_registered"]),
                    int(stats["num_points"]),
                    -float(stats["mean_reproj_error"]),
                    str(model_dir),
                )
                scored.append((key, Path(model_dir), stats))
            except Exception as exc:
                errors.append(f"{model_dir}: {exc}")

        if not scored:
            detail = "; ".join(errors) if errors else "No parseable sparse models found"
            raise RuntimeError(detail)

        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_dir, best_stats = scored[0]
        return best_dir, best_stats

    @staticmethod
    def _pycolmap_available() -> bool:
        # Subprocess qualification FIRST: a wheel whose import crashes must
        # never be imported into this (GUI) process.
        qual = qualify_pycolmap_backend()
        if not qual.get("importable"):
            return False
        # Provenance enforcement (final-audit finding 1): a generic
        # `pip install -U pycolmap` silently replaces the custom CUDA wheel
        # with PyPI's CPU-only Windows build. A CPU wheel is functional but a
        # silent regression on this machine — exclude it and fall back to the
        # CUDA CLI instead. Reinstall from ~/COLMAP/release-v4.1.0/ (by hash).
        if not qual.get("has_cuda"):
            logger.warning(
                "pycolmap wheel (version %s) is CPU-only — excluded; using the "
                "CLI backend. Restore the custom CUDA wheel from the archived "
                "release artifact (see ACQUISITION-RECORD).",
                qual.get("version") or "unknown",
            )
            return False
        try:
            import pycolmap  # noqa: F401
            return True
        except ImportError:
            return False

    def _use_pycolmap(self) -> bool:
        """Select the execution backend for stock COLMAP work.

        SphereSfM is always subprocess. For stock COLMAP the explicit
        preference wins: "cli" forces subprocess, "pycolmap" requires a
        qualified wheel (raises if unavailable — no silent downgrade),
        "auto" uses pycolmap when qualified, else the CLI.
        """
        if self._is_spheresfm_like():
            return False
        if self.backend_preference == "cli":
            return False
        available = self._pycolmap_available()
        if self.backend_preference == "pycolmap":
            if not available:
                qual = qualify_pycolmap_backend()
                raise RuntimeError(
                    "Backend preference is 'pycolmap' but the pycolmap wheel "
                    "failed qualification for this session"
                    + (f": {qual.get('error')}" if qual.get("error") else "")
                    + ". Fix the wheel and restart, or switch backend to auto/cli."
                )
            return True
        return available

    def parse_model(self, model_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Parse a COLMAP model and compute summary stats.

        Uses pycolmap if available (reads binary models directly, no text
        conversion needed). Falls back to text parsers otherwise.
        """
        if model_dir is None:
            self._ensure_active_run()
            selected = self.current_run_info.get("selected_model_dir", "") if self.current_run_info else ""
            if selected:
                model_dir = Path(selected)
            else:
                best_dir, _ = self.pick_best_model()
                model_dir = best_dir

        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Try pycolmap first — reads binary models directly. Gated on
        # _use_pycolmap() (not just availability): SphereSfM models use the
        # fork camera-model namespace, which pycolmap would misinterpret.
        if self._use_pycolmap():
            try:
                parsed = self._parse_model_pycolmap(model_path)
                parsed["stats"] = self._model_stats_from_parsed(parsed)
                return parsed
            except Exception as exc:
                logger.warning("pycolmap model read failed, falling back to text: %s", exc)

        # Fallback: convert to text and parse
        return self._parse_model_text(model_path)

    def _parse_model_pycolmap(self, model_path: Path) -> Dict[str, Any]:
        """Parse a model using pycolmap (binary or text format)."""
        import pycolmap
        try:
            from colmap_validation import from_pycolmap_reconstruction
        except ImportError:
            from reconstruction_gui.colmap_validation import from_pycolmap_reconstruction

        rec = pycolmap.Reconstruction()
        rec.read(str(model_path))
        parsed = from_pycolmap_reconstruction(rec)
        parsed["model_dir"] = str(model_path.resolve())
        return parsed

    def _parse_model_binary(self, model_path: Path) -> Dict[str, Any]:
        """Parse a binary model with the native reader (no external tools)."""
        try:
            from colmap_binary_validation_adapter import read_colmap_binary_full_for_validation
        except ImportError:
            from reconstruction_gui.colmap_binary_validation_adapter import (
                read_colmap_binary_full_for_validation,
            )

        variant = "spheresfm" if self._is_spheresfm_like() else "colmap"
        parsed = read_colmap_binary_full_for_validation(model_path, variant=variant)
        parsed["model_dir"] = str(model_path.resolve())
        parsed["camera_model_namespace"] = variant
        return parsed

    def _parse_model_text(self, model_path: Path) -> Dict[str, Any]:
        """Parse a model using text-format parsers (legacy path)."""
        parse_cameras_txt, parse_images_txt, parse_points3d_txt = _import_colmap_parsers()

        self._convert_model_to_text(model_path)

        cameras_path = model_path / "cameras.txt"
        images_path = model_path / "images.txt"
        points_path = model_path / "points3D.txt"
        for path in (cameras_path, images_path, points_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing COLMAP text file: {path}")

        cameras = parse_cameras_txt(cameras_path)
        images = parse_images_txt(images_path)
        points3d = parse_points3d_txt(points_path)
        parsed = {
            "model_dir": str(model_path.resolve()),
            "cameras": cameras,
            "images": images,
            "points3D": points3d,
        }
        parsed["stats"] = self._model_stats_from_parsed(parsed)
        return parsed

    def run_all(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        max_features: int = 8192,
        max_image_size: int = 0,
        feature_type: str = "SIFT",
        strategy: str = "exhaustive",
        guided: bool = False,
        max_num_matches: int = 32768,
        vocab_tree_path: Optional[str] = None,
        mapper: str = "incremental",
        min_num_inliers: int = 15,
        ba_backend: str = "AUTO",
        ba_use_gpu: bool = True,
        ba_gpu_index: int = -1,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        extract_extra_args: Optional[Dict[str, Any]] = None,
        match_extra_args: Optional[Dict[str, Any]] = None,
        reconstruct_extra_args: Optional[Dict[str, Any]] = None,
    ) -> List[StageResult]:
        """Run extract -> match -> reconstruct in a fresh run directory."""
        self.start_run(images_dir=images_dir, masks_dir=masks_dir)
        results: List[StageResult] = []

        if progress_callback:
            progress_callback("=== Stage 1/3: Feature Extraction ===")
        r1 = self.extract_features(
            images_dir=images_dir,
            masks_dir=masks_dir,
            max_features=max_features,
            max_image_size=max_image_size,
            feature_type=feature_type,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            extra_args=extract_extra_args,
        )
        results.append(r1)
        if not r1.success:
            return results

        if cancel_event and cancel_event.is_set():
            return results

        if progress_callback:
            progress_callback("=== Stage 2/3: Feature Matching ===")
        r2 = self.match_features(
            strategy=strategy,
            guided=guided,
            max_num_matches=max_num_matches,
            vocab_tree_path=vocab_tree_path,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            extra_args=match_extra_args,
        )
        results.append(r2)
        if not r2.success:
            return results

        if cancel_event and cancel_event.is_set():
            return results

        if progress_callback:
            progress_callback("=== Stage 3/3: Sparse Reconstruction ===")
        r3 = self.reconstruct(
            mapper=mapper,
            min_num_inliers=min_num_inliers,
            images_dir=images_dir,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            extra_args=reconstruct_extra_args,
            ba_backend=ba_backend,
            ba_use_gpu=ba_use_gpu,
            ba_gpu_index=ba_gpu_index,
        )
        results.append(r3)
        return results

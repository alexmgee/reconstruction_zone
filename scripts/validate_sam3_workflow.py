"""Validate SAM3 token/access/weight workflow in an isolated HF home.

The script never prints token values. It can reuse the currently saved
HuggingFace token to simulate the setup wizard's "paste token and verify" path
inside a disposable HF_HOME.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = f"sam3-workflow-{time.strftime('%Y%m%d-%H%M%S')}"


def marker(key: str, value: object = "") -> None:
    if value == "":
        print(key, flush=True)
    else:
        print(f"{key} {value}", flush=True)


def safe_reset(path: Path) -> None:
    base = (ROOT / "dist_test" / "new-install-sandbox").resolve()
    resolved = path.resolve()
    if base not in resolved.parents and resolved != base:
        raise RuntimeError(f"Refusing to reset outside sandbox: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def report(label: str, obj: object) -> None:
    marker(label, obj)


def read_saved_token_without_importing_hf() -> str | None:
    """Read the currently saved HF token without importing huggingface_hub."""
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token.strip() or None

    homes: list[Path] = []
    if os.environ.get("HF_HOME"):
        homes.append(Path(os.environ["HF_HOME"]))
    homes.append(Path.home() / ".cache" / "huggingface")

    for home in homes:
        token_path = home / "token"
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if token:
            return token
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default=DEFAULT_CASE)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--use-saved-token", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    saved_token = None
    if args.use_saved_token:
        saved_token = read_saved_token_without_importing_hf()
        marker("SAVED_TOKEN_PRESENT", bool(saved_token))
        if not saved_token:
            marker("SAM3_WORKFLOW_FAILED", "no saved token available")
            return 1

    case_root = ROOT / "dist_test" / "new-install-sandbox" / args.case_name
    app_home = case_root / "app-home"
    model_dir = app_home / "models"
    hf_home = case_root / "hf-home"

    if args.reset:
        safe_reset(case_root)

    model_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["RECONSTRUCTION_ZONE_APP_HOME"] = str(app_home)
    os.environ["RECONSTRUCTION_ZONE_MODEL_DIR"] = str(model_dir)
    os.environ["RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"] = "1"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reconstruction_gui"))

    marker("CASE_ROOT", case_root)
    marker("APP_HOME", app_home)
    marker("MODEL_DIR", model_dir)
    marker("HF_HOME", hf_home)

    from reconstruction_gui.sam3_setup import (
        check_sam3_setup,
        download_model_weights,
        verify_hf_token_detailed,
    )

    report("INITIAL_SAM3_SETUP", check_sam3_setup())

    if saved_token:
        verified = verify_hf_token_detailed(saved_token)
        report("VERIFY_SAVED_TOKEN_REPORT", verified)
        after_verify = check_sam3_setup()
        report("AFTER_VERIFY_SAM3_SETUP", after_verify)
        if verified.overall_stage not in {"needs_weights", "ready", "ready_to_install"}:
            marker("SAM3_WORKFLOW_FAILED", "token verification did not reach an install/download state")
            return 1

    if args.download:
        marker("SAM3_DOWNLOAD_START")

        def progress(message: str) -> None:
            marker("SAM3_PROGRESS", message)

        ok = download_model_weights(on_progress=progress)
        marker("SAM3_DOWNLOAD_RESULT", ok)
        if not ok:
            marker("SAM3_WORKFLOW_FAILED", "download_model_weights returned false")
            return 1

    final = check_sam3_setup()
    report("FINAL_SAM3_SETUP", final)

    if args.download and final.overall_stage != "ready":
        marker("SAM3_WORKFLOW_FAILED", "download completed but setup is not ready")
        return 1

    marker("SAM3_WORKFLOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate real model download workflows in an isolated app-home.

This script exercises the same RF-DETR and YOLO26 download helpers used by the
first-launch setup wizard. SAM3 is gated and large, so this script reports its
setup state by default and does not download SAM3 unless explicitly extended.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = f"real-model-downloads-{time.strftime('%Y%m%d-%H%M%S')}"


def safe_reset(path: Path) -> None:
    base = (ROOT / "dist_test" / "new-install-sandbox").resolve()
    resolved = path.resolve()
    if base not in resolved.parents and resolved != base:
        raise RuntimeError(f"Refusing to reset outside sandbox: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def print_marker(key: str, value: object = "") -> None:
    if value == "":
        print(key, flush=True)
    else:
        print(f"{key} {value}", flush=True)


def drain_queue(q: queue.Queue) -> None:
    while not q.empty():
        msg = q.get_nowait()
        print_marker("QUEUE", repr(msg))


def describe_file(path: Path) -> None:
    if path.exists():
        print_marker("FILE", f"{path} {path.stat().st_size}")
    else:
        print_marker("FILE_MISSING", path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default=DEFAULT_CASE)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-rfdetr", action="store_true")
    parser.add_argument("--skip-yolo26", action="store_true")
    parser.add_argument("--sam3-status-only", action="store_true", default=True)
    args = parser.parse_args()

    case_root = ROOT / "dist_test" / "new-install-sandbox" / args.case_name
    app_home = case_root / "app-home"
    model_dir = app_home / "models"
    hf_home = case_root / "hf-home"
    transformers_cache = hf_home / "transformers"

    if args.reset:
        safe_reset(case_root)

    model_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ["RECONSTRUCTION_ZONE_APP_HOME"] = str(app_home)
    os.environ["RECONSTRUCTION_ZONE_MODEL_DIR"] = str(model_dir)
    os.environ["RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"] = "1"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reconstruction_gui"))

    print_marker("CASE_ROOT", case_root)
    print_marker("APP_HOME", app_home)
    print_marker("MODEL_DIR", model_dir)
    print_marker("STRICT_MODEL_DIRS", os.environ["RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS"])
    print_marker("HF_HOME", hf_home)

    from reconstruction_gui.model_paths import (
        resolve_rfdetr_seg_weights,
        resolve_sam3_weights,
        resolve_yolo26_weights,
    )
    from reconstruction_gui.setup_wizard import (
        _download_rfdetr,
        _download_yolo26,
        check_all_models,
    )
    from reconstruction_gui.sam3_setup import check_sam3_setup

    initial = {m.key: m.ready for m in check_all_models()}
    print_marker("INITIAL_MODEL_READY", initial)
    print_marker("INITIAL_SAM3_SETUP", check_sam3_setup())

    q: queue.Queue = queue.Queue()

    if not args.skip_rfdetr:
        print_marker("RFDETR_DOWNLOAD_START")
        _download_rfdetr(q)
        drain_queue(q)
        rfdetr_path = model_dir / "rf-detr-seg-small.pt"
        describe_file(rfdetr_path)
        print_marker("RFDETR_RESOLVED", resolve_rfdetr_seg_weights("small"))
        print_marker("RFDETR_DOWNLOAD_OK")

    if not args.skip_yolo26:
        print_marker("YOLO26_DOWNLOAD_START")
        _download_yolo26(q)
        drain_queue(q)
        yolo_path = model_dir / "yolo26n-seg.pt"
        describe_file(yolo_path)
        print_marker("YOLO26_RESOLVED", resolve_yolo26_weights("n"))
        print_marker("YOLO26_DOWNLOAD_OK")

    final = {m.key: m.ready for m in check_all_models()}
    print_marker("FINAL_MODEL_READY", final)
    print_marker("FINAL_SAM3_RESOLVED", resolve_sam3_weights())
    print_marker("FINAL_SAM3_SETUP", check_sam3_setup())

    expected_rfdetr = args.skip_rfdetr or final.get("rfdetr") is True
    expected_yolo = args.skip_yolo26 or final.get("yolo26") is True
    if not expected_rfdetr or not expected_yolo:
        print_marker("MODEL_DOWNLOAD_WORKFLOW_FAILED")
        return 1

    print_marker("MODEL_DOWNLOAD_WORKFLOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

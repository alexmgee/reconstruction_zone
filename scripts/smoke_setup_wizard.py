"""Source-level setup wizard smoke for new-install validation.

This intentionally avoids model downloads. It only verifies that the wizard can
be constructed, made modal, and run briefly inside a redirected app-home.
"""
from __future__ import annotations

import os
import sys
import traceback
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_HOME = ROOT / "dist_test" / "new-install-sandbox" / "source-setup-wizard-smoke" / "app-home"
HF_HOME = ROOT / "dist_test" / "new-install-sandbox" / "source-setup-wizard-smoke" / "hf-home"

os.environ["RECONSTRUCTION_ZONE_APP_HOME"] = str(APP_HOME)
os.environ["RECONSTRUCTION_ZONE_MODEL_DIR"] = str(APP_HOME / "models")
os.environ["HF_HOME"] = str(HF_HOME)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reconstruction_gui"))


def marker(text: str) -> None:
    print(text, flush=True)


def run_direct() -> int:
    import customtkinter as ctk
    from reconstruction_gui.setup_wizard import ModelInfo, SetupWizard

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.withdraw()
    marker("SETUP_WIZARD_ROOT_OK")

    models = [
        ModelInfo("SAM3", "sam3", "3.3 GB", gated=True, ready=False),
        ModelInfo("RF-DETR-Seg", "rfdetr", "129 MB", ready=False),
        ModelInfo("YOLO26-n-seg", "yolo26", "6.5 MB", ready=False),
    ]

    wizard = SetupWizard(root, models)
    marker("SETUP_WIZARD_CONSTRUCTED")

    wizard.transient(root)
    wizard.grab_set()
    wizard.focus()
    marker("SETUP_WIZARD_MODAL_APPLIED")

    def close() -> None:
        try:
            wizard.grab_release()
        except Exception:
            pass
        try:
            wizard.destroy()
        except Exception:
            pass
        root.destroy()

    root.after(1200, close)
    root.mainloop()
    marker("SETUP_WIZARD_MAINLOOP_OK")
    return 0


def run_if_needed() -> int:
    import customtkinter as ctk
    from reconstruction_gui.setup_wizard import run_setup_wizard_if_needed

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.withdraw()
    marker("SETUP_WIZARD_API_ROOT_OK")

    result = run_setup_wizard_if_needed(root)
    marker(f"SETUP_WIZARD_IF_NEEDED_RETURNED {result}")

    def close_api() -> None:
        for child in root.winfo_children():
            try:
                child.grab_release()
            except Exception:
                pass
            try:
                child.destroy()
            except Exception:
                pass
        root.destroy()

    root.after(1200, close_api)
    root.mainloop()
    marker("SETUP_WIZARD_IF_NEEDED_MAINLOOP_OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "if-needed"), default="direct")
    args = parser.parse_args()

    marker(f"APP_HOME {os.environ['RECONSTRUCTION_ZONE_APP_HOME']}")
    marker(f"MODEL_DIR {os.environ['RECONSTRUCTION_ZONE_MODEL_DIR']}")
    marker(f"HF_HOME {os.environ['HF_HOME']}")

    try:
        import customtkinter  # noqa: F401
        import reconstruction_gui.setup_wizard  # noqa: F401
        marker("SETUP_WIZARD_IMPORT_OK")
        if args.mode == "direct":
            return run_direct()
        return run_if_needed()
    except Exception:
        traceback.print_exc()
        marker("SETUP_WIZARD_SMOKE_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Load ERP rig helpers into the Metashape Python console.

DO NOT paste exec(...) lines from chat into the Metashape console — they often
contain hidden U+2029 characters and fail with SyntaxError.

Use: Tools > Run Script > select THIS file.
Then in the console: verify_alignment_status(), apply_erp_rig(...), etc.
"""
from __future__ import annotations

import importlib.util
import sys

_SCRIPT = r"D:\Projects\reconstruction-zone\scripts\metashape_apply_erp_rig.py"
_VERSION = "2026-07-02-verify-v2"
_EXPORTS = (
    "apply_erp_rig",
    "verify_alignment_status",
    "verify_aligned_rig",
    "verify_erp_rig",
    "enable_single_rig_pose",
    "enable_all_rig_poses",
)

# Drop cached module so Run Script always picks up file edits.
if "erp_rig" in sys.modules:
    del sys.modules["erp_rig"]

spec = importlib.util.spec_from_file_location("erp_rig", _SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

if not hasattr(mod, "_tie_point_count"):
    raise RuntimeError(f"Stale script at {_SCRIPT} — missing _tie_point_count")

main = sys.modules.get("__main__")
for name in _EXPORTS:
    fn = getattr(mod, name)
    setattr(main, name, fn)
    globals()[name] = fn

print(f"ERP rig tools loaded ({_VERSION}).")
print("  verify_alignment_status()  — or Tools > Run Script > metashape_verify_alignment.py")
print("  apply_erp_rig(r'D:\\path\\to\\reframe_metadata.json')")

# Fail loudly if console still has an old pasted copy.
src = getattr(verify_alignment_status, "__code__", None)
if src and "point_count" in (src.co_consts or ()):
    print("WARNING: stale verify_alignment_status still in memory — restart Metashape or use metashape_verify_alignment.py")

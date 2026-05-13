# Build Issues Found in Phase 3 Nuitka Build (2026-05-11)

**Build:** Full GUI Nuitka build with YAML singledispatch fix + console=attach
**Build time:** 2h 33m (02:13 → 04:46)
**Exe location:** `dist_nuitka/reconstruction_zone.dist/ReconstructionZone.exe`
**Tested:** 2026-05-11 ~10:53

---

## Issue 1: cv2 Recursive Loading in Nuitka Bundle

**Error:**
```
ERROR: recursion is detected during loading of "cv2" binary extensions. Check OpenCV installation.
```

**When it happens:** During the setup wizard's model download step. The old download functions instantiated the full model objects (`rfdetr.RFDETRSegSmall()` and `YOLO("yolo26n-seg")`), which import cv2. Inside the Nuitka bundle, cv2 has a recursive loading problem.

**Root cause:** The Nuitka dist contains two copies of `cv2.pyd`:
- `dist_nuitka/reconstruction_zone.dist/cv2.pyd` (root level)
- `dist_nuitka/reconstruction_zone.dist/cv2/cv2.pyd` (inside cv2 directory)

When cv2 loads, it finds itself in the directory structure and recurses.

**Impact:** RF-DETR and YOLO26 downloads fail. SAM3 is unaffected (uses `huggingface_hub`, not cv2).

**Fix applied in source (`setup_wizard.py`):**

The download functions were rewritten to avoid importing cv2 entirely:

**Before (old code, broken):**
```python
def _download_rfdetr(q):
    import rfdetr                          # imports cv2 transitively
    model = rfdetr.RFDETRSegSmall()        # instantiates model, loads weights
    del model

def _download_yolo26(q):
    from ultralytics import YOLO           # imports cv2 transitively
    model = YOLO("yolo26n-seg")            # instantiates model, loads weights
    del model
```

**After (new code, fix):**
```python
def _download_rfdetr(q):
    # Direct URL download for seg weights — no cv2 needed
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "rf-detr-seg-small.pt"
    )
    # DINOv2 backbone via torch.hub — also no cv2
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/nicolo-savioli/romav2/resolve/main/romav2.pt",
        model_dir=str(torch_cache),
        map_location="cpu",
    )

def _download_yolo26(q):
    # HuggingFace hub download — no cv2 needed
    from huggingface_hub import hf_hub_download
    hf_hub_download("openvision/yolo26-n-seg", "model.pt")
```

**Verification status:** Source-level fix only. Not yet tested in a Nuitka build. The cv2 recursion issue may also affect the main masking pipeline at runtime (when models are actually loaded for inference, not just downloaded). That is a separate concern from the wizard.

**Open question:** Does cv2 load correctly during normal app operation (masking, extraction) in the current Nuitka build? The crash happened before we could test because the wizard error cascaded. The cv2 recursion may be limited to the download-time import path, or it may be a general Nuitka+OpenCV issue that affects all cv2 usage.

---

## Issue 2: Wizard Shows "Setup Complete!" Despite Download Failures

**Error:** No error shown to user — the wizard said "Setup complete!" and showed the "Start Using Reconstruction Zone" button even though both RF-DETR and YOLO26 failed to download.

**When it happens:** After the download worker encounters exceptions for one or more models.

**Root cause:** The download worker always sends `("all_done",)` at the end, regardless of whether any downloads failed:

```python
# Old code (broken):
def _download_worker(self):
    for i, m in enumerate(self._to_download):
        try:
            fn(self._queue)
            self._queue.put(("model_done", m.key))
        except Exception as e:
            self._queue.put(("model_error", m.key, str(e)))
    self._queue.put(("all_done",))  # <-- always sent, even after failures
```

**Impact:** User thinks setup succeeded, launches the app, then masking fails with confusing errors because models aren't actually downloaded.

**Fix applied in source (`setup_wizard.py`):**

```python
# New code (fix):
def _download_worker(self):
    failures = []
    for i, m in enumerate(self._to_download):
        try:
            fn(self._queue)
            self._queue.put(("model_done", m.key))
        except Exception as e:
            self._queue.put(("model_error", m.key, str(e)))
            failures.append(m.name)
    if failures:
        self._queue.put(("all_done_with_errors", failures))
    else:
        self._queue.put(("all_done",))
```

The handler for `all_done_with_errors` shows which models failed and tells the user they'll retry on first use:

```python
elif msg_type == "all_done_with_errors":
    failures = msg[1]
    self._dl_label.configure(
        text=f"{len(failures)} model(s) failed to download",
        text_color="#EF4444")
    self._dl_status.configure(
        text=f"Failed: {', '.join(failures)}. The app will still launch — "
             f"these models will retry on first use.",
        text_color="#F59E0B")
    self.protocol("WM_DELETE_WINDOW", self.destroy)
```

**Verification status:** Source-level fix only. Not yet tested in a build.

---

## Issue 3: App Crash After Wizard Closes — "'ReconstructionZone' object is not callable"

**Error:**
```
TypeError: 'ReconstructionZone' object is not callable
```

**Traceback:**
```
tkinter/__init__.py line 2081, in __call__
tkinter/__init__.py line 1787, in _substitute
tkinter/__init__.py line 1685, in nametowidget
TypeError: 'ReconstructionZone' object is not callable
```

**When it happens:** After the user clicks "Start Using Reconstruction Zone" in the wizard and the wizard closes. The main app's `mainloop()` then crashes.

**Root cause (hypothesis):** The wizard (a `CTkToplevel`) was destroyed while the main loop was running. The destruction left a stale reference or callback in tkinter's event system. When the main loop tries to process the next event, it calls `nametowidget()` which encounters the `ReconstructionZone` class object instead of a valid widget path string.

This may be caused by:
- The wizard's `wait_window()` interaction with `after()` callbacks
- A Nuitka-specific issue with how compiled CTkToplevel destruction propagates
- The error cascade from Issue 1 (cv2 failures) leaving the wizard in an inconsistent state

**Why it doesn't reproduce from source:** From source, all models are cached, so the wizard never appears. The crash only happens when the wizard shows, fails, and closes.

**Fix approach:** The crash is likely a symptom of Issues 1 and 2. If the downloads succeed (Issue 1 fixed) and the error state is handled properly (Issue 2 fixed), the wizard should close cleanly. If the crash persists after fixing Issues 1 and 2, we'll need to investigate the tkinter/CTkToplevel destruction further.

**Verification status:** Not directly fixed — expected to be resolved by fixing Issues 1 and 2. Needs testing in the next build.

---

## Issue 4: Model Check Functions Return Wrong Results in Packaged App

**Symptom:** Wizard shows RF-DETR and YOLO26 as "missing" (red dots) even though SAM3 is correctly detected as "ready" (green check).

**Root cause for RF-DETR:** `_rfdetr_ready()` checked for `rf-detr-seg-small.pt` in the current working directory. In the packaged app, CWD is `dist_nuitka/reconstruction_zone.dist/`, not the project root where the file was previously downloaded.

**Root cause for YOLO26:** `_yolo26_ready()` only checked the HuggingFace cache, but ultralytics downloads to CWD (`yolo26n-seg.pt`), not the HF cache.

**Fix applied in source (`setup_wizard.py`):**

Both check functions now search multiple locations:

```python
def _rfdetr_ready():
    # Check torch hub cache for DINOv2 backbone
    # Check HF cache for rfdetr
    # Check CWD as fallback

def _yolo26_ready():
    # Check CWD (yolo26n-seg.pt) — ultralytics default
    # Check ultralytics cache (~/.cache/ultralytics/)
    # Check HF cache (our wizard downloads here)
```

**Verified from source:**
```
SAM3: True
RF-DETR: True
YOLO26: True
```

**Verification status:** Passes from source. Not yet tested in a Nuitka build.

---

## Summary: What Needs Rebuilding

All four issues require a new Nuitka build to test. The source fixes are:

| File | Changes |
|---|---|
| `reconstruction_gui/setup_wizard.py` | New download functions (no cv2), error handling for failures, fixed check functions |

The fixes do NOT require changes to the Nuitka YAML config, the spec file, or any other build infrastructure. The singledispatch fix from the previous build is still valid and will carry forward.

**Estimated rebuild time:** ~2.5 hours based on previous builds.

**Open risks for the next build:**
1. cv2 recursion may affect masking inference, not just downloads — needs testing
2. The `nametowidget` crash may persist if it's a Nuitka+CTkToplevel issue, not just an error cascade
3. RF-DETR's DINOv2 backbone download URL (`huggingface.co/nicolo-savioli/romav2`) needs verification — this URL was found in the torch hub cache, not from rfdetr's source code
4. YOLO26's HF hub download (`openvision/yolo26-n-seg`) produces a file that ultralytics may not find at runtime — ultralytics expects `yolo26n-seg.pt` in CWD, not in the HF cache

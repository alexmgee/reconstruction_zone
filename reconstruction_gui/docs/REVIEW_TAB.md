# Review Tab

The Review tab is where you inspect, approve, reject, and hand-edit masks after the Mask tab finishes processing. It provides a thumbnail grid for scanning results at a glance, a large preview panel for detailed inspection, and a subprocess-based OpenCV editor for precise brush/lasso/flood-fill corrections. Review status is persisted to disk so you can close the app and resume later.

## Sections

The left panel is organized into five vertically stacked sections:

| Section | Purpose |
|---------|---------|
| **Data Source** | Point to your masks and source images, then load them |
| **Filter & Sort** | Control which masks appear and in what order |
| **Thumbnails** | 2-column scrollable grid of overlay previews |
| **Current Mask** | Info line + action buttons for the selected mask |
| **Batch** | Bulk actions and progress summary |

The right panel is the **shared Preview panel** (also used by the Mask tab). It shows a large overlay of the selected mask on its source image, with zoom, navigator, and optional mask-only view.

---

## Data Source

### Masks and Images

- **Masks** — Directory containing mask PNG files. Each mask is a binary image: black pixels = regions to remove, white pixels = regions to keep.
- **Images** — Directory containing the original source images that were masked.

The two directories are matched by filename stem — `frame_0001.png` in masks matches `frame_0001.jpg` in images.

### Buttons

| Button | Purpose |
|--------|---------|
| **Load Masks** | Scans both directories, discovers matching pairs, loads thumbnails, and populates the grid |
| **Auto-detect from Output** | Reads the Output path from the Mask tab and automatically finds the masks and images directories. Looks for `output/masks/` and common image locations (`output/images/`, `output/frames/`, parent directory) |

### Pair discovery

Two strategies are tried in order:

1. **Prefixed masks** — looks for files named `mask_*.png` or `*_mask.png`, strips the prefix/suffix to find the stem, then matches against the images directory.
2. **Same-stem matching** — if no prefixed files are found, matches masks and images by identical filename stem (e.g. `frame_0001.png` in masks ↔ `frame_0001.jpg` in images).

---

## Filter & Sort

### Sort

| Option | Behavior |
|--------|----------|
| **Filename** (default) | Alphabetical by stem name |
| **Confidence** | By detection confidence score, ascending (lowest first — worst masks surface first) |
| **Quality** | By quality tier: reject → poor → review → good → excellent |
| **Area %** | By mask area as percentage of image, descending (largest masks first) |

### Filter

| Option | Shows |
|--------|-------|
| **All** (default) | Every loaded mask |
| **Needs Review** | Only masks with quality = "review" or "poor" |
| **Poor** | Only masks with quality = "poor" |
| **Unreviewed** | Only masks with status = "pending" (not yet accepted/rejected/skipped) |

Changing either control immediately rebuilds the thumbnail grid and resets the navigator.

---

## Thumbnails

A 2-column scrollable grid showing overlay previews of each mask on its source image.

### What each cell shows

- **Overlay thumbnail** — Source image with a red tint on masked (black) regions, resized to fit the cell
- **Filename** — Stem name, truncated to 20 characters
- **Status line** — Quality tier, confidence percentage, and review status (e.g. `"excellent  87%  accepted"`)

### Border colors

Borders are color-coded by status first, quality second:

**By status** (takes priority):
| Status | Color |
|--------|-------|
| Edited | Blue (#3b82f6) |
| Accepted | Green (#22c55e) |
| Rejected | Red (#ef4444) |
| Skipped | Gray (#6b7280) |

**By quality** (fallback if status is "pending"):
| Quality | Color |
|---------|-------|
| Excellent | Green (#22c55e) |
| Good | Yellow (#eab308) |
| Review | Orange (#f97316) |
| Poor | Red (#ef4444) |
| Reject | Dark red (#991b1b) |

### Interaction

- **Click a thumbnail** to jump the navigator to that mask and show it in the preview panel.
- The selected thumbnail gets a blue highlight border.

### Loading behavior

- First 40 thumbnails are loaded immediately.
- Remaining thumbnails load in background batches of 10.
- A progress label shows `"X/Y loaded..."` until complete.
- Mask area percentages are calculated in a separate background thread (needed for Area % sorting).

---

## Current Mask

Shows details and action buttons for whichever mask is currently selected in the navigator.

### Info line

Multi-line text showing:
```
frame_0001
Quality: excellent | Conf: 87% | Area: 45.2% | Status: accepted
```

### Action buttons

| Button | Color | What it does |
|--------|-------|-------------|
| **Edit (OpenCV)** | Blue | Opens the mask in the OpenCV editor for brush/lasso/flood-fill corrections |
| **Accept** | Green | Marks the mask as accepted and advances to the next pending mask |
| **Reject** | Red | Marks the mask as rejected and advances to the next pending mask |
| **Skip** | Gray | Marks the mask as skipped and advances to the next pending mask |

All actions are recorded with timestamps and persisted to `review_status.json` in the masks directory.

---

## Batch

### Accept All Good

Bulk-accepts every mask that is currently "pending" AND has quality of "good" or "excellent". Saves you from clicking Accept on dozens of high-quality masks one by one.

### Hide Done

Removes all accepted and edited masks from the current thumbnail grid. Unlike the "Unreviewed" filter, this keeps rejected and skipped masks visible so you can reconsider them. Stacks with whatever filter is already active — if you're viewing "Needs Review" and click Hide Done, it removes any you've already dealt with from that subset.

### Summary label

Shows a running count: `"12 done | 3 rejected | 8 pending"`. Updated after every action.

---

## Preview Panel (right side)

The preview panel is shared between the Mask and Review tabs. When the Review tab is active, it shows:

### Overlay view

The source image with red-tinted masked regions overlaid. Label reads `"Overlay (red = masked)"`.

### Mask view (optional)

Toggle the **Mask** checkbox in the preview header to show the raw mask below the overlay. Off by default.

### Zoom

Slider from 25% to 300%, default 65%. Adjusting zoom re-renders the cached image at the new scale. Zoom level is preserved when navigating between masks.

### Navigator

| Control | Purpose |
|---------|---------|
| **< button** | Previous mask |
| **Slider** | Jump to any mask by index |
| **> button** | Next mask |
| **Counter** | Shows `"X / Y"` (current index / total) |

The navigator operates on the filtered+sorted list, so if you're filtered to "Needs Review" with 15 results, the navigator range is 1–15.

### Preview button

Runs a single-image preview of the masking pipeline on the currently displayed image (from the Mask tab's settings). Useful for testing settings before a full batch run.

---

## OpenCV Mask Editor

Launched via the **Edit (OpenCV)** button. Opens in a separate subprocess so it doesn't block the GUI. The editor communicates back via temporary files — when you save, the GUI detects the save, invalidates the thumbnail cache, and pushes the next mask to the editor. Saving distinguishes between modified and unmodified masks: if you painted changes, the mask is marked "edited" (blue border); if you viewed it and saved without changes, it's marked "accepted" (green border).

### Drawing tools

**Brush** — The primary tool. Click and drag to paint.
- Left click = add to mask (mark for removal)
- Right click = remove from mask (restore)
- Brush size: 1–100 px, adjust with `+` / `-` keys

**Flood fill** — Shift + click to fill connected regions of similar color.
- Shift + left click = add region to mask
- Shift + right click = remove region from mask
- Tolerance: 5–100, adjust with `[` / `]` keys
- Two fill modes (toggle with `b`):
  - **Color mode** — LAB color space flood fill with edge detection
  - **Brightness mode** — Relative darkness (good for shadows on pavement)

**Lasso** — Ctrl + drag to draw a freeform selection polygon.
- Ctrl + left drag = add enclosed area to mask
- Ctrl + right drag = remove enclosed area from mask
- Uses Otsu thresholding inside the lasso for intelligent shadow detection
- Morphological opening removes noise (e.g. grass texture)

### View modes

Press `v` to cycle through:
| Mode | Shows |
|------|-------|
| **Overlay** (default) | Original image + red mask tint + green contour |
| **Original** | Source image only |
| **Mask-only** | White foreground on black background |

Press `m` to toggle between overlay and original. Press `a` for A/B flicker mode (alternates overlay/original every ~500ms).

### Zoom and pan

- **Scroll wheel** — Zoom in/out (1.0x to 8.0x), centered on cursor
- **Middle mouse drag** — Pan the view

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `s` | Save mask and advance to next image |
| `n` | Skip to next image without saving |
| `q` | Quit editor |
| `u` | Undo last change |
| `r` | Reset to original mask |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |
| `[` | Decrease flood tolerance |
| `]` | Increase flood tolerance |
| `v` | Cycle view mode (overlay → original → mask) |
| `m` | Toggle overlay ↔ original |
| `a` | Toggle A/B flicker comparison |
| `b` | Toggle fill mode (color ↔ brightness) |
| `0` | Reset zoom to 100% |
| `h` | Toggle help overlay |

### Visual feedback

- **Cursor indicator** — Green circle (add mode) or red circle (remove mode), sized to match brush
- **Contour highlight** — Yellow contour near cursor for edge-aware editing
- **Lasso path** — Cyan line (add) or orange line (remove) drawn while dragging
- **Status bar** (bottom) — Shows brush size, flood tolerance, zoom, fill mode, view mode, and mask coverage %
- **SAVED! flash** — Green text appears for 1.5 seconds after saving

### Mask convention

- **On disk**: Black (0) = foreground to remove, White (255) = keep
- **In editor**: The overlay shows red tint on regions that will be removed
- **On save**: Mask is written back in the same convention (black = remove)

---

## Status Management

### Review status file

`review_status.json` is saved alongside the mask files in the masks directory. It tracks:

| Field | Purpose |
|-------|---------|
| **status** | pending, accepted, edited, rejected, or skipped |
| **quality** | excellent, good, review, poor, or reject (set by the masking pipeline based on review threshold) |
| **confidence** | Detection confidence score (0.0–1.0) |
| **area_percent** | Masked area as percentage of total image pixels |
| **edited_at** | Timestamp of last edit |
| **action_history** | Timestamped log of all actions taken on this mask |

The file is created automatically on first load and updated after every action. It survives app restarts — you can close Masking Studio and resume reviewing later.

### Quality tiers

Quality is assigned by the masking pipeline based on the **Review threshold** slider on the Mask tab (default 0.85). Masks scoring below the threshold are flagged for review. The quality score is based on mask area, edge quality, and detection confidence.

---

## Typical workflow

```
1. Run masking on the Mask tab
2. Switch to Review tab
3. Click "Auto-detect from Output" (or manually set Masks + Images paths)
4. Click "Load Masks"
5. Filter → "Needs Review" to see flagged masks first
6. For each mask:
   - Glance at the overlay in the preview panel
   - Accept if good, Reject if bad, Edit if fixable
7. Click "Accept All Good" to bulk-approve the rest
8. Check summary: "X done | Y rejected | Z pending"
```

## Typical workflow: hand-editing a mask

```
1. Navigate to a mask that needs correction
2. Click "Edit (OpenCV)" — editor opens in a new window
3. Use brush to paint missing mask regions (left click = add)
4. Use flood fill (Shift+click) for large shadow areas
5. Use lasso (Ctrl+drag) for irregular regions
6. Press 's' to save — editor auto-advances to next mask
7. GUI updates: thumbnail refreshes, status changes to "edited" (if changed) or "accepted" (if unchanged)
```

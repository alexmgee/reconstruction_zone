# Panoex GUI Enhancement - Implementation Handoff Document

## Overview

This document covers the implementation of significant GUI enhancements to `panoex_gui.py`, adding collapsible sections, an extraction planning system, LUT preview capabilities, a standalone segmentation window, and integrated post-processing in the extraction workflow.

**Files Modified:**
- `panoex_gui.py` - Main GUI application

**Date:** 2026-02-05

---

## 1. New Helper Classes

### 1.1 PlannedFrame Dataclass (Line 31-37)

```python
@dataclass
class PlannedFrame:
    """Represents a frame planned for extraction."""
    index: int
    frame_number: int
    timestamp: float
    reason: str  # "interval", "scene", "manual"
    included: bool = True
```

**Purpose:** Stores metadata for each frame in the extraction plan. Used by the Plan tab to display and manage which frames will be extracted.

**Fields:**
- `index`: Sequential index in the plan
- `frame_number`: Actual video frame number
- `timestamp`: Time position in seconds
- `reason`: How the frame was selected ("interval", "scene", or "manual")
- `included`: Whether to include this frame in extraction (checkbox state)

---

### 1.2 CollapsibleSection Class (Lines 40-79)

```python
class CollapsibleSection(ctk.CTkFrame):
    """A collapsible section with toggle button (▼/▶) and show/hide content."""
```

**Purpose:** Reusable UI component that provides expandable/collapsible sections to reduce visual clutter while keeping advanced options accessible.

**Constructor:**
```python
def __init__(self, master, title: str, expanded: bool = False, **kwargs)
```

**Key Attributes:**
- `self.header`: Frame containing the toggle button
- `self.toggle_btn`: Button with ▼/▶ indicator and title
- `self.content`: Frame where child widgets are placed

**Public Methods:**
- `is_expanded() -> bool`: Check current state
- `expand()`: Force expand the section
- `collapse()`: Force collapse the section

**Usage Pattern:**
```python
section = CollapsibleSection(parent, "Section Title", expanded=False)
section.pack(fill="x", pady=5)

# Add widgets to section.content
ctk.CTkCheckBox(section.content, text="Option").pack()
ctk.CTkSlider(section.content, ...).pack()
```

---

## 2. Extract Tab Reorganization

### 2.1 Structural Changes

**Left Panel:** Changed from fixed `CTkFrame` to `CTkScrollableFrame` to accommodate additional controls.

```python
# Old:
left = ctk.CTkFrame(master=container, fg_color="transparent", width=380)

# New:
left = ctk.CTkScrollableFrame(master=container, width=380)
```

**Right Panel:** Now uses a nested `CTkTabview` with three tabs:
- **Queue**: Original video queue functionality
- **Plan**: New extraction planning interface
- **LUT Preview**: Before/after LUT comparison

### 2.2 Color & LUT Collapsible Section (Lines 372-427)

**Location:** Left panel, after Time Range section

**Components:**
| Widget | Variable | Purpose |
|--------|----------|---------|
| Enable checkbox | `extract_lut_enabled_var` | Toggle LUT application |
| LUT file entry | `extract_lut_file_entry` | Path to .cube file |
| Browse button | - | Opens file dialog |
| Strength slider | `extract_lut_strength_var` | 0-1 (displayed as 0-100%) |
| Shadows slider | `extract_shadow_var` | 0-100 (50 = neutral) |
| Highlights slider | `extract_highlight_var` | 0-100 (50 = neutral) |

**Helper Method:**
```python
def _on_extract_lut_toggle(self):
    """Called when LUT enabled checkbox changes."""
    # Auto-expands section when enabled
```

```python
def _browse_extract_lut_file(self):
    """Browse for LUT file in extract tab."""
```

### 2.3 Sky Filter Collapsible Section (Lines 428-472)

**Location:** Left panel, below Color & LUT section

**Components:**
| Widget | Variable | Purpose |
|--------|----------|---------|
| Enable checkbox | `extract_sky_enabled_var` | Toggle sky filtering |
| Brightness slider | `extract_sky_brightness_var` | 0.5-1.0 threshold |
| Keypoints slider | `extract_sky_keypoints_var` | 10-200 threshold |
| Min Pitch slider | `extract_sky_pitch_var` | 30-90 degrees |

**Note:** The pitch threshold is displayed but not currently used in the worker - it was intended for filtering only zenith-direction views. This could be enhanced in future iterations.

### 2.4 Launch Mask Tool Button (Lines 474-480)

```python
self.extract_mask_btn = ctk.CTkButton(
    left, text="Launch Mask Tool...",
    command=self._launch_segmentation_window,
    fg_color="#555555", hover_color="#666666", height=32
)
```

**Behavior:** Opens `SegmentationWindow` with the current output folder pre-populated.

### 2.5 Right Panel Tab Structure (Lines 482-654)

#### Queue Tab
Contains the original video queue interface, moved from the main right panel into a sub-tab.

#### Plan Tab (Lines 555-599)
**Components:**
- Header label and description
- Control buttons: "Analyze Plan", "Insert Frame", "Delete Selected"
- Scrollable frame list (`plan_scroll`)
- Statistics label (`plan_stats_label`)

**Related Methods:**
```python
def _analyze_extraction_plan(self):
    """Analyze video to generate extraction plan."""
    # Gets first selected or first pending video
    # Spawns worker thread

def _analyze_plan_worker(self, video_path):
    """Worker thread to analyze video and generate extraction plan."""
    # Uses VideoAnalyzer to get video info
    # Generates PlannedFrame list based on interval
    # Calls _refresh_plan_display()

def _refresh_plan_display(self):
    """Refresh the plan display with current planned frames."""
    # Creates checkbox + info label for each frame
    # Updates statistics

def _on_plan_include_change(self, idx: int, var):
    """Handle checkbox change for planned frame."""

def _plan_insert_frame(self):
    """Insert a manual frame into the plan."""
    # Shows input dialog for timestamp
    # Inserts at correct position

def _plan_delete_selected(self):
    """Delete unchecked frames from the plan."""
    # Removes frames where included=False
```

#### LUT Preview Tab (Lines 601-654)
**Components:**
- Frame slider with label (`lut_preview_frame_var`, 0-100 range)
- Side-by-side preview frames:
  - `lut_preview_before`: Original image
  - `lut_preview_after`: LUT-processed image
- "Update Preview" button

**Related Methods:**
```python
def _on_lut_preview_frame_change(self, value):
    """Handle LUT preview frame slider change."""

def _update_lut_preview(self):
    """Update LUT preview with current settings."""
    # Validates inputs
    # Spawns worker thread

def _lut_preview_worker(self, video_path, lut_path):
    """Worker to generate LUT preview images."""
    # Extracts single frame from video using OpenCV
    # Applies LUT and shadow/highlight
    # Resizes and displays both images
```

---

## 3. Extraction Worker Integration

### 3.1 Modified Method: `_extract_queue_worker()` (Lines 891-967)

The extraction worker now includes three post-processing steps after successful frame extraction:

#### Step 1: LUT Application (Lines 897-916)
```python
lut_enabled = self.extract_lut_enabled_var.get()
lut_path = self.extract_lut_file_entry.get()
if lut_enabled and lut_path and Path(lut_path).exists():
    processor = LUTProcessor()
    strength = self.extract_lut_strength_var.get()
    # Process each image in-place
```

#### Step 2: Shadow/Highlight Adjustments (Lines 917-935)
```python
shadow = self.extract_shadow_var.get()
highlight = self.extract_highlight_var.get()
if shadow != 50 or highlight != 50:
    from panoex.core.adjustments import apply_shadow_highlight
    # Process each image in-place
```

#### Step 3: Sky Filter (Lines 937-960)
```python
sky_enabled = self.extract_sky_enabled_var.get()
if sky_enabled:
    sky_config = SkyFilterConfig(
        brightness_threshold=self.extract_sky_brightness_var.get(),
        keypoint_threshold=self.extract_sky_keypoints_var.get(),
    )
    sky_filter = SkyFilter(sky_config)
    # Analyze each image, delete if is_sky
```

**Important Notes:**
- All three steps process images in-place (overwriting originals)
- Processing order is LUT -> Adjustments -> Sky Filter
- Sky filter runs last so it can evaluate the color-corrected images
- Final frame count is updated after sky removal

---

## 4. Reframe Tab Enhancements

### 4.1 Views Summary Display (Lines 1313-1319)

Added after the preset description label:

```python
self.reframe_views_summary = ctk.CTkTextbox(
    left, height=100, font=ctk.CTkFont(family="Consolas", size=10),
    fg_color="#1a1a1a", state="disabled"
)
```

### 4.2 New Method: `_update_views_summary()` (Lines 1488-1515)

```python
def _update_views_summary(self, config):
    """Update the views summary with a monospace tree display."""
```

**Output Format:**
```
Ring 0: pitch= +0°, 8 views @ 45°, V00-V07
Ring 1: pitch=-20°, 4 views @ 90°, V08-V11
Zenith: V12, pitch=+90°, FOV=90°
Nadir:  V13, pitch=-90°, FOV=90°
```

**Called by:** `_on_reframe_preset_change()` when preset selection changes.

---

## 5. SegmentationWindow Class

### 5.1 Class Definition (Lines 2992-3278)

```python
class SegmentationWindow(ctk.CTkToplevel):
    """Standalone window for YOLO segmentation with categorized class selection."""
```

### 5.2 Class Categories (Lines 2997-3008)

```python
CATEGORIES = {
    "People": [0],
    "Vehicles": [1, 2, 3, 4, 5, 6, 7, 8],
    "Animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "Sports": [32, 33, 34, 35, 36, 37, 38],
    "Food": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "Furniture": [56, 57, 58, 59, 60, 61, 62, 63],
    "Electronics": [62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
    "Kitchen": [39, 40, 41, 42, 43, 44, 45, 71, 72, 73, 74, 75],
    "Other": [24, 25, 26, 27, 28, 29, 30, 31, 76, 77, 78, 79],
}
```

**Note:** Some class IDs overlap between categories (e.g., Electronics and Furniture share 62, 63). This is intentional to allow logical groupings but could cause confusion.

### 5.3 Constructor

```python
def __init__(self, parent, input_folder: str = ""):
```

- `parent`: Reference to PanoexApp for logging
- `input_folder`: Pre-populated input path (from Extract tab)

**Window Properties:**
- Size: 900x700 (minimum 800x600)
- Modal: Uses `transient()` and `grab_set()`

### 5.4 UI Structure

#### Left Panel (Settings)
| Section | Components |
|---------|------------|
| Input/Output | Folder entries with browse buttons |
| Detection Classes | Category headers with checkbox grids |
| Quick Select | Default/All/None buttons |
| Model Settings | Model dropdown, confidence slider, dilate slider |
| Options | Invert mask checkbox |
| Actions | "Generate Masks" button |

#### Right Panel (Preview)
| Section | Components |
|---------|------------|
| Test Run | Frame slider, "Run Test" button |
| Preview | Side-by-side Original/With Mask display |

### 5.5 Key Methods

```python
def _load_test_frames(self):
    """Load frame list for test slider."""
    # Scans input folder for images
    # Configures slider range

def _select_default(self):
    """Select default classes (person, backpack, etc.)."""
    # Sets: {0, 24, 25, 26, 28}

def _run_test(self):
    """Run test segmentation on current frame."""
    # Spawns _test_worker thread

def _test_worker(self, frame_idx: int):
    """Worker thread for test segmentation."""
    # Runs single-frame segmentation
    # Creates green overlay visualization
    # Updates preview_masked label

def _run_segmentation(self):
    """Run full segmentation on all images."""
    # Validates inputs
    # Spawns _segmentation_worker thread

def _segmentation_worker(self, input_dir: str, output_dir: str):
    """Worker thread for full segmentation."""
    # Uses Segmenter.segment_batch()
    # Logs progress through parent.log()
```

---

## 6. Variable Reference

### 6.1 New Extract Tab Variables

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `extract_planned_frames` | `List[PlannedFrame]` | `[]` | Planned extraction frames |
| `extract_lut_enabled_var` | `BooleanVar` | `False` | Enable LUT processing |
| `extract_lut_file_entry` | `CTkEntry` | - | LUT file path |
| `extract_lut_strength_var` | `DoubleVar` | `1.0` | LUT strength (0-1) |
| `extract_sky_enabled_var` | `BooleanVar` | `False` | Enable sky filter |
| `extract_sky_brightness_var` | `DoubleVar` | `0.85` | Sky brightness threshold |
| `extract_sky_keypoints_var` | `IntVar` | `50` | Sky keypoints threshold |
| `extract_sky_pitch_var` | `IntVar` | `60` | Sky pitch threshold (unused) |
| `lut_preview_frame_var` | `IntVar` | `0` | Preview frame slider (0-100) |

### 6.2 New Reframe Tab Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `reframe_views_summary` | `CTkTextbox` | Monospace view configuration display |

---

## 7. Known Issues & Limitations

### 7.1 Pitch Threshold Not Used
The `extract_sky_pitch_var` slider is displayed but not used in the extraction worker. The original plan was to only filter images that came from views above a certain pitch angle, but this would require parsing view metadata from filenames.

### 7.2 Plan Tab Limitations
- Plan is regenerated from scratch each time "Analyze Plan" is clicked
- No persistence of plan between sessions
- Frame numbers are estimated using video FPS (may not match exact extraction)

### 7.3 LUT Preview Frame Mapping
The preview slider maps 0-100 to the video's frame range. This is a rough approximation and may not match exact extraction intervals.

### 7.4 SegmentationWindow Category Overlaps
Some COCO class IDs appear in multiple categories. This could cause UI confusion where checking a class in one category doesn't update its appearance in another.

### 7.5 Test Preview Visualization
The test segmentation uses a simple green overlay. A more sophisticated visualization showing individual instances with different colors would be beneficial.

---

## 8. Testing Checklist

- [ ] CollapsibleSection expands/collapses correctly
- [ ] LUT section auto-expands when checkbox enabled
- [ ] LUT file browser works and filters .cube files
- [ ] Sky filter sliders update labels correctly
- [ ] "Launch Mask Tool" opens SegmentationWindow
- [ ] SegmentationWindow receives input folder from Extract tab
- [ ] Plan tab "Analyze Plan" generates frame list
- [ ] Plan checkboxes toggle frame inclusion
- [ ] Insert Frame dialog accepts timestamps
- [ ] Delete Selected removes unchecked frames
- [ ] LUT Preview loads frame from first queued video
- [ ] Before/after preview updates correctly
- [ ] Views Summary updates when preset changes
- [ ] Views Summary shows correct ring information
- [ ] Extraction worker applies LUT when enabled
- [ ] Extraction worker applies shadow/highlight when not neutral
- [ ] Extraction worker removes sky images when enabled
- [ ] SegmentationWindow Default/All/None buttons work
- [ ] SegmentationWindow test run shows overlay
- [ ] SegmentationWindow full run processes all images

---

## 9. File Structure After Changes

```
panoex_gui.py
├── Imports (1-38)
├── PlannedFrame dataclass (31-37)
├── CollapsibleSection class (40-79)
├── PanoexApp class (81-2990)
│   ├── __init__ (82-133)
│   ├── _create_widgets (134-167)
│   ├── _build_analyze_tab (170-281)
│   ├── _build_extract_tab (285-655)
│   │   ├── Left panel with scrollable settings
│   │   ├── CollapsibleSection: Color & LUT (372-427)
│   │   ├── CollapsibleSection: Sky Filter (428-472)
│   │   ├── Launch Mask Tool button (474-480)
│   │   └── Right panel tabs: Queue/Plan/LUT Preview
│   ├── Queue methods (656-807)
│   ├── Extraction worker (808-968)
│   │   └── Post-processing: LUT → Adjustments → Sky Filter
│   ├── Plan/Preview helper methods (975-1263)
│   ├── _build_reframe_tab (1265-1416)
│   │   └── Views Summary textbox (1313-1319)
│   ├── Reframe methods (1418-1700)
│   │   └── _update_views_summary (1488-1515)
│   ├── _build_lut_tab (1702-1819)
│   ├── _build_sky_tab (1936-2103)
│   ├── _build_segment_tab (2105-2251)
│   ├── _build_pipeline_tab (2339-2575)
│   └── Shared helpers (2577-2901)
├── SegmentationWindow class (2992-3278)
│   ├── CATEGORIES dict (2997-3008)
│   ├── __init__ (3010-3025)
│   ├── _build_ui (3027-3179)
│   ├── Browse/load methods (3181-3206)
│   ├── Class selection methods (3208-3231)
│   ├── Preview methods (3233-3256)
│   ├── Test methods (3258-3302)
│   └── Full segmentation methods (3304-3278)
└── Main entry point (3280-3284)
```

---

## 10. Dependencies

No new external dependencies were added. The implementation uses:
- `customtkinter` - GUI framework
- `PIL/Pillow` - Image handling for previews
- `opencv-python` - Image processing
- `panoex.core` modules:
  - `VideoAnalyzer` - Video metadata analysis
  - `LUTProcessor` - LUT application
  - `SkyFilter`, `SkyFilterConfig` - Sky detection
  - `Segmenter`, `SegmentConfig` - YOLO segmentation
  - `adjustments.apply_shadow_highlight` - Color adjustments

---

## 11. Future Enhancements

1. **Plan Persistence**: Save/load extraction plans to JSON
2. **Plan from Scene Detection**: Use scene detection mode to pre-populate plan
3. **Pitch-Based Sky Filtering**: Parse view metadata to apply pitch threshold
4. **Multi-Video Preview**: Allow switching between queued videos in preview
5. **Instance Visualization**: Show individual detected objects with unique colors
6. **Class Category Deduplication**: Ensure each class only appears once in UI
7. **Progress Indicators**: Add progress bars to post-processing steps
8. **Batch LUT Preview**: Preview multiple frames as a grid

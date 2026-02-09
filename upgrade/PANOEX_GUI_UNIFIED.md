# Panoex GUI Restructure — Unified Layout with Inline Help

## Overview

Restructure the panoex GUI from tab-based navigation to a unified two-panel layout. Every control includes a subtle help description so users understand what each feature does.

---

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ panoex - Panorama Extractor                                      [─][□][×] │
├─────────────────────────────┬───────────────────────────────────────────────┤
│ LEFT PANEL                  │ RIGHT PANEL                                   │
│ (400px, scrollable)         │ (flexible width)                              │
│                             │                                               │
│ ┌─────────────────────────┐ │ ┌───────────────────────────────────────────┐ │
│ │ ▼ INPUT                 │ │ │ PREVIEW                                   │ │
│ └─────────────────────────┘ │ │                                           │ │
│ ┌─────────────────────────┐ │ │   [equirectangular image with overlays]   │ │
│ │ ▼ EXTRACTION            │ │ │                                           │ │
│ └─────────────────────────┘ │ │                                           │ │
│ ┌─────────────────────────┐ │ └───────────────────────────────────────────┘ │
│ │ ▼ REFRAME               │ │ ┌───────────────────────────────────────────┐ │
│ └─────────────────────────┘ │ │ LOG                                       │ │
│ ┌─────────────────────────┐ │ │   [scrollable log messages]               │ │
│ │ ▼ SEGMENTATION          │ │ └───────────────────────────────────────────┘ │
│ └─────────────────────────┘ │ ┌───────────────────────────────────────────┐ │
│ ┌─────────────────────────┐ │ │ PROGRESS                                  │ │
│ │ ▼ COLOR                 │ │ │   [████████░░░░░░] 45% - Extracting...   │ │
│ └─────────────────────────┘ │ └───────────────────────────────────────────┘ │
│ ┌─────────────────────────┐ │                                               │
│ │ ▼ SKY FILTER            │ │                                               │
│ └─────────────────────────┘ │                                               │
├─────────────────────────────┴───────────────────────────────────────────────┤
│ Ready │ 2 videos loaded │ 0 frames extracted │ Queue: empty                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Help Text Styling

All help/description text should use a consistent subtle style:

```python
# Help text style - smaller, dimmed, appears below or beside controls
HELP_TEXT_STYLE = {
    "font": ("Segoe UI", 11),      # Slightly smaller than main text
    "text_color": "#888888",        # Dimmed gray
}

# Create a helper function for consistent help labels
def add_help_text(parent, text):
    """Add subtle help text below a control."""
    label = ctk.CTkLabel(
        parent, 
        text=text,
        font=("Segoe UI", 11),
        text_color="#888888",
        anchor="w",
        wraplength=350  # Wrap long text
    )
    label.pack(anchor="w", padx=(20, 10), pady=(0, 8))
    return label
```

---

## Section: INPUT

**Section header:** `▼ INPUT`
**Section description:** *"Select source videos and configure input/output paths"*

### Controls

#### Folder
```
Label: "Source Folder"
Control: [Entry][Browse...]
Help: "Folder containing your 360° video files (.mp4, .mov, .insv)"
```

#### Output
```
Label: "Output Folder"
Control: [Entry][Browse...]
Help: "Where extracted frames and processed files will be saved. Auto-creates subfolders for each step."
```

#### LUT
```
Label: "LUT File (optional)"
Control: [Entry][Browse...][×]
Help: "Color lookup table (.cube) for log footage. Converts flat/log video to standard colors (e.g., D-Log to Rec.709)"
```

#### File List
```
Label: "Videos Found"
Control: [Scrollable checkbox list]
Help: "Check the videos you want to process. Shows duration, resolution, and detected camera preset."

Item format:
  ☑ NS-F-05-12.mp4
    6:28 • 30fps • 7680×3840
    Detected: DJI Osmo 360
```

---

## Section: EXTRACTION

**Section header:** `▼ EXTRACTION`
**Section description:** *"Extract frames from video at specified intervals"*

### Controls

#### Interval
```
Label: "Interval"
Control: [Entry: 2.0] seconds
Help: "Time between extracted frames. Lower = more frames, larger dataset. 2.0s is good for walking pace."
```

#### Max Gap
```
Label: "Max Gap"
Control: [Entry: 3.0] seconds
Help: "Maximum allowed gap between frames, even in static scenes. Ensures coverage if interval frames are skipped."
```

#### Mode
```
Label: "Mode"
Control: [Dropdown: Fixed ▼]
Options: "Fixed", "Scene", "Adaptive"
Help: 
  - Fixed: Extract at exact intervals (most predictable)
  - Scene: Extract on scene changes + interval (good for varied footage)  
  - Adaptive: Adjusts density based on motion (fewer frames when stationary)
```

#### Quality
```
Label: "Quality"
Control: [Slider: 70━━━━●━━━━100] 95
Help: "JPEG quality for extracted frames. 95 is ideal for photogrammetry (high quality, reasonable file size)."
```

#### Format
```
Label: "Format"
Control: [Dropdown: JPEG ▼]
Options: "JPEG", "PNG"
Help: "JPEG is smaller and faster. PNG is lossless but 5-10× larger files."
```

#### Time Range
```
Label: "Time Range"
Control: Start [Entry: 0] End [Entry: ∞] seconds
Help: "Process only a portion of the video. Leave End as ∞ for entire video."
```

#### Actions
```
Buttons: [Extract] [Cancel]
Help: "Extract will process all checked videos. Progress shown in right panel."
```

---

## Section: REFRAME

**Section header:** `▼ REFRAME`
**Section description:** *"Convert 360° equirectangular images to multiple perspective views for photogrammetry"*

### Controls

#### Input Folder
```
Label: "Input"
Control: [Entry][Browse...]
Help: "Folder containing equirectangular frames (usually the extraction output folder)"
```

#### Output Folder
```
Label: "Output"
Control: [Entry][Browse...]
Help: "Where perspective views will be saved. Creates many images (e.g., 13 per input frame)"
```

#### Preset
```
Label: "Preset"
Control: [Dropdown: panoex_default ▼]
Help: "Determines how many views and at what angles. See preview on right for visual representation."

Options with descriptions:
  - panoex_default: "13 views: 8 at horizon + 4 below + zenith (recommended)"
  - dense_coverage: "18 views: maximum overlap for difficult scenes"
  - lightweight: "7 views: faster processing, good for initial tests"
  - octa_horizon: "9 views: 8 at horizon + zenith (no downward views)"
  - full_sphere: "18 views: full spherical coverage including nadir"
```

#### Size
```
Label: "Output Size"
Control: [Dropdown: 1920 ▼]
Options: "1920", "2048", "2560", "3840", "Auto"
Help: "Resolution of each perspective view. 1920 balances quality and processing speed. Auto matches input resolution."
```

#### FOV
```
Label: "Field of View"
Control: [Slider: 50━━━●━━━━90] 65°
Help: "Wider FOV = more overlap between views but more distortion. 65° is a good balance."
```

#### Quality
```
Label: "Quality"  
Control: [Slider: 70━━━━●━━━━100] 95
Help: "JPEG quality for output perspective images."
```

#### Workers
```
Label: "Workers"
Control: [Slider: 1━●━━━━━━━16] 4
Help: "Parallel processing threads. More = faster but uses more RAM. Start with 4, increase if stable."
```

#### Options
```
☑ Include zenith (top view)
Help: "Add a straight-up view. Useful for capturing ceilings/sky structures. Disable for outdoor scenes."

☐ Include nadir (bottom view)  
Help: "Add a straight-down view. Usually disabled because tripod/operator is visible."

☐ Apply LUT during reframe
Help: "Apply the selected LUT to perspective views. Enable if you didn't apply LUT during extraction."
```

#### Actions
```
Buttons: [Reframe] [Cancel]
Help: "Generates perspective views from equirectangular frames. This multiplies your image count significantly."
```

---

## Section: SEGMENTATION

**Section header:** `▼ SEGMENTATION`
**Section description:** *"Generate masks to exclude people, vehicles, and other moving objects from 3D reconstruction"*

### Controls

#### Input Folder
```
Label: "Input"
Control: [Entry][Browse...]
Help: "Folder containing perspective views to generate masks for"
```

#### Output Folder
```
Label: "Output"
Control: [Dropdown: Sibling 'masks' folder ▼][Browse...]
Options: "Sibling 'masks' folder", "Same folder", "Custom..."
Help: "Where mask images are saved. 'Sibling' creates a parallel masks/ folder (recommended for Metashape)"
```

#### Model
```
Label: "Model"
Control: [Dropdown: yolo11s-seg ▼]
Help: "AI model for detecting objects. Larger = more accurate but slower."

Options with descriptions:
  - yolo11n-seg: "Nano (2.9M) — Fastest, lower accuracy"
  - yolo11s-seg: "Small (10M) — Balanced speed/accuracy (recommended)"  
  - yolo11m-seg: "Medium (22M) — Higher accuracy"
  - yolo11l-seg: "Large (47M) — High accuracy, slower"
  - yolo11x-seg: "XLarge (62M) — Highest accuracy, slowest"
```

#### Confidence
```
Label: "Confidence"
Control: [Slider: 0.1━━━━●━━━━0.9] 0.50
Help: "Detection threshold. Lower = more detections (may include false positives). 0.5 is a good default."
```

#### Classes
```
Label: "Target Classes"
Control: [Dropdown: People + Accessories ▼]
Help: "What objects to mask out."

Options:
  - People + Accessories: "person, backpack, handbag, umbrella, suitcase"
  - People only: "person"
  - Vehicles: "car, truck, bus, motorcycle, bicycle"
  - All moving objects: "people + vehicles + animals"
  - Custom...: [opens class selection dialog]
```

#### Dilate
```
Label: "Dilate"
Control: [Slider: 0━━●━━━━━━━50] 10 px
Help: "Expand masks by this many pixels. Helps cover edges/shadows that detection might miss."
```

#### Options
```
☐ Invert masks
Help: "Swap black/white. Some software expects masked areas as black instead of white."

☐ Save overlay preview
Help: "Save a copy of each image with the mask overlaid for visual verification."
```

#### Actions
```
Buttons: [Segment] [Cancel]
Help: "Runs AI detection on all images. First run downloads the model (~25MB)."
```

---

## Section: COLOR

**Section header:** `▼ COLOR`
**Section description:** *"Adjust exposure and apply color grading"*

### Controls

#### LUT
```
Label: "Apply LUT"
Control: [Dropdown: None ▼][Browse...]
Help: "Color lookup table to transform log/flat footage to vivid colors. Select 'None' if already applied."
```

#### LUT Strength
```
Label: "LUT Strength"
Control: [Slider: 0━━━━━━━━●━100] 100%
Help: "Blend between original (0%) and full LUT (100%). Useful for subtle color grading."
```

#### Shadow
```
Label: "Shadow"
Control: [Slider: 0━━━━━━●━━━100] 50
Help: "Lift dark areas. >50 brightens shadows, <50 darkens them. 50 = no change."
```

#### Highlight
```
Label: "Highlight"
Control: [Slider: 0━━━━━━●━━━100] 50
Help: "Compress bright areas. >50 recovers highlights, <50 increases contrast. 50 = no change."
```

#### Options
```
☐ Process in-place
Help: "Overwrite original files. If unchecked, saves to a 'color_adjusted' subfolder."
```

#### Actions
```
Buttons: [Apply Color] [Cancel]
Help: "Processes all images in the selected folder with the specified color adjustments."
```

---

## Section: SKY FILTER

**Section header:** `▼ SKY FILTER`
**Section description:** *"Remove frames that are mostly sky — they hurt photogrammetry alignment"*

### Controls

#### Input Folder
```
Label: "Input"
Control: [Entry][Browse...]
Help: "Folder containing perspective views to filter"
```

#### Brightness Threshold
```
Label: "Brightness"
Control: [Slider: 0━━━━━━━●━━1.0] 0.85
Help: "Maximum average brightness (0-1). Frames brighter than this may be sky. Default 0.85."
```

#### Keypoint Threshold
```
Label: "Min Keypoints"
Control: [Entry: 50]
Help: "Minimum feature points required. Sky has few features. Frames with fewer keypoints are likely sky."
```

#### Saturation Threshold
```
Label: "Max Saturation"
Control: [Slider: 0━━●━━━━━━━1.0] 0.15
Help: "Maximum color saturation. Sky is usually desaturated (blue/white). Default 0.15."
```

#### Options
```
☐ Move instead of delete
Help: "Move rejected frames to a 'sky_rejected' folder instead of deleting. Safer for review."

☐ Dry run (preview only)
Help: "Show what would be filtered without actually moving/deleting files."
```

#### Actions
```
Buttons: [Filter] [Cancel]
Help: "Analyzes all images and removes/moves sky-dominated frames."
```

---

## Right Panel: PREVIEW

**Section header:** `PREVIEW`

### Controls

#### Navigation (when images loaded)
```
Buttons: [← Prev] [Entry: frame number] [Next →]
Help: "Navigate through frames. Enter a number to jump to specific frame."
```

#### View Toggle
```
Buttons: [Original] [With Overlays] [Mask Preview]
Help: "Switch between viewing the raw image, image with view boundaries, or segmentation mask overlay."
```

#### Preview Info
```
Label: "Preset: panoex_default"
Label: "13 views: 8@+0°, 4@-20°, zenith"
Help: "Shows current reframe preset configuration. The overlay shows where each perspective view will be extracted."
```

---

## Right Panel: LOG

**Section header:** `LOG` with `[Clear]` button

```
Control: Scrollable text area (read-only)
Help: "Processing messages appear here. Timestamps show when each action occurred."

Format:
[12:34:56] Loaded 2 videos from E:/project/
[12:34:58] Starting extraction: NS-F-05-12.mp4
[12:35:02]   Extracted frame 1/500 (0:02)
[12:35:04]   Extracted frame 2/500 (0:04)
...
```

---

## Right Panel: PROGRESS

**Section header:** `PROGRESS`

```
Control: Progress bar + status label
[████████████░░░░░░░░░░░░░░░░░░] 42%
Status: "Extracting: NS-F-05-12.mp4 — Frame 210/500"

Help: "Shows progress of current operation. Cancel button appears when processing."
```

---

## Status Bar (Bottom)

```
Format: Ready │ 2 videos │ 1,234 frames │ Queue: 3 pending │ Errors: 0
         ↑          ↑           ↑              ↑              ↑
       State    Loaded     Extracted       Jobs waiting    Error count

Help: (shown as tooltip on hover)
  - State: Current application state (Ready, Processing, Error)
  - Videos: Number of video files loaded
  - Frames: Total frames extracted in current session
  - Queue: Background jobs waiting to process
  - Errors: Click to view error details
```

---

## Collapsible Section Implementation

```python
class CollapsibleSection(ctk.CTkFrame):
    """A collapsible section with header, description, and content area."""
    
    def __init__(self, parent, title: str, description: str = "", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.collapsed = False
        
        # Header frame (always visible)
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=5, pady=(5, 0))
        
        # Collapse toggle button
        self.toggle_btn = ctk.CTkButton(
            self.header_frame,
            text=f"▼ {title}",
            font=("Segoe UI", 14, "bold"),
            fg_color="transparent",
            hover_color="#333333",
            anchor="w",
            command=self.toggle
        )
        self.toggle_btn.pack(side="left", fill="x", expand=True)
        
        # Section description (subtle help text)
        if description:
            self.desc_label = ctk.CTkLabel(
                self,
                text=description,
                font=("Segoe UI", 11),
                text_color="#888888",
                anchor="w"
            )
            self.desc_label.pack(fill="x", padx=20, pady=(0, 5))
        
        # Content frame (collapsible)
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="x", padx=10, pady=5)
    
    def toggle(self):
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.content.pack_forget()
            if hasattr(self, 'desc_label'):
                self.desc_label.pack_forget()
            self.toggle_btn.configure(text=f"▶ {self.title}")
        else:
            if hasattr(self, 'desc_label'):
                self.desc_label.pack(fill="x", padx=20, pady=(0, 5))
            self.content.pack(fill="x", padx=10, pady=5)
            self.toggle_btn.configure(text=f"▼ {self.title}")
    
    def add_control(self, label: str, control_widget, help_text: str = ""):
        """Add a labeled control with optional help text."""
        row = ctk.CTkFrame(self.content, fg_color="transparent")
        row.pack(fill="x", pady=2)
        
        # Label
        lbl = ctk.CTkLabel(row, text=label, width=100, anchor="w")
        lbl.pack(side="left", padx=(0, 10))
        
        # Control
        control_widget.pack(side="left", fill="x", expand=True)
        
        # Help text (below the control row)
        if help_text:
            help_lbl = ctk.CTkLabel(
                self.content,
                text=help_text,
                font=("Segoe UI", 11),
                text_color="#666666",
                anchor="w",
                wraplength=350
            )
            help_lbl.pack(fill="x", padx=(110, 10), pady=(0, 8))
```

---

## Usage Example

```python
# Create the main layout
app = ctk.CTk()
app.geometry("1400x900")
app.title("panoex - Panorama Extractor")

# Left panel (scrollable)
left_panel = ctk.CTkScrollableFrame(app, width=400)
left_panel.pack(side="left", fill="y", padx=5, pady=5)

# Right panel
right_panel = ctk.CTkFrame(app)
right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

# Add INPUT section
input_section = CollapsibleSection(
    left_panel, 
    title="INPUT",
    description="Select source videos and configure input/output paths"
)
input_section.pack(fill="x", pady=5)

# Add folder control with help
folder_entry = ctk.CTkEntry(input_section.content, placeholder_text="Select folder...")
input_section.add_control(
    "Source Folder", 
    folder_entry,
    help_text="Folder containing your 360° video files (.mp4, .mov, .insv)"
)

# Add EXTRACTION section
extract_section = CollapsibleSection(
    left_panel,
    title="EXTRACTION", 
    description="Extract frames from video at specified intervals"
)
extract_section.pack(fill="x", pady=5)

interval_entry = ctk.CTkEntry(extract_section.content, placeholder_text="2.0")
extract_section.add_control(
    "Interval",
    interval_entry,
    help_text="Time between extracted frames. Lower = more frames. 2.0s is good for walking pace."
)

# ... continue for other sections
```

---

## Tooltips (Alternative/Additional Help)

For controls where inline help is too verbose, use tooltips:

```python
class ToolTip:
    """Hover tooltip for widgets."""
    
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip = None
        self.scheduled = None
        
        widget.bind("<Enter>", self.schedule_show)
        widget.bind("<Leave>", self.hide)
    
    def schedule_show(self, event):
        self.scheduled = self.widget.after(self.delay, self.show)
    
    def show(self):
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + 30
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(
            self.tooltip, 
            text=self.text, 
            background="#333333",
            foreground="#ffffff",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4,
            wraplength=300
        )
        label.pack()
    
    def hide(self, event=None):
        if self.scheduled:
            self.widget.after_cancel(self.scheduled)
            self.scheduled = None
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# Usage:
ToolTip(interval_entry, "Time between extracted frames. Lower = more frames, larger dataset.")
```

---

## Summary for Claude Code

1. **Replace tab-based layout** with unified two-panel layout
2. **Left panel**: Scrollable, 400px fixed width, contains collapsible sections
3. **Right panel**: Preview (60%), Log (30%), Progress (10%)
4. **Every control has help text** — displayed below the control in subtle gray
5. **Collapsible sections** let users hide what they don't need
6. **Workflow is top-to-bottom**: INPUT → EXTRACTION → REFRAME → SEGMENTATION → COLOR → SKY FILTER
7. **Preview updates contextually** based on which section is active
8. **Status bar** shows overall state at bottom

All help text is included inline. You can edit out obvious ones later.

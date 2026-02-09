# Panoex UI Replication Specification

## Overview

Replicate the UI layout and features of STUDIO DUCKBILL's Panoex tool. This document provides exact specifications for Claude Code to implement.

---

## Window 1: Panorama Extractor (Main Window)

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ panoex - Panorama Extractor                                      [─][□][×] │
├────────────────────────────────┬────────────────────────────────────────────┤
│ LEFT PANEL (400px)             │ RIGHT PANEL (flexible)                     │
│                                │                                            │
│ ┌────────────────────────────┐ │ ┌────────────────────────────────────────┐ │
│ │ INPUT                      │ │ │ Results  [Insert][Delete][Save] [▼video]│ │
│ │ Folder: [__________][...]  │ │ │ ┌──────────────────────────────────┐  │ │
│ │ Extract:[__________][...]  │ │ │ │ #00085 #05040 [FIX] 2:48.0      │  │ │
│ │ LUT:    [__________][...][×]│ │ │ │ #00086 #05100 [FIX] 2:50.0      │  │ │
│ │                            │ │ │ │ #00087 #05160 [FIX] 2:52.0      │  │ │
│ │ ┌──────────────────────┐   │ │ │ │ ...                              │  │ │
│ │ │☑ NS-F-05-12.mp4      │   │ │ │ └──────────────────────────────────┘  │ │
│ │ │  6:28 30fps 7680x3840│   │ │ │                                        │ │
│ │ │  [Analyzed][Extracted]│   │ │ │ View: [Preview][Frame #5040-FIX]      │ │
│ │ │  H5M-16mm 975img     │   │ │ │              [←Prev] #5040 [Next→]    │ │
│ │ │☑ NS-T-03-13.mp4      │   │ │ │ ┌──────────────────────────────────┐  │ │
│ │ │  ...                 │   │ │ │ │                                  │  │ │
│ │ └──────────────────────┘   │ │ │ │      DUAL FISHEYE PREVIEW        │  │ │
│ │                            │ │ │ │      with view overlays          │  │ │
│ │ [Launch Mask Tool]         │ │ │ │      and numbered labels         │  │ │
│ │                            │ │ │ │                                  │  │ │
│ ├────────────────────────────┤ │ │ └──────────────────────────────────┘  │ │
│ │ Analysis                   │ │ ├────────────────────────────────────────┤ │
│ │ Interval:[2.0] fps         │ │ │ Log                            [Clear]│ │
│ │ Max Gap: [3.0] sec         │ │ │ [17:41:57] Loaded presets...          │ │
│ │ Mode: [Fixed      ▼]       │ │ │ [17:44:28] Found 2 video files        │ │
│ │ □ Auto-Extract             │ │ │ [17:44:34] Switched to NS-T-03-13.mp4 │ │
│ │ [Analyze (2)][Cancel]      │ │ └────────────────────────────────────────┘ │
│ ├────────────────────────────┤ │                                            │
│ │ Extraction                 │ │                                            │
│ │ File: [____________][...]  │ │                                            │
│ │ Preset:[F13MD20D90-16▼]    │ │                                            │
│ │ □ Also extract Full Pixel  │ │                                            │
│ │ □ Sky Filter P:[30]KP:[50] │ │                                            │
│ │              Color:[85]    │ │                                            │
│ │ Size:[Auto▼] Quality:[95]  │ │                                            │
│ │ Start:[0   ] End:[∞     ]  │ │                                            │
│ │ Shadow:[====●====] 70      │ │                                            │
│ │ Highlight:[●=======] 10    │ │                                            │
│ │ [Extract (2)][Cancel]      │ │                                            │
│ ├────────────────────────────┤ │                                            │
│ │ Views                 (13) │ │                                            │
│ │ Focal: 16.0mm              │ │                                            │
│ │ ├ Ring 0: P:+0° 8v @45°    │ │                                            │
│ │ ├ Ring 1: P:-20° 4v @90°   │ │                                            │
│ │ └ Direct views: 1          │ │                                            │
│ └────────────────────────────┘ │                                            │
├────────────────────────────────┴────────────────────────────────────────────┤
│ Ready | 2/2 selected | 2 analyzed | 2 extracted                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Section: INPUT

#### Controls

| Control | Type | Description |
|---------|------|-------------|
| Folder | Entry + Browse | Source folder containing video files |
| Extract | Entry + Browse | Output folder for extracted frames (auto-fills to `{folder}/extract`) |
| LUT | Entry + Browse + Clear(×) | Optional .cube LUT file path |
| File List | Scrollable list with checkboxes | Shows all video files found in folder |

#### File List Item Display

Each video file shows:
```
☑ NS-F-05-12.mp4
  6:28 30fps 7680x3840
  [Analyzed] [Extracted]          ← Status badges (color-coded)
  H5M-16mm 975img                 ← Auto-detected preset + extracted frame count
```

**Status badges:**
- `Analyzed` - Orange/yellow badge
- `Extracted` - Green badge

**Implementation:**
```python
# File list item structure
class VideoFileItem:
    path: str
    duration: str          # "6:28"
    fps: float             # 30
    resolution: str        # "7680x3840"
    status: List[str]      # ["Analyzed", "Extracted"]
    detected_preset: str   # "H5M-16mm"
    extracted_count: int   # 975
    selected: bool         # Checkbox state
```

---

### Section: Analysis

#### Controls

| Control | Type | Default | Description |
|---------|------|---------|-------------|
| Interval | Entry (float) | 2.0 | Extraction interval in seconds |
| Max Gap | Entry (float) | 3.0 | Maximum gap between frames |
| Mode | Dropdown | "Fixed" | Options: "Fixed", "Adaptive", "Scene" |
| Auto-Extract | Checkbox | False | Automatically run extraction after analysis |
| Analyze | Button | - | Shows count of selected videos: "Analyze (2)" |
| Cancel | Button | - | Disabled unless analysis running |

---

### Section: Extraction

#### Controls

| Control | Type | Default | Description |
|---------|------|---------|-------------|
| File | Entry + Browse | - | Output filename pattern or folder |
| Preset | Dropdown | - | Camera/lens presets (auto-detected from video) |
| Also extract Full Pixel | Checkbox | False | Extract full resolution equirect alongside perspectives |
| Sky Filter | Checkbox | False | Enable sky filtering |
| Sky Filter - Pitch | Entry (int) | 30 | Pitch threshold for sky detection |
| Sky Filter - KP | Entry (int) | 50 | Keypoint threshold |
| Sky Filter - Color | Entry (int) | 85 | Color/brightness threshold |
| Size | Dropdown | "Auto" | Output size: "Auto", "1920", "2048", "3840" |
| Quality | Entry (int) | 95 | JPEG quality (1-100) |
| Start(sec) | Entry (float) | 0 | Start time in seconds |
| End(sec) | Entry | ∞ | End time (∞ = entire video) |
| Shadow | Slider | 70 | Shadow lift (0-100) |
| Highlight | Slider | 10 | Highlight compression (0-100) |
| Extract | Button | - | Shows count: "Extract (2)" |
| Cancel | Button | - | Disabled unless extracting |

---

### Section: Views

Shows the current preset's view configuration:

```
Views                                    (13)
Focal: 16.0mm
├ Ring 0: P:+0° 8 views @45°
├ Ring 1: P:-20° 4 views @90°
└ Direct views: 1
```

**Implementation:**
- Tree-style display with indent characters (├ └)
- Total view count in header
- Focal length from preset
- Each ring shows: pitch, count, spacing
- "Direct views" = zenith + nadir count

---

### Section: Results (Right Panel)

#### Header Controls

| Control | Type | Description |
|---------|------|-------------|
| Video Dropdown | Dropdown | Select which video's results to view |
| Insert | Button | Insert new extraction point |
| Delete | Button | Delete selected extraction point |
| Save | Button | Save extraction list to file |

#### Frame List

Scrollable list showing extraction points:
```
#00085  #05040  [FIX]  2:48.0
#00086  #05100  [FIX]  2:50.0
```

Columns:
- Extraction index (#00085)
- Frame number (#05040)
- Status tag [FIX] (fixed interval) or [SCN] (scene change)
- Timestamp (2:48.0)

#### Preview Section

| Control | Type | Description |
|---------|------|-------------|
| View Toggle | Segmented Button | "Preview" / "Frame #XXXX - FIX" |
| Prev | Button | Navigate to previous frame |
| Frame Number | Entry | Jump to specific frame |
| Next | Button | Navigate to next frame |
| Preview Canvas | Image | Shows current frame with overlays |

---

### Preview Overlay System

**CRITICAL IMPLEMENTATION DETAIL:**

Panoex shows the **equirectangular image** with perspective view boundaries projected as **curved polygons**. The curves arise because straight lines in 3D space become curves when projected onto equirectangular coordinates, especially near the poles.

#### How to Draw View Overlays on Equirectangular

```python
import numpy as np
import cv2
import math

def spherical_to_equirect(yaw_deg: float, pitch_deg: float, 
                          img_width: int, img_height: int) -> tuple:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.
    
    Args:
        yaw_deg: Horizontal angle, 0° = center, -180° to +180° (or 0° to 360°)
        pitch_deg: Vertical angle, 0° = horizon, +90° = zenith, -90° = nadir
        img_width: Equirectangular image width
        img_height: Equirectangular image height
    
    Returns:
        (x, y) pixel coordinates
    """
    # Normalize yaw to 0-360 range
    yaw = yaw_deg % 360
    
    # Convert to pixel coordinates
    # x: 0° at left edge, 360° at right edge
    x = (yaw / 360.0) * img_width
    
    # y: +90° (zenith) at top, -90° (nadir) at bottom
    # pitch 0° = middle of image
    y = ((90.0 - pitch_deg) / 180.0) * img_height
    
    return (int(x), int(y))


def get_view_boundary_points(
    center_yaw: float,    # View center yaw in degrees
    center_pitch: float,  # View center pitch in degrees  
    fov: float,           # Field of view in degrees
    img_width: int,
    img_height: int,
    num_points: int = 32  # Points per edge for smooth curves
) -> list:
    """
    Generate boundary points for a perspective view projected onto equirectangular.
    
    The boundary of a perspective view forms a curved shape when projected
    onto equirectangular coordinates because:
    1. Great circles appear as sinusoidal curves
    2. The distortion increases toward the poles
    
    Args:
        center_yaw: View center horizontal angle
        center_pitch: View center vertical angle
        fov: Field of view (assuming square view)
        img_width, img_height: Equirectangular image dimensions
        num_points: Number of points per edge for smooth curves
    
    Returns:
        List of (x, y) pixel coordinates forming the boundary polygon
    """
    half_fov = fov / 2.0
    points = []
    
    # We need to trace the boundary of the perspective view
    # The boundary in the perspective view is a rectangle, but we need to
    # project each point back to spherical coordinates
    
    # For each edge of the rectangle, sample points and convert to spherical
    # Edge coordinates in perspective view: corners at (±1, ±1) normalized
    
    def perspective_to_spherical(px, py, center_yaw, center_pitch, fov):
        """
        Convert normalized perspective coordinates to spherical.
        px, py in range [-1, 1] representing the perspective image plane.
        """
        # Focal length for the given FOV
        f = 1.0 / math.tan(math.radians(fov / 2))
        
        # Direction vector in camera space (looking down -Z)
        # x = right, y = up, z = forward
        dx = px
        dy = py
        dz = f
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/length, dy/length, dz/length
        
        # Rotate by pitch (around X axis)
        pitch_rad = math.radians(center_pitch)
        dy2 = dy * math.cos(pitch_rad) - dz * math.sin(pitch_rad)
        dz2 = dy * math.sin(pitch_rad) + dz * math.cos(pitch_rad)
        dy, dz = dy2, dz2
        
        # Rotate by yaw (around Y axis)
        yaw_rad = math.radians(center_yaw)
        dx2 = dx * math.cos(yaw_rad) + dz * math.sin(yaw_rad)
        dz2 = -dx * math.sin(yaw_rad) + dz * math.cos(yaw_rad)
        dx, dz = dx2, dz2
        
        # Convert to spherical
        yaw_out = math.degrees(math.atan2(dx, dz))
        pitch_out = math.degrees(math.asin(dy))
        
        return yaw_out, pitch_out
    
    # Trace the four edges of the perspective rectangle
    edges = [
        # Top edge: y = 1, x from -1 to 1
        [(x/num_points * 2 - 1, 1) for x in range(num_points + 1)],
        # Right edge: x = 1, y from 1 to -1
        [(1, 1 - y/num_points * 2) for y in range(num_points + 1)],
        # Bottom edge: y = -1, x from 1 to -1
        [(1 - x/num_points * 2, -1) for x in range(num_points + 1)],
        # Left edge: x = -1, y from -1 to 1
        [(-1, -1 + y/num_points * 2) for y in range(num_points + 1)],
    ]
    
    for edge in edges:
        for px, py in edge:
            yaw, pitch = perspective_to_spherical(
                px, py, center_yaw, center_pitch, fov
            )
            x, y = spherical_to_equirect(yaw, pitch, img_width, img_height)
            points.append((x, y))
    
    return points


def draw_view_overlay(
    image: np.ndarray,
    center_yaw: float,
    center_pitch: float,
    fov: float,
    color: tuple,          # BGR color
    label: str,            # "V01", "V02", etc.
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,  # Use cv2.LINE_4 for dashed effect
    filled: bool = False,
    fill_alpha: float = 0.2
):
    """
    Draw a single view's curved boundary on an equirectangular image.
    """
    h, w = image.shape[:2]
    points = get_view_boundary_points(
        center_yaw, center_pitch, fov, w, h, num_points=32
    )
    
    # Handle wrap-around at 0°/360° boundary
    # Split into segments that don't cross the boundary
    segments = []
    current_segment = [points[0]]
    
    for i in range(1, len(points)):
        prev_x = points[i-1][0]
        curr_x = points[i][0]
        
        # If jump is more than half the image width, it's a wrap
        if abs(curr_x - prev_x) > w / 2:
            segments.append(current_segment)
            current_segment = [points[i]]
        else:
            current_segment.append(points[i])
    
    segments.append(current_segment)
    
    # Draw each segment
    for segment in segments:
        if len(segment) > 1:
            pts = np.array(segment, dtype=np.int32)
            
            if filled:
                overlay = image.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)
            
            cv2.polylines(image, [pts], False, color, thickness, line_type)
    
    # Draw label at center
    cx, cy = spherical_to_equirect(center_yaw, center_pitch, w, h)
    
    # Label background for readability
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(image, 
                  (cx - label_size[0]//2 - 2, cy - label_size[1]//2 - 2),
                  (cx + label_size[0]//2 + 2, cy + label_size[1]//2 + 2),
                  (40, 40, 40), -1)
    cv2.putText(image, label, 
                (cx - label_size[0]//2, cy + label_size[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_all_view_overlays(
    image: np.ndarray,
    preset: dict,
    show_excluded: bool = True
):
    """
    Draw all view overlays for a preset configuration.
    
    Colors:
    - Ring 0 (horizon): Magenta (#FF00FF)
    - Ring 1 (below): Magenta or Cyan
    - Ring 2 (above): Magenta or different color
    - Excluded/sky views: Blue dashed
    - Nadir: Red
    """
    COLORS = {
        'ring_0': (255, 0, 255),    # Magenta (BGR)
        'ring_1': (255, 0, 255),    # Magenta
        'ring_2': (205, 0, 205),    # Darker magenta
        'excluded': (255, 100, 100), # Light blue for excluded
        'nadir': (0, 0, 255),       # Red
        'zenith': (255, 100, 100),  # Blue (excluded by default)
    }
    
    view_num = 0
    
    for ring_idx, ring in enumerate(preset.get('rings', [])):
        pitch = ring['pitch']
        count = ring['count']
        fov = ring.get('fov', 65)
        start_yaw = ring.get('start_yaw', 0)
        yaw_step = 360.0 / count
        
        color = COLORS.get(f'ring_{ring_idx}', (255, 0, 255))
        
        for i in range(count):
            yaw = (start_yaw + i * yaw_step) % 360
            # Normalize yaw to -180 to 180 for drawing
            if yaw > 180:
                yaw -= 360
            
            label = f"V{view_num:02d}"
            draw_view_overlay(image, yaw, pitch, fov, color, label)
            view_num += 1
    
    # Zenith
    if preset.get('include_zenith', False):
        label = f"V{view_num:02d}"
        # Zenith views often shown as excluded (dashed blue)
        draw_view_overlay(image, 0, 85, preset['rings'][0].get('fov', 65),
                         COLORS['zenith'], label)
        view_num += 1
    
    # Show excluded sky regions if enabled
    if show_excluded:
        # Draw dashed circles/regions for sky analysis zones
        # These are the E08, E09, etc. labels in Panoex
        pass  # Implementation depends on specific analysis zones
```

#### View Label Convention

From Panoex screenshots:
```
V00-V07  - Ring 0: 8 views at horizon (0° pitch)
V08-V11  - Ring 1: 4 views below horizon (-20° pitch)  
V12      - Direct view (zenith or primary)

E08-E11  - "Excluded" or "Evaluated" regions (sky/zenith)
           Shown with dashed lines, conditionally included
```

#### Analysis Metrics Overlay (PanoDyn Feature)

The top bar shows real-time analysis metrics:
```
Seq: 00052          - Frame sequence number
Motion: STRAIGHT    - Detected motion type (STRAIGHT, TURNING, STATIONARY)
Q-Score: 1200.0     - Overall quality score
Q-Thr: 1200.0       - Quality threshold (frames below this are rejected)
LapVar: 2144.7      - Laplacian variance (sharpness/blur detection)
Z_F: -0.54          - Zenith feature density
Z-Cov: 0.084        - Zenith coverage score
Z-Bright: 159.4     - Zenith brightness (sky detection)
Z-Dec: ON           - Zenith decision (include zenith views or not)
```

This informs automated decisions:
- **Frame selection**: Q-Score vs Q-Thr
- **Blur rejection**: LapVar threshold
- **Zenith inclusion**: Z-Dec based on Z_F, Z-Cov, Z-Bright
- **Density adjustment**: Motion type affects extraction interval

#### View Label Convention

```
V00, V01, V02... V07  - Ring 0 (8 views at horizon)
V08, V09, V10, V11   - Ring 1 (4 views below horizon)
V12                  - Zenith (direct view)
```

#### Overlay Colors (from screenshot)

```python
OVERLAY_COLORS = {
    'ring_0': (255, 0, 255),    # Magenta/Pink
    'ring_1': (255, 255, 0),    # Cyan (BGR)
    'zenith': (0, 255, 255),    # Yellow
    'nadir': (0, 165, 255),     # Orange
}
```

---

### Section: Log

```
Log                                                              [Clear]
[17:41:57] Loaded presets from: presets_panoex.json
[17:44:28] Folder selected: Video_NoLUT
[17:44:28]   Found 2 video file(s)
[17:44:28]   Auto-selected preset: H5M-16mm
[17:44:28]   2 file(s) with existing metadata
[17:44:34] Switched to: NS-T-03-13.mp4
```

- Scrollable text area
- Timestamps in [HH:MM:SS] format
- Indented sub-messages with 2 spaces
- Clear button to reset

---

## Window 2: Panoex Segmentation

Launched via "Launch Mask Tool" button from main window.

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Panoex Segmentation                                              [─][□][×] │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT                                                                       │
│ [Browse Folder] E:\20251023奈良井宿\Video_NoLUT\extract                      │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ NS-F-05-12/  NS-F-05-12_000000_00001_01.jpg ... _00195_05.jpg      975 │ │
│ │ NS-T-03-13/  NS-T-03-13_000000_00001_01.jpg ... _00188_13.jpg    2,444 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                       Total: 3,419 images  │
├─────────────────────────────────┬───────────────────────────────────────────┤
│ SEGMENTATION                    │ TARGET CLASSES                            │
│                                 │ [Default] [All] [None]                    │
│ Model: [YOLO26s-seg (10.4M) ▼] │                                            │
│ YOLO26 Small - Latest arch...  │ People & Accessories                       │
│                                 │ ☑ person        ☑ backpack               │
│ Confidence: [=======●==] 0.50  │ ☐ umbrella      ☑ handbag                 │
│ Mask Color: [White (det=w) ▼]  │ ☑ tie           ☑ suitcase                │
│ Filename:   [PNG format    ▼]  │                                            │
│ Output:     [Sibling masks ▼]  │ Vehicles                                   │
│             [Browse]           │ ☐ bicycle   ☐ car      ☐ airplane        │
│                                 │ ☐ motorcycle ☐ bus     ☐ train           │
│ → E:\...\Video_NoLUT\masks     │ ☐ truck     ☐ boat                        │
│                                 │                                            │
│ ☐ Save overlay images          │ Animals                                    │
│ ☐ Save JSON metadata           │ ☐ bird   ☐ cat    ☐ dog    ☐ horse       │
│                                 │ ☐ sheep  ☐ cow    ☐ elephant             │
├─────────────────────────────────┤ ☐ zebra  ☐ bear   ☐ giraffe              │
│ TEST RUN                        │                                            │
│ Image #: [=======●] 1/3419 [±] │                                            │
│ NS-F-05-12_000000_00001_01.jpg │                                            │
│ [Test]                          │                                            │
├─────────────────────────────────┴───────────────────────────────────────────┤
│ [Start Segmentation] [Cancel]                                               │
│ [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 24%                                  │
│ Ready                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Section: INPUT

| Control | Type | Description |
|---------|------|-------------|
| Browse Folder | Button | Select input folder |
| Path Display | Label | Shows selected path |
| Folder List | Scrollable list | Shows subfolders with image counts |

#### Folder List Format

```
NS-F-05-12/  NS-F-05-12_000000_00001_01.jpg ... NS-F-05-12_011640_00195_05.jpg    975
```

Shows:
- Folder name
- First image filename
- "..." 
- Last image filename
- Image count (right-aligned)

Total shown at bottom right.

---

### Section: SEGMENTATION

| Control | Type | Default | Options |
|---------|------|---------|---------|
| Model | Dropdown | YOLO26s-seg | See model list below |
| Model Description | Label | - | Auto-updates with model selection |
| Confidence | Slider | 0.50 | Range 0.10 - 0.90 |
| Mask Color | Dropdown | White | "White (detected = white)", "Black (detected = black)" |
| Filename | Dropdown | PNG | "PNG format (IMG_001.png)", "Match input name" |
| Output | Dropdown | Sibling | "Sibling 'masks' folder (../masks/)", "Custom..." |
| Browse | Button | - | Only enabled when Output = Custom |
| Output Preview | Label | - | Shows resolved output path |
| Save overlay images | Checkbox | False | Save image with mask overlay visualization |
| Save JSON metadata | Checkbox | False | Save detection metadata as JSON |

#### Model Options

```python
MODELS = [
    ("yolo11n-seg", "2.9M params", "YOLO11 Nano - Fastest, lowest accuracy"),
    ("yolo11s-seg", "10.1M params", "YOLO11 Small - Balanced speed/accuracy"),
    ("yolo11m-seg", "22.4M params", "YOLO11 Medium - Higher accuracy"),
    ("yolo11l-seg", "46.8M params", "YOLO11 Large - High accuracy"),
    ("yolo11x-seg", "62.1M params", "YOLO11 Extra Large - Highest accuracy"),
]
```

Note: Panoex uses "YOLO26s-seg" which may be their internal/custom model. Use YOLO11 equivalents.

---

### Section: TARGET CLASSES

#### Quick Selection Buttons

| Button | Action |
|--------|--------|
| Default | Select: person, backpack, handbag, tie, suitcase |
| All | Select all classes |
| None | Deselect all classes |

#### Class Categories

**People & Accessories:**
- person (0)
- backpack (24)
- umbrella (25)
- handbag (26)
- tie (27)
- suitcase (28)

**Vehicles:**
- bicycle (1)
- car (2)
- motorcycle (3)
- airplane (4)
- bus (5)
- train (6)
- truck (7)
- boat (8)

**Animals:**
- bird (14)
- cat (15)
- dog (16)
- horse (17)
- sheep (18)
- cow (19)
- elephant (20)
- bear (21)
- zebra (22)
- giraffe (23)

Layout: 2 columns of checkboxes per category.

---

### Section: TEST RUN

| Control | Type | Description |
|---------|------|-------------|
| Image # Slider | Slider | Select image index (1 to total) |
| +/- Buttons | Buttons | Increment/decrement image index |
| Filename | Label | Shows current image filename |
| Test | Button | Run segmentation on current image and show result |

---

### Bottom Controls

| Control | Type | Description |
|---------|------|-------------|
| Start Segmentation | Button | Begin batch processing |
| Cancel | Button | Stop processing (disabled when not running) |
| Progress Bar | Progress | Shows completion percentage |
| Status | Label | "Ready", "Processing...", "Completed", etc. |

---

## Color Theme

Based on screenshots, Panoex uses a dark theme:

```python
COLORS = {
    'bg_dark': '#1a1a2e',      # Main background
    'bg_medium': '#16213e',    # Panel background
    'bg_light': '#0f3460',     # Input field background
    'accent': '#e94560',       # Buttons, highlights
    'text': '#eaeaea',         # Primary text
    'text_dim': '#a0a0a0',     # Secondary text
    'success': '#4ecca3',      # Extracted badge
    'warning': '#ffc107',      # Analyzed badge
    'border': '#333355',       # Panel borders
}
```

---

## Implementation Notes for Claude Code

### Priority Order

1. **Restructure main window layout** to match two-panel design
2. **Add Results panel** with frame list and preview
3. **Implement dual fisheye preview** with view overlays (most complex)
4. **Add Views section** showing ring configuration
5. **Update Segmentation window** layout
6. **Add target class categories** with checkbox grid

### Key Challenges

1. **Dual fisheye overlay rendering** - Requires understanding fisheye projection math
2. **Frame list management** - Insert/delete/save extraction points
3. **Auto-preset detection** - Match video metadata to camera presets
4. **Status badges** - Color-coded status indicators in file list

### For Equirectangular (Simplified Alternative)

If dual fisheye is too complex, keep equirectangular preview but improve overlays:
- Add numbered labels to each view rectangle
- Use curved boundaries (not rectangles) that show actual equirect distortion
- Match Panoex's color scheme

### Fisheye vs Equirectangular Detection

```python
def detect_image_type(image_path: str) -> str:
    """Detect if image is dual fisheye or equirectangular."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if abs(w / h - 2.0) < 0.1:
        # 2:1 aspect ratio = equirectangular
        return "equirectangular"
    elif abs(w / h - 2.0) < 0.1 and has_two_circles(img):
        # 2:1 with two circular regions = dual fisheye
        return "dual_fisheye"
    else:
        return "unknown"

def has_two_circles(img: np.ndarray) -> bool:
    """Check if image contains two circular fisheye images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
                                minDist=img.shape[1]//3,
                                param1=50, param2=30,
                                minRadius=img.shape[0]//3,
                                maxRadius=img.shape[0]//2)
    return circles is not None and len(circles[0]) == 2
```

---

## Sample Prompt for Claude Code

```
Restructure the panoex GUI to match the Panoex UI specification in PANOEX_UI_SPEC.md.

Key changes:
1. Split main window into left panel (Input, Analysis, Extraction, Views) and right panel (Results with frame list, preview, log)
2. Add file list with checkboxes, status badges (Analyzed/Extracted), and auto-detected preset display
3. Add Results section with:
   - Video dropdown selector
   - Insert/Delete/Save buttons
   - Scrollable frame list showing index, frame#, status, timestamp
   - Preview/Frame toggle
   - Prev/Next navigation
   - Preview canvas with view overlays
4. Add Views section showing ring configuration tree
5. Keep equirectangular preview for now, but add numbered labels to view rectangles
6. Update color scheme to dark theme matching Panoex

For the Segmentation window:
1. Reorganize to match spec layout
2. Add categorized class checkboxes (People, Vehicles, Animals)
3. Add Default/All/None quick selection buttons
4. Add folder list showing image ranges and counts
5. Add Test Run section with image slider

Use CustomTkinter. Maintain the modular core library architecture.
```

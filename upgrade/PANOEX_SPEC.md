# Panoex-Style 360° Processing Tool — Technical Specification

## Overview

Build a desktop application for processing 360° video into photogrammetry/3DGS-ready image sets. The tool handles the complete pipeline: video analysis, frame extraction, multi-perspective reframing, LUT application, sky filtering, and AI-powered masking.

Reference: STUDIO DUCKBILL's "Panoex" tool (Japan-based photogrammetry studio).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI Application                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Panorama        │  │ Segmentation    │  │ Settings/       │ │
│  │ Extractor       │  │ (Masking)       │  │ Presets         │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼─────────┐
│                        Core Library                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │ Video    │ │ Reframe  │ │ LUT      │ │ YOLO     │ │ Preset ││
│  │ Analyzer │ │ Engine   │ │ Processor│ │ Segmenter│ │ Manager││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Video Analyzer

### Purpose
Analyze input video files to extract metadata and recommend extraction parameters.

### Inputs
- Video file path (.mp4, .mov, .insv, .360)
- Optional: Camera preset name

### Outputs
```python
{
    "path": str,
    "format": str,           # "mp4", "mov", etc.
    "codec": str,            # "h264", "hevc", etc.
    "width": int,            # e.g., 7680
    "height": int,           # e.g., 3840
    "fps": float,            # e.g., 30.0
    "duration_seconds": float,
    "frame_count": int,
    "is_equirectangular": bool,  # True if 2:1 aspect ratio
    "is_log_format": bool,       # Detect D-Log, V-Log, etc.
    "recommended_interval": float,
    "recommended_lut": str or None
}
```

### Implementation
```python
import subprocess
import json

def analyze_video(video_path: str) -> dict:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames,codec_name,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    # Parse and return structured dict
```

---

## Module 2: Frame Extractor

### Purpose
Extract frames from video with smart selection (interval + max gap + scene detection).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | float | 2.0 | Base extraction interval in seconds |
| `max_gap` | float | 3.0 | Maximum allowed gap between frames |
| `mode` | str | "fixed" | "fixed", "adaptive", "scene" |
| `start_sec` | float | 0 | Start time |
| `end_sec` | float | None | End time (None = end of video) |
| `quality` | int | 95 | JPEG quality (1-100) |

### Selection Modes

**Fixed Mode:**
Extract at exact intervals.
```
ffmpeg -i video.mp4 -vf "fps=1/{interval}" -qscale:v 2 out/%05d.jpg
```

**Adaptive Mode:**
Extract at interval, but ensure no gap exceeds max_gap even in static scenes.
```python
def adaptive_select(video_path, interval, max_gap):
    # 1. Extract scene change scores
    # 2. Select frames at interval
    # 3. If gap > max_gap, insert intermediate frame
```

**Scene Mode:**
Use scene detection with minimum interval.
```
ffmpeg -i video.mp4 -vf "select='gt(scene,0.2)*gte(t-prev_selected_t,{interval})'" -vsync vfr out/%05d.jpg
```

---

## Module 3: Reframe Engine

### Purpose
Convert equirectangular frames to multiple pinhole perspective views.

### View Configuration System

Views are organized in "rings" at different pitch angles:

```python
@dataclass
class Ring:
    pitch: float        # Degrees from horizon (-90 to +90)
    count: int          # Number of views in this ring
    fov: float          # Field of view in degrees
    start_yaw: float = 0  # Starting yaw offset

@dataclass 
class ViewConfig:
    rings: List[Ring]
    include_zenith: bool = True    # Top view (pitch=90)
    include_nadir: bool = False    # Bottom view (pitch=-90)
```

### Reference Configuration (from Panoex screenshots)

```python
PANOEX_DEFAULT = ViewConfig(
    rings=[
        Ring(pitch=0, count=8, fov=65, start_yaw=0),     # Horizon: 8 views @ 45° spacing
        Ring(pitch=-20, count=4, fov=65, start_yaw=0),  # Below horizon: 4 views @ 90° spacing
    ],
    include_zenith=True,
    include_nadir=False,  # Usually has tripod
)
# Total: 8 + 4 + 1 = 13 views per frame
```

### Preset Examples

```python
PRESETS = {
    "panoex_default": {
        "rings": [
            {"pitch": 0, "count": 8, "fov": 65},
            {"pitch": -20, "count": 4, "fov": 65},
        ],
        "zenith": True,
        "nadir": False,
    },
    "dense_coverage": {
        "rings": [
            {"pitch": 0, "count": 8, "fov": 60},
            {"pitch": 30, "count": 4, "fov": 60},
            {"pitch": -30, "count": 4, "fov": 60},
        ],
        "zenith": True,
        "nadir": True,
    },
    "lightweight": {
        "rings": [
            {"pitch": 0, "count": 6, "fov": 75},
        ],
        "zenith": True,
        "nadir": False,
    },
}
```

### Output Naming Convention

```
{original_stem}_{frame:06d}_{ring:02d}_{view:02d}.jpg

Examples:
  NS-T-03-13_000001_00_00.jpg  # Frame 1, Ring 0, View 0
  NS-T-03-13_000001_00_07.jpg  # Frame 1, Ring 0, View 7
  NS-T-03-13_000001_01_00.jpg  # Frame 1, Ring 1, View 0
  NS-T-03-13_000001_ZN_00.jpg  # Frame 1, Zenith
```

### Implementation

```python
import numpy as np
import cv2

# Use py360convert if available, else custom implementation
try:
    import py360convert
    def reframe(equirect, fov, yaw, pitch, size):
        return py360convert.e2p(equirect, fov_deg=fov, u_deg=yaw, v_deg=pitch, 
                                 out_hw=(size, size), mode="bilinear")
except ImportError:
    # Fallback implementation using cv2.remap
    pass

def extract_views(equirect_path: str, config: ViewConfig, output_dir: str, size: int = 1920):
    """Extract all perspective views from an equirectangular image."""
    equirect = cv2.imread(equirect_path)
    stem = Path(equirect_path).stem
    
    views = []
    
    # Process rings
    for ring_idx, ring in enumerate(config.rings):
        yaw_step = 360.0 / ring.count
        for view_idx in range(ring.count):
            yaw = ring.start_yaw + view_idx * yaw_step
            
            persp = reframe(equirect, ring.fov, yaw, ring.pitch, size)
            
            out_name = f"{stem}_{ring_idx:02d}_{view_idx:02d}.jpg"
            cv2.imwrite(str(Path(output_dir) / out_name), persp, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            views.append(out_name)
    
    # Zenith
    if config.include_zenith:
        persp = reframe(equirect, config.rings[0].fov, 0, 90, size)
        out_name = f"{stem}_ZN_00.jpg"
        cv2.imwrite(str(Path(output_dir) / out_name), persp,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        views.append(out_name)
    
    # Nadir
    if config.include_nadir:
        persp = reframe(equirect, config.rings[0].fov, 0, -90, size)
        out_name = f"{stem}_ND_00.jpg"
        cv2.imwrite(str(Path(output_dir) / out_name), persp,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
        views.append(out_name)
    
    return views
```

---

## Module 4: LUT Processor

### Purpose
Apply color lookup tables to convert log footage to standard color space.

### Supported Formats
- .cube (3D LUT)
- .3dl (Autodesk)

### Common LUTs for 360 Cameras

| Camera | Log Format | LUT |
|--------|------------|-----|
| DJI Osmo Action | D-Log M | DJI D-Log M to Rec.709 |
| Insta360 X3/X4 | I-Log | Insta360 I-Log to Rec.709 |
| GoPro MAX | Protune Flat | GoPro Protune to Rec.709 |

### Implementation

```python
import numpy as np

def load_cube_lut(cube_path: str) -> tuple:
    """Load .cube LUT file."""
    with open(cube_path, 'r') as f:
        lines = f.readlines()
    
    size = None
    lut_data = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[1])
        elif line and not line.startswith('#') and not line.startswith('TITLE') and not line.startswith('DOMAIN'):
            values = [float(x) for x in line.split()]
            if len(values) == 3:
                lut_data.append(values)
    
    lut_3d = np.array(lut_data).reshape(size, size, size, 3)
    return lut_3d, size

def apply_lut(image: np.ndarray, lut_3d: np.ndarray) -> np.ndarray:
    """Apply 3D LUT to image using trilinear interpolation."""
    # Normalize to 0-1
    img_float = image.astype(np.float32) / 255.0
    
    size = lut_3d.shape[0]
    
    # Scale to LUT indices
    img_scaled = img_float * (size - 1)
    
    # Get integer and fractional parts
    idx = img_scaled.astype(np.int32)
    frac = img_scaled - idx
    
    # Clamp indices
    idx = np.clip(idx, 0, size - 2)
    
    # Trilinear interpolation
    # ... (implement or use scipy.ndimage.map_coordinates)
    
    return (result * 255).astype(np.uint8)

# Alternative: Use colour-science library
# pip install colour-science
import colour
def apply_lut_colour(image: np.ndarray, cube_path: str) -> np.ndarray:
    lut = colour.read_LUT(cube_path)
    img_float = image.astype(np.float32) / 255.0
    result = lut.apply(img_float)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)
```

---

## Module 5: Sky Filter

### Purpose
Detect and optionally skip frames that are predominantly sky (useless for photogrammetry).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pitch_threshold` | float | 30 | Skip views with pitch > this |
| `brightness_threshold` | float | 0.85 | Mean brightness threshold (0-1) |
| `saturation_threshold` | float | 0.15 | Low saturation = likely sky |
| `keypoint_threshold` | int | 50 | Minimum feature count |

### Implementation

```python
import cv2
import numpy as np

def is_sky_frame(image: np.ndarray, 
                 brightness_thresh: float = 0.85,
                 saturation_thresh: float = 0.15,
                 keypoint_thresh: int = 50) -> tuple[bool, dict]:
    """
    Detect if image is predominantly sky.
    
    Returns: (is_sky, metrics_dict)
    """
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Brightness (V channel or grayscale mean)
    brightness = gray.mean() / 255.0
    
    # Saturation
    saturation = hsv[:, :, 1].mean() / 255.0
    
    # Feature detection (sky has few features)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    kp_count = len(keypoints)
    
    # Blue ratio (sky is often blue)
    b, g, r = cv2.split(image)
    blue_ratio = b.mean() / (r.mean() + g.mean() + b.mean() + 1e-6)
    
    metrics = {
        "brightness": brightness,
        "saturation": saturation,
        "keypoint_count": kp_count,
        "blue_ratio": blue_ratio,
    }
    
    is_sky = (
        brightness > brightness_thresh and
        saturation < saturation_thresh and
        kp_count < keypoint_thresh
    )
    
    return is_sky, metrics
```

---

## Module 6: YOLO Segmentation

### Purpose
Generate masks for people, objects, and other unwanted elements using YOLO instance segmentation.

### Model Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n-seg | 2.9M | Fastest | Good |
| yolo11s-seg | 10.1M | Fast | Better |
| yolo11m-seg | 22.4M | Medium | High |
| yolo11x-seg | 62.1M | Slow | Highest |

### COCO Classes for Masking

```python
MASK_CLASSES = {
    # People & Accessories
    0: "person",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    
    # Vehicles (optional)
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    
    # Animals (optional)
    14: "bird",
    15: "cat",
    16: "dog",
}

DEFAULT_MASK_CLASSES = [0, 24, 25, 26, 27, 28]  # Person + accessories
```

### Implementation

```python
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

class YOLOSegmenter:
    def __init__(self, model_name: str = "yolo11s-seg.pt"):
        self.model = YOLO(model_name)
    
    def segment(self, 
                image_path: str, 
                classes: list[int] = None,
                confidence: float = 0.5,
                mask_color: int = 255) -> np.ndarray:
        """
        Generate segmentation mask for specified classes.
        
        Args:
            image_path: Path to input image
            classes: List of COCO class IDs to mask (None = all)
            confidence: Minimum detection confidence
            mask_color: Value for masked pixels (255=white, 0=black)
        
        Returns:
            Binary mask (H, W) where mask_color = detected object
        """
        if classes is None:
            classes = DEFAULT_MASK_CLASSES
        
        # Run inference
        results = self.model(image_path, conf=confidence, classes=classes)
        
        # Get image dimensions
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine all detection masks
        if results[0].masks is not None:
            for seg_mask in results[0].masks.data:
                # Resize mask to image size
                seg_mask = seg_mask.cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (w, h))
                mask[seg_mask > 0.5] = mask_color
        
        return mask
    
    def batch_segment(self,
                      input_dir: str,
                      output_dir: str,
                      classes: list[int] = None,
                      confidence: float = 0.5,
                      mask_color: int = 255,
                      callback=None):
        """Process all images in directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        for i, img_path in enumerate(images):
            mask = self.segment(str(img_path), classes, confidence, mask_color)
            
            out_path = output_path / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), mask)
            
            if callback:
                callback(i + 1, len(images), img_path.name)
```

---

## Module 7: Preset Manager

### Purpose
Save and load camera/extraction presets for different hardware configurations.

### Preset Schema

```json
{
  "name": "DJI Osmo 360 D-Log",
  "camera": {
    "model": "DJI Osmo Action 360",
    "lens_mm": 16.0,
    "sensor_size": "1/2.3\""
  },
  "extraction": {
    "interval": 2.0,
    "max_gap": 3.0,
    "mode": "adaptive"
  },
  "reframe": {
    "rings": [
      {"pitch": 0, "count": 8, "fov": 65},
      {"pitch": -20, "count": 4, "fov": 65}
    ],
    "zenith": true,
    "nadir": false,
    "output_size": 1920
  },
  "color": {
    "lut": "DJI D-Log M to Rec.709 V1.cube",
    "shadow": 70,
    "highlight": 10
  },
  "sky_filter": {
    "enabled": true,
    "pitch_threshold": 30,
    "keypoint_threshold": 50,
    "brightness_threshold": 0.85
  },
  "masking": {
    "model": "yolo11s-seg",
    "classes": [0, 24, 25, 26, 27, 28],
    "confidence": 0.5
  }
}
```

### Implementation

```python
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Preset:
    name: str
    camera_model: str
    lens_mm: float
    
    # Extraction
    interval: float = 2.0
    max_gap: float = 3.0
    extraction_mode: str = "adaptive"
    
    # Reframe
    rings: list = None
    include_zenith: bool = True
    include_nadir: bool = False
    output_size: int = 1920
    
    # Color
    lut_path: Optional[str] = None
    shadow: int = 50
    highlight: int = 50
    
    # Sky filter
    sky_filter_enabled: bool = True
    sky_pitch_threshold: float = 30
    sky_kp_threshold: int = 50
    
    # Masking
    mask_model: str = "yolo11s-seg"
    mask_classes: list = None
    mask_confidence: float = 0.5

class PresetManager:
    def __init__(self, presets_dir: str = "./presets"):
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(exist_ok=True)
    
    def save(self, preset: Preset):
        path = self.presets_dir / f"{preset.name}.json"
        with open(path, 'w') as f:
            json.dump(asdict(preset), f, indent=2)
    
    def load(self, name: str) -> Preset:
        path = self.presets_dir / f"{name}.json"
        with open(path, 'r') as f:
            data = json.load(f)
        return Preset(**data)
    
    def list_presets(self) -> list[str]:
        return [p.stem for p in self.presets_dir.glob("*.json")]
```

---

## GUI Specification

### Recommended Framework
**DearPyGui** — Fast, GPU-accelerated, good for image preview and real-time sliders.

Alternative: **PyQt6** for more native appearance.

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [File] [Edit] [Presets] [Help]                                    [─][□][×] │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────┐  ┌────────────────────────────────────┐ │
│ │ INPUT                       │  │ RESULTS                            │ │
│ │ Folder: [_______________][…]│  │ ┌──────────────────────────────┐  │ │
│ │ Extract:[_______________][…]│  │ │ #00085  [FIX] 2:48.0        │  │ │
│ │ LUT:    [_______________][×]│  │ │ #00086  [FIX] 2:50.0        │  │ │
│ │                             │  │ │ #00087  [FIX] 2:52.0        │  │ │
│ │ ┌─────────────────────────┐ │  │ │ ...                          │  │ │
│ │ │ File        │ Status    │ │  │ └──────────────────────────────┘  │ │
│ │ │ NS-F-05.mp4 │ Extracted │ │  │                                    │ │
│ │ │ NS-T-03.mp4 │ Analyzed  │ │  │ View: [Preview] [Frame #5040-FIX]  │ │
│ │ └─────────────────────────┘ │  │ ┌──────────────────────────────┐  │ │
│ │                             │  │ │                              │  │ │
│ │ [Launch Mask Tool]          │  │ │     EQUIRECT PREVIEW         │  │ │
│ │                             │  │ │     with view overlays       │  │ │
│ ├─────────────────────────────┤  │ │                              │  │ │
│ │ ANALYSIS                    │  │ └──────────────────────────────┘  │ │
│ │ Interval: [2.0] fps         │  │                                    │ │
│ │ Max Gap:  [3.0] sec         │  ├────────────────────────────────────┤ │
│ │ Mode: [Fixed     ▼]         │  │ LOG                                │ │
│ │ □ Auto-Extract              │  │ [17:44:28] Loaded presets...       │ │
│ │ [Analyze (2)] [Cancel]      │  │ [17:44:28] Found 2 video files     │ │
│ ├─────────────────────────────┤  │ [17:44:34] Switched to NS-T-03.mp4 │ │
│ │ EXTRACTION                  │  └────────────────────────────────────┘ │
│ │ File: [________________][…] │                                          │
│ │ Preset: [F13MD20D90-16mm ▼] │                                          │
│ │ □ Also extract Full Pixel   │                                          │
│ │ □ Sky Filter  P:[30] K:[50] │                                          │
│ │ Size: [Auto▼] Quality: [95] │                                          │
│ │ Start: [0   ] End: [∞     ] │                                          │
│ │ Shadow: [====●====] 70      │                                          │
│ │ Highlight: [●========] 10   │                                          │
│ │                             │                                          │
│ │ VIEWS                  (13) │                                          │
│ │ Focal: 16.0mm               │                                          │
│ │ ├ Ring 0: P:+0° 8v @45°     │                                          │
│ │ ├ Ring 1: P:-20° 4v @90°    │                                          │
│ │ └ Zenith: 1                 │                                          │
│ │                             │                                          │
│ │ [Extract (2)] [Cancel]      │                                          │
│ └─────────────────────────────┘                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Ready | 2/2 selected | 2 analyzed | 2 extracted                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Segmentation Window Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Panoex Segmentation                                              [─][□][×] │
├─────────────────────────────────────────────────────────────────────────┤
│ INPUT                                                                    │
│ [Browse Folder] E:\project\Video_NoLUT\extract                          │
│ ┌───────────────────────────────────────────────────────────────┐       │
│ │ NS-F-05-12/  NS-F-05-12_000000_00001_01.jpg ... _00195_05.jpg │  975  │
│ │ NS-T-03-13/  NS-T-03-13_000000_00001_01.jpg ... _00188_13.jpg │ 2,444 │
│ └───────────────────────────────────────────────────────────────┘       │
│                                                    Total: 3,419 images  │
├─────────────────────────────────────────────────────────────────────────┤
│ SEGMENTATION                      │ TARGET CLASSES                      │
│ Model: [YOLO26s-seg (10.4M) ▼]   │ [Default] [All] [None]              │
│                                   │                                      │
│ Confidence: [========●==] 0.50   │ People & Accessories                 │
│ Mask Color: [White (detected)▼]  │ ☑ person      ☑ backpack            │
│ Filename:   [PNG format     ▼]   │ ☐ umbrella    ☑ handbag             │
│ Output:     [Sibling masks/ ▼]   │ ☑ tie         ☑ suitcase            │
│                                   │                                      │
│ → E:\project\masks               │ Vehicles                             │
│                                   │ ☐ bicycle  ☐ car    ☐ airplane     │
│ ☐ Save overlay images            │ ☐ motorcycle ☐ bus  ☐ train        │
│ ☐ Save JSON metadata             │ ☐ truck    ☐ boat                   │
│                                   │                                      │
│ TEST RUN                          │ Animals                              │
│ Image #: [========●] 1/3419 [+]  │ ☐ bird  ☐ cat   ☐ dog   ☐ horse    │
│ NS-F-05-12_000000_00001_01.jpg   │ ☐ sheep ☐ cow   ☐ elephant          │
│ [Test]                            │ ☐ zebra ☐ bear  ☐ giraffe          │
├───────────────────────────────────┴──────────────────────────────────────┤
│ [Start Segmentation] [Cancel]                                            │
│ [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 24%  819/3419                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
panoex/
├── main.py                 # Entry point
├── requirements.txt
├── presets/
│   ├── dji_osmo_360.json
│   ├── insta360_x3.json
│   └── gopro_max.json
├── luts/
│   └── DJI_DLog_M_to_Rec709.cube
├── core/
│   ├── __init__.py
│   ├── analyzer.py         # Video analysis
│   ├── extractor.py        # Frame extraction
│   ├── reframer.py         # Equirect → perspective
│   ├── lut.py              # LUT processing
│   ├── sky_filter.py       # Sky detection
│   ├── segmenter.py        # YOLO masking
│   └── presets.py          # Preset management
├── gui/
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── extractor_panel.py  # Extraction controls
│   ├── segmentation_window.py
│   ├── preview_widget.py   # Equirect preview with overlays
│   └── styles.py           # Theme/colors
└── tests/
    └── ...
```

---

## Dependencies

```
# requirements.txt

# Core
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0

# 360 Processing
py360convert>=0.1.0

# Color/LUT
colour-science>=0.4.0

# AI/Segmentation
ultralytics>=8.0.0
torch>=2.0.0

# GUI (choose one)
dearpygui>=1.10.0
# or: PyQt6>=6.5.0

# Video
ffmpeg-python>=0.2.0
# Note: Requires ffmpeg and ffprobe in PATH

# Utilities
tqdm>=4.65.0
```

---

## CLI Interface (for testing without GUI)

```bash
# Analyze video
python -m panoex analyze video.mp4

# Extract frames
python -m panoex extract video.mp4 ./output --preset dji_osmo_360

# Reframe equirectangular images
python -m panoex reframe ./equirect_frames ./perspectives --preset panoex_default

# Generate masks
python -m panoex segment ./perspectives ./masks --model yolo11s-seg --classes 0,24,25

# Full pipeline
python -m panoex pipeline video.mp4 ./output --preset dji_osmo_360 --mask
```

---

## Implementation Priority

### Phase 1: Core Pipeline (MVP)
1. Video analyzer
2. Frame extractor (fixed interval)
3. Reframe engine with configurable rings
4. Basic CLI interface

### Phase 2: Quality Features
5. LUT processor
6. Sky filter
7. YOLO segmentation
8. Preset manager

### Phase 3: GUI
9. Main window with DearPyGui
10. Preview with view overlays
11. Segmentation window
12. Progress tracking

### Phase 4: Polish
13. Adaptive extraction mode
14. Batch processing
15. Error handling & logging
16. Packaging/distribution

"""
prep360 Core Library

Modular components for 360° video processing pipeline.
"""

# Segmenter imports are lazy to avoid pulling ultralytics at package import time.
# Use: from prep360.core.segmenter import Segmenter
# Or:  from prep360.core import Segmenter  (triggers lazy load)
from .adjustments import apply_shadow_highlight, batch_adjust_images
from .analyzer import VideoAnalyzer, VideoInfo
from .blur_filter import BlurFilter, BlurFilterConfig
from .bridge_extractor import BridgeExtractor, BridgeFrameInfo, BridgeRequest, BridgeResult
from .colmap_export import (
    ColmapExportConfig,
    ColmapExporter,
    ExportResult,
    MetashapeProject,
    parse_metashape_xml,
)
from .extractor import ExtractionConfig, ExtractionMode, FrameExtractor
from .fisheye_calibration import (
    DualFisheyeCalibration,
    FisheyeCalibration,
    FisheyeCalibrator,
)
from .fisheye_reframer import (
    FISHEYE_PRESETS,
    FisheyeReframer,
    FisheyeView,
    FisheyeViewConfig,
    default_osmo360_calibration,
)
from .fisheye_reframer import (
    batch_extract as fisheye_batch_extract,
)
from .gap_detector import CameraPosition as GapCameraPosition
from .gap_detector import GapDetector, GapReport, SpatialGap
from .lut import LUTProcessor
from .motion_selector import FrameScore, MotionSelector, SelectionResult
from .osv import OSVHandler, OSVInfo
from .paired_split_video_extractor import (
    PairedSplitConfig,
    PairedSplitResult,
    PairedSplitVideoExtractor,
)
from .presets import Preset, PresetManager
from .reframer import Reframer, Ring, ViewConfig
from .sharpest_extractor import SharpestConfig, SharpestExtractor, SharpestResult
from .sky_filter import SkyFilter, SkyFilterConfig

__all__ = [
    "VideoAnalyzer",
    "VideoInfo",
    "FrameExtractor",
    "ExtractionMode",
    "ExtractionConfig",
    "Reframer",
    "Ring",
    "ViewConfig",
    "Preset",
    "PresetManager",
    "SkyFilter",
    "SkyFilterConfig",
    "LUTProcessor",
    "Segmenter",
    "SegmentConfig",
    "COCO_CLASSES",
    "CLASS_PRESETS",
    "apply_shadow_highlight",
    "batch_adjust_images",
    "BlurFilter",
    "BlurFilterConfig",
    "ColmapExporter",
    "ColmapExportConfig",
    "MetashapeProject",
    "parse_metashape_xml",
    "ExportResult",
    "OSVHandler",
    "OSVInfo",
    "FisheyeCalibrator",
    "FisheyeCalibration",
    "DualFisheyeCalibration",
    "FisheyeReframer",
    "FisheyeView",
    "FisheyeViewConfig",
    "FISHEYE_PRESETS",
    "fisheye_batch_extract",
    "default_osmo360_calibration",
    "MotionSelector",
    "FrameScore",
    "SelectionResult",
    "GapDetector",
    "GapReport",
    "SpatialGap",
    "BridgeExtractor",
    "BridgeRequest",
    "BridgeResult",
    "BridgeFrameInfo",
    "SharpestExtractor",
    "SharpestConfig",
    "SharpestResult",
    "PairedSplitVideoExtractor",
    "PairedSplitConfig",
    "PairedSplitResult",
    "Segmenter",
    "SegmentConfig",
    "COCO_CLASSES",
    "CLASS_PRESETS",
]


# Lazy imports for heavy optional dependencies (ultralytics)
_LAZY_SEGMENTER = {"Segmenter", "SegmentConfig", "COCO_CLASSES", "CLASS_PRESETS"}

def __getattr__(name):
    if name in _LAZY_SEGMENTER:
        from .segmenter import CLASS_PRESETS, COCO_CLASSES, SegmentConfig, Segmenter
        _map = {"Segmenter": Segmenter, "SegmentConfig": SegmentConfig,
                "COCO_CLASSES": COCO_CLASSES, "CLASS_PRESETS": CLASS_PRESETS}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

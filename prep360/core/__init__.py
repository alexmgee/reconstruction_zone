"""
prep360 Core Library

Modular components for 360° video processing pipeline.
"""

from .analyzer import VideoAnalyzer, VideoInfo
from .extractor import FrameExtractor, ExtractionMode, ExtractionConfig
from .reframer import Reframer, Ring, ViewConfig
from .presets import Preset, PresetManager
from .sky_filter import SkyFilter, SkyFilterConfig
from .lut import LUTProcessor
# Segmenter imports are lazy to avoid pulling ultralytics at package import time.
# Use: from prep360.core.segmenter import Segmenter
# Or:  from prep360.core import Segmenter  (triggers lazy load)
from .adjustments import apply_shadow_highlight, batch_adjust_images
from .blur_filter import BlurFilter, BlurFilterConfig
from .colmap_export import (
    ColmapExporter, ColmapExportConfig, MetashapeProject,
    parse_metashape_xml, ExportResult,
)
from .osv import OSVHandler, OSVInfo
from .fisheye_calibration import (
    FisheyeCalibrator, FisheyeCalibration, DualFisheyeCalibration,
)
from .motion_selector import MotionSelector, FrameScore, SelectionResult
from .gap_detector import GapDetector, GapReport, SpatialGap, CameraPosition as GapCameraPosition
from .fisheye_reframer import (
    FisheyeReframer, FisheyeView, FisheyeViewConfig,
    FISHEYE_PRESETS, batch_extract as fisheye_batch_extract,
    default_osmo360_calibration,
)
from .bridge_extractor import BridgeExtractor, BridgeRequest, BridgeResult, BridgeFrameInfo
from .sharpest_extractor import SharpestExtractor, SharpestConfig, SharpestResult
from .paired_split_video_extractor import (
    PairedSplitVideoExtractor,
    PairedSplitConfig,
    PairedSplitResult,
)

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
        from .segmenter import Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS
        _map = {"Segmenter": Segmenter, "SegmentConfig": SegmentConfig,
                "COCO_CLASSES": COCO_CLASSES, "CLASS_PRESETS": CLASS_PRESETS}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

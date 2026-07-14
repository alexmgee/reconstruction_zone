"""
prep360 - 360° Video Processing Pipeline

A modular toolkit for processing 360° video into photogrammetry/3DGS-ready image sets.

Pipeline stages:
1. Analyze - Extract video metadata and recommendations
2. Extract - Extract frames from video
3. Reframe - Convert equirectangular to perspective views
4. LUT - Apply color correction
5. Sky Filter - Remove useless sky frames
6. Segment - AI-powered masking (YOLO)

Usage:
    # As library
    from prep360.core import VideoAnalyzer, FrameExtractor, Reframer

    # As CLI
    python -m prep360 analyze video.mp4
    python -m prep360 extract video.mp4 ./frames
    python -m prep360 reframe ./frames ./perspectives
"""

__version__ = "0.1.0"
__author__ = "Edgeworks"

from .core import (
    FreeView,
    OutputLayout,
    VIEW_PRESETS,
    DEFAULT_PRESET,
    get_view_preset,
    copy_view_config,
    resolve_preset_name,
    generate_rig_config,
    write_rig_config,
    BlurFilter,
    BlurFilterConfig,
    ColmapExportConfig,
    ColmapExporter,
    ExportResult,
    ExtractionMode,
    FrameExtractor,
    LUTProcessor,
    MetashapeProject,
    Preset,
    PresetManager,
    Reframer,
    Ring,
    SkyFilter,
    SkyFilterConfig,
    VideoAnalyzer,
    VideoInfo,
    ViewConfig,
    parse_metashape_xml,
)

__all__ = [
    "VideoAnalyzer",
    "VideoInfo",
    "FrameExtractor",
    "ExtractionMode",
    "Reframer",
    "Ring",
    "FreeView",
    "ViewConfig",
    "OutputLayout",
    "VIEW_PRESETS",
    "DEFAULT_PRESET",
    "get_view_preset",
    "copy_view_config",
    "resolve_preset_name",
    "generate_rig_config",
    "write_rig_config",
    "Preset",
    "PresetManager",
    "SkyFilter",
    "SkyFilterConfig",
    "LUTProcessor",
    "Segmenter",
    "SegmentConfig",
    "COCO_CLASSES",
    "CLASS_PRESETS",
    "BlurFilter",
    "BlurFilterConfig",
    "ColmapExporter",
    "ColmapExportConfig",
    "MetashapeProject",
    "parse_metashape_xml",
    "ExportResult",
]


# Lazy imports for heavy optional dependencies (ultralytics via prep360.core.segmenter)
_LAZY_SEGMENTER = {"Segmenter", "SegmentConfig", "COCO_CLASSES", "CLASS_PRESETS"}

def __getattr__(name):
    if name in _LAZY_SEGMENTER:
        from .core.segmenter import CLASS_PRESETS, COCO_CLASSES, SegmentConfig, Segmenter
        _map = {"Segmenter": Segmenter, "SegmentConfig": SegmentConfig,
                "COCO_CLASSES": COCO_CLASSES, "CLASS_PRESETS": CLASS_PRESETS}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

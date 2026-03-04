"""
Panoex - 360° Video Processing Pipeline

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
    from panoex.core import VideoAnalyzer, FrameExtractor, Reframer

    # As CLI
    python -m panoex analyze video.mp4
    python -m panoex extract video.mp4 ./frames
    python -m panoex reframe ./frames ./perspectives
"""

__version__ = "0.1.0"
__author__ = "Edgeworks"

from .core import (
    VideoAnalyzer,
    VideoInfo,
    FrameExtractor,
    ExtractionMode,
    Reframer,
    Ring,
    ViewConfig,
    Preset,
    PresetManager,
    SkyFilter,
    SkyFilterConfig,
    LUTProcessor,
    Segmenter,
    SegmentConfig,
    COCO_CLASSES,
    CLASS_PRESETS,
)

__all__ = [
    "VideoAnalyzer",
    "VideoInfo",
    "FrameExtractor",
    "ExtractionMode",
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
]

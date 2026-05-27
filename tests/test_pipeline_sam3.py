"""Diagnostic: test SAM3 through the actual MaskingPipeline code path.

Compares standalone Sam3Processor calls vs pipeline's segment_image method
to find exactly where detections are lost.

Usage: python reconstruction_gui/test_pipeline_sam3.py <equirect_image_path>
"""
import sys, os

# Ensure reconstruction_gui and sam3 are importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3'))

import cv2
import numpy as np
from PIL import Image

from reconstruction_pipeline import (
    MaskConfig, MaskingPipeline, SAM3Segmenter, CubemapProjection,
    ImageGeometry, SegmentationModel
)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path or not os.path.exists(img_path):
        print("Usage: python reconstruction_gui/test_pipeline_sam3.py <equirect_image_path>")
        return

    # Load equirect
    equirect = cv2.imread(img_path)
    h, w = equirect.shape[:2]
    print(f"Equirect: {w}x{h}")

    # Extract cubemap faces (same as pipeline does)
    face_size = min(1024, w // 4)
    cubemap = CubemapProjection(face_size, overlap_degrees=0)
    faces = cubemap.equirect2cubemap(equirect)

    # Create config with defaults (mimicking GUI with empty prompts entry)
    config = MaskConfig(
        model=SegmentationModel.SAM3,
        device='cuda',
        confidence_threshold=0.5,
    )
    print(f"Config remove_prompts: {config.remove_prompts}")
    print(f"Config keep_prompts: {config.keep_prompts}")
    print(f"Config confidence_threshold: {config.confidence_threshold}")

    # Create SAM3Segmenter directly (same as pipeline does)
    segmenter = SAM3Segmenter(config)
    segmenter.initialize()
    print(f"Sam3Processor confidence_threshold: {segmenter.processor.confidence_threshold}")

    # Test on each face using segment_image (the pipeline method)
    print(f"\n{'='*60}")
    print("Test 1: Using segment_image with prompts=None (pipeline default path)")
    print(f"{'='*60}")
    for name, face_img in faces.items():
        results = segmenter.segment_image(face_img, prompts=None, geometry=ImageGeometry.PINHOLE)
        print(f"  {name}: {len(results)} detections")
        for r in results:
            print(f"    score={r.confidence:.3f} prompt={r.metadata.get('prompt')}")

    # Test with explicit prompts dict
    print(f"\n{'='*60}")
    print("Test 2: Using segment_image with explicit prompts dict")
    print(f"{'='*60}")
    explicit_prompts = {'remove': ['person', 'tripod', 'backpack'], 'keep': []}
    for name, face_img in faces.items():
        results = segmenter.segment_image(face_img, prompts=explicit_prompts, geometry=ImageGeometry.PINHOLE)
        print(f"  {name}: {len(results)} detections")
        for r in results:
            print(f"    score={r.confidence:.3f} prompt={r.metadata.get('prompt')}")

    # Test the full pipeline process_image
    print(f"\n{'='*60}")
    print("Test 3: Using MaskingPipeline.process_image (full pipeline path)")
    print(f"{'='*60}")
    pipeline = MaskingPipeline(config=config, auto_select_model=False)
    result = pipeline.process_image(equirect, ImageGeometry.EQUIRECTANGULAR)
    mask_pixels = np.count_nonzero(result.mask)
    total_pixels = result.mask.size
    pct = mask_pixels / total_pixels * 100
    print(f"  Final mask: {mask_pixels}/{total_pixels} pixels ({pct:.1f}%)")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Quality: {result.quality:.3f}")
    print(f"  Metadata: {result.metadata}")

    print("\nDone.")


if __name__ == "__main__":
    main()

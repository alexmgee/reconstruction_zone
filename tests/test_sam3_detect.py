"""Quick SAM3 detection diagnostic — run from command line (not pythonw).

Usage: python reconstruction_gui/test_sam3_detect.py <image_path>
"""
import sys, os

# Ensure sam3 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3'))

import torch
import cv2
import numpy as np
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path or not os.path.exists(img_path):
        print("Usage: python reconstruction_gui/test_sam3_detect.py <image_path>")
        return

    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    model = model.cuda()
    model.eval()

    # Test with TWO thresholds
    for thresh in [0.5, 0.3, 0.1]:
        processor = Sam3Processor(model, confidence_threshold=thresh)
        print(f"\n{'='*60}")
        print(f"Confidence threshold: {thresh}")
        print(f"{'='*60}")

        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]
        print(f"Image: {w}x{h}")

        # If equirect, extract a rough "left" face (where person likely is)
        if w >= h * 1.8:
            face_w = min(1024, w // 4)
            # Person is in the left portion of this equirect
            face = cv2.resize(img_bgr[:, :w//4, :], (face_w, face_w))
            print(f"Equirect detected — using left quarter resized to {face_w}x{face_w}")
        else:
            face = img_bgr
            print("Using full image as-is")

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        state = processor.set_image(pil_image)

        prompts = ["person", "photographer", "man", "human", "tripod",
                   "camera operator person", "equipment gear"]
        for prompt in prompts:
            # Reset prompts between calls to avoid stale state
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(prompt=prompt, state=state)

            masks = output.get('masks')
            scores = output.get('scores')
            boxes = output.get('boxes')

            if masks is not None and len(masks) > 0:
                print(f"  '{prompt}': {len(masks)} detections, "
                      f"scores={scores.cpu().numpy().round(3)}, "
                      f"mask shapes={masks.shape}")
            else:
                print(f"  '{prompt}': 0 detections")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Diagnostic: extract cubemap faces from equirect and run SAM3 on each.

Saves face images to disk so we can visually inspect what SAM3 receives.

Usage: python reconstruction_gui/test_cubemap_sam3.py <equirect_image_path>
"""
import sys, os

# Ensure reconstruction_gui and sam3 are importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'sam3'))

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path or not os.path.exists(img_path):
        print("Usage: python reconstruction_gui/test_cubemap_sam3.py <equirect_image_path>")
        return

    # Output dir for debug faces
    out_dir = Path(img_path).parent / "debug_cubemap_faces"
    out_dir.mkdir(exist_ok=True)
    print(f"Saving face images to: {out_dir}")

    # Load equirect
    equirect = cv2.imread(img_path)
    h, w = equirect.shape[:2]
    print(f"Equirect: {w}x{h}")

    # Import CubemapProjection from reconstruction_pipeline
    from reconstruction_pipeline import CubemapProjection

    face_size = min(1024, w // 4)
    cubemap = CubemapProjection(face_size, overlap_degrees=0)
    faces = cubemap.equirect2cubemap(equirect)

    # Save each face
    for name, face_img in faces.items():
        face_path = out_dir / f"face_{name}.jpg"
        cv2.imwrite(str(face_path), face_img)
        print(f"  Saved {name}: {face_img.shape[1]}x{face_img.shape[0]} -> {face_path}")

    # Load SAM3
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model()
    model = model.cuda()
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.3)

    prompts = ["person", "tripod", "backpack"]

    for name, face_img in faces.items():
        print(f"\n{'='*50}")
        print(f"Face: {name} ({face_img.shape[1]}x{face_img.shape[0]})")
        print(f"{'='*50}")

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        state = processor.set_image(pil_image)

        for prompt in prompts:
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(prompt=prompt, state=state)

            masks = output.get('masks')
            scores = output.get('scores')

            if masks is not None and len(masks) > 0:
                print(f"  '{prompt}': {len(masks)} detections, "
                      f"scores={scores.cpu().numpy().round(3)}")

                # Save mask overlay for first detection
                mask_np = masks[0].cpu().numpy()
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                mask_path = out_dir / f"mask_{name}_{prompt}.png"
                cv2.imwrite(str(mask_path), mask_uint8)

                # Overlay
                overlay = face_img.copy()
                overlay[mask_np > 0.5] = [0, 0, 255]  # Red overlay
                alpha = cv2.addWeighted(face_img, 0.6, overlay, 0.4, 0)
                overlay_path = out_dir / f"overlay_{name}_{prompt}.jpg"
                cv2.imwrite(str(overlay_path), alpha)
            else:
                print(f"  '{prompt}': 0 detections")

    print(f"\nDone. Check faces at: {out_dir}")


if __name__ == "__main__":
    main()

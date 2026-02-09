#!/usr/bin/env python3
"""
Quick shadow detection test using SAM3 directly.
Bypasses all masking_v2.py pipeline complexity.
"""
import os
import numpy as np
from PIL import Image
import cv2

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import sam3

# Setup
SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = os.path.join(SAM3_ROOT, "assets", "bpe_simple_vocab_16e6.txt.gz")

print("Loading SAM3...")
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.1)

# Load image
img_path = "/mnt/d/Michal/Tree/360/video/frames/frame_000001.png"
img = Image.open(img_path)
print(f"Image size: {img.size}")

# Set image
state = processor.set_image(img)

# Combine masks from multiple shadow prompts
combined_shadow = None
prompts_to_try = ["shadow", "shadow on ground", "human shadow", "person shadow"]

for prompt in prompts_to_try:
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    
    masks = state.get('masks')
    scores = state.get('scores')
    
    if masks is not None and masks.numel() > 0:
        print(f"'{prompt}': Found {masks.shape[0]} masks, shape={masks.shape}")
        
        # Convert to numpy and combine all masks for this prompt
        masks_np = masks.cpu().numpy()
        
        # Squeeze any (1,) dimensions: (N, 1, H, W) -> (N, H, W)
        masks_np = masks_np.squeeze()
        
        # If single mask, add back the batch dimension
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]
        
        # OR all masks together to get combined shadow
        prompt_combined = np.any(masks_np > 0.5, axis=0)  # Shape: (H, W)
            
        if combined_shadow is None:
            combined_shadow = prompt_combined
        else:
            combined_shadow = np.logical_or(combined_shadow, prompt_combined)
    else:
        print(f"'{prompt}': No masks found")

# Also get the person mask so we can compare
processor.reset_all_prompts(state)
state = processor.set_text_prompt(state=state, prompt="person")
person_masks = state.get('masks')
person_combined = None
if person_masks is not None and person_masks.numel() > 0:
    print(f"'person': Found {person_masks.shape[0]} masks, shape={person_masks.shape}")
    masks_np = person_masks.cpu().numpy().squeeze()
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]
    person_combined = np.any(masks_np > 0.5, axis=0)

# Save results
output_dir = "/mnt/d/Michal/Tree/360/video/masks"
os.makedirs(output_dir, exist_ok=True)

# Save shadow mask
if combined_shadow is not None:
    shadow_mask = (combined_shadow.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "shadow_only.png"), shadow_mask)
    print(f"Saved shadow mask to {output_dir}/shadow_only.png")

# Save person mask
if person_combined is not None:
    person_mask = (person_combined.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "person_only.png"), person_mask)
    print(f"Saved person mask to {output_dir}/person_only.png")

# Save combined (person + shadow)
if combined_shadow is not None and person_combined is not None:
    full_mask = np.logical_or(combined_shadow, person_combined).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, "person_and_shadow.png"), full_mask)
    print(f"Saved combined mask to {output_dir}/person_and_shadow.png")

print("Done!")

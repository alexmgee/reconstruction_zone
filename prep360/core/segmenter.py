"""
Segmenter Module

AI-powered instance segmentation using YOLO for generating masks.
Useful for masking out people, equipment, and other unwanted objects
before photogrammetry processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Dict, Tuple, Set
import json
import time

import cv2
import numpy as np

# Try to import ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# COCO class names (80 classes)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Reverse mapping: name -> id
COCO_NAME_TO_ID = {name: id for id, name in COCO_CLASSES.items()}

# Common presets for photogrammetry masking
CLASS_PRESETS = {
    "person": [0],  # Just people
    "photographer": [0, 24, 25, 26, 28],  # Person + bags/umbrella/suitcase
    "equipment": [0, 24, 25, 26, 27, 28, 67],  # Person + accessories + phone
    "vehicles": [1, 2, 3, 5, 6, 7, 8],  # All vehicles
    "animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # All animals
    "all_dynamic": [0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19],  # People + vehicles + common animals
}


@dataclass
class SegmentConfig:
    """Configuration for segmentation."""
    model_name: str = "yolo11n-seg.pt"  # YOLO model (n/s/m/l/x)
    classes: List[int] = field(default_factory=lambda: [0])  # COCO class IDs to detect
    confidence: float = 0.5  # Detection confidence threshold
    iou_threshold: float = 0.5  # NMS IoU threshold
    dilate_pixels: int = 0  # Dilate masks by N pixels
    invert_mask: bool = False  # True = mask is black on white (COLMAP style)
    device: str = ""  # "" for auto, "cpu", "cuda:0", etc.


@dataclass
class SegmentResult:
    """Result of segmentation for a single image."""
    image_name: str
    mask_path: Optional[str]
    detections: int  # Number of objects detected
    classes_found: List[str]  # Class names found
    error: Optional[str] = None


@dataclass
class BatchSegmentResult:
    """Result of batch segmentation."""
    success: bool
    total_images: int
    images_with_detections: int
    total_detections: int
    output_dir: str
    results: List[SegmentResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class Segmenter:
    """YOLO-based instance segmentation for mask generation."""

    def __init__(self, config: Optional[SegmentConfig] = None):
        if not HAS_YOLO:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        self.config = config or SegmentConfig()
        self._model = None

    @property
    def model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            self._model = YOLO(self.config.model_name)
            if self.config.device:
                self._model.to(self.config.device)
        return self._model

    def segment_image(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment a single image and return mask + detection info.

        Args:
            image: Input BGR image
            classes: Class IDs to detect (None = use config)

        Returns:
            (mask, detections) where mask is binary uint8 and
            detections is list of {class_id, class_name, confidence, bbox}
        """
        target_classes = classes if classes is not None else self.config.classes

        # Run inference
        results = self.model(
            image,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            classes=target_classes,
            verbose=False,
        )

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Check if segmentation masks exist
            if result.masks is not None and len(result.masks) > 0:
                for i, seg_mask in enumerate(result.masks.data):
                    # Get detection info
                    cls_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    bbox = result.boxes.xyxy[i].cpu().numpy().tolist()

                    detections.append({
                        "class_id": cls_id,
                        "class_name": COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
                        "confidence": conf,
                        "bbox": bbox,
                    })

                    # Add to combined mask
                    seg_np = seg_mask.cpu().numpy()
                    # Resize mask to image size if needed
                    if seg_np.shape != (h, w):
                        seg_np = cv2.resize(seg_np, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = np.maximum(mask, (seg_np > 0.5).astype(np.uint8) * 255)

        # Dilate if configured
        if self.config.dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.dilate_pixels * 2 + 1, self.config.dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)

        # Invert if configured (for COLMAP-style masks)
        if self.config.invert_mask:
            mask = 255 - mask

        return mask, detections

    def segment_single(
        self,
        image_path: str,
        output_dir: str,
        log: Optional[Callable[[str], None]] = None,
    ) -> SegmentResult:
        """
        Segment a single image file and save mask.

        Returns:
            SegmentResult with details
        """
        def _log(msg):
            if log:
                log(msg)

        image_path = Path(image_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return SegmentResult(
                    image_name=image_path.name,
                    mask_path=None,
                    detections=0,
                    classes_found=[],
                    error=f"Failed to load image: {image_path}"
                )

            # Segment
            mask, detections = self.segment_image(image)

            if detections:
                _log(f"  {image_path.name}: {len(detections)} detections")
                for det in detections:
                    _log(f"    {det['class_name']} conf={det['confidence']:.2f}")
            else:
                _log(f"  {image_path.name}: no detections")

            # Save mask
            mask_name = f"{image_path.stem}_mask.png"
            mask_path = output_path / mask_name
            cv2.imwrite(str(mask_path), mask)

            # Get unique classes found
            classes_found = list(set(d["class_name"] for d in detections))

            return SegmentResult(
                image_name=image_path.name,
                mask_path=str(mask_path),
                detections=len(detections),
                classes_found=classes_found,
            )

        except Exception as e:
            return SegmentResult(
                image_name=image_path.name,
                mask_path=None,
                detections=0,
                classes_found=[],
                error=str(e)
            )

    def segment_batch(
        self,
        input_dir: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> BatchSegmentResult:
        """
        Segment all images in a directory.

        Args:
            input_dir: Input directory with images
            output_dir: Output directory for masks
            progress_callback: Called with (current, total, filename)

        Returns:
            BatchSegmentResult with operation details
        """
        def _log(msg):
            if log:
                log(msg)

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

        if not images:
            return BatchSegmentResult(
                success=False,
                total_images=0,
                images_with_detections=0,
                total_detections=0,
                output_dir=str(output_path),
                errors=["No images found in input directory"]
            )

        results = []
        errors = []
        images_with_detections = 0
        total_detections = 0

        t0 = time.perf_counter()

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, len(images), img_path.name)

            result = self.segment_single(str(img_path), str(output_path), log=log)
            results.append(result)

            if result.error:
                errors.append(f"{img_path.name}: {result.error}")
            elif result.detections > 0:
                images_with_detections += 1
                total_detections += result.detections

        elapsed = time.perf_counter() - t0
        _log(f"Segmentation complete: {len(images)} images in {elapsed:.1f}s")
        _log(f"  With detections: {images_with_detections}/{len(images)}, total: {total_detections}")

        # Class breakdown
        class_counts = {}
        for r in results:
            if r.detections > 0:
                for cls_name in r.classes_found:
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        if class_counts:
            _log(f"  Classes: {', '.join(f'{name}={count}' for name, count in sorted(class_counts.items(), key=lambda x: -x[1]))}")

        return BatchSegmentResult(
            success=len(errors) == 0,
            total_images=len(images),
            images_with_detections=images_with_detections,
            total_detections=total_detections,
            output_dir=str(output_path),
            results=results,
            errors=errors,
        )

    def get_class_names(self, class_ids: List[int]) -> List[str]:
        """Get class names for a list of class IDs."""
        return [COCO_CLASSES.get(cid, f"unknown_{cid}") for cid in class_ids]


def get_class_id(name: str) -> Optional[int]:
    """Get class ID from name."""
    return COCO_NAME_TO_ID.get(name.lower())


def get_class_ids(names: List[str]) -> List[int]:
    """Get class IDs from a list of names."""
    ids = []
    for name in names:
        cid = get_class_id(name)
        if cid is not None:
            ids.append(cid)
    return ids


def list_classes() -> Dict[int, str]:
    """Return all COCO classes."""
    return COCO_CLASSES.copy()


def list_presets() -> Dict[str, List[int]]:
    """Return all class presets."""
    return CLASS_PRESETS.copy()


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLO instance segmentation for mask generation")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output directory for masks")
    parser.add_argument("--model", "-m", default="yolo11n-seg.pt",
                        help="YOLO model (default: yolo11n-seg.pt)")
    parser.add_argument("--classes", "-c", nargs="+", type=str, default=["person"],
                        help="Class names or IDs to detect (default: person)")
    parser.add_argument("--preset", "-p", choices=list(CLASS_PRESETS.keys()),
                        help="Use a class preset instead of --classes")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--dilate", "-d", type=int, default=0,
                        help="Dilate masks by N pixels (default: 0)")
    parser.add_argument("--invert", "-i", action="store_true",
                        help="Invert mask (white background, black objects)")
    parser.add_argument("--device", default="",
                        help="Device to use (auto, cpu, cuda:0)")
    parser.add_argument("--list-classes", action="store_true",
                        help="List all COCO classes and exit")
    parser.add_argument("--list-presets", action="store_true",
                        help="List class presets and exit")

    args = parser.parse_args()

    # List modes
    if args.list_classes:
        print("COCO Classes:")
        for cid, name in sorted(COCO_CLASSES.items()):
            print(f"  {cid:2d}: {name}")
        return 0

    if args.list_presets:
        print("Class Presets:")
        for name, ids in CLASS_PRESETS.items():
            class_names = [COCO_CLASSES[i] for i in ids]
            print(f"  {name}: {', '.join(class_names)}")
        return 0

    if not HAS_YOLO:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return 1

    # Parse classes
    if args.preset:
        class_ids = CLASS_PRESETS[args.preset]
    else:
        class_ids = []
        for c in args.classes:
            try:
                # Try as ID first
                class_ids.append(int(c))
            except ValueError:
                # Try as name
                cid = get_class_id(c)
                if cid is not None:
                    class_ids.append(cid)
                else:
                    print(f"Warning: Unknown class '{c}'")

    if not class_ids:
        class_ids = [0]  # Default to person

    config = SegmentConfig(
        model_name=args.model,
        classes=class_ids,
        confidence=args.confidence,
        dilate_pixels=args.dilate,
        invert_mask=args.invert,
        device=args.device,
    )

    segmenter = Segmenter(config)

    print(f"Classes: {segmenter.get_class_names(class_ids)}")
    print(f"Model: {config.model_name}")

    def progress(curr, total, name):
        print(f"[{curr}/{total}] {name}")

    input_path = Path(args.input)

    if input_path.is_dir():
        result = segmenter.segment_batch(args.input, args.output, progress)
        print(f"\nProcessed {result.total_images} images")
        print(f"Images with detections: {result.images_with_detections}")
        print(f"Total detections: {result.total_detections}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
    else:
        result = segmenter.segment_single(args.input, args.output)
        if result.error:
            print(f"Error: {result.error}")
            return 1
        print(f"Detections: {result.detections}")
        print(f"Classes: {result.classes_found}")
        print(f"Mask: {result.mask_path}")

    return 0


if __name__ == "__main__":
    exit(main())

# Contributing

How to extend the masking pipeline — adding new segmentation models, refinement modules, and shadow detection methods.

## Adding a new segmentation model

This is the most common extension. The system uses a clean plugin pattern: implement `BaseSegmenter`, register in the factory, done.

### Step 1: Add the import check

At the top of `reconstruction_pipeline.py`, add a guarded import alongside the existing ones (~line 50):

```python
try:
    from your_model import YourDetector
    HAS_YOURMODEL = True
except ImportError:
    HAS_YOURMODEL = False
```

This lets the system gracefully handle missing dependencies — your model appears in the auto-fallback chain only when installed.

### Step 2: Add the enum value

In `SegmentationModel` (~line 148):

```python
class SegmentationModel(Enum):
    SAM3 = "sam3"
    YOLO26 = "yolo26"
    RFDETR = "rfdetr"
    FASTSAM = "fastsam"
    YOURMODEL = "yourmodel"       # ← add here
```

### Step 3: Implement the segmenter

Create a subclass of `BaseSegmenter`. Here's the minimum viable implementation:

```python
class YourModelSegmenter(BaseSegmenter):
    """One-line description of your model."""

    def initialize(self):
        """Load model weights onto self.device."""
        if not HAS_YOURMODEL:
            raise ImportError("YourModel not available. Install with: pip install your-model")

        # Load your model — weights auto-download is preferred
        self.model = YourDetector(device=self.device)

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Run detection + segmentation on a single image.

        Args:
            image: BGR uint8 numpy array (H, W, 3)
            prompts: Optional dict — may contain 'remove_prompts' (text),
                     'classes' (COCO IDs), or model-specific keys
            geometry: Image geometry (for preprocessing)

        Returns:
            List of MaskResult — one per detected object
        """
        # 1. Preprocess (handles CLAHE for equirect poles, etc.)
        image_processed = self.preprocess_image(image, geometry)

        # 2. Run your model
        detections = self.model.predict(image_processed)

        # 3. Convert each detection to a MaskResult
        results = []
        for det in detections:
            # Get binary mask — MUST be (H, W) uint8 with values 0 or 1
            mask = det.mask.astype(np.uint8)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)

            # Geometry-aware postprocessing (lightweight — final=False by default)
            mask = self.postprocess_mask(mask, geometry)

            # Quality scoring (inherited from BaseSegmenter)
            confidence = float(det.confidence)
            quality = self._evaluate_mask_quality(mask, confidence)

            results.append(MaskResult(
                mask=mask,
                confidence=confidence,
                quality=quality,
                metadata={
                    'geometry': geometry.value,
                    'model': 'yourmodel',
                    'class': det.class_name,
                    'class_id': det.class_id,
                }
            ))

        return results
```

**Critical requirements:**
- Masks must be `(H, W)` uint8 arrays with values `0` or `1` (not 0/255)
- Masks must match the input image dimensions
- Call `self.preprocess_image()` before detection (handles geometry-specific enhancement)
- Call `self.postprocess_mask()` after detection (handles morphological cleanup)
- Call `self._evaluate_mask_quality()` to score each mask

### Step 4: Register in the factory

In `MaskingPipeline._create_segmenter()` (~line 1595):

```python
elif self.config.model == SegmentationModel.YOURMODEL:
    return YourModelSegmenter(self.config)
```

In `MaskingPipeline._create_segmenter_by_name()` (~line 1608), add to the map:

```python
segmenter_map = {
    'yolo26': (HAS_YOLO, YOLO26Segmenter),
    'rfdetr': (HAS_RFDETR, RFDETRSegmenter),
    'fastsam': (HAS_FASTSAM, FastSAMSegmenter),
    'sam3': (HAS_SAM3, SAM3Segmenter),
    'yourmodel': (HAS_YOURMODEL, YourModelSegmenter),  # ← add here
}
```

### Step 5: Add to auto-fallback (optional)

In `_auto_select_model()` (~line 1575), insert at the appropriate priority level:

```python
elif HAS_YOURMODEL:
    logger.info("Preferred models not available, using YourModel")
    return SegmentationModel.YOURMODEL
```

### That's it

Your model now works with:
- The full pipeline (cubemap decomposition, postprocessing, quality scoring)
- Ensemble mode (add `'yourmodel'` to `ensemble_models` list)
- The GUI (add a radio button or it appears in auto-select)
- Batch processing, video processing, directory processing

## Adding a refinement module

Refinement modules are optional post-segmentation stages. They're loaded lazily in `MaskingPipeline.__init__()` based on `MaskConfig` flags.

### Pattern

1. Create a new file (e.g., `your_refinement.py`) with:
   - A `Config` dataclass for parameters
   - A main class with `initialize()` and a processing method

2. Add a flag to `MaskConfig`:
   ```python
   your_feature: bool = False
   your_feature_config: Optional[dict] = None
   ```

3. Lazy-load in `MaskingPipeline.__init__()`:
   ```python
   if self.config.your_feature:
       from your_refinement import YourRefiner, YourConfig
       cfg = YourConfig(**self.config.your_feature_config) if self.config.your_feature_config else YourConfig()
       self.your_refiner = YourRefiner(cfg)
       self.your_refiner.initialize()
   ```

4. Call in the pipeline at the appropriate stage of `process_image()`.

### Existing refinement modules as examples

| Module | Config flag | When it runs | Input → Output |
|--------|-------------|-------------|----------------|
| `sam_refinement.py` | `sam_refine` | After cubemap merge | Coarse binary mask → refined binary mask |
| `matting.py` | `matting` | Last stage | Binary mask → soft alpha (0.0–1.0) |
| `vos_propagation.py` | `vos_propagation` | Per-frame in sequences | Image + optional mask → propagated mask |
| `shadow_detection.py` | `detect_shadows` | After SAM refine | Image + person mask → expanded shadow mask |
| `colmap_validation.py` | `colmap_validate` | Post-batch | Masks + COLMAP model → consistency scores |

## Adding a shadow detection method

Shadow detection is handled by `shadow_detection.py` which already supports multiple algorithms via `ShadowDetectorType` enum.

To add a new method:

1. Add to the `ShadowDetectorType` enum
2. Implement a detector function that takes `(image, object_mask, config)` → binary shadow mask
3. Register in `ShadowPipeline._get_detector()`

## Code conventions

- **Type hints everywhere** — all function signatures use type hints
- **Dataclasses for config** — every module's parameters are a `@dataclass` with sensible defaults
- **Lazy imports** — optional dependencies are imported inside functions/`__init__`, never at module top level
- **Logging via `logging` module** — use `logger = logging.getLogger(__name__)`; never `print()`
- **Masks are 0/1 uint8** — the single most common bug is treating them as 0/255. Multiply by 255 only at file save time.
- **GUI-independent pipeline** — if your feature can't be tested from Python without the GUI, refactor until it can

## File naming

- Pipeline modules: `snake_case.py` in the root `reconstruction_gui/` directory
- GUI tab builders: `tabs/snake_case_tab.py` with a `build_X_tab(app, parent)` function
- Shared widgets: add to `widgets.py`
- Documentation: `docs/UPPER_CASE.md`

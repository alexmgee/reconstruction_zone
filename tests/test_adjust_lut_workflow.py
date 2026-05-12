import json
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from prep360.core.adjustment_recipe import AdjustmentRecipe
from prep360.core.color_pipeline import apply_adjustment_recipe, load_image_float, write_image_float
from prep360.core.lut import LUTProcessor
from prep360.core.queue_manager import ExtractionSettings
from reconstruction_gui.adjust_workflow import (
    detect_adjust_input,
    ensure_output_ready,
    export_adjusted_dataset,
    validate_adjust_export,
)
from reconstruction_gui.tabs.source_tab import (
    _run_post_processing,
    _validate_adjust_recipe_before_extraction,
)


@pytest.fixture
def workdir(request):
    path = Path("test_tmp") / "adjust_lut_workflow" / f"{request.node.name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_cube(path: Path, size: int, fn) -> Path:
    # .cube spec: B outermost, G middle, R innermost (R changes fastest)
    lines = ['TITLE "test"\n', f"LUT_3D_SIZE {size}\n"]
    for b_i in range(size):
        b = b_i / (size - 1)
        for g_i in range(size):
            g = g_i / (size - 1)
            for r_i in range(size):
                r = r_i / (size - 1)
                out = fn(r, g, b)
                lines.append(f"{out[0]:.8f} {out[1]:.8f} {out[2]:.8f}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _identity_cube(path: Path) -> Path:
    return _write_cube(path, 2, lambda r, g, b: (r, g, b))


def _invert_cube(path: Path) -> Path:
    return _write_cube(path, 2, lambda r, g, b: (1.0 - r, 1.0 - g, 1.0 - b))


def _write_img(path: Path, value=(10, 40, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((3, 4, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return path


def test_lut_identity(workdir):
    lut_path = _identity_cube(workdir / "identity.cube")
    image = np.array([[[0, 64, 255], [25, 128, 200]]], dtype=np.uint8)
    processor = LUTProcessor()
    lut, _ = processor.load_cube(str(lut_path))

    result = processor.apply_uint8(image, lut, 1.0)

    np.testing.assert_allclose(result, image, atol=1)


def test_lut_strength_zero_one_and_half(workdir):
    lut_path = _invert_cube(workdir / "invert.cube")
    image = np.array([[[10, 60, 200]]], dtype=np.uint8)
    processor = LUTProcessor()
    lut, _ = processor.load_cube(str(lut_path))

    zero = processor.apply_uint8(image, lut, 0.0)
    one = processor.apply_uint8(image, lut, 1.0)
    half = processor.apply_uint8(image, lut, 0.5)

    expected_one = 255 - image
    expected_half = np.rint((image.astype(np.float32) + expected_one.astype(np.float32)) / 2).astype(np.uint8)
    np.testing.assert_array_equal(zero, image)
    np.testing.assert_allclose(one, expected_one, atol=1)
    np.testing.assert_allclose(half, expected_half, atol=1)


def test_lut_channel_order_sanity(workdir):
    lut_path = _write_cube(workdir / "shuffle.cube", 2, lambda r, g, b: (b, r, g))
    image = np.array([[[26, 51, 179]]], dtype=np.uint8)  # BGR
    processor = LUTProcessor()

    result = processor.apply_lut_uint8(image, str(lut_path), 1.0)

    # Input RGB is (179, 51, 26). LUT outputs RGB=(26,179,51), so BGR=(51,179,26).
    np.testing.assert_allclose(result[0, 0], np.array([51, 179, 26], dtype=np.uint8), atol=1)


def test_lut_missing_and_malformed(workdir):
    processor = LUTProcessor()
    with pytest.raises(FileNotFoundError):
        processor.load_cube(str(workdir / "missing.cube"))

    bad = workdir / "bad.cube"
    bad.write_text("LUT_3D_SIZE 2\n0 0 nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed LUT data"):
        processor.load_cube(str(bad))


def test_recipe_json_roundtrip_and_validation(workdir):
    lut_path = _identity_cube(workdir / "identity.cube")
    recipe = AdjustmentRecipe(name="Roundtrip")
    recipe.input_lut.enabled = True
    recipe.input_lut.path = str(lut_path)
    recipe.input_lut.strength = 0.5
    recipe.tone.exposure = 0.25
    path = workdir / "adjustment_recipe.json"

    recipe.save(path)
    loaded = AdjustmentRecipe.load(path)

    assert loaded.to_dict() == recipe.to_dict()
    assert loaded.validate() == []

    loaded.input_lut.path = str(workdir / "missing.cube")
    assert any("LUT file not found" in err for err in loaded.validate())


def test_color_pipeline_noop_and_lut_enabled(workdir):
    image = np.random.default_rng(1).random((4, 5, 3), dtype=np.float32)
    noop = AdjustmentRecipe()

    unchanged = apply_adjustment_recipe(image, noop)

    np.testing.assert_allclose(unchanged, image, atol=1e-6)

    lut_path = _invert_cube(workdir / "invert.cube")
    recipe = AdjustmentRecipe()
    recipe.input_lut.enabled = True
    recipe.input_lut.path = str(lut_path)
    changed = apply_adjustment_recipe(image, recipe)

    assert not np.allclose(changed, image)


def test_adjust_paired_folder_detection(workdir):
    root = workdir / "clip_raw"
    _write_img(root / "front" / "frames" / "frame_0001.jpg")
    _write_img(root / "back" / "frames" / "frame_0001.jpg")
    (root / "paired_extraction_manifest.json").write_text("{}", encoding="utf-8")

    dataset = detect_adjust_input(root)

    assert dataset.kind == "paired_extraction"
    assert len(dataset.front_images) == 1
    assert len(dataset.back_images) == 1
    assert dataset.source_manifest.name == "paired_extraction_manifest.json"


def test_adjust_paired_detection_uses_manifest_order(workdir):
    root = workdir / "clip_raw"
    _write_img(root / "front" / "frames" / "z.jpg")
    _write_img(root / "front" / "frames" / "a.jpg")
    _write_img(root / "back" / "frames" / "z.jpg")
    _write_img(root / "back" / "frames" / "a.jpg")
    manifest = {
        "pairs": [
            {"front_image": "front/frames/z.jpg", "back_image": "back/frames/z.jpg"},
            {"front_image": "front/frames/a.jpg", "back_image": "back/frames/a.jpg"},
        ]
    }
    (root / "paired_extraction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = detect_adjust_input(root)

    assert [p.name for p in dataset.front_images] == ["z.jpg", "a.jpg"]
    assert [p.name for p in dataset.back_images] == ["z.jpg", "a.jpg"]


def test_paired_export_count_mismatch_blocks_export(workdir):
    root = workdir / "clip_raw"
    _write_img(root / "front" / "frames" / "frame_0001.jpg")
    _write_img(root / "front" / "frames" / "frame_0002.jpg")
    _write_img(root / "back" / "frames" / "frame_0001.jpg")
    dataset = detect_adjust_input(root)

    errors = validate_adjust_export(dataset, AdjustmentRecipe())

    assert any("count mismatch" in err for err in errors)
    with pytest.raises(ValueError, match="count mismatch"):
        export_adjusted_dataset(dataset, workdir / "out", AdjustmentRecipe())


def test_export_refuses_non_empty_output(workdir):
    out = workdir / "out"
    out.mkdir()
    (out / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        ensure_output_ready(out, refuse_non_empty=True)


def test_cancel_cleanup_removes_written_files(workdir):
    root = workdir / "frames"
    _write_img(root / "a.jpg")
    _write_img(root / "b.jpg")
    dataset = detect_adjust_input(root)
    out = workdir / "out"
    cancel = {"stop": False}

    def progress(_curr, _total, _name):
        cancel["stop"] = True

    result = export_adjusted_dataset(
        dataset,
        out,
        AdjustmentRecipe(),
        cancel_check=lambda: cancel["stop"],
        progress_callback=progress,
    )

    assert result.cancelled
    assert not list(out.rglob("*"))


def test_load_write_float_preserves_16bit_png(workdir):
    src = workdir / "source.png"
    image = np.zeros((2, 3, 3), dtype=np.uint16)
    image[0, 0] = (1024, 32768, 65535)
    assert cv2.imwrite(str(src), image)

    loaded = load_image_float(src)

    assert loaded.source_dtype == "uint16"
    assert loaded.source_max_value == 65535.0
    np.testing.assert_allclose(loaded.image[0, 0], image[0, 0].astype(np.float32) / 65535.0, atol=1e-6)

    recipe = AdjustmentRecipe()
    recipe.output.format = "png"
    recipe.output.bit_depth = "16-bit"
    out = workdir / "out.png"
    write_image_float(out, loaded.image, recipe.output)

    written = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert written.dtype == np.uint16
    np.testing.assert_allclose(written[0, 0], image[0, 0], atol=1)


def test_adjustment_manifest_and_recipe_written(workdir):
    root = workdir / "clip_raw"
    _write_img(root / "front" / "frames" / "frame_0001.jpg")
    _write_img(root / "back" / "frames" / "frame_0001.jpg")
    (root / "paired_extraction_manifest.json").write_text('{"ok": true}', encoding="utf-8")
    dataset = detect_adjust_input(root)
    out = workdir / "clip_adjusted"

    result = export_adjusted_dataset(dataset, out, AdjustmentRecipe())

    assert result.success
    assert (out / "front" / "frames" / "frame_0001.jpg").exists()
    assert (out / "back" / "frames" / "frame_0001.jpg").exists()
    assert (out / "paired_extraction_manifest.json").exists()
    assert (out / "adjustment_recipe.json").exists()
    manifest = json.loads((out / "adjustment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["front_count"] == 1
    assert manifest["back_count"] == 1


def test_extract_post_processing_uses_existing_lut_api(workdir):
    class App:
        cancel_flag = threading.Event()

        def __init__(self):
            self.messages = []

        def log(self, message):
            self.messages.append(message)

    frames = workdir / "frames"
    _write_img(frames / "frame_0001.jpg")
    lut_path = _identity_cube(workdir / "identity.cube")
    settings = ExtractionSettings(lut_enabled=True, lut_path=str(lut_path), lut_strength=1.0)

    stats = _run_post_processing(App(), settings, frames)

    assert stats["after_lut"] == 1
    assert (frames / "frame_0001.jpg").exists()


def test_extract_invalid_saved_recipe_fails_before_extraction(workdir):
    recipe_path = workdir / "recipe.json"
    recipe = AdjustmentRecipe()
    recipe.input_lut.enabled = True
    recipe.input_lut.path = str(workdir / "missing.cube")
    recipe.save(recipe_path)
    settings = ExtractionSettings(
        adjust_recipe_enabled=True,
        adjust_recipe_path=str(recipe_path),
    )

    with pytest.raises(ValueError, match="Adjust recipe is invalid"):
        _validate_adjust_recipe_before_extraction(settings)

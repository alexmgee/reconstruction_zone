"""Tests for prep360 BlurFilter — sharpness scoring and filtering logic."""

import cv2
import numpy as np

from prep360.core.blur_filter import (
    BlurFilter, BlurFilterConfig, BlurFilterResult, BlurScore, _score_image,
)


class TestScoreImage:
    """Test the per-image scoring worker function."""

    def test_sharp_image_scores_higher(self, tmp_dir):
        """A high-frequency pattern should score higher than a flat gray image."""
        # Create a sharp image (checkerboard)
        sharp = np.zeros((128, 128), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp[1::2, 1::2] = 255
        sharp_path = str(tmp_dir / "sharp.png")
        cv2.imwrite(sharp_path, sharp)

        # Create a blurry image (solid gray)
        blurry = np.full((128, 128), 128, dtype=np.uint8)
        blurry_path = str(tmp_dir / "blurry.png")
        cv2.imwrite(blurry_path, blurry)

        _, sharp_score, sharp_metrics = _score_image(sharp_path)
        _, blurry_score, blurry_metrics = _score_image(blurry_path)

        assert sharp_score > blurry_score
        assert sharp_score > 0
        assert blurry_score == 0.0 or blurry_score < sharp_score * 0.01

    def test_returns_all_metrics(self, tmp_dir):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        path = str(tmp_dir / "test.png")
        cv2.imwrite(path, img)

        _, score, metrics = _score_image(path)
        assert "laplacian_var" in metrics
        assert "sobel_mag" in metrics
        assert "brenner" in metrics
        assert score == metrics["laplacian_var"]

    def test_nonexistent_image_returns_zero(self, tmp_dir):
        # ultralytics monkey-patches cv2.imread to raise FileNotFoundError;
        # vanilla OpenCV returns None.  Both are valid outcomes.
        try:
            _, score, metrics = _score_image(str(tmp_dir / "nonexistent.jpg"))
            assert score == 0.0
            assert metrics == {}
        except FileNotFoundError:
            pass  # ultralytics patched cv2.imread — acceptable


class TestBlurFilterAnalyze:
    def test_analyze_single_image(self, tmp_dir):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_dir / "frame.jpg"), img)

        bf = BlurFilter()
        result = bf.analyze_image(str(tmp_dir / "frame.jpg"))
        assert isinstance(result, BlurScore)
        assert result.image_name == "frame.jpg"
        assert result.score >= 0

    def test_analyze_batch_sorted_descending(self, tmp_dir):
        """Results should be sorted sharpest-first."""
        # Create images with varying sharpness
        for i in range(5):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            if i < 2:
                # Blur the first two heavily
                img = cv2.GaussianBlur(img, (31, 31), 10)
            cv2.imwrite(str(tmp_dir / f"frame_{i:03d}.jpg"), img)

        bf = BlurFilter(BlurFilterConfig(workers=1))
        scores = bf.analyze_batch(str(tmp_dir))
        assert len(scores) == 5
        # Verify sorted descending
        for i in range(len(scores) - 1):
            assert scores[i].score >= scores[i + 1].score

    def test_analyze_empty_directory(self, tmp_dir):
        bf = BlurFilter()
        scores = bf.analyze_batch(str(tmp_dir))
        assert scores == []


class TestBlurFilterFilter:
    def _create_test_images(self, tmp_dir, count=10):
        """Create test images with controllable sharpness."""
        for i in range(count):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            if i >= count // 2:
                img = cv2.GaussianBlur(img, (21, 21), 8)
            cv2.imwrite(str(tmp_dir / f"frame_{i:03d}.jpg"), img)

    def test_percentile_filter(self, tmp_dir):
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        self._create_test_images(input_dir, 10)

        bf = BlurFilter()
        result = bf.filter_images(
            str(input_dir), str(output_dir),
            config=BlurFilterConfig(percentile=50, workers=1),
        )
        assert isinstance(result, BlurFilterResult)
        assert result.success is True
        assert result.total_images == 10
        assert result.kept_count > 0
        assert result.rejected_count > 0
        assert result.kept_count + result.rejected_count == 10

    def test_keep_top_n(self, tmp_dir):
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        self._create_test_images(input_dir, 10)

        bf = BlurFilter()
        result = bf.filter_images(
            str(input_dir), str(output_dir),
            config=BlurFilterConfig(keep_top_n=3, workers=1),
        )
        assert result.kept_count == 3

    def test_output_files_exist(self, tmp_dir):
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        self._create_test_images(input_dir, 5)

        bf = BlurFilter()
        result = bf.filter_images(
            str(input_dir), str(output_dir),
            config=BlurFilterConfig(percentile=50, workers=1),
        )
        output_files = list(output_dir.iterdir())
        assert len(output_files) == result.kept_count

    def test_empty_input(self, tmp_dir):
        input_dir = tmp_dir / "empty"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()

        bf = BlurFilter()
        result = bf.filter_images(str(input_dir), str(output_dir))
        assert result.success is True
        assert result.total_images == 0

    def test_score_stats_populated(self, tmp_dir):
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir()
        self._create_test_images(input_dir, 5)

        bf = BlurFilter()
        result = bf.filter_images(
            str(input_dir), str(output_dir),
            config=BlurFilterConfig(percentile=50, workers=1),
        )
        assert "min" in result.score_stats
        assert "max" in result.score_stats
        assert "mean" in result.score_stats
        assert "median" in result.score_stats
        assert "cutoff" in result.score_stats
        assert result.score_stats["max"] >= result.score_stats["min"]

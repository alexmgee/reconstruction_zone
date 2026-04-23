"""Tests for prep360 VideoAnalyzer — parsing helpers and detection logic.

The analyzer wraps ffprobe, which we don't call in tests. Instead we test
the parsing, detection, and recommendation functions with crafted inputs.
"""

from prep360.core.analyzer import (
    VideoAnalyzer, VideoInfo, LOG_FORMATS,
    _safe_float, _safe_int,
)


class TestSafeParsingHelpers:
    def test_safe_float_normal(self):
        assert _safe_float("29.97") == 29.97

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_na(self):
        assert _safe_float("N/A") == 0.0

    def test_safe_float_garbage(self):
        assert _safe_float("not_a_number") == 0.0

    def test_safe_int_normal(self):
        assert _safe_int("1920") == 1920

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_na(self):
        assert _safe_int("N/A") == 0


class TestDetect360:
    """_detect_360 sets is_equirectangular based on 2:1 aspect ratio."""

    def setup_method(self):
        self.analyzer = VideoAnalyzer()

    def _make_info(self, w, h):
        return VideoInfo(
            path="test.mp4", filename="test.mp4", format="mp4",
            codec="h264", width=w, height=h, fps=30.0,
            duration_seconds=60.0, frame_count=1800,
        )

    def test_7680x3840_is_equirect(self):
        info = self._make_info(7680, 3840)
        self.analyzer._detect_360(info)
        assert info.is_equirectangular is True

    def test_5760x2880_is_equirect(self):
        info = self._make_info(5760, 2880)
        self.analyzer._detect_360(info)
        assert info.is_equirectangular is True

    def test_3840x2160_is_not_equirect(self):
        """16:9 is NOT 2:1."""
        info = self._make_info(3840, 2160)
        self.analyzer._detect_360(info)
        assert info.is_equirectangular is False

    def test_1920x1080_is_not_equirect(self):
        info = self._make_info(1920, 1080)
        self.analyzer._detect_360(info)
        assert info.is_equirectangular is False

    def test_zero_dimensions(self):
        info = self._make_info(0, 0)
        self.analyzer._detect_360(info)
        assert info.is_equirectangular is False


class TestDetectLogFormat:
    def setup_method(self):
        self.analyzer = VideoAnalyzer()

    def _make_info(self):
        return VideoInfo(
            path="test.mp4", filename="test.mp4", format="mp4",
            codec="h264", width=7680, height=3840, fps=30.0,
            duration_seconds=60.0, frame_count=1800,
        )

    def test_dlog_in_filename(self):
        from pathlib import Path
        info = self._make_info()
        self.analyzer._detect_log_format(info, Path("DJI_dlog_0001.mp4"))
        assert info.is_log_format is True
        assert info.detected_log_type == "dlog"
        assert info.recommended_lut == LOG_FORMATS["dlog"]["lut"]

    def test_insv_extension(self):
        from pathlib import Path
        info = self._make_info()
        self.analyzer._detect_log_format(info, Path("VID_20240101_120000.insv"))
        assert info.is_log_format is True
        assert info.detected_log_type == "ilog"

    def test_360_extension_gopro(self):
        from pathlib import Path
        info = self._make_info()
        self.analyzer._detect_log_format(info, Path("GS010042.360"))
        assert info.is_log_format is True
        assert info.detected_log_type == "protune"

    def test_no_log_detected(self):
        from pathlib import Path
        info = self._make_info()
        self.analyzer._detect_log_format(info, Path("random_clip.mp4"))
        assert info.is_log_format is False
        assert info.detected_log_type is None


class TestRecommendations:
    def setup_method(self):
        self.analyzer = VideoAnalyzer()

    def test_8k_equirect_gets_2s(self):
        info = VideoInfo(
            path="x", filename="x", format="mp4", codec="h264",
            width=7680, height=3840, fps=30, duration_seconds=60, frame_count=1800,
            is_equirectangular=True,
        )
        self.analyzer._generate_recommendations(info)
        assert info.recommended_interval == 2.0

    def test_6k_equirect_gets_1_5s(self):
        info = VideoInfo(
            path="x", filename="x", format="mp4", codec="h264",
            width=5760, height=2880, fps=30, duration_seconds=60, frame_count=1800,
            is_equirectangular=True,
        )
        self.analyzer._generate_recommendations(info)
        assert info.recommended_interval == 1.5

    def test_4k_equirect_gets_1s(self):
        info = VideoInfo(
            path="x", filename="x", format="mp4", codec="h264",
            width=3840, height=1920, fps=30, duration_seconds=60, frame_count=1800,
            is_equirectangular=True,
        )
        self.analyzer._generate_recommendations(info)
        assert info.recommended_interval == 1.0

    def test_non_equirect_gets_half_second(self):
        info = VideoInfo(
            path="x", filename="x", format="mp4", codec="h264",
            width=3840, height=2160, fps=30, duration_seconds=60, frame_count=1800,
            is_equirectangular=False,
        )
        self.analyzer._generate_recommendations(info)
        assert info.recommended_interval == 0.5


class TestDurationFormatting:
    def setup_method(self):
        self.analyzer = VideoAnalyzer()

    def _make_info(self, duration):
        return VideoInfo(
            path="x", filename="x", format="mp4", codec="h264",
            width=1920, height=1080, fps=30, duration_seconds=duration, frame_count=0,
        )

    def test_seconds_only(self):
        assert self.analyzer.get_duration_formatted(self._make_info(45)) == "0:45"

    def test_minutes_seconds(self):
        assert self.analyzer.get_duration_formatted(self._make_info(125)) == "2:05"

    def test_hours(self):
        assert self.analyzer.get_duration_formatted(self._make_info(3661)) == "1:01:01"

    def test_estimate_frame_count(self):
        info = self._make_info(120.0)
        assert self.analyzer.estimate_frame_count(info, 2.0) == 60


class TestVideoInfoDict:
    def test_to_dict_roundtrip_fields(self):
        info = VideoInfo(
            path="/test.mp4", filename="test.mp4", format="mp4", codec="h264",
            width=7680, height=3840, fps=29.97, duration_seconds=120.5,
            frame_count=3600, bitrate=50_000_000, pixel_format="yuv420p",
            is_equirectangular=True, is_log_format=True,
            detected_log_type="dlog", recommended_lut="DJI_DLog_M_to_Rec709.cube",
        )
        d = info.to_dict()
        assert d["width"] == 7680
        assert d["is_equirectangular"] is True
        assert d["detected_log_type"] == "dlog"
        assert d["recommended_lut"] == "DJI_DLog_M_to_Rec709.cube"

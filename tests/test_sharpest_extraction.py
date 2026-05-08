"""Regression tests for sharpest-frame extraction semantics."""

from prep360.core.paired_split_video_extractor import PairedSplitVideoExtractor
from prep360.core.sharpest_extractor import SharpestExtractor


def test_fractional_fps_seek_range_uses_user_time_windows():
    """A 10s range at fractional FPS should still produce five 2s windows."""
    fps = 49.566668618609825
    range_start_sec, start_frame, end_frame = SharpestExtractor._frame_range_for_seconds(
        fps=fps,
        total_frame_count=2000,
        start_sec=10.0,
        end_sec=20.0,
    )

    assert range_start_sec == 10.0
    assert start_frame == 496
    assert end_frame == 992
    assert start_frame / fps >= 10.0
    assert (end_frame - 1) / fps < 20.0

    window_indices = [
        SharpestExtractor._time_window_index(
            frame_idx,
            fps=fps,
            range_start_sec=range_start_sec,
            interval_sec=2.0,
        )
        for frame_idx in range(start_frame, end_frame)
    ]

    assert sorted(set(window_indices)) == [0, 1, 2, 3, 4]
    assert window_indices[-1] == 4


def test_paired_selection_does_not_emit_fractional_fps_tail_window():
    """Paired selection should not add an extra winner for a rounded tail frame."""
    fps = 49.566668618609825
    range_start_sec, start_frame, end_frame = SharpestExtractor._frame_range_for_seconds(
        fps=fps,
        total_frame_count=2000,
        start_sec=10.0,
        end_sec=20.0,
    )

    paired_entries = []
    for frame_idx in range(start_frame, end_frame):
        window_index = SharpestExtractor._time_window_index(
            frame_idx,
            fps=fps,
            range_start_sec=range_start_sec,
            interval_sec=2.0,
        )
        relative_in_window = frame_idx - start_frame
        paired_entries.append({
            "relative_frame": frame_idx - start_frame,
            "absolute_frame": frame_idx,
            "window_index": window_index,
            # Make the latest frame in each window the deterministic winner.
            "front_score": float(relative_in_window),
            "back_score": float(relative_in_window),
            "scene_change": False,
        })

    selected, front_scores, back_scores = PairedSplitVideoExtractor()._select_from_paired_entries(
        paired_entries,
        scene_aware=False,
    )

    assert len(selected) == 5
    assert selected == [594, 693, 793, 892, 991]
    assert selected[-1] == end_frame - 1
    assert selected[-1] - selected[-2] > 1
    assert front_scores == back_scores


def test_nvdec_failure_hint_explains_dnxhr_fallback():
    hint = SharpestExtractor._nvdec_failure_hint(
        "DNXHR HQX / dnxhd / AVdh, yuv422p10le, 3840x3840, 50/1 fps"
    )

    assert "DNxHD/DNxHR is not supported by NVDEC" in hint
    assert "HEVC/H.265 or H.264" in hint

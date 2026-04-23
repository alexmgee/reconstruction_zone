"""
Sharpness Benchmark — compare Laplacian vs Tenengrad on the same video.

Reads every frame once, scores with both methods, compares which frames
each would select per window, and reports agreement/disagreement.

Usage:
    python -m prep360.core.sharpness_benchmark VIDEO [--interval 0.5] [--scene]
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2

from .sharpest_extractor import SharpestExtractor


def benchmark(
    video_path: str,
    interval: float = 0.5,
    scene_detection: bool = True,
    scene_threshold: float = 0.3,
    scale_width: int = 1920,
    end_sec: float = None,
):
    ext = SharpestExtractor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(int(end_sec * fps), total) if end_sec else total
    window_size = max(1, int(fps * interval))
    duration = end_frame / fps

    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps:.2f}, frames: {end_frame:,}, duration: {duration:.1f}s")
    print(f"Window: {window_size} frames ({interval:.1f}s), scale: {scale_width}px")
    print(f"Scene detection: {scene_detection} (threshold: {scene_threshold})")
    print()

    # Score every frame with both methods in a single pass
    lap_scores: List[Tuple[int, float, bool]] = []
    ten_scores: List[Tuple[int, float, bool]] = []
    prev_small = None
    t0 = time.perf_counter()

    for idx in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if scale_width and w > scale_width:
            scale = scale_width / w
            small = cv2.resize(frame, (scale_width, int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap = ext._laplacian_sharpness(gray)
        ten = ext._tenengrad_sharpness(gray)

        is_scene = False
        if scene_detection and prev_small is not None:
            is_scene = ext._detect_scene_change(prev_small, small, scene_threshold)

        lap_scores.append((idx, lap, is_scene))
        ten_scores.append((idx, ten, is_scene))

        if scene_detection:
            prev_small = small

        if idx % 1000 == 0 and idx > 0:
            elapsed = time.perf_counter() - t0
            pct = idx / end_frame * 100
            speed = idx / elapsed
            print(f"  {pct:.0f}% ({idx:,}/{end_frame:,}) — {speed:.0f} fps", end="\r")

    cap.release()
    scoring_time = time.perf_counter() - t0

    n = len(lap_scores)
    print(f"Scored {n:,} frames in {scoring_time:.1f}s ({n/scoring_time:.0f} fps)        ")
    print()

    # Select winners per window
    def select_winners(scores, scene_aware):
        winners = []
        for i in range(0, len(scores), window_size):
            window = scores[i : i + window_size]
            if scene_aware:
                chunks = ext._split_at_scenes(window)
            else:
                chunks = [window]
            for chunk in chunks:
                best = max(chunk, key=lambda x: x[1])
                winners.append(best[0])
        return winners

    lap_winners = select_winners(lap_scores, scene_detection)
    ten_winners = select_winners(ten_scores, scene_detection)

    # Compare
    lap_set = set(lap_winners)
    ten_set = set(ten_winners)
    agree = lap_set & ten_set
    lap_only = lap_set - ten_set
    ten_only = ten_set - lap_set

    total_unique = len(lap_set | ten_set)
    agreement_pct = len(agree) / total_unique * 100 if total_unique else 100

    print(f"{'':─<60}")
    print(f"  Laplacian selected:  {len(lap_winners)} frames")
    print(f"  Tenengrad selected:  {len(ten_winners)} frames")
    print(f"  Agreement:           {len(agree)}/{total_unique} ({agreement_pct:.1f}%)")
    print(f"  Laplacian-only:      {len(lap_only)}")
    print(f"  Tenengrad-only:      {len(ten_only)}")
    print()

    # Score statistics
    lap_vals = [s for _, s, _ in lap_scores]
    ten_vals = [s for _, s, _ in ten_scores]
    print(f"  Laplacian scores:    min={min(lap_vals):.1f}  max={max(lap_vals):.1f}  mean={sum(lap_vals)/len(lap_vals):.1f}")
    print(f"  Tenengrad scores:    min={min(ten_vals):.1f}  max={max(ten_vals):.1f}  mean={sum(ten_vals)/len(ten_vals):.1f}")

    if scene_detection:
        scene_count = sum(1 for _, _, sc in lap_scores if sc)
        print(f"  Scene changes:       {scene_count}")
    print()

    # Show disagreements with context
    if lap_only or ten_only:
        print(f"Disagreements (showing first 20):")
        print(f"  {'Window':>8}  {'Lap frame':>10}  {'Ten frame':>10}  {'Lap score':>10}  {'Ten score':>10}  {'Delta':>8}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

        # Build window-level comparison
        disagree_windows = []
        for i in range(0, len(lap_scores), window_size):
            lap_win = lap_scores[i : i + window_size]
            ten_win = ten_scores[i : i + window_size]
            lap_best = max(lap_win, key=lambda x: x[1])
            ten_best = max(ten_win, key=lambda x: x[1])
            if lap_best[0] != ten_best[0]:
                win_num = i // window_size
                frame_delta = abs(lap_best[0] - ten_best[0])
                disagree_windows.append((win_num, lap_best, ten_best, frame_delta))

        for win, lap_b, ten_b, delta in disagree_windows[:20]:
            time_sec = win * interval
            print(f"  {time_sec:>7.1f}s  {lap_b[0]:>10}  {ten_b[0]:>10}  {lap_b[1]:>10.1f}  {ten_b[1]:>10.1f}  {delta:>7} fr")

        if len(disagree_windows) > 20:
            print(f"  ... and {len(disagree_windows) - 20} more")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare Laplacian vs Tenengrad sharpness scoring")
    p.add_argument("video", help="Path to video file")
    p.add_argument("--interval", type=float, default=0.5, help="Window interval in seconds (default: 0.5)")
    p.add_argument("--scene", action="store_true", default=True, help="Enable scene detection (default: on)")
    p.add_argument("--no-scene", dest="scene", action="store_false", help="Disable scene detection")
    p.add_argument("--end", type=float, default=None, help="Stop at this many seconds")
    p.add_argument("--scale", type=int, default=1920, help="Analysis width (default: 1920)")
    args = p.parse_args()

    benchmark(args.video, args.interval, args.scene, end_sec=args.end, scale_width=args.scale)

"""
Extraction Benchmark — Compare frame scoring + extraction approaches.

Tests three methods on the same video:
  1. CURRENT: cv2.VideoCapture decode → score all → re-read winners (two-pass)
  2. SINGLE-PASS: cv2.VideoCapture decode → score + buffer winner → write on window close
  3. NVDEC PIPE: ffmpeg hwaccel cuda → pipe grayscale for scoring → OpenCV seek for winners

Usage:
    python benchmarks/extraction_benchmark.py <video_path> [--interval 2.0] [--output ./bench_output]

Outputs:
    - Timing comparison table
    - Extracted frames in separate dirs for visual comparison
    - Score correlation check (do methods select the same frames?)
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


@dataclass
class BenchResult:
    method: str
    total_frames_scored: int
    winners_extracted: int
    scoring_time: float
    extraction_time: float
    total_time: float
    winner_frame_numbers: List[int]


# ── Sharpness scoring ────────────────────────────────────────────────

def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Method 1: Current two-pass approach ──────────────────────────────

def method_current_twopass(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
) -> BenchResult:
    """Replicate current SharpestExtractor: score all, then re-read winners."""
    t_start = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_size = max(1, int(fps * interval))

    # Pass 1: score every frame
    scores: List[Tuple[int, float]] = []
    t_score_start = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > scale_width:
            scale = scale_width / w
            small = cv2.resize(frame, (scale_width, int(h * scale)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_sharpness(gray)
        scores.append((frame_idx, sharpness))
        frame_idx += 1

    cap.release()
    t_score_end = time.perf_counter()

    # Select winners
    best_frames = []
    for i in range(0, len(scores), window_size):
        window = scores[i:i + window_size]
        winner = max(window, key=lambda x: x[1])
        best_frames.append(winner[0])

    # Pass 2: re-open and seek to winners
    t_extract_start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, fnum in enumerate(best_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if ret:
            out_path = output_dir / f"{idx + 1:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    cap.release()
    t_end = time.perf_counter()

    return BenchResult(
        method="current_twopass",
        total_frames_scored=len(scores),
        winners_extracted=len(best_frames),
        scoring_time=t_score_end - t_score_start,
        extraction_time=t_end - t_extract_start,
        total_time=t_end - t_start,
        winner_frame_numbers=best_frames,
    )


# ── Method 2: Single-pass (buffer winner, write on window close) ─────

def method_single_pass(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
) -> BenchResult:
    """Single pass: score + buffer best frame, write when window closes."""
    t_start = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    window_size = max(1, int(fps * interval))

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    window_start = 0
    best_score = -1.0
    best_frame: Optional[np.ndarray] = None
    best_frame_idx = -1
    winners_written = 0
    winner_frame_numbers = []
    total_scored = 0

    t_score_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Score on downscaled version
        h, w = frame.shape[:2]
        if w > scale_width:
            scale = scale_width / w
            small = cv2.resize(frame, (scale_width, int(h * scale)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_sharpness(gray)
        total_scored += 1

        # Update best for current window
        if sharpness > best_score:
            best_score = sharpness
            best_frame = frame  # keep full-res reference
            best_frame_idx = frame_idx

        # Window closed?
        frames_in_window = frame_idx - window_start + 1
        if frames_in_window >= window_size:
            # Write winner
            winners_written += 1
            out_path = output_dir / f"{winners_written:05d}.jpg"
            cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            winner_frame_numbers.append(best_frame_idx)

            # Reset window
            window_start = frame_idx + 1
            best_score = -1.0
            best_frame = None
            best_frame_idx = -1

        frame_idx += 1

    # Flush final partial window
    if best_frame is not None:
        winners_written += 1
        out_path = output_dir / f"{winners_written:05d}.jpg"
        cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        winner_frame_numbers.append(best_frame_idx)

    cap.release()
    t_end = time.perf_counter()

    return BenchResult(
        method="single_pass",
        total_frames_scored=total_scored,
        winners_extracted=winners_written,
        scoring_time=t_end - t_score_start,  # scoring IS the whole thing
        extraction_time=0.0,  # no separate extraction phase
        total_time=t_end - t_start,
        winner_frame_numbers=winner_frame_numbers,
    )


# ── Method 3: NVDEC pipe for scoring + OpenCV seek for winners ───────

def method_nvdec_pipe(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
) -> BenchResult:
    """GPU-accelerated decode via ffmpeg NVDEC pipe, OpenCV seek for winners.

    ffmpeg decodes on GPU and outputs raw grayscale frames at scale_width.
    Python scores from the pipe. After scoring, OpenCV seeks to winners
    for full-res extraction (one seek per window, not per frame).
    """
    t_start = time.perf_counter()

    # Get video info first
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    window_size = max(1, int(fps * interval))

    # Compute output dimensions for scoring
    if orig_w > scale_width:
        scale = scale_width / orig_w
        score_w = scale_width
        score_h = int(orig_h * scale)
    else:
        score_w = orig_w
        score_h = orig_h

    frame_size = score_w * score_h  # grayscale = 1 byte per pixel

    # Launch ffmpeg with NVDEC, output scaled grayscale raw frames
    cmd = [
        "ffmpeg", "-hide_banner",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", f"scale={score_w}:{score_h}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "pipe:1",
    ]

    t_score_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        **_SUBPROCESS_FLAGS,
    )

    # Score frames from pipe
    scores: List[Tuple[int, float]] = []
    frame_idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(score_h, score_w)
        sharpness = laplacian_sharpness(gray)
        scores.append((frame_idx, sharpness))
        frame_idx += 1

    proc.wait()
    t_score_end = time.perf_counter()

    # Select winners
    best_frames = []
    for i in range(0, len(scores), window_size):
        window = scores[i:i + window_size]
        winner = max(window, key=lambda x: x[1])
        best_frames.append(winner[0])

    # Extract winners via OpenCV seek (full resolution)
    t_extract_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    for idx, fnum in enumerate(best_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ret, frame = cap.read()
        if ret:
            out_path = output_dir / f"{idx + 1:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    cap.release()
    t_end = time.perf_counter()

    return BenchResult(
        method="nvdec_pipe",
        total_frames_scored=len(scores),
        winners_extracted=len(best_frames),
        scoring_time=t_score_end - t_score_start,
        extraction_time=t_end - t_extract_start,
        total_time=t_end - t_start,
        winner_frame_numbers=best_frames,
    )


# ── Method 4 (optional): cv2.cudacodec if available ──────────────────

def method_cudacodec(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
) -> Optional[BenchResult]:
    """Single-pass using cv2.cudacodec.VideoReader if available."""
    if not hasattr(cv2, "cudacodec"):
        return None

    t_start = time.perf_counter()

    reader = cv2.cudacodec.createVideoReader(video_path)
    # Get fps from standard VideoCapture (cudacodec doesn't expose it easily)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    window_size = max(1, int(fps * interval))
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    window_start = 0
    best_score = -1.0
    best_frame: Optional[np.ndarray] = None
    best_frame_idx = -1
    winners_written = 0
    winner_frame_numbers = []
    total_scored = 0

    t_score_start = time.perf_counter()

    while True:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break

        # Download from GPU
        frame = gpu_frame.download()
        # cudacodec returns uint16 BGRA — convert to uint8 BGR
        if frame.dtype != np.uint8:
            frame = (frame >> 8).astype(np.uint8)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Score on downscaled
        h, w = frame.shape[:2]
        if w > scale_width:
            scale = scale_width / w
            small = cv2.resize(frame, (scale_width, int(h * scale)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_sharpness(gray)
        total_scored += 1

        if sharpness > best_score:
            best_score = sharpness
            best_frame = frame
            best_frame_idx = frame_idx

        frames_in_window = frame_idx - window_start + 1
        if frames_in_window >= window_size:
            winners_written += 1
            out_path = output_dir / f"{winners_written:05d}.jpg"
            cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            winner_frame_numbers.append(best_frame_idx)
            window_start = frame_idx + 1
            best_score = -1.0
            best_frame = None
            best_frame_idx = -1

        frame_idx += 1

    # Flush final window
    if best_frame is not None:
        winners_written += 1
        out_path = output_dir / f"{winners_written:05d}.jpg"
        cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        winner_frame_numbers.append(best_frame_idx)

    t_end = time.perf_counter()

    return BenchResult(
        method="cudacodec_single_pass",
        total_frames_scored=total_scored,
        winners_extracted=winners_written,
        scoring_time=t_end - t_score_start,
        extraction_time=0.0,
        total_time=t_end - t_start,
        winner_frame_numbers=winner_frame_numbers,
    )


# ── Method 5: cudacodec optimized (GPU resize, minimal PCIe) ─────────

def method_cudacodec_optimized(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
) -> Optional[BenchResult]:
    """Single-pass with GPU resize — only download full-res for winners.

    For scoring: GPU decodes → GPU resizes to scale_width → GPU converts
    to grayscale → download small frame → CPU scores.

    For winners: download the full-res GpuMat only when the window closes.
    """
    if not hasattr(cv2, "cudacodec"):
        return None

    t_start = time.perf_counter()

    reader = cv2.cudacodec.createVideoReader(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    window_size = max(1, int(fps * interval))
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    window_start = 0
    best_score = -1.0
    best_gpu_frame: Optional[cv2.cuda.GpuMat] = None
    best_frame_idx = -1
    winners_written = 0
    winner_frame_numbers = []
    total_scored = 0

    t_score_start = time.perf_counter()

    # Detect color format from first frame
    ret, first_gpu = reader.nextFrame()
    if not ret:
        return BenchResult(
            method="cudacodec_optimized", total_frames_scored=0,
            winners_extracted=0, scoring_time=0, extraction_time=0,
            total_time=0, winner_frame_numbers=[],
        )
    channels = first_gpu.channels()
    color_code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
    bgr_code = cv2.COLOR_BGRA2BGR if channels == 4 else None

    # Process first frame
    def _score_gpu(gpu_frm):
        w, h = gpu_frm.size()
        if w > scale_width:
            gpu_small = cv2.cuda.resize(gpu_frm, (scale_width, int(h * scale_width / w)))
        else:
            gpu_small = gpu_frm
        gpu_gray = cv2.cuda.cvtColor(gpu_small, color_code)
        gray = gpu_gray.download()
        # Convert to uint8 for consistent scoring with CPU methods
        if gray.dtype != np.uint8:
            gray = (gray >> 8).astype(np.uint8)
        return laplacian_sharpness(gray)

    def _save_winner(gpu_frm, path):
        """Download full-res winner, convert to uint8 BGR, write."""
        frame = gpu_frm.download()
        if frame.dtype != np.uint8:
            frame = (frame >> 8).astype(np.uint8)
        if bgr_code is not None:
            frame = cv2.cvtColor(frame, bgr_code)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # Score first frame
    sharpness = _score_gpu(first_gpu)
    best_score = sharpness
    best_gpu_frame = first_gpu.clone()
    best_frame_idx = 0
    total_scored = 1
    frame_idx = 1

    while True:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break

        sharpness = _score_gpu(gpu_frame)
        total_scored += 1

        if sharpness > best_score:
            best_score = sharpness
            best_gpu_frame = gpu_frame.clone()
            best_frame_idx = frame_idx

        # Window closed?
        frames_in_window = frame_idx - window_start + 1
        if frames_in_window >= window_size:
            winners_written += 1
            _save_winner(best_gpu_frame, output_dir / f"{winners_written:05d}.jpg")
            winner_frame_numbers.append(best_frame_idx)

            # Reset window
            window_start = frame_idx + 1
            best_score = -1.0
            best_gpu_frame = None
            best_frame_idx = -1

        frame_idx += 1

    # Flush final partial window
    if best_gpu_frame is not None:
        winners_written += 1
        _save_winner(best_gpu_frame, output_dir / f"{winners_written:05d}.jpg")
        winner_frame_numbers.append(best_frame_idx)

    t_end = time.perf_counter()

    return BenchResult(
        method="cudacodec_optimized",
        total_frames_scored=total_scored,
        winners_extracted=winners_written,
        scoring_time=t_end - t_score_start,
        extraction_time=0.0,
        total_time=t_end - t_start,
        winner_frame_numbers=winner_frame_numbers,
    )


# ── Method 6: cudacodec full GPU (GPU scoring with Tenengrad) ────────

def method_cudacodec_full_gpu(
    video_path: str,
    output_dir: Path,
    interval: float,
    scale_width: int,
    quality: int,
    scoring: str = "tenengrad",
) -> Optional[BenchResult]:
    """Single-pass with GPU decode, GPU resize, GPU scoring.

    The only data crossing PCIe per frame is:
      - A single scalar (score) — 8 bytes
    Full-res frames are only downloaded for the ~170 winners.

    Args:
        scoring: "tenengrad" or "laplacian"
    """
    if not hasattr(cv2, "cudacodec"):
        return None

    t_start = time.perf_counter()
    method_name = f"cudacodec_full_gpu_{scoring}"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    try:
        reader = cv2.cudacodec.createVideoReader(video_path)
    except cv2.error as e:
        print(f"  cudacodec failed: {e}")
        print(f"  (NVDEC H.264 max width is 4096px — this video may exceed it)")
        return None

    window_size = max(1, int(fps * interval))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect format from first frame
    ret, first_gpu = reader.nextFrame()
    if not ret:
        return BenchResult(
            method=method_name, total_frames_scored=0,
            winners_extracted=0, scoring_time=0, extraction_time=0,
            total_time=0, winner_frame_numbers=[],
        )
    channels = first_gpu.channels()
    color_code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
    bgr_code = cv2.COLOR_BGRA2BGR if channels == 4 else None

    # Pre-create GPU filters (reusable across frames)
    if scoring == "tenengrad":
        sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)
    else:
        lap_filter = cv2.cuda.createLaplacianFilter(cv2.CV_32FC1, cv2.CV_32FC1, ksize=1)

    def _prepare_gray(gpu_frm):
        """Resize + grayscale + uint8 conversion on GPU."""
        w, h = gpu_frm.size()
        if w > scale_width:
            gpu_small = cv2.cuda.resize(gpu_frm, (scale_width, int(h * scale_width / w)))
        else:
            gpu_small = gpu_frm
        gpu_gray = cv2.cuda.cvtColor(gpu_small, color_code)
        if gpu_gray.type() != cv2.CV_8UC1:
            gpu_8u = cv2.cuda.GpuMat(gpu_gray.size(), cv2.CV_8UC1)
            gpu_gray.convertTo(cv2.CV_8UC1, gpu_8u, alpha=1.0 / 256.0)
        else:
            gpu_8u = gpu_gray
        return gpu_8u

    def _score_tenengrad(gpu_8u):
        """Tenengrad: Sobel gradient energy mean, all on GPU."""
        gx = sobel_x.apply(gpu_8u)
        gy = sobel_y.apply(gpu_8u)
        gx2 = cv2.cuda.multiply(gx, gx)
        gy2 = cv2.cuda.multiply(gy, gy)
        energy = cv2.cuda.add(gx2, gy2)
        s = cv2.cuda.sum(energy)
        pixel_count = energy.size()[0] * energy.size()[1]
        return s[0] / pixel_count

    def _score_laplacian(gpu_8u):
        """Laplacian variance, all on GPU. Input uint8 → float32 → filter → variance."""
        gpu_32f = cv2.cuda.GpuMat(gpu_8u.size(), cv2.CV_32FC1)
        gpu_8u.convertTo(cv2.CV_32FC1, gpu_32f)
        dst = lap_filter.apply(gpu_32f)
        # Variance = E[x^2] - E[x]^2
        sq = cv2.cuda.multiply(dst, dst)
        sum_sq = cv2.cuda.sum(sq)
        sum_val = cv2.cuda.sum(dst)
        n = dst.size()[0] * dst.size()[1]
        mean = sum_val[0] / n
        return sum_sq[0] / n - mean * mean

    _score_gpu = _score_tenengrad if scoring == "tenengrad" else _score_laplacian

    def _save_winner(gpu_frm, path):
        """Download full-res winner, convert to uint8 BGR, write."""
        frame = gpu_frm.download()
        if frame.dtype != np.uint8:
            frame = (frame >> 8).astype(np.uint8)
        if bgr_code is not None:
            frame = cv2.cvtColor(frame, bgr_code)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

    frame_idx = 0
    window_start = 0
    best_score = -1.0
    best_gpu_frame = None
    best_frame_idx = -1
    winners_written = 0
    winner_frame_numbers = []
    total_scored = 0

    t_score_start = time.perf_counter()

    # Score first frame
    sharpness = _score_gpu(_prepare_gray(first_gpu))
    best_score = sharpness
    best_gpu_frame = first_gpu.clone()
    best_frame_idx = 0
    total_scored = 1
    frame_idx = 1

    while True:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break

        sharpness = _score_gpu(_prepare_gray(gpu_frame))
        total_scored += 1

        if sharpness > best_score:
            best_score = sharpness
            best_gpu_frame = gpu_frame.clone()
            best_frame_idx = frame_idx

        # Window closed?
        frames_in_window = frame_idx - window_start + 1
        if frames_in_window >= window_size:
            winners_written += 1
            _save_winner(best_gpu_frame, output_dir / f"{winners_written:05d}.jpg")
            winner_frame_numbers.append(best_frame_idx)

            window_start = frame_idx + 1
            best_score = -1.0
            best_gpu_frame = None
            best_frame_idx = -1

        frame_idx += 1

    # Flush final partial window
    if best_gpu_frame is not None:
        winners_written += 1
        _save_winner(best_gpu_frame, output_dir / f"{winners_written:05d}.jpg")
        winner_frame_numbers.append(best_frame_idx)

    t_end = time.perf_counter()

    return BenchResult(
        method=method_name,
        total_frames_scored=total_scored,
        winners_extracted=winners_written,
        scoring_time=t_end - t_score_start,
        extraction_time=0.0,
        total_time=t_end - t_start,
        winner_frame_numbers=winner_frame_numbers,
    )


# ── Comparison utilities ─────────────────────────────────────────────

def compare_winners(results: List[BenchResult]):
    """Check how much the methods agree on frame selection."""
    if len(results) < 2:
        return

    print("\n── Winner Agreement ──────────────────────────────────────────")
    baseline = results[0]
    for other in results[1:]:
        if not other.winner_frame_numbers:
            continue
        matches = sum(
            1 for a, b in zip(baseline.winner_frame_numbers, other.winner_frame_numbers)
            if a == b
        )
        total = min(len(baseline.winner_frame_numbers), len(other.winner_frame_numbers))
        pct = matches / total * 100 if total > 0 else 0
        # Also check "within N frames" agreement
        close = sum(
            1 for a, b in zip(baseline.winner_frame_numbers, other.winner_frame_numbers)
            if abs(a - b) <= 2
        )
        print(f"  {baseline.method} vs {other.method}:")
        print(f"    Exact match: {matches}/{total} ({pct:.1f}%)")
        print(f"    Within 2 frames: {close}/{total} ({close/total*100:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark extraction methods")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--interval", type=float, default=2.0, help="Extraction interval in seconds")
    parser.add_argument("--output", default="./bench_output", help="Output directory")
    parser.add_argument("--scale-width", type=int, default=1920, help="Analysis resolution width")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--skip-current", action="store_true", help="Skip the slow two-pass method")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this method (single_pass, nvdec_pipe, cudacodec, cudacodec_optimized, cudacodec_full_gpu)")
    parser.add_argument("--scoring", type=str, default="both", choices=["tenengrad", "laplacian", "both"],
                        help="Scoring method for cudacodec_full_gpu (default: both)")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Error: Video not found: {video}")
        sys.exit(1)

    output_base = Path(args.output)

    # Quick video info
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {video.name}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Interval: {args.interval}s → ~{int(duration / args.interval)} windows")
    print(f"  Analysis width: {args.scale_width}px")
    print(f"  cudacodec available: {hasattr(cv2, 'cudacodec')}")
    print()

    results: List[BenchResult] = []
    only = args.only

    def should_run(name):
        return only is None or only == name

    # Method 1: Current two-pass
    if should_run("current_twopass") and not args.skip_current:
        print("Running: current_twopass (score all → re-read winners)...")
        r = method_current_twopass(
            str(video), output_base / "current", args.interval, args.scale_width, args.quality)
        results.append(r)
        print(f"  Done: {r.total_time:.1f}s (score: {r.scoring_time:.1f}s, extract: {r.extraction_time:.1f}s)")
    elif should_run("current_twopass"):
        print("Skipping: current_twopass (--skip-current)")

    # Method 2: Single-pass
    if should_run("single_pass"):
        print("Running: single_pass (score + buffer + write inline)...")
        r = method_single_pass(
            str(video), output_base / "single_pass", args.interval, args.scale_width, args.quality)
        results.append(r)
        print(f"  Done: {r.total_time:.1f}s")

    # Method 3: NVDEC pipe
    if should_run("nvdec_pipe"):
        print("Running: nvdec_pipe (ffmpeg GPU decode → pipe → OpenCV seek for winners)...")
        r = method_nvdec_pipe(
            str(video), output_base / "nvdec_pipe", args.interval, args.scale_width, args.quality)
        results.append(r)
        print(f"  Done: {r.total_time:.1f}s (score: {r.scoring_time:.1f}s, extract: {r.extraction_time:.1f}s)")

    # Method 4: cudacodec (if available)
    if should_run("cudacodec") and hasattr(cv2, "cudacodec"):
        print("Running: cudacodec_single_pass (GPU decode + single-pass)...")
        r = method_cudacodec(
            str(video), output_base / "cudacodec", args.interval, args.scale_width, args.quality)
        if r:
            results.append(r)
            print(f"  Done: {r.total_time:.1f}s")
    elif should_run("cudacodec"):
        print("Skipping: cudacodec (not available)")

    # Method 5: cudacodec optimized (GPU resize, minimal PCIe transfer)
    if should_run("cudacodec_optimized") and hasattr(cv2, "cudacodec"):
        print("Running: cudacodec_optimized (GPU resize + score, download only winners)...")
        r = method_cudacodec_optimized(
            str(video), output_base / "cudacodec_opt", args.interval, args.scale_width, args.quality)
        if r:
            results.append(r)
            print(f"  Done: {r.total_time:.1f}s")
    elif should_run("cudacodec_optimized"):
        print("Skipping: cudacodec_optimized (not available)")

    # Method 6: cudacodec full GPU scoring
    if should_run("cudacodec_full_gpu") and hasattr(cv2, "cudacodec"):
        scorers = []
        if args.scoring in ("tenengrad", "both"):
            scorers.append("tenengrad")
        if args.scoring in ("laplacian", "both"):
            scorers.append("laplacian")
        for scorer in scorers:
            print(f"Running: cudacodec_full_gpu_{scorer} (GPU decode + GPU {scorer} scoring)...")
            r = method_cudacodec_full_gpu(
                str(video), output_base / f"cudacodec_full_gpu_{scorer}",
                args.interval, args.scale_width, args.quality, scoring=scorer)
            if r:
                results.append(r)
                print(f"  Done: {r.total_time:.1f}s")
    elif should_run("cudacodec_full_gpu"):
        print("Skipping: cudacodec_full_gpu (not available)")

    # Summary table
    print("\n── Results ───────────────────────────────────────────────────")
    print(f"{'Method':<35} {'Scored':>7} {'Extracted':>9} {'Score(s)':>9} {'Extract(s)':>11} {'Total(s)':>9} {'vs baseline':>11}")
    print("─" * 95)
    baseline_time = results[0].total_time if results else 1.0
    for r in results:
        speedup = baseline_time / r.total_time if r.total_time > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r.method != results[0].method else "baseline"
        print(f"{r.method:<35} {r.total_frames_scored:>7} {r.winners_extracted:>9} "
              f"{r.scoring_time:>9.1f} {r.extraction_time:>11.1f} {r.total_time:>9.1f} {speedup_str:>11}")

    compare_winners(results)

    print(f"\nOutput frames in: {output_base.resolve()}")
    print("Compare visually: each method's folder should contain identical (or near-identical) frame selections.")


if __name__ == "__main__":
    main()

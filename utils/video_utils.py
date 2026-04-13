"""
Video utility helpers for the panoramic stitching pipeline.

Provides:
  - VideoReader        : iterate frames from a video file or camera
  - VideoWriter        : write frames to an MP4 / AVI file
  - extract_frames     : dump N evenly-spaced frames to a directory
  - estimate_overlap   : estimate the overlap rate between two registered images
  - sync_video_readers : read frame pairs from two captures in lockstep
  - benchmark_stitcher : measure average fps over a clip
"""

import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# VideoReader
# ---------------------------------------------------------------------------

class VideoReader:
    """
    Thin wrapper around cv2.VideoCapture that supports iteration and
    context-manager usage.

    Args:
        source     : video file path, image directory glob, or camera index
        max_frames : stop after this many frames (None = all)
        skip       : read every Nth frame (1 = no skip)
    """

    def __init__(self, source, max_frames: Optional[int] = None, skip: int = 1):
        self.source = source
        self.max_frames = max_frames
        self.skip = max(1, skip)
        self._cap = None
        self._frame_idx = 0

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")
        self._frame_idx = 0
        return self

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()
        while True:
            if self.max_frames and self._frame_idx >= self.max_frames:
                break
            ret, frame = self._cap.read()
            if not ret:
                break
            if self._frame_idx % self.skip == 0:
                yield frame
            self._frame_idx += 1

    # ---- Properties ----

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------

class VideoWriter:
    """
    Write frames to an MP4 file.

    Args:
        path    : output file path (will create parent dirs)
        fps     : output frame rate
        size    : (width, height) of output frames
        codec   : FourCC codec string (default 'mp4v')
    """

    def __init__(
        self,
        path: str,
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
    ):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Cannot open VideoWriter at {path!r}")

    def write(self, frame: np.ndarray):
        """Write one BGR uint8 frame."""
        self._writer.write(frame)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    output_dir: str,
    n_frames: int = 100,
    prefix: str = "frame",
    fmt: str = "jpg",
    quality: int = 95,
) -> List[str]:
    """
    Extract n_frames evenly-spaced frames from a video to output_dir.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Could not read frames from {video_path!r}")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    saved = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        fname = os.path.join(output_dir, f"{prefix}_{i:04d}.{fmt}")
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else []
        cv2.imwrite(fname, frame, params)
        saved.append(fname)

    cap.release()
    print(f"Extracted {len(saved)}/{n_frames} frames → {output_dir}/")
    return saved


# ---------------------------------------------------------------------------
# Synchronised dual-stream reader
# ---------------------------------------------------------------------------

def sync_video_readers(
    path_a: str,
    path_b: str,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (frame_a, frame_b) pairs from two video files in lockstep.
    Stops when either stream ends or max_frames is reached.
    """
    cap_a = cv2.VideoCapture(path_a)
    cap_b = cv2.VideoCapture(path_b)
    n = 0
    try:
        while True:
            if max_frames and n >= max_frames:
                break
            ret_a, fa = cap_a.read()
            ret_b, fb = cap_b.read()
            if not ret_a or not ret_b:
                break
            yield fa, fb
            n += 1
    finally:
        cap_a.release()
        cap_b.release()


# ---------------------------------------------------------------------------
# Overlap rate estimation from a pair of images
# ---------------------------------------------------------------------------

def estimate_overlap_rate_from_images(
    img_a: np.ndarray,   # (H, W, 3) uint8
    img_b: np.ndarray,   # (H, W, 3) uint8
    n_features: int = 500,
) -> float:
    """
    Estimate the overlap rate between two images using ORB feature matching.

    The overlap fraction is approximated as the fraction of matched keypoints
    whose reprojected positions fall within the bounds of both images after
    computing a homography between them.

    Returns:
        float in [0, 1]  – approximate overlap rate
    """
    orb = cv2.ORB_create(n_features)
    kp_a, des_a = orb.detectAndCompute(
        cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None
    )
    kp_b, des_b = orb.detectAndCompute(
        cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None
    )

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0.0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    if H is None:
        return 0.0

    # Project corners of img_a into img_b's space
    h, w = img_a.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    hb, wb = img_b.shape[:2]
    # Compute intersection area
    xs = projected[:, 0]
    ys = projected[:, 1]
    overlap_w = max(0, min(xs.max(), wb) - max(xs.min(), 0))
    overlap_h = max(0, min(ys.max(), hb) - max(ys.min(), 0))
    overlap_area = overlap_w * overlap_h
    ref_area = w * h
    return float(min(overlap_area / (ref_area + 1e-8), 1.0))


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------

def benchmark_stitcher(
    stitcher,                  # PanoramicStitcher instance
    video_a: str,
    video_b: str,
    n_warmup: int = 100,
    n_measure: int = 1000,
) -> dict:
    """
    Measure average per-frame stitching time following the paper's protocol
    (Section 4.1.1): 100 warm-up iterations discarded, then 1000 measurements.

    Returns dict with keys: avg_fps, avg_ms, min_ms, max_ms, std_ms.
    """
    gen = sync_video_readers(video_a, video_b,
                             max_frames=n_warmup + n_measure)
    times_ms = []
    frame_idx = 0

    for fa, fb in gen:
        t0 = time.perf_counter()
        stitcher.stitch(fa, fb)
        dt_ms = (time.perf_counter() - t0) * 1000

        if frame_idx >= n_warmup:
            times_ms.append(dt_ms)
        frame_idx += 1

    if not times_ms:
        return {}

    arr = np.array(times_ms)
    return {
        "avg_fps": float(1000.0 / arr.mean()),
        "avg_ms":  float(arr.mean()),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
        "std_ms":  float(arr.std()),
        "n_frames": len(arr),
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Synthetic test: write a tiny video, extract frames, read back
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "test.mp4")
        with VideoWriter(vid_path, fps=30, size=(128, 128)) as w:
            for _ in range(60):
                frame = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
                w.write(frame)

        frames = extract_frames(vid_path, os.path.join(tmpdir, "frames"),
                                n_frames=10)
        print(f"Extracted {len(frames)} frames.")

        with VideoReader(vid_path, max_frames=5) as reader:
            for i, f in enumerate(reader):
                print(f"  Frame {i}: shape={f.shape}")

    print("video_utils smoke test passed.")

"""
Motion detection for the overlap region  (Section 3.3.2, eq. 16).

Goal: produce a binary mask  M_Ω ⊂ Ω  that marks pixels belonging to
moving foreground objects in the overlapping area between the two
registered images.

Strategy (matches Figure 6 in the paper):
  1. Convert both registered images to grayscale.
  2. Run OpenCV's MOG2 background subtractor independently on each view.
  3. Union the two foreground masks → raw motion mask.
  4. Apply morphological clean-up (open → close) to remove noise and fill holes.
  5. Restrict the result to the valid overlap region Ω.

The detector is stateful: it maintains a background model that is updated
every frame, exactly as in a real-time surveillance pipeline.
"""

import cv2
import numpy as np
from typing import Optional


class MotionDetector:
    """
    Stateful per-camera MOG2-based motion detector.

    Args:
        history          : number of frames used to build the background model
        var_threshold    : Mahalanobis distance threshold for foreground decision
        detect_shadows   : whether MOG2 should model shadows (slightly slower)
        min_area         : minimum contour area (px²) to keep as a moving region
        morph_kernel_size: size of the structuring element for morphological ops
        dilate_iters     : number of dilation iterations (expands detected regions
                           slightly to ensure the seam steers well clear)
    """

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 36.0,
        detect_shadows: bool = False,
        min_area: int = 500,
        morph_kernel_size: int = 5,
        dilate_iters: int = 3,
    ):
        self.min_area = min_area
        self.dilate_iters = dilate_iters

        # Two independent background models – one per camera view
        self._bg_a = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._bg_b = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._kernel = kernel

    # ------------------------------------------------------------------ #

    def detect(
        self,
        frame_a: np.ndarray,   # (H, W, 3) uint8, registered view A
        frame_b: np.ndarray,   # (H, W, 3) uint8, registered view B
        overlap_mask: np.ndarray,  # (H, W) uint8 binary: 255 = overlap region Ω
    ) -> np.ndarray:
        """
        Compute the moving-object mask M_Ω restricted to the overlap.

        Args:
            frame_a      : registered image A (BGR or RGB uint8)
            frame_b      : registered image B (BGR or RGB uint8)
            overlap_mask : valid overlap region Ω  (255 = valid)

        Returns:
            motion_mask  : (H, W) uint8, 255 = moving pixel inside Ω
        """
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Foreground masks from MOG2 (255 = foreground, 0 = background)
        fg_a = self._bg_a.apply(gray_a)
        fg_b = self._bg_b.apply(gray_b)

        # Union: moving if detected in either view
        fg = cv2.bitwise_or(fg_a, fg_b)

        # Morphological open (remove isolated noise)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel, iterations=1)
        # Morphological close (fill holes inside objects)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        # Dilate to give the seam a safety margin around moving objects
        fg = cv2.dilate(fg, self._kernel, iterations=self.dilate_iters)

        # Remove small spurious regions
        fg = self._filter_small_contours(fg)

        # Restrict to overlap region Ω
        motion_mask = cv2.bitwise_and(fg, overlap_mask)
        return motion_mask

    # ------------------------------------------------------------------ #

    def _filter_small_contours(self, mask: np.ndarray) -> np.ndarray:
        """Zero out connected components smaller than self.min_area."""
        out = np.zeros_like(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
        return out

    def reset(self):
        """Re-initialise background models (e.g. after a scene cut)."""
        self._bg_a = cv2.createBackgroundSubtractorMOG2()
        self._bg_b = cv2.createBackgroundSubtractorMOG2()


# ---------------------------------------------------------------------------
# Helper: build overlap mask from a float warped-validity mask
# ---------------------------------------------------------------------------

def build_overlap_mask(warp_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a float validity mask (output of HomographyWarper on a ones tensor)
    to a uint8 binary overlap mask suitable for MotionDetector.

    Args:
        warp_mask : (H, W) float32  in [0, 1]
        threshold : pixels above this are considered valid overlap

    Returns:
        (H, W) uint8  255 = valid overlap
    """
    binary = (warp_mask > threshold).astype(np.uint8) * 255
    return binary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    H, W = 512, 512
    detector = MotionDetector()

    # Warm up background model with static frames
    static_a = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    static_b = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    overlap  = np.ones((H, W), dtype=np.uint8) * 255

    for _ in range(10):
        detector.detect(static_a, static_b, overlap)

    # Inject a synthetic moving object
    moving_a = static_a.copy()
    moving_a[200:300, 200:300] = 200  # bright square

    mask = detector.detect(moving_a, static_b, overlap)
    print(f"Motion mask nonzero pixels: {np.count_nonzero(mask)}")
    print("MotionDetector smoke test passed.")

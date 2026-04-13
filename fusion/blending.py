"""
Image blending / compositing strategies  (Section 3.3, Section 4.2).

Two modes are implemented:

1. WeightedAverageBlender
   Simple linear alpha blending in the overlap region.  Used as the
   default fast path when no moving objects are detected (and also as
   a baseline in the paper's comparison, Table 5).

2. SeamBlender
   Hard-cut compositing guided by the label map produced by
   GraphCutSeamFinder.  Pixels labelled 0 come from img0, pixels
   labelled 1 from img1.  A narrow feathering strip along the seam
   boundary prevents hard colour discontinuities.

Both blenders accept float32 images in [0, 1] and return float32 [0, 1].
"""

import numpy as np
import cv2
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Weighted Average Blender
# ---------------------------------------------------------------------------

class WeightedAverageBlender:
    """
    Linear alpha blend in the overlap; outside the overlap each image
    contributes fully.

    alpha_0(p) = dist_to_right_edge / overlap_width   (ramps 1→0 left-to-right)
    alpha_1(p) = 1 − alpha_0(p)
    """

    def blend(
        self,
        img0: np.ndarray,           # (H, W, 3) float32
        img1: np.ndarray,           # (H, W, 3) float32
        overlap_mask: np.ndarray,   # (H, W) uint8, 255 = overlap
        canvas_size: Optional[tuple] = None,  # (H_out, W_out) if panorama canvas
    ) -> np.ndarray:
        """
        Returns blended image (H, W, 3) float32.
        """
        H, W = img0.shape[:2]
        overlap = (overlap_mask > 0)

        # Build horizontal distance-based alpha for the overlap region
        alpha = np.zeros((H, W), dtype=np.float32)
        for r in range(H):
            cols = np.where(overlap[r])[0]
            if len(cols) < 2:
                continue
            left, right = cols[0], cols[-1]
            width = right - left
            if width == 0:
                continue
            # alpha for img0: 1 at left edge, 0 at right edge
            alpha[r, left:right + 1] = np.linspace(1.0, 0.0, right - left + 1)

        alpha3 = alpha[:, :, np.newaxis]

        # Inside overlap: blend; outside: take whichever is valid
        result = np.where(
            overlap[:, :, np.newaxis],
            alpha3 * img0 + (1.0 - alpha3) * img1,
            img0 + img1,   # outside overlap one of them is 0 (black-padded)
        )
        return result.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Seam-based Hard-Cut Blender (with feathering)
# ---------------------------------------------------------------------------

class SeamBlender:
    """
    Composites img0 and img1 according to a seam label map:
        output(p) = img0(p) if labels(p) == 0
                  = img1(p) if labels(p) == 1

    A small feathering band (Gaussian-blurred distance from the seam)
    softens the transition to avoid visible colour discontinuities.

    Args:
        feather_width : half-width of the feathering band in pixels
                        (set to 0 to disable feathering)
    """

    def __init__(self, feather_width: int = 8):
        self.feather_width = feather_width

    def blend(
        self,
        img0: np.ndarray,           # (H, W, 3) float32
        img1: np.ndarray,           # (H, W, 3) float32
        labels: np.ndarray,         # (H, W) int32 {0, 1}
        overlap_mask: np.ndarray,   # (H, W) uint8  255 = Ω
    ) -> np.ndarray:
        """
        Returns composited panoramic image (H, W, 3) float32.
        """
        H, W = img0.shape[:2]
        overlap = (overlap_mask > 0)

        if self.feather_width > 0:
            alpha = self._feather_alpha(labels, overlap)
        else:
            # Hard cut
            alpha = (labels == 0).astype(np.float32)

        alpha3 = alpha[:, :, np.newaxis]

        # Inside overlap: use seam-guided blend
        # Outside overlap: full contribution from the respective image
        result = np.where(
            overlap[:, :, np.newaxis],
            alpha3 * img0 + (1.0 - alpha3) * img1,
            img0 + img1,   # one of them is zero outside overlap
        )
        return result.clip(0.0, 1.0).astype(np.float32)

    def _feather_alpha(
        self,
        labels: np.ndarray,
        overlap: np.ndarray,
    ) -> np.ndarray:
        """
        Build a smooth alpha channel that transitions from 1 (img0 side) to 0
        (img1 side) over a band of width `feather_width` around the seam.
        """
        H, W = labels.shape
        fw = self.feather_width

        # Binary mask: 1 where label == 0 (img0 region)
        mask0 = (labels == 0).astype(np.float32)

        # Distance transform from the seam in both directions
        # (signed distance: positive in img0 territory, negative in img1)
        dist_to_seam = cv2.distanceTransform(
            mask0.astype(np.uint8) * 255, cv2.DIST_L2, 3
        ).astype(np.float32)
        dist_to_seam1 = cv2.distanceTransform(
            ((1 - mask0) * 255).astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float32)

        # Signed distance: positive = img0 side
        signed = dist_to_seam - dist_to_seam1

        # Map to [0,1] over [-fw, fw]
        alpha = ((signed + fw) / (2.0 * fw)).clip(0.0, 1.0)

        # Apply only inside overlap; elsewhere use hard labels
        hard = mask0.copy()
        alpha = np.where(overlap, alpha, hard)
        return alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Panorama canvas compositor
# ---------------------------------------------------------------------------

def composite_panorama(
    img0: np.ndarray,       # full registered view 0  (H, W, 3) float32
    img1: np.ndarray,       # full registered view 1  (H, W, W) float32
    labels: np.ndarray,     # (H, W_overlap, ...) seam labels  OR  None
    overlap_mask: np.ndarray,
    blender: Optional[SeamBlender] = None,
) -> np.ndarray:
    """
    Convenience wrapper: blend and return the final panoramic frame.
    If labels is None, falls back to weighted average blending.
    """
    if labels is None or blender is None:
        return WeightedAverageBlender().blend(img0, img1, overlap_mask)
    return blender.blend(img0, img1, labels, overlap_mask)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    H, W = 256, 512

    img0 = np.random.rand(H, W, 3).astype(np.float32)
    img1 = np.random.rand(H, W, 3).astype(np.float32)

    overlap = np.zeros((H, W), dtype=np.uint8)
    overlap[:, W//4: 3*W//4] = 255

    # Fake label map (left half = 0, right half = 1 within overlap)
    labels = np.zeros((H, W), dtype=np.int32)
    labels[:, W//2:] = 1

    # Weighted average
    wa_blender = WeightedAverageBlender()
    wa_out = wa_blender.blend(img0, img1, overlap)
    assert wa_out.shape == (H, W, 3), "WA blender shape mismatch"
    print(f"WeightedAverage output: {wa_out.shape}  range [{wa_out.min():.3f}, {wa_out.max():.3f}]")

    # Seam blender
    seam_blender = SeamBlender(feather_width=8)
    seam_out = seam_blender.blend(img0, img1, labels, overlap)
    assert seam_out.shape == (H, W, 3), "Seam blender shape mismatch"
    print(f"SeamBlender output    : {seam_out.shape}  range [{seam_out.min():.3f}, {seam_out.max():.3f}]")

    print("Blending smoke test passed.")

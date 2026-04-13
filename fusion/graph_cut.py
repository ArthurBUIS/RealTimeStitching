"""
Motion-aware graph cut seam search  (Section 3.3).

Implements the energy functions from equations (6) and (7):

    E_first(l) = α·ΣEs(p,q) + β·ΣEg(p,q)          over N(Ω)
    E_sub(l)   = α·ΣEs(p,q) + β·ΣEg(p,q)          over N(Ω ∖ M_Ω)

with α = β = 1 (paper default).

Terms:
    Es(p,q) – smoothness:  |lp−lq| · (Id(p) + Id(q))          eq. (9,10)
    Eg(p,q) – gradient  :  |lp−lq| · (W(p) + W(q))            eq. (11-14)

where
    Id(p) = ||I0(p) − I1(p)||²_2                               eq. (10)
    W(p)  = σ( Wx(p) + Wy(p) )                                 eq. (12)
    Wx/Wy – squared Sobel responses of the image difference     eq. (13,14)

The minimum cut is found with the PyMaxflow library, which provides an
efficient Python wrapper around the Boykov-Kolmogorov max-flow algorithm.

Install:  pip install PyMaxflow
Fallback: if PyMaxflow is unavailable we fall back to a simple scanline
          dynamic-programming seam (much faster, lower quality).
"""

import numpy as np
import cv2
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Attempt to import PyMaxflow; fall back gracefully
# ---------------------------------------------------------------------------

try:
    import maxflow
    _MAXFLOW_AVAILABLE = True
except ImportError:
    _MAXFLOW_AVAILABLE = False
    import warnings
    warnings.warn(
        "PyMaxflow not found. Install with: pip install PyMaxflow\n"
        "Falling back to dynamic-programming seam search.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Energy-map utilities
# ---------------------------------------------------------------------------

def _color_diff_map(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """
    Id(p) = ||I0(p) − I1(p)||²_2  (eq. 10), normalised to [0,1].

    Args:
        img0, img1 : (H, W, 3) float32 in [0, 1]
    Returns:
        (H, W) float32
    """
    diff = img0.astype(np.float32) - img1.astype(np.float32)
    return (diff ** 2).sum(axis=-1)   # sum over colour channels


def _gradient_weight_map(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """
    W(p) = sigmoid( Wx(p) + Wy(p) )  (eq. 12)

    Wx(p) = [Sx * (I_lp - I_lq)](p)²   (eq. 13)
    Wy(p) = [Sy * (I_lp - I_lq)](p)²   (eq. 14)

    Sobel kernels used: Sx as in eq. (15) – note the paper uses a
    non-standard 3×3 kernel. We implement it exactly as written:
        Sx = [[-2,0,2],[-1,0,1],[-2,0,2]]
        Sy = [[-2,-1,-2],[0,0,0],[2,1,2]]

    Args:
        img0, img1 : (H, W, 3) float32 in [0, 1]
    Returns:
        (H, W) float32
    """
    # Paper's custom Sobel kernels (eq. 15)
    Sx = np.array([[-2, 0, 2],
                   [-1, 0, 1],
                   [-2, 0, 2]], dtype=np.float32)
    Sy = np.array([[-2, -1, -2],
                   [ 0,  0,  0],
                   [ 2,  1,  2]], dtype=np.float32)

    # Work on grayscale difference
    diff_gray = cv2.cvtColor(
        ((img0 - img1) * 255).clip(0, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    wx = cv2.filter2D(diff_gray, -1, Sx) ** 2
    wy = cv2.filter2D(diff_gray, -1, Sy) ** 2

    w = wx + wy
    # Sigmoid
    w = 1.0 / (1.0 + np.exp(-w))
    return w.astype(np.float32)


def _edge_weight_map(
    img0: np.ndarray,
    img1: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Combined per-pixel weight used for graph edges:
        cost(p) = α·Id(p) + β·W(p)

    For a boundary (p,q): edge_cost = cost(p) + cost(q)  (eqs 9, 11).
    """
    Id = _color_diff_map(img0, img1)
    W  = _gradient_weight_map(img0, img1)
    return alpha * Id + beta * W


# ---------------------------------------------------------------------------
# PyMaxflow-based graph cut seam search
# ---------------------------------------------------------------------------

def _find_seam_graphcut(
    cost_map: np.ndarray,       # (H, W) float32 – per-pixel cost
    valid_mask: np.ndarray,     # (H, W) uint8   – 255 = node belongs to graph
) -> np.ndarray:
    """
    Build a graph over valid pixels and find the min-cut seam using
    Boykov-Kolmogorov max-flow (via PyMaxflow).

    The seam separates the left portion (label 0, source side) from the
    right portion (label 1, sink side) in the overlap region.

    Returns:
        labels : (H, W) int32  with values {0, 1}
    """
    H, W = cost_map.shape
    valid = (valid_mask > 0)

    g = maxflow.Graph[float](H * W, H * W * 2)
    nodes = g.add_nodes(H * W)

    # ---- Terminal (source/sink) capacities --------------------------------
    # Source (label 0) is connected to the LEFT column of the overlap.
    # Sink   (label 1) is connected to the RIGHT column of the overlap.
    INF = 1e9
    for r in range(H):
        for c in range(W):
            if not valid[r, c]:
                continue
            idx = r * W + c
            # Find leftmost / rightmost valid pixel in this row
            row_valid = np.where(valid[r])[0]
            if len(row_valid) == 0:
                continue
            left_c  = row_valid[0]
            right_c = row_valid[-1]
            src_cap  = INF if c == left_c  else 0.0
            sink_cap = INF if c == right_c else 0.0
            g.add_tedge(idx, src_cap, sink_cap)

    # ---- Neighbour (n-link) capacities ------------------------------------
    # 4-connectivity; weight = cost(p) + cost(q)
    for r in range(H):
        for c in range(W):
            if not valid[r, c]:
                continue
            idx = r * W + c
            cp = cost_map[r, c]

            # Right neighbour
            if c + 1 < W and valid[r, c + 1]:
                cap = cp + cost_map[r, c + 1]
                g.add_edge(idx, idx + 1, cap, cap)

            # Down neighbour
            if r + 1 < H and valid[r + 1, c]:
                cap = cp + cost_map[r + 1, c]
                g.add_edge(idx, idx + W, cap, cap)

    g.maxflow()

    segments = g.get_grid_segments(nodes).reshape(H, W).astype(np.int32)
    # get_grid_segments: True = source side (label 0), False = sink (label 1)
    labels = 1 - segments   # flip so left=0, right=1
    # Zero out invalid pixels
    labels[~valid] = 0
    return labels


# ---------------------------------------------------------------------------
# DP fallback seam search (scanline, O(H·W))
# ---------------------------------------------------------------------------

def _find_seam_dp(
    cost_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Dynamic-programming vertical seam search (fallback when PyMaxflow
    is not installed).  Finds the lowest-cost vertical path from top to
    bottom of the overlap region.

    Returns:
        labels : (H, W) int32 {0, 1}  – pixels left of seam = 0, right = 1
    """
    H, W = cost_map.shape
    valid = (valid_mask > 0).astype(np.float32)
    BIG = 1e9

    cost = cost_map.copy()
    cost[~valid.astype(bool)] = BIG

    # Cumulative cost table
    dp = cost.copy()
    for r in range(1, H):
        for c in range(W):
            if not valid[r, c]:
                dp[r, c] = BIG
                continue
            candidates = []
            if c > 0     and valid[r-1, c-1]: candidates.append(dp[r-1, c-1])
            if             valid[r-1, c    ]: candidates.append(dp[r-1, c    ])
            if c < W - 1 and valid[r-1, c+1]: candidates.append(dp[r-1, c+1])
            if candidates:
                dp[r, c] += min(candidates)

    # Traceback
    seam_cols = np.zeros(H, dtype=np.int32)
    seam_cols[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        c = seam_cols[r + 1]
        lo, hi = max(0, c - 1), min(W, c + 2)
        seam_cols[r] = lo + int(np.argmin(dp[r, lo:hi]))

    labels = np.zeros((H, W), dtype=np.int32)
    for r in range(H):
        labels[r, seam_cols[r]:] = 1
    labels[~valid.astype(bool)] = 0
    return labels


# ---------------------------------------------------------------------------
# Public seam finder
# ---------------------------------------------------------------------------

class GraphCutSeamFinder:
    """
    Motion-aware optimal seam finder.

    Usage::

        finder = GraphCutSeamFinder()

        # First frame – search over all of Ω
        labels = finder.find_seam(img0, img1, overlap_mask)

        # Subsequent frames with detected motion
        labels = finder.find_seam(img0, img1, overlap_mask,
                                  motion_mask=motion_mask)

    Args:
        alpha       : weight for smoothness term  (paper: α=1)
        beta        : weight for gradient term     (paper: β=1)
        use_graphcut: True=PyMaxflow, False=DP fallback (auto-detected)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_graphcut: Optional[bool] = None,
    ):
        self.alpha = alpha
        self.beta  = beta

        if use_graphcut is None:
            self.use_graphcut = _MAXFLOW_AVAILABLE
        else:
            self.use_graphcut = use_graphcut and _MAXFLOW_AVAILABLE

    # ------------------------------------------------------------------ #

    def find_seam(
        self,
        img0: np.ndarray,           # (H, W, 3) float32 [0,1]  – first registered image
        img1: np.ndarray,           # (H, W, 3) float32 [0,1]  – second registered image
        overlap_mask: np.ndarray,   # (H, W) uint8, 255 = Ω
        motion_mask: Optional[np.ndarray] = None,  # (H, W) uint8, 255 = M_Ω
    ) -> np.ndarray:
        """
        Find the optimal seam label map.

        For the first frame (motion_mask=None):  search over all of Ω  (eq. 6).
        For subsequent frames:                   search over Ω ∖ M_Ω  (eq. 7).

        Returns:
            labels : (H, W) int32  {0 = take pixel from img0, 1 = from img1}
        """
        # Build cost map
        cost_map = _edge_weight_map(img0, img1, self.alpha, self.beta)

        # Build valid domain
        if motion_mask is not None:
            # Ω ∖ M_Ω  – exclude moving regions
            valid = overlap_mask.copy()
            valid[motion_mask > 0] = 0
        else:
            valid = overlap_mask.copy()

        if self.use_graphcut:
            labels = _find_seam_graphcut(cost_map, valid)
        else:
            labels = _find_seam_dp(cost_map, valid)

        return labels

    # ------------------------------------------------------------------ #

    @staticmethod
    def seam_to_mask(labels: np.ndarray) -> np.ndarray:
        """
        Convert label map to a uint8 display mask (0=black, 255=white).
        Useful for visualisation.
        """
        return (labels * 255).astype(np.uint8)

    @staticmethod
    def should_update_seam(
        motion_mask: np.ndarray,
        current_labels: np.ndarray,
        seam_buffer: int = 10,
    ) -> bool:
        """
        Decide whether the seam needs recalculation.

        A seam update is triggered when moving pixels are detected within
        `seam_buffer` pixels of the current seam boundary.

        Args:
            motion_mask    : (H, W) uint8  from MotionDetector
            current_labels : (H, W) int32  current seam labels
            seam_buffer    : proximity threshold in pixels

        Returns:
            True if seam should be recomputed
        """
        if motion_mask is None or np.count_nonzero(motion_mask) == 0:
            return False

        # Find seam boundary pixels (where label changes horizontally)
        boundary = np.zeros_like(current_labels, dtype=np.uint8)
        boundary[:, :-1] = ((current_labels[:, 1:] != current_labels[:, :-1])
                            .astype(np.uint8) * 255)

        # Dilate boundary by seam_buffer
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (seam_buffer * 2 + 1, seam_buffer * 2 + 1)
        )
        dilated_boundary = cv2.dilate(boundary, kernel)

        # Check overlap with motion mask
        overlap = cv2.bitwise_and(dilated_boundary, motion_mask)
        return np.count_nonzero(overlap) > 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    H, W = 128, 256

    img0 = np.random.rand(H, W, 3).astype(np.float32)
    img1 = np.random.rand(H, W, 3).astype(np.float32)

    # Simulate an overlap region (centre strip)
    overlap = np.zeros((H, W), dtype=np.uint8)
    overlap[:, W//4: 3*W//4] = 255

    finder = GraphCutSeamFinder()

    # First frame (no motion mask)
    labels = finder.find_seam(img0, img1, overlap)
    print(f"Seam labels unique: {np.unique(labels)}")
    print(f"Label=0 count: {(labels==0).sum()}  Label=1 count: {(labels==1).sum()}")

    # Subsequent frame with motion
    motion = np.zeros((H, W), dtype=np.uint8)
    motion[40:80, W//4: W//2] = 255   # fake moving object in overlap
    labels2 = finder.find_seam(img0, img1, overlap, motion_mask=motion)
    print(f"With motion – seam labels unique: {np.unique(labels2)}")

    update_needed = GraphCutSeamFinder.should_update_seam(motion, labels)
    print(f"Should update seam: {update_needed}")
    print("GraphCutSeamFinder smoke test passed.")

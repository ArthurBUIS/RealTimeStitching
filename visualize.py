"""
Qualitative visualisation tools.

Reproduces the style of Figures 8–12 in the paper:
  - Side-by-side method comparison grids (Figs 8–11)
  - Dynamic-scene fusion comparison with before/during/after columns (Fig 12)
  - Seam overlay on the panoramic output
  - Motion mask overlay on the overlap region
  - Registration quality heatmaps (photometric difference map)

All functions accept standard numpy uint8 BGR images (OpenCV convention)
and write PNG outputs or return figure arrays.

Usage (standalone):
    python visualize.py \
        --checkpoint checkpoints/stage2_best.pth \
        --img_a      samples/cam_a.jpg \
        --img_b      samples/cam_b.jpg \
        --output_dir vis_output/
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------------

def draw_seam_overlay(
    panorama: np.ndarray,      # (H, W, 3) uint8 BGR
    labels: np.ndarray,        # (H, W) int32 {0, 1}
    colour: Tuple[int, int, int] = (0, 0, 255),  # BGR red
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw the seam boundary on top of the panoramic image.
    The seam is where adjacent pixels have different labels.
    """
    out = panorama.copy()
    # Detect horizontal boundaries
    boundary = np.zeros(labels.shape, dtype=np.uint8)
    boundary[:, :-1] = ((labels[:, 1:] != labels[:, :-1])
                        .astype(np.uint8))
    boundary[:, 1:]  |= ((labels[:, :-1] != labels[:, 1:])
                         .astype(np.uint8))
    # Thicken
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1)
    )
    boundary = cv2.dilate(boundary * 255, kernel)
    out[boundary > 0] = colour
    return out


def draw_motion_overlay(
    panorama: np.ndarray,          # (H, W, 3) uint8 BGR
    motion_mask: np.ndarray,       # (H, W) uint8, 255 = motion
    colour: Tuple[int, int, int] = (0, 255, 0),  # BGR green
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a semi-transparent colour overlay on detected motion regions.
    """
    out = panorama.copy().astype(np.float32)
    mask = (motion_mask > 0)
    overlay = np.zeros_like(out)
    overlay[mask] = colour
    out = (1 - alpha) * out + alpha * overlay
    return out.clip(0, 255).astype(np.uint8)


def draw_diff_heatmap(
    img_ref: np.ndarray,    # (H, W, 3) uint8 BGR  – reference
    img_warp: np.ndarray,   # (H, W, 3) uint8 BGR  – warped
) -> np.ndarray:
    """
    Return a colourised absolute-difference heatmap (uint8 BGR).
    Bright red = large error; dark blue = small error.
    """
    diff = cv2.absdiff(img_ref, img_warp).mean(axis=-1).astype(np.float32)
    diff_norm = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return heatmap


def add_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    colour: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Burn a text label into an image (in-place copy)."""
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, colour, thickness, cv2.LINE_AA)
    return out


def resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def make_tile(images: List[np.ndarray], labels: List[str],
              rows: int, cols: int, cell_h: int = 256,
              cell_w: int = 384) -> np.ndarray:
    """
    Arrange images in a rows×cols grid with text labels.
    Images are resized to (cell_h, cell_w).
    """
    cells = []
    for img, lbl in zip(images, labels):
        cell = resize_to(img, cell_h, cell_w)
        cell = add_label(cell, lbl)
        cells.append(cell)

    # Pad to fill grid
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank)

    rows_list = []
    for r in range(rows):
        row_imgs = cells[r * cols: (r + 1) * cols]
        rows_list.append(np.hstack(row_imgs))
    return np.vstack(rows_list)


# ---------------------------------------------------------------------------
# Figure 8–11 style: method comparison grid
# ---------------------------------------------------------------------------

def plot_method_comparison(
    results: Dict[str, np.ndarray],   # method_name → stitched BGR image
    scenario: str = "large_parallax",
    output_path: Optional[str] = None,
    cell_h: int = 256,
    cell_w: int = 512,
    zoom_rect: Optional[Tuple[int, int, int, int]] = None,  # (x,y,w,h)
) -> np.ndarray:
    """
    Produce a side-by-side comparison grid matching Figures 8–11.

    Each method gets one column.  If zoom_rect is provided a magnified
    crop is shown in a second row (mimicking the red-rectangle insets
    in the paper).

    Args:
        results    : ordered dict of method name → stitched image
        scenario   : title string
        output_path: if given, save PNG to this path
        zoom_rect  : (x, y, w, h) region to magnify in the second row

    Returns:
        grid image (uint8 BGR)
    """
    methods = list(results.keys())
    n = len(methods)

    rows = 2 if zoom_rect else 1
    grid_imgs, grid_labels = [], []

    for name, img in results.items():
        full = resize_to(img, cell_h, cell_w)
        if zoom_rect:
            x, y, zw, zh = zoom_rect
            crop = img[y: y + zh, x: x + zw]
            # Draw red rectangle on full image
            full_rect = full.copy()
            # scale rect to cell size
            sx = cell_w / img.shape[1]
            sy = cell_h / img.shape[0]
            rx = int(x * sx); ry = int(y * sy)
            rw = int(zw * sx); rh = int(zh * sy)
            cv2.rectangle(full_rect, (rx, ry), (rx + rw, ry + rh),
                          (0, 0, 255), 2)
            grid_imgs.append(full_rect)
            grid_labels.append(name)
            zoom = resize_to(crop, cell_h, cell_w)
            grid_imgs.append(zoom)
            grid_labels.append(f"{name} (zoom)")
        else:
            grid_imgs.append(full)
            grid_labels.append(name)

    if zoom_rect:
        # Reorder: first row = full images, second row = zooms
        full_row  = grid_imgs[0::2]
        full_lbls = grid_labels[0::2]
        zoom_row  = grid_imgs[1::2]
        zoom_lbls = grid_labels[1::2]
        top    = np.hstack([add_label(r, l) for r, l in zip(full_row, full_lbls)])
        bottom = np.hstack([add_label(z, l) for z, l in zip(zoom_row, zoom_lbls)])

        # Add scenario title bar
        title_bar = np.zeros((40, top.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, f"Scenario: {scenario}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2, cv2.LINE_AA)
        grid = np.vstack([title_bar, top, bottom])
    else:
        row = np.hstack([add_label(resize_to(img, cell_h, cell_w), lbl)
                         for img, lbl in zip(grid_imgs, grid_labels)])
        title_bar = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, f"Scenario: {scenario}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2, cv2.LINE_AA)
        grid = np.vstack([title_bar, row])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, grid)

    return grid


# ---------------------------------------------------------------------------
# Figure 12 style: dynamic scene fusion (before / during / after)
# ---------------------------------------------------------------------------

def plot_fusion_comparison(
    method_frames: Dict[str, List[np.ndarray]],
    # e.g. {"Weighted Average": [before, during, after], "Ours": [...]}
    output_path: Optional[str] = None,
    cell_h: int = 200,
    cell_w: int = 400,
) -> np.ndarray:
    """
    Produce a methods × phases grid matching Figure 12.

    Rows = methods, Columns = [Before Crossing, During Crossing, After Crossing]
    """
    phases = ["Before Crossing", "During Crossing", "After Crossing"]
    methods = list(method_frames.keys())

    col_w = cell_w
    col_h = cell_h
    header_h = 40
    label_w  = 160   # left margin for method names

    total_w = label_w + len(phases) * col_w
    total_h = header_h + len(methods) * col_h
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # Column headers
    for ci, phase in enumerate(phases):
        x = label_w + ci * col_w + 5
        cv2.putText(canvas, phase, (x, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # Rows
    for ri, (method, frames) in enumerate(method_frames.items()):
        y0 = header_h + ri * col_h

        # Method label on left
        cv2.putText(canvas, method,
                    (5, y0 + col_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)

        # Frame cells
        for ci, frame in enumerate(frames[:3]):
            cell = resize_to(frame, col_h, col_w)
            x0 = label_w + ci * col_w
            canvas[y0: y0 + col_h, x0: x0 + col_w] = cell

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)

    return canvas


# ---------------------------------------------------------------------------
# Registration quality visualisation
# ---------------------------------------------------------------------------

def plot_registration_quality(
    img_a: np.ndarray,       # (H, W, 3) uint8 BGR – reference
    img_b: np.ndarray,       # (H, W, 3) uint8 BGR – target
    warped_a: np.ndarray,    # (H, W, 3) uint8 BGR – warped reference
    output_path: Optional[str] = None,
    cell_h: int = 256,
    cell_w: int = 384,
) -> np.ndarray:
    """
    Three-panel view: reference | target | warped | diff heatmap.
    """
    panels = [
        (img_a,   "Input A (reference)"),
        (img_b,   "Input B (target)"),
        (warped_a,"Warped A"),
        (draw_diff_heatmap(img_b, warped_a), "Diff heatmap"),
    ]
    row = np.hstack([
        add_label(resize_to(img, cell_h, cell_w), lbl)
        for img, lbl in panels
    ])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, row)
    return row


# ---------------------------------------------------------------------------
# Seam + motion diagnostic visualisation
# ---------------------------------------------------------------------------

def plot_seam_diagnostic(
    panorama: np.ndarray,
    labels: np.ndarray,
    overlap_mask: np.ndarray,
    motion_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    cell_h: int = 360,
    cell_w: int = 640,
) -> np.ndarray:
    """
    Stitch diagnostics: panorama | seam overlay | motion overlay (if any).
    """
    panels = [
        (panorama, "Panorama"),
        (draw_seam_overlay(panorama, labels), "Seam overlay"),
    ]
    if motion_mask is not None:
        panels.append((
            draw_motion_overlay(panorama, motion_mask),
            "Motion regions (green)"
        ))

    row = np.hstack([
        add_label(resize_to(img, cell_h, cell_w), lbl)
        for img, lbl in panels
    ])
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, row)
    return row


# ---------------------------------------------------------------------------
# Matplotlib-based metric bar chart (Tables 1-3)
# ---------------------------------------------------------------------------

def plot_metric_bars(
    metric_name: str,        # e.g. "PSNR (dB)"
    methods: List[str],
    buckets: List[str],      # e.g. ["0-30%", "30-60%", "60-100%", "Average"]
    values: np.ndarray,      # (n_methods, n_buckets)
    highlight_method: str = "Proposed Method",
    output_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart matching the style of Tables 1–3 visualised.
    """
    n_methods = len(methods)
    n_buckets = len(buckets)
    x = np.arange(n_buckets)
    width = 0.7 / n_methods

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, (method, row) in enumerate(zip(methods, values)):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, row, width,
                      label=method, color=colours[i],
                      edgecolor="white", linewidth=0.5)
        if method == highlight_method:
            for bar in bars:
                bar.set_edgecolor("red")
                bar.set_linewidth(2)

    ax.set_xlabel("Overlap Rate")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Overlap Rate and Method")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI: run full visualisation on a single image pair
# ---------------------------------------------------------------------------

def _run_vis(args):
    import torch
    from fusion.stitcher import PanoramicStitcher, preprocess_frame, tensor_to_uint8
    from fusion.graph_cut import GraphCutSeamFinder
    from fusion.motion_detector import MotionDetector, build_overlap_mask

    os.makedirs(args.output_dir, exist_ok=True)

    frame_a = cv2.imread(args.img_a)
    frame_b = cv2.imread(args.img_b)
    if frame_a is None or frame_b is None:
        raise FileNotFoundError("Could not read input images.")

    # Run stitcher
    stitcher = PanoramicStitcher(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
    )
    panorama, info = stitcher.stitch(frame_a, frame_b)
    state = stitcher.state

    # Registration quality
    from models.homography_net import HomographyNet, HomographyWarper
    device = stitcher.device
    ta, _ = preprocess_frame(frame_a, args.img_size)
    ta = ta.to(device)

    import torch
    warper = HomographyWarper(args.img_size, args.img_size).to(device)
    with torch.no_grad():
        warped_a_t = warper(ta, state.H2)
    warped_a = tensor_to_uint8(warped_a_t)

    fa_resized = cv2.resize(frame_a, (args.img_size, args.img_size))
    fb_resized = cv2.resize(frame_b, (args.img_size, args.img_size))

    # Plot registration quality
    reg_vis = plot_registration_quality(
        fa_resized, fb_resized, warped_a,
        output_path=os.path.join(args.output_dir, "registration_quality.png"),
    )
    print(f"Saved registration_quality.png")

    # Plot seam diagnostic
    if state.seam_labels is not None:
        seam_vis = plot_seam_diagnostic(
            panorama,
            state.seam_labels,
            state.overlap_mask,
            output_path=os.path.join(args.output_dir, "seam_diagnostic.png"),
        )
        print(f"Saved seam_diagnostic.png")

    # Plot paper's metric bars (using paper's Table 2 values as example)
    methods  = ["I3×3", "SIFT+RANSAC", "UDIS", "UDIS++", "NIS", "Proposed"]
    buckets  = ["0-30%", "30-60%", "60-100%", "Average"]
    psnr_vals = np.array([
        [15.62, 12.53, 10.45, 12.625],
        [24.85, 21.96, 17.32, 20.971],
        [26.91, 23.78, 20.45, 23.387],
        [31.16, 26.86, 22.63, 26.458],
        [28.33, 25.06, 21.32, 24.545],
        [30.02, 25.73, 21.57, 25.338],
    ])
    plot_metric_bars(
        "PSNR (dB)", methods, buckets, psnr_vals,
        highlight_method="Proposed",
        output_path=os.path.join(args.output_dir, "psnr_comparison.png"),
    )
    print(f"Saved psnr_comparison.png")

    ssim_vals = np.array([
        [0.3736, 0.1586, 0.0658, 0.1860],
        [0.8143, 0.7261, 0.5145, 0.6679],
        [0.8945, 0.7992, 0.6357, 0.7624],
        [0.9573, 0.8915, 0.7791, 0.8663],
        [0.9037, 0.8638, 0.7302, 0.8223],
        [0.9312, 0.8728, 0.7385, 0.8366],
    ])
    plot_metric_bars(
        "SSIM", methods, buckets, ssim_vals,
        highlight_method="Proposed",
        output_path=os.path.join(args.output_dir, "ssim_comparison.png"),
    )
    print(f"Saved ssim_comparison.png")

    print(f"\nAll visualisations written to {args.output_dir}/")
    print(f"Inference info: fps={info['fps']:.1f}  seam={info['seam']}  motion={info['motion_detected']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualisation tools")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--img_a",       required=True)
    p.add_argument("--img_b",       required=True)
    p.add_argument("--output_dir",  default="vis_output")
    p.add_argument("--img_size",    type=int, default=512)
    _run_vis(p.parse_args())

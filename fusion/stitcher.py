"""
Real-time panoramic video stitcher  (Section 3.1, Figure 1).

Orchestrates the full pipeline frame-by-frame:

    ┌─────────────────────────────────────────────────────────┐
    │  Frame N                                                │
    │                                                         │
    │  preprocess → is first frame?                          │
    │       YES:  register → find seam → cache params        │
    │       NO :  use cached registration params             │
    │                                                         │
    │  detect motion in overlap                               │
    │       motion near seam?                                 │
    │           YES: recompute seam (Ω ∖ M_Ω)                │
    │           NO : use cached seam                         │
    │                                                         │
    │  blend → output panoramic frame                        │
    └─────────────────────────────────────────────────────────┘

The stitcher targets the 23 fps reported in Table 5 of the paper.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import cv2
import torch

from models.homography_net import HomographyNet
from fusion.motion_detector import MotionDetector, build_overlap_mask
from fusion.graph_cut import GraphCutSeamFinder
from fusion.blending import SeamBlender, WeightedAverageBlender

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image pre / post processing helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(
    frame: np.ndarray,   # (H, W, 3) uint8  BGR (OpenCV default)
    img_size: int = 512,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Resize + ImageNet-normalise a frame for the network, and also return
    the float32 [0,1] version for energy-map computation.

    Returns:
        tensor   : (1, 3, img_size, img_size) torch.Tensor (normalised)
        float_im : (img_size, img_size, 3) float32  RGB [0,1]
    """
    resized = cv2.resize(frame, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD          # (H,W,3)
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1))   # (3,H,W)
    tensor = tensor.unsqueeze(0)                               # (1,3,H,W)
    return tensor, rgb


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """
    (1, 3, H, W) or (3, H, W) normalised tensor → (H, W, 3) uint8 BGR.
    """
    if t.dim() == 4:
        t = t.squeeze(0)
    # Denormalise
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(3, 1, 1)
    img = (t * std + mean).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Noise reduction + contrast enhancement preprocessing step
    (Section 3.1 – "preprocessing steps include noise reduction and
    contrast enhancement").
    """
    # Gaussian denoise
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    # CLAHE contrast enhancement on the L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Cached state for the stitcher
# ---------------------------------------------------------------------------

@dataclass
class StitcherState:
    """Persisted state between frames."""
    H2: Optional[torch.Tensor] = None          # (1,3,3) refined homography
    overlap_mask: Optional[np.ndarray] = None  # (H,W) uint8
    seam_labels: Optional[np.ndarray] = None   # (H,W) int32
    frame_count: int = 0
    seam_update_count: int = 0
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main Stitcher
# ---------------------------------------------------------------------------

class PanoramicStitcher:
    """
    Real-time dual-camera panoramic video stitcher.

    Args:
        checkpoint_path : path to a trained HomographyNet checkpoint
        img_size        : spatial size fed to the network (default 512)
        device          : 'cuda' | 'cpu' | None (auto-detect)
        feather_width   : seam feathering half-width in pixels
        seam_buffer     : pixel margin around seam for motion trigger
        use_graphcut    : True = PyMaxflow GC, False = DP fallback
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 512,
        device: Optional[str] = None,
        feather_width: int = 8,
        seam_buffer: int = 10,
        use_graphcut: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = img_size

        # ---- Registration network ----
        self.net = HomographyNet(img_size=img_size, pretrained_backbone=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model", ckpt)   # handle both formats
        self.net.load_state_dict(state_dict)
        self.net.to(self.device).eval()
        log.info(f"Loaded checkpoint from {checkpoint_path}")

        # ---- Fusion components ----
        self.motion_detector = MotionDetector()
        self.seam_finder     = GraphCutSeamFinder(use_graphcut=use_graphcut)
        self.seam_blender    = SeamBlender(feather_width=feather_width)
        self.wa_blender      = WeightedAverageBlender()
        self.seam_buffer     = seam_buffer

        # ---- State ----
        self.state = StitcherState()

        log.info(
            f"PanoramicStitcher ready | device={device} | "
            f"img_size={img_size} | graphcut={use_graphcut}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset stitcher state (e.g. after a camera cut or restart)."""
        self.state = StitcherState()
        self.motion_detector.reset()
        log.info("Stitcher state reset.")

    def stitch(
        self,
        frame_a: np.ndarray,   # (H, W, 3) uint8 BGR – camera A
        frame_b: np.ndarray,   # (H, W, 3) uint8 BGR – camera B
    ) -> Tuple[np.ndarray, dict]:
        """
        Process one frame pair and return the stitched panoramic frame.

        Returns:
            panorama : (img_size, img_size, 3) uint8 BGR
            info     : diagnostic dict (timings, seam_updated, motion_detected)
        """
        t_total = time.perf_counter()
        state = self.state
        state.frame_count += 1
        info = {"frame": state.frame_count, "seam_updated": False,
                "motion_detected": False}

        # ---- 1. Preprocessing (noise reduction + contrast) ----
        t0 = time.perf_counter()
        fa_enhanced = enhance_frame(frame_a)
        fb_enhanced = enhance_frame(frame_b)
        info["t_preprocess"] = time.perf_counter() - t0

        # Convert to tensors for the network
        ta, fa_float = preprocess_frame(fa_enhanced, self.img_size)
        tb, fb_float = preprocess_frame(fb_enhanced, self.img_size)
        ta = ta.to(self.device)
        tb = tb.to(self.device)

        # ---- 2. Registration -------------------------------------------- #
        t0 = time.perf_counter()

        if state.frame_count == 1:
            # First frame: run full registration
            with torch.no_grad():
                out = self.net(ta, tb)

            state.H2           = out["H2"].detach()
            mask_float         = out["mask"].squeeze().cpu().numpy()   # (H,W)
            state.overlap_mask = build_overlap_mask(mask_float)

            # Warp images for seam search
            warped_a_t = out["warped_A"]
            warped_b_t = out["warped_B"]

            info["registration"] = "full"
        else:
            # Subsequent frames: reuse cached H2
            with torch.no_grad():
                from models.homography_net import HomographyWarper
                # Re-use the stored homography; only apply the warper
                warper = HomographyWarper(self.img_size, self.img_size)
                warper = warper.to(self.device)
                warped_a_t = warper(ta, state.H2)
                warped_b_t = warper(tb, state.H2)

            info["registration"] = "cached"

        info["t_registration"] = time.perf_counter() - t0

        # Float numpy versions for blending / energy maps
        warped_a_np = tensor_to_uint8(warped_a_t).astype(np.float32) / 255.0
        warped_b_np = tensor_to_uint8(warped_b_t).astype(np.float32) / 255.0

        # ---- 3. Motion detection ---------------------------------------- #
        t0 = time.perf_counter()
        warped_a_u8 = (warped_a_np * 255).astype(np.uint8)
        warped_b_u8 = (warped_b_np * 255).astype(np.uint8)
        motion_mask = self.motion_detector.detect(
            warped_a_u8, warped_b_u8, state.overlap_mask
        )
        motion_detected = np.count_nonzero(motion_mask) > 0
        info["motion_detected"] = motion_detected
        info["t_motion"] = time.perf_counter() - t0

        # ---- 4. Seam search / update ------------------------------------ #
        t0 = time.perf_counter()

        if state.seam_labels is None:
            # First frame: unconditional seam search over all of Ω
            state.seam_labels = self.seam_finder.find_seam(
                warped_a_np, warped_b_np,
                state.overlap_mask,
                motion_mask=None,
            )
            info["seam"] = "search_first"
            state.seam_update_count += 1
        elif motion_detected and GraphCutSeamFinder.should_update_seam(
            motion_mask, state.seam_labels, seam_buffer=self.seam_buffer
        ):
            # Motion near seam: update seam over Ω ∖ M_Ω
            state.seam_labels = self.seam_finder.find_seam(
                warped_a_np, warped_b_np,
                state.overlap_mask,
                motion_mask=motion_mask,
            )
            info["seam"] = "updated"
            info["seam_updated"] = True
            state.seam_update_count += 1
        else:
            info["seam"] = "cached"

        info["t_seam"] = time.perf_counter() - t0

        # ---- 5. Blending / fusion --------------------------------------- #
        t0 = time.perf_counter()

        if state.seam_labels is not None:
            panorama_f = self.seam_blender.blend(
                warped_a_np, warped_b_np,
                state.seam_labels,
                state.overlap_mask,
            )
        else:
            panorama_f = self.wa_blender.blend(
                warped_a_np, warped_b_np,
                state.overlap_mask,
            )

        info["t_blend"] = time.perf_counter() - t0

        # ---- 6. Finalise output ----------------------------------------- #
        panorama = (panorama_f * 255).clip(0, 255).astype(np.uint8)
        info["t_total"] = time.perf_counter() - t_total
        info["fps"]     = 1.0 / max(info["t_total"], 1e-9)

        return panorama, info

    # ------------------------------------------------------------------ #
    # Multi-camera extension (3-camera, Section 3.1 / Figure 2)
    # ------------------------------------------------------------------ #

    def stitch_multi(
        self,
        frames: list,   # [frame_0, frame_1, ..., frame_N]  all (H,W,3) uint8
    ) -> np.ndarray:
        """
        Sequentially stitch N frames: 0↔1, result↔2, ...
        Returns the full multi-camera panorama.
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames.")
        panorama, _ = self.stitch(frames[0], frames[1])
        for f in frames[2:]:
            panorama, _ = self.stitch(panorama, f)
        return panorama


# ---------------------------------------------------------------------------
# CLI: process a pair of video files and write stitched output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Real-time panoramic video stitcher")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video_a",    required=True)
    p.add_argument("--video_b",    required=True)
    p.add_argument("--output",     default="stitched.mp4")
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--max_frames", type=int, default=None)
    args = p.parse_args()

    stitcher = PanoramicStitcher(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
    )

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)

    fps_in = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in,
                             (args.img_size, args.img_size))

    frame_idx = 0
    fps_history = []

    while True:
        ret_a, fa = cap_a.read()
        ret_b, fb = cap_b.read()
        if not ret_a or not ret_b:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        panorama, info = stitcher.stitch(fa, fb)
        writer.write(panorama)
        fps_history.append(info["fps"])

        if frame_idx % 30 == 0:
            avg_fps = sum(fps_history[-30:]) / len(fps_history[-30:])
            log.info(
                f"Frame {frame_idx:04d} | fps={avg_fps:.1f} | "
                f"seam={info['seam']} | motion={info['motion_detected']}"
            )
        frame_idx += 1

    cap_a.release()
    cap_b.release()
    writer.release()

    if fps_history:
        avg = sum(fps_history) / len(fps_history)
        log.info(f"Done. {frame_idx} frames | avg fps={avg:.1f} → {args.output}")

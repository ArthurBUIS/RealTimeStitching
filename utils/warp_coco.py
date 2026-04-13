"""
Generate the Warped MS-COCO dataset for Stage-1 pre-training.

The Warped MS-COCO dataset (used in UDIS and this paper) is derived from
the raw MS-COCO images by:
  1. Randomly sampling a homography perturbation (4-point method).
  2. Cropping a fixed-size patch from the original image (view A).
  3. Applying the random homography to obtain a second patch (view B).
  4. Saving the pair to  <output_root>/train/input1/  and  .../input2/

The homography generation follows the standard protocol from:
  "Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model"
  (DeTone et al., 2018) and is consistent with the UDIS codebase.

Usage:
    python utils/warp_coco.py \
        --coco_dir   /data/coco/train2017 \
        --output_dir /data/WarpedCOCO \
        --n_train    10000 \
        --n_test     1000  \
        --patch_size 512   \
        --rho        32
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Homography generation
# ---------------------------------------------------------------------------

def random_homography(
    patch_size: int,
    rho: int,
    img_h: int,
    img_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random homography by perturbing the four corners of a patch.

    Args:
        patch_size : size of the square patch
        rho        : maximum corner displacement in pixels
        img_h, img_w : source image dimensions

    Returns:
        H          : (3, 3) homography  (maps patch coords → perturbed coords)
        src_corners: (4, 2) source corners
        dst_corners: (4, 2) destination corners (perturbed)
    """
    # Random top-left of the patch (must fit within image)
    margin = rho + 1
    max_x = img_w - patch_size - margin
    max_y = img_h - patch_size - margin
    if max_x <= margin or max_y <= margin:
        raise ValueError(
            f"Image too small ({img_h}×{img_w}) for patch_size={patch_size} "
            f"and rho={rho}."
        )
    x = random.randint(margin, max_x)
    y = random.randint(margin, max_y)

    # Source corners (TL, TR, BR, BL)
    src = np.array([
        [x,              y             ],
        [x + patch_size, y             ],
        [x + patch_size, y + patch_size],
        [x,              y + patch_size],
    ], dtype=np.float32)

    # Random perturbations
    delta = np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32)
    dst = src + delta

    # Homography from src → dst
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        H = np.eye(3, dtype=np.float32)

    return H.astype(np.float32), src, dst


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pair(
    img: np.ndarray,
    patch_size: int,
    rho: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a source image, generate an (input1, input2) pair:
      input1 : a patch_size × patch_size crop from the original image
      input2 : the same region warped by a random homography

    Returns:
        patch_a : (patch_size, patch_size, 3) uint8
        patch_b : (patch_size, patch_size, 3) uint8
        H       : (3, 3) ground-truth homography
    """
    h, w = img.shape[:2]

    # Ensure minimum size
    if h < patch_size + 2 * rho + 2 or w < patch_size + 2 * rho + 2:
        # Upscale image to fit
        scale = (patch_size + 2 * rho + 10) / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    H, src_corners, _ = random_homography(patch_size, rho, h, w)

    # Patch A: crop from source (top-left of first src corner)
    x, y = int(src_corners[0, 0]), int(src_corners[0, 1])
    patch_a = img[y: y + patch_size, x: x + patch_size]

    # Patch B: warp the full image and crop the same region
    warped = cv2.warpPerspective(img, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    patch_b = warped[y: y + patch_size, x: x + patch_size]

    # Validate shapes
    if patch_a.shape[:2] != (patch_size, patch_size):
        patch_a = cv2.resize(patch_a, (patch_size, patch_size))
    if patch_b.shape[:2] != (patch_size, patch_size):
        patch_b = cv2.resize(patch_b, (patch_size, patch_size))

    return patch_a, patch_b, H


# ---------------------------------------------------------------------------
# Dataset generation loop
# ---------------------------------------------------------------------------

def generate_split(
    image_paths: list,
    output_root: str,
    split: str,            # "train" or "test"
    n_pairs: int,
    patch_size: int,
    rho: int,
    seed: int = 42,
) -> None:
    """
    Generate n_pairs image pairs and save to output_root/{split}/input1|2/.
    """
    random.seed(seed)
    np.random.seed(seed)

    out1 = Path(output_root) / split / "input1"
    out2 = Path(output_root) / split / "input2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts  = 0
    max_attempts = n_pairs * 5

    pbar = tqdm(total=n_pairs, desc=f"Generating {split} pairs")

    while generated < n_pairs and attempts < max_attempts:
        attempts += 1
        img_path = random.choice(image_paths)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            patch_a, patch_b, H = generate_pair(img, patch_size, rho)
        except ValueError:
            continue

        idx = generated
        fname = f"{idx:06d}.jpg"
        cv2.imwrite(str(out1 / fname), patch_a,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out2 / fname), patch_b,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        generated += 1
        pbar.update(1)

    pbar.close()
    print(f"{split}: {generated} pairs written to {output_root}/{split}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Warped MS-COCO dataset for homography pre-training"
    )
    p.add_argument("--coco_dir",    required=True,
                   help="Directory containing raw COCO JPEG images "
                        "(e.g. train2017/)")
    p.add_argument("--output_dir",  required=True,
                   help="Where to write WarpedCOCO dataset")
    p.add_argument("--n_train",     type=int, default=10000,
                   help="Number of training pairs")
    p.add_argument("--n_test",      type=int, default=1000,
                   help="Number of test pairs")
    p.add_argument("--patch_size",  type=int, default=512,
                   help="Patch (image pair) size in pixels")
    p.add_argument("--rho",         type=int, default=32,
                   help="Maximum corner displacement (pixels)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Collect all JPEG/PNG images from coco_dir
    exts = {".jpg", ".jpeg", ".png"}
    all_paths = [
        p for p in Path(args.coco_dir).rglob("*")
        if p.suffix.lower() in exts
    ]
    if not all_paths:
        raise FileNotFoundError(f"No images found in {args.coco_dir}")
    print(f"Found {len(all_paths)} source images.")

    # Shuffle and split source images so train/test don't share sources
    random.seed(args.seed)
    random.shuffle(all_paths)
    n_test_src = max(1, len(all_paths) // 10)
    test_paths  = all_paths[:n_test_src]
    train_paths = all_paths[n_test_src:]

    generate_split(train_paths, args.output_dir, "train",
                   args.n_train, args.patch_size, args.rho, args.seed)
    generate_split(test_paths,  args.output_dir, "test",
                   args.n_test,  args.patch_size, args.rho, args.seed + 1)

    print("Done. Dataset layout:")
    print(f"  {args.output_dir}/train/input1/  ← reference patches")
    print(f"  {args.output_dir}/train/input2/  ← warped patches")
    print(f"  {args.output_dir}/test/input1|2/ ← test pairs")

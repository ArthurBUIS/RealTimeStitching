"""
Dataset loaders for the two training stages (Section 4.1.1):

Stage 1  – Warped MS-COCO  (synthetic homography pairs, known ground truth)
Stage 2  – Composite dataset: UDIS-D + real industrial scenes

Expected directory layouts
---------------------------

Warped MS-COCO  (UDIS convention, widely used):
    <root>/
        train/
            input1/  *.jpg   (reference images)
            input2/  *.jpg   (target images, already warped)
        test/
            input1/  *.jpg
            input2/  *.jpg

UDIS-D  (same layout):
    <root>/
        training/
            input1/  *.jpg
            input2/  *.jpg
        testing/
            input1/  *.jpg
            input2/  *.jpg

The composite dataset used in Stage 2 is formed by combining UDIS-D
with the real industrial images; both use the same two-folder layout.
"""

import os
import glob
from pathlib import Path
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# ---------------------------------------------------------------------------
# Base paired-image dataset
# ---------------------------------------------------------------------------

class ImagePairDataset(Dataset):
    """
    Loads (input1, input2) image pairs from two mirrored directories.

    Args:
        root         : dataset root  (parent of input1/ and input2/)
        input1_dir   : name of the reference-image sub-folder
        input2_dir   : name of the target-image sub-folder
        img_size     : resize both images to this square side length
        augment      : apply random augmentations (flip, colour jitter)
        extensions   : accepted image file extensions
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root: str,
        input1_dir: str = "input1",
        input2_dir: str = "input2",
        img_size: int = 512,
        augment: bool = False,
        extensions: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.extensions = extensions or self.IMG_EXTENSIONS

        dir1 = self.root / input1_dir
        dir2 = self.root / input2_dir

        if not dir1.exists():
            raise FileNotFoundError(f"input1 directory not found: {dir1}")
        if not dir2.exists():
            raise FileNotFoundError(f"input2 directory not found: {dir2}")

        # Collect and sort files; match by sorted index
        self.paths1 = sorted(
            p for p in dir1.iterdir()
            if p.suffix.lower() in self.extensions
        )
        self.paths2 = sorted(
            p for p in dir2.iterdir()
            if p.suffix.lower() in self.extensions
        )

        if len(self.paths1) != len(self.paths2):
            raise ValueError(
                f"Mismatched pair counts: {len(self.paths1)} vs {len(self.paths2)}"
            )
        if len(self.paths1) == 0:
            raise ValueError(f"No images found under {dir1}")

        # Base transforms (always applied)
        self.base_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths1)

    def __getitem__(self, idx: int) -> dict:
        img1 = Image.open(self.paths1[idx]).convert("RGB")
        img2 = Image.open(self.paths2[idx]).convert("RGB")

        if self.augment:
            img1, img2 = self._augment_pair(img1, img2)

        t1 = self.base_tf(img1)   # (3, H, W)
        t2 = self.base_tf(img2)

        return {
            "img_a": t1,
            "img_b": t2,
            "path_a": str(self.paths1[idx]),
            "path_b": str(self.paths2[idx]),
        }

    # ------------------------------------------------------------------ #
    # Augmentations applied consistently to both images in a pair
    # ------------------------------------------------------------------ #

    def _augment_pair(
        self, img1: Image.Image, img2: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Geometric augmentations are applied identically to both images;
        colour jitter is applied independently (simulating different cameras).
        """
        # Random horizontal flip
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # Random vertical flip
        if random.random() < 0.3:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # Independent colour jitter (different cameras may have different
        # brightness / contrast settings)
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05)
        img1 = jitter(img1)
        img2 = jitter(img2)

        return img1, img2


# ---------------------------------------------------------------------------
# Convenience constructors for each dataset variant
# ---------------------------------------------------------------------------

def make_coco_dataset(
    root: str,
    split: str = "train",
    img_size: int = 512,
    augment: bool = True,
) -> ImagePairDataset:
    """
    Warped MS-COCO dataset.
    split : 'train' | 'test'
    """
    split_dir = "train" if split == "train" else "test"
    return ImagePairDataset(
        root=os.path.join(root, split_dir),
        input1_dir="input1",
        input2_dir="input2",
        img_size=img_size,
        augment=(augment and split == "train"),
    )


def make_udis_dataset(
    root: str,
    split: str = "train",
    img_size: int = 512,
    augment: bool = True,
) -> ImagePairDataset:
    """
    UDIS-D dataset.
    split : 'train' | 'test'
    """
    split_dir = "training" if split == "train" else "testing"
    return ImagePairDataset(
        root=os.path.join(root, split_dir),
        input1_dir="input1",
        input2_dir="input2",
        img_size=img_size,
        augment=(augment and split == "train"),
    )


def make_composite_dataset(
    udis_root: str,
    industrial_root: str,
    split: str = "train",
    img_size: int = 512,
    augment: bool = True,
) -> ConcatDataset:
    """
    Composite dataset for Stage-2 fine-tuning:
    UDIS-D  +  real industrial images (same directory layout).
    """
    ds_udis = make_udis_dataset(udis_root, split, img_size, augment)
    ds_industrial = ImagePairDataset(
        root=industrial_root,
        input1_dir="input1",
        input2_dir="input2",
        img_size=img_size,
        augment=(augment and split == "train"),
    )
    return ConcatDataset([ds_udis, ds_industrial])


# ---------------------------------------------------------------------------
# Inverse normalisation utility (for visualisation)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation → [0, 1]."""
    return (tensor * IMAGENET_STD.to(tensor.device)
            + IMAGENET_MEAN.to(tensor.device)).clamp(0, 1)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, numpy as np

    # Build a tiny synthetic dataset in a temp folder
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("input1", "input2"):
            os.makedirs(os.path.join(tmpdir, sub))
            for i in range(4):
                arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(
                    os.path.join(tmpdir, sub, f"{i:04d}.jpg")
                )

        ds = ImagePairDataset(tmpdir, augment=True)
        print(f"Dataset length : {len(ds)}")
        sample = ds[0]
        print(f"img_a shape    : {sample['img_a'].shape}")
        print(f"img_b shape    : {sample['img_b'].shape}")

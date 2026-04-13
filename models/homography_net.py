"""
Homography estimation network  (Section 3.2.1, Figure 3).

Two-scale pipeline:
    Coarse scale (1/16):
        GlobalCorrelation(Fa_1/16, Fb_1/16) → RegressionNet → TensorDLT → H1

    Fine scale (1/8):
        warp Fb_1/8 with H1  →  GlobalCorrelation(Fa_1/8, warped_Fb_1/8)
        → RegressionNet → TensorDLT → H2

    Final output:
        Warp original images with H2 → registered pair
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import SiameseFeatureExtractor


# ---------------------------------------------------------------------------
# Global Correlation Layer
# ---------------------------------------------------------------------------

class GlobalCorrelationLayer(nn.Module):
    """
    Computes a dense correlation volume between two feature maps by sliding
    a local d×d block of Fa over the full Fb map, normalised by channel dim.

    Output shape: (B, d*d, H, W)  – one correlation score per block-shift.
    """

    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.d = max_displacement
        self.pad = max_displacement

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fa, fb : (B, C, H, W)
        Returns:
            corr   : (B, (2d+1)^2, H, W)
        """
        B, C, H, W = fa.shape
        d = self.d

        # L2-normalise along channel axis for cosine-like similarity
        fa = F.normalize(fa, p=2, dim=1)
        fb = F.normalize(fb, p=2, dim=1)

        # Pad fb to allow shifts
        fb_padded = F.pad(fb, [d, d, d, d])  # (B, C, H+2d, W+2d)

        corr_list = []
        for dy in range(2 * d + 1):
            for dx in range(2 * d + 1):
                fb_shift = fb_padded[:, :, dy: dy + H, dx: dx + W]  # (B,C,H,W)
                # Dot product over channel dim → (B, H, W)
                dot = (fa * fb_shift).sum(dim=1)
                corr_list.append(dot)

        corr = torch.stack(corr_list, dim=1)  # (B, (2d+1)^2, H, W)
        return corr


# ---------------------------------------------------------------------------
# Regression Network
# ---------------------------------------------------------------------------

class RegressionNet(nn.Module):
    """
    Takes the flattened correlation volume and regresses 8 corner offsets Δ.

    Input: (B, (2d+1)^2, H, W)  via a small CNN → FC → 8 values
    """

    def __init__(self, in_channels: int, spatial_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        fc_in = 64 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )
        # Initialise final layer small to start near identity
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            delta : (B, 8) – predicted corner displacements
        """
        x = self.encoder(corr)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# TensorDLT: corner offsets → homography matrix  (3×3)
# ---------------------------------------------------------------------------

class TensorDLT(nn.Module):
    """
    Converts 8 corner displacements into a 3×3 homography matrix via the
    Direct Linear Transform, implemented as a differentiable batch operation.

    Reference corners are at the four corners of a patch_size × patch_size square,
    which we normalise to [-1, 1] before DLT for numerical stability.
    """

    def __init__(self, patch_size: int = 128):
        super().__init__()
        # Source corners (normalised to [-1,1]) – fixed, not learnable
        half = patch_size / 2.0
        src = torch.tensor([
            [-1.0, -1.0], [1.0, -1.0],
            [1.0,  1.0], [-1.0,  1.0],
        ]) * (half / half)  # already normalised
        self.register_buffer("src_corners", src)   # (4, 2)
        self.patch_size = patch_size

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta : (B, 8) – displacements in [-1,1] normalised space
        Returns:
            H     : (B, 3, 3) homography matrices
        """
        B = delta.shape[0]
        device = delta.device
        dtype = delta.dtype

        src = self.src_corners.to(device=device, dtype=dtype)  # (4, 2)
        src = src.unsqueeze(0).expand(B, -1, -1)               # (B, 4, 2)

        # Keep predictions in a sane range and remove NaN/Inf before solving.
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        delta = delta.clamp(min=-2.0, max=2.0)
        dst = src + delta.view(B, 4, 2)                        # (B, 4, 2)

        # Build an 8x8 linear system with h33 fixed to 1:
        # [xs ys 1 0  0  0 -xd*xs -xd*ys] [h11..h32]^T = xd
        # [0  0  0 xs ys 1 -yd*xs -yd*ys] [h11..h32]^T = yd
        xs = src[:, :, 0]  # (B, 4)
        ys = src[:, :, 1]
        xd = dst[:, :, 0]
        yd = dst[:, :, 1]

        zeros = torch.zeros_like(xs)
        ones = torch.ones_like(xs)

        row1 = torch.stack([
            xs, ys, ones, zeros, zeros, zeros, -xd * xs, -xd * ys
        ], dim=-1)  # (B, 4, 8)
        row2 = torch.stack([
            zeros, zeros, zeros, xs, ys, ones, -yd * xs, -yd * ys
        ], dim=-1)  # (B, 4, 8)

        M = torch.cat([row1, row2], dim=1)                    # (B, 8, 8)
        b = torch.cat([xd, yd], dim=1).unsqueeze(-1)          # (B, 8, 1)

        M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

        # Try a direct solve first; fall back to pseudo-inverse per sample.
        try:
            h8 = torch.linalg.solve(M, b).squeeze(-1)         # (B, 8)
        except RuntimeError:
            h8_list = []
            for i in range(B):
                Mi = M[i]
                bi = b[i]
                hi = torch.matmul(torch.linalg.pinv(Mi), bi).squeeze(-1)
                h8_list.append(hi)
            h8 = torch.stack(h8_list, dim=0)

        h9 = torch.ones(B, 1, device=device, dtype=dtype)
        h = torch.cat([h8, h9], dim=1)                        # (B, 9)
        H = h.view(B, 3, 3)
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        return H


# ---------------------------------------------------------------------------
# Differentiable Warping Module (Transformer-based bilinear sampler)
# ---------------------------------------------------------------------------

class HomographyWarper(nn.Module):
    """
    Warps an image / feature map with a given homography matrix using
    differentiable bilinear sampling (torch.nn.functional.grid_sample).
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        # Pre-compute normalised grid [-1,1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )
        ones = torch.ones_like(grid_x)
        grid = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
            H : (B, 3, 3) homography
        Returns:
            warped : (B, C, H, W)
        """
        B, C, height, width = x.shape
        grid = self.grid.to(x.device)                      # (H, W, 3)
        grid_flat = grid.view(-1, 3).unsqueeze(-1)         # (H*W, 3, 1)

        # Apply inverse homography to the output grid
        H_inv = torch.linalg.inv(H)                        # (B, 3, 3)
        H_inv_exp = H_inv.unsqueeze(1).expand(-1, height * width, -1, -1)
        # (B, H*W, 3, 1)
        grid_exp = grid_flat.unsqueeze(0).expand(B, -1, -1, -1)
        mapped = torch.matmul(H_inv_exp, grid_exp).squeeze(-1)  # (B,H*W,3)

        # Normalise homogeneous coordinates
        mapped_xy = mapped[..., :2] / (mapped[..., 2:3] + 1e-8)  # (B,H*W,2)
        sample_grid = mapped_xy.view(B, height, width, 2)

        warped = F.grid_sample(
            x, sample_grid,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return warped


# ---------------------------------------------------------------------------
# Full Homography Estimation Network
# ---------------------------------------------------------------------------

class HomographyNet(nn.Module):
    """
    Two-scale unsupervised homography estimation network (Figure 3).

    Input  : image pair (I_A, I_B)  each (B, 3, 512, 512)
    Output :
        H1          – coarse homography  (B, 3, 3)
        H2          – refined homography (B, 3, 3)
        warped_A    – I_A warped by H2  (B, 3, 512, 512)
        warped_B    – I_B warped by H2  (B, 3, 512, 512)
        mask        – valid overlap mask after warping (B, 1, 512, 512)
    """

    # Displacement budget for the correlation layer
    MAX_DISP = 4
    PATCH_SIZE = 128   # TensorDLT normalisation reference

    def __init__(self, img_size: int = 512, pretrained_backbone: bool = True):
        super().__init__()
        self.img_size = img_size

        # ---------- Shared Siamese backbone ----------
        self.backbone = SiameseFeatureExtractor(pretrained=pretrained_backbone)

        # ---------- Correlation layers ----------
        self.corr_coarse = GlobalCorrelationLayer(self.MAX_DISP)
        self.corr_fine = GlobalCorrelationLayer(self.MAX_DISP)

        num_corr_ch = (2 * self.MAX_DISP + 1) ** 2  # = 81

        # ---------- Regression networks (coarse / fine) ----------
        self.reg_coarse = RegressionNet(num_corr_ch, spatial_size=img_size // 16)
        self.reg_fine = RegressionNet(num_corr_ch, spatial_size=img_size // 8)

        # ---------- TensorDLT ----------
        self.dlt = TensorDLT(patch_size=self.PATCH_SIZE)

        # ---------- Warpers (feature maps + full resolution) ----------
        self.warper_feat_8 = HomographyWarper(img_size // 8, img_size // 8)
        self.warper_full = HomographyWarper(img_size, img_size)

    # ------------------------------------------------------------------ #

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """
        Args:
            img_a, img_b : (B, 3, 512, 512)  normalised to [-1, 1] or [0, 1]

        Returns dict with keys:
            H1, H2, warped_A, warped_B, mask
        """
        # ---------- 1. Feature extraction (shared weights) ----------
        fa_8, fa_16 = self.backbone(img_a)
        fb_8, fb_16 = self.backbone(img_b)

        # ---------- 2. Coarse scale (1/16) ----------
        corr_c = self.corr_coarse(fa_16, fb_16)   # (B, 81, H/16, W/16)
        delta1 = self.reg_coarse(corr_c)           # (B, 8)
        H1 = self.dlt(delta1)                      # (B, 3, 3)

        # ---------- 3. Warp Fb_1/8 with H1 ----------
        warped_fb_8 = self.warper_feat_8(fb_8, H1)

        # ---------- 4. Fine scale (1/8) ----------
        corr_f = self.corr_fine(fa_8, warped_fb_8)  # (B, 81, H/8, W/8)
        delta2 = self.reg_fine(corr_f)               # (B, 8)
        H2 = self.dlt(delta2)                        # (B, 3, 3)

        # ---------- 5. Warp full-resolution images with H2 ----------
        warped_a = self.warper_full(img_a, H2)
        warped_b = self.warper_full(img_b, H2)

        # ---------- 6. Compute valid-overlap mask ----------
        ones_a = torch.ones(img_a.shape[0], 1, self.img_size, self.img_size,
                            device=img_a.device)
        mask = self.warper_full(ones_a, H2)
        mask = (mask > 0.5).float()

        return {
            "H1": H1,
            "H2": H2,
            "warped_A": warped_a,
            "warped_B": warped_b,
            "mask": mask,
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = HomographyNet(img_size=512, pretrained_backbone=False)
    ia = torch.randn(2, 3, 512, 512)
    ib = torch.randn(2, 3, 512, 512)
    out = net(ia, ib)
    for k, v in out.items():
        print(f"{k:12s}: {v.shape}")

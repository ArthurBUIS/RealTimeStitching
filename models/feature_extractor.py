"""
Feature extraction module for the unsupervised homography estimation network.

Architecture (Section 3.2.2):
  - Siamese ResNet-50 backbone (shared weights)
  - ECA (Efficient Channel Attention) after Layer0
  - CA  (Coordinate Attention) after each residual block (Layer1-3)
  - Outputs two feature pyramid levels:
      * F_1/8  (after Layer2)  – fine scale
      * F_1/16 (after Layer3)  – coarse scale
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ---------------------------------------------------------------------------
# ECA – Efficient Channel Attention  (Wang et al., CVPR 2020)
# ---------------------------------------------------------------------------

class ECAModule(nn.Module):
    """
    Global average pooling → 1-D convolution with adaptive kernel size k = ψ(C).
    Channel-wise sigmoid weights are multiplied back onto the feature map.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = odd( log2(C)/gamma + b/gamma )
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)               # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)                   # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# CA – Coordinate Attention  (Hou et al., CVPR 2021)
# ---------------------------------------------------------------------------

class CAModule(nn.Module):
    """
    Decomposes channel attention into two 1-D spatial encodings (X-pool / Y-pool)
    to capture long-range spatial dependencies with positional information.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv_1x1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # X-pool: average along W  → (B, C, H, 1)
        x_h = x.mean(dim=-1, keepdim=True)
        # Y-pool: average along H  → (B, C, 1, W)
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)  # (B,C,W,1)

        # Concatenate along spatial axis and encode jointly
        y = torch.cat([x_h, x_w], dim=2)   # (B, C, H+W, 1)
        y = self.act(self.bn(self.conv_1x1(y)))

        # Split back
        y_h, y_w = y[:, :, :H, :], y[:, :, H:, :]   # (B,mid,H,1), (B,mid,W,1)
        y_w = y_w.permute(0, 1, 3, 2)                # (B,mid,1,W)

        attn_h = self.sigmoid(self.conv_h(y_h))  # (B,C,H,1)
        attn_w = self.sigmoid(self.conv_w(y_w))  # (B,C,1,W)

        return x * attn_h * attn_w


# ---------------------------------------------------------------------------
# Siamese Feature Extractor
# ---------------------------------------------------------------------------

class SiameseFeatureExtractor(nn.Module):
    """
    Modified ResNet-50 Siamese backbone that produces two-level feature pyramids.

    Network stages (Figure 4):
        Layer0  : 7×7 conv + BN + ReLU + 3×3 MaxPool  → stride 4
        ECA     : applied after Layer0
        Layer1  : ResNet Stage 1 (64 channels)         → stride 4
        CA      : applied after Layer1
        Layer2  : ResNet Stage 2 (128 channels)        → stride 8   ← F_1/8
        CA      : applied after Layer2
        Layer3  : ResNet Stage 3 (256 channels)        → stride 16  ← F_1/16
        CA      : applied after Layer3

    Returns:
        feat_1_8  : (B, 128, H/8,  W/8)
        feat_1_16 : (B, 256, H/16, W/16)
    """

    # Output channel counts for each stage of ResNet-50
    _STAGE_CHANNELS = {
        "layer0": 64,
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
    }

    # We project to smaller channel counts to keep memory and compute light
    _PROJ_CHANNELS = {
        "layer2": 128,
        "layer3": 256,
    }

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        # ---- Stage 0: conv1 + bn1 + relu + maxpool ----
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.eca0 = ECAModule(self._STAGE_CHANNELS["layer0"])

        # ---- Residual stages ----
        self.layer1 = backbone.layer1
        self.ca1 = CAModule(self._STAGE_CHANNELS["layer1"])

        self.layer2 = backbone.layer2
        self.ca2 = CAModule(self._STAGE_CHANNELS["layer2"])
        self.proj2 = nn.Conv2d(
            self._STAGE_CHANNELS["layer2"], self._PROJ_CHANNELS["layer2"],
            kernel_size=1, bias=False
        )

        self.layer3 = backbone.layer3
        self.ca3 = CAModule(self._STAGE_CHANNELS["layer3"])
        self.proj3 = nn.Conv2d(
            self._STAGE_CHANNELS["layer3"], self._PROJ_CHANNELS["layer3"],
            kernel_size=1, bias=False
        )

    # The Siamese nature is achieved by calling forward() twice with the
    # same module (shared weights) – no need for separate branch objects.

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)  – H=W=512 expected

        Returns:
            feat_1_8  : (B, 128, H/8,  W/8)
            feat_1_16 : (B, 256, H/16, W/16)
        """
        x = self.layer0(x)          # (B, 64, H/4, W/4)
        x = self.eca0(x)

        x = self.layer1(x)          # (B, 256, H/4, W/4)
        x = self.ca1(x)

        x = self.layer2(x)          # (B, 512, H/8, W/8)
        x = self.ca2(x)
        feat_1_8 = self.proj2(x)

        x = self.layer3(x)          # (B, 1024, H/16, W/16)
        x = self.ca3(x)
        feat_1_16 = self.proj3(x)

        return feat_1_8, feat_1_16


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SiameseFeatureExtractor(pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    f8, f16 = model(dummy)
    print(f"F_1/8  shape : {f8.shape}")   # expect (2, 128, 64, 64)
    print(f"F_1/16 shape : {f16.shape}")  # expect (2, 256, 32, 32)

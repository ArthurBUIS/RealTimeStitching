"""
Loss functions for the unsupervised homography estimation network.

Section 3.2.3 of the paper:

    L_w = L_sim + λ * L_smooth        (eq. 1)

    L_sim   – L1 photometric loss over the valid overlapping region (eq. 2)
    L_smooth = L_inter + L_intra      (eq. 3)
        L_intra – penalises excessively long mesh edges (eq. 4)
        L_inter – preserves collinearity of adjacent grid edges (eq. 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Similarity Loss  (L_sim)
# ---------------------------------------------------------------------------

class SimilarityLoss(nn.Module):
    """
    Equation (2):
        L_sim = || M ⊙ (H(I_A) - I_B) ||_1

    where
        M        – binary overlap mask  (1 where both views are valid)
        H(I_A)   – warped reference image
        I_B      – target image
    """

    def forward(
        self,
        warped_a: torch.Tensor,   # (B, C, H, W)  H(I_A)
        img_b: torch.Tensor,      # (B, C, H, W)  I_B
        mask: torch.Tensor,       # (B, 1, H, W)  binary overlap mask
    ) -> torch.Tensor:
        diff = (warped_a - img_b).abs()          # (B, C, H, W)
        diff = diff * mask                        # zero-out outside overlap
        # Mean over all pixels (including zero-masked ones, consistent with
        # the standard UDIS formulation)
        return diff.mean()


# ---------------------------------------------------------------------------
# Smoothness Loss  (L_smooth = L_intra + L_inter)
# ---------------------------------------------------------------------------

class SmoothnessLoss(nn.Module):
    """
    Mesh-based geometric regularisation applied to the estimated deformation
    field (displacement grid).

    The homography is sampled onto a coarse G×G mesh grid; the displacement
    of each mesh vertex from its canonical position forms the deformation field
    on which the two terms operate.

    L_intra  (eq. 4): penalises edge lengths exceeding threshold η.
    L_inter  (eq. 5): penalises deviations from collinearity of neighbouring
                       edge pairs (encourages parallelism).
    """

    def __init__(self, grid_size: int = 8, eta: float = 1.5):
        """
        Args:
            grid_size : number of mesh divisions per side (G×G interior points)
            eta       : relaxation threshold for edge lengths (eq. 4)
        """
        super().__init__()
        self.G = grid_size
        self.eta = eta

    # ------------------------------------------------------------------ #
    # Helper: build mesh vertices and compute displacement from homography
    # ------------------------------------------------------------------ #

    def _get_displacements(
        self, H: torch.Tensor, img_size: int = 512
    ) -> torch.Tensor:
        """
        Sample the homography H at G×G grid points and return the
        per-vertex displacement vectors.

        Args:
            H        : (B, 3, 3)
            img_size : source image side length (assumed square)
        Returns:
            disp     : (B, G, G, 2)  displacement (dx, dy) in normalised [-1,1]
        """
        B = H.shape[0]
        G = self.G
        device = H.device

        # Canonical grid positions in [-1, 1]
        pts_1d = torch.linspace(-1.0, 1.0, G, device=device)
        gy, gx = torch.meshgrid(pts_1d, pts_1d, indexing="ij")  # (G, G)
        ones = torch.ones_like(gx)
        grid_h = torch.stack([gx, gy, ones], dim=-1)            # (G, G, 3)
        grid_flat = grid_h.view(-1, 3).unsqueeze(-1)             # (G*G, 3, 1)
        grid_flat = grid_flat.unsqueeze(0).expand(B, -1, -1, -1) # (B,G*G,3,1)

        # Apply H
        H_exp = H.unsqueeze(1).expand(-1, G * G, -1, -1)        # (B,G*G,3,3)
        mapped = torch.matmul(H_exp, grid_flat).squeeze(-1)      # (B,G*G,3)
        mapped_xy = mapped[..., :2] / (mapped[..., 2:3] + 1e-8) # (B,G*G,2)
        mapped_xy = mapped_xy.view(B, G, G, 2)

        # Canonical positions
        canon = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        disp = mapped_xy - canon   # (B, G, G, 2)
        return disp

    # ------------------------------------------------------------------ #
    # L_intra  (eq. 4)
    # ------------------------------------------------------------------ #

    def _l_intra(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Penalise mesh edge lengths that exceed threshold η.

        Edges are taken along the horizontal (right-neighbour) and vertical
        (down-neighbour) directions.

        Args:
            disp : (B, G, G, 2)
        """
        # Vertex positions = canon + disp, but since we only care about
        # edge vectors, we compute them directly from the absolute mapped
        # positions stored in disp (Δ relative to canon, which has unit spacing).
        G = self.G
        eta = self.eta

        # Horizontal edges: vertex (i,j) → (i, j+1)
        eh = disp[:, :, 1:, :] - disp[:, :, :-1, :]  # (B, G, G-1, 2)
        # Add the canonical spacing (2/(G-1) in normalised coords)
        spacing = 2.0 / (G - 1) if G > 1 else 1.0
        eh = eh + torch.tensor([spacing, 0.0], device=disp.device)

        # Vertical edges: vertex (i,j) → (i+1, j)
        ev = disp[:, 1:, :, :] - disp[:, :-1, :, :]  # (B, G-1, G, 2)
        ev = ev + torch.tensor([0.0, spacing], device=disp.device)

        # Edge lengths
        len_h = eh.norm(dim=-1)   # (B, G, G-1)
        len_v = ev.norm(dim=-1)   # (B, G-1, G)

        l_intra = F.relu(len_h - eta).mean() + F.relu(len_v - eta).mean()
        return l_intra

    # ------------------------------------------------------------------ #
    # L_inter  (eq. 5)
    # ------------------------------------------------------------------ #

    def _l_inter(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Preserve collinearity / parallelism of adjacent edge pairs.

        For each interior vertex, we take the two horizontal neighbours and
        the two vertical neighbours, forming adjacent edge pairs.
        The penalty is 1 − cos(e1, e2).

        Args:
            disp : (B, G, G, 2)
        """
        G = self.G
        spacing = 2.0 / (G - 1) if G > 1 else 1.0

        # ---- Horizontal ----
        # Edges from (i, j-1)→(i,j) and (i, j)→(i, j+1)  for interior j
        e_h_left  = (disp[:, :, 1:, :]  - disp[:, :, :-1, :])  # (B,G,G-1,2)
        e_h_left  = e_h_left + torch.tensor([spacing, 0.0], device=disp.device)
        e_h_right = e_h_left[:, :, 1:, :]  # (B,G,G-2,2)
        e_h_left  = e_h_left[:, :, :-1, :] # (B,G,G-2,2)

        cos_h = self._cosine_penalty(e_h_left, e_h_right)

        # ---- Vertical ----
        e_v_up   = (disp[:, 1:, :, :]  - disp[:, :-1, :, :])   # (B,G-1,G,2)
        e_v_up   = e_v_up + torch.tensor([0.0, spacing], device=disp.device)
        e_v_down = e_v_up[:, 1:, :, :]   # (B,G-2,G,2)
        e_v_up   = e_v_up[:, :-1, :, :]  # (B,G-2,G,2)

        cos_v = self._cosine_penalty(e_v_up, e_v_down)

        return cos_h.mean() + cos_v.mean()

    @staticmethod
    def _cosine_penalty(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        Equation (5): 1 - (e1 · e2) / (||e1|| ||e2||)
        """
        dot = (e1 * e2).sum(dim=-1)
        norm1 = e1.norm(dim=-1).clamp(min=1e-8)
        norm2 = e2.norm(dim=-1).clamp(min=1e-8)
        return 1.0 - dot / (norm1 * norm2)

    # ------------------------------------------------------------------ #

    def forward(self, H: torch.Tensor, img_size: int = 512) -> torch.Tensor:
        """
        Args:
            H        : (B, 3, 3) estimated homography
            img_size : input image side length
        Returns:
            L_smooth scalar
        """
        disp = self._get_displacements(H, img_size)
        return self._l_intra(disp) + self._l_inter(disp)


# ---------------------------------------------------------------------------
# Combined Total Loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    L_w = L_sim + λ * L_smooth        (eq. 1)

    Paper uses λ = 10.
    """

    def __init__(self, lambda_smooth: float = 10.0, grid_size: int = 8,
                 eta: float = 1.5):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.sim_loss = SimilarityLoss()
        self.smooth_loss = SmoothnessLoss(grid_size=grid_size, eta=eta)

    def forward(
        self,
        warped_a: torch.Tensor,   # H(I_A)
        img_b: torch.Tensor,      # I_B
        mask: torch.Tensor,       # overlap mask
        H: torch.Tensor,          # estimated homography (for smoothness)
        img_size: int = 512,
    ) -> dict:
        l_sim = self.sim_loss(warped_a, img_b, mask)
        l_smooth = self.smooth_loss(H, img_size)
        l_total = l_sim + self.lambda_smooth * l_smooth
        return {
            "loss": l_total,
            "l_sim": l_sim.detach(),
            "l_smooth": l_smooth.detach(),
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 2
    loss_fn = TotalLoss(lambda_smooth=10.0)

    warped_a = torch.randn(B, 3, 512, 512)
    img_b = torch.randn(B, 3, 512, 512)
    mask = torch.ones(B, 1, 512, 512)
    H = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    H += torch.randn(B, 3, 3) * 0.05   # small perturbation

    out = loss_fn(warped_a, img_b, mask, H)
    print("Total loss :", out["loss"].item())
    print("L_sim      :", out["l_sim"].item())
    print("L_smooth   :", out["l_smooth"].item())

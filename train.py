"""
Training script for the attention-enhanced unsupervised homography network.

Two-stage training (Section 4.1.1):
    Stage 1 : 100 epochs on Warped MS-COCO
    Stage 2 :  50 epochs fine-tuning on composite dataset (UDIS-D + industrial)

Optimiser : Adam  lr=1e-4, β1=0.9, β2=0.999, ε=1e-8
Scheduler : ExponentialLR  γ=0.97 per epoch

Usage:
    python train.py \
        --coco_root    /data/WarpedCOCO \
        --udis_root    /data/UDIS-D \
        --industrial_root /data/industrial \
        --output_dir   checkpoints/

    # Resume from checkpoint
    python train.py ... --resume checkpoints/latest.pth
"""

import argparse
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from models.homography_net import HomographyNet
from losses import TotalLoss
from dataset import make_coco_dataset, make_composite_dataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Training / validation for a single epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: TotalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool = True,
    epoch: int = 0,
    log_interval: int = 50,
) -> dict:

    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    meters = {k: AverageMeter() for k in ("loss", "l_sim", "l_smooth")}
    t0 = time.time()

    with ctx:
        for step, batch in enumerate(loader):
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)

            out = model(img_a, img_b)
            loss_dict = loss_fn(
                warped_a=out["warped_A"],
                img_b=img_b,
                mask=out["mask"],
                H=out["H2"],
                img_size=img_a.shape[-1],
            )

            if train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            B = img_a.size(0)
            for k in meters:
                meters[k].update(loss_dict[k].item(), B)

            if train and (step + 1) % log_interval == 0:
                elapsed = time.time() - t0
                log.info(
                    f"Epoch {epoch:03d} | step {step+1:04d}/{len(loader):04d} | "
                    f"loss={meters['loss'].avg:.4f}  "
                    f"l_sim={meters['l_sim'].avg:.4f}  "
                    f"l_smooth={meters['l_smooth'].avg:.4f}  "
                    f"({elapsed:.1f}s)"
                )
                t0 = time.time()

    return {k: v.avg for k, v in meters.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: ExponentialLR):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_stage = ckpt.get("stage", 1)
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt.get("best_val", float("inf"))
    log.info(f"Resumed from {path}  (stage {start_stage}, epoch {start_epoch})")
    return start_stage, start_epoch, best_val


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    )
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    model = HomographyNet(
        img_size=args.img_size,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {total_params:,}")

    # ---- Loss ----
    loss_fn = TotalLoss(
        lambda_smooth=args.lambda_smooth,
        grid_size=args.grid_size,
        eta=args.eta,
    )

    # ---- Optimiser & scheduler ----
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    # ---- Optional resume ----
    start_stage, start_epoch, best_val = 1, 1, float("inf")
    if args.resume:
        start_stage, start_epoch, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # ====================================================================
    # STAGE 1 – Warped MS-COCO pre-training
    # ====================================================================
    if start_stage <= 1:
        log.info("=== Stage 1: Warped MS-COCO pre-training ===")

        train_ds = make_coco_dataset(args.coco_root, split="train",
                                     img_size=args.img_size, augment=True)
        val_ds   = make_coco_dataset(args.coco_root, split="test",
                                     img_size=args.img_size, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        stage1_start = start_epoch if start_stage == 1 else 1

        for epoch in range(stage1_start, args.stage1_epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, loss_fn, optimizer, device,
                train=True, epoch=epoch, log_interval=args.log_interval,
            )
            val_metrics = run_epoch(
                model, val_loader, loss_fn, optimizer, device,
                train=False, epoch=epoch,
            )

            scheduler.step()

            log.info(
                f"[Stage 1] Epoch {epoch:03d}/{args.stage1_epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Save latest
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "stage": 1, "epoch": epoch,
                    "best_val": best_val,
                },
                os.path.join(args.output_dir, "stage1_latest.pth"),
            )

            # Save best
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    {"model": model.state_dict(), "stage": 1, "epoch": epoch},
                    os.path.join(args.output_dir, "stage1_best.pth"),
                )

        log.info("Stage 1 complete.")

    # ====================================================================
    # STAGE 2 – Composite dataset fine-tuning
    # ====================================================================
    log.info("=== Stage 2: Composite fine-tuning ===")

    train_ds2 = make_composite_dataset(
        udis_root=args.udis_root,
        industrial_root=args.industrial_root,
        split="train",
        img_size=args.img_size,
        augment=True,
    )
    val_ds2 = make_composite_dataset(
        udis_root=args.udis_root,
        industrial_root=args.industrial_root,
        split="test",
        img_size=args.img_size,
        augment=False,
    )

    train_loader2 = DataLoader(
        train_ds2, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader2 = DataLoader(
        val_ds2, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Reset scheduler for stage 2 (lower LR for fine-tuning)
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr * 0.1
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    stage2_start = start_epoch if start_stage == 2 else 1
    best_val = float("inf")

    for epoch in range(stage2_start, args.stage2_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader2, loss_fn, optimizer, device,
            train=True, epoch=epoch, log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model, val_loader2, loss_fn, optimizer, device,
            train=False, epoch=epoch,
        )

        scheduler.step()

        log.info(
            f"[Stage 2] Epoch {epoch:03d}/{args.stage2_epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "stage": 2, "epoch": epoch,
                "best_val": best_val,
            },
            os.path.join(args.output_dir, "stage2_latest.pth"),
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                {"model": model.state_dict(), "stage": 2, "epoch": epoch},
                os.path.join(args.output_dir, "stage2_best.pth"),
            )

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train attention-enhanced unsupervised homography network"
    )

    # Data
    p.add_argument("--coco_root",       required=True,
                   help="Root of Warped MS-COCO dataset")
    p.add_argument("--udis_root",       required=True,
                   help="Root of UDIS-D dataset")
    p.add_argument("--industrial_root", required=True,
                   help="Root of industrial image pairs")
    p.add_argument("--output_dir",      default="checkpoints",
                   help="Where to save checkpoints")

    # Training
    p.add_argument("--stage1_epochs",   type=int, default=100)
    p.add_argument("--stage2_epochs",   type=int, default=50)
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--log_interval",    type=int, default=50,
                   help="Print metrics every N steps")

    # Model / loss
    p.add_argument("--img_size",            type=int, default=512)
    p.add_argument("--lambda_smooth",       type=float, default=10.0)
    p.add_argument("--grid_size",           type=int, default=8)
    p.add_argument("--eta",                 type=float, default=1.5)
    p.add_argument("--pretrained_backbone", action="store_true", default=True)
    p.add_argument("--no_pretrained",       dest="pretrained_backbone",
                   action="store_false")

    # Misc
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    p.add_argument("--cpu",    action="store_true", help="Force CPU training")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

"""
train.py — Training script for the Sea Ice Reasoning Segmentation pipeline.

Features:
  - Mixed precision (FP16/BF16)
  - Gradient accumulation
  - Learning rate scheduling with warmup
  - Checkpointing with best model tracking
  - WandB logging (optional)
  - Evaluation every N steps
  - Resumable from checkpoint

Usage:
    python train.py --config config.py --output_dir outputs/exp1
"""

import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import cfg, Config
from data.dataset import build_dataloaders
from models.pipeline import SeaIceSegmentationPipeline
from utils.losses import SeaIceLoss
from utils.metrics import MetricAccumulator


# ─── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, train_cfg, num_training_steps):
    """Cosine LR schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_steps = int(num_training_steps * train_cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    best_metric: float,
    save_path: Path,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path: Path,
    device: str,
) -> Dict:
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt["scaler_state_dict"]:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"✓ Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
    return ckpt


# ─── Training step ────────────────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    batch: Dict,
    criterion: nn.Module,
    optimizer,
    scaler: Optional[GradScaler],
    device: str,
    use_amp: bool,
) -> Dict:
    """Single training step."""
    images = batch["image"].to(device)              # (B, 3, H, W)
    masks = batch["mask"].to(device)                # (B, 1, H, W)
    labels = batch["label"].to(device)              # (B,)
    descriptions = batch["long_desc"]               # List[str]

    # Convert images to uint8 numpy for SAM (if using SAM)
    if model.use_sam:
        images_np = [
            (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            for img in images
        ]
    else:
        images_np = None

    # path layout: .../dataset/<class>/images/<file> → parent.parent.name = class name
    seq_ids = [str(Path(p).parent.parent.name) for p in batch["image_path"]]

    # Forward pass — request aux logits for deep supervision during training
    with autocast('cuda', enabled=use_amp):
        outputs = model(
            images=images,
            descriptions=descriptions,
            images_np=images_np,
            sequence_ids=seq_ids,
            return_aux=True,
        )
        loss_dict = criterion(
            outputs=outputs,
            targets={"mask": masks, "label": labels},
        )
        loss = loss_dict["loss"]

    # NaN guard: skip batch and clear temporal bank to prevent poisoning
    if not torch.isfinite(loss):
        model.temporal.reset_all()
        return None

    # Backward pass
    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return {
        "loss": loss.item(),
        "loss_mask": loss_dict["loss_mask"].item(),
        "loss_cls": loss_dict["loss_cls"].item(),
        "loss_attn": loss_dict["loss_attn"].item(),
        "loss_aux": loss_dict["loss_aux"].item(),
        "outputs": outputs,
    }


# ─── Validation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict:
    """Run validation and return metrics."""
    model.eval()
    metrics = MetricAccumulator()

    # Use "val_" prefix so validation sequences never mix with training banks.
    # Clean up val-specific bank entries after the loop.
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        descriptions = batch["long_desc"]

        images_np = None
        if model.use_sam:
            images_np = [
                (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for img in images
            ]

        seq_ids = ["val_" + str(Path(p).parent.parent.name)
                   for p in batch["image_path"]]

        outputs = model(
            images=images,
            descriptions=descriptions,
            images_np=images_np,
            sequence_ids=seq_ids,
        )
        loss_dict = criterion(outputs, {"mask": masks, "label": labels})

        metrics.update(
            outputs=outputs,
            targets={"mask": masks, "label": labels},
            loss=loss_dict["loss"].item(),
        )

    # Purge val_ entries to keep the bank from growing unboundedly
    for key in list(model.temporal._banks.keys()):
        if key.startswith("val_"):
            del model.temporal._banks[key]

    model.train()
    return metrics.compute()


# ─── Main training loop ───────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    train_cfg,
    output_dir: Path,
    wandb_run=None,
    start_epoch: int = 0,
    start_step: int = 0,
    best_metric: float = 0.0,
):
    device = train_cfg.device
    epochs = train_cfg.epochs
    grad_accum = train_cfg.grad_accum_steps
    use_amp = train_cfg.fp16 or train_cfg.bf16
    patience = getattr(train_cfg, "early_stop_patience", 6)
    clip_norm = getattr(train_cfg, "grad_clip_norm", 1.0)

    global_step = start_step
    # Segmentation is the primary task, so the best checkpoint and early
    # stopping are driven by mIoU. F1 is tracked and reported separately
    # (as in standard segmentation papers — the two are not combined).
    best_miou = best_metric       # passed in as 0.0 on fresh runs
    best_f1 = 0.0                 # reported separately, not used for stopping
    no_improve_count = 0          # early-stopping counter (on mIoU)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_metrics = MetricAccumulator()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            step_result = train_step(
                model, batch, criterion, optimizer, scaler, device, use_amp
            )

            # NaN batch: gradients are already zeroed inside train_step
            if step_result is None:
                optimizer.zero_grad()
                continue

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Update metrics
            epoch_metrics.update(
                outputs=step_result["outputs"],
                targets={
                    "mask": batch["mask"].to(device),
                    "label": batch["label"].to(device),
                },
                loss=step_result["loss"],
            )

            # Logging
            if global_step % train_cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{step_result['loss']:.4f}",
                    "aux": f"{step_result.get('loss_aux', 0.0):.4f}",
                    "lr": f"{lr:.2e}",
                })
                if wandb_run:
                    wandb_run.log({
                        "train/loss": step_result["loss"],
                        "train/loss_mask": step_result["loss_mask"],
                        "train/loss_cls": step_result["loss_cls"],
                        "train/loss_attn": step_result["loss_attn"],
                        "train/loss_aux": step_result.get("loss_aux", 0.0),
                        "train/lr": lr,
                        "step": global_step,
                    })

            # Validation
            if global_step % train_cfg.eval_every == 0 and global_step > 0:
                val_metrics = validate(model, val_loader, criterion, device)
                val_iou = val_metrics["mean_iou"]
                val_f1  = val_metrics["weighted_f1"]

                print(f"\n{'='*60}")
                print(f"Validation @ step {global_step}")
                print(val_metrics.get("classification_report", ""))
                print(f"mIoU: {val_iou:.4f} (best={best_miou:.4f}) | "
                      f"F1: {val_f1:.4f} (best={best_f1:.4f})")
                print(f"No-improve count (mIoU): {no_improve_count}/{patience}")
                print(f"{'='*60}\n")

                if wandb_run:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()
                                   if isinstance(v, (int, float))}, step=global_step)

                # Save best-F1 checkpoint separately (classification reporting)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_miou,
                        output_dir / "best_model_f1.pth",
                    )
                    print(f"  ✓ New best F1={val_f1:.4f} (mIoU here={val_iou:.4f})")

                # Best model + early stopping driven by mIoU (primary task)
                if val_iou > best_miou:
                    best_miou = val_iou
                    no_improve_count = 0
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_miou,
                        output_dir / "best_model.pth",
                    )
                    print(f"  ✓ New best mIoU={val_iou:.4f} (F1 here={val_f1:.4f})")
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"\n⏹  Early stopping: mIoU did not improve for "
                              f"{patience} evaluations. "
                              f"Best mIoU={best_miou:.4f}  Best F1={best_f1:.4f}")
                        return best_miou

            # Periodic checkpoint
            if global_step % train_cfg.save_every == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_miou,
                    output_dir / f"checkpoint_step_{global_step}.pth",
                )
                # Keep only last N checkpoints
                cleanup_old_checkpoints(output_dir, keep_last=train_cfg.keep_last_n)

        # End-of-epoch summary
        train_summary = epoch_metrics.compute()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary")
        print(f"{'='*60}")
        print(f"Train Loss: {train_summary['mean_loss']:.4f}")
        print(f"Train mIoU: {train_summary['mean_iou']:.4f}")
        print(f"Train F1:   {train_summary['weighted_f1']:.4f}")
        print(f"{'='*60}\n")

    print(f"\nTraining complete! Best mIoU: {best_miou:.4f} | Best F1: {best_f1:.4f}")
    return best_miou


def cleanup_old_checkpoints(output_dir: Path, keep_last: int = 3):
    """Keep only the N most recent checkpoints."""
    ckpts = sorted(
        output_dir.glob("checkpoint_step_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for ckpt in ckpts[:-keep_last]:
        ckpt.unlink()
        print(f"Removed old checkpoint: {ckpt.name}")


# ─── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset",
                        help="Path to dataset root folder")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use_sam", action="store_true",
                        help="Use SAM for mask decoding (else lightweight decoder)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sea-ice-seg")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Override config if CLI args provided
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.lr:
        cfg.train.lr = args.lr
    cfg.data.data_root = args.data_root
    cfg.train.output_dir = str(output_dir)

    device = cfg.train.device
    print(f"Device: {device}")
    print(f"Using SAM: {args.use_sam}")

    # WandB
    wandb_run = None
    if args.use_wandb or cfg.train.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(cfg),
            name=f"seaice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # Data
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data, cfg.train)

    # Model
    print("Building model...")
    model = SeaIceSegmentationPipeline(cfg.model, use_sam=args.use_sam).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    param_groups = model.get_param_groups(cfg.model, cfg.train)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas,
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // cfg.train.grad_accum_steps
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = get_lr_scheduler(optimizer, cfg.train, total_steps)

    # Mixed precision scaler
    scaler = None
    if cfg.train.fp16:
        scaler = GradScaler('cuda')

    # Loss
    criterion = SeaIceLoss(cfg.train).to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    start_step = 0
    best_metric = 0.0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, scheduler, scaler,
                                Path(args.resume), device)
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_metric = ckpt["best_metric"]

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_cfg=cfg.train,
        output_dir=output_dir,
        wandb_run=wandb_run,
        start_epoch=start_epoch,
        start_step=start_step,
        best_metric=best_metric,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()

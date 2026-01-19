"""
Train a standard SiT teacher model on CIFAR-10.

This is Step 1 of distillation:
1. Train teacher (this script)
2. Generate trajectories from teacher
3. Train unrolled student on trajectories

Usage:
    torchrun --nproc_per_node=4 train_teacher.py --data-path ./data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import sys
from copy import deepcopy
from collections import OrderedDict
from datetime import datetime

from models import SiT, SiT_models


# ============================================================================
# CIFAR-10 adapted SiT (pixel space, 32x32)
# ============================================================================

def SiT_S_CIFAR(num_classes=10, **kwargs):
    """SiT-S adapted for CIFAR-10 (pixel space)."""
    return SiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=num_classes,
        learn_sigma=False,
        **kwargs
    )


def SiT_B_CIFAR(num_classes=10, **kwargs):
    """SiT-B adapted for CIFAR-10 (pixel space)."""
    return SiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=num_classes,
        learn_sigma=False,
        **kwargs
    )


# ============================================================================
# Training utilities
# ============================================================================

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def safe_save(obj, path):
    """
    Save checkpoint atomically to avoid corruption.
    Saves to temp file first, then renames.
    """
    path = Path(path)
    tmp_path = path.with_suffix('.tmp')
    try:
        torch.save(obj, tmp_path)
        tmp_path.rename(path)
        return True
    except Exception as e:
        print_flush(f"  WARNING: Failed to save checkpoint: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass
        return False


# ============================================================================
# Flow Matching Loss (Rectified Flow)
# ============================================================================

class FlowMatchingLoss:
    """
    Rectified Flow / Flow Matching loss.

    Forward process: x_t = t * x1 + (1 - t) * x0
    where x0 ~ N(0, I), x1 ~ data, t ~ U(0, 1)

    Target velocity: v = x1 - x0
    """

    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, model, x1, y):
        """
        Compute flow matching loss.

        Args:
            model: velocity prediction model v(x_t, t, y)
            x1: clean images (B, C, H, W)
            y: class labels (B,)

        Returns:
            loss: scalar
        """
        B = x1.shape[0]
        device = x1.device

        # Sample noise
        x0 = torch.randn_like(x1)

        # Sample timesteps uniformly
        t = torch.rand(B, device=device) * (1 - 2 * self.eps) + self.eps  # [eps, 1-eps]

        # Interpolate: x_t = t * x1 + (1 - t) * x0
        t_expand = t.view(B, 1, 1, 1)
        x_t = t_expand * x1 + (1 - t_expand) * x0

        # Target velocity: v = x1 - x0
        v_target = x1 - x0

        # Predict velocity
        v_pred = model(x_t, t, y)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss


# ============================================================================
# ODE Sampler for generation
# ============================================================================

@torch.no_grad()
def sample_ode(model, shape, y, device, num_steps=50, cfg_scale=1.0):
    """
    Sample from the model using Euler ODE solver.

    Args:
        model: velocity prediction model
        shape: (B, C, H, W)
        y: class labels (B,)
        device: torch device
        num_steps: number of ODE steps
        cfg_scale: classifier-free guidance scale

    Returns:
        samples: (B, C, H, W)
    """
    B = shape[0]

    # Start from noise
    x = torch.randn(shape, device=device)

    # Time steps from 0 to 1
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]

        t_batch = torch.full((B,), t, device=device)

        if cfg_scale > 1.0:
            # Classifier-free guidance
            v_cond = model(x, t_batch, y)
            v_uncond = model(x, t_batch, torch.full_like(y, 10))  # null class
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, y)

        # Euler step
        x = x + v * dt

    return x


# ============================================================================
# Training loop
# ============================================================================

def train(args):
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print_flush("=" * 60)
        print_flush("  SiT Teacher Training on CIFAR-10")
        print_flush("=" * 60)
        print_flush(f"  GPUs: {world_size}")
        print_flush(f"  Model: {args.model}")
        print_flush(f"  Batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        print_flush(f"  Epochs: {args.epochs}")
        print_flush(f"  Learning rate: {args.lr}")
        print_flush("=" * 60)

    # Seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Model
    if args.model == "small":
        model = SiT_S_CIFAR(num_classes=10)
    elif args.model == "base":
        model = SiT_B_CIFAR(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print_flush(f"  Parameters: {num_params:,}")

    # EMA
    ema_model = None
    if args.use_ema and rank == 0:
        ema_model = deepcopy(model.module).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print_flush(f"  Dataset: {len(train_dataset):,} images")
        print_flush(f"  Batches/epoch: {len(train_loader)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss
    loss_fn = FlowMatchingLoss()

    # Log directory
    log_dir = Path(args.log_dir)
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume: find latest checkpoint if not specified
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        latest_ckpt = log_dir / "teacher_latest.pt"
        if latest_ckpt.exists():
            resume_path = str(latest_ckpt)
            if rank == 0:
                print_flush(f"Auto-resume: found {resume_path}")

    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    best_loss = float('inf')

    if resume_path:
        if rank == 0:
            print_flush(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        global_step = ckpt.get('global_step', 0)
        if ema_model is not None and ckpt.get('ema_state_dict'):
            ema_model.load_state_dict(ckpt['ema_state_dict'])
        if rank == 0:
            print_flush(f"  Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                       dynamic_ncols=True, file=sys.stdout)
        else:
            pbar = train_loader

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss = loss_fn(model, x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_model is not None:
                update_ema(ema_model, model.module, args.ema_decay)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch stats
        avg_loss = epoch_loss / num_batches
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = avg_loss_tensor.item() / world_size

        if rank == 0:
            print_flush(f"\nEpoch {epoch}: loss = {avg_loss_global:.6f}")

            # Save checkpoint (only keep best and latest)
            is_best = avg_loss_global < best_loss
            if is_best:
                best_loss = avg_loss_global

            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'ema_state_dict': ema_model.state_dict() if ema_model else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss_global,
                'best_loss': best_loss,
                'args': vars(args),
            }
            # Always save latest (overwrite)
            safe_save(ckpt, log_dir / "teacher_latest.pt")

            if is_best:
                safe_save(ckpt, log_dir / "teacher_best.pt")
                print_flush(f"  New best model! Loss: {avg_loss_global:.6f}")

            # Generate samples
            if epoch % args.sample_every == 0:
                sample_model = ema_model if ema_model else model.module
                sample_model.eval()

                z_shape = (64, 3, 32, 32)
                y_sample = torch.randint(0, 10, (64,), device=device)

                samples = sample_ode(sample_model, z_shape, y_sample, device,
                                    num_steps=50, cfg_scale=args.cfg_scale)
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)

                save_image(samples, log_dir / f"teacher_samples_epoch{epoch}.png", nrow=8)
                print_flush(f"  Saved samples")

        dist.barrier()

    if rank == 0:
        print_flush("\nTeacher training complete!")
        print_flush(f"Best loss: {best_loss:.6f}")
        print_flush(f"Checkpoint: {log_dir / 'teacher_best.pt'}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./results/teacher")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true", default=True,
                        help="Auto-resume from latest checkpoint (default: True)")
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume",
                        help="Disable auto-resume")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
Multi-GPU training script for Unrolled SiT using PyTorch DDP.

Usage:
    # Auto-detect GPUs and run
    ./train.sh

    # Or manually with torchrun
    torchrun --nproc_per_node=4 train_ddp.py --data-path ./data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import os
import sys
from copy import deepcopy
from collections import OrderedDict
from datetime import datetime

from unrolled_sit import UnrolledSiT_S, UnrolledSiT_B
from trajectory_dataset import (
    TeacherTrajectoryDataset,
    compute_uniform_weights,
    compute_snr_weights,
)


def print_flush(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_logging(log_dir, rank):
    """Setup logging for main process only."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"{log_dir}/train.log")
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def save_checkpoint(model, ema_model, optimizer, epoch, loss, global_step, path, args):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'ema_state_dict': ema_model.state_dict() if ema_model else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args),
    }
    torch.save(checkpoint, path)


def train(args):
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Logging
    logger = setup_logging(args.log_dir, rank)

    if rank == 0:
        print_flush("=" * 60)
        print_flush("Unrolled SiT - Distillation Training")
        print_flush("=" * 60)
        print_flush(f"World size: {world_size} GPUs")
        print_flush(f"Model: {args.model}")
        print_flush(f"Trajectories: {args.trajectory_dir}")
        print_flush(f"Batch size: {args.batch_size} (per GPU) x {world_size} = {args.batch_size * world_size} (total)")
        print_flush(f"Epochs: {args.epochs}")
        print_flush(f"Learning rate: {args.lr}")
        print_flush(f"Loss weighting: {args.loss_weighting}")
        print_flush(f"Log directory: {args.log_dir}")
        print_flush("=" * 60)

    # Seed for reproducibility
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Create model
    if args.model == "small":
        model = UnrolledSiT_S(num_classes=10)
    elif args.model == "base":
        model = UnrolledSiT_B(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print_flush(f"Model parameters: {num_params:,}")

    # EMA model (only on rank 0)
    ema_model = None
    if args.use_ema and rank == 0:
        ema_model = deepcopy(model.module).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # Loss weights
    num_steps = model.module.depth
    if args.loss_weighting == "uniform":
        loss_weights = compute_uniform_weights(num_steps)
    elif args.loss_weighting == "snr":
        loss_weights = compute_snr_weights(num_steps, gamma=args.snr_gamma)
    else:
        raise ValueError(f"Unknown loss weighting: {args.loss_weighting}")
    loss_weights = loss_weights.to(device)

    if rank == 0:
        print_flush(f"Loss weights: {[f'{w:.4f}' for w in loss_weights.tolist()]}")

    # Dataset - load pre-generated teacher trajectories
    train_dataset = TeacherTrajectoryDataset(
        trajectory_dir=args.trajectory_dir,
        limit=args.train_limit,
    )

    # Verify num_steps matches model depth
    if train_dataset.num_steps != num_steps:
        raise ValueError(
            f"Trajectory num_steps ({train_dataset.num_steps}) != model depth ({num_steps})"
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
        print_flush(f"Dataset size: {len(train_dataset):,}")
        print_flush(f"Batches per epoch: {len(train_loader)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    global_step = 0
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        layer_losses_sum = [0.0] * num_steps
        num_batches = 0

        # Progress bar only on rank 0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                       dynamic_ncols=True, file=sys.stdout)
        else:
            pbar = train_loader

        for batch in pbar:
            trajectory = batch['trajectory'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            x_input = trajectory[:, 0]
            targets = trajectory[:, 1:]

            # Forward
            _, intermediates = model(x_input, labels, return_intermediates=True)

            # Layer-wise loss
            total_loss = 0
            layer_losses = []
            for i, (pred, target) in enumerate(zip(intermediates, targets)):
                layer_loss = F.mse_loss(pred, target)
                layer_losses.append(layer_loss.item())
                total_loss = total_loss + loss_weights[i] * layer_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update EMA
            if ema_model is not None:
                update_ema(ema_model, model.module, args.ema_decay)

            # Track metrics
            epoch_loss += total_loss.item()
            for i, ll in enumerate(layer_losses):
                layer_losses_sum[i] += ll
            num_batches += 1
            global_step += 1

            # Update progress bar
            if rank == 0:
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        # Epoch stats
        avg_loss = epoch_loss / num_batches
        avg_layer_losses = [ll / num_batches for ll in layer_losses_sum]

        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = avg_loss_tensor.item() / world_size

        if rank == 0:
            print_flush(f"\nEpoch {epoch} complete:")
            print_flush(f"  Average loss: {avg_loss_global:.6f}")
            print_flush(f"  Layer losses: {[f'{l:.4f}' for l in avg_layer_losses]}")

            # Save checkpoint
            is_best = avg_loss_global < best_loss
            if is_best:
                best_loss = avg_loss_global

            if epoch % args.save_every == 0 or is_best:
                ckpt_path = Path(args.log_dir) / f"checkpoint_epoch{epoch}.pt"
                save_checkpoint(
                    model, ema_model, optimizer, epoch,
                    avg_loss_global, global_step, ckpt_path, args
                )
                print_flush(f"  Saved checkpoint: {ckpt_path}")

                if is_best:
                    best_path = Path(args.log_dir) / "checkpoint_best.pt"
                    save_checkpoint(
                        model, ema_model, optimizer, epoch,
                        avg_loss_global, global_step, best_path, args
                    )
                    print_flush(f"  New best model! Loss: {avg_loss_global:.6f}")

            # Generate samples
            if epoch % args.sample_every == 0:
                generate_samples(ema_model if ema_model else model.module,
                               device, args.log_dir, epoch)

        # Sync all processes
        dist.barrier()

    if rank == 0:
        print_flush("\nTraining complete!")
        print_flush(f"Best loss: {best_loss:.6f}")
        print_flush(f"Results saved to: {args.log_dir}")

    cleanup_distributed()


@torch.no_grad()
def generate_samples(model, device, log_dir, epoch, num_samples=64):
    """Generate and save sample images."""
    from torchvision.utils import save_image

    model.eval()
    z = torch.randn(num_samples, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (num_samples,), device=device)

    samples = model(z, y)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    save_path = Path(log_dir) / f"samples_epoch{epoch}.png"
    save_image(samples, save_path, nrow=8)
    print_flush(f"  Saved samples: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Unrolled SiT with DDP")

    # Data - teacher trajectories
    parser.add_argument("--trajectory-dir", type=str, required=True,
                        help="Path to pre-generated teacher trajectories")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Limit number of training samples")
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)

    # Loss
    parser.add_argument("--loss-weighting", type=str, default="uniform", choices=["uniform", "snr"])
    parser.add_argument("--snr-gamma", type=float, default=5.0)

    # Logging
    parser.add_argument("--log-dir", type=str, default="./results/unrolled_sit")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=10)

    args = parser.parse_args()

    # Add timestamp to log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = f"{args.log_dir}_{timestamp}"

    train(args)


if __name__ == "__main__":
    main()

"""
Baseline 2: Direct One-Step Prediction

Same architecture as Unrolled SiT, but only supervise the final output.
No intermediate layer supervision - this isolates the contribution of
layer-wise distillation.

Usage:
    torchrun --nproc_per_node=4 baselines/train_direct_onestep.py \
        --trajectory-dir ./results/trajectories
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
import sys
import os
from copy import deepcopy
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from unrolled_sit import UnrolledSiT_S, UnrolledSiT_B
from trajectory_dataset import TeacherTrajectoryDataset


def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def safe_save(obj, path):
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


def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print_flush("=" * 60)
        print_flush("Baseline: Direct One-Step Prediction")
        print_flush("=" * 60)
        print_flush(f"  World size: {world_size} GPUs")
        print_flush(f"  Model: {args.model}")
        print_flush(f"  Only final output supervision (no layer-wise loss)")
        print_flush("=" * 60)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Same architecture as Unrolled SiT
    if args.model == "small":
        model = UnrolledSiT_S(num_classes=10)
    else:
        model = UnrolledSiT_B(num_classes=10)

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
    train_dataset = TeacherTrajectoryDataset(
        trajectory_dir=args.trajectory_dir,
        limit=args.train_limit,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
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
        print_flush(f"  Dataset: {len(train_dataset):,}")
        print_flush(f"  Batches/epoch: {len(train_loader)}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Auto-resume
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        latest_ckpt = log_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            resume_path = str(latest_ckpt)
            if rank == 0:
                print_flush(f"Auto-resume: {resume_path}")

    start_epoch = 1
    global_step = 0
    best_loss = float('inf')

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        global_step = ckpt.get('global_step', 0)
        if ema_model and ckpt.get('ema_state_dict'):
            ema_model.load_state_dict(ckpt['ema_state_dict'])
        if rank == 0:
            print_flush(f"  Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0)

        for batch in pbar:
            trajectory = batch['trajectory'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            x_input = trajectory[:, 0]  # noise
            x_target = trajectory[:, -1]  # final output (clean)

            # Forward - only use final output
            x_pred = model(x_input, labels, return_intermediates=False)

            # Single MSE loss on final output only
            loss = F.mse_loss(x_pred, x_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_model:
                update_ema(ema_model, model.module, args.ema_decay)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = avg_loss_tensor.item() / world_size

        if rank == 0:
            print_flush(f"\nEpoch {epoch}: loss = {avg_loss_global:.6f}")

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
            safe_save(ckpt, log_dir / "checkpoint_latest.pt")

            if is_best:
                safe_save(ckpt, log_dir / "checkpoint_best.pt")
                print_flush(f"  New best! Loss: {avg_loss_global:.6f}")

            if epoch % args.sample_every == 0:
                generate_samples(ema_model if ema_model else model.module,
                               device, log_dir, epoch)

        dist.barrier()

    if rank == 0:
        print_flush(f"\nTraining complete! Best loss: {best_loss:.6f}")

    cleanup_distributed()


@torch.no_grad()
def generate_samples(model, device, log_dir, epoch, num_samples=64):
    from torchvision.utils import save_image
    model.eval()
    z = torch.randn(num_samples, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (num_samples,), device=device)
    samples = model(z, y)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    save_image(samples, Path(log_dir) / f"samples_epoch{epoch}.png", nrow=8)
    print_flush(f"  Saved samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="/scratch2/jiwan_chung/layerdistill/results/baseline_direct")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

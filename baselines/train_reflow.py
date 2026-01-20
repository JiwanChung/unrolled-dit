"""
Baseline 5: Rectified Flow Reflow

Straighten the ODE trajectories by:
1. Generate (x_0, x_1) pairs from teacher
2. Retrain flow model on direct x_0 → x_1 mapping
3. Optionally iterate this process (reflow multiple times)

After reflowing, the trajectories become straighter, enabling
fewer sampling steps.

Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"

Usage:
    python baselines/train_reflow.py \
        --teacher-ckpt ./results/teacher/teacher_best.pt \
        --log-dir ./results/baseline_reflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
from copy import deepcopy
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_teacher import SiT_S_CIFAR, SiT_B_CIFAR


def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


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
        print_flush(f"  WARNING: Failed to save: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


@torch.no_grad()
def generate_pairs(model, dataloader, device, num_samples, num_steps=50, cfg_scale=1.0):
    """
    Generate (x_0, x_1, y) pairs by running ODE from noise to data.

    Returns:
        x_0: noise samples (N, C, H, W)
        x_1: generated samples (N, C, H, W)
        y: labels (N,)
    """
    print_flush(f"Generating {num_samples} pairs with {num_steps} ODE steps...")

    x_0_list = []
    x_1_list = []
    y_list = []

    total = 0
    pbar = tqdm(total=num_samples, desc="Generating pairs")

    for x_real, y in dataloader:
        if total >= num_samples:
            break

        B = min(x_real.shape[0], num_samples - total)
        y = y[:B].to(device)

        # Start from noise
        x_0 = torch.randn(B, 3, 32, 32, device=device)
        x = x_0.clone()

        # Run ODE from t=0 to t=1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_batch = torch.full((B,), t.item(), device=device)

            if cfg_scale > 1.0:
                v_cond = model(x, t_batch, y)
                v_uncond = model(x, t_batch, torch.full_like(y, 10))
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t_batch, y)

            x = x + v * dt

        x_1 = x

        x_0_list.append(x_0.cpu())
        x_1_list.append(x_1.cpu())
        y_list.append(y.cpu())

        total += B
        pbar.update(B)

    pbar.close()

    x_0 = torch.cat(x_0_list, dim=0)[:num_samples]
    x_1 = torch.cat(x_1_list, dim=0)[:num_samples]
    y = torch.cat(y_list, dim=0)[:num_samples]

    return x_0, x_1, y


@torch.no_grad()
def sample_ode(model, shape, y, device, num_steps=50, cfg_scale=1.0):
    """Sample from model using Euler ODE."""
    B = shape[0]
    x = torch.randn(shape, device=device)
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_batch = torch.full((B,), t.item(), device=device)

        if cfg_scale > 1.0:
            v_cond = model(x, t_batch, y)
            v_uncond = model(x, t_batch, torch.full_like(y, 10))
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, y)

        x = x + v * dt

    return x


def train_reflow_iteration(model, x_0, x_1, y, device, args, log_dir, iteration):
    """
    Train one iteration of reflow.

    Given (x_0, x_1) pairs, train model to predict v = x_1 - x_0
    along the straight line x_t = t * x_1 + (1-t) * x_0.
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Reflow Iteration {iteration}")
    print_flush(f"{'='*60}")

    # Create dataset
    dataset = TensorDataset(x_0, x_1, y)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA
    ema_model = deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    best_loss = float('inf')

    for epoch in range(1, args.epochs_per_iter + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Iter {iteration} Epoch {epoch}")

        for x_0_batch, x_1_batch, y_batch in pbar:
            x_0_batch = x_0_batch.to(device, non_blocking=True)
            x_1_batch = x_1_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            B = x_0_batch.shape[0]

            # Sample random timesteps
            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1, 1)

            # Interpolate: x_t = t * x_1 + (1-t) * x_0
            x_t = t_expand * x_1_batch + (1 - t_expand) * x_0_batch

            # Target velocity: v = x_1 - x_0 (straight line)
            v_target = x_1_batch - x_0_batch

            # Predict velocity
            v_pred = model(x_t, t, y_batch)

            # MSE loss
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(ema_model, model, args.ema_decay)

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        print_flush(f"Iter {iteration} Epoch {epoch}: loss = {avg_loss:.6f}")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }
        safe_save(ckpt, log_dir / f"iter{iteration}_latest.pt")

        if is_best:
            safe_save(ckpt, log_dir / f"iter{iteration}_best.pt")

        # Generate samples at various step counts
        if epoch % args.sample_every == 0:
            for n_steps in [1, 2, 4, 8]:
                ema_model.eval()
                samples = sample_ode(
                    ema_model, (64, 3, 32, 32),
                    torch.randint(0, 10, (64,), device=device),
                    device, num_steps=n_steps, cfg_scale=1.0
                )
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)
                save_image(samples, log_dir / f"iter{iteration}_epoch{epoch}_{n_steps}steps.png", nrow=8)
            print_flush(f"  Saved samples at 1/2/4/8 steps")

    return ema_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_flush(f"Device: {device}")

    # Load teacher
    print_flush(f"Loading teacher from {args.teacher_ckpt}")
    if args.model == "small":
        model = SiT_S_CIFAR(num_classes=10)
    else:
        model = SiT_B_CIFAR(num_classes=10)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict']:
        model.load_state_dict(ckpt['ema_state_dict'])
        print_flush("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)

    # For generating initial pairs
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cifar_dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform
    )

    cifar_loader = DataLoader(
        cifar_dataset,
        batch_size=args.gen_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print_flush("=" * 60)
    print_flush("Baseline: Rectified Flow Reflow")
    print_flush(f"  Iterations: {args.num_iterations}")
    print_flush(f"  Pairs per iteration: {args.num_pairs}")
    print_flush("=" * 60)

    current_model = model

    for iteration in range(1, args.num_iterations + 1):
        # Generate (x_0, x_1) pairs using current model
        current_model.eval()
        x_0, x_1, y = generate_pairs(
            current_model, cifar_loader, device,
            num_samples=args.num_pairs,
            num_steps=args.gen_steps,
            cfg_scale=args.cfg_scale
        )

        # Save pairs for debugging
        if iteration == 1:
            vis_x0 = (x_0[:64] + 1) / 2
            vis_x1 = (x_1[:64] + 1) / 2
            save_image(vis_x0.clamp(0, 1), log_dir / f"iter{iteration}_x0.png", nrow=8)
            save_image(vis_x1.clamp(0, 1), log_dir / f"iter{iteration}_x1.png", nrow=8)

        # Create fresh model for this iteration (or continue from previous)
        if args.model == "small":
            new_model = SiT_S_CIFAR(num_classes=10)
        else:
            new_model = SiT_B_CIFAR(num_classes=10)

        # Initialize from current model
        new_model.load_state_dict(current_model.state_dict())
        new_model = new_model.to(device)

        # Train reflow
        current_model = train_reflow_iteration(
            new_model, x_0, x_1, y, device, args, log_dir, iteration
        )

    # Save final model
    final_ckpt = {
        'model_state_dict': current_model.state_dict(),
        'num_iterations': args.num_iterations,
    }
    safe_save(final_ckpt, log_dir / "final_reflowed.pt")

    print_flush(f"\n{'='*60}")
    print_flush("Reflow complete!")
    print_flush(f"Final model: {log_dir / 'final_reflowed.pt'}")
    print_flush(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="/scratch2/jiwan_chung/layerdistill/results/baseline_reflow")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])

    # Reflow config
    parser.add_argument("--num-iterations", type=int, default=2,
                        help="Number of reflow iterations")
    parser.add_argument("--num-pairs", type=int, default=50000,
                        help="Number of (x_0, x_1) pairs per iteration")
    parser.add_argument("--gen-steps", type=int, default=50,
                        help="ODE steps for generating pairs")
    parser.add_argument("--gen-batch-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.5)

    # Training config
    parser.add_argument("--epochs-per-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)

"""
Generate teacher trajectories for distillation.

This is Step 2 of distillation:
1. Train teacher (train_teacher.py)
2. Generate trajectories (this script)
3. Train unrolled student on trajectories

Usage:
    python generate_trajectories.py \
        --teacher-ckpt ./results/teacher/teacher_best.pt \
        --output-dir ./trajectories \
        --num-samples 50000 \
        --num-steps 12
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os

from train_teacher import SiT_S_CIFAR, SiT_B_CIFAR


@torch.no_grad()
def generate_trajectory_ode(model, x0, y, num_steps, device):
    """
    Generate a trajectory by running the ODE from noise to data.

    Args:
        model: trained velocity model v(x_t, t, y)
        x0: starting noise (B, C, H, W)
        y: class labels (B,)
        num_steps: number of ODE steps (= number of trajectory points - 1)
        device: torch device

    Returns:
        trajectory: (B, num_steps+1, C, H, W) - from noise to generated image
        timesteps: (num_steps+1,) - timestep for each point
    """
    B = x0.shape[0]

    # Timesteps from 0 to 1
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    trajectory = [x0.clone()]
    x = x0.clone()

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]

        t_batch = torch.full((B,), t, device=device)

        # Predict velocity
        v = model(x, t_batch, y)

        # Euler step
        x = x + v * dt

        trajectory.append(x.clone())

    # Stack: (B, num_steps+1, C, H, W)
    trajectory = torch.stack(trajectory, dim=1)

    return trajectory, timesteps


@torch.no_grad()
def generate_trajectory_with_cfg(model, x0, y, num_steps, device, cfg_scale=1.5):
    """
    Generate trajectory with classifier-free guidance.
    """
    B = x0.shape[0]
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    trajectory = [x0.clone()]
    x = x0.clone()

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]

        t_batch = torch.full((B,), t, device=device)

        # CFG
        v_cond = model(x, t_batch, y)
        v_uncond = model(x, t_batch, torch.full_like(y, 10))  # null class = 10
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x = x + v * dt
        trajectory.append(x.clone())

    trajectory = torch.stack(trajectory, dim=1)
    return trajectory, timesteps


def load_teacher(checkpoint_path, model_type, device):
    """Load trained teacher model."""
    if model_type == "small":
        model = SiT_S_CIFAR(num_classes=10)
    elif model_type == "base":
        model = SiT_B_CIFAR(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try EMA weights first
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
        model.load_state_dict(ckpt['ema_state_dict'])
        print(f"Loaded EMA weights from {checkpoint_path}")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded weights from {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load teacher
    print(f"Loading teacher from {args.teacher_ckpt}")
    teacher = load_teacher(args.teacher_ckpt, args.model, device)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 dataset (for labels and reference clean images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform
    )

    # We generate trajectories from random noise with dataset labels
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"Generating {args.num_samples} trajectories with {args.num_steps} steps each")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Output directory: {output_dir}")

    # Generate trajectories
    total_generated = 0
    batch_idx = 0

    pbar = tqdm(total=args.num_samples, desc="Generating trajectories")

    for x_real, y in dataloader:
        if total_generated >= args.num_samples:
            break

        B = min(x_real.shape[0], args.num_samples - total_generated)
        y = y[:B].to(device)

        # Random noise (fixed seed for reproducibility per batch)
        if args.fix_noise:
            torch.manual_seed(args.seed + batch_idx)
        x0 = torch.randn(B, 3, 32, 32, device=device)

        # Generate trajectory
        if args.cfg_scale > 1.0:
            trajectory, timesteps = generate_trajectory_with_cfg(
                teacher, x0, y, args.num_steps, device, args.cfg_scale
            )
        else:
            trajectory, timesteps = generate_trajectory_ode(
                teacher, x0, y, args.num_steps, device
            )

        # Save batch
        save_data = {
            'trajectory': trajectory.cpu(),  # (B, T+1, C, H, W)
            'labels': y.cpu(),
            'timesteps': timesteps.cpu(),
            'x0': x0.cpu(),  # original noise
        }

        save_path = output_dir / f"batch_{batch_idx:05d}.pt"
        torch.save(save_data, save_path)

        # Visualize first batch
        if batch_idx == 0:
            # Save trajectory visualization for first sample
            traj_vis = trajectory[0]  # (T+1, C, H, W)
            traj_vis = (traj_vis + 1) / 2  # denormalize
            traj_vis = traj_vis.clamp(0, 1)
            save_image(traj_vis, output_dir / "trajectory_example.png", nrow=args.num_steps + 1)
            print(f"\nSaved trajectory visualization to {output_dir / 'trajectory_example.png'}")

        total_generated += B
        batch_idx += 1
        pbar.update(B)

    pbar.close()

    # Save metadata
    metadata = {
        'num_samples': total_generated,
        'num_steps': args.num_steps,
        'num_batches': batch_idx,
        'batch_size': args.batch_size,
        'cfg_scale': args.cfg_scale,
        'model': args.model,
        'teacher_ckpt': str(args.teacher_ckpt),
        'timesteps': timesteps.cpu().tolist(),
    }

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"  Total trajectories: {total_generated}")
    print(f"  Batches: {batch_idx}")
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-ckpt", type=str, required=True,
                        help="Path to teacher checkpoint")
    parser.add_argument("--output-dir", type=str, default="./trajectories",
                        help="Output directory for trajectories")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Path to CIFAR-10 data (for labels)")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"],
                        help="Teacher model type")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of trajectories to generate")
    parser.add_argument("--num-steps", type=int, default=12,
                        help="Number of ODE steps (should match student depth)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for generation")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fix-noise", action="store_true",
                        help="Use fixed noise per batch for reproducibility")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)

"""
Evaluation script for Unrolled SiT.

Computes:
- FID score against CIFAR-10 test set
- Sample generation
- Layer-wise analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
from torchvision import datasets, transforms
import os

from unrolled_sit import UnrolledSiT_S, UnrolledSiT_B
from trajectory_dataset import TeacherTrajectoryDataset

# FID computation
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not installed. Install with: pip install pytorch-fid")


@torch.no_grad()
def generate_samples(model, num_samples, batch_size, device, use_cfg=False, cfg_scale=1.0):
    """Generate samples from the model."""
    model.eval()
    samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch = min(batch_size, num_samples - i * batch_size)

        # Random noise
        z = torch.randn(current_batch, 3, 32, 32, device=device)

        # Random labels
        y = torch.randint(0, 10, (current_batch,), device=device)

        # Generate
        if use_cfg and cfg_scale > 1.0:
            out = model.forward_with_cfg(
                torch.cat([z, z], dim=0),
                torch.cat([y, torch.full_like(y, 10)], dim=0),  # 10 = null class
                cfg_scale
            )
            out = out[:current_batch]
        else:
            out = model(z, y)

        # Denormalize
        out = (out + 1) / 2
        out = out.clamp(0, 1)

        samples.append(out.cpu())

    return torch.cat(samples, dim=0)[:num_samples]


@torch.no_grad()
def generate_samples_per_class(model, num_per_class, device):
    """Generate equal samples per class."""
    model.eval()
    all_samples = []

    for class_idx in range(10):
        z = torch.randn(num_per_class, 3, 32, 32, device=device)
        y = torch.full((num_per_class,), class_idx, device=device)

        out = model(z, y)
        out = (out + 1) / 2
        out = out.clamp(0, 1)

        all_samples.append(out.cpu())

    return torch.cat(all_samples, dim=0)


def compute_fid(model, device, num_samples=10000, batch_size=256, real_stats_path=None):
    """
    Compute FID score.

    If real_stats_path is provided, uses precomputed statistics.
    Otherwise computes real statistics from CIFAR-10 test set.
    """
    if not FID_AVAILABLE:
        print("FID computation not available. Install pytorch-fid.")
        return None

    # Create temp directories
    fake_dir = Path("./temp_fid/fake")
    real_dir = Path("./temp_fid/real")
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # Generate fake samples
    print("Generating samples for FID...")
    samples = generate_samples(model, num_samples, batch_size, device)

    # Save fake samples
    print("Saving generated samples...")
    for i, img in enumerate(tqdm(samples)):
        save_image(img, fake_dir / f"{i:05d}.png")

    # Get real samples if needed
    if real_stats_path is None:
        print("Saving real samples...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        for i, (img, _) in enumerate(tqdm(cifar_test)):
            if i >= num_samples:
                break
            save_image(img, real_dir / f"{i:05d}.png")

        real_path = str(real_dir)
    else:
        real_path = real_stats_path

    # Compute FID
    print("Computing FID...")
    fid = fid_score.calculate_fid_given_paths(
        [real_path, str(fake_dir)],
        batch_size=50,
        device=device,
        dims=2048
    )

    # Cleanup
    import shutil
    shutil.rmtree("./temp_fid")

    return fid


@torch.no_grad()
def analyze_intermediates(model, dataset, device, num_samples=100):
    """
    Analyze intermediate representations.

    Computes:
    - MSE between predicted and target intermediates
    - Frequency content at each layer
    """
    model.eval()
    results = {
        'mse_per_layer': [],
        'low_freq_ratio': [],  # Ratio of low-frequency energy
    }

    num_layers = model.depth
    mse_accum = [0.0] * num_layers
    count = 0

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing"):
        sample = dataset[i]
        trajectory = sample['trajectory'].unsqueeze(0).to(device)
        label = torch.tensor([sample['label']], device=device)

        x_input = trajectory[:, 0]
        targets = trajectory[:, 1:]

        _, intermediates = model(x_input, label, return_intermediates=True)

        for j, pred in enumerate(intermediates):
            target = targets[:, j]  # (1, C, H, W)
            mse = F.mse_loss(pred, target).item()
            mse_accum[j] += mse

        count += 1

    results['mse_per_layer'] = [m / count for m in mse_accum]

    return results


@torch.no_grad()
def visualize_trajectory(model, device, output_path="trajectory_viz.png"):
    """Visualize the generation trajectory through intermediate outputs."""
    model.eval()

    # Generate one sample
    z = torch.randn(1, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (1,), device=device)

    _, intermediates = model(z, y, return_intermediates=True)

    # Add noise input at the beginning
    all_images = [((z[0] + 1) / 2).clamp(0, 1).cpu()]

    for img in intermediates:
        img = ((img[0] + 1) / 2).clamp(0, 1).cpu()
        all_images.append(img)

    # Stack and save
    all_images = torch.stack(all_images, dim=0)
    save_image(all_images, output_path, nrow=len(all_images))
    print(f"Saved trajectory visualization to {output_path}")

    return all_images


def load_model(checkpoint_path, model_type="small", device="cuda"):
    """Load model from checkpoint."""
    if model_type == "small":
        model = UnrolledSiT_S(num_classes=10)
    elif model_type == "base":
        model = UnrolledSiT_B(num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("Loaded EMA weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded weights directly")

    return model.to(device)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.model, device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    if args.generate:
        print(f"Generating {args.num_samples} samples...")
        samples = generate_samples(model, args.num_samples, args.batch_size, device)
        save_image(samples[:64], output_dir / "samples.png", nrow=8)
        print(f"Saved sample grid to {output_dir / 'samples.png'}")

        # Save all samples
        if args.save_all:
            samples_dir = output_dir / "all_samples"
            samples_dir.mkdir(exist_ok=True)
            for i, img in enumerate(samples):
                save_image(img, samples_dir / f"{i:05d}.png")
            print(f"Saved all samples to {samples_dir}")

    # Compute FID
    if args.fid:
        fid = compute_fid(model, device, args.num_samples, args.batch_size)
        if fid is not None:
            print(f"FID: {fid:.2f}")
            with open(output_dir / "fid.txt", "w") as f:
                f.write(f"FID: {fid:.2f}\n")

    # Visualize trajectory
    if args.trajectory:
        visualize_trajectory(model, device, output_dir / "trajectory.png")

    # Analyze intermediates (requires pre-generated trajectories)
    if args.analyze:
        if args.trajectory_dir and Path(args.trajectory_dir).exists():
            dataset = TeacherTrajectoryDataset(args.trajectory_dir)
            results = analyze_intermediates(model, dataset, device, num_samples=args.num_samples)
            print("\nMSE per layer:")
            for i, mse in enumerate(results['mse_per_layer']):
                print(f"  Layer {i+1}: {mse:.6f}")
        else:
            print("Skipping analysis: --trajectory-dir not provided or doesn't exist")

    # Per-class samples
    if args.per_class:
        print("Generating per-class samples...")
        samples = generate_samples_per_class(model, 10, device)
        save_image(samples, output_dir / "per_class_samples.png", nrow=10)
        print(f"Saved per-class samples to {output_dir / 'per_class_samples.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Unrolled SiT")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--trajectory-dir", type=str, default=None, help="Path to trajectories (for --analyze)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10000)

    # Evaluation modes
    parser.add_argument("--generate", action="store_true", help="Generate samples")
    parser.add_argument("--save-all", action="store_true", help="Save all generated samples")
    parser.add_argument("--fid", action="store_true", help="Compute FID score")
    parser.add_argument("--trajectory", action="store_true", help="Visualize generation trajectory")
    parser.add_argument("--analyze", action="store_true", help="Analyze intermediate representations")
    parser.add_argument("--per-class", action="store_true", help="Generate per-class samples")

    args = parser.parse_args()

    # Default to generate if no mode specified
    if not any([args.generate, args.fid, args.trajectory, args.analyze, args.per_class]):
        args.generate = True

    main(args)

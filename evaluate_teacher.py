"""
Evaluation script for teacher model (standard SiT with ODE sampling).
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
from torchvision import datasets, transforms

from train_teacher import SiT_S_CIFAR, SiT_B_CIFAR, sample_ode

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False


def load_model(checkpoint_path, model_type="small", device="cuda"):
    """Load teacher model from checkpoint."""
    if model_type == "small":
        model = SiT_S_CIFAR(num_classes=10)
    else:
        model = SiT_B_CIFAR(num_classes=10)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("Loaded EMA weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)

    return model.to(device)


@torch.no_grad()
def generate_samples(model, num_samples, batch_size, device, num_steps=12, cfg_scale=1.5):
    """Generate samples using ODE sampling."""
    model.eval()
    samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=f"Generating ({num_steps} steps)"):
        current_batch = min(batch_size, num_samples - i * batch_size)
        shape = (current_batch, 3, 32, 32)
        y = torch.randint(0, 10, (current_batch,), device=device)

        out = sample_ode(model, shape, y, device, num_steps=num_steps, cfg_scale=cfg_scale)
        out = (out + 1) / 2
        out = out.clamp(0, 1)
        samples.append(out.cpu())

    return torch.cat(samples, dim=0)[:num_samples]


@torch.no_grad()
def generate_per_class(model, num_per_class, device, num_steps=12, cfg_scale=1.5):
    """Generate samples per class."""
    model.eval()
    all_samples = []

    for class_idx in range(10):
        shape = (num_per_class, 3, 32, 32)
        y = torch.full((num_per_class,), class_idx, device=device)
        out = sample_ode(model, shape, y, device, num_steps=num_steps, cfg_scale=cfg_scale)
        out = (out + 1) / 2
        out = out.clamp(0, 1)
        all_samples.append(out.cpu())

    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def visualize_trajectory(model, device, num_steps=12, cfg_scale=1.5, output_path="trajectory.png"):
    """Visualize the ODE trajectory."""
    model.eval()

    z = torch.randn(1, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (1,), device=device)

    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
    trajectory = [((z[0] + 1) / 2).clamp(0, 1).cpu()]

    x = z.clone()
    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_batch = torch.full((1,), t, device=device)

        if cfg_scale > 1.0:
            v_cond = model(x, t_batch, y)
            v_uncond = model(x, t_batch, torch.full_like(y, 10))
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, y)

        x = x + v * dt
        trajectory.append(((x[0] + 1) / 2).clamp(0, 1).cpu())

    trajectory = torch.stack(trajectory, dim=0)
    save_image(trajectory, output_path, nrow=len(trajectory))
    print(f"Saved trajectory to {output_path}")


def compute_fid(model, device, num_samples, batch_size, num_steps, cfg_scale):
    """Compute FID score."""
    if not FID_AVAILABLE:
        print("FID not available. Install pytorch-fid.")
        return None

    fake_dir = Path("./temp_fid/fake")
    real_dir = Path("./temp_fid/real")
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    print("Generating samples for FID...")
    samples = generate_samples(model, num_samples, batch_size, device, num_steps, cfg_scale)

    print("Saving generated samples...")
    for i, img in enumerate(tqdm(samples)):
        save_image(img, fake_dir / f"{i:05d}.png")

    print("Saving real samples...")
    transform = transforms.ToTensor()
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    for i, (img, _) in enumerate(tqdm(cifar_test)):
        if i >= num_samples:
            break
        save_image(img, real_dir / f"{i:05d}.png")

    print("Computing FID...")
    fid = fid_score.calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=50,
        device=device,
        dims=2048
    )

    import shutil
    shutil.rmtree("./temp_fid")

    return fid


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model, device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate:
        print(f"Generating {args.num_samples} samples with {args.num_steps} steps...")
        samples = generate_samples(model, min(args.num_samples, 1000), args.batch_size,
                                   device, args.num_steps, args.cfg_scale)
        save_image(samples[:64], output_dir / "samples.png", nrow=8)
        print(f"Saved to {output_dir / 'samples.png'}")

    if args.fid:
        fid = compute_fid(model, device, args.num_samples, args.batch_size,
                         args.num_steps, args.cfg_scale)
        if fid is not None:
            print(f"FID ({args.num_steps} steps): {fid:.2f}")
            with open(output_dir / "fid.txt", "w") as f:
                f.write(f"FID ({args.num_steps} steps): {fid:.2f}\n")

    if args.trajectory:
        visualize_trajectory(model, device, args.num_steps, args.cfg_scale,
                           output_dir / "trajectory.png")

    if args.per_class:
        print("Generating per-class samples...")
        samples = generate_per_class(model, 10, device, args.num_steps, args.cfg_scale)
        save_image(samples, output_dir / "per_class_samples.png", nrow=10)
        print(f"Saved to {output_dir / 'per_class_samples.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--cfg-scale", type=float, default=1.5)

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--per-class", action="store_true")

    args = parser.parse_args()

    if not any([args.generate, args.fid, args.trajectory, args.per_class]):
        args.generate = True

    main(args)

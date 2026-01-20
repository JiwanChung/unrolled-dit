"""
Evaluation script for consistency distillation baseline.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.utils import save_image
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_teacher import SiT_S_CIFAR, SiT_B_CIFAR
from baselines.train_consistency import ConsistencyModel

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False


def load_model(checkpoint_path, model_type="small", device="cuda"):
    """Load consistency model from checkpoint."""
    if model_type == "small":
        base_model = SiT_S_CIFAR(num_classes=10)
    else:
        base_model = SiT_B_CIFAR(num_classes=10)

    model = ConsistencyModel(base_model)

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
def generate_samples(model, num_samples, batch_size, device):
    """Generate samples using single-step consistency model."""
    model.eval()
    samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating (1 step)"):
        current_batch = min(batch_size, num_samples - i * batch_size)

        z = torch.randn(current_batch, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (current_batch,), device=device)
        t = torch.zeros(current_batch, device=device)  # t=0 for one-step generation

        out = model(z, t, y)
        out = (out + 1) / 2
        out = out.clamp(0, 1)
        samples.append(out.cpu())

    return torch.cat(samples, dim=0)[:num_samples]


@torch.no_grad()
def generate_per_class(model, num_per_class, device):
    """Generate samples per class."""
    model.eval()
    all_samples = []

    for class_idx in range(10):
        z = torch.randn(num_per_class, 3, 32, 32, device=device)
        y = torch.full((num_per_class,), class_idx, device=device)
        t = torch.zeros(num_per_class, device=device)

        out = model(z, t, y)
        out = (out + 1) / 2
        out = out.clamp(0, 1)
        all_samples.append(out.cpu())

    return torch.cat(all_samples, dim=0)


def compute_fid(model, device, num_samples, batch_size):
    """Compute FID score."""
    if not FID_AVAILABLE:
        print("FID not available. Install pytorch-fid.")
        return None

    fake_dir = Path("./temp_fid/fake")
    real_dir = Path("./temp_fid/real")
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    print("Generating samples for FID...")
    samples = generate_samples(model, num_samples, batch_size, device)

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
        print(f"Generating samples (1-step consistency)...")
        samples = generate_samples(model, min(args.num_samples, 1000), args.batch_size, device)
        save_image(samples[:64], output_dir / "samples.png", nrow=8)
        print(f"Saved to {output_dir / 'samples.png'}")

    if args.fid:
        fid = compute_fid(model, device, args.num_samples, args.batch_size)
        if fid is not None:
            print(f"FID (1 step): {fid:.2f}")
            with open(output_dir / "fid.txt", "w") as f:
                f.write(f"FID (1 step): {fid:.2f}\n")

    if args.per_class:
        print("Generating per-class samples...")
        samples = generate_per_class(model, 10, device)
        save_image(samples, output_dir / "per_class_samples.png", nrow=10)
        print(f"Saved to {output_dir / 'per_class_samples.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10000)

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--per-class", action="store_true")

    args = parser.parse_args()

    if not any([args.generate, args.fid, args.per_class]):
        args.generate = True

    main(args)

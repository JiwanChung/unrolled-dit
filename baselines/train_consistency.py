"""
Baseline 4: Consistency Distillation

Train the model to be self-consistent: predictions from different points
along the trajectory should map to the same final output.

Key idea: f(x_t, t) and f(x_{t'}, t') should produce the same x_1
when both x_t and x_{t'} are on the same trajectory.

Reference: Song et al., "Consistency Models"

Usage:
    python baselines/train_consistency.py \
        --teacher-ckpt ./results/teacher/teacher_best.pt \
        --log-dir ./results/baseline_consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
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


class ConsistencyModel(nn.Module):
    """
    Wrapper that turns a velocity model into a consistency model.

    Given x_t at time t, predicts the final x_1 directly.
    Uses skip connection: f(x_t, t) = c_skip(t) * x_t + c_out(t) * F(x_t, t)
    """
    def __init__(self, velocity_model, sigma_data=0.5):
        super().__init__()
        self.velocity_model = velocity_model
        self.sigma_data = sigma_data

    def get_scalings(self, t):
        """Get skip and output scalings based on time."""
        # For flow matching: x_t = t * x_1 + (1-t) * x_0
        # At t=1, we want f(x_1, 1) = x_1 (identity)
        # At t=0, we want f(x_0, 0) to predict x_1

        # Simple linear scaling
        c_skip = t  # At t=1, c_skip=1 (identity); at t=0, c_skip=0
        c_out = 1 - t  # At t=1, c_out=0; at t=0, c_out=1

        return c_skip, c_out

    def forward(self, x_t, t, y):
        """
        Predict x_1 from x_t at time t.

        Args:
            x_t: noisy input at time t
            t: timestep (scalar or batch)
            y: class labels

        Returns:
            x_1_pred: predicted clean output
        """
        B = x_t.shape[0]
        if t.dim() == 0:
            t = t.expand(B)

        c_skip, c_out = self.get_scalings(t)
        c_skip = c_skip.view(B, 1, 1, 1)
        c_out = c_out.view(B, 1, 1, 1)

        # Get velocity prediction and integrate to x_1
        # For flow matching: x_1 = x_t + (1-t) * v
        v = self.velocity_model(x_t, t, y)
        t_expand = t.view(B, 1, 1, 1)
        x_1_from_v = x_t + (1 - t_expand) * v

        # Apply skip connection
        x_1_pred = c_skip * x_t + c_out * x_1_from_v

        return x_1_pred


@torch.no_grad()
def get_trajectory_point(teacher, x_0, y, t_target, device, num_steps=100):
    """
    Get x_t at time t_target by running teacher ODE from x_0.
    """
    B = x_0.shape[0]
    x = x_0.clone()

    # Run ODE from 0 to t_target
    timesteps = torch.linspace(0, t_target, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_batch = torch.full((B,), t.item(), device=device)
        v = teacher(x, t_batch, y)
        x = x + v * dt

    return x


@torch.no_grad()
def get_teacher_endpoint(teacher, x_t, t_start, y, device, num_steps=50):
    """
    Run teacher ODE from x_t at t_start to get x_1.
    """
    B = x_t.shape[0]
    x = x_t.clone()

    timesteps = torch.linspace(t_start, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_batch = torch.full((B,), t.item(), device=device)
        v = teacher(x, t_batch, y)
        x = x + v * dt

    return x


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_flush(f"Device: {device}")

    # Load teacher
    print_flush(f"Loading teacher from {args.teacher_ckpt}")
    if args.model == "small":
        teacher = SiT_S_CIFAR(num_classes=10)
        student_base = SiT_S_CIFAR(num_classes=10)
    else:
        teacher = SiT_B_CIFAR(num_classes=10)
        student_base = SiT_B_CIFAR(num_classes=10)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict']:
        teacher.load_state_dict(ckpt['ema_state_dict'])
        student_base.load_state_dict(ckpt['ema_state_dict'])
        print_flush("  Loaded EMA weights")
    else:
        teacher.load_state_dict(ckpt['model_state_dict'])
        student_base.load_state_dict(ckpt['model_state_dict'])

    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Create consistency model (student)
    student = ConsistencyModel(student_base).to(device)

    # EMA of student
    ema_student = deepcopy(student)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    if args.model == "small":
        num_params = sum(p.numel() for p in student.parameters())
        print_flush(f"  Parameters: {num_params:,}")

    # Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print_flush("=" * 60)
    print_flush("Baseline: Consistency Distillation")
    print_flush("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for x_real, y in pbar:
            x_real = x_real.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = x_real.shape[0]

            # Sample random noise
            x_0 = torch.randn_like(x_real)

            # Sample two different timesteps t and t' where t' > t
            # We want consistency: f(x_t, t) ≈ f(x_{t'}, t')
            t = torch.rand(B, device=device) * 0.8  # t in [0, 0.8]
            t_prime = t + torch.rand(B, device=device) * (1 - t) * 0.5  # t' > t

            # Get x_t and x_{t'} from same trajectory using teacher
            with torch.no_grad():
                # Run teacher ODE to get points on trajectory
                x_t = get_trajectory_point(teacher, x_0, y, t.mean().item(), device, num_steps=20)
                x_t_prime = get_trajectory_point(teacher, x_0, y, t_prime.mean().item(), device, num_steps=20)

                # Get teacher's prediction of x_1 from x_t (target)
                x_1_target = get_teacher_endpoint(teacher, x_t, t.mean().item(), y, device, num_steps=20)

            # Student predictions
            x_1_pred_t = student(x_t, t, y)

            # EMA student prediction at t' (for consistency)
            with torch.no_grad():
                x_1_pred_t_prime = ema_student(x_t_prime, t_prime, y)

            # Consistency loss: student at t should match EMA student at t'
            consistency_loss = F.mse_loss(x_1_pred_t, x_1_pred_t_prime.detach())

            # Distillation loss: match teacher's endpoint
            distill_loss = F.mse_loss(x_1_pred_t, x_1_target)

            # Combined loss
            loss = args.consistency_weight * consistency_loss + args.distill_weight * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA
            update_ema(ema_student, student, args.ema_decay)

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cons': f'{consistency_loss.item():.4f}',
                'dist': f'{distill_loss.item():.4f}'
            })

        avg_loss = epoch_loss / num_batches
        print_flush(f"Epoch {epoch}: loss = {avg_loss:.6f}")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        ckpt = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'ema_state_dict': ema_student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }
        safe_save(ckpt, log_dir / "checkpoint_latest.pt")

        if is_best:
            safe_save(ckpt, log_dir / "checkpoint_best.pt")
            print_flush(f"  New best! Loss: {avg_loss:.6f}")

        # Generate samples
        if epoch % args.sample_every == 0:
            generate_samples(ema_student, device, log_dir, epoch)

    print_flush(f"\nTraining complete! Best loss: {best_loss:.6f}")


@torch.no_grad()
def generate_samples(model, device, log_dir, epoch, num_samples=64):
    """Generate samples using consistency model (single step from noise)."""
    model.eval()

    # Start from pure noise (t=0)
    z = torch.randn(num_samples, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (num_samples,), device=device)
    t = torch.zeros(num_samples, device=device)  # t=0

    # Single forward pass to predict x_1
    samples = model(z, t, y)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    save_image(samples, Path(log_dir) / f"samples_epoch{epoch}.png", nrow=8)
    print_flush(f"  Saved samples (1-step)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="/scratch2/jiwan_chung/layerdistill/results/baseline_consistency")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)

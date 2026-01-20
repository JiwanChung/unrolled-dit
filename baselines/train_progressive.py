"""
Baseline 3: Progressive Distillation

Iteratively halve the number of sampling steps:
- Stage 1: 12 steps → 6 steps
- Stage 2: 6 steps → 3 steps
- Stage 3: 3 steps → 2 steps
- Stage 4: 2 steps → 1 step

At each stage, train a student to match teacher's 2-step output in 1 step.

Reference: Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models"

Usage:
    python baselines/train_progressive.py \
        --teacher-ckpt ./results/teacher/teacher_best.pt \
        --log-dir ./results/baseline_progressive
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


@torch.no_grad()
def teacher_two_steps(model, x, t_start, dt, y, device):
    """
    Run teacher for 2 steps: t_start → t_start+dt → t_start+2*dt
    Returns the final x after 2 steps.
    """
    B = x.shape[0]

    # Step 1
    t1 = torch.full((B,), t_start, device=device)
    v1 = model(x, t1, y)
    x_mid = x + v1 * dt

    # Step 2
    t2 = torch.full((B,), t_start + dt, device=device)
    v2 = model(x_mid, t2, y)
    x_final = x_mid + v2 * dt

    return x_final


def student_one_step(model, x, t_start, dt_double, y, device):
    """
    Student does 1 step covering same distance as teacher's 2 steps.
    t_start → t_start + 2*dt (same as teacher's 2 steps)
    """
    B = x.shape[0]
    t = torch.full((B,), t_start, device=device)
    v = model(x, t, y)
    x_final = x + v * dt_double
    return x_final


@torch.no_grad()
def sample_ode(model, shape, y, device, num_steps=50, cfg_scale=1.0):
    """Sample from model using Euler ODE."""
    B = shape[0]
    x = torch.randn(shape, device=device)
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_batch = torch.full((B,), t, device=device)

        if cfg_scale > 1.0:
            v_cond = model(x, t_batch, y)
            v_uncond = model(x, t_batch, torch.full_like(y, 10))
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_batch, y)

        x = x + v * dt

    return x


def train_one_stage(teacher, student, train_loader, num_steps_teacher,
                    device, args, stage_dir, stage_num):
    """
    Train student to match teacher's 2-step output in 1 step.

    Teacher uses num_steps_teacher steps total.
    Student will use num_steps_teacher // 2 steps.
    """
    print_flush(f"\n{'='*60}")
    print_flush(f"Stage {stage_num}: {num_steps_teacher} steps → {num_steps_teacher // 2} steps")
    print_flush(f"{'='*60}")

    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA
    ema_student = deepcopy(student)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    # Timesteps for this stage
    # Teacher uses num_steps_teacher steps from t=0 to t=1
    # dt for teacher = 1 / num_steps_teacher
    dt_teacher = 1.0 / num_steps_teacher
    dt_student = 2 * dt_teacher  # student covers 2x distance per step

    num_student_steps = num_steps_teacher // 2

    best_loss = float('inf')

    for epoch in range(1, args.epochs_per_stage + 1):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Stage {stage_num} Epoch {epoch}")

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = x.shape[0]

            # Sample random starting point along trajectory
            # Student step i covers teacher steps 2i and 2i+1
            step_idx = np.random.randint(0, num_student_steps)
            t_start = step_idx * dt_student

            # Get starting x by running from noise
            x_t = torch.randn_like(x)
            if step_idx > 0:
                # Run teacher to get x at t_start
                timesteps = torch.linspace(0, t_start, step_idx * 2 + 1, device=device)
                for i in range(step_idx * 2):
                    t = timesteps[i]
                    dt = timesteps[i+1] - timesteps[i]
                    t_batch = torch.full((B,), t, device=device)
                    with torch.no_grad():
                        v = teacher(x_t, t_batch, y)
                    x_t = x_t + v * dt

            # Teacher: 2 steps from t_start
            with torch.no_grad():
                x_teacher = teacher_two_steps(teacher, x_t, t_start, dt_teacher, y, device)

            # Student: 1 step covering same distance
            x_student = student_one_step(student, x_t, t_start, dt_student, y, device)

            # Loss
            loss = F.mse_loss(x_student, x_teacher)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(ema_student, student, args.ema_decay)

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        print_flush(f"Stage {stage_num} Epoch {epoch}: loss = {avg_loss:.6f}")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Save checkpoints
        ckpt = {
            'epoch': epoch,
            'student_state_dict': student.state_dict(),
            'ema_state_dict': ema_student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'num_steps': num_student_steps,
        }
        safe_save(ckpt, stage_dir / "checkpoint_latest.pt")

        if is_best:
            safe_save(ckpt, stage_dir / "checkpoint_best.pt")
            print_flush(f"  New best! Loss: {avg_loss:.6f}")

        # Generate samples
        if epoch % args.sample_every == 0:
            ema_student.eval()
            samples = sample_ode(ema_student, (64, 3, 32, 32),
                               torch.randint(0, 10, (64,), device=device),
                               device, num_steps=num_student_steps, cfg_scale=args.cfg_scale)
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            save_image(samples, stage_dir / f"samples_epoch{epoch}.png", nrow=8)
            print_flush(f"  Saved samples ({num_student_steps} steps)")

    # Return EMA student as new teacher for next stage
    return ema_student


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_flush(f"Device: {device}")

    # Load initial teacher
    print_flush(f"Loading teacher from {args.teacher_ckpt}")
    if args.model == "small":
        teacher = SiT_S_CIFAR(num_classes=10)
    else:
        teacher = SiT_B_CIFAR(num_classes=10)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict']:
        teacher.load_state_dict(ckpt['ema_state_dict'])
        print_flush("  Loaded EMA weights")
    else:
        teacher.load_state_dict(ckpt['model_state_dict'])

    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Progressive distillation stages
    # 12 → 6 → 3 → 2 → 1 (or stop earlier)
    stages = []
    n = args.initial_steps
    while n > 1:
        stages.append(n)
        n = n // 2
    stages.append(1)  # final 1-step

    print_flush(f"\nProgressive distillation stages: {stages}")
    print_flush(f"  {' → '.join(str(s) for s in stages)} steps")

    current_teacher = teacher

    for stage_idx, num_steps in enumerate(stages[:-1]):
        # Create fresh student (same architecture)
        if args.model == "small":
            student = SiT_S_CIFAR(num_classes=10)
        else:
            student = SiT_B_CIFAR(num_classes=10)

        # Initialize student from current teacher
        student.load_state_dict(current_teacher.state_dict())
        student = student.to(device)

        # Train this stage
        stage_dir = log_dir / f"stage{stage_idx+1}_{num_steps}to{num_steps//2}"
        new_teacher = train_one_stage(
            current_teacher, student, train_loader, num_steps,
            device, args, stage_dir, stage_idx + 1
        )

        # New teacher for next stage
        current_teacher = new_teacher
        for p in current_teacher.parameters():
            p.requires_grad = False

    # Save final 1-step model
    final_ckpt = {
        'model_state_dict': current_teacher.state_dict(),
        'num_steps': 1,
    }
    safe_save(final_ckpt, log_dir / "final_1step.pt")

    print_flush(f"\n{'='*60}")
    print_flush("Progressive distillation complete!")
    print_flush(f"Final 1-step model: {log_dir / 'final_1step.pt'}")
    print_flush(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="/scratch2/jiwan_chung/layerdistill/results/baseline_progressive")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"])
    parser.add_argument("--initial-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs-per-stage", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)

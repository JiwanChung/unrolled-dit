"""
Training script for Unrolled SiT on CIFAR-10.

Usage:
    python train_unrolled.py --data-path ./data --batch-size 128 --epochs 100

    # Multi-GPU:
    torchrun --nproc_per_node=4 train_unrolled.py --data-path ./data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import os
from copy import deepcopy
from collections import OrderedDict

from unrolled_sit import UnrolledSiT_S, UnrolledSiT_B
from trajectory_dataset import (
    TrajectoryDataset,
    get_cifar10_loaders,
    compute_uniform_weights,
    compute_snr_weights,
)

# Optional: for FID computation
try:
    from pytorch_fid import fid_score
    from torchvision.utils import save_image
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. FID evaluation disabled.")


def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/train.log")
        ]
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class UnrolledTrainer:
    """Trainer for Unrolled SiT with layer-wise supervision."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_weights: torch.Tensor,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        device: str = "cuda",
        log_dir: str = "./results",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights for each layer
        self.loss_weights = loss_weights.to(device)
        self.num_steps = len(loss_weights)

        # EMA model
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = deepcopy(model).to(device)
            self.ema_model.eval()
            self.ema_decay = ema_decay

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Logger
        self.logger = setup_logging(log_dir)
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Loss weights: {loss_weights.tolist()}")

        # Metrics tracking
        self.global_step = 0
        self.best_loss = float('inf')

    def compute_loss(self, batch):
        """
        Compute layer-wise loss.

        Args:
            batch: dict with 'trajectory' (B, T+1, C, H, W) and 'label' (B,)

        Returns:
            total_loss: scalar
            layer_losses: list of per-layer losses
        """
        trajectory = batch['trajectory'].to(self.device)
        labels = batch['label'].to(self.device)

        B, T_plus_1, C, H, W = trajectory.shape

        # Input is pure noise (t=0)
        x_input = trajectory[:, 0]

        # Targets are intermediate and final states
        targets = trajectory[:, 1:]  # (B, T, C, H, W)

        # Forward pass with intermediates
        final_out, intermediates = self.model(x_input, labels, return_intermediates=True)

        # Compute layer-wise loss
        layer_losses = []
        total_loss = 0

        for i, (pred, target) in enumerate(zip(intermediates, targets)):
            layer_loss = F.mse_loss(pred, target)
            layer_losses.append(layer_loss.item())
            total_loss = total_loss + self.loss_weights[i] * layer_loss

        return total_loss, layer_losses

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        layer_losses_sum = [0] * self.num_steps
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()

            loss, layer_losses = self.compute_loss(batch)

            loss.backward()
            self.optimizer.step()

            # Update EMA
            if self.use_ema:
                update_ema(self.ema_model, self.model, self.ema_decay)

            # Track metrics
            total_loss += loss.item()
            for i, ll in enumerate(layer_losses):
                layer_losses_sum[i] += ll
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        avg_layer_losses = [ll / num_batches for ll in layer_losses_sum]

        return avg_loss, avg_layer_losses

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set."""
        model = self.ema_model if self.use_ema else self.model
        model.eval()

        total_loss = 0
        layer_losses_sum = [0] * self.num_steps
        num_batches = 0

        for batch in self.test_loader:
            trajectory = batch['trajectory'].to(self.device)
            labels = batch['label'].to(self.device)

            x_input = trajectory[:, 0]
            targets = trajectory[:, 1:]

            final_out, intermediates = model(x_input, labels, return_intermediates=True)

            for i, (pred, target) in enumerate(zip(intermediates, targets)):
                layer_loss = F.mse_loss(pred, target)
                layer_losses_sum[i] += layer_loss.item()
                total_loss += self.loss_weights[i].item() * layer_loss.item()

            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_layer_losses = [ll / num_batches for ll in layer_losses_sum]

        return avg_loss, avg_layer_losses

    @torch.no_grad()
    def generate_samples(self, num_samples=64, use_ema=True):
        """Generate samples from random noise."""
        model = self.ema_model if (use_ema and self.use_ema) else self.model
        model.eval()

        # Random noise
        z = torch.randn(num_samples, 3, 32, 32, device=self.device)

        # Random labels
        y = torch.randint(0, 10, (num_samples,), device=self.device)

        # Generate
        samples = model(z, y)

        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)

        return samples

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step,
        }
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        path = self.log_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.log_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with loss {loss:.4f}")

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        return checkpoint.get('epoch', 0)

    def train(self, num_epochs, save_every=10, sample_every=10):
        """Full training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_layer_losses = self.train_epoch(epoch)

            # Evaluate
            eval_loss, eval_layer_losses = self.evaluate()

            # Log
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}"
            )
            self.logger.info(f"  Train layer losses: {[f'{l:.4f}' for l in train_layer_losses]}")
            self.logger.info(f"  Eval layer losses: {[f'{l:.4f}' for l in eval_layer_losses]}")

            # Save checkpoint
            is_best = eval_loss < self.best_loss
            if is_best:
                self.best_loss = eval_loss

            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, eval_loss, is_best)

            # Generate samples
            if epoch % sample_every == 0:
                samples = self.generate_samples(num_samples=64)
                sample_path = self.log_dir / f"samples_epoch{epoch}.png"
                save_image(samples, sample_path, nrow=8)
                self.logger.info(f"Saved samples to {sample_path}")


def main(args):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    if args.model == "small":
        model = UnrolledSiT_S(num_classes=10)
    elif args.model == "base":
        model = UnrolledSiT_B(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        root=args.data_path,
        batch_size=args.batch_size,
        num_steps=model.depth,
        num_workers=args.num_workers,
        fix_noise=args.fix_noise,
    )

    # Loss weights
    if args.loss_weighting == "uniform":
        loss_weights = compute_uniform_weights(model.depth)
    elif args.loss_weighting == "snr":
        loss_weights = compute_snr_weights(model.depth, gamma=args.snr_gamma)
    else:
        raise ValueError(f"Unknown loss weighting: {args.loss_weighting}")

    print(f"Loss weighting: {args.loss_weighting}")
    print(f"Loss weights: {loss_weights.tolist()}")

    # Trainer
    trainer = UnrolledTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_weights=loss_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        log_dir=args.log_dir,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )

    # Resume from checkpoint
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unrolled SiT on CIFAR-10")

    # Data
    parser.add_argument("--data-path", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--fix-noise", action="store_true", help="Use fixed noise per sample")

    # Model
    parser.add_argument("--model", type=str, default="small", choices=["small", "base"],
                        help="Model size")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay")

    # Loss weighting
    parser.add_argument("--loss-weighting", type=str, default="uniform",
                        choices=["uniform", "snr"], help="Loss weighting scheme")
    parser.add_argument("--snr-gamma", type=float, default=5.0, help="SNR gamma for min-SNR weighting")

    # Logging
    parser.add_argument("--log-dir", type=str, default="./results/unrolled_sit",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--sample-every", type=int, default=10, help="Generate samples every N epochs")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    main(args)

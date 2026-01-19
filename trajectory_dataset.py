"""
Trajectory Dataset for Unrolled SiT training.

Loads pre-generated teacher trajectories from disk.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json


class TeacherTrajectoryDataset(Dataset):
    """
    Dataset that loads pre-generated teacher trajectories.

    Each trajectory file contains:
        - trajectory: (B, num_steps+1, C, H, W) - from noise to generated image
        - labels: (B,) - class labels
        - timesteps: (num_steps+1,) - timestep for each point
        - x0: (B, C, H, W) - original noise

    Returns per sample:
        - trajectory: (num_steps+1, C, H, W)
        - label: int
    """

    def __init__(self, trajectory_dir: str, limit: int = None):
        """
        Args:
            trajectory_dir: Directory containing trajectory batch files
            limit: Optional limit on number of samples to use
        """
        self.trajectory_dir = Path(trajectory_dir)

        # Load metadata
        metadata_path = self.trajectory_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            self.num_steps = self.metadata['num_steps']
            print(f"Loaded metadata: {self.metadata['num_samples']} samples, {self.num_steps} steps")
        else:
            self.metadata = None
            self.num_steps = None

        # Find all trajectory files
        self.files = sorted(self.trajectory_dir.glob("batch_*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No trajectory files found in {trajectory_dir}")

        # Build index: map global idx to (file_idx, local_idx)
        self.index = []
        for file_idx, file_path in enumerate(self.files):
            # Load just to get batch size
            data = torch.load(file_path, map_location='cpu')
            batch_size = data['trajectory'].shape[0]

            if self.num_steps is None:
                self.num_steps = data['trajectory'].shape[1] - 1

            for local_idx in range(batch_size):
                self.index.append((file_idx, local_idx))

                if limit and len(self.index) >= limit:
                    break

            if limit and len(self.index) >= limit:
                break

        print(f"TeacherTrajectoryDataset: {len(self.index)} samples from {len(self.files)} files")

        # Cache for loaded batches
        self._cache = {}
        self._cache_size = 10  # Keep last N batches in memory

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, local_idx = self.index[idx]

        # Load batch (with caching)
        if file_idx not in self._cache:
            # Evict old cache entries if needed
            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]

            self._cache[file_idx] = torch.load(self.files[file_idx], map_location='cpu')

        data = self._cache[file_idx]

        return {
            'trajectory': data['trajectory'][local_idx],  # (T+1, C, H, W)
            'label': data['labels'][local_idx].item() if isinstance(data['labels'][local_idx], torch.Tensor) else data['labels'][local_idx],
        }


class PrecomputedTrajectoryDataset(Dataset):
    """
    Dataset for loading pre-computed trajectories from disk.
    Useful when using a trained teacher model to generate non-linear trajectories.
    """

    def __init__(self, trajectory_dir: str, num_steps: int = 12):
        """
        Args:
            trajectory_dir: Directory containing trajectory files
            num_steps: Expected number of steps (for validation)
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.num_steps = num_steps

        # Find all trajectory files
        self.files = sorted(self.trajectory_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No trajectory files found in {trajectory_dir}")

        print(f"Found {len(self.files)} trajectory files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data


def compute_loss_weights_from_curvature(dataloader, num_samples=1000):
    """
    Compute loss weights based on trajectory curvature.

    For perfectly straight (rectified) flows, curvature is zero everywhere,
    so this will return uniform weights. But for real data, there may be
    slight variations we can exploit.

    Args:
        dataloader: DataLoader yielding trajectory batches
        num_samples: Number of samples to analyze

    Returns:
        weights: (num_steps,) tensor of normalized loss weights
    """
    curvatures = None
    count = 0

    for batch in dataloader:
        traj = batch['trajectory']  # (B, T+1, C, H, W)
        B, T_plus_1, C, H, W = traj.shape
        T = T_plus_1 - 1

        if curvatures is None:
            curvatures = torch.zeros(T)

        # Compute velocity at each step
        velocities = traj[:, 1:] - traj[:, :-1]  # (B, T, C, H, W)

        # Compute curvature as change in velocity
        for i in range(T - 1):
            v_before = velocities[:, i]
            v_after = velocities[:, i + 1]
            curvature = (v_after - v_before).pow(2).mean()
            curvatures[i] += curvature.item()

        count += B
        if count >= num_samples:
            break

    # Normalize
    curvatures = curvatures / count

    # Add small epsilon and normalize to sum to 1
    weights = curvatures + 1e-6
    weights = weights / weights.sum()

    return weights


def compute_snr_weights(num_steps, gamma=5.0):
    """
    Compute loss weights based on signal-to-noise ratio.
    Uses Min-SNR-gamma weighting from SD training.

    For linear interpolation x_t = t * x + (1-t) * noise:
        SNR(t) = t^2 / (1-t)^2

    Args:
        num_steps: Number of trajectory steps
        gamma: SNR clipping value (default 5.0)

    Returns:
        weights: (num_steps,) tensor of normalized loss weights
    """
    timesteps = torch.linspace(0, 1, num_steps + 1)

    weights = []
    for i in range(num_steps):
        t = timesteps[i + 1]  # Target timestep for this step

        # SNR = t^2 / (1-t)^2, but avoid division by zero
        if t >= 0.999:
            snr = 1000.0  # Large value for t close to 1
        elif t <= 0.001:
            snr = 0.001  # Small value for t close to 0
        else:
            snr = (t ** 2) / ((1 - t) ** 2)

        # Min-SNR-gamma weighting
        weight = min(snr, gamma) / snr
        weights.append(weight)

    weights = torch.tensor(weights)
    weights = weights / weights.sum()  # Normalize

    return weights


def compute_uniform_weights(num_steps):
    """Simple uniform weighting."""
    return torch.ones(num_steps) / num_steps


def get_trajectory_loaders(
    trajectory_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    train_limit: int = None,
    val_split: float = 0.1,
):
    """
    Get train and validation dataloaders for teacher trajectories.

    Args:
        trajectory_dir: Path to pre-generated trajectories
        batch_size: Batch size
        num_workers: DataLoader workers
        train_limit: Optional limit on training samples
        val_split: Fraction of data for validation

    Returns:
        train_loader, val_loader, num_steps
    """
    full_dataset = TeacherTrajectoryDataset(trajectory_dir, limit=train_limit)

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.num_steps


if __name__ == "__main__":
    import sys

    print("Testing trajectory datasets...\n")

    # Test loss weights
    print("Loss weights:")
    print(f"  Uniform (12 steps): {compute_uniform_weights(12).tolist()[:4]}...")
    print(f"  SNR gamma=5 (12 steps): {[f'{w:.4f}' for w in compute_snr_weights(12, gamma=5.0).tolist()[:4]]}...")

    # Test teacher trajectory dataset if available
    if len(sys.argv) > 1:
        trajectory_dir = sys.argv[1]
        print(f"\nTesting TeacherTrajectoryDataset from {trajectory_dir}")

        dataset = TeacherTrajectoryDataset(trajectory_dir)
        print(f"Dataset size: {len(dataset)}")
        print(f"Num steps: {dataset.num_steps}")

        sample = dataset[0]
        print(f"Trajectory shape: {sample['trajectory'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Trajectory range: [{sample['trajectory'].min():.2f}, {sample['trajectory'].max():.2f}]")

        # Test dataloader
        train_loader, val_loader, num_steps = get_trajectory_loaders(
            trajectory_dir, batch_size=4
        )
        batch = next(iter(train_loader))
        print(f"\nBatch trajectory shape: {batch['trajectory'].shape}")
        print(f"Batch labels: {batch['label']}")
    else:
        print("\nTo test TeacherTrajectoryDataset, run:")
        print("  python trajectory_dataset.py /path/to/trajectories")

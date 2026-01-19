#!/bin/bash
# Quick start script for Unrolled SiT training on CIFAR-10

set -e

# Configuration
DATA_PATH="./data"
LOG_DIR="./results/unrolled_sit_run1"
MODEL="small"  # or "base"
BATCH_SIZE=128
EPOCHS=100
LR=1e-4
LOSS_WEIGHTING="uniform"  # or "snr"

# Create directories
mkdir -p $DATA_PATH
mkdir -p $LOG_DIR

echo "============================================"
echo "Unrolled SiT Training on CIFAR-10"
echo "============================================"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Loss weighting: $LOSS_WEIGHTING"
echo "Log directory: $LOG_DIR"
echo "============================================"

# Install dependencies if needed
pip install torch torchvision timm tqdm --quiet

# Optional: install FID computation
pip install pytorch-fid --quiet 2>/dev/null || echo "Note: pytorch-fid not installed, FID evaluation disabled"

# Run training
python train_unrolled.py \
    --data-path $DATA_PATH \
    --log-dir $LOG_DIR \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --loss-weighting $LOSS_WEIGHTING \
    --use-ema \
    --save-every 10 \
    --sample-every 10

echo "Training complete! Results saved to $LOG_DIR"

#!/bin/bash
#
# Full distillation pipeline for Unrolled SiT
#
# Steps:
#   1. Train teacher model on CIFAR-10
#   2. Generate trajectories from teacher
#   3. Train unrolled student on trajectories
#
# Usage:
#   ./train.sh                     # Run full pipeline
#   ./train.sh --step teacher      # Only train teacher
#   ./train.sh --step trajectories # Only generate trajectories
#   ./train.sh --step student      # Only train student
#

set -e

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH="./data"
RESULTS_DIR="/scratch2/jiwan_chung/layerdistill/results"
MODEL="small"
BATCH_SIZE=128
NUM_WORKERS=4

# Teacher config
TEACHER_EPOCHS=500
TEACHER_DIR="${RESULTS_DIR}/teacher"

# Trajectory config
TRAJECTORY_DIR="${RESULTS_DIR}/trajectories"
NUM_TRAJECTORIES=50000
NUM_STEPS=12
CFG_SCALE=1.5

# Student config
STUDENT_EPOCHS=100
STUDENT_DIR="${RESULTS_DIR}/student"
LOSS_WEIGHTING="uniform"

# =============================================================================
# Parse arguments
# =============================================================================
STEP="all"
GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --teacher-epochs)
            TEACHER_EPOCHS="$2"
            shift 2
            ;;
        --student-epochs)
            STUDENT_EPOCHS="$2"
            shift 2
            ;;
        --num-trajectories)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --loss-weighting)
            LOSS_WEIGHTING="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --step STEP          Run specific step: teacher, trajectories, student, or all (default)"
            echo "  --gpus GPUS          Comma-separated GPU IDs (default: auto-detect)"
            echo "  --model MODEL        Model size: small or base (default: small)"
            echo "  --teacher-epochs N   Teacher training epochs (default: 500)"
            echo "  --student-epochs N   Student training epochs (default: 100)"
            echo "  --num-trajectories N Number of trajectories to generate (default: 50000)"
            echo "  --loss-weighting W   Student loss weighting: uniform or snr (default: uniform)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Auto-resume is enabled by default. Training will resume from the latest"
            echo "checkpoint if one exists in the output directory."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Auto-detect GPUs
# =============================================================================
if [ -z "$GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ "$NUM_GPUS" -gt 0 ]; then
            GPUS=$(seq -s, 0 $((NUM_GPUS - 1)))
        else
            echo "Error: No GPUs detected"
            exit 1
        fi
    else
        echo "Error: nvidia-smi not found"
        exit 1
    fi
fi

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================================"
echo "  Unrolled SiT - Distillation Pipeline"
echo "============================================================"
echo ""
echo "  Step:              $STEP"
echo "  GPUs:              $GPUS ($NUM_GPUS total)"
echo "  Model:             $MODEL"
echo ""
echo "  Teacher epochs:    $TEACHER_EPOCHS"
echo "  Trajectories:      $NUM_TRAJECTORIES"
echo "  Student epochs:    $STUDENT_EPOCHS"
echo "  Loss weighting:    $LOSS_WEIGHTING"
echo ""
echo "============================================================"
echo ""

# Create directories
mkdir -p "$DATA_PATH"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 1: Train Teacher
# =============================================================================
train_teacher() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== STEP 1: Train Teacher =========="
    echo ""

    mkdir -p "$TEACHER_DIR"

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" \
            train_teacher.py \
            --data-path "$DATA_PATH" \
            --log-dir "$TEACHER_DIR" \
            --model "$MODEL" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$TEACHER_EPOCHS" \
            --num-workers "$NUM_WORKERS" \
            --use-ema \
            --save-every 50 \
            --sample-every 50 \
            2>&1 | tee -a "$TEACHER_DIR/train.log"
    else
        python train_teacher.py \
            --data-path "$DATA_PATH" \
            --log-dir "$TEACHER_DIR" \
            --model "$MODEL" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$TEACHER_EPOCHS" \
            --num-workers "$NUM_WORKERS" \
            --use-ema \
            --save-every 50 \
            --sample-every 50 \
            2>&1 | tee -a "$TEACHER_DIR/train.log"
    fi

    echo ""
    echo "[$(date '+%H:%M:%S')] Teacher training complete!"
    echo "  Checkpoint: $TEACHER_DIR/teacher_best.pt"
}

# =============================================================================
# Step 2: Generate Trajectories
# =============================================================================
generate_trajectories() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== STEP 2: Generate Trajectories =========="
    echo ""

    TEACHER_CKPT="$TEACHER_DIR/teacher_best.pt"

    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "Error: Teacher checkpoint not found at $TEACHER_CKPT"
        echo "Run with --step teacher first"
        exit 1
    fi

    mkdir -p "$TRAJECTORY_DIR"

    python generate_trajectories.py \
        --teacher-ckpt "$TEACHER_CKPT" \
        --output-dir "$TRAJECTORY_DIR" \
        --data-path "$DATA_PATH" \
        --model "$MODEL" \
        --num-samples "$NUM_TRAJECTORIES" \
        --num-steps "$NUM_STEPS" \
        --batch-size 256 \
        --cfg-scale "$CFG_SCALE" \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "$TRAJECTORY_DIR/generate.log"

    echo ""
    echo "[$(date '+%H:%M:%S')] Trajectory generation complete!"
    echo "  Output: $TRAJECTORY_DIR"
}

# =============================================================================
# Step 3: Train Student
# =============================================================================
train_student() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== STEP 3: Train Student =========="
    echo ""

    if [ ! -d "$TRAJECTORY_DIR" ] || [ ! -f "$TRAJECTORY_DIR/metadata.json" ]; then
        echo "Error: Trajectories not found at $TRAJECTORY_DIR"
        echo "Run with --step trajectories first"
        exit 1
    fi

    mkdir -p "$STUDENT_DIR"

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" \
            train_ddp.py \
            --trajectory-dir "$TRAJECTORY_DIR" \
            --log-dir "$STUDENT_DIR" \
            --model "$MODEL" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$STUDENT_EPOCHS" \
            --loss-weighting "$LOSS_WEIGHTING" \
            --num-workers "$NUM_WORKERS" \
            --use-ema \
            --save-every 10 \
            --sample-every 10 \
            2>&1 | tee -a "$STUDENT_DIR/train.log"
    else
        python train_ddp.py \
            --trajectory-dir "$TRAJECTORY_DIR" \
            --log-dir "$STUDENT_DIR" \
            --model "$MODEL" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$STUDENT_EPOCHS" \
            --loss-weighting "$LOSS_WEIGHTING" \
            --num-workers "$NUM_WORKERS" \
            --use-ema \
            --save-every 10 \
            --sample-every 10 \
            2>&1 | tee -a "$STUDENT_DIR/train.log"
    fi

    echo ""
    echo "[$(date '+%H:%M:%S')] Student training complete!"
    echo "  Checkpoint: $STUDENT_DIR"
}

# =============================================================================
# Run pipeline
# =============================================================================
case $STEP in
    teacher)
        train_teacher
        ;;
    trajectories)
        generate_trajectories
        ;;
    student)
        train_student
        ;;
    all)
        # Skip steps that are already complete
        if [ -f "$TEACHER_DIR/teacher_best.pt" ]; then
            echo "[$(date '+%H:%M:%S')] Teacher already trained, skipping..."
        else
            train_teacher
        fi

        if [ -f "$TRAJECTORY_DIR/metadata.json" ]; then
            echo "[$(date '+%H:%M:%S')] Trajectories already generated, skipping..."
        else
            generate_trajectories
        fi

        train_student
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid steps: teacher, trajectories, student, all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
echo "  Teacher:      $TEACHER_DIR"
echo "  Trajectories: $TRAJECTORY_DIR"
echo "  Student:      $STUDENT_DIR"
echo "============================================================"

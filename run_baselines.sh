#!/bin/bash
#
# Run all baseline methods for comparison
#
# Prerequisites:
#   - Trained teacher at ./results/teacher/teacher_best.pt
#   - Generated trajectories at ./results/trajectories/
#
# Usage:
#   ./run_baselines.sh                    # Run all baselines
#   ./run_baselines.sh --baseline direct  # Run specific baseline
#

set -e

# =============================================================================
# Configuration
# =============================================================================
RESULTS_DIR="/scratch2/jiwan_chung/layerdistill/results"
TEACHER_CKPT="${RESULTS_DIR}/teacher/teacher_best.pt"
TRAJECTORY_DIR="${RESULTS_DIR}/trajectories"
DATA_PATH="./data"
MODEL="small"
NUM_WORKERS=4

# Check teacher exists
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Error: Teacher checkpoint not found at $TEACHER_CKPT"
    echo "Run ./train.sh --step teacher first"
    exit 1
fi

# =============================================================================
# Parse arguments
# =============================================================================
BASELINE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --gpus)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --baseline NAME   Run specific baseline: direct, progressive, consistency, reflow, or all"
            echo "  --gpus GPUS       Comma-separated GPU IDs"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  Running Baselines"
echo "============================================================"
echo "  Baseline: $BASELINE"
echo "  Teacher:  $TEACHER_CKPT"
echo "============================================================"
echo ""

# =============================================================================
# Baseline 2: Direct One-Step
# =============================================================================
run_direct() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== Baseline: Direct One-Step =========="
    echo ""

    # Check trajectories exist
    if [ ! -d "$TRAJECTORY_DIR" ]; then
        echo "Error: Trajectories not found at $TRAJECTORY_DIR"
        echo "Run ./train.sh --step trajectories first"
        exit 1
    fi

    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" \
            baselines/train_direct_onestep.py \
            --trajectory-dir "$TRAJECTORY_DIR" \
            --log-dir "${RESULTS_DIR}/baseline_direct" \
            --model "$MODEL" \
            --epochs 100 \
            --batch-size 128 \
            --num-workers "$NUM_WORKERS" \
            2>&1 | tee "${RESULTS_DIR}/baseline_direct/train.log"
    else
        python baselines/train_direct_onestep.py \
            --trajectory-dir "$TRAJECTORY_DIR" \
            --log-dir "${RESULTS_DIR}/baseline_direct" \
            --model "$MODEL" \
            --epochs 100 \
            --batch-size 128 \
            --num-workers "$NUM_WORKERS" \
            2>&1 | tee "${RESULTS_DIR}/baseline_direct/train.log"
    fi

    echo "[$(date '+%H:%M:%S')] Direct one-step baseline complete!"
}

# =============================================================================
# Baseline 3: Progressive Distillation
# =============================================================================
run_progressive() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== Baseline: Progressive Distillation =========="
    echo ""

    mkdir -p "${RESULTS_DIR}/baseline_progressive"

    python baselines/train_progressive.py \
        --teacher-ckpt "$TEACHER_CKPT" \
        --log-dir "${RESULTS_DIR}/baseline_progressive" \
        --data-path "$DATA_PATH" \
        --model "$MODEL" \
        --initial-steps 12 \
        --epochs-per-stage 50 \
        --batch-size 128 \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "${RESULTS_DIR}/baseline_progressive/train.log"

    echo "[$(date '+%H:%M:%S')] Progressive distillation complete!"
}

# =============================================================================
# Baseline 4: Consistency Distillation
# =============================================================================
run_consistency() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== Baseline: Consistency Distillation =========="
    echo ""

    mkdir -p "${RESULTS_DIR}/baseline_consistency"

    python baselines/train_consistency.py \
        --teacher-ckpt "$TEACHER_CKPT" \
        --log-dir "${RESULTS_DIR}/baseline_consistency" \
        --data-path "$DATA_PATH" \
        --model "$MODEL" \
        --epochs 100 \
        --batch-size 64 \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "${RESULTS_DIR}/baseline_consistency/train.log"

    echo "[$(date '+%H:%M:%S')] Consistency distillation complete!"
}

# =============================================================================
# Baseline 5: Rectified Flow Reflow
# =============================================================================
run_reflow() {
    echo ""
    echo "[$(date '+%H:%M:%S')] ========== Baseline: Rectified Flow Reflow =========="
    echo ""

    mkdir -p "${RESULTS_DIR}/baseline_reflow"

    python baselines/train_reflow.py \
        --teacher-ckpt "$TEACHER_CKPT" \
        --log-dir "${RESULTS_DIR}/baseline_reflow" \
        --data-path "$DATA_PATH" \
        --model "$MODEL" \
        --num-iterations 2 \
        --num-pairs 50000 \
        --epochs-per-iter 50 \
        --batch-size 128 \
        --num-workers "$NUM_WORKERS" \
        2>&1 | tee "${RESULTS_DIR}/baseline_reflow/train.log"

    echo "[$(date '+%H:%M:%S')] Reflow complete!"
}

# =============================================================================
# Run baselines
# =============================================================================
case $BASELINE in
    direct)
        run_direct
        ;;
    progressive)
        run_progressive
        ;;
    consistency)
        run_consistency
        ;;
    reflow)
        run_reflow
        ;;
    all)
        run_direct
        run_progressive
        run_consistency
        run_reflow
        ;;
    *)
        echo "Unknown baseline: $BASELINE"
        echo "Valid baselines: direct, progressive, consistency, reflow, all"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Baselines complete!"
echo "============================================================"
echo "  Direct:       ${RESULTS_DIR}/baseline_direct"
echo "  Progressive:  ${RESULTS_DIR}/baseline_progressive"
echo "  Consistency:  ${RESULTS_DIR}/baseline_consistency"
echo "  Reflow:       ${RESULTS_DIR}/baseline_reflow"
echo "============================================================"

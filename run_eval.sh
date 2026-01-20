#!/bin/bash
#
# Evaluate all trained models
#
# Usage:
#   ./run_eval.sh              # Evaluate all available models
#   ./run_eval.sh --fid        # Include FID computation (slower)
#

set -e

RESULTS_DIR="/scratch2/jiwan_chung/layerdistill/results"
NUM_SAMPLES=10000
BATCH_SIZE=256

# Parse arguments
COMPUTE_FID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fid)
            COMPUTE_FID="--fid"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  Evaluating All Models"
echo "============================================================"
echo ""

# Teacher (multi-step baseline)
eval_teacher() {
    CKPT="$RESULTS_DIR/teacher/teacher_best.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Teacher (12-step ODE)"
        python evaluate_teacher.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/teacher/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --generate --trajectory --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping teacher: checkpoint not found"
    fi
}

# Student (our method)
eval_student() {
    CKPT="$RESULTS_DIR/student/checkpoint_best.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Student (Unrolled SiT - Ours)"
        python evaluate.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/student/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --generate --trajectory --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping student: checkpoint not found"
    fi
}

# Direct one-step baseline
eval_direct() {
    CKPT="$RESULTS_DIR/baseline_direct/checkpoint_best.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Direct One-Step Baseline"
        python evaluate.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/baseline_direct/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --generate --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping direct baseline: checkpoint not found"
    fi
}

# Progressive distillation baseline
eval_progressive() {
    CKPT="$RESULTS_DIR/baseline_progressive/final_1step.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Progressive Distillation (1-step)"
        python evaluate_teacher.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/baseline_progressive/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --num-steps 1 \
            --generate --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping progressive baseline: checkpoint not found"
    fi
}

# Consistency distillation baseline
eval_consistency() {
    CKPT="$RESULTS_DIR/baseline_consistency/checkpoint_best.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Consistency Distillation"
        python evaluate_consistency.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/baseline_consistency/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --generate --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping consistency baseline: checkpoint not found"
    fi
}

# Reflow baseline
eval_reflow() {
    CKPT="$RESULTS_DIR/baseline_reflow/final_reflowed.pt"
    if [ -f "$CKPT" ]; then
        echo "[$(date '+%H:%M:%S')] Evaluating: Reflow (1-step)"
        python evaluate_teacher.py \
            --checkpoint "$CKPT" \
            --output-dir "$RESULTS_DIR/baseline_reflow/eval" \
            --num-samples $NUM_SAMPLES \
            --batch-size $BATCH_SIZE \
            --num-steps 1 \
            --generate --per-class $COMPUTE_FID
        echo ""
    else
        echo "Skipping reflow baseline: checkpoint not found"
    fi
}

# Run all evaluations
eval_teacher
eval_student
eval_direct
eval_progressive
eval_consistency
eval_reflow

echo "============================================================"
echo "  Evaluation Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  Teacher:      $RESULTS_DIR/teacher/eval"
echo "  Student:      $RESULTS_DIR/student/eval"
echo "  Direct:       $RESULTS_DIR/baseline_direct/eval"
echo "  Progressive:  $RESULTS_DIR/baseline_progressive/eval"
echo "  Consistency:  $RESULTS_DIR/baseline_consistency/eval"
echo "  Reflow:       $RESULTS_DIR/baseline_reflow/eval"
echo ""

# Print summary if FID was computed
if [ -n "$COMPUTE_FID" ]; then
    echo "============================================================"
    echo "  FID Summary"
    echo "============================================================"
    for dir in teacher student baseline_direct baseline_progressive baseline_consistency baseline_reflow; do
        FID_FILE="$RESULTS_DIR/$dir/eval/fid.txt"
        if [ -f "$FID_FILE" ]; then
            echo "  $dir: $(cat $FID_FILE)"
        fi
    done
    echo ""
fi

#!/bin/bash
# SLURM job array for DUCK+ PHEME — random 5-fold CV (4 variants x 5 folds = 20 jobs)
# Juno runs 4 at a time (max concurrent jobs per user).
#
# Usage:
#   sbatch run_pheme_random5fold.sh
#
# Monitor:
#   squeue --me
#   tail -f logs/pheme_r5f_<JOB_ID>_<ARRAY_ID>.out

#SBATCH --job-name=pheme_r5fold
#SBATCH --output=logs/pheme_r5f_%A_%a.out
#SBATCH --error=logs/pheme_r5f_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=2-00:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Map array index -> (variant, fold)
# Layout: variant changes every 5, fold cycles 0-4
VARIANTS=(baseline temp gated full)
VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
VARIANT=${VARIANTS[$VARIANT_IDX]}

WORK_DIR=~/work/duck-rumor-detection
SCRATCH_DIR=~/scratch/pheme_r5f_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/results" "$WORK_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR"

echo "Task $SLURM_ARRAY_TASK_ID: variant=$VARIANT fold=$FOLD split=random5fold"

module load miniconda
conda activate duck

cd "$WORK_DIR"

# Copy data to scratch for faster IO
cp -r data/pheme_npz    "$SCRATCH_DIR/"
cp -r data/pheme_5fold  "$SCRATCH_DIR/"

python duck_plus_pheme.py \
    --stage     random5fold \
    --variant   "$VARIANT" \
    --fold      "$FOLD" \
    --data-root "$SCRATCH_DIR" \
    --gpu       0

# Copy results back and clean up
cp -r "$SCRATCH_DIR"/results/* "$WORK_DIR/results/" 2>/dev/null || true
rm -rf "$SCRATCH_DIR"

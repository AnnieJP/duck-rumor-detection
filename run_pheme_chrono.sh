#!/bin/bash
set -euo pipefail
# SLURM job array for DUCK+ PHEME — chrono split (4 variants x 1 run = 4 jobs)
#
# Usage:
#   sbatch run_pheme_chrono.sh
#
# Monitor:
#   squeue --me
#   tail -f logs/pheme_chrono_<JOB_ID>_<ARRAY_ID>.out

#SBATCH --job-name=pheme_chrono
#SBATCH --output=logs/pheme_chrono_%A_%a.out
#SBATCH --error=logs/pheme_chrono_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=2-00:00:00
#SBATCH --partition=h100,a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

VARIANTS=(baseline temp gated full)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

WORK_DIR=~/work/duck-rumor-detection
SCRATCH_DIR=~/scratch/pheme_chrono_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/results" "$WORK_DIR/checkpoints"
mkdir -p "$SCRATCH_DIR"

echo "Task $SLURM_ARRAY_TASK_ID: variant=$VARIANT split=chrono"

module load miniconda
unset LD_LIBRARY_PATH
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duck_venv
nvidia-smi

cd "$WORK_DIR"

# Detect GPU and set batch size accordingly
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -q "A30"; then
    BATCH_SIZE=16
else
    BATCH_SIZE=32
fi
echo "GPU: $GPU_NAME — batch size: $BATCH_SIZE"

# Copy data to scratch for faster IO
cp -r data/pheme_npz    "$SCRATCH_DIR/"
cp -r data/pheme_chrono "$SCRATCH_DIR/"

python duck_plus_pheme.py \
    --stage      chrono \
    --variant    "$VARIANT" \
    --data-root  "$SCRATCH_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gpu        0

# Copy results back and clean up
cp -r "$SCRATCH_DIR"/results/* "$WORK_DIR/results/" 2>/dev/null || true
rm -rf "$SCRATCH_DIR"

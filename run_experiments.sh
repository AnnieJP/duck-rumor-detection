#!/bin/bash
# SLURM job array for DUCK+ factorial experiment on Juno HPC.
#
# Submits 80 jobs (4 variants x 2 datasets x 2 splits x 5 folds/runs).
# Each job is one (variant, dataset, split, fold) combination.
#
# Usage:
#   sbatch run_experiments.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/duck_plus_<JOB_ID>_<ARRAY_ID>.out

#SBATCH --job-name=duck_plus
#SBATCH --output=logs/duck_plus_%A_%a.out
#SBATCH --error=logs/duck_plus_%A_%a.err
#SBATCH --array=0-79
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p logs checkpoints results

# ---- Map SLURM_ARRAY_TASK_ID -> (variant, dataset, split, fold) ----
# Layout: 4 variants x 2 datasets x 2 splits x 5 folds = 80
VARIANTS=(baseline temp gated full)
DATASETS=(twitter15 twitter16)
SPLITS=(random chrono)
N_FOLDS=5

TASK_ID=$SLURM_ARRAY_TASK_ID

FOLD=$(( TASK_ID % N_FOLDS ))
REST=$(( TASK_ID / N_FOLDS ))
SPLIT_IDX=$(( REST % 2 ))
REST=$(( REST / 2 ))
DATASET_IDX=$(( REST % 2 ))
VARIANT_IDX=$(( REST / 2 ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}

echo "Task $TASK_ID: variant=$VARIANT dataset=$DATASET split=$SPLIT fold=$FOLD"

# ---- Environment setup ----
module load miniconda
conda activate duck

# ---- Run ----
python train_duck_plus.py \
    --variant   "$VARIANT" \
    --dataset   "$DATASET" \
    --split     "$SPLIT" \
    --fold      "$FOLD" \
    --run       "$FOLD" \
    --data_root data \
    --ckpt_dir  checkpoints \
    --result_csv results/results.csv \
    --hid_feats  64 \
    --ct_out     64 \
    --ut_out     64 \
    --lr_bert    2e-5 \
    --lr_other   1e-3 \
    --weight_decay 5e-5 \
    --n_epochs   50 \
    --batch_size 8 \
    --patience   10 \
    --num_workers 4 \
    --gpu        0

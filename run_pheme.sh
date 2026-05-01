#!/bin/bash
set -euo pipefail
# SLURM job array for DUCK+ PHEME experiments on Juno HPC.
#
# Submits 40 jobs: 4 variants x 2 splits x (9 LOEO events + 1 chrono) = 40
# Layout: variant (4) x split (2) x fold (5 for chrono=1 slot, 9 for loeo)
#
# Actual array indices:
#   0-35  : LOEO  (4 variants x 9 events)
#   36-39 : Chrono (4 variants x 1 run)
# Total: 40 jobs
#
# Usage:
#   sbatch run_pheme.sh
#
# Monitor:
#   squeue --me
#   tail -f logs/pheme_<JOB_ID>_<ARRAY_ID>.out
#
# Data layout expected on Juno (set up once before submitting):
#   ~/work/duck-rumor-detection/data/pheme_npz/
#   ~/work/duck-rumor-detection/data/pheme_loeo/
#   ~/work/duck-rumor-detection/data/pheme_chrono/

#SBATCH --job-name=duck_pheme
#SBATCH --output=logs/pheme_%A_%a.out
#SBATCH --error=logs/pheme_%A_%a.err
#SBATCH --array=0-39
#SBATCH --time=2-00:00:00
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ── Directories ──────────────────────────────────────────────────────────────
WORK_DIR=~/work/duck-rumor-detection
SCRATCH_DIR=~/scratch/duck_pheme_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/checkpoints" "$WORK_DIR/results"
mkdir -p "$SCRATCH_DIR"

# ── Map SLURM_ARRAY_TASK_ID -> (variant, split, fold/event) ──────────────────
VARIANTS=(baseline temp gated full)

LOEO_EVENTS=(
    charliehebdo
    ebola-essien
    ferguson
    germanwings-crash
    gurlitt
    ottawashooting
    prince-toronto
    putinmissing
    sydneysiege
)

N_LOEO=9    # events
N_LOEO_JOBS=$(( ${#VARIANTS[@]} * N_LOEO ))   # 36

TASK_ID=$SLURM_ARRAY_TASK_ID

if [ "$TASK_ID" -lt "$N_LOEO_JOBS" ]; then
    # LOEO block: jobs 0-35
    SPLIT=loeo
    EVENT_IDX=$(( TASK_ID % N_LOEO ))
    VARIANT_IDX=$(( TASK_ID / N_LOEO ))
    EVENT=${LOEO_EVENTS[$EVENT_IDX]}
    FOLD=$EVENT_IDX
else
    # Chrono block: jobs 36-39
    SPLIT=chrono
    CHRONO_IDX=$(( TASK_ID - N_LOEO_JOBS ))
    VARIANT_IDX=$CHRONO_IDX
    EVENT=chrono
    FOLD=0
fi

VARIANT=${VARIANTS[$VARIANT_IDX]}

echo "Task $TASK_ID: variant=$VARIANT split=$SPLIT event=$EVENT fold=$FOLD"

# ── Environment ──────────────────────────────────────────────────────────────
module load miniconda
source ~/.bashrc
conda activate duck

cd "$WORK_DIR"

# ── Copy data to scratch for faster IO ───────────────────────────────────────
echo "Copying data to scratch..."
cp -r data/pheme_npz    "$SCRATCH_DIR/"
cp -r data/pheme_loeo   "$SCRATCH_DIR/"
cp -r data/pheme_chrono "$SCRATCH_DIR/"

# ── Run one (variant, split, fold) per array task ────────────────────────────
if [ "$SPLIT" = "loeo" ]; then
    STAGE=realistic
else
    STAGE=chrono
fi

python duck_plus_pheme.py \
    --stage     "$STAGE" \
    --variant   "$VARIANT" \
    --fold      "$FOLD" \
    --data-root "$SCRATCH_DIR" \
    --batch-size 32 \
    --gpu       0

# ── Copy results back and clean up scratch ───────────────────────────────────
cp -r "$SCRATCH_DIR"/results/* "$WORK_DIR/results/" 2>/dev/null || true
echo "Cleaning scratch..."
rm -rf "$SCRATCH_DIR"

# Running DUCK+ PHEME on Juno

## One-time setup

```bash
# 1. SSH into Juno (must be on UTD network or VPN)
ssh <netid>@juno.utdallas.edu

# 2. Pull latest code (conda env already installed)
cd ~/work/duck-rumor-detection
git fetch origin
git checkout kishan/duck-plus
git reset --hard origin/kishan/duck-plus
```

```bash
# 3. Transfer PHEME data — run this from your Mac, not Juno
scp -r data/pheme_npz    <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_chrono <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_5fold  <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_loeo   <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
```

```bash
# 4. Create output directories (SLURM opens log files before the script body runs)
ssh <netid>@juno.utdallas.edu "mkdir -p ~/work/duck-rumor-detection/{logs,results,checkpoints}"
```

```bash
# 5. One-time conda setup on Juno (run interactively on the login node, only needed once)
module load miniconda
conda init bash
source ~/.bashrc
# The duck env should already exist; if not: conda create -p ~/work/envs/duck python=3.10
conda activate duck
```

```bash
# 6. Pre-cache BERT on the login node (compute nodes have no internet)
#    The cache lives in ~/.cache/huggingface and is shared with compute nodes.
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')"
```

---

## Running experiments

### Smoke test first (always do this)
Runs 1 epoch on 16 samples — confirms everything works before committing GPU time.

```bash
cd ~/work/duck-rumor-detection
module load miniconda && source ~/.bashrc && conda activate duck
python duck_plus_pheme.py --stage smoke --data-root data --gpu 0
```

### Chrono split (4 jobs, ~5 hrs total)
```bash
sbatch run_pheme_chrono.sh
```
- Submits 4 array jobs (one per variant: baseline, temp, gated, full)
- All 4 run in parallel immediately
- Results → `results/pheme_chrono.csv`

### Random 5-fold CV (20 jobs, ~26 hrs total)
```bash
sbatch run_pheme_random5fold.sh
```
- Submits 20 array jobs (4 variants × 5 folds)
- Juno runs 4 at a time automatically
- Results → `results/pheme_random5fold.csv`

### Full run — LOEO + Chrono (40 jobs, ~52 hrs total)
```bash
sbatch run_pheme.sh
```
- Submits 40 array jobs (4 variants × 9 LOEO events + 4 variants × 1 chrono)
- Jobs 0-35 run LOEO variants; jobs 36-39 run chrono variants
- Results → `results/pheme_realistic.csv` (LOEO) and `results/pheme_chrono.csv` (chrono)
- Requires `data/pheme_loeo/` on Juno (transferred in step 3 above)

---

## Monitoring

```bash
squeue --me                                        # see your running jobs
tail -f logs/pheme_chrono_<JOBID>_0.out           # watch chrono job 0
tail -f logs/pheme_r5f_<JOBID>_0.out              # watch random5fold job 0
scancel <JOBID>                                    # cancel a job if needed
```

---

## GPU / partition options

Both scripts target `h100,a30` — SLURM picks whichever has a free GPU first. The scripts auto-detect the GPU at runtime and set batch size accordingly (32 for H100, 16 for A30), so no manual changes are needed.

---

## If a job fails and you need to resume

The script skips already-completed runs automatically by reading the result CSV.
Just resubmit the same `sbatch` command — it will pick up where it left off.

---

## File reference

| File | Purpose |
|---|---|
| `duck_plus_pheme.py` | Main training script — run directly or via sbatch |
| `run_pheme_chrono.sh` | SLURM array for chrono split (4 jobs) |
| `run_pheme_random5fold.sh` | SLURM array for random 5-fold CV (20 jobs) |
| `run_pheme.sh` | SLURM array for full LOEO + chrono run (40 jobs) |
| `data/pheme_npz/` | Preprocessed PHEME threads |
| `data/pheme_chrono/` | Chrono split pkl files |
| `data/pheme_5fold/` | Random 5-fold split pkl files |
| `data/pheme_loeo/` | LOEO split pkl files (needed for smoke test and run_pheme.sh) |
| `results/pheme_chrono.csv` | Chrono results (written incrementally) |
| `results/pheme_random5fold.csv` | Random5fold results (written incrementally) |
| `results/pheme_realistic.csv` | LOEO results (written incrementally) |
| `checkpoints/pheme_chrono/` | Best model per variant |
| `checkpoints/pheme_random5fold/` | Best model per variant/fold |

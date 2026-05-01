# Running DUCK+ PHEME on Juno

## One-time setup

```bash
# SSH into Juno (must be on UTD network or VPN)
ssh <netid>@juno.utdallas.edu

# Clone repo into your work directory
cd ~/work
git clone --branch kishan/duck-plus https://github.com/AnnieJP/duck-rumor-detection.git
cd duck-rumor-detection

# Create conda environment
module load miniconda
conda create -n duck python=3.10 -y
conda activate duck
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric transformers scikit-learn numpy pandas

# Transfer PHEME data (run this from your Mac, not Juno)
scp -r data/pheme_npz    <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_loeo   <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_chrono <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_5fold  <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
```

---

## Running experiments

### Smoke test first (always do this)
Runs 1 epoch on 16 samples — confirms everything works before committing GPU time.

```bash
cd ~/work/duck-rumor-detection
module load miniconda && conda activate duck
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

Both scripts target `h100` with `--batch-size 32`. To use `a30` instead:

1. Edit the `.sh` file: change `--partition=h100` → `--partition=a30`
2. Change `--batch-size 32` → `--batch-size 16`

---

## If a job fails and you need to resume

The script skips already-completed runs automatically by reading the result CSV.
Just resubmit the same `sbatch` command — it will pick up where it left off.

---

## File reference

| File | Purpose |
|---|---|
| `duck_plus_pheme.py` | Main training script — run directly or via sbatch |
| `run_pheme_chrono.sh` | SLURM array for chrono split |
| `run_pheme_random5fold.sh` | SLURM array for random 5-fold CV |
| `data/pheme_npz/` | Preprocessed PHEME threads |
| `data/pheme_chrono/` | Chrono split pkl files |
| `data/pheme_5fold/` | Random 5-fold split pkl files |
| `results/pheme_chrono.csv` | Chrono results (written incrementally) |
| `results/pheme_random5fold.csv` | Random5fold results (written incrementally) |
| `checkpoints/pheme_chrono/` | Best model per variant |
| `checkpoints/pheme_random5fold/` | Best model per variant/fold |

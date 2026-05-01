# Juno Run Steps — chrono + random5fold

---

## Step 0 — Upload data (run from your Mac terminal, NOT Juno)

One-time only. The PHEME data folders are not in git, so they must be copied
manually. Total size is ~58 MB. Replace `<netid>` with your UTD NetID.

```bash
cd ~/Projects/duck-rumor-detection
scp -r data/pheme_npz    <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_chrono <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_5fold  <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
scp -r data/pheme_loeo   <netid>@juno.utdallas.edu:~/work/duck-rumor-detection/data/
```

Then ssh into Juno: `ssh <netid>@juno.utdallas.edu`

---

## First-time setup on Juno (steps 1–7)

```bash
# 1. Pull latest code
cd ~/work/duck-rumor-detection
git fetch origin
git checkout kishan/duck-plus
git reset --hard origin/kishan/duck-plus
```

```bash
# 2. Create output directories (one-time)
mkdir -p logs results checkpoints
```

```bash
# 3. One-time conda setup
module load miniconda
conda init bash
source ~/.bashrc
```

```bash
# 4. Verify the duck env exists and dependencies work
conda env list                   # confirm "duck" is listed
conda activate duck
python -c "import torch, torch_geometric, transformers; print('deps ok')"
```

```bash
# 5. Verify data is in place
ls data/pheme_npz | head -3      # should show .npz files
ls data/pheme_chrono             # train.pkl, val.pkl, test.pkl
ls data/pheme_5fold              # fold0/ ... fold4/
ls data/pheme_loeo               # event_charliehebdo/ ... (only needed for smoke + run_pheme.sh)
```

```bash
# 6. Pre-cache BERT (login node has internet, compute nodes do not)
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')"
```

```bash
# 7. Smoke test on a compute node (NOT the login node)
salloc -p h100,a30 --gres=gpu:1 --time=00:15:00 --mem=16G srun --pty bash -c '
    module load miniconda && source ~/.bashrc && conda activate duck && \
    cd ~/work/duck-rumor-detection && \
    python duck_plus_pheme.py --stage smoke --data-root data --gpu 0
'
```

---

## Submit experiments (step 8)

```bash
# 8. Submit chrono and random5fold jobs
sbatch run_pheme_chrono.sh
sbatch run_pheme_random5fold.sh
```

---

## Monitor (step 9)

```bash
# 9. Monitor
squeue --me                                      # see your queued/running jobs
tail -f logs/pheme_chrono_<JOBID>_0.out          # watch chrono job 0 live
tail -f logs/pheme_r5f_<JOBID>_0.out             # watch random5fold job 0 live
scancel <JOBID>                                  # cancel a job if needed
```

Results land in:
- `results/pheme_chrono.csv`
- `results/pheme_random5fold.csv`

---

## Subsequent sessions

After the first run, just do:

```bash
cd ~/work/duck-rumor-detection
git fetch origin && git reset --hard origin/kishan/duck-plus
sbatch run_pheme_chrono.sh
sbatch run_pheme_random5fold.sh
squeue --me
```

The scripts skip already-completed runs by reading the result CSV, so resubmitting after a partial failure picks up where it left off.

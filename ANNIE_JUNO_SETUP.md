# Annie's Juno Setup — DUCK+ PHEME Experiments

Your Juno username: `ajp230014`

---

## Step 0 — SSH into Juno

```bash
ssh ajp230014@juno-l-01.utdallas.edu
```

---

## Step 1 — Clone the repo

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/AnnieJP/duck-rumor-detection.git
cd duck-rumor-detection
git checkout kishan/duck-plus
```

---

## Step 2 — Create output directories

```bash
mkdir -p ~/work/duck-rumor-detection/logs
mkdir -p ~/work/duck-rumor-detection/results
mkdir -p ~/work/duck-rumor-detection/checkpoints
mkdir -p ~/scratch
mkdir -p ~/tmp_pip
```

---

## Step 3 — Set up conda environment

```bash
module load miniconda
conda create -n duck_venv python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duck_venv
cd ~
TMPDIR=~/tmp_pip pip install --no-cache-dir -r ~/work/duck-rumor-detection/requirements-juno.txt
```

This downloads ~3 GB of packages. Takes 5–10 minutes.

---

## Step 4 — Pre-cache BERT model

Compute nodes may not have internet access. Cache BERT now from the login node:

```bash
module load miniconda
unset LD_LIBRARY_PATH
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duck_venv
python -c "from transformers import BertModel, BertTokenizer; BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased'); print('BERT cached ok')"
```

---

## Step 5 — Edit the run scripts (one-time)

In both `run_pheme_chrono.sh` and `run_pheme_random5fold.sh`, lines 13–14 have
Kishan's username hardcoded. Change `kxr240006` to `ajp230014`:

```bash
sed -i 's/kxr240006/ajp230014/g' run_pheme_chrono.sh run_pheme_random5fold.sh
```

Verify:
```bash
grep ajp230014 run_pheme_chrono.sh run_pheme_random5fold.sh
```

Should show two lines each with your username in the log paths.

---

## Step 6 — Verify data is in place

Data is included in the repo — no upload needed.

```bash
ls ~/work/duck-rumor-detection/data/pheme_npz    | head -3   # .npz files
ls ~/work/duck-rumor-detection/data/pheme_chrono             # train.pkl val.pkl test.pkl
ls ~/work/duck-rumor-detection/data/pheme_5fold              # fold0/ ... fold4/
```

---

## Step 7 — Submit jobs

```bash
cd ~/work/duck-rumor-detection
sbatch --partition=h100 run_pheme_chrono.sh
sbatch --partition=a30  run_pheme_chrono.sh
sbatch --partition=h100 run_pheme_random5fold.sh
sbatch --partition=a30  run_pheme_random5fold.sh
squeue --me
```

Submitting to both `h100` and `a30` is intentional — whichever GPU frees up
first will run your job. If both run simultaneously, the result CSV will just
have duplicate rows (safe to dedupe with pandas later).

---

## Step 8 — Monitor

```bash
squeue --me                                        # see queue status
ls logs/                                           # check for log files
tail -f logs/pheme_chrono_*_0.out                 # watch chrono job live
tail -f logs/pheme_r5f_*_0.out                    # watch random5fold job live
```

Results land in:
- `results/pheme_chrono.csv`
- `results/pheme_random5fold.csv`

---

## Troubleshooting

**`conda activate` fails:** Run `conda init bash && exec bash` first, then retry.

**`import torch` fails with oneMKL error:** Run `unset LD_LIBRARY_PATH` before activating the env. This is caused by Juno's miniconda module polluting the library path.

**Jobs fail instantly (exit code 1, no logs):** Make sure you ran the `sed` command in Step 5 to replace the username in the log paths.

**Jobs stuck in PD (pending):** Normal — H100 and A30 nodes are shared. Check queue with `squeue -p h100` or `squeue -p a30`. Jobs will run when a slot opens.

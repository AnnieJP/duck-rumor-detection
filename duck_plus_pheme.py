"""
DUCK+ PHEME experiment script — runs on Juno HPC (or any Linux machine with GPU).

Usage:
    python duck_plus_pheme.py --stage chrono
    python duck_plus_pheme.py --stage smoke
    python duck_plus_pheme.py --stage full

Stages:
    smoke    — 1 variant, 1 epoch, 16 samples  (~5 min, sanity check)
    chrono   — 4 variants x chrono split        (~5 hrs on A100)
    realistic— 1 variant x loeo + chrono        (~10 hrs on A100)
    full     — 4 variants x loeo + chrono        (~52 hrs, use Juno job array)

Data expected at --data-root (default: data/):
    pheme_npz/        — preprocessed .npz files from preprocess_pheme.py
    pheme_loeo/       — LOEO split pkl files
    pheme_chrono/     — chronological split pkl files
"""

import os
import sys
import csv
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from dataset import DuckPlusDataset
from duck_plus import DuckPlus
from train_duck_plus import set_seed, EarlyStopping, evaluate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOEO_EVENTS = [
    'charliehebdo', 'ebola-essien', 'ferguson', 'germanwings-crash',
    'gurlitt', 'ottawashooting', 'prince-toronto', 'putinmissing', 'sydneysiege'
]

PRESETS = {
    'smoke': {
        'variants': ['full'],
        'splits': ['loeo'],
        'loeo_events': ['charliehebdo'],
        'n_epochs': 1,
        'batch_size': 4,
        'smoke_n': 16,
        'result_csv': 'results/pheme_smoke.csv',
        'ckpt_dir': 'checkpoints/pheme_smoke',
    },
    'chrono': {
        'variants': ['baseline', 'temp', 'gated', 'full'],
        'splits': ['chrono'],
        'loeo_events': [],
        'n_epochs': 50,
        'batch_size': 8,
        'smoke_n': None,
        'result_csv': 'results/pheme_chrono.csv',
        'ckpt_dir': 'checkpoints/pheme_chrono',
    },
    'random5fold': {
        'variants': ['baseline', 'temp', 'gated', 'full'],
        'splits': ['random5fold'],
        'loeo_events': [],
        'n_epochs': 50,
        'batch_size': 8,
        'smoke_n': None,
        'result_csv': 'results/pheme_random5fold.csv',
        'ckpt_dir': 'checkpoints/pheme_random5fold',
    },
    'realistic': {
        'variants': ['full'],
        'splits': ['loeo', 'chrono'],
        'loeo_events': LOEO_EVENTS,
        'n_epochs': 50,
        'batch_size': 8,
        'smoke_n': None,
        'result_csv': 'results/pheme_realistic.csv',
        'ckpt_dir': 'checkpoints/pheme_realistic',
    },
    'full': {
        'variants': ['baseline', 'temp', 'gated', 'full'],
        'splits': ['loeo', 'chrono'],
        'loeo_events': LOEO_EVENTS,
        'n_epochs': 50,
        'batch_size': 8,
        'smoke_n': None,
        'result_csv': 'results/pheme_results.csv',
        'ckpt_dir': 'checkpoints/pheme',
    },
}

PHEME_RESULT_COLS = [
    'variant', 'dataset', 'split', 'run', 'event',
    'test_acc', 'test_macro_f1',
    'f1_NR', 'f1_FR', 'f1_TR', 'f1_UR',
    'val_f1', 'val_acc',
]

HID_FEATS    = 64
CT_OUT       = 64
UT_OUT       = 64
LR_BERT      = 2e-5
LR_OTHER     = 1e-3
WEIGHT_DECAY = 5e-5
PATIENCE     = 10
NUM_WORKERS  = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def append_result(result, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PHEME_RESULT_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result.get(k, '') for k in PHEME_RESULT_COLS})


def load_splits(split, event, data_root, fold=0):
    if split == 'loeo':
        fold_dir = os.path.join(data_root, 'pheme_loeo', f'event_{event}')
        with open(os.path.join(fold_dir, 'train.pkl'), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(fold_dir, 'test.pkl'), 'rb') as f:
            test_ids = pickle.load(f)
        val_ids = test_ids
    elif split == 'random5fold':
        fold_dir = os.path.join(data_root, 'pheme_5fold', f'fold{fold}')
        with open(os.path.join(fold_dir, '_x_train.pkl'), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(fold_dir, '_x_test.pkl'), 'rb') as f:
            test_ids = pickle.load(f)
        val_ids = test_ids   # same protocol as original DUCK
    else:
        chrono_dir = os.path.join(data_root, 'pheme_chrono')
        with open(os.path.join(chrono_dir, 'train.pkl'), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(chrono_dir, 'val.pkl'), 'rb') as f:
            val_ids = pickle.load(f)
        with open(os.path.join(chrono_dir, 'test.pkl'), 'rb') as f:
            test_ids = pickle.load(f)
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=list(PRESETS), default='chrono')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    cfg = PRESETS[args.stage]
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Stage: {args.stage} | Device: {device}')

    npz_dir  = os.path.join(args.data_root, 'pheme_npz')
    ckpt_dir = cfg['ckpt_dir']
    result_csv = cfg['result_csv']
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Build job list
    jobs = []
    for variant in cfg['variants']:
        for split in cfg['splits']:
            if split == 'loeo':
                for i, event in enumerate(cfg['loeo_events']):
                    jobs.append((variant, split, i, event))
            elif split == 'random5fold':
                for fold_idx in range(5):
                    jobs.append((variant, split, fold_idx, None))
            else:
                jobs.append((variant, split, 0, None))

    print(f'Total jobs: {len(jobs)}')

    # Resume support
    completed = set()
    if os.path.exists(result_csv):
        with open(result_csv) as f:
            for row in csv.DictReader(f):
                completed.add((row['variant'], row['split'], row['run'], row.get('event', '')))
        print(f'Resuming: {len(completed)} already done.')

    t_start = time.time()
    done = 0
    total = len(jobs)

    for variant, split, fold, event in jobs:
        key = (variant, split, str(fold), event or '')
        if key in completed:
            print(f'  SKIP: variant={variant} split={split} event={event}')
            done += 1
            continue

        label = event or 'chrono'
        print(f'\n[{done+1}/{total}] variant={variant} split={split} fold={label}')

        set_seed(42 + fold)

        train_ids, val_ids, test_ids = load_splits(split, event, args.data_root, fold=fold)

        if cfg['smoke_n'] is not None:
            train_ids = train_ids[:cfg['smoke_n']]
            val_ids   = val_ids[:cfg['smoke_n']]
            test_ids  = test_ids[:cfg['smoke_n']]

        train_loader = DataLoader(DuckPlusDataset(train_ids, npz_dir),
                                  batch_size=cfg['batch_size'], shuffle=True,
                                  num_workers=NUM_WORKERS)
        val_loader   = DataLoader(DuckPlusDataset(val_ids, npz_dir),
                                  batch_size=cfg['batch_size'], shuffle=False,
                                  num_workers=NUM_WORKERS)
        test_loader  = DataLoader(DuckPlusDataset(test_ids, npz_dir),
                                  batch_size=cfg['batch_size'], shuffle=False,
                                  num_workers=NUM_WORKERS)

        model = DuckPlus(
            variant=variant, hid_feats=HID_FEATS,
            ct_out=CT_OUT, ut_out=UT_OUT, num_classes=4,
        ).to(device)

        bert_ids = set()
        for branch in ['comment_tree', 'comment_chain', 'user_tree']:
            m = getattr(model, branch, None)
            if m and hasattr(m, 'bert'):
                bert_ids.update(id(p) for p in m.bert.parameters())
        optimizer = torch.optim.Adam([
            {'params': [p for p in model.parameters() if id(p) in bert_ids],     'lr': LR_BERT},
            {'params': [p for p in model.parameters() if id(p) not in bert_ids], 'lr': LR_OTHER},
        ], weight_decay=WEIGHT_DECAY)

        ckpt_path = os.path.join(ckpt_dir, f'{variant}_pheme_{split}_r{fold}.pt')
        stopper   = EarlyStopping(patience=PATIENCE, ckpt_path=ckpt_path)

        for epoch in range(cfg['n_epochs']):
            model.train()
            tr_loss, tr_correct, tr_total = [], 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                labels = batch.y.view(-1)
                loss = F.nll_loss(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss.append(loss.item())
                tr_correct += logits.argmax(-1).eq(labels).sum().item()
                tr_total   += labels.size(0)

            val_loss, val_acc, val_f1, val_f1_cls = evaluate(model, val_loader, device)
            tr_acc = tr_correct / max(tr_total, 1)
            print(f'  Ep {epoch:03d} tr_loss={np.mean(tr_loss):.3f} tr_acc={tr_acc:.3f} '
                  f'val_loss={val_loss:.3f} val_f1={val_f1:.3f}')

            stopper(val_f1, val_acc, val_f1_cls, model)
            if stopper.early_stop:
                print(f'  Early stop at epoch {epoch}.')
                break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        _, test_acc, test_f1, test_f1_cls = evaluate(model, test_loader, device)

        result = {
            'variant': variant, 'dataset': 'pheme', 'split': split,
            'run': fold, 'event': event or 'chrono',
            'test_acc': round(test_acc, 4), 'test_macro_f1': round(test_f1, 4),
            'f1_NR': round(test_f1_cls[0], 4), 'f1_FR': round(test_f1_cls[1], 4),
            'f1_TR': round(test_f1_cls[2], 4), 'f1_UR': round(test_f1_cls[3], 4),
            'val_f1': round(stopper.best_f1, 4), 'val_acc': round(stopper.best_acc, 4),
        }
        append_result(result, result_csv)
        completed.add(key)
        done += 1

        elapsed     = (time.time() - t_start) / 60
        avg_per_job = elapsed / done if done else 0
        print(f'  TEST acc={test_acc:.4f} macro-F1={test_f1:.4f} '
              f'NR={test_f1_cls[0]:.4f} FR={test_f1_cls[1]:.4f} '
              f'TR={test_f1_cls[2]:.4f} UR={test_f1_cls[3]:.4f}')
        print(f'  Progress: {done}/{total} | ~{avg_per_job*(total-done):.0f} min remaining')

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\nDone. Results saved to {result_csv}')


if __name__ == '__main__':
    main()

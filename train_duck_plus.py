"""
Training script for DUCK+ factorial experiment.

Runs one (variant, dataset, split_type, fold/run) combination per invocation.
Results are appended to results/results.csv for easy aggregation.

Usage examples:
  # single run
  python train_duck_plus.py --variant full --dataset twitter15 --split random --fold 0

  # all 4 variants x 2 datasets x 2 splits x 5 folds = 80 runs (use run_experiments.sh)
  python train_duck_plus.py --variant baseline --dataset twitter16 --split chrono --run 0

Juno:
  sbatch run_experiments.sh
"""

import os
import sys
import csv
import time
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from dataset import DuckPlusDataset
from duck_plus import DuckPlus


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Early stopping (fixed version — monitors val macro-F1)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, ckpt_path: str = 'best_model.pt'):
        self.patience   = patience
        self.ckpt_path  = ckpt_path
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.best_f1    = 0.0
        self.best_acc   = 0.0
        self.best_f1_per_class = [0.0, 0.0, 0.0, 0.0]

    def __call__(self, val_f1: float, val_acc: float, f1_per_class: list,
                 model: torch.nn.Module):
        score = val_f1
        if self.best_score is None or score > self.best_score:
            self.best_score        = score
            self.best_f1           = val_f1
            self.best_acc          = val_acc
            self.best_f1_per_class = f1_per_class
            torch.save(model.state_dict(), self.ckpt_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            labels = batch.y.view(-1)
            loss = F.nll_loss(logits, labels)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc    = accuracy_score(all_labels, all_preds)
    macro  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_cls = f1_score(all_labels, all_preds, average=None,
                       labels=[0, 1, 2, 3], zero_division=0).tolist()
    return float(np.mean(losses)), acc, macro, per_cls


# ---------------------------------------------------------------------------
# One train/eval loop for a single fold or chrono run
# ---------------------------------------------------------------------------

def run_one(
    variant: str,
    dataset: str,
    split_type: str,
    fold_or_run: int,
    data_root: str,
    hid_feats: int,
    ct_out: int,
    ut_out: int,
    lr_bert: float,
    lr_other: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    patience: int,
    num_workers: int,
    ckpt_dir: str,
    device: torch.device,
):
    npz_dir = os.path.join(data_root, f'{dataset}_npz')

    # Load split IDs
    if split_type == 'random':
        fold_dir = os.path.join(data_root, f'{dataset}_5fold',
                                f'fold{fold_or_run}')
        with open(os.path.join(fold_dir, '_x_train.pkl'), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(fold_dir, '_x_test.pkl'), 'rb') as f:
            val_ids = pickle.load(f)
        test_ids = val_ids   # 5-fold: test == val fold
    else:
        chrono_dir = os.path.join(data_root, f'{dataset}_chrono')
        with open(os.path.join(chrono_dir, 'train.pkl'), 'rb') as f:
            train_ids = pickle.load(f)
        with open(os.path.join(chrono_dir, 'val.pkl'), 'rb') as f:
            val_ids = pickle.load(f)
        with open(os.path.join(chrono_dir, 'test.pkl'), 'rb') as f:
            test_ids = pickle.load(f)

    train_ds = DuckPlusDataset(train_ids, npz_dir)
    val_ds   = DuckPlusDataset(val_ids,   npz_dir)
    test_ds  = DuckPlusDataset(test_ids,  npz_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    model = DuckPlus(
        variant=variant,
        hid_feats=hid_feats,
        ct_out=ct_out,
        ut_out=ut_out,
        num_classes=4,
    ).to(device)

    # Separate LR for BERT params vs the rest (following DUCK appendix)
    bert_param_ids = set()
    for branch in ['comment_tree', 'comment_chain', 'user_tree']:
        module = getattr(model, branch, None)
        if module is None:
            continue
        bert_mod = getattr(module, 'bert', None)
        if bert_mod is not None:
            bert_param_ids.update(id(p) for p in bert_mod.parameters())

    bert_params  = [p for p in model.parameters() if id(p) in bert_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in bert_param_ids]

    optimizer = torch.optim.Adam(
        [{'params': bert_params,  'lr': lr_bert},
         {'params': other_params, 'lr': lr_other}],
        weight_decay=weight_decay,
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir, f'{variant}_{dataset}_{split_type}_r{fold_or_run}.pt')
    stopper = EarlyStopping(patience=patience, ckpt_path=ckpt_path)

    for epoch in range(n_epochs):
        model.train()
        train_losses, train_correct, train_total = [], 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            labels = batch.y.view(-1)
            loss = F.nll_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_correct += logits.argmax(-1).eq(labels).sum().item()
            train_total   += labels.size(0)

        val_loss, val_acc, val_f1, val_f1_cls = evaluate(model, val_loader, device)
        train_acc = train_correct / max(train_total, 1)

        print(
            f'[{variant}|{dataset}|{split_type}|r{fold_or_run}] '
            f'Epoch {epoch:03d} '
            f'train_loss={np.mean(train_losses):.4f} train_acc={train_acc:.4f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}'
        )

        stopper(val_f1, val_acc, val_f1_cls, model)
        if stopper.early_stop:
            print(f'  Early stopping at epoch {epoch}')
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, test_f1, test_f1_cls = evaluate(model, test_loader, device)

    print(
        f'\n=== TEST [{variant}|{dataset}|{split_type}|r{fold_or_run}] ==='
        f'\n  acc={test_acc:.4f}  macro-F1={test_f1:.4f}'
        f'\n  NR={test_f1_cls[0]:.4f}  FR={test_f1_cls[1]:.4f}'
        f'  TR={test_f1_cls[2]:.4f}  UR={test_f1_cls[3]:.4f}\n'
    )

    return {
        'variant':      variant,
        'dataset':      dataset,
        'split':        split_type,
        'run':          fold_or_run,
        'test_acc':     round(test_acc, 4),
        'test_macro_f1': round(test_f1, 4),
        'f1_NR':        round(test_f1_cls[0], 4),
        'f1_FR':        round(test_f1_cls[1], 4),
        'f1_TR':        round(test_f1_cls[2], 4),
        'f1_UR':        round(test_f1_cls[3], 4),
        'val_f1':       round(stopper.best_f1, 4),
        'val_acc':      round(stopper.best_acc, 4),
    }


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

RESULT_COLS = [
    'variant', 'dataset', 'split', 'run',
    'test_acc', 'test_macro_f1',
    'f1_NR', 'f1_FR', 'f1_TR', 'f1_UR',
    'val_f1', 'val_acc',
]


def append_result(result: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result.get(k, '') for k in RESULT_COLS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train DUCK+ variant')
    p.add_argument('--variant',   choices=['baseline', 'temp', 'gated', 'full'],
                   default='full')
    p.add_argument('--dataset',   choices=['twitter15', 'twitter16'],
                   default='twitter15')
    p.add_argument('--split',     choices=['random', 'chrono'],
                   default='random')
    p.add_argument('--fold',      type=int, default=0,
                   help='Fold index (0-4) for random split')
    p.add_argument('--run',       type=int, default=0,
                   help='Run index for chrono split (sets random seed)')
    p.add_argument('--data_root', default='data')
    p.add_argument('--ckpt_dir',  default='checkpoints')
    p.add_argument('--result_csv', default='results/results.csv')

    # Hyperparameters
    p.add_argument('--hid_feats',     type=int,   default=64)
    p.add_argument('--ct_out',        type=int,   default=64)
    p.add_argument('--ut_out',        type=int,   default=64)
    p.add_argument('--lr_bert',       type=float, default=2e-5)
    p.add_argument('--lr_other',      type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=5e-5)
    p.add_argument('--n_epochs',      type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--gpu',           type=int,   default=0,
                   help='GPU index; -1 for CPU')
    return p.parse_args()


def main():
    args = parse_args()

    fold_or_run = args.fold if args.split == 'random' else args.run
    seed = args.seed + fold_or_run
    set_seed(seed)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    print(f'Running: variant={args.variant} dataset={args.dataset} '
          f'split={args.split} fold/run={fold_or_run}')

    result = run_one(
        variant=args.variant,
        dataset=args.dataset,
        split_type=args.split,
        fold_or_run=fold_or_run,
        data_root=args.data_root,
        hid_feats=args.hid_feats,
        ct_out=args.ct_out,
        ut_out=args.ut_out,
        lr_bert=args.lr_bert,
        lr_other=args.lr_other,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        num_workers=args.num_workers,
        ckpt_dir=args.ckpt_dir,
        device=device,
    )

    append_result(result, args.result_csv)
    print(f'Result appended to {args.result_csv}')


if __name__ == '__main__':
    main()

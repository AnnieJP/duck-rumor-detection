"""
Preprocessing pipeline for DUCK+ on Twitter15/Twitter16.

Reads raw tree files and converts each story into a .npz file containing:
  - edgematrix : (2, E) int array  — COO edge list (parent_idx -> child_idx)
  - root       : (1,) str array    — source tweet text
  - nodecontent: (N,) str array    — text of each node (empty string if retweet)
  - timestamps : (N,) float array  — time delay in minutes from source post
  - node_ids   : (N,) str array    — tweet IDs (index 0 = source)
  - user_ids   : (N,) str array    — user IDs per node
  - y          : scalar int        — label (0=NR,1=FR,2=TR,3=UR for 4-class)
  - rootindex  : scalar int        — always 0
  - topindex   : (K,) int array    — indices of direct children of root
  - triIndex   : (T,) int array    — indices of nodes at depth <= 2

Also produces:
  - data/<dataset>_5fold/         — 5 fold train/test splits (random)
  - data/<dataset>_chrono/        — chronological 60/20/20 split

Usage:
  python preprocess.py --dataset twitter15
  python preprocess.py --dataset twitter16
  python preprocess.py --dataset all
"""

import os
import re
import ast
import argparse
import pickle
import random
import numpy as np
from collections import defaultdict


LABEL_MAP = {
    'non-rumor': 0,
    'false':     1,
    'true':      2,
    'unverified': 3,
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_tree_file(filepath):
    """Parse one tree file into a list of (parent_tuple, child_tuple) edges."""
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('->')
            if len(parts) != 2:
                continue
            try:
                parent = ast.literal_eval(parts[0].strip())
                child  = ast.literal_eval(parts[1].strip())
            except Exception:
                continue
            edges.append((parent, child))
    return edges


def build_graph(edges, source_text, source_id):
    """
    Convert raw edge list into indexed graph arrays.

    Returns:
        node_order   : list of tweet_ids in BFS order (index 0 = source)
        uid_map      : tweet_id -> user_id
        time_map     : tweet_id -> time delay (minutes)
        edge_list    : list of (parent_idx, child_idx)
        node_content : list of str (empty unless we have text)
    """
    # Collect all unique tweet nodes (skip ROOT)
    uid_map  = {}   # tweet_id -> user_id
    time_map = {}   # tweet_id -> float time delay
    parent_map = defaultdict(list)  # parent_tweet_id -> [child_tweet_id]

    for parent, child in edges:
        p_uid, p_tid, p_time = parent
        c_uid, c_tid, c_time = child

        if p_tid != 'ROOT':
            uid_map[p_tid]  = p_uid
            time_map[p_tid] = float(p_time)
        uid_map[c_tid]  = c_uid
        time_map[c_tid] = float(c_time)

        if p_tid == 'ROOT':
            p_tid = source_id
        parent_map[p_tid].append(c_tid)

    # BFS order starting from source
    node_order = [source_id]
    visited = {source_id}
    queue = [source_id]
    while queue:
        curr = queue.pop(0)
        for child in sorted(parent_map.get(curr, []),
                            key=lambda t: time_map.get(t, 0.0)):
            if child not in visited:
                visited.add(child)
                node_order.append(child)
                queue.append(child)

    idx_map = {tid: i for i, tid in enumerate(node_order)}

    edge_list = []
    for parent, child in edges:
        p_uid, p_tid, p_time = parent
        c_uid, c_tid, c_time = child
        if p_tid == 'ROOT':
            p_tid = source_id
        if p_tid in idx_map and c_tid in idx_map:
            pi = idx_map[p_tid]
            ci = idx_map[c_tid]
            if pi != ci:
                edge_list.append((pi, ci))

    node_content = []
    for tid in node_order:
        if tid == source_id:
            node_content.append(source_text)
        else:
            node_content.append('')

    return node_order, uid_map, time_map, edge_list, node_content


def compute_temporal_features(node_order, time_map, edge_list, source_id):
    """
    Compute the 6 temporal/structural edge features needed by DUCK+.
    Returns dict: edge (p_idx, c_idx) -> feature vector of length 6.
    """
    n = len(node_order)
    idx_map = {tid: i for i, tid in enumerate(node_order)}
    times = np.array([time_map.get(tid, 0.0) for tid in node_order], dtype=np.float32)
    t0 = times[0]

    # Depth via BFS from root
    depth = np.full(n, -1, dtype=np.int32)
    depth[0] = 0
    children_of = defaultdict(list)
    for pi, ci in edge_list:
        children_of[pi].append(ci)
    queue = [0]
    while queue:
        curr = queue.pop(0)
        for child in children_of[curr]:
            if depth[child] < 0:
                depth[child] = depth[curr] + 1
                queue.append(child)

    # Subtree sizes via reverse BFS
    subtree_size = np.ones(n, dtype=np.int32)
    order = list(range(n))
    parent_of = {}
    for pi, ci in edge_list:
        parent_of[ci] = pi
    for ci in reversed(order):
        if ci in parent_of:
            subtree_size[parent_of[ci]] += subtree_size[ci]

    # Sibling rank: among siblings of same parent, rank by timestamp
    sibling_rank = np.zeros(n, dtype=np.float32)
    siblings_by_parent = defaultdict(list)
    for pi, ci in edge_list:
        siblings_by_parent[pi].append(ci)
    for pi, sibs in siblings_by_parent.items():
        sorted_sibs = sorted(sibs, key=lambda c: times[c])
        for rank, ci in enumerate(sorted_sibs):
            sibling_rank[ci] = rank / max(1, len(sorted_sibs) - 1)

    edge_features = {}
    for pi, ci in edge_list:
        dt     = np.log1p(max(0.0, float(times[ci] - times[pi])))
        dr     = np.log1p(max(0.0, float(times[ci] - t0)))
        d      = float(depth[ci]) if depth[ci] >= 0 else 0.0
        is_root_reply = 1.0 if pi == 0 else 0.0
        srank  = float(sibling_rank[ci])
        ssize  = np.log1p(float(subtree_size[ci]))
        edge_features[(pi, ci)] = np.array([dt, dr, d, is_root_reply, srank, ssize], dtype=np.float32)

    return edge_features


def process_story(story_id, tree_path, source_text, label_int):
    """Convert one story into the dict that will be saved as .npz."""
    edges = parse_tree_file(tree_path)
    if not edges:
        return None

    node_order, uid_map, time_map, edge_list, node_content = build_graph(
        edges, source_text, story_id)

    n = len(node_order)
    if n < 2 or not edge_list:
        return None

    # Build COO edge matrix shape (2, E)
    edge_array = np.array(edge_list, dtype=np.int64).T  # (2, E)

    times = np.array([time_map.get(tid, 0.0) for tid in node_order], dtype=np.float32)

    # topindex: direct children of root (index 0)
    top_index = np.array([ci for pi, ci in edge_list if pi == 0], dtype=np.int64)

    # triIndex: nodes at depth <= 2
    depth = np.full(n, -1, dtype=np.int32)
    depth[0] = 0
    children_of = defaultdict(list)
    for pi, ci in edge_list:
        children_of[pi].append(ci)
    queue = [0]
    while queue:
        curr = queue.pop(0)
        for child in children_of[curr]:
            if depth[child] < 0:
                depth[child] = depth[curr] + 1
                queue.append(child)
    tri_index = np.array([i for i in range(n) if 0 <= depth[i] <= 2], dtype=np.int64)

    edge_features = compute_temporal_features(node_order, time_map, edge_list, story_id)
    # edge_feat_array: (E, 6) matching edge_list order
    edge_feat_array = np.array([edge_features.get((pi, ci), np.zeros(6))
                                 for pi, ci in edge_list], dtype=np.float32)

    # user features: 6-dim zero vector (real user features require Twitter API)
    # shape (N, 6) — filled with zeros since we don't have API access
    user_x = np.zeros((n, 6), dtype=np.float32)

    return {
        'edgematrix':   edge_array,           # (2, E)
        'edge_features': edge_feat_array,     # (E, 6) temporal features
        'root':         np.array([source_text], dtype=object),
        'nodecontent':  np.array(node_content, dtype=object),
        'timestamps':   times,                # (N,)
        'node_ids':     np.array(node_order, dtype=object),
        'user_ids':     np.array([uid_map.get(tid, '') for tid in node_order], dtype=object),
        'userx':        user_x,               # (N, 6)
        'y':            np.array(label_int, dtype=np.int64),
        'rootindex':    np.int64(0),
        'topindex':     top_index,
        'triIndex':     tri_index,
        'source_time':  np.float32(times[0]),  # for chronological splitting
    }


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def load_labels(label_path):
    label_map = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue
            label_str, sid = line.split(':', 1)
            label_map[sid.strip()] = LABEL_MAP.get(label_str.strip(), -1)
    return label_map


def load_source_tweets(source_path):
    source_map = {}
    with open(source_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' not in line:
                continue
            sid, text = line.split('\t', 1)
            source_map[sid.strip()] = text.strip()
    return source_map


def process_dataset(dataset_name, data_root, out_root):
    dataset_dir = os.path.join(data_root, dataset_name)
    tree_dir    = os.path.join(dataset_dir, 'tree')
    label_path  = os.path.join(dataset_dir, 'label.txt')
    source_path = os.path.join(dataset_dir, 'source_tweets.txt')

    out_npz_dir = os.path.join(out_root, dataset_name + '_npz')
    os.makedirs(out_npz_dir, exist_ok=True)

    labels  = load_labels(label_path)
    sources = load_source_tweets(source_path)

    story_ids = []
    skipped   = 0

    tree_files = sorted(os.listdir(tree_dir))
    print(f"Processing {len(tree_files)} stories for {dataset_name}...")

    for fname in tree_files:
        if not fname.endswith('.txt'):
            continue
        story_id = fname[:-4]
        label_int = labels.get(story_id, -1)
        if label_int < 0:
            skipped += 1
            continue
        source_text = sources.get(story_id, '')
        tree_path   = os.path.join(tree_dir, fname)

        result = process_story(story_id, tree_path, source_text, label_int)
        if result is None:
            skipped += 1
            continue

        np.savez(os.path.join(out_npz_dir, story_id + '.npz'), **result)
        story_ids.append(story_id)

    print(f"  Saved {len(story_ids)} stories, skipped {skipped}.")
    return story_ids


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def make_5fold_splits(story_ids, out_dir, seed=42):
    """Random 5-fold CV splits matching DUCK's protocol."""
    os.makedirs(out_dir, exist_ok=True)
    ids = story_ids.copy()
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    fold_size = n // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        end   = start + fold_size if i < 4 else n
        folds.append(ids[start:end])

    for fold_idx in range(5):
        fold_dir = os.path.join(out_dir, f'fold{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        test  = folds[fold_idx]
        train = []
        for j in range(5):
            if j != fold_idx:
                train.extend(folds[j])
        with open(os.path.join(fold_dir, '_x_train.pkl'), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(fold_dir, '_x_test.pkl'), 'wb') as f:
            pickle.dump(test, f)
        print(f"  Fold {fold_idx}: {len(train)} train, {len(test)} test")


def make_chrono_splits(story_ids, npz_dir, out_dir):
    """Chronological 60/20/20 split by source post timestamp."""
    os.makedirs(out_dir, exist_ok=True)

    timed = []
    for sid in story_ids:
        npz = np.load(os.path.join(npz_dir, sid + '.npz'), allow_pickle=True)
        source_time = float(npz['source_time'])
        timed.append((source_time, sid))

    timed.sort(key=lambda x: x[0])
    ordered_ids = [sid for _, sid in timed]

    n      = len(ordered_ids)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)
    train   = ordered_ids[:n_train]
    val     = ordered_ids[n_train:n_train + n_val]
    test    = ordered_ids[n_train + n_val:]

    with open(os.path.join(out_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(out_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(out_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test, f)

    print(f"  Chronological: {len(train)} train / {len(val)} val / {len(test)} test")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dataset_name, data_root='data', out_root='data'):
    story_ids = process_dataset(dataset_name, data_root, out_root)

    npz_dir = os.path.join(out_root, dataset_name + '_npz')

    print(f"Creating 5-fold splits for {dataset_name}...")
    fold_dir = os.path.join(out_root, dataset_name + '_5fold')
    make_5fold_splits(story_ids, fold_dir)

    print(f"Creating chronological splits for {dataset_name}...")
    chrono_dir = os.path.join(out_root, dataset_name + '_chrono')
    make_chrono_splits(story_ids, npz_dir, chrono_dir)

    print(f"Done: {dataset_name}")
    return story_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['twitter15', 'twitter16', 'all'],
                        default='all')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_root',  default='data')
    args = parser.parse_args()

    datasets = ['twitter15', 'twitter16'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        run(ds, args.data_root, args.out_root)

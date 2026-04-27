"""
Preprocessing pipeline for DUCK+ on the PHEME dataset.

Reads the PHEME all-rnr-annotated-threads directory and converts each thread
into a .npz file with the identical schema produced by preprocess.py for
Twitter15/16, so DuckPlusDataset and all training code work unchanged.

Label mapping (4-class, matching Twitter15/16):
  non-rumour              -> 0  (NR)
  rumour + misinformation -> 1  (FR)
  rumour + true           -> 2  (TR)
  rumour + unverified     -> 3  (UR)

Splits:
  - Leave-One-Event-Out (LOEO): standard for PHEME in the literature.
    Each of the 9 events takes a turn as the test set; the rest are train.
    Saved as data/pheme_loeo/event_<name>/{train.pkl, test.pkl}
  - Chronological 60/20/20: same as Twitter15/16 pipeline.
    Saved as data/pheme_chrono/{train.pkl, val.pkl, test.pkl}

Usage:
  python preprocess_pheme.py --pheme-root /path/to/all-rnr-annotated-threads
  python preprocess_pheme.py --pheme-root /path/to/all-rnr-annotated-threads --out-root data
"""

import os
import json
import pickle
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone


LABEL_MAP = {
    'nonrumour': 0,
    'false':     1,
    'true':      2,
    'unverified': 3,
}

TWITTER_EPOCH = datetime(2006, 3, 21, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_created_at(s):
    """Parse Twitter's created_at string -> UTC timestamp (seconds)."""
    try:
        dt = datetime.strptime(s, '%a %b %d %H:%M:%S +0000 %Y')
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def load_tweet_json(path):
    """Load a tweet JSON file. Returns dict or None."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def user_features(user):
    """
    Extract 6 user profile features matching data/README.txt:
      [followers, friends, ratio, history_tweets, reg_year, verified]
    """
    followers = float(user.get('followers_count', 0) or 0)
    friends   = float(user.get('friends_count', 0) or 0)
    ratio     = followers / max(1.0, friends)
    tweets    = float(user.get('statuses_count', 0) or 0)
    created   = user.get('created_at', '')
    reg_year  = 0.0
    if created:
        try:
            reg_year = float(datetime.strptime(
                created, '%a %b %d %H:%M:%S +0000 %Y').year)
        except Exception:
            pass
    verified = 1.0 if user.get('verified', False) else 0.0
    return [followers, friends, ratio, tweets, reg_year, verified]


def parse_structure(structure_dict, source_id):
    """
    Recursively walk structure.json nested dict and return list of
    (parent_id_str, child_id_str) edges.
    """
    edges = []

    def recurse(node_dict, parent_id):
        for child_id, grandchildren in node_dict.items():
            edges.append((parent_id, child_id))
            if isinstance(grandchildren, dict):
                recurse(grandchildren, child_id)

    # Top level: {source_id: {child: {...}, ...}}
    children = structure_dict.get(str(source_id), {})
    if isinstance(children, dict):
        recurse(children, str(source_id))

    return edges


# ---------------------------------------------------------------------------
# Graph construction (mirrors preprocess.py exactly)
# ---------------------------------------------------------------------------

def compute_temporal_features(node_order, time_map, edge_list):
    n = len(node_order)
    times = np.array([time_map.get(tid, 0.0) for tid in node_order],
                     dtype=np.float32)
    t0 = times[0]

    # Depth via BFS
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

    # Subtree sizes via reverse pass
    subtree_size = np.ones(n, dtype=np.int32)
    parent_of = {}
    for pi, ci in edge_list:
        parent_of[ci] = pi
    for ci in reversed(range(n)):
        if ci in parent_of:
            subtree_size[parent_of[ci]] += subtree_size[ci]

    # Sibling rank
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
        dt    = np.log1p(max(0.0, float(times[ci] - times[pi])))
        dr    = np.log1p(max(0.0, float(times[ci] - t0)))
        d     = float(depth[ci]) if depth[ci] >= 0 else 0.0
        is_root_reply = 1.0 if pi == 0 else 0.0
        srank = float(sibling_rank[ci])
        ssize = np.log1p(float(subtree_size[ci]))
        edge_features[(pi, ci)] = np.array(
            [dt, dr, d, is_root_reply, srank, ssize], dtype=np.float32)

    return edge_features


def process_thread(thread_dir, source_id, label_int):
    """
    Convert one PHEME thread directory into the .npz dict.
    Returns None if the thread is unusable (no edges, missing source, etc.).
    """
    # Load source tweet
    src_path = os.path.join(thread_dir, 'source-tweets', f'{source_id}.json')
    src_tweet = load_tweet_json(src_path)
    if src_tweet is None:
        return None

    src_text = src_tweet.get('text', '') or ''
    src_time_str = src_tweet.get('created_at', '')
    src_ts = parse_created_at(src_time_str)
    if src_ts is None:
        src_ts = float(source_id)  # fallback: snowflake ordering

    # Load structure.json for tree edges
    struct_path = os.path.join(thread_dir, 'structure.json')
    try:
        with open(struct_path, 'r', encoding='utf-8') as f:
            structure = json.load(f)
    except Exception:
        return None

    raw_edges = parse_structure(structure, source_id)
    if not raw_edges:
        return None

    # Load all reaction tweets into a dict: tweet_id -> tweet_json
    reactions_dir = os.path.join(thread_dir, 'reactions')
    reaction_tweets = {}
    if os.path.isdir(reactions_dir):
        for fname in os.listdir(reactions_dir):
            if not fname.endswith('.json'):
                continue
            tid = fname[:-5]
            tw = load_tweet_json(os.path.join(reactions_dir, fname))
            if tw:
                reaction_tweets[tid] = tw

    # Build time_map and uid_map for all nodes
    time_map = {str(source_id): 0.0}   # relative minutes from source
    uid_map  = {str(source_id): str(src_tweet.get('user', {}).get('id_str', ''))}
    tweet_map = {str(source_id): src_tweet}

    for tid, tw in reaction_tweets.items():
        ts = parse_created_at(tw.get('created_at', ''))
        if ts is not None:
            time_map[tid] = (ts - src_ts) / 60.0   # convert to minutes
        else:
            time_map[tid] = 0.0
        uid_map[tid] = str(tw.get('user', {}).get('id_str', ''))
        tweet_map[tid] = tw

    # Filter edges to nodes we actually have data for
    valid_nodes = set(time_map.keys())
    raw_edges = [(p, c) for p, c in raw_edges
                 if p in valid_nodes and c in valid_nodes]
    if not raw_edges:
        return None

    # BFS order from source
    parent_map = defaultdict(list)
    for p, c in raw_edges:
        parent_map[p].append(c)

    sid_str = str(source_id)
    node_order = [sid_str]
    visited = {sid_str}
    queue = [sid_str]
    while queue:
        curr = queue.pop(0)
        for child in sorted(parent_map.get(curr, []),
                            key=lambda t: time_map.get(t, 0.0)):
            if child not in visited:
                visited.add(child)
                node_order.append(child)
                queue.append(child)

    n = len(node_order)
    if n < 2:
        return None

    idx_map = {tid: i for i, tid in enumerate(node_order)}

    edge_list = []
    for p, c in raw_edges:
        pi = idx_map.get(p)
        ci = idx_map.get(c)
        if pi is not None and ci is not None and pi != ci:
            edge_list.append((pi, ci))

    if not edge_list:
        return None

    # Node content and user features
    node_content = []
    user_x = np.zeros((n, 6), dtype=np.float32)
    for i, tid in enumerate(node_order):
        tw = tweet_map.get(tid, {})
        node_content.append(tw.get('text', '') or '')
        u = tw.get('user', {})
        if u:
            user_x[i] = np.array(user_features(u), dtype=np.float32)

    times = np.array([time_map.get(tid, 0.0) for tid in node_order],
                     dtype=np.float32)

    edge_array = np.array(edge_list, dtype=np.int64).T   # (2, E)

    # topindex: direct children of root
    top_index = np.array([ci for pi, ci in edge_list if pi == 0],
                         dtype=np.int64)

    # triIndex: depth <= 2
    depth_arr = np.full(n, -1, dtype=np.int32)
    depth_arr[0] = 0
    children_of = defaultdict(list)
    for pi, ci in edge_list:
        children_of[pi].append(ci)
    queue = [0]
    while queue:
        curr = queue.pop(0)
        for child in children_of[curr]:
            if depth_arr[child] < 0:
                depth_arr[child] = depth_arr[curr] + 1
                queue.append(child)
    tri_index = np.array([i for i in range(n) if 0 <= depth_arr[i] <= 2],
                         dtype=np.int64)

    edge_features = compute_temporal_features(node_order, time_map, edge_list)
    edge_feat_array = np.array(
        [edge_features.get((pi, ci), np.zeros(6)) for pi, ci in edge_list],
        dtype=np.float32)

    return {
        'edgematrix':    edge_array,
        'edge_features': edge_feat_array,
        'root':          np.array([src_text], dtype=object),
        'nodecontent':   np.array(node_content, dtype=object),
        'timestamps':    times,
        'node_ids':      np.array(node_order, dtype=object),
        'user_ids':      np.array([uid_map.get(tid, '') for tid in node_order],
                                  dtype=object),
        'userx':         user_x,
        'y':             np.array(label_int, dtype=np.int64),
        'rootindex':     np.int64(0),
        'topindex':      top_index,
        'triIndex':      tri_index,
        'source_time':   np.float64(src_ts),
        'event':         np.array([os.path.basename(
                             os.path.dirname(os.path.dirname(thread_dir)))],
                             dtype=object),
    }


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------

def get_label(ann):
    """Map annotation.json -> 4-class int label. Returns -1 if unclear."""
    is_rumour = ann.get('is_rumour', '')
    if is_rumour == 'nonrumour':
        return LABEL_MAP['nonrumour']
    if is_rumour == 'rumour':
        true_val = str(ann.get('true', ''))
        misinfo  = str(ann.get('misinformation', ''))
        if true_val == '1':
            return LABEL_MAP['true']
        elif misinfo == '1':
            return LABEL_MAP['false']
        else:
            return LABEL_MAP['unverified']
    return -1   # 'unclear' or missing


def process_dataset(pheme_root, out_root):
    npz_dir = os.path.join(out_root, 'pheme_npz')
    os.makedirs(npz_dir, exist_ok=True)

    story_ids = []    # list of source_id strings
    story_events = {} # source_id -> event name
    skipped = 0

    events = sorted(e for e in os.listdir(pheme_root) if not e.startswith('.'))
    print(f"Found {len(events)} events.")

    for event in events:
        event_dir = os.path.join(pheme_root, event)
        event_label = event.replace('-all-rnr-threads', '')
        for split in ['rumours', 'non-rumours']:
            split_dir = os.path.join(event_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for source_id in os.listdir(split_dir):
                if source_id.startswith('.'):
                    continue
                thread_dir = os.path.join(split_dir, source_id)
                ann_path   = os.path.join(thread_dir, 'annotation.json')
                if not os.path.isfile(ann_path):
                    skipped += 1
                    continue
                with open(ann_path) as f:
                    ann = json.load(f)
                label_int = get_label(ann)
                if label_int < 0:
                    skipped += 1
                    continue

                result = process_thread(thread_dir, source_id, label_int)
                if result is None:
                    skipped += 1
                    continue

                np.savez(os.path.join(npz_dir, source_id + '.npz'), **result)
                story_ids.append(source_id)
                story_events[source_id] = event_label

    print(f"Saved {len(story_ids)} threads, skipped {skipped}.")
    return story_ids, story_events


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def make_loeo_splits(story_ids, story_events, out_dir):
    """
    Leave-One-Event-Out splits — standard evaluation protocol for PHEME.
    Each of the 9 events is held out as test once; the rest are train.
    """
    os.makedirs(out_dir, exist_ok=True)

    events = sorted(set(story_events.values()))
    for test_event in events:
        fold_dir = os.path.join(out_dir, f'event_{test_event}')
        os.makedirs(fold_dir, exist_ok=True)
        train = [sid for sid in story_ids if story_events[sid] != test_event]
        test  = [sid for sid in story_ids if story_events[sid] == test_event]
        with open(os.path.join(fold_dir, 'train.pkl'), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(fold_dir, 'test.pkl'), 'wb') as f:
            pickle.dump(test, f)
        print(f"  LOEO {test_event}: {len(train)} train, {len(test)} test")


def make_chrono_splits(story_ids, npz_dir, out_dir):
    """Chronological 60/20/20 split by source tweet timestamp."""
    os.makedirs(out_dir, exist_ok=True)

    timed = []
    for sid in story_ids:
        npz = np.load(os.path.join(npz_dir, sid + '.npz'), allow_pickle=True)
        timed.append((float(npz['source_time']), sid))
    timed.sort(key=lambda x: x[0])
    ordered = [sid for _, sid in timed]

    n       = len(ordered)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)
    train   = ordered[:n_train]
    val     = ordered[n_train:n_train + n_val]
    test    = ordered[n_train + n_val:]

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

def run(pheme_root, out_root='data'):
    print(f"Processing PHEME from: {pheme_root}")
    story_ids, story_events = process_dataset(pheme_root, out_root)

    npz_dir = os.path.join(out_root, 'pheme_npz')

    print("Creating LOEO splits...")
    make_loeo_splits(story_ids, story_events,
                     os.path.join(out_root, 'pheme_loeo'))

    print("Creating chronological splits...")
    make_chrono_splits(story_ids, npz_dir,
                       os.path.join(out_root, 'pheme_chrono'))

    print(f"Done. {len(story_ids)} threads ready in {npz_dir}")
    return story_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pheme-root', required=True,
                        help='Path to all-rnr-annotated-threads directory')
    parser.add_argument('--out-root', default='data',
                        help='Output directory (default: data)')
    args = parser.parse_args()
    run(args.pheme_root, args.out_root)

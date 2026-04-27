"""
Rehydrate tweet text and user profile features for Twitter15/Twitter16.

Scans all tree files, collects every tweet ID and user ID, then fetches
them from the Twitter v2 API in batches of 100.

Outputs (written incrementally, safe to interrupt and resume):
  data/tweet_texts.json    — {tweet_id: text}
  data/user_features.json  — {user_id: [followers, friends, ratio, tweets, reg_year, verified]}

Usage:
  export TWITTER_BEARER_TOKEN=<your token>
  python tools/rehydrate.py
  python tools/rehydrate.py --datasets twitter15        # single dataset
  python tools/rehydrate.py --rate-limit 15             # requests per 15 min window
"""

import os
import ast
import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
import urllib.error
from collections import defaultdict


DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

# Output paths
TWEET_TEXT_PATH   = os.path.join(DATA_ROOT, 'tweet_texts.json')
USER_FEAT_PATH    = os.path.join(DATA_ROOT, 'user_features.json')

# Twitter v2 endpoint
TWEETS_URL = 'https://api.twitter.com/2/tweets'
USERS_URL  = 'https://api.twitter.com/2/users'

TWEET_FIELDS = 'id,text,author_id'
USER_FIELDS  = 'id,public_metrics,created_at,verified'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def bearer_headers(token):
    return {'Authorization': f'Bearer {token}'}


def api_get(url, params, headers):
    """Make a GET request, return parsed JSON or raise on HTTP error."""
    query = '&'.join(f'{k}={urllib.parse.quote(str(v))}' for k, v in params.items())
    full_url = f'{url}?{query}'
    req = urllib.request.Request(full_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {e.code}: {body}') from e


# ---------------------------------------------------------------------------
# ID collection
# ---------------------------------------------------------------------------

def collect_ids(datasets):
    """Walk tree files and return (tweet_ids set, user_ids set)."""
    tweet_ids = set()
    user_ids  = set()

    for ds in datasets:
        tree_dir = os.path.join(DATA_ROOT, ds, 'tree')
        if not os.path.isdir(tree_dir):
            print(f'  Warning: {tree_dir} not found, skipping.')
            continue
        for fname in os.listdir(tree_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(tree_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '->' not in line:
                        continue
                    parts = line.split('->')
                    if len(parts) != 2:
                        continue
                    try:
                        parent = ast.literal_eval(parts[0].strip())
                        child  = ast.literal_eval(parts[1].strip())
                    except Exception:
                        continue
                    for node in (parent, child):
                        uid, tid, _ = node
                        if tid != 'ROOT':
                            tweet_ids.add(str(tid))
                        if uid not in ('ROOT', ''):
                            user_ids.add(str(uid))

    return tweet_ids, user_ids


# ---------------------------------------------------------------------------
# Fetch tweets
# ---------------------------------------------------------------------------

def fetch_tweets(tweet_ids, existing, token, rate_limit, dry_run=False):
    """Fetch text for tweet IDs not already in `existing`. Updates existing in place."""

    needed = [tid for tid in tweet_ids if tid not in existing]
    if not needed:
        print('  All tweet texts already fetched.')
        return

    print(f'  Fetching text for {len(needed)} tweets ({len(existing)} already cached)...')
    headers = bearer_headers(token)
    batch_size = 100
    window_requests = 0
    window_start = time.time()

    for i in range(0, len(needed), batch_size):
        batch = needed[i:i + batch_size]

        if dry_run:
            for tid in batch:
                existing[tid] = '[dry-run]'
            print(f'  [dry-run] batch {i // batch_size + 1}: {len(batch)} IDs')
            continue

        # Rate-limit: sleep if we've used the window allowance
        if window_requests >= rate_limit:
            elapsed = time.time() - window_start
            sleep_for = max(0, 900 - elapsed) + 5  # 15-min window + buffer
            print(f'  Rate limit reached. Sleeping {sleep_for:.0f}s...', flush=True)
            time.sleep(sleep_for)
            window_requests = 0
            window_start = time.time()

        params = {
            'ids': ','.join(batch),
            'tweet.fields': TWEET_FIELDS,
        }
        try:
            data = api_get(TWEETS_URL, params, headers)
        except RuntimeError as e:
            print(f'  Error on batch {i // batch_size + 1}: {e}')
            print('  Saving progress and stopping.')
            save_json(TWEET_TEXT_PATH, existing)
            sys.exit(1)

        for tweet in data.get('data', []):
            existing[str(tweet['id'])] = tweet.get('text', '')

        # Tweets the API couldn't find (deleted/private) — store empty string
        found_ids = {str(t['id']) for t in data.get('data', [])}
        for tid in batch:
            if tid not in found_ids:
                existing[tid] = ''

        window_requests += 1
        batch_num = i // batch_size + 1
        total_batches = (len(needed) + batch_size - 1) // batch_size
        print(f'  Tweets: batch {batch_num}/{total_batches} done ({len(existing)} cached)', flush=True)

        save_json(TWEET_TEXT_PATH, existing)


# ---------------------------------------------------------------------------
# Fetch users
# ---------------------------------------------------------------------------

def user_features_from_obj(user):
    """
    Extract the 6 user features defined in data/README.txt:
      [followers, friends, ratio, history_tweets, reg_year, verified]
    """
    metrics  = user.get('public_metrics', {})
    followers = int(metrics.get('followers_count', 0))
    friends   = int(metrics.get('following_count', 0))
    ratio     = followers / max(1, friends)
    tweets    = int(metrics.get('tweet_count', 0))

    created_at = user.get('created_at', '')
    reg_year = 0
    if created_at:
        try:
            reg_year = int(created_at[:4])
        except ValueError:
            pass

    verified = 1.0 if user.get('verified', False) else 0.0

    return [float(followers), float(friends), float(ratio),
            float(tweets), float(reg_year), verified]


def fetch_users(user_ids, existing, token, rate_limit, dry_run=False):
    """Fetch user profile features for user IDs not already in `existing`."""

    needed = [uid for uid in user_ids if uid not in existing]
    if not needed:
        print('  All user features already fetched.')
        return

    print(f'  Fetching features for {len(needed)} users ({len(existing)} already cached)...')
    headers = bearer_headers(token)
    batch_size = 100
    window_requests = 0
    window_start = time.time()

    for i in range(0, len(needed), batch_size):
        batch = needed[i:i + batch_size]

        if dry_run:
            for uid in batch:
                existing[uid] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            print(f'  [dry-run] batch {i // batch_size + 1}: {len(batch)} IDs')
            continue

        if window_requests >= rate_limit:
            elapsed = time.time() - window_start
            sleep_for = max(0, 900 - elapsed) + 5
            print(f'  Rate limit reached. Sleeping {sleep_for:.0f}s...', flush=True)
            time.sleep(sleep_for)
            window_requests = 0
            window_start = time.time()

        params = {
            'ids': ','.join(batch),
            'user.fields': USER_FIELDS,
        }
        try:
            data = api_get(USERS_URL, params, headers)
        except RuntimeError as e:
            print(f'  Error on batch {i // batch_size + 1}: {e}')
            print('  Saving progress and stopping.')
            save_json(USER_FEAT_PATH, existing)
            sys.exit(1)

        for user in data.get('data', []):
            existing[str(user['id'])] = user_features_from_obj(user)

        found_ids = {str(u['id']) for u in data.get('data', [])}
        for uid in batch:
            if uid not in found_ids:
                existing[uid] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        window_requests += 1
        batch_num = i // batch_size + 1
        total_batches = (len(needed) + batch_size - 1) // batch_size
        print(f'  Users: batch {batch_num}/{total_batches} done ({len(existing)} cached)', flush=True)

        save_json(USER_FEAT_PATH, existing)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Rehydrate Twitter15/16 tweet text and user features.')
    parser.add_argument('--datasets', nargs='+', default=['twitter15', 'twitter16'],
                        help='Dataset folders to scan (default: both)')
    parser.add_argument('--rate-limit', type=int, default=15,
                        help='Max API requests per 15-minute window (default: 15 for Basic tier)')
    parser.add_argument('--tweets-only', action='store_true',
                        help='Only fetch tweet text, skip user features')
    parser.add_argument('--users-only', action='store_true',
                        help='Only fetch user features, skip tweet text')
    parser.add_argument('--dry-run', action='store_true',
                        help='Walk all IDs and log batches without making API calls')
    args = parser.parse_args()

    token = os.environ.get('TWITTER_BEARER_TOKEN', '')
    if not token and not args.dry_run:
        print('Error: set TWITTER_BEARER_TOKEN environment variable.')
        print('  export TWITTER_BEARER_TOKEN=<your bearer token>')
        sys.exit(1)

    print(f'Scanning tree files for {args.datasets}...')
    tweet_ids, user_ids = collect_ids(args.datasets)
    print(f'  Found {len(tweet_ids)} unique tweet IDs, {len(user_ids)} unique user IDs.')

    if not args.users_only:
        tweet_cache = load_json(TWEET_TEXT_PATH)
        fetch_tweets(tweet_ids, tweet_cache, token, args.rate_limit, dry_run=args.dry_run)
        print(f'  Tweet texts saved to {TWEET_TEXT_PATH}')

    if not args.tweets_only:
        user_cache = load_json(USER_FEAT_PATH)
        fetch_users(user_ids, user_cache, token, args.rate_limit, dry_run=args.dry_run)
        print(f'  User features saved to {USER_FEAT_PATH}')

    print('Done.')


if __name__ == '__main__':
    main()

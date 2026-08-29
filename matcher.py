#!/usr/bin/env python3
"""Match tpdb_all_performers.json entries to local performer ids via the
stashface face-matching pipeline.

For every entry: downloads its image, runs it through the pipeline, and —
if the result clears the 55% cosine threshold — records
{"local_id": ..., "tpdb_id": ...} into matched_identities.json.

Built for large inputs (100k+ items):
  - Processes in batches; saves matched_identities.json after every batch,
    so a crash never loses matches already found.
  - Writes are atomic (temp file + rename), so the output file is never
    left half-written even if the process dies mid-save.
  - Only ever holds ONE downloaded image in memory at a time — the 114k
    input entries themselves (just id/name/url strings) are small enough
    to load in full, but their images are never all held at once.
  - Re-running after a crash/stop skips tpdb_ids that are already in
    matched_identities.json, instead of reprocessing them.
  - A failed download is retried up to 4 times (with a short backoff)
    before that entry is given up on — a blip in the network doesn't cost
    it a match.
  - An image that downloads fine but has no detectable face just counts as
    "no match" for that entry — the run keeps going.
  - Any other unexpected error on one entry is caught, logged, and
    skipped — nothing stops the whole run.

Usage:
    python matcher.py                 # process the full dataset
    python matcher.py --limit 30      # only process the first 30 entries, for testing
    python matcher.py --batch-size 100
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

import requests

from stashface_adapter import stashface_pipeline

INPUT_FILE = "tpdb_all_performers.json"
OUTPUT_FILE = "matched_identities.json"
COSINE_THRESHOLD = 55.0
DOWNLOAD_TIMEOUT = 15  # seconds
DEFAULT_BATCH_SIZE = 50
MAX_DOWNLOAD_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2  # 2s, 4s, 8s between attempts


def load_input(path: str, limit: Optional[int]) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def load_existing_matches(path: str):
    """Load whatever matches are already on disk, so re-running after a
    crash/stop doesn't redo work that already succeeded."""
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], set()
    done_ids = {m["tpdb_id"] for m in existing if "tpdb_id" in m}
    return existing, done_ids


def save_matches_atomic(path: str, matches: list):
    """Write to a temp file then atomically replace the real file, so
    matched_identities.json is never left half-written."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def download_image(url: str, retries: int = MAX_DOWNLOAD_RETRIES) -> Optional[bytes]:
    """Try to download the image up to `retries` times before giving up.
    A transient network blip shouldn't permanently cost this entry a match."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            last_error = e
            if attempt < retries:
                wait = RETRY_BACKOFF_SECONDS * attempt  # 2s, 4s, 6s...
                print(f"  [download attempt {attempt}/{retries} failed] {url} -> {e} (retrying in {wait}s)",
                      file=sys.stderr)
                time.sleep(wait)
    print(f"  [download failed after {retries} attempts] {url} -> {last_error}", file=sys.stderr)
    return None


def process_entry(entry: dict) -> Optional[dict]:
    """Download one entry's image, run the pipeline, and return a
    matched_identities.json record — or None if there's no confident match."""
    tpdb_id = entry.get("tpdb_id")
    image_url = entry.get("image")
    if not tpdb_id or not image_url:
        return None

    image_bytes = download_image(image_url)
    if image_bytes is None:
        return None

    try:
        result = stashface_pipeline(image_bytes)
    except Exception as e:
        print(f"  [pipeline failed] {tpdb_id} -> {e}", file=sys.stderr)
        return None
    finally:
        del image_bytes  # don't hold it any longer than needed

    if not result:
        return None

    cosine_score = result.get("cosine_score", 0)
    if cosine_score < COSINE_THRESHOLD:
        return None

    return {"local_id": result["matched_id"], "tpdb_id": tpdb_id, "cosine_score": cosine_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N entries (default: all)")
    parser.add_argument("--input", default=INPUT_FILE, help=f"Input file (default: {INPUT_FILE})")
    parser.add_argument("--output", default=OUTPUT_FILE, help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Save progress every N entries (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    entries = load_input(args.input, args.limit)
    matches, done_ids = load_existing_matches(args.output)

    total = len(entries)
    print(f"Loaded {total} entries" + (f" (--limit {args.limit})" if args.limit else "") +
          f"; {len(done_ids)} already matched in a previous run.")

    processed_since_save = 0
    for i, entry in enumerate(entries, start=1):
        tpdb_id = entry.get("tpdb_id")
        if tpdb_id in done_ids:
            continue  # already matched previously, skip re-downloading/re-matching

        try:
            match = process_entry(entry)
        except Exception as e:
            # Last-resort safety net: no single bad entry should ever kill
            # a run that's already hours into 114k items.
            print(f"[{i}/{total}] [unexpected error, skipping]  {tpdb_id} -> {e}", file=sys.stderr)
            match = None
        if match:
            matches.append(match)
            done_ids.add(tpdb_id)
            print(f"[{i}/{total}] MATCH     {tpdb_id} -> {match['local_id']}")
        else:
            print(f"[{i}/{total}] no match  {tpdb_id}")

        processed_since_save += 1
        if processed_since_save >= args.batch_size:
            save_matches_atomic(args.output, matches)
            processed_since_save = 0
            print(f"  -- progress saved: {len(matches)} matches so far --")

    save_matches_atomic(args.output, matches)  # final save for the last partial batch
    print(f"Done. {len(matches)} confirmed matches written to {args.output}.")


if __name__ == "__main__":
    main()

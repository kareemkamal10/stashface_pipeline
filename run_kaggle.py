#!/usr/bin/env python3
"""Multi-GPU (2x T4) batched version of matcher.py, built for Kaggle.

Batch-sequential design (each --batch-size chunk goes through two clear
phases, one after the other, with its own progress bar):
  1. Download phase: a pool of download threads fetches every image in the
     current batch (retrying failures up to 4 times), saving each one to
     disk as data/_download_cache/{tpdb_id}.{ext}. A tqdm progress bar
     tracks this phase; the next phase only starts once the whole batch
     has finished downloading.
  2. Match phase: once the batch is fully downloaded, two GPU worker
     threads (one for device 0, one for device 1) pull from that batch's
     queue independently and run detection + matching, each on its own
     model instances pinned to its own GPU (see models/gpu_worker.py). A
     second tqdm progress bar tracks this phase.
  3. matched_identities.json is saved atomically at the end of each batch,
     and a re-run after a crash skips tpdb_ids already recorded there —
     so it resumes at the next un-started batch, not from scratch.

Usage:
    python run_kaggle.py                    # full dataset, 2 GPUs
    python run_kaggle.py --limit 200         # quick test
    python run_kaggle.py --batch-size 10000  # smaller batches
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional

import requests
from tqdm import tqdm

from models.data_manager import DataManager
from models.gpu_worker import FaceMatcher
from models.paths import DATA_DIR

INPUT_FILE = "tpdb_all_performers.json"
OUTPUT_FILE = "matched_identities.json"
COSINE_THRESHOLD = 55.0
DOWNLOAD_TIMEOUT = 15
MAX_DOWNLOAD_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2
DOWNLOAD_WORKERS = 16
DEFAULT_BATCH_SIZE = 15000
GPU_DEVICE_IDS = [0, 1]


# --- input/output helpers (same contract as matcher.py) --------------------

def load_input(path: str, limit: Optional[int]) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def load_existing_matches(path: str):
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
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --- download stage ----------------------------------------------------

def _guess_extension(url: str) -> str:
    ext = Path(url.split("?")[0]).suffix
    if not ext or len(ext) > 5:
        return ".jpg"
    return ext


def download_one(entry: dict, out_dir: Path) -> Optional[tuple]:
    """Download one entry's image with retries, save it as {tpdb_id}{ext}.
    Returns (tpdb_id, local_path) on success, None if it never succeeded."""
    tpdb_id = entry.get("tpdb_id")
    url = entry.get("image")
    if not tpdb_id or not url:
        return None

    dest = out_dir / f"{tpdb_id}{_guess_extension(url)}"
    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return tpdb_id, dest
        except (requests.RequestException, OSError) as e:
            last_error = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    tqdm.write(f"  [download failed after {MAX_DOWNLOAD_RETRIES} attempts] {tpdb_id} -> {last_error}")
    return None


def download_batch(batch: list, download_dir: Path, download_workers: int) -> list:
    """Downloads every entry in this batch, blocking until all of them have
    been attempted. Returns the list of (tpdb_id, local_path) that
    succeeded. Shows one tqdm bar for the whole batch."""
    results = []
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = [pool.submit(download_one, entry, download_dir) for entry in batch]
        for fut in tqdm(futures, desc="  downloading batch", unit="img", leave=False):
            result = fut.result()
            if result:
                results.append(result)
    return results


# --- GPU matching stage --------------------------------------------------

def gpu_worker_loop(matcher: FaceMatcher, work_queue: Queue,
                     state: dict, state_lock: threading.Lock, pbar: tqdm):
    while True:
        item = work_queue.get()
        if item is None:  # sentinel: no more work coming
            work_queue.task_done()
            break

        tpdb_id, path = item
        try:
            result = matcher.match(str(path))
        except Exception as e:
            tqdm.write(f"  [match failed] {tpdb_id} -> {e}")
            result = None
        finally:
            try:
                os.remove(path)  # free disk space immediately, don't wait for batch end
            except OSError:
                pass

        record = None
        if result and result.get("cosine_score", 0) >= COSINE_THRESHOLD:
            record = {"local_id": result["matched_id"], "tpdb_id": tpdb_id, "cosine_score": result["cosine_score"]}

        with state_lock:
            if record:
                state["matches"].append(record)
                state["done_ids"].add(tpdb_id)
            pbar.update(1)

        work_queue.task_done()


def match_batch(downloaded: list, matchers: list, gpu_ids: list, state: dict, state_lock: threading.Lock):
    """Matches every downloaded image in this batch, splitting the work
    across GPU worker threads that pull independently from a shared queue
    (whichever GPU frees up first grabs the next image — not a fixed
    50/50 split). Blocks until the whole batch is matched. Shows one tqdm
    bar for the whole batch."""
    work_queue: Queue = Queue()
    for item in downloaded:
        work_queue.put(item)
    for _ in matchers:
        work_queue.put(None)

    with tqdm(total=len(downloaded), desc="  matching batch", unit="img", leave=False) as pbar:
        threads = [
            threading.Thread(target=gpu_worker_loop, args=(m, work_queue, state, state_lock, pbar), daemon=True)
            for m in matchers
        ]
        for t in threads:
            t.start()
        work_queue.join()
        for t in threads:
            t.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N entries (default: all)")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Entries per download-then-match batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--download-workers", type=int, default=DOWNLOAD_WORKERS)
    parser.add_argument("--gpus", type=int, nargs="+", default=GPU_DEVICE_IDS,
                         help="GPU device ids to use, e.g. --gpus 0 1 (default: 0 1). "
                              "Use --gpus (with no numbers) or a single CPU-only machine to fall back to CPU.")
    args = parser.parse_args()

    entries = load_input(args.input, args.limit)
    matches, done_ids = load_existing_matches(args.output)
    todo = [e for e in entries if e.get("tpdb_id") not in done_ids]

    print(f"Loaded {len(entries)} entries" + (f" (--limit {args.limit})" if args.limit else "") +
          f"; {len(done_ids)} already matched previously; {len(todo)} left to process.")

    if not todo:
        print("Nothing to do.")
        return

    # Downloaded images live under --data-dir (which defaults to /kaggle/temp
    # on Kaggle — see models/paths.py) rather than a fixed relative path, so
    # they land on the same big, always-clean disk as everything else. Each
    # image is removed right after it's matched, but wipe any leftovers from
    # a previous crashed run too, just in case.
    download_dir = Path(args.data_dir) / "_download_cache"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = args.gpus if args.gpus else [None]  # empty --gpus -> single CPU worker

    print("Opening performers.zvec (shared across all GPU workers) ...")
    shared_data_manager = DataManager(collection_path=str(Path(args.data_dir) / "performers.zvec"))
    if shared_data_manager.collection is None:
        print(f"ERROR: could not open performers.zvec under {args.data_dir} — "
              f"did setup.py finish syncing the bucket?", file=sys.stderr)
        sys.exit(1)

    matchers = [FaceMatcher(data_dir=args.data_dir, device_id=g, data_manager=shared_data_manager) for g in gpu_ids]

    state = {"matches": matches, "done_ids": done_ids}
    state_lock = threading.Lock()

    batches = list(chunked(todo, args.batch_size))
    print(f"Starting: {len(batches)} batch(es) of up to {args.batch_size} entries each, "
          f"{args.download_workers} download workers, {len(matchers)} GPU worker(s).")

    for batch_num, batch in enumerate(batches, start=1):
        print(f"Batch {batch_num}/{len(batches)} ({len(batch)} entries):")
        downloaded = download_batch(batch, download_dir, args.download_workers)
        failed = len(batch) - len(downloaded)
        if failed:
            print(f"  downloaded {len(downloaded)}/{len(batch)} ({failed} failed after retries)")

        match_batch(downloaded, matchers, gpu_ids, state, state_lock)

        with state_lock:
            save_matches_atomic(args.output, state["matches"])
            total_matches = len(state["matches"])
        print(f"  batch {batch_num} done — {total_matches} matches total so far. [saved]")

    print(f"Done. {len(state['matches'])} confirmed matches written to {args.output}.")


if __name__ == "__main__":
    main()

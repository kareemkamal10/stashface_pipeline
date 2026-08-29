#!/usr/bin/env python3
"""Multi-GPU (2x T4) batched version of matcher.py, built for Kaggle.

How it avoids a bottleneck:
  - A pool of download threads keeps fetching images (retrying failures up
    to 4 times) and saving each one to disk as data/_download_cache/{tpdb_id}.{ext}.
  - Downloaded images are handed off through a queue whose capacity equals
    --batch-size. Once that queue is full, new downloads simply pause until
    a GPU worker frees up a slot — so at most ~one batch's worth of images
    is ever sitting on disk, and the next batch keeps downloading in the
    background the entire time the current batch is being matched. There's
    no explicit "wait for batch 1, then start batch 2" step — it's a
    continuous pipeline, which keeps both stages busy at all times instead
    of alternating between them.
  - Two GPU worker threads (one for device 0, one for device 1) pull
    from that queue independently and run detection + matching, each on
    its own model instances pinned to its own GPU (see models/gpu_worker.py).
  - matched_identities.json is saved atomically every --batch-size results,
    and a re-run after a crash skips tpdb_ids already recorded there.

Usage:
    python run_kaggle.py                    # full dataset, 2 GPUs
    python run_kaggle.py --limit 200         # quick test
    python run_kaggle.py --batch-size 10000  # smaller queue / save interval
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
# How often to print a progress line, independent of --batch-size (which
# also controls how often matched_identities.json is saved to disk).
# Printing one line per image (114k+ lines) is what floods the notebook
# cell and hangs the browser tab, so this is decoupled and kept small.
PROGRESS_PRINT_INTERVAL = 1000
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
    print(f"  [download failed after {MAX_DOWNLOAD_RETRIES} attempts] {tpdb_id} -> {last_error}", file=sys.stderr)
    return None


# --- GPU matching stage --------------------------------------------------

def gpu_worker_loop(matcher: FaceMatcher, worker_label: str, work_queue: Queue,
                     state: dict, state_lock: threading.Lock, output_path: str, batch_size: int):
    while True:
        item = work_queue.get()
        if item is None:  # sentinel: no more work coming
            work_queue.task_done()
            break

        tpdb_id, path = item
        try:
            result = matcher.match(str(path))
        except Exception as e:
            print(f"  [{worker_label}] [match failed] {tpdb_id} -> {e}", file=sys.stderr)
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
                state["batch_match_count"] = state.get("batch_match_count", 0) + 1

            state["processed_total"] = state.get("processed_total", 0) + 1
            state["processed_since_save"] += 1
            state["processed_since_print"] = state.get("processed_since_print", 0) + 1

            do_save = state["processed_since_save"] >= batch_size
            if do_save:
                save_matches_atomic(output_path, state["matches"])
                state["processed_since_save"] = 0

            # No per-image print here on purpose — printing one line per image
            # (114k+ lines total) floods the notebook cell output and hangs the
            # browser tab long before the pipeline itself is anywhere near done.
            # A compact line every PROGRESS_PRINT_INTERVAL images instead
            # (always also printed right after a save, so "saved" state stays visible).
            if state["processed_since_print"] >= PROGRESS_PRINT_INTERVAL or do_save:
                batch_matches = state.get("batch_match_count", 0)
                tag = " [saved]" if do_save else ""
                print(
                    f"  -- progress: {state['processed_total']} processed, "
                    f"{len(state['matches'])} matches total "
                    f"(+{batch_matches} recently){tag} --"
                )
                state["processed_since_print"] = 0
                state["batch_match_count"] = 0

        work_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N entries (default: all)")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help=f"Download queue capacity + save interval (default: {DEFAULT_BATCH_SIZE})")
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
    # image is removed right after it's matched (see gpu_worker_loop), but
    # wipe any leftovers from a previous crashed run too, just in case.
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

    work_queue: Queue = Queue(maxsize=args.batch_size)
    state = {
        "matches": matches, "done_ids": done_ids,
        "processed_since_save": 0, "processed_since_print": 0,
        "processed_total": 0, "batch_match_count": 0,
    }
    state_lock = threading.Lock()

    gpu_threads = [
        threading.Thread(
            target=gpu_worker_loop,
            args=(m, f"gpu{gpu_ids[i]}" if gpu_ids[i] is not None else "cpu", work_queue,
                  state, state_lock, args.output, args.batch_size),
            daemon=True,
        )
        for i, m in enumerate(matchers)
    ]
    for t in gpu_threads:
        t.start()

    def enqueue_download(entry):
        result = download_one(entry, download_dir)
        if result:
            work_queue.put(result)  # blocks here once the queue is full — this IS the backpressure

    print(f"Starting download pool ({args.download_workers} workers) and {len(gpu_threads)} GPU worker(s) ...")
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        list(pool.map(enqueue_download, todo))

    # all downloads attempted; tell each GPU worker to stop once the queue drains
    for _ in gpu_threads:
        work_queue.put(None)
    work_queue.join()
    for t in gpu_threads:
        t.join()

    with state_lock:
        save_matches_atomic(args.output, state["matches"])
        total_matches = len(state["matches"])

    print(f"Done. {total_matches} confirmed matches written to {args.output}.")


if __name__ == "__main__":
    main()

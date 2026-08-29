#!/usr/bin/env python3
"""Install dependencies and sync the cc1234/stashface-data bucket to disk.

This is the "just make it work" setup step: run it once and search.py has
everything it needs (the performer index + the AdaFace ONNX model).

By default this ALWAYS starts from a completely clean slate: it deletes any
existing data-dir and clears the local huggingface_hub cache before syncing,
then downloads a fresh copy of the bucket. This is deliberate — a previous
run that got interrupted (Restart, a closed tab, a Kaggle session dying)
can leave partially-written files behind, and reusing those instead of
downloading fresh copies is exactly what produces RocksDB "Corruption"
errors. Re-syncing from scratch every time is slower but never flaky.

The data also goes to /kaggle/temp by default when running on Kaggle (see
models/paths.py) instead of the project folder under /kaggle/working:
/kaggle/temp has more free space, and is guaranteed to be wiped whenever
the session stops, so corrupted leftovers can never quietly survive into
the next run.

Usage:
    python setup.py                          # clean sync -> DATA_DIR (see models/paths.py)
    python setup.py --data-dir /mnt/data      # custom location
    python setup.py --skip-install            # only sync the bucket
    python setup.py --keep-existing           # DON'T wipe first — reuse whatever's
                                               # already on disk / cached (not
                                               # recommended; only for debugging)

Requires an HF token with read access to the bucket if it's private:
    hf auth login
    # or: export HF_TOKEN=hf_xxx
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from models.paths import DATA_DIR

DEFAULT_BUCKET = "cc1234/stashface-data"

# Versions before 1.13.0 have an interactive "update now? [Y/n]" prompt at
# startup that hangs forever in a notebook (no stdin to answer it) — pin to
# a version that dropped it, and also disable the update check as a
# second safety net in case an even older version is already installed.
MIN_HF_HUB_VERSION = "1.13.0"
_ENV = {**os.environ, "HF_HUB_DISABLE_UPDATE_CHECK": "1"}

# huggingface_hub's own download cache (separate from --data-dir). If a
# previous sync left a corrupted file in here, re-syncing to a fresh
# --data-dir won't help — hf would just hand back the same broken file
# from cache. Wiped by default for the same reason --data-dir is wiped.
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


def install_requirements(req_file: str = "requirements.txt"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True, env=_ENV)
    # `hf` CLI + bucket support ships with huggingface_hub[cli]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", f"huggingface_hub[cli]>={MIN_HF_HUB_VERSION}"],
        check=True,
        env=_ENV,
    )


def wipe_everything(data_dir: Path):
    if data_dir.exists():
        print(f"Removing existing '{data_dir}/' ...")
        shutil.rmtree(data_dir)
    if HF_CACHE_DIR.exists():
        print(f"Clearing huggingface_hub cache at '{HF_CACHE_DIR}/' ...")
        shutil.rmtree(HF_CACHE_DIR)


def sync_bucket(data_dir: Path, bucket_id: str):
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "buckets", "sync", f"hf://buckets/{bucket_id}", str(data_dir)],
        check=True,
        env=_ENV,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                         help=f"Where to sync the bucket to (default: {DATA_DIR})")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"Bucket id (default: {DEFAULT_BUCKET})")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install, only sync the bucket")
    parser.add_argument("--keep-existing", action="store_true",
                         help="Don't wipe the data-dir / HF cache first — reuse whatever's already there. "
                              "Not recommended: this is how corrupted leftovers from an interrupted run "
                              "end up getting reused instead of re-downloaded.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not args.keep_existing:
        wipe_everything(data_dir)

    if not args.skip_install:
        install_requirements()

    sync_bucket(data_dir, args.bucket)
    print(f"Done. Bucket '{args.bucket}' synced into '{data_dir}/'.")


if __name__ == "__main__":
    main()

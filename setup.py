#!/usr/bin/env python3
"""Install dependencies and sync the cc1234/stashface-data bucket to disk.

This is the "just make it work" setup step: run it once and search.py has
everything it needs (the performer index + the AdaFace ONNX model).

Usage:
    python setup.py                          # -> ./data
    python setup.py --data-dir /mnt/data      # custom location
    python setup.py --skip-install            # only sync the bucket

Requires an HF token with read access to the bucket if it's private:
    hf auth login
    # or: export HF_TOKEN=hf_xxx
"""
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "cc1234/stashface-data"


def install_requirements(req_file: str = "requirements.txt"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)
    # `hf` CLI + bucket support ships with huggingface_hub[cli]
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"], check=True)


def sync_bucket(data_dir: Path, bucket_id: str):
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "buckets", "sync", f"hf://buckets/{bucket_id}", str(data_dir)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="Where to sync the bucket to (default: ./data)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"Bucket id (default: {DEFAULT_BUCKET})")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install, only sync the bucket")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements()

    sync_bucket(Path(args.data_dir), args.bucket)
    print(f"Done. Bucket '{args.bucket}' synced into '{args.data_dir}/'.")


if __name__ == "__main__":
    main()

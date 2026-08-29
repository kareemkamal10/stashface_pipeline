"""Central place for where the stashface-data bucket lives on disk.

The original Space (cc1234/stashface_onnx) mounts the `cc1234/stashface-data`
bucket directly as a volume, so DATA_DIR just points at wherever that's
mounted. Here (running outside of a Space) we don't have that native mount,
so `setup.py` downloads the same bucket into a local folder instead, and
DATA_DIR points at that folder.

On Kaggle specifically, we default that folder to /kaggle/temp instead of
the project directory under /kaggle/working:
  - /kaggle/temp has much more free space than /kaggle/working, which is
    what was causing RocksDB "Corruption" errors mid-sync (the bucket ran
    out of room partway through and left partial/corrupted files behind).
  - /kaggle/temp is guaranteed to be wiped whenever the session stops, so a
    corrupted or partial previous sync can never silently linger and get
    reused by a later run — every run starts from a genuinely clean disk.

Override with the STASHFACE_DATA_DIR env var if you want the data somewhere
else, e.g.:

    export STASHFACE_DATA_DIR=/mnt/stashface-data
"""

import os
from pathlib import Path


def _default_data_dir() -> Path:
    kaggle_temp = Path("/kaggle/temp")
    if kaggle_temp.is_dir():
        return kaggle_temp / "stashface-data"
    # Not on Kaggle (or /kaggle/temp isn't available) — fall back to
    # <project_root>/data, same as before.
    return Path(__file__).resolve().parent.parent / "data"


DATA_DIR = Path(os.environ.get("STASHFACE_DATA_DIR", str(_default_data_dir())))

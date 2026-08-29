"""Central place for where the stashface-data bucket lives on disk.

The original Space (cc1234/stashface_onnx) mounts the `cc1234/stashface-data`
bucket directly as a volume, so DATA_DIR just points at wherever that's
mounted. Here (running outside of a Space) we don't have that native mount,
so `setup.py` downloads the same bucket into a local folder instead, and
DATA_DIR points at that folder.

Override with the STASHFACE_DATA_DIR env var if you want the data somewhere
else, e.g.:

    export STASHFACE_DATA_DIR=/mnt/stashface-data
"""

import os
from pathlib import Path

# Defaults to <project_root>/data
DATA_DIR = Path(os.environ.get("STASHFACE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))

"""Adapter: implements `stashface_pipeline(image_data)` on top of this
project's real matching logic, so matcher.py has something real to import
instead of a stub.

`image_data` is raw image bytes (e.g. straight from a `requests.get(...)`
download). Returns `{"matched_id": ..., "cosine_score": ...}` (cosine_score
0-100) or `None` if no face was found or nothing cleared the similarity
floor.
"""
from io import BytesIO

from PIL import Image

from models.data_manager import DataManager
from models.image_processor import best_match
from models.paths import DATA_DIR

_data_manager = None  # loaded once, reused across every call


def _get_data_manager() -> DataManager:
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager(collection_path=str(DATA_DIR / "performers.zvec"))
    return _data_manager


def stashface_pipeline(image_data: bytes):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return best_match(image, _get_data_manager())

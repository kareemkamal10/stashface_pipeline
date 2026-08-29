#!/usr/bin/env python3
"""Plain (non-Gradio) face search over the stashface performer index.

This is a stripped-down clone of cc1234/stashface_onnx with the Gradio UI
removed. It's meant to be called as a plain function/CLI step inside a
larger pipeline, not run as an interactive app. The core matching logic
(models/data_manager.py, models/face_recognition.py, models/image_processor.py)
is untouched — only the Gradio wrapper (app.py + web/interface.py) is gone.

Each match already includes the performer's `id` (the stash_id used to key
the vector index) alongside name/confidence/country/etc. — that's not
something we add here, `image_search_performers` already returns it per
match, it just wasn't surfaced by the Gradio "Visual Search" HTML tab.

Usage as a CLI:
    python search.py path/to/photo.jpg --top-k 5

Usage as a library:
    from search import load_data_manager, search_image
    dm = load_data_manager()
    results = search_image("photo.jpg", dm, top_k=5)
    for face in results:
        for performer in face["performers"]:
            print(performer["id"], performer["name"], performer["confidence"])
"""
import argparse
import json
import sys

from PIL import Image

from models.data_manager import DataManager
from models.image_processor import MIN_COSINE_SIMILARITY, image_search_performers
from models.paths import DATA_DIR


def load_data_manager(collection_path: str | None = None) -> DataManager:
    """Open the performer vector index (performers.zvec).

    Defaults to DATA_DIR/performers.zvec, i.e. wherever setup.py synced the
    cc1234/stashface-data bucket to.
    """
    path = collection_path or str(DATA_DIR / "performers.zvec")
    return DataManager(collection_path=path)


def search_image(image_path: str, data_manager: DataManager, top_k: int = 5, min_similarity: float = MIN_COSINE_SIMILARITY):
    """Run face detection + AdaFace embedding + vector search on one image.

    `min_similarity` is a hard floor on cosine similarity (0-1): any match
    below it is dropped and never appears in `performers`. Defaults to 0.55.

    Returns a list with one entry per detected face:
        {
            "area": {...}, "confidence": 0.93, "landmarks": [...],
            "performers": [
                {"id": "...", "name": "...", "confidence": 87, "country": "US", ...},
                ...
            ],
        }
    """
    image = Image.open(image_path).convert("RGB")
    return image_search_performers(image, data_manager, results=top_k, min_cosine=min_similarity)


def _json_default(obj):
    """Safety net: convert any numpy scalar that slips through into a plain
    Python type, so json.dump never crashes on e.g. numpy.int64/float32."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=5, help="Max performers per detected face (default: 5)")
    parser.add_argument(
        "--collection",
        default=None,
        help="Path to performers.zvec (defaults to DATA_DIR/performers.zvec)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=MIN_COSINE_SIMILARITY,
        help=f"Minimum cosine similarity (0-1) a match must have to be kept (default: {MIN_COSINE_SIMILARITY})",
    )
    args = parser.parse_args()

    dm = load_data_manager(args.collection)

    try:
        results = search_image(args.image, dm, top_k=args.top_k, min_similarity=args.min_similarity)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        sys.exit(1)

    # Quick human-readable summary on stderr (ids front and center)...
    for i, face in enumerate(results):
        ids = [p["id"] for p in face["performers"]]
        print(f"face {i}: {ids}", file=sys.stderr)

    # ...full JSON (with ids inside each performer) on stdout, for piping
    # into the next stage of a pipeline.
    json.dump(results, sys.stdout, ensure_ascii=False, indent=2, default=_json_default)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()

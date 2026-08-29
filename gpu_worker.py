"""A self-contained face-matching engine pinned to one GPU (or CPU).

Everything here (analyzer, embedder, index handle) is instance state, not
module-level globals — so you can safely create one FaceMatcher per GPU and
run them in parallel threads, each doing its own detection + embedding +
index lookup without interfering with the others. Used by run_kaggle.py.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from models.data_manager import DataManager
from models.face_recognition import (
    AdaFaceEmbedder,
    MIN_FACE_CONFIDENCE,
    extract_faces,
)
from models.image_processor import MIN_COSINE_SIMILARITY, _score_to_cosine


class FaceMatcher:
    def __init__(
        self,
        data_dir,
        device_id: Optional[int] = None,
        collection_path: Optional[str] = None,
        data_manager: Optional[DataManager] = None,
    ):
        self.device_id = device_id
        label = f"GPU {device_id}" if device_id is not None else "CPU"
        print(f"[{label}] loading AdaFace model ...")

        # Analyzers are built lazily, per detection size, the first time
        # they're needed (see extract_faces(analyzer_cache=..., device_id=...))
        # — same adaptive-size + retry-on-miss behavior as the default
        # single-GPU path, just kept in this worker's own cache so it never
        # touches another worker's GPU.
        self._analyzer_cache: dict = {}

        adaface_model_path = str(Path(data_dir) / "adaface" / "adaface_vit_b_mha_fused.int8q.onnx")
        self.embedder = AdaFaceEmbedder(adaface_model_path, device_id=device_id)

        # IMPORTANT: don't open a second independent handle on the same
        # performers.zvec file per GPU worker — pass one shared DataManager
        # in (built once by the caller) instead. Two workers opening it
        # concurrently is not something to rely on being safe, and if the
        # second open silently fails, every single query from that worker
        # comes back empty — i.e. "no match" for every image, which looks
        # exactly like a matching bug but is actually a data-loading one.
        self.data_manager = data_manager or DataManager(
            collection_path=collection_path or str(Path(data_dir) / "performers.zvec")
        )
        if self.data_manager.collection is None:
            raise RuntimeError(
                f"[{label}] performers.zvec failed to load from "
                f"'{collection_path or str(Path(data_dir) / 'performers.zvec')}' — "
                "check that setup.py finished syncing the bucket before running this."
            )

        print(f"[{label}] ready.")

    def match(
        self,
        image_path: str,
        min_cosine: float = MIN_COSINE_SIMILARITY,
        detection_threshold: float = MIN_FACE_CONFIDENCE,
    ) -> Optional[dict]:
        """Detect faces in the image and return the single strongest match
        across all of them, as {"matched_id": ..., "cosine_score": ...}
        (cosine_score 0-100), or None if nothing clears min_cosine."""
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        if image_array.ndim < 2 or image_array.size == 0:
            return None

        faces = extract_faces(
            image_array,
            min_confidence=detection_threshold,
            analyzer_cache=self._analyzer_cache,  # this worker's own cache
            device_id=self.device_id,              # ... pinned to its own GPU
        )
        if not faces:
            return None

        best = None
        for face_info in faces:
            face_batch = np.expand_dims(face_info["face"], axis=0)
            adaface_emb = self.embedder.embed_batch(face_batch)[0]

            raw_entries = self.data_manager.query_with_vectors(adaface_emb, 50)
            raw_entries = [e for e in raw_entries if _score_to_cosine(e["score"]) >= min_cosine]
            if not raw_entries:
                continue

            top = max(raw_entries, key=lambda e: e["score"])
            cosine_score = round(_score_to_cosine(top["score"]) * 100, 2)
            if best is None or cosine_score > best["cosine_score"]:
                best = {"matched_id": top["id"], "cosine_score": cosine_score}

        return best

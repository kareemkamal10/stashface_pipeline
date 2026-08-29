import os
import numpy as np
import zvec
from typing import Dict, Any, Optional, List, Tuple


def embedding_to_ints(emb: np.ndarray) -> list:
    """Convert L2-normalized float32 embedding to int8 list for VECTOR_INT8."""
    return np.round(emb * 127).clip(-128, 127).astype(np.int8).tolist()


class DataManager:
    def __init__(self, collection_path: str = "data/performers.zvec"):
        self.collection_path = collection_path
        self._metadata_cache: Dict[str, Dict] = {}
        self.collection = None
        self._load_collection()

    def _load_collection(self):
        try:
            self.collection = zvec.open(
                path=self.collection_path,
                option=zvec.CollectionOption(read_only=True, enable_mmap=True),
            )
        except Exception as e:
            print(f"Error loading collection: {e}")

    def get_performer_info(self, stash_id: str, confidence: float, raw_score: float = 0.0) -> Optional[Dict[str, Any]]:
        meta = self._metadata_cache.get(stash_id)
        if not meta:
            return None

        confidence_int = int(confidence * 100)
        source = meta.get("source", "")
        return {
            'id': stash_id,
            'name': meta.get("name", ""),
            'confidence': confidence_int,
            'image': meta.get("image", ""),
            'country': meta.get("country") or None,
            'gender': meta.get("gender") or None,
            'source': source,
            'hits': 1,
            'distance': raw_score,
            'performer_url': meta.get("url") or f"https://stashdb.org/performers/{stash_id}"
        }

    def _query(self, field_name: str, embedding, limit: int) -> Tuple[List[str], List[float]]:
        if self.collection is None:
            return [], []

        results = self.collection.query(
            queries=zvec.Query(field_name=field_name, vector=embedding_to_ints(embedding)),
            topk=limit,
        )

        ids = []
        distances = []
        for doc in results:
            doc_id = doc.id if hasattr(doc, 'id') else doc['id']
            doc_score = doc.score if hasattr(doc, 'score') else doc['score']
            doc_fields = doc.fields if hasattr(doc, 'fields') else doc.get('fields', {})

            ids.append(doc_id)
            distances.append(doc_score)
            self._metadata_cache[doc_id] = doc_fields

        return ids, distances

    def query_with_vectors(self, embedding, limit: int) -> List[dict]:
        if self.collection is None:
            return []

        results = self.collection.query(
            queries=zvec.Query(field_name="adaface", vector=embedding_to_ints(embedding)),
            topk=limit,
            include_vector=True,
        )

        entries = []
        for doc in results:
            doc_id = doc.id if hasattr(doc, 'id') else doc['id']
            doc_score = doc.score if hasattr(doc, 'score') else doc['score']
            doc_fields = doc.fields if hasattr(doc, 'fields') else doc.get('fields', {})
            doc_vectors = doc.vectors if hasattr(doc, 'vectors') else doc.get('vectors', {})
            int8_vec = np.array(doc_vectors.get("adaface", []), dtype=np.int8)
            entries.append({
                "id": doc_id,
                "score": doc_score,
                "fields": doc_fields,
                "vector": int8_vec,
            })
            self._metadata_cache[doc_id] = doc_fields

        return entries

    def query_adaface_index(self, embedding, limit: int) -> Tuple[List[str], List[float]]:
        return self._query("adaface", embedding, limit)

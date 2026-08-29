import re

import numpy as np

from models.face_recognition import AdaFaceRecognition, extract_faces, _face_to_base64

# zvec stores each L2-normalized embedding as int8 = round(value * 127), so the
# raw IP ("score") returned by the index is cosine_similarity * 127 * 127.
_INT8_SCORE_SCALE = 127 * 127  # 16129

# Hard floor: any match whose true cosine similarity is below this is dropped
# and never appears in `performers`, no matter how consolidation groups it.
# 0.55 cosine ~ 0.74 on the 0-1 "confidence" scale used below (score/12000),
# which is roughly what shows up as ~80% in the Space-style display.
MIN_COSINE_SIMILARITY = 0.55


def _score_to_cosine(score):
    return score / _INT8_SCORE_SCALE


def _int8_similarity(v1, v2):
    dot = np.dot(v1.astype(np.float32), v2.astype(np.float32))
    norm = max(np.linalg.norm(v1.astype(np.float32)), 1e-12) * max(np.linalg.norm(v2.astype(np.float32)), 1e-12)
    return dot / norm


def _int8_score_to_confidence(score):
    return min(1.0, max(0.0, score / 12000.0))


def _normalize_name(name):
    name = name.strip().lower()
    name = re.sub(r'[-_\'.]+', ' ', name)
    return re.sub(r'\s+', ' ', name)


def _consolidate(entries, sim_threshold=0.85, min_cosine=MIN_COSINE_SIMILARITY):
    # Drop anything below the hard similarity floor first, so it can never
    # surface in the output — not directly, and not by getting absorbed into
    # a group with a stronger match either.
    entries = [e for e in entries if _score_to_cosine(e["score"]) >= min_cosine]
    if not entries:
        return []

    entries = sorted(entries, key=lambda e: e["score"], reverse=True)
    used = [False] * len(entries)
    groups = []

    # Pass 1: group by exact lowercase name
    for i in range(len(entries)):
        if used[i]:
            continue
        name = _normalize_name(entries[i]["fields"].get("name", ""))
        if not name:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            if _normalize_name(entries[j]["fields"].get("name", "")) == name:
                group.append(j)
                used[j] = True
        groups.append(group)

    # Pass 2: merge unnamed entries into existing groups by vector similarity
    for i in range(len(entries)):
        if used[i]:
            continue
        for g_idx, group in enumerate(groups):
            for m_idx in group:
                sim = _int8_similarity(entries[i]["vector"], entries[m_idx]["vector"])
                if sim > sim_threshold:
                    groups[g_idx].append(i)
                    used[i] = True
                    break
            if used[i]:
                break

    # Pass 3: remaining entries group among themselves
    for i in range(len(entries)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            sim = _int8_similarity(entries[i]["vector"], entries[j]["vector"])
            if sim > sim_threshold:
                group.append(j)
                used[j] = True
        groups.append(group)

    result = []
    for group in groups:
        members = [entries[idx] for idx in group]
        sources = {}
        source_urls = {}
        for m in members:
            src = m["fields"].get("source", "")
            if src:
                sources.setdefault(src, []).append(m)
                if src not in source_urls:
                    url = m["fields"].get("url", "")
                    if url:
                        source_urls[src] = url

        all_images = []
        for m in members:
            img = m["fields"].get("image", "")
            if img and img not in all_images:
                all_images.append(img)

        def completeness(m):
            name = m["fields"].get("name", "")
            image = m["fields"].get("image", "")
            return (1 if name and image else 0 if name else -1, m["score"])

        best = max(members, key=completeness)
        result.append({
            "id": best["id"],
            "name": best["fields"].get("name", ""),
            "score": best["score"],
            "confidence": _int8_score_to_confidence(best["score"]),
            "image": best["fields"].get("image", ""),
            "all_images": all_images,
            "country": best["fields"].get("country") or None,
            "gender": best["fields"].get("gender") or None,
            "url": best["fields"].get("url", ""),
            "sources": list(sources.keys()),
            "source_urls": source_urls,
            "duplicates": len(members) - 1,
        })

    return result


def image_search_performers(image, data_manager, results=3, detection_threshold=0.5, min_cosine=MIN_COSINE_SIMILARITY):
    image_array = np.array(image)
    if image_array.ndim < 2 or image_array.size == 0:
        raise ValueError("No faces found: uploaded image is empty or corrupt")

    ensemble = AdaFaceRecognition()

    faces = extract_faces(image_array, min_confidence=detection_threshold)
    if not faces:
        raise ValueError("No faces found")

    response = []
    for face_info in faces:
        face = face_info['face']
        face_batch = np.expand_dims(face, axis=0)
        embeddings_batch = ensemble.get_adaface_embeddings_batch(face_batch)
        adaface_emb = embeddings_batch[0]

        raw_entries = data_manager.query_with_vectors(adaface_emb, max(results * 5, 50))
        consolidated = _consolidate(raw_entries, sim_threshold=0.75, min_cosine=min_cosine)
        consolidated = consolidated[:results]

        im_b64 = _face_to_base64(face)

        performers = []
        for entry in consolidated:
            performers.append({
                'id': entry['id'],
            })

        response.append({
            'image': im_b64,
            'area': face_info['facial_area'],
            'confidence': face_info['confidence'],
            'landmarks': face_info['landmarks'],
            'performers': performers,
        })
    return response


def best_match(image, data_manager, detection_threshold=0.5, min_cosine=MIN_COSINE_SIMILARITY):
    """Return the single strongest performer match across all faces in an
    image, as `{"matched_id": ..., "cosine_score": ...}` (cosine_score is a
    true cosine similarity percentage, 0-100), or `None` if nothing clears
    `min_cosine`.

    Unlike `image_search_performers` (which returns every detected face with
    up to `results` candidates each, trimmed to id/name/confidence for
    display), this is meant for bulk 1-image-in -> 1-best-id-out matching
    pipelines, e.g. matcher.py.
    """
    image_array = np.array(image)
    if image_array.ndim < 2 or image_array.size == 0:
        return None

    faces = extract_faces(image_array, min_confidence=detection_threshold)
    if not faces:
        return None

    ensemble = AdaFaceRecognition()
    best = None
    for face_info in faces:
        face_batch = np.expand_dims(face_info['face'], axis=0)
        adaface_emb = ensemble.get_adaface_embeddings_batch(face_batch)[0]

        raw_entries = data_manager.query_with_vectors(adaface_emb, 50)
        raw_entries = [e for e in raw_entries if _score_to_cosine(e["score"]) >= min_cosine]
        if not raw_entries:
            continue

        top = max(raw_entries, key=lambda e: e["score"])
        cosine_score = round(_score_to_cosine(top["score"]) * 100, 2)
        if best is None or cosine_score > best["cosine_score"]:
            best = {"matched_id": top["id"], "cosine_score": cosine_score}

    return best

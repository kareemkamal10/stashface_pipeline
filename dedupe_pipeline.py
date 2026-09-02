#!/usr/bin/env python3
"""
dedupe_pipeline.py

يبني قاعدة بيانات مؤقتة (في الرام) من صور performers_with_tpdb.json و
performers_without_tpdb.json مع بعض، باستخدام نفس detection/embedding
القديمين (buffalo_l + AdaFace ViT-B) - من غير أي اعتماد على performers.zvec
أو DataManager.

المرحلة 1 - البناء:
  لكل عنصر (بعد استبعاد أي عنصر مفيهوش image من الأساس):
    - تحميل الصورة (retries + timeout)
    - كشف الوجوه:
        - فشل تحميل        -> download_failed
        - مفيش وش          -> no_face_detected
        - أكتر من وش        -> multiple_faces_detected
        - وش واحد بالظبط    -> accepted (يتحسب embedding وتدخل الـ DB)

المرحلة 2 - الاستعلام:
  لكل عنصر accepted، تشوف أقرب top-10 في نفس الـ DB (شامل نفسه)،
  تستبعد أي نتيجة بنفس (file_type, id) بتاعت الاستعلام، وتاخد أول نتيجة
  باقية:
    - >= 70%        -> confirmed_duplicates.json
    - 55% - 70%      -> needs_human_review.json
    - < 55%          -> لا تتسجل

المرحلة 3 - الرفع:
  كل ملفات التقارير + قاعدة الـ embeddings (.npz) بترفع لمجلد ثابت
  "reports/" جوا نفس الـ HF dataset repo.

الاستخدام (على Kaggle، بعد ما setup.py يزامن الموديل):
    python dedupe_pipeline.py \
        --hf-dataset-id <repo_id> \
        --hf-token $HF_TOKEN \
        --data-dir /kaggle/temp/stashface-data
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from models.face_recognition import AdaFaceEmbedder, MIN_FACE_CONFIDENCE, extract_faces
from models.paths import DATA_DIR

INPUT_WITH_TPDB = "performers_with_tpdb.json"
INPUT_WITHOUT_TPDB = "performers_without_tpdb.json"

REPORTS_DIR_IN_REPO = "reports"  # مجلد ثابت جوا الـ HF dataset repo

TOP_K = 10
CONFIRMED_THRESHOLD = 70.0
REVIEW_THRESHOLD = 55.0

DOWNLOAD_TIMEOUT = 15
MAX_DOWNLOAD_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2
DOWNLOAD_WORKERS = 16
DEFAULT_BATCH_SIZE = 5000

OUT_ACCEPTED_DB = "face_db.npz"
OUT_DOWNLOAD_FAILED = "download_failed.json"
OUT_NO_FACE = "no_face_detected.json"
OUT_MULTI_FACE = "multiple_faces_detected.json"
OUT_CONFIRMED = "confirmed_duplicates.json"
OUT_REVIEW = "needs_human_review.json"


# --- تحميل المدخلات --------------------------------------------------------

def load_records(path: str, file_type: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data:
        image = (r.get("image") or "").strip()
        if not image:
            continue  # مفيش image خالص -> يتجاهل تمامًا، مش بيدخل أي تقرير
        out.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "image": image,
            "source": r.get("source"),
            "file_type": file_type,
        })
    return out


def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _key(entry: dict) -> str:
    return f"{entry['file_type']}:{entry['id']}"


# --- تحميل الصور -----------------------------------------------------------

def download_one(entry: dict) -> tuple:
    """يرجع (entry, image_array أو None, error أو None)."""
    url = entry["image"]
    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            image = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
            return entry, np.array(image), None
        except Exception as e:  # noqa: BLE001 - أي فشل تحميل/فك صورة يتسجل كفشل تحميل
            last_error = e
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return entry, None, str(last_error)


def download_batch(batch: list, workers: int) -> tuple[list, float]:
    start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(download_one, batch):
            results.append(result)
    return results, time.monotonic() - start


# --- checkpoint (resumability) ---------------------------------------------

def load_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_json_atomic(path: str, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_db_checkpoint(path: str):
    if not os.path.exists(path):
        return [], [], [], [], np.zeros((0, 512), dtype=np.float32)
    npz = np.load(path, allow_pickle=True)
    ids = list(npz["ids"])
    names = list(npz["names"])
    file_types = list(npz["file_types"])
    images = list(npz["images"])
    embeddings = npz["embeddings"]
    return ids, names, file_types, images, embeddings


def save_db_atomic(path: str, ids, names, file_types, images, embeddings):
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(
        tmp,
        ids=np.array(ids, dtype=object),
        names=np.array(names, dtype=object),
        file_types=np.array(file_types, dtype=object),
        images=np.array(images, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    os.replace(tmp, path)


# --- المرحلة 1: البناء -------------------------------------------------

def build_phase(all_entries: list, embedder: AdaFaceEmbedder, out_dir: Path,
                 batch_size: int, download_workers: int):
    download_failed = load_json_list(str(out_dir / OUT_DOWNLOAD_FAILED))
    no_face = load_json_list(str(out_dir / OUT_NO_FACE))
    multi_face = load_json_list(str(out_dir / OUT_MULTI_FACE))
    db_ids, db_names, db_file_types, db_images, db_embeddings = load_db_checkpoint(str(out_dir / OUT_ACCEPTED_DB))

    done_keys = set()
    done_keys.update(f"{e['file_type']}:{e['id']}" for e in download_failed)
    done_keys.update(f"{e['file_type']}:{e['id']}" for e in no_face)
    done_keys.update(f"{e['file_type']}:{e['id']}" for e in multi_face)
    done_keys.update(f"{ft}:{i}" for ft, i in zip(db_file_types, db_ids))

    todo = [e for e in all_entries if _key(e) not in done_keys]
    print(f"[بناء] إجمالي {len(all_entries)}، اتعالج قبل كده {len(done_keys)}، متبقي {len(todo)}.")

    embeddings_buffer = list(db_embeddings)

    batches = list(chunked(todo, batch_size))
    for batch_num, batch in enumerate(batches, start=1):
        downloaded, download_seconds = download_batch(batch, download_workers)

        batch_new_embeddings = []
        n_ok = n_dl_fail = n_no_face = n_multi = 0

        for entry, image_array, error in downloaded:
            if image_array is None:
                download_failed.append({**{k: entry[k] for k in ("id", "name", "image", "source", "file_type")},
                                         "error": error})
                n_dl_fail += 1
                continue

            faces = extract_faces(image_array, min_confidence=MIN_FACE_CONFIDENCE)
            if len(faces) == 0:
                no_face.append({k: entry[k] for k in ("id", "name", "image", "source", "file_type")})
                n_no_face += 1
                continue
            if len(faces) > 1:
                multi_face.append({k: entry[k] for k in ("id", "name", "image", "source", "file_type")})
                n_multi += 1
                continue

            face_batch = np.expand_dims(faces[0]["face"], axis=0)
            emb = embedder.embed_batch(face_batch)[0]
            db_ids.append(entry["id"])
            db_names.append(entry["name"])
            db_file_types.append(entry["file_type"])
            db_images.append(entry["image"])
            batch_new_embeddings.append(emb)
            n_ok += 1

        if batch_new_embeddings:
            embeddings_buffer.extend(batch_new_embeddings)

        save_json_atomic(str(out_dir / OUT_DOWNLOAD_FAILED), download_failed)
        save_json_atomic(str(out_dir / OUT_NO_FACE), no_face)
        save_json_atomic(str(out_dir / OUT_MULTI_FACE), multi_face)
        save_db_atomic(str(out_dir / OUT_ACCEPTED_DB), db_ids, db_names, db_file_types, db_images, embeddings_buffer)

        print(f"[بناء] batch {batch_num}/{len(batches)}: تحميل {len(batch)} في {_fmt_duration(download_seconds)} | "
              f"مقبول {n_ok}, فشل تحميل {n_dl_fail}, مفيش وش {n_no_face}, أكتر من وش {n_multi}. [saved]")

    return db_ids, db_names, db_file_types, db_images, np.asarray(embeddings_buffer, dtype=np.float32)


# --- المرحلة 2: الاستعلام -----------------------------------------------

def query_phase(db_ids, db_names, db_file_types, db_images, embeddings: np.ndarray, out_dir: Path):
    if embeddings.shape[0] == 0:
        print("[استعلام] القاعدة فاضية، مفيش حاجة تتقارن.")
        return

    # الـ embeddings من AdaFace متوقع تكون L2-normalized بالفعل، فالـ dot
    # product = cosine similarity مباشرة.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = embeddings / norms
    sims = normed @ normed.T  # matrix مربعة: sims[i, j] = تشابه العنصر i مع j

    confirmed = []
    review = []
    n = len(db_ids)
    for i in range(n):
        order = np.argsort(-sims[i])[:TOP_K + 1]  # +1 لأن نفسه هيكون غالبًا أول واحد
        best = None
        for j in order:
            if db_file_types[j] == db_file_types[i] and db_ids[j] == db_ids[i]:
                continue  # استبعاد نفس العنصر (استبعاد بالـ id، مش بالترتيب)
            score = round(float(sims[i, j]) * 100, 2)
            best = (j, score)
            break

        if best is None:
            continue
        j, score = best
        record = {
            "query_id": db_ids[i], "query_file_type": db_file_types[i], "query_name": db_names[i],
            "match_id": db_ids[j], "match_file_type": db_file_types[j], "match_name": db_names[j],
            "cosine_score": score,
        }
        if score >= CONFIRMED_THRESHOLD:
            confirmed.append(record)
        elif score >= REVIEW_THRESHOLD:
            review.append(record)

    save_json_atomic(str(out_dir / OUT_CONFIRMED), confirmed)
    save_json_atomic(str(out_dir / OUT_REVIEW), review)
    print(f"[استعلام] تطابقات مؤكدة (>= {CONFIRMED_THRESHOLD}%): {len(confirmed)}")
    print(f"[استعلام] تحتاج مراجعة بشرية ({REVIEW_THRESHOLD}%-{CONFIRMED_THRESHOLD}%): {len(review)}")


# --- المرحلة 3: الرفع لـ HF -------------------------------------------------

def upload_reports(out_dir: Path, hf_dataset_id: str, hf_token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    for filename in [OUT_ACCEPTED_DB, OUT_DOWNLOAD_FAILED, OUT_NO_FACE, OUT_MULTI_FACE, OUT_CONFIRMED, OUT_REVIEW]:
        local_path = out_dir / filename
        if not local_path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{REPORTS_DIR_IN_REPO}/{filename}",
            repo_id=hf_dataset_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"dedupe_pipeline: add {filename}",
        )
        print(f"[رفع] {filename} -> {hf_dataset_id}/{REPORTS_DIR_IN_REPO}/{filename}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-tpdb", default=INPUT_WITH_TPDB)
    ap.add_argument("--without-tpdb", default=INPUT_WITHOUT_TPDB)
    ap.add_argument("--out-dir", default="dedupe_reports")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--device-id", type=int, default=None, help="GPU device id (افتراضي: CPU)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--download-workers", type=int, default=DOWNLOAD_WORKERS)
    ap.add_argument("--hf-dataset-id", default=None, help="لو محدد، هيترفع كل حاجة لـ reports/ جوا الـ repo ده")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_records(args.with_tpdb, "with_tpdb") + load_records(args.without_tpdb, "without_tpdb")
    print(f"تم تحميل {len(entries)} عنصر ليهم image (بعد استبعاد اللي مفيهوش).")

    adaface_model_path = str(Path(args.data_dir) / "adaface" / "adaface_vit_b_mha_fused.int8q.onnx")
    embedder = AdaFaceEmbedder(adaface_model_path, device_id=args.device_id)

    db_ids, db_names, db_file_types, db_images, embeddings = build_phase(
        entries, embedder, out_dir, args.batch_size, args.download_workers,
    )
    print(f"انتهت مرحلة البناء: {len(db_ids)} عنصر دخلوا القاعدة فعليًا.")

    query_phase(db_ids, db_names, db_file_types, db_images, embeddings, out_dir)

    if args.hf_dataset_id:
        if not args.hf_token:
            print("WARNING: --hf-dataset-id محدد بس مفيش --hf-token/HF_TOKEN، هتخطى الرفع.", file=sys.stderr)
        else:
            upload_reports(out_dir, args.hf_dataset_id, args.hf_token)

    print("خلص.")


if __name__ == "__main__":
    main()

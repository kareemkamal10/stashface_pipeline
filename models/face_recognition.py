import io
import base64
import threading
import warnings
import os
import sys
import numpy as np
import cv2
from pathlib import Path

from PIL import Image
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

from models.paths import DATA_DIR

warnings.filterwarnings("ignore", message=r"`estimate` is deprecated", category=FutureWarning, module="insightface")

_photo_analyzer = None
_sprite_analyzer = None
_tile_analyzer = None
_adaptive_analyzers: dict[tuple[int, int], FaceAnalysis] = {}


_DETECTOR_MODEL = "buffalo_l"

# Guards two things that are NOT thread-safe when multiple GPU worker threads
# build their own analyzer at the same time (e.g. run_kaggle.py's multi-GPU
# pipeline, where each worker lazily builds its first analyzer on its own
# thread the moment it pulls its first item off the queue):
#   1. sys.stdout is process-global. Redirecting it to devnull to silence
#      insightface's setup logging, then restoring it, is only safe if one
#      thread does the whole redirect-build-restore sequence at a time —
#      otherwise thread A can end up restoring stdout to thread B's (already
#      closed) devnull handle, which surfaces later as
#      "ValueError: I/O operation on closed file" from an unrelated print().
#   2. The first FaceAnalysis(name=...) call for a given model auto-downloads
#      and unzips that model into the shared ~/.insightface/models/<name>
#      directory. Two threads doing this at once race on the same files,
#      which shows up as "[Errno 17] File exists" for one thread and a
#      corrupted/partial .onnx ("Protobuf parsing failed") for the other.
# Serializing analyzer construction fixes both, and only costs a bit of
# one-time startup latency (subsequent calls hit the in-memory cache and
# never touch this lock).
_analyzer_build_lock = threading.Lock()


def set_detector_model(name: str):
    global _DETECTOR_MODEL
    global _photo_analyzer, _sprite_analyzer, _tile_analyzer, _adaptive_analyzers
    _photo_analyzer = _sprite_analyzer = _tile_analyzer = None
    _adaptive_analyzers.clear()
    _DETECTOR_MODEL = name
    print(f"Detector model set to: {name}")


def _create_face_analysis(name, providers, ctx_id, det_size):
    """Construct + prepare a FaceAnalysis instance with stdout silenced.

    Must only ever be called while holding _analyzer_build_lock — see the
    comment above it for why.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            analyzer = FaceAnalysis(
                name=name,
                allowed_modules=["detection"],
                providers=providers,
            )
            analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
        finally:
            sys.stdout = old_stdout
    return analyzer


def _build_analyzer(det_size, model_name=None):
    providers = _get_onnx_providers()
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    name = model_name or _DETECTOR_MODEL
    with _analyzer_build_lock:
        return _create_face_analysis(name, providers, ctx_id, det_size)


def _get_face_analyzer(det_size=None):
    global _photo_analyzer, _sprite_analyzer, _tile_analyzer
    if det_size is not None:
        if det_size == (640, 640):
            if _tile_analyzer is None:
                _tile_analyzer = _build_analyzer((640, 640), model_name="buffalo_sc")
                print("Tile detector loaded (buffalo_sc, 640x640).")
            return _tile_analyzer
        if _sprite_analyzer is None:
            _sprite_analyzer = _build_analyzer((320, 320), model_name="buffalo_sc")
            print("Sprite detector loaded (buffalo_sc, 320x320).")
        return _sprite_analyzer
    if _photo_analyzer is None:
        _photo_analyzer = _build_analyzer((640, 640))
        print("Photo detector loaded (buffalo_l, 640x640).")
    return _photo_analyzer


def build_sprite_analyzer():
    return _build_analyzer((320, 320), model_name="buffalo_sc")


MIN_FACE_CONFIDENCE = 0.5

REFERENCE_LANDMARKS = np.array([
    [30.5, 22.5],
    [81.5, 22.5],
    [56.5, 50.0],
    [39.5, 86.0],
    [73.5, 86.0],
], dtype=np.float32)


def extract_faces(image, min_confidence=MIN_FACE_CONFIDENCE, det_size=None, analyzer=None,
                   analyzer_cache=None, device_id=None):
    """
    analyzer_cache / device_id: for multi-GPU pipelines. Pass an empty dict
    you own (one per GPU worker) as analyzer_cache and that worker's GPU id
    as device_id, and this function will build+cache analyzers pinned to
    that device (same adaptive-sizing + fallback-on-miss behavior as the
    default single-GPU path below, just scoped to your own cache/device
    instead of the module-level one). Leave both as None for the default
    single-GPU/CPU behavior.
    """
    cache = analyzer_cache if analyzer_cache is not None else _adaptive_analyzers
    build = (lambda size: build_face_analyzer(device_id=device_id, det_size=size)) if device_id is not None else _build_analyzer

    if analyzer is None:
        if det_size is None:
            h, w = image.shape[:2]
            target = max(h, w, 320)
            target = min(target, 640)
            target = ((target + 31) // 32) * 32
            det_size = (target, target)
        analyzer = cache.get(det_size)
        if analyzer is None:
            analyzer = build(det_size)
            cache[det_size] = analyzer
    # else: use the analyzer that was explicitly passed in, as-is.
    detected = analyzer.get(image)
    if not detected and det_size is not None and det_size[0] >= 576:
        for sz in [544, 480, 416, 352]:
            key = (sz, sz)
            alt = cache.get(key)
            if alt is None:
                alt = build(key)
                cache[key] = alt
            detected = alt.get(image)
            if detected:
                break
    results = []
    for face in detected:
        if face.det_score < min_confidence:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = norm_crop(image, face.kps, image_size=112)
        kps = face.kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(kps, REFERENCE_LANDMARKS)
        ones = np.ones((5, 1), dtype=np.float32)
        crop_kps = M @ np.hstack([kps, ones]).T
        landmarks = [{'x': int(pt[0]), 'y': int(pt[1])} for pt in crop_kps.T]
        results.append({
            'face': crop,
            'facial_area': {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)},
            'confidence': float(face.det_score),
            'landmarks': landmarks,
        })
    return results


def extract_faces_batch(
    frames: list[np.ndarray],
    min_confidence: float = MIN_FACE_CONFIDENCE,
    frames_per_tile: int = 4,
    analyzer=None,
) -> list[tuple[dict, int]]:
    """Tile frames into composites and detect faces in batch.

    Groups frames into a grid layout, runs SCRFD detection once per composite,
    maps bbox results back to original frame coordinates.

    Returns list of (result_dict, original_frame_index).
    """
    if not frames:
        return []

    n_frames = len(frames)
    results: list[tuple[dict, int]] = []

    if analyzer is None:
        analyzer = _get_face_analyzer(det_size=(640, 640))

    for batch_start in range(0, n_frames, frames_per_tile):
        batch = frames[batch_start:batch_start + frames_per_tile]
        n_in = len(batch)

        cols = int(np.ceil(np.sqrt(n_in)))
        rows = int(np.ceil(n_in / cols))

        cell_w = max(f.shape[1] for f in batch)
        cell_h = max(f.shape[0] for f in batch)
        comp_h = rows * cell_h
        comp_w = cols * cell_w
        composite = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)

        tile_map = []
        for i, frame in enumerate(batch):
            r, c = divmod(i, cols)
            y_off, x_off = r * cell_h, c * cell_w
            h, w = frame.shape[:2]
            composite[y_off:y_off + h, x_off:x_off + w] = frame
            tile_map.append((batch_start + i, x_off, y_off, w, h))

        dets = analyzer.get(composite)
        for face in dets:
            if face.det_score < min_confidence:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            for frame_idx, tx, ty, fw, fh in tile_map:
                if tx <= cx < tx + fw and ty <= cy < ty + fh:
                    crop = norm_crop(composite, face.kps, image_size=112)
                    kps = face.kps.astype(np.float32)
                    M, _ = cv2.estimateAffinePartial2D(kps, REFERENCE_LANDMARKS)
                    ones = np.ones((5, 1), dtype=np.float32)
                    crop_kps = M @ np.hstack([kps, ones]).T
                    landmarks = [{'x': int(pt[0]), 'y': int(pt[1])} for pt in crop_kps.T]

                    results.append(({
                        'face': crop,
                        'facial_area': {
                            'x': int(x1 - tx), 'y': int(y1 - ty),
                            'w': int(x2 - x1), 'h': int(y2 - y1),
                        },
                        'confidence': float(face.det_score),
                        'landmarks': landmarks,
                    }, frame_idx))
                    break

    return results


def _get_onnx_providers():
    if not _cuda_runtime_available():
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _cuda_runtime_available() -> bool:
    """True only if CUDAExecutionProvider is both compiled in AND its
    runtime shared libraries (cuDNN/cuBLAS) actually load. Compile-time
    presence in ort.get_available_providers() is not enough — the library
    can still fail to load at session-creation time if the host's
    CUDA/cuDNN versions don't match what onnxruntime-gpu was built against."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        import ctypes
        ctypes.CDLL("libcudnn.so.9")
        return True
    except OSError:
        return False


def _providers_for_device(device_id: int | None):
    """Like _get_onnx_providers(), but pinned to a specific GPU via
    onnxruntime's device_id provider option — used by build_face_analyzer()
    and AdaFaceEmbedder for multi-GPU pipelines. device_id=None means CPU."""
    if device_id is None:
        return ["CPUExecutionProvider"]
    if not _cuda_runtime_available():
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]


def build_face_analyzer(device_id: int | None = None, det_size=(640, 640), model_name="buffalo_l"):
    """Build a standalone FaceAnalysis instance pinned to a specific GPU
    (or CPU if device_id is None).

    Unlike the module-level analyzer cache used by extract_faces()'s default
    path (a single shared analyzer for the whole process), this always
    returns a brand new instance — so each GPU worker in a multi-GPU
    pipeline can own one bound to its own device. Pass the result via
    extract_faces(..., analyzer=...).
    """
    providers = _providers_for_device(device_id)
    ctx_id = device_id if device_id is not None else -1
    with _analyzer_build_lock:
        return _create_face_analysis(model_name, providers, ctx_id, det_size)


# ---------------------------------------------------------------------------
# AdaFace ViT-B ONNX embedder — query (server) path.
#
# Uses adaface_vit_b_mha_fused.int8q.onnx (INT8, 111 MB), optimized for CPU
# inference on Hugging Face Spaces via MHA fusion + per-channel quantization.
#
# The indexing pipeline (facepipe/embed.py) uses the fp32 variant instead
# (adaface_vit_b_webface4m.onnx) for fast GPU batch processing. The two
# models produce functionally identical embeddings (cosine similarity 0.9986
# measured on real crops), so ranking quality is preserved across paths.
# ---------------------------------------------------------------------------
_ada_session: ort.InferenceSession | None = None
_ada_input: str | None = None
_ada_output: str | None = None


def _get_adaface_model_path():
    return str(DATA_DIR / "adaface" / "adaface_vit_b_mha_fused.int8q.onnx")


def _compute_adaface_embeddings(x: np.ndarray) -> np.ndarray:
    global _ada_session, _ada_input, _ada_output
    if _ada_session is None:
        _ada_session = ort.InferenceSession(
            _get_adaface_model_path(),
            providers=_get_onnx_providers(),
        )
        _ada_input = _ada_session.get_inputs()[0].name
        _ada_output = _ada_session.get_outputs()[0].name
        device = 'GPU' if _get_onnx_providers()[0] == 'CUDAExecutionProvider' else 'CPU'
        print(f"AdaFace model loaded (mha_fused.int8q, {device}).")
    return _ada_session.run([_ada_output], {_ada_input: x})[0]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


class AdaFaceRecognition:
    @staticmethod
    def get_adaface_embeddings_batch(faces: np.ndarray) -> np.ndarray:
        x = (faces.astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (0, 3, 1, 2))
        embs = _compute_adaface_embeddings(x)
        return _l2_normalize(embs)


EnsembleFaceRecognition = AdaFaceRecognition


class AdaFaceEmbedder:
    """Standalone AdaFace ONNX embedder pinned to one specific device.

    Unlike AdaFaceRecognition (which lazily builds and reuses a single
    module-level session — fine for the single-GPU/CPU search.py/matcher.py
    path), each AdaFaceEmbedder instance owns its own onnxruntime session,
    so multiple GPU workers can each hold one bound to their own device
    without stepping on each other. Used by gpu_worker.py.
    """

    def __init__(self, model_path: str | None = None, device_id: int | None = None):
        self.session = ort.InferenceSession(
            model_path or _get_adaface_model_path(),
            providers=_providers_for_device(device_id),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_batch(self, faces: np.ndarray) -> np.ndarray:
        x = (faces.astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (0, 3, 1, 2))
        embs = self.session.run([self.output_name], {self.input_name: x})[0]
        return _l2_normalize(embs)


def _dist_to_confidence(score):
    """Convert IP score (from VECTOR_INT8) to confidence [0,1].

    VECTOR_INT8 stores l2_normalized(emb) * 127 as int8, so zvec returns
    sum(int8_q[i] * int8_db[i]) which is scaled by 127^2 = 16129.
    Dividing by 16129 recovers cosine similarity (for L2-normalized vectors).
    """
    cos_sim = float(score) / 16129.0
    return max(0.0, (cos_sim - 0.3) / 0.7)


def _face_to_base64(face_array):
    buf = io.BytesIO()
    Image.fromarray(face_array).save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

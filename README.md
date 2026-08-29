# stashface-pipeline

A plain-script clone of [cc1234/stashface_onnx](https://huggingface.co/spaces/cc1234/stashface_onnx),
stripped of the Gradio UI so it can be dropped into a bigger pipeline as a
regular Python step instead of a web app.

## What changed vs the original Space

- **No Gradio.** `app.py` and `web/interface.py` (the Gradio tabs) are gone.
  `search.py` replaces them with a plain function / CLI.
- **`id` is front and center.** The matching logic
  (`models/image_processor.py::image_search_performers`) already returns each
  performer's `id` (their stash_id) — the Gradio "Visual Search" tab just
  never displayed it in the HTML cards. `search.py` prints it straight to
  stderr for a quick look, and it's in every performer dict in the JSON
  output.
- **`models/paths.py` added.** The original Space mounts the
  `cc1234/stashface-data` bucket directly as a Space volume, so it didn't
  need a download step. Running outside of a Space, `setup.py` downloads
  the same bucket to a local `data/` folder instead, and `DATA_DIR` in
  `models/paths.py` points there.
- Everything else (`models/data_manager.py`, `models/face_recognition.py`,
  `models/image_processor.py`) is untouched — same detection (SCRFD /
  buffalo_l via insightface), same AdaFace ONNX embedder, same zvec index
  query + consolidation logic.
- Dropped for now (not needed for a single-image pipeline step): the
  sprite/VTT batch search and the raw-vector search endpoint. Say the word
  if you want those ported over too.

## Setup

```bash
python setup.py
```

This installs `requirements.txt` + the `hf` CLI, then runs
`hf buckets sync hf://buckets/cc1234/stashface-data ./data`.

If the bucket is private you'll need a token first:

```bash
hf auth login
# or: export HF_TOKEN=hf_xxx
```

## Run

```bash
python search.py path/to/photo.jpg --top-k 5
```

stderr gets a quick `face 0: ['id1', 'id2', ...]` summary; stdout gets the
full JSON (one entry per detected face, each with a `performers` list where
every performer dict includes `id`), ready to pipe into the next step.

## Bulk matching (matcher.py)

`matcher.py` matches every entry in `tpdb_all_performers.json` against the
local index and writes confirmed matches (cosine ≥ 55%) to
`matched_identities.json`. Batched, crash-safe (atomic saves + resumes
where it left off), retries failed downloads up to 4 times, and never lets
one bad entry stop the whole run.

```bash
python matcher.py --limit 30    # quick test
python matcher.py               # full dataset
```

## Bulk matching at scale on Kaggle, 2x T4 (run_kaggle.py + stashface_kaggle.ipynb)

For running the full dataset on Kaggle with 2 T4 GPUs, use `run_kaggle.py`
instead of `matcher.py`. Same input/output contract and the same 55%
threshold, but:

- Downloading and matching run **concurrently**, not one after another —
  a pool of download threads keeps feeding a queue (capacity =
  `--batch-size`, default 15,000) while two GPU worker threads (one per
  T4) drain it independently. Once the queue is full, downloading just
  pauses until a GPU worker frees a slot, so the next batch is always
  downloading in the background while the current one is being matched —
  no explicit "wait for batch 1, then start batch 2" step.
- Each downloaded image is saved as `data/_download_cache/{tpdb_id}.{ext}`
  and deleted right after it's matched, so disk usage stays bounded.
- `matched_identities.json` is saved atomically every `--batch-size`
  results, and re-running after a crash/stop skips `tpdb_id`s already
  recorded there.

Open `stashface_kaggle.ipynb` in a new Kaggle notebook (set Accelerator to
**GPU T4 x2** first), fill in `GITHUB_REPO_URL` and `HF_TOKEN` in the first
cell, and run the cells top to bottom. It clones this repo, installs
`onnxruntime-gpu`, downloads the bucket data and `tpdb_all_performers.json`
from your Hugging Face dataset, runs the matcher, and uploads
`matched_identities.json` back to that same dataset at the end.


## Calling search.py from Python

```python
from search import load_data_manager, search_image

dm = load_data_manager()
results = search_image("photo.jpg", dm, top_k=5)
for face in results:
    for performer in face["performers"]:
        print(performer["id"], performer["name"], performer["confidence"])
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PhysicsLENS is a web tool that evaluates physics accuracy in AI-generated video. It's structured as a
four-stage medical diagnostic pipeline (triage → localize → specialize → diagnose):

| Stage | Role | Cost |
|-------|------|------|
| 1 — Screening | Cheap signals flag suspicious regions (temporal smoothness, optical flow, embeddings, VLM, camera motion) | Cheap |
| 2 — Differential Diagnosis | Localize failures, extract trajectories, rank hypotheses | Medium |
| 3 — Specialist Evaluation | Confirm/reject specific physics-failure types (collision, gravity, momentum, friction, ...) | Expensive |
| 4 — Final Diagnosis | Aggregate score, severity, breakdown timing, report | Output |

FastAPI backend (`backend/`) + a single self-contained static HTML frontend (`frontend/index.html`, no
build step, no npm). No test framework (pytest etc.) is configured — "tests" are standalone async scripts
under `backend/scripts/` run directly with `python`.

## Commands

```bash
# Setup (CPU-only; base deps)
conda create -n physicslens python=3.11 -y && conda activate physicslens
cd backend && pip install -r requirements.txt

# GPU stack (needed for the object tracker / embedding pipelines: torch, transformers>=5.5, SAM3, DINOv2)
pip install -r requirements-gpu.txt
hf auth login   # or export HF_TOKEN=<token> — SAM3 (facebook/sam3) is gated; DINOv2 is not

# Run the server (serves API + static frontend together on one port)
cd backend && uvicorn main:app --reload --port 8000
# then open http://localhost:8000

# Sanity-check the GPU model stack (torch/CUDA, transformers SAM3/DINOv2 classes, gating)
python backend/scripts/check_models.py
```

### Running "tests"

There is no pytest suite. Each script under `backend/scripts/` is a standalone entry point — run it
directly with `python` from the repo root or `backend/`; each inserts the right path itself via
`sys.path.insert`. There is no way to run a "single test function" — run the whole script; some assert
internally, some just print a PASS/FAIL summary.

```bash
python backend/scripts/test_vlm_scoring.py      # pure unit tests: JSON parsing + payload building (no network)
python backend/scripts/test_vlm_pipeline.py     # integration test: full VLM suspicion pipeline (needs OPENROUTER key)
python backend/scripts/test_object_tracker.py   # smoke test: stage-2 object tracker against test_videos/ clips
python backend/scripts/vlm_failure_mode_eval.py # multi-frame vs single-frame AUC eval; writes vlm_failure_mode_eval.json
```

No linter/formatter is configured in this repo.

## Architecture

### Pipeline contract (the core abstraction)

Every pipeline module exports a single async generator:

```python
async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    ...
    yield {"type": "log", "level": "info", "text": "..."}
    yield {"type": "metric", "label": "...", "value": "...", "sub": "..."}
    yield {"type": "severity", "label": "...", "value": 0-100, "color": "#hex"}
    yield {"type": "done"}
```

`settings` is a raw JSON string (parsed inside the pipeline, not by the caller). Event types: `log`,
`metric`, `severity`, `image`, `plotly`, `video`/`marker_video`, `signal`, `done`, `error` — the full schema
with required keys is documented at the top of `backend/main.py` and in the README. The frontend renders
whatever stream of events it receives; adding a new event type requires updating both the pipeline and the
frontend's renderer in `frontend/index.html`.

### Pipeline registration (`backend/main.py`)

`PIPELINES` is a single dict registry mapping pipeline id → `{id, name, desc, badge, dummy, requires_pair,
settings, run}`. This is the one place that wires a pipeline module into the app:
- `dummy: True` marks a stub (returns placeholder output) — shown with a STUB badge in the UI. Most of
  stage 2's generator/ranker and all of stage 3/4 (except `s4_report`) are currently stubs.
- `settings` describes the UI-editable knobs (number/text/select/password) for that pipeline; the frontend
  builds its settings form from this list, and the values round-trip back as the `settings` JSON string.
- `GET /pipelines` exposes this registry (minus the `run` callables) for the frontend to build the stage
  tabs and test lists.
- `POST /run` is the original single-video entrypoint (accepts an uploaded file, optionally a second
  `video_ai` when `requires_pair`, streams NDJSON events back).

### Dataset/batch API (`backend/dataset_api.py`)

A self-contained router (`build_dataset_router(PIPELINES)`, mounted under `/dataset`) added on top of the
single-video flow without modifying it. It registers uploaded/downloaded files under a short id
(`_FILES: id -> path`) so the frontend can run pipelines against a video repeatedly by id instead of
re-uploading. Key routes: `/dataset/upload` (browser files/folder), `/dataset/hf_download` (streams
progress while pulling every video file out of a HuggingFace dataset repo), `/dataset/run` (same NDJSON
contract as `/run`, but by `file_id`), `/dataset/file/{id}` (serves for in-browser preview).

The frontend has one unified Dataset view (grid of videos → per-video report) — there is no separate
single-video mode; one video is just a batch of one.

### Cross-stage shared state (`backend/tools/`)

This is what makes multi-stage evidence propagation work despite each pipeline's `run()` only receiving
`(video_path, settings)`:

- **`tools/evidence.py`** — `EvidenceStore` (LRU, thread-safe, in-process singleton `EVIDENCE`), keyed by
  `file_hash(video_path)` (content hash of size + head/tail bytes, memoized). Stage 2 writes structured
  results; Stage 3/4 read them back by the same video-content key. This is how "Stage 2 trajectories →
  Stage 3 specialists → Stage 4 scorer" data passing happens without a real message bus. Evidence is
  cleared on server restart (intentionally ephemeral).
- **`tools/tracking.py`** — `get_tracks(video_path)` is the canonical, cached Shi-Tomasi + LK object
  tracker. It exists so every downstream stage sees the *same* tracks (same params → same cache entry, via
  `file_hash`); before this was centralized, each stage re-ran tracking independently and could disagree
  about what the tracked objects even were.
- **`tools/video.py`** — `load_frames` decodes any container (mp4/webm/mov/avi/mkv/...) via OpenCV/ffmpeg,
  and animated GIFs via Pillow separately (OpenCV's `VideoCapture` is unreliable on GIFs). All pipelines
  should decode through this rather than calling `cv2.VideoCapture` directly.
- **`tools/flow.py`** — shared keypoint detection/tracking primitives used by `tools/tracking.py` and the
  stage-1 optical-flow pipeline.
- **`tools/embeddings.py`** — DINOv2/CLIP/SigLIP embedding helpers, L2-normalized, batched, cached.
- **`tools/vlm.py`** — OpenRouter-based multi-frame VLM suspicion scoring (`OPENROUTER_MODELS` maps
  friendly keys to OpenRouter model ids; `build_suspicion_payload` builds the deterministic multi-frame
  request; `parse_vlm_json` robustly extracts JSON from fenced/prose-wrapped/plain responses). Requires an
  OpenRouter API key, passed as a pipeline setting.

Because every pipeline in a batch run operates on the same on-disk file, the track cache and evidence bus
are shared across the whole batch automatically — no extra wiring needed per pipeline.

### Adding a new pipeline

1. Add `backend/pipelines/stageN/my_test.py` exporting the `run()` async generator described above.
2. Import it and add an entry to `PIPELINES` in `backend/main.py` (id, name, desc, badge, dummy,
   requires_pair, settings, run).
3. Add the pipeline id to the relevant stage's `pipelines: [...]` list in `frontend/index.html`'s `STAGES`
   array — the frontend otherwise has no way to know which stage tab a pipeline belongs in (the pipeline
   list itself is auto-loaded from `GET /pipelines`).

### Frontend

`frontend/index.html` is a single self-contained file (~2,350 lines: HTML/CSS/JS inline, no build step,
no framework, no npm install). FastAPI mounts it as static files at `/`, so the backend serves both the API
and the UI on one port. When making frontend changes, edit this file directly.

### Remote/H100 access

If running the server on a remote GPU box, the frontend calls `http://localhost:8000`, so forward the port
over SSH (`ssh -L 8000:localhost:8000 <user>@<server>`) rather than changing the frontend's base URL.

### `backend/archive_files/`

Old flat (pre-4-stage) pipeline implementations, kept for reference only — not imported or registered
anywhere. Don't build on these; look at the equivalent `pipelines/stageN/` module instead.

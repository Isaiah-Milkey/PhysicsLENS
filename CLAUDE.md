# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PhysicsLENS is a web tool that evaluates physics accuracy in AI-generated video. It's structured as a
four-stage medical diagnostic pipeline (triage → localize → specialize → diagnose):

| Stage | Role | Cost |
|-------|------|------|
| 1 — Screening | Cheap signals flag suspicious regions (temporal smoothness, optical flow, embeddings, VLM, camera motion) | Cheap |
| 2 — Differential Diagnosis | Localize failures, extract trajectories, rank hypotheses | Medium |
| 3 — Specialist Evaluation | Confirm/reject specific physics-failure types (collision, gravity, momentum, friction, deformation, fluid, causality) | Expensive |
| 4 — Final Diagnosis | Aggregate score, severity, semantic timeline, LLM-written report | Output |

FastAPI backend (`backend/`) + a single self-contained static HTML frontend (`frontend/index.html`, no
build step, no npm). No test framework (pytest etc.) is configured — "tests" are standalone async scripts
under `backend/scripts/` run directly with `python`.

Every registered pipeline is now live — no `"dummy": True` entries remain in the registry (the flag and the
UI's STUB badge still exist for future stubs).

## Commands

```bash
# Setup (CPU-only; base deps)
conda create -n physicslens python=3.11 -y && conda activate physicslens
cd backend && pip install -r requirements.txt

# GPU stack (needed for SAM3 tracking, DINOv2/CLIP embeddings, local VLMs)
pip install -r requirements-gpu.txt
hf auth login   # or export HF_TOKEN=<token> — SAM3 (facebook/sam3) is gated; DINOv2 is not

# Run the server (serves API + static frontend together on one port)
./launch.sh                    # sources .env, activates the conda env, uvicorn on $PORT (default 8000)
# or manually:
cd backend && uvicorn main:app --reload --port 8000
# then open http://localhost:8000

# Sanity-check the GPU model stack (torch/CUDA, transformers SAM3/DINOv2 classes, gating)
python backend/scripts/check_models.py

# Run any pipeline headless, no server
python backend/scripts/run_pipeline.py <pipeline_id> <video_path> --set key=val
```

### Credentials (`.env` at the repo root, gitignored)

`launch.sh` and `tools/createai.py` (via python-dotenv) both load `<repo>/.env`:

- `CREATEAI_TOKEN` — ASU CreateAI proxy (Gemini/GPT); the default VLM provider for most pipelines.
- `CREATEAI_BASE_URL` — optional; `tools/createai.py` falls back to `https://api-main.aiml.asu.edu`.
- `OPENROUTER_API_KEY` — only for the OpenRouter-tagged models.
- Local VLMs (`qwen2.5-vl-7b`, `internvl3-*`, `smolvlm2-2.2b`) need **no** key — just a GPU.

Any key can also be typed into a pipeline's `api_key` settings field, which overrides `.env` for that run.
The frontend never persists keys (see the Frontend section).

### Running "tests"

There is no pytest suite. Each script under `backend/scripts/` is a standalone entry point — run it
directly with `python` from the repo root or `backend/`; each inserts the right path itself via
`sys.path.insert`. There is no way to run a "single test function" — run the whole script; some assert
internally, some just print a PASS/FAIL summary.

```bash
python backend/scripts/test_vlm_scoring.py      # pure unit tests: JSON parsing + payload building (no network)
python backend/scripts/test_vlm_pipeline.py     # integration test: full VLM suspicion pipeline (IO stubbed)
python backend/scripts/test_object_tracker.py   # smoke test: stage-2 object tracker against test_videos/ clips
python backend/scripts/sam3_smoke.py            # de-risk SAM3 alone: segment+track one concept in one video
python backend/scripts/createai_vision.py IMG   # manual probe of the CreateAI vision endpoint
python backend/scripts/vlm_failure_mode_eval.py # multi-frame vs single-frame AUC eval
python backend/scripts/vlm_multimodel_eval.py   # AI-vs-real AUC across local VLM families (source of the AUCs in the UI)
python backend/scripts/vlm_rapidata_eval.py     # agreement with Rapidata human Likert ratings on ~200 Sora clips
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

`settings` is a raw JSON string (parsed inside the pipeline, not by the caller). Event types actually
rendered today: `log`, `metric`, `severity`, `image`, `plotly`, `video`, `marker_video`, `signal`,
`result`, `llm_summary`, `timing`, `done`, `error`.

The authoritative renderer is `handleEvent()` in `frontend/index.html` (single-video/detail path) and
`applyEventToEntry()` (batch path, which persists events onto the run entry for export/benchmark). The
docstring at the top of `backend/main.py` and the README table are both older and omit several types —
trust `handleEvent()`. Adding a new event type means touching the pipeline **and both** frontend functions.

### Automatic cost instrumentation

Both `POST /run` and `POST /dataset/run` wrap the pipeline generator in `tools.costs.instrument(gen,
badge=...)`. It swallows the pipeline's `done`, appends a machine-readable `timing` event
(`duration_ms`, `gpu_mb`, `badge`) plus `Runtime` / `Peak GPU memory` / `Cost tier` metrics, then re-emits
`done`. Pipelines therefore should **not** hand-roll timing or GPU-memory metrics — they arrive free and
are what the frontend's benchmark table aggregates.

### Pipeline registration (`backend/main.py`)

`PIPELINES` is a single dict registry mapping pipeline id → `{id, name, desc, badge, dummy, requires_pair,
settings, run}`. This is the one place that wires a pipeline module into the app:
- `settings` describes the UI-editable knobs (number/text/select/password) for that pipeline; the frontend
  builds its settings form from this list, and the values round-trip back as the `settings` JSON string.
- Use the shared settings builders at the top of `main.py` rather than hand-writing model/key selects:
  `_vlm_model_setting(label, include_local=…, default=…)`, `_vlm_key_setting()`,
  `_createai_key_setting()` (CreateAI-only, e.g. the report's text LLM), `_naming_model_setting()`.
  This keeps every VLM pipeline on the same one-dropdown-plus-one-key convention.
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
contract as `/run`, but by `file_id`), `/dataset/file/{id}` (serves for in-browser preview),
`/dataset/clear` (drops the in-memory registry).

The frontend has one unified Dataset view (grid of videos → per-video report) — there is no separate
single-video mode; one video is just a batch of one.

### VLM access — three backends behind one router

- **`tools/vlm_router.py`** — the entry point pipelines should use. A dropdown value encodes provider +
  model (`createai:geminiflash2_5`, `openrouter:gpt-4o`, …); `resolve()` maps it, `key_status()` reports
  whether credentials exist so a pipeline can degrade cleanly, and `ask_vision()` /`ask_vision_json()` /
  `name_subjects()` are provider-agnostic. `DEFAULT_MODEL_KEY = "createai:geminiflash2_5"`.
- **`tools/createai.py`** — ASU CreateAI proxy client: `query_vision`, `query_text` (text-only, used by the
  Stage 4 LLM summary), `response_text`. Always sends an explicit `model` + `model_provider`; omitting the
  provider silently routes to a degraded default and some tokens 403 on the bare payload.
- **`tools/vlm.py`** — OpenRouter multi-frame suspicion scoring (`OPENROUTER_MODELS`,
  `build_suspicion_payload`, `parse_vlm_json` — the last is the shared robust JSON extractor used by all
  three backends).
- **`tools/vlm_local.py`** — local open-weight VLMs, no API key: `name_objects`, `score_video`,
  `analyze_physics`, plus token-probability scoring. Only one local model is resident on the GPU at a time
  (selecting another evicts the previous, 5–17 GB each). Measured on this benchmark, **bigger is worse**:
  Qwen2.5-VL-7B AUC 0.92 > InternVL3-8B 0.70 > InternVL3-14B 0.60 ≈ Qwen2.5-VL-32B 0.58 ≈
  SmolVLM2-2.2B 0.50. The 7B is the default judge; don't "upgrade" it without re-running
  `vlm_multimodel_eval.py`.

### Cross-stage shared state (`backend/tools/`)

This is what makes multi-stage evidence propagation work despite each pipeline's `run()` only receiving
`(video_path, settings)`:

- **`tools/evidence.py`** — `EvidenceStore` (LRU, thread-safe, in-process singleton `EVIDENCE`), keyed by
  `file_hash(video_path)` (content hash of size + head/tail bytes, memoized). Stage 2 writes structured
  results; Stage 3/4 read them back by the same video-content key. Entries are keyed by pipeline id
  (`s2_object_tracker`, `s3_gravity`, …). Evidence is cleared on server restart (intentionally ephemeral).
- **`tools/evidence_planner.py`** — Stage-3 pre-step. A specialist with an `auto_deps` setting can ask the
  planner to look at what's already on the bus and auto-run the missing Stage-2 producers inline
  (`mode="agent"`: one VLM call picks them; `mode="rules"`: deterministic dependency order — also the
  fallback when the agent has no credentials or misbehaves). Only the producers' `log` events are
  re-yielded, prefixed `[auto <id>]`. Currently wired into the gravity specialist.
- **`tools/tracking.py`** — `get_tracks(video_path)` is the canonical, cached Shi-Tomasi + LK object
  tracker. It exists so every downstream stage sees the *same* tracks (same params → same cache entry, via
  `file_hash`); before this was centralized, each stage re-ran tracking independently and could disagree
  about what the tracked objects even were.
- **`tools/sam3.py`** — SAM 3 promptable-concept video segmentation/tracking (gated `facebook/sam3`, GPU
  only). Process-wide singletons behind a load lock + a GPU lock, since FastAPI may call concurrently.
- **`tools/locate_anything.py`** — NVIDIA LocateAnything-3B open-set detection, single-image, matched onto
  existing tracks by IOU (NVIDIA non-commercial research license).
- **`tools/video.py`** — `load_frames` decodes any container (mp4/webm/mov/avi/mkv/...) via OpenCV/ffmpeg,
  and animated GIFs via Pillow separately (OpenCV's `VideoCapture` is unreliable on GIFs). All pipelines
  should decode through this rather than calling `cv2.VideoCapture` directly.
- **`tools/flow.py`** — shared keypoint detection/tracking primitives used by `tools/tracking.py` and the
  stage-1 optical-flow pipeline.
- **`tools/embeddings.py`** — DINOv2/CLIP/SigLIP embedding helpers, L2-normalized, batched, cached. Keep
  the DINOv2 extraction path aligned with `check_models.py` — the latent-kinematics thresholds are only
  meaningful in that space.

Because every pipeline in a batch run operates on the same on-disk file, the track cache and evidence bus
are shared across the whole batch automatically — no extra wiring needed per pipeline.

**Second channel — settings injected by the frontend.** There is no server-side orchestrator; the frontend
batch driver sequences stages and hands some data across as settings keys:
- `stage1_signals` → `s2_event_localizer` (the `signal` events emitted by Stage 1 tests for that video).
- `previous_results` + `video_name` → `s4_report` (every non-excluded prior test's rendered report).
So a pipeline reading these keys only receives them when driven through the UI, not from a bare
`/run` call — `scripts/run_pipeline.py` needs them passed explicitly.

### Stage 4's semantic timeline (what a new specialist must publish)

`_collect_semantic_findings()` in `pipelines/stage4/diagnostic_report.py` walks `SPECIALIST_DISPLAY` and
reads each specialist's bus entry to build the chronological "what went wrong, when, and why" timeline.
It understands these shapes:
- most specialists: `{"violations": [...]}`
- deformation: `{"verdicts": [...], "vanish_events": [...]}`
- fluid: `{"violations": [...], "holistic": {...}}`
- causality: `{"rules": [...]}` (whole-clip, `t` stays None)

Each violation/verdict is normalized by `_norm_finding()`, which looks for `t`, `t_end`, `label` or
`object_name`, `confidence`, `explanation` or `desc`, and a flagged/verdict field. A new specialist that
publishes a different shape — or doesn't publish to the bus at all — silently contributes nothing to the
report, even if its own UI panel looks fine. `use_llm_summary` then feeds this timeline to a CreateAI text
model and emits an `llm_summary` event.

### Adding a new pipeline

1. Add `backend/pipelines/stageN/my_test.py` exporting the `run()` async generator described above.
2. Import it and add an entry to `PIPELINES` in `backend/main.py` (id, name, desc, badge, dummy,
   requires_pair, settings, run) — use the shared VLM settings builders for model/key fields.
3. Add the pipeline id to the relevant stage's `pipelines: [...]` list in `frontend/index.html`'s `STAGES`
   array — the frontend otherwise has no way to know which stage tab a pipeline belongs in (the pipeline
   list itself is auto-loaded from `GET /pipelines`).
4. For a **Stage 3 specialist**, three more registries have to agree or it gets skipped by the automation:
   `SPECIALISTS` in `pipelines/stage2/physics_hypothesis_generator.py` (so triage can rank it),
   `SPECIALIST_PIPES` in `frontend/index.html` (so hypothesis→specialist routing can queue it), and
   `SPECIALIST_DISPLAY` in `pipelines/stage4/diagnostic_report.py` (so its findings reach the report).
   Pipeline ids are `s3_<specialist>`.

### Frontend

`frontend/index.html` is a single self-contained file (~3,140 lines: HTML/CSS/JS inline, no build step,
no framework, no npm install). FastAPI mounts it as static files at `/`, so the backend serves both the API
and the UI on one port. When making frontend changes, edit this file directly.

Beyond rendering events, it carries most of the batch orchestration:
- **Resumable batch driver** — `batchState` is a cursor over (video → tool queue); Pause/Abort both cancel
  the in-flight request via `AbortController`, but Pause keeps the cursor so Resume re-runs the cancelled
  tool and continues. Tools run in stage order so producers precede consumers (Hypothesis Generator before
  the specialists it routes to; `s4_report` last).
- **Per-tool batch settings** — `batchSettings[pipelineId]` overrides defaults per tool (the ⚙ gear on each
  checklist row). `"__"`-prefixed keys are frontend-only orchestration (`__route`, `__routeN` control
  hypothesis→specialist routing) and are stripped before hitting the backend.
- **Session-only API keys** — `batchKeys` lives in memory for the tab only; never localStorage, never
  exported. `redactSettings()` blanks anything key/token/secret-shaped before it is stored on a run entry
  or written to an export file. `applyBatchKeys()` picks the provider from the selected model.
- **Export / import** — `PL_EXPORT_SCHEMA` JSON of the whole batch (or one video), optionally including
  images/plots/marker videos; importable back into the Dataset view for offline inspection.
- **Benchmark modal** — cross-video table over severity, server-measured run time, or peak GPU (the latter
  two from the `timing` event), aggregated by stage/tool.

### Remote/H100 access

If running the server on a remote GPU box, the frontend calls `http://localhost:8000`, so forward the port
over SSH (`ssh -L 8000:localhost:8000 <user>@<server>`) rather than changing the frontend's base URL.
The README also documents an internal auto-deployed instance that tracks `main`.

### Reference-only code

- `backend/archive_files/` — old flat (pre-4-stage) pipeline implementations. Not imported or registered
  anywhere; don't build on these, look at the equivalent `pipelines/stageN/` module instead.
- `backend/pipelines/stage3/contact_specialist.py` — merged into `collision_specialist.py`, unregistered.
- `backend/pipelines/stage4/diagnostic_report_old.py` — previous report version, unregistered.

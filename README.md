# PhysicsLENS — Physics Diagnostic Pipeline

Web interface for evaluating physics accuracy in AI-generated video.  
Structured as a four-stage medical diagnostic workflow.

## Internal Hosting URL

Live internal instance — auto-deployed from `main`:

```
https://10.218.107.89:8000
```

> [!IMPORTANT]
> - On first visit your browser shows **"Your connection is not private"** because the TLS certificate is **self-signed**. Click **Advanced → Proceed to 10.218.107.89** — the traffic is still encrypted; the warning only means the cert isn't issued by a public authority. One click per machine.
> - You must be on the **ASU network / VPN** to reach this internal IP.
> - This site **auto-updates from `main`** — it checks GitHub every 5 minutes, so any pushed commit goes live automatically within ~5 minutes. No manual redeploy needed.

---

## Architecture

PhysicsLENS follows a **triage → localize → specialize → diagnose** pipeline,
analogous to a clinical pathway:

| Stage | Medical analogy | Role | Cost |
|-------|----------------|------|------|
| 1 — Screening | Triage / vital signs | Rapidly flag suspicious regions | Cheap |
| 2 — Differential Diagnosis | Primary care evaluation | Localize failures, rank hypotheses | Medium |
| 3 — Specialist Evaluation | Expert adjudication | Confirm/reject specific failures | Expensive |
| 4 — Final Diagnosis | Treatment plan | Score, severity, PBT, report | Output |

---

## Project structure

```
physicslens/
├── backend/
│   ├── main.py                      # FastAPI server + pipeline registry
│   ├── dataset_api.py               # Batch eval router: upload / HF download / run-by-id
│   ├── requirements.txt             # base / CPU-only deps (no torch)
│   ├── requirements-gpu.txt         # torch, transformers, SAM3, etc.
│   ├── pipelines/
│   │   ├── stage1/                  # Screening
│   │   │   ├── temporal_smoothness.py           ✅ verified
│   │   │   ├── optical_flow_irregularities.py   ✅ verified
│   │   │   ├── camera_motion.py                 🔵 implemented
│   │   │   ├── embedding_biomarkers.py          ✅ verified
│   │   │   └── vlm_suspicion.py                 ✅ verified
│   │   ├── stage2/                  # Failure localisation & hypothesis testing
│   │   │   ├── object_tracker.py                🔵 implemented (SAM3 subject masks + Gemini naming; LK fallback)
│   │   │   ├── event_localizer.py               🔵 implemented
│   │   │   ├── trajectory_extractor.py          🔵 implemented (reuses tracker masks; static-track filter)
│   │   │   └── physics_hypothesis_generator.py  🔵 implemented (VLM triage → ranks Stage 3 specialists;
│   │   │                                           absorbed the former hypothesis_ranker.py — removed)
│   │   ├── stage3/                  # Specialist evaluation (one file per failure type)
│   │   │   ├── deformation_specialist.py        🔵 implemented (DINOv2 drift detects shape/appearance
│   │   │   │                                       change + vanish; VLM explains — absorbed the former
│   │   │   │                                       consistency_specialist, removed)
│   │   │   ├── collision_specialist.py          🔵 implemented (contact episodes, restitution, phantom
│   │   │   │                                       bounces; absorbed contact_specialist — unregistered)
│   │   │   ├── gravity_specialist.py            🔶 stub
│   │   │   ├── momentum_specialist.py           🔵 implemented (motion signature + VLM mass proxy;
│   │   │   │                                       flags causeless momentum jumps & bad transfer)
│   │   │   ├── friction_specialist.py           🔵 implemented
│   │   │   ├── fluid_specialist.py              🔶 stub
│   │   │   └── causality_specialist.py          🔶 stub
│   │   └── stage4/                  # Final diagnosis outputs
│   │       ├── diagnostic_report.py             🔵 implemented (only Stage 4 pipeline —
│   │       │                                       physics_consistency_scorer.py, severity_assessor.py,
│   │       │                                       physics_breakdown_timer.py, and failure_explainer.py
│   │       │                                       were unimplemented stubs, removed)
│   │       └── diagnostic_report_old.py         (previous version, kept for reference)
│   ├── tools/                       # Shared utilities
│   │   ├── video.py
│   │   ├── flow.py
│   │   ├── tracking.py              # Cached Shi-Tomasi+LK tracks (one canonical set per video)
│   │   ├── evidence.py             # Cross-stage evidence bus (Stage 2→3→4 data passing)
│   │   ├── embeddings.py            # DINOv2 / CLIP / SigLIP — L2-normalised, batched, cached
│   │   ├── sam3.py                  # SAM3 video segmentation (gated facebook/sam3; GPU)
│   │   ├── createai.py              # ASU CreateAI client (Gemini vision; subject naming, judging)
│   │   ├── locate_anything.py       # NVIDIA LocateAnything-3B open-set detection (GPU, optional)
│   │   └── vlm.py                   # OpenRouter multi-frame suspicion scoring
│   ├── scripts/
│   │   ├── check_models.py          # Sanity-check ML model availability
│   │   ├── test_vlm_scoring.py      # Unit tests — VLM JSON parsing + payload build
│   │   ├── test_vlm_pipeline.py     # Integration test — VLM suspicion pipeline
│   │   ├── test_object_tracker.py   # Object-tracker smoke test
│   │   └── vlm_failure_mode_eval.py # multi-frame vs single-frame AUC eval (+ .json results)
│   └── archive_files/               # Old flat pipelines, kept for reference
├── frontend/
│   └── index.html                   # Self-contained UI — no build step
└── test_videos/                     # Real vs AI-generated clips (note: *.mp4 is gitignored)
    ├── README.md                    # describes the set + the matched real/AI demo pair
    ├── real/
    │   ├── physics_iq/              # real Physics-IQ benchmark footage
    │   └── wikimedia/               # short real clips
    └── ai_generated/               # text/image-to-video model output
```

| Status | Meaning |
|--------|---------|
| ✅ verified | Implemented and tested end-to-end on real videos |
| 🔵 implemented | Code complete, not yet fully validated |
| 🔶 stub | Placeholder — returns dummy output, not yet implemented |

> In the UI, stub pipelines show a yellow **STUB** badge and a dashed card border.
> Flip `"dummy": False` in `PIPELINES` (main.py) to promote a pipeline to live.

---

## Setup

### 1. Create / activate the conda environment

```bash
conda create -n physicslens python=3.11 -y
conda activate physicslens
cd backend
pip install -r requirements.txt
```

### 2. Start the server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open `http://localhost:8000` in a browser.  FastAPI also serves the
frontend statically, so no separate web server is needed.

### 3. Accessing from a remote server (e.g. H100)

The frontend calls `http://localhost:8000`.  Forward the port over SSH, then
open `http://localhost:8000` locally:

```bash
ssh -L 8000:localhost:8000 <user>@<server>
# on the server:
conda activate physicslens && cd backend && uvicorn main:app --reload --port 8000
```

### GPU pipelines

The object tracker and embedding pipelines benefit from a CUDA GPU.  Install
the extended ML stack:

```bash
pip install -r requirements-gpu.txt   # torch, transformers>=5.5, accelerate, …
```

The **SAM 3** model (`facebook/sam3`) is **gated** on HuggingFace.  Request
access on its model page, then authenticate once:

```bash
hf auth login          # or: export HF_TOKEN=<token>
```

**DINOv2** (`facebook/dinov2-base`) is Apache-2.0 — no gating required.

Sanity-check the model stack:

```bash
python scripts/check_models.py
```

---

## Adding a pipeline

### 1. Create the module

```python
# backend/pipelines/stageN/my_test.py
import asyncio, json
from typing import AsyncGenerator

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}

    yield {"type": "log", "level": "info", "text": "Starting…"}
    await asyncio.sleep(0)

    # --- your implementation here ---

    yield {"type": "metric",   "label": "Score",         "value": "0.85", "sub": "description"}
    yield {"type": "severity", "label": "Physics score",  "value": 85,     "color": "#1a7a3c"}
    yield {"type": "done"}
```

### 2. Register it in `main.py`

```python
from pipelines.stageN.my_test import run as run_my_test

PIPELINES["my_test"] = {
    "id":           "my_test",
    "name":         "My Test Name",
    "desc":         "Short description shown on the test card.",
    "badge":        "medium",       # cheap | medium | expensive | output
    "dummy":        False,          # True = STUB badge + dashed border in UI
    "requires_pair": False,
    "settings": [
        {"id": "threshold", "label": "Threshold", "type": "number",
         "default": 0.5, "min": 0.0, "max": 1.0},
    ],
    "run": run_my_test,
}
```

### 3. Add the pipeline ID to `STAGES` in `frontend/index.html`

```js
{ id: 'specialist', ..., pipelines: [..., 'my_test'] },
```

The UI auto-loads the pipeline list from `GET /pipelines` on startup.

---

## Event schema (pipeline → frontend)

| `type`     | Required keys                              | Notes                          |
|------------|--------------------------------------------|--------------------------------|
| `log`      | `level` (info/warn/error/success), `text`  | Appears in live log stream     |
| `metric`   | `label`, `value`, `sub`                    | `"PASS"` / `"FAIL"` colored   |
| `severity` | `label`, `value` (0–100), `color` (hex)    | Renders animated progress bar  |
| `image`    | `data` (base64), `mime`, `caption`         | Rendered inline                |
| `plotly`   | `data` (JSON string), `caption`            | Interactive Plotly chart       |
| `video`    | `data` (base64), `mime`, `caption`         | Inline video player            |
| `done`     | —                                          | Sets status to Done            |
| `error`    | `text`                                     | Logs error + sets status       |

---

## UI overview

The tool has a single unified **Dataset** view — there's no separate single-video
mode. You load **one or many** videos the same way; one video is just a batch of
one. The top-bar **Dataset** button is the home/back control: from a per-video
report it returns to the grid.

**Grid view** (home)

1. **Add videos** — either:
   - **Upload** — drag-drop videos of any format (mp4, webm, mov, gif, …), or click
     *choose a folder* to add a whole directory (non-video files are ignored).
   - **HuggingFace** — enter a dataset repo id (+ optional subfolder, max count, and a
     token for gated/private repos). The backend lists and downloads every video file
     in the repo, streaming progress as it goes.
2. **Select tests** — a multi-select checklist of the live (non-stub) pipelines, grouped by stage.
3. **Run batch** — each video runs the selected pipelines sequentially. A progress
   bar tracks `done / total` jobs. (Optional — you can also open a single video and
   run individual tests from its report without running a batch first.)

**Per-video report** (click any card)

Opens a full-screen stage view for that video — the same stage tabs, test list,
settings, run button, metrics, plots, overlay videos, and marker viewers used
throughout the tool. On open it **auto-loads every test that already ran**: result
dots mark which tests have data, the first completed stage/test is selected, and
its report is shown immediately. Any test can be re-run individually from here.

Each video card shows status and the worst severity across its tests. Because every
pipeline runs against the *same* on-disk file, the per-video track cache and evidence
bus are shared across the batch automatically. All of this is backed by the
`/dataset/*` API in `dataset_api.py`.

> **Any video format.** `tools/video.load_frames` decodes standard containers
> (mp4, webm, mov, avi, mkv, …) through OpenCV/ffmpeg, and animated GIFs through
> Pillow (OpenCV's `VideoCapture` is unreliable on GIFs, especially on Windows).
> Every pipeline accepts any of these.

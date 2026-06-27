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
│   │   │   ├── object_tracker.py                🔵 implemented
│   │   │   ├── event_localizer.py               🔵 implemented
│   │   │   ├── trajectory_extractor.py          🔶 stub
│   │   │   ├── physics_hypothesis_generator.py  🔶 stub
│   │   │   └── hypothesis_ranker.py             🔶 stub
│   │   ├── stage3/                  # Specialist evaluation (one file per failure type)
│   │   │   ├── collision_specialist.py          🔶 stub
│   │   │   ├── gravity_specialist.py            🔶 stub
│   │   │   ├── momentum_specialist.py           🔶 stub
│   │   │   ├── friction_specialist.py           🔶 stub
│   │   │   ├── deformation_specialist.py        🔶 stub
│   │   │   ├── contact_specialist.py            🔶 stub
│   │   │   ├── fluid_specialist.py              🔶 stub
│   │   │   └── causality_specialist.py          🔶 stub
│   │   └── stage4/                  # Final diagnosis outputs
│   │       ├── diagnostic_report.py             🔵 implemented
│   │       ├── diagnostic_report_old.py         (previous version, kept for reference)
│   │       ├── physics_consistency_scorer.py    🔶 stub
│   │       ├── severity_assessor.py             🔶 stub
│   │       ├── physics_breakdown_timer.py       🔶 stub
│   │       └── failure_explainer.py             🔶 stub
│   ├── tools/                       # Shared utilities
│   │   ├── video.py
│   │   ├── flow.py
│   │   ├── embeddings.py            # DINOv2 / CLIP / SigLIP — L2-normalised, batched, cached
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

- **Left pane** — stage selector, test list, settings, run button
- **Right pane → Report tab** — live output for the selected test; restores the last result when you switch between tests
- **Right pane → Previous Reports tab** — full run history for the selected test, collapsible per-run detail
- **Full Report button** (top right) — modal with a cross-test diagnostic report grouped by stage

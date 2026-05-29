"""
PhysicsLENS Physics Diagnostic — FastAPI Backend
------------------------------------------------
Pipeline types:
  - Single-video:  run(video_path: str)
  - Paired-video:  run(gt_path: str, ai_path: str)   requires_pair=True

To add a pipeline, create a function in pipelines/ and register it below.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipelines.screening         import run as run_screening
from pipelines.collision         import run as run_collision
from pipelines.full_diagnostic   import run as run_full
from pipelines.optical_flow_compare  import run as run_optical_flow
from pipelines.sparse_flow_compare   import run as run_sparse_flow

# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Keys understood by the frontend:
#   id, name, desc, badge         — display info
#   requires_pair: bool           — True = expects (gt_path, ai_path)
#   run: callable                 — the async generator function
#
# Event schema yielded by each pipeline:
#   {"type": "log",      "level": "info|warn|error|success", "text": "..."}
#   {"type": "metric",   "label": "...", "value": "...", "sub": "..."}
#   {"type": "severity", "label": "...", "value": 0-100, "color": "#hex"}
#   {"type": "image",    "data": "<base64>", "mime": "image/png", "caption": "..."}
#   {"type": "done"}
#   {"type": "error",    "text": "..."}
# ---------------------------------------------------------------------------

PIPELINES = {
    "optical_flow": {
        "id":            "optical_flow",
        "name":          "Optical flow comparison (dense)",
        "desc":          "Compares GT vs AI optical flow distributions using DIS dense flow: speed, direction, and flow derivatives.",
        "badge":         "L1",
        "requires_pair": True,
        "run":           run_optical_flow,
    },
    "sparse_flow": {
        "id":            "sparse_flow",
        "name":          "Sparse flow comparison (LK)",
        "desc":          "Tracks Shi-Tomasi corners with Lucas-Kanade across GT and AI videos. Shows coloured trajectory trails and compares flow distributions.",
        "badge":         "L1",
        "requires_pair": True,
        "run":           run_sparse_flow,
    },
    "screening": {
        "id":            "screening",
        "name":          "Dummy Screening test",
        "desc":          "FAKE VALUES - Cheap, broad anomaly detection. Optical flow, temporal smoothness, VLM suspicion score.",
        "badge":         "L1",
        "requires_pair": False,
        "run":           run_screening,
    },
    "collision": {
        "id":            "collision",
        "name":          "Dummy Collision specialist",
        "desc":          "FAKE VALUES - Impulse consistency, rebound angle, and momentum transfer checks.",
        "badge":         "L3",
        "requires_pair": False,
        "run":           run_collision,
    },
    "full_diagnostic": {
        "id":            "full_diagnostic",
        "name":          "Full diagnostic",
        "desc":          "All stages: screening → differential diagnosis → specialist tests → adjudication.",
        "badge":         "Full",
        "requires_pair": False,
        "run":           run_full,
    },
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PhysicsLENS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/pipelines")
def list_pipelines():
    return [
        {k: v for k, v in p.items() if k != "run"}
        for p in PIPELINES.values()
    ]


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.flush()
    return tmp.name


@app.post("/run")
async def run_pipeline(
    pipeline_id: str = Form(...),
    video:       UploadFile = File(...),
    video_ai:    Optional[UploadFile] = File(None),
):
    if pipeline_id not in PIPELINES:
        return {"error": f"Unknown pipeline: {pipeline_id}"}

    p = PIPELINES[pipeline_id]
    paths = []

    try:
        paths.append(_save_upload(video))
        if p["requires_pair"]:
            if video_ai is None:
                return {"error": "This pipeline requires both a GT and AI video."}
            paths.append(_save_upload(video_ai))

        pipeline_fn = p["run"]

        async def event_stream():
            try:
                if p["requires_pair"]:
                    gen = pipeline_fn(paths[0], paths[1])
                else:
                    gen = pipeline_fn(paths[0])
                async for event in gen:
                    yield json.dumps(event) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
            finally:
                for path in paths:
                    Path(path).unlink(missing_ok=True)

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    except Exception as exc:
        for path in paths:
            Path(path).unlink(missing_ok=True)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")

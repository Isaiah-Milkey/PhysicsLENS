# PhysicsLENS — Physics Diagnostic Pipeline

Web interface for evaluating physics accuracy in AI-generated video.

## Project structure

```
newtonbench/
├── backend/
│   ├── main.py               # FastAPI server — edit to add routes
│   ├── requirements.txt
│   └── pipelines/
│       ├── __init__.py
│       ├── screening.py       # Stage 1 — replace stubs with real code
│       ├── collision.py       # Stage 3 — collision specialist
│       └── full_diagnostic.py # All stages end-to-end
└── frontend/
    └── index.html            # Self-contained UI, no build step needed
```

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser. The UI fetches pipelines from
`http://localhost:8000` on load.

For production, FastAPI also serves the frontend — just run uvicorn and visit
`http://localhost:8000`.

---

## Adding a pipeline

1. Create `backend/pipelines/your_pipeline.py`:

```python
import asyncio
from typing import AsyncGenerator

async def run(video_path: str) -> AsyncGenerator[dict, None]:
    yield {"type": "log", "level": "info", "text": "Starting..."}
    await asyncio.sleep(0.5)

    # --- your real code here ---

    yield {"type": "metric", "label": "Score", "value": "0.85", "sub": "description"}
    yield {"type": "severity", "label": "Physics score", "value": 85, "color": "#1a7a3c"}
    yield {"type": "done"}
```

2. Register it in `backend/main.py`:

```python
from pipelines.your_pipeline import run as run_yours

PIPELINES["your_pipeline"] = {
    "id": "your_pipeline",
    "name": "Your pipeline name",
    "desc": "Short description shown in the UI.",
    "badge": "L2",
    "run": run_yours,
}
```

That's it — the UI auto-loads the pipeline list from `/pipelines` on startup.

---

## Event schema (pipeline → frontend)

| `type`     | Required keys                              | Notes                        |
|------------|--------------------------------------------|------------------------------|
| `log`      | `level` (info/warn/error/success), `text`  | Appears in log stream        |
| `metric`   | `label`, `value`, `sub`                    | Value "PASS"/"FAIL" colored  |
| `severity` | `label`, `value` (0–100), `color` (hex)    | Renders progress bar         |
| `done`     | —                                          | Sets status to Done          |
| `error`    | `text`                                     | Logs + sets status to Error  |

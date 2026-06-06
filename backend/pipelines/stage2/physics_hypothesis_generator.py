"""
Stage 2 · Step 4 — Physics Hypothesis Generator
--------------------------------------------------
For each candidate failure window (from event_localizer.py), maps trajectory
evidence to the most likely physics failure categories using cheap heuristics:

  Evidence pattern                          → Hypothesis
  ─────────────────────────────────────────────────────────
  Large accel spike near contact event      → collision / momentum
  Object accelerates upward freely          → gravity
  Sliding object fails to decelerate        → friction
  Two tracks' boxes overlap significantly   → collision / contact
  Bounding-box shape changes abruptly       → deformation
  Effect frame precedes cause frame         → causality / temporal drift

Each hypothesis is emitted with a raw evidence score (not yet ranked).
hypothesis_ranker.py converts these into a final sorted list.

Output schema (per hypothesis):
  {
    "window_id": int,
    "failure_type": str,     # one of the 7 failure categories
    "raw_score": float,      # 0–1, higher = stronger evidence
    "evidence": [str, ...]   # human-readable evidence bullets
  }
"""
import asyncio, json
from typing import AsyncGenerator

FAILURE_CATEGORIES = [
    "collision",
    "gravity",
    "momentum",
    "friction",
    "deformation",
    "contact",
    "causality",
]


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg               = json.loads(settings) if settings else {}
    accel_spike_thresh = float(cfg.get("accel_spike_threshold", 4.0))   # std-devs
    overlap_thresh     = float(cfg.get("overlap_threshold", 0.05))      # IoU fraction
    candidate_windows  = cfg.get("candidate_windows", [])

    yield {"type": "log", "level": "info", "text": "Physics Hypothesis Generator — dummy mode"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "accel_spike_threshold": accel_spike_thresh,
        "overlap_threshold": overlap_thresh,
        "windows_processed": len(candidate_windows),
        "hypotheses": [],   # list of hypothesis dicts described above
        "note": "Replace with real heuristic evidence → hypothesis mapping.",
    }

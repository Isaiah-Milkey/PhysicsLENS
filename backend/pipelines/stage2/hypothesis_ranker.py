"""
Stage 2 · Step 5 — Hypothesis Ranker
---------------------------------------
Aggregates all raw hypotheses from physics_hypothesis_generator.py, deduplicates
overlapping windows for the same failure type, and produces a final ranked list
of candidate failures sorted by confidence score.

This ranked list is the sole output of Stage 2.  Stage 3 specialist modules are
invoked in rank order, stopping once a diagnosis is confirmed or the confidence
of remaining hypotheses falls below a threshold.

Output schema:
  [
    {"failure": "collision", "score": 0.82, "time_seconds": 12.4, "window_id": 3},
    {"failure": "momentum",  "score": 0.61, "time_seconds": 12.4, "window_id": 3},
    {"failure": "gravity",   "score": 0.34, "time_seconds":  8.1, "window_id": 1},
    ...
  ]
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg              = json.loads(settings) if settings else {}
    min_score        = float(cfg.get("min_score", 0.20))     # prune below this
    max_hypotheses   = int(cfg.get("max_hypotheses", 5))     # cap list length
    raw_hypotheses   = cfg.get("hypotheses", [])             # from hypothesis_generator

    yield {"type": "log", "level": "info", "text": "Hypothesis Ranker — dummy mode"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "min_score": min_score,
        "max_hypotheses": max_hypotheses,
        "input_hypotheses": len(raw_hypotheses),
        "ranked_hypotheses": [],   # sorted list described above
        "note": "Replace with real deduplication, aggregation, and ranking logic.",
    }

"""
Stage 3 · Specialist — Momentum Specialist
--------------------------------------------
Called when Stage 2 hypothesises a momentum discontinuity.
Estimates object masses via bounding-box area or depth cues and checks whether
linear and angular momentum are conserved across each interaction event within
expected tolerances.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    momentum_tol  = float(cfg.get("momentum_tolerance", 0.15))
    angular_tol   = float(cfg.get("angular_tolerance", 0.20))
    target_events = cfg.get("target_events", [])

    yield {"type": "log", "level": "info", "text": "Momentum Specialist — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "momentum_tolerance": momentum_tol,
        "angular_tolerance": angular_tol,
        "target_events": target_events,
        "conservation_violations": [],
        "verdict": None,
        "note": "Replace with real momentum specialist analysis.",
    }

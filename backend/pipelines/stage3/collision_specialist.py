"""
Stage 3 · Specialist — Collision Specialist
---------------------------------------------
Called when Stage 2 hypothesises a collision inconsistency.
Verifies approach/exit velocities satisfy restitution bounds, checks contact
normals for geometric plausibility, and confirms no interpenetration during
or after impact.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    restitution_range = cfg.get("restitution_range", [0.0, 1.0])
    target_events     = cfg.get("target_events", [])

    yield {"type": "log", "level": "info", "text": "Collision Specialist — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "restitution_range": restitution_range,
        "target_events": target_events,
        "verdict": None,
        "evidence": [],
        "note": "Replace with real collision specialist analysis.",
    }

"""
Stage 3 · Specialist — Gravity Specialist
-------------------------------------------
Called when Stage 2 hypothesises a gravity violation.
Fits projectile trajectories to each free-falling object and compares the
inferred gravitational acceleration against the expected value.  Also checks
for objects that rise without an applied force.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    g_expected   = float(cfg.get("g_expected", 9.81))   # m/s² (scene-scale normalised)
    g_tolerance  = float(cfg.get("g_tolerance", 0.20))  # fractional deviation allowed
    target_objects = cfg.get("target_objects", [])

    yield {"type": "log", "level": "info", "text": "Gravity Specialist — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "g_expected": g_expected,
        "g_tolerance": g_tolerance,
        "target_objects": target_objects,
        "inferred_g_per_object": {},
        "verdict": None,
        "note": "Replace with real gravity specialist analysis.",
    }

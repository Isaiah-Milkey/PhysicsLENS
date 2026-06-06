"""
Stage 3 · Specialist — Contact Specialist
-------------------------------------------
Called when Stage 2 hypothesises contact instability.
Detects object interpenetration, phantom contacts, and sudden separation events
that violate non-penetration constraints.  Also checks that surface normals at
contact points are consistent with rigid-body contact geometry.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    overlap_thresh  = float(cfg.get("overlap_threshold", 0.02))   # fraction of object area
    target_pairs    = cfg.get("target_pairs", [])                  # [(obj_a_id, obj_b_id), ...]

    yield {"type": "log", "level": "info", "text": "Contact Specialist — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "overlap_threshold": overlap_thresh,
        "target_pairs": target_pairs,
        "contact_violations": [],
        "verdict": None,
        "note": "Replace with real contact specialist analysis.",
    }

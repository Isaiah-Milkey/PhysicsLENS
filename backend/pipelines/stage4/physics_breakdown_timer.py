"""
Stage 4 · Output 3 — Physics Breakdown Timer (PBT)
----------------------------------------------------
Identify the precise frame (or frame range) at which the video first deviates
from physical law — the Physics Breakdown Time (PBT).
Uses the anomaly onset signals from Stages 1–3 and applies a change-point
detection algorithm to pinpoint the earliest statistically significant break.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    change_point_method = cfg.get("method", "cusum")   # "cusum" | "bocpd" | "pelt"
    min_segment_frames  = int(cfg.get("min_segment_frames", 5))

    yield {"type": "log", "level": "info", "text": "Physics Breakdown Timer — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "method": change_point_method,
        "min_segment_frames": min_segment_frames,
        "pbt_frame": None,
        "pbt_seconds": None,
        "confidence": None,
        "note": "Replace with real change-point detection and PBT estimation implementation.",
    }

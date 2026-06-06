"""
Screening pipeline — Stage 1
-----------------------------
Replace the stub logic below with your real implementation.
Yield dicts that the frontend understands (see main.py for the event schema).
"""

import asyncio
from typing import AsyncGenerator


async def run(video_path: str) -> AsyncGenerator[dict, None]:
    # -----------------------------------------------------------------------
    # STUB — replace with real screening logic
    # e.g. optical flow, temporal smoothness, VLM suspicion scoring
    # -----------------------------------------------------------------------

    yield {"type": "log", "level": "info",    "text": f"Loading video: {video_path}"}
    await asyncio.sleep(0.4)

    yield {"type": "log", "level": "info",    "text": "Extracting optical flow..."}
    await asyncio.sleep(0.8)

    yield {"type": "log", "level": "info",    "text": "Computing temporal smoothness..."}
    await asyncio.sleep(0.6)

    yield {"type": "log", "level": "info",    "text": "Running VLM suspicion scoring..."}
    await asyncio.sleep(1.0)

    yield {"type": "log", "level": "success", "text": "Screening complete."}

    # Metrics — replace values with real results
    yield {"type": "metric", "label": "Suspicion score",   "value": "0.61", "sub": "threshold 0.5"}
    yield {"type": "metric", "label": "Anomalous frames",  "value": "14",   "sub": "of 120 total"}
    yield {"type": "metric", "label": "Optical flow δ",    "value": "2.3",  "sub": "avg px/frame"}

    yield {"type": "severity", "label": "Suspicion level", "value": 61, "color": "#EF9F27"}
    yield {"type": "done"}

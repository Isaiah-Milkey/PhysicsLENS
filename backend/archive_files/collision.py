"""
Collision specialist pipeline — Stage 3
----------------------------------------
Replace the stub logic below with your real implementation.
"""

import asyncio
from typing import AsyncGenerator


async def run(video_path: str) -> AsyncGenerator[dict, None]:
    yield {"type": "log", "level": "info",    "text": f"Loading video: {video_path}"}
    await asyncio.sleep(0.4)

    yield {"type": "log", "level": "info",    "text": "Detecting collision events..."}
    await asyncio.sleep(0.8)

    yield {"type": "log", "level": "info",    "text": "Checking impulse consistency..."}
    await asyncio.sleep(0.6)

    yield {"type": "log", "level": "warn",    "text": "Rebound angle anomaly at frame 23"}
    await asyncio.sleep(0.3)

    yield {"type": "log", "level": "warn",    "text": "Momentum transfer inconsistency detected"}
    await asyncio.sleep(0.3)

    yield {"type": "log", "level": "success", "text": "Collision analysis complete."}

    yield {"type": "metric", "label": "Rebound angle",      "value": "FAIL", "sub": "expected ~42°, got 71°"}
    yield {"type": "metric", "label": "Momentum transfer",  "value": "FAIL", "sub": "energy not conserved"}
    yield {"type": "metric", "label": "Contact timing",     "value": "PASS", "sub": "within tolerance"}

    yield {"type": "severity", "label": "Physics breakdown time (frame)", "value": 83, "color": "#E24B4A"}
    yield {"type": "done"}

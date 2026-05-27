"""
Full diagnostic pipeline — All stages
--------------------------------------
Replace the stub logic below with your real implementation.
"""

import asyncio
from typing import AsyncGenerator


async def run(video_path: str) -> AsyncGenerator[dict, None]:
    yield {"type": "log", "level": "info",    "text": f"Loading video: {video_path}"}
    await asyncio.sleep(0.3)

    yield {"type": "log", "level": "info",    "text": "[Stage 1] Running screening tests..."}
    await asyncio.sleep(0.9)

    yield {"type": "log", "level": "info",    "text": "[Stage 2] Differential diagnosis: collision inconsistency (0.82)"}
    await asyncio.sleep(0.7)

    yield {"type": "log", "level": "info",    "text": "[Stage 3] Running collision specialist tests..."}
    await asyncio.sleep(0.8)

    yield {"type": "log", "level": "warn",    "text": "Rebound angle: FAIL. Momentum transfer: FAIL."}
    await asyncio.sleep(0.4)

    yield {"type": "log", "level": "info",    "text": "[Stage 4] Requesting VLM adjudication..."}
    await asyncio.sleep(1.0)

    yield {"type": "log", "level": "info",    "text": 'VLM: "Ball loses too much height after first bounce."'}
    await asyncio.sleep(0.4)

    yield {"type": "log", "level": "success", "text": "[Stage 5] Diagnosis complete."}

    yield {"type": "metric", "label": "Consistency score", "value": "0.31", "sub": "low = bad physics"}
    yield {"type": "metric", "label": "Severity",          "value": "HIGH", "sub": "energy loss in bounce"}
    yield {"type": "metric", "label": "Confidence",        "value": "0.91", "sub": "PBT: frame 23"}

    yield {"type": "severity", "label": "Overall physics score", "value": 31, "color": "#E24B4A"}
    yield {"type": "done"}

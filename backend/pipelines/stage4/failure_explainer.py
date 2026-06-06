"""
Stage 4 · Output 4 — Failure Explainer
-----------------------------------------
Generate a human-readable, structured explanation of the detected physics
failure.  Combines evidence summaries from prior stages with a templated
narrative and, optionally, a VLM-generated free-text description.
Output is suitable for display in a diagnostic report UI.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    use_vlm       = bool(cfg.get("use_vlm", False))
    affected_objs = cfg.get("affected_objects", [])

    yield {"type": "log", "level": "info", "text": "Failure Explainer — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "use_vlm": use_vlm,
        "affected_objects": affected_objs,
        "failure_type": None,
        "explanation": None,
        "supporting_evidence": [],
        "note": "Replace with real explanation generation implementation.",
    }

"""
Stage 4 · Output 2 — Severity Assessor
-----------------------------------------
Map the confirmed failure type and confidence onto a severity scale:
  • Critical   — clear, high-confidence physics violation
  • Moderate   — probable violation, moderate confidence
  • Minor      — possible artefact, low confidence
  • Inconclusive — insufficient evidence
Also estimates the spatial extent (fraction of frame affected) and temporal
duration of the anomaly.
"""
import asyncio, json
from typing import AsyncGenerator


SEVERITY_LEVELS = ["critical", "moderate", "minor", "inconclusive"]


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    confidence_thresholds = cfg.get(
        "confidence_thresholds",
        {"critical": 0.85, "moderate": 0.60, "minor": 0.35},
    )

    yield {"type": "log", "level": "info", "text": "Severity Assessor — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "confidence_thresholds": confidence_thresholds,
        "severity": None,
        "spatial_extent": None,
        "temporal_duration_frames": None,
        "note": "Replace with real severity mapping and extent estimation implementation.",
    }

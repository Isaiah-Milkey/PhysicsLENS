"""
Stage 3 · Specialist — Fluid Specialist
-----------------------------------------
Called when Stage 2 hypothesises a fluid dynamics anomaly.
Analyses flow patterns of liquids or gases for violations of continuity,
incompressibility, or viscosity constraints.  Uses dense optical flow on fluid
regions and checks for unphysical sources, sinks, or discontinuous splashes.
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    fluid_regions  = cfg.get("fluid_regions", [])     # bounding boxes of fluid areas
    viscosity_mode = cfg.get("viscosity_mode", "low") # "low" | "high" (water vs. syrup)

    yield {"type": "log", "level": "info", "text": "Fluid Specialist — dummy mode (not yet implemented)"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "fluid_regions": fluid_regions,
        "viscosity_mode": viscosity_mode,
        "flow_anomalies": [],
        "verdict": None,
        "note": "Replace with real fluid dynamics specialist analysis.",
    }

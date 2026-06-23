"""
Stage 3 · Water Specialist — Mass / Area Conservation
-----------------------------------------------------
Segment the water region each frame and track its area (a 2-D volume proxy).
Physical water volume changes smoothly (inflow/outflow); discontinuous jumps
mean water popped into or out of existence — a generative artifact. Steady
monotonic drift (a filling tank) is tolerated; abrupt frame-to-frame jumps
are flagged.
"""
import asyncio, json
from typing import AsyncGenerator, List

import numpy as np

from tools.video import load_frames
from tools.fluid import water_mask, severity_color, timeseries_figure


def analyze(frames: List[np.ndarray], fps: float, cfg: dict) -> dict:
    mask_method = cfg.get("mask_method", "auto")
    jump_threshold = float(cfg.get("jump_threshold", 0.20))  # frac change/frame
    areas = []
    for f in frames:
        mask, _ = water_mask(f, method=mask_method)
        areas.append(float(mask.mean()))  # fraction of frame that is water

    flagged, signals, deltas = [], [], [0.0]
    for i in range(1, len(areas)):
        denom = max(areas[i - 1], 1e-4)
        d = abs(areas[i] - areas[i - 1]) / denom
        deltas.append(d)
        if d > jump_threshold and max(areas[i], areas[i - 1]) > 0.01:
            flagged.append(i)
            signals.append({"frame": int(i), "signal_type": "mass_discontinuity",
                            "score": round(float(d / max(jump_threshold, 1e-6)), 3)})

    n = max(len(frames), 1)
    severity = min(int(len(flagged) / n * 250), 100)  # *250 saturates: ~40% flagged frames -> severity 100
    time = [i / fps for i in range(len(frames))]
    return {
        "time": time,
        "series": {"water area fraction": areas, "|Δarea| rate": deltas},
        "flagged": flagged,
        "severity": severity,
        "color": severity_color(severity),
        "signals": signals,
        "metrics": [
            {"label": "Area jumps", "value": str(len(flagged)), "sub": "discontinuous mass change"},
            {"label": "Peak |Δarea| rate", "value": f"{max(deltas, default=0.0):.2f}", "sub": "per frame"},
            {"label": "Mean water area", "value": f"{np.mean(areas) if areas else 0:.3f}", "sub": "frame fraction"},
        ],
        "summary": (f"{len(flagged)} mass-discontinuity event(s) detected."
                    if flagged else "Water mass is conserved (no sudden jumps)."),
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps — tracking water area…"}
    await asyncio.sleep(0)

    r = analyze(frames, fps, cfg)

    yield {"type": "plotly",
           "data": timeseries_figure(
               r["time"],
               [("water area fraction", r["series"]["water area fraction"], "#1a54c4"),
                ("|Δarea| rate", r["series"]["|Δarea| rate"], "#E24B4A")],
               "Mass / area conservation",
               threshold=float(cfg.get("jump_threshold", 0.20)),
               ythresh_label="jump threshold"),
           "caption": "Blue: water area over time. Red: frame-to-frame change rate (jumps = mass artifacts)."}
    yield {"type": "signal", "source": "s3_water_mass_conservation",
           "source_name": "Water Mass Conservation", "fps": float(fps),
           "n_frames": int(len(frames)), "severity": r["severity"], "signals": r["signals"]}
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "Mass conservation violation score",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "warn" if r["flagged"] else "success", "text": r["summary"]}
    yield {"type": "done"}

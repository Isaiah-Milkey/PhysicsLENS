"""
Stage 3 · Water Specialist — Incompressibility (Divergence)
-----------------------------------------------------------
Dense optical flow → Helmholtz divergence ∇·v inside the water region,
normalized by flow magnitude. Real water is ~incompressible (∇·v ≈ 0); large
persistent divergence means water is being created (source) or destroyed
(sink) — a hallmark generative artifact.
"""
import asyncio, json
from typing import AsyncGenerator, List, Optional

import numpy as np

from tools.video import load_frames
from tools.fluid import (compute_flow_sequence, masked_mean, severity_color,
                         timeseries_figure)


def analyze(frames: List[np.ndarray], fps: float, cfg: dict,
            flow_seq: Optional[list] = None) -> dict:
    backend = cfg.get("backend", "auto")
    mask_method = cfg.get("mask_method", "auto")
    # Calibrated on real water (IMG_7513): the optical-flow divergence noise
    # floor on genuine moving water is ~0.11 (max ~0.17) at 480p, so 0.25 gives
    # ~1.5x margin above real-water false positives. (One-sided: kills false
    # alarms on real footage; catching true fakes needs an AI clip to confirm.)
    threshold = float(cfg.get("divergence_threshold", 0.25))  # normalized |∇·v|/|v|
    if flow_seq is None:
        flow_seq = compute_flow_sequence(frames, backend=backend, mask_method=mask_method)

    series, flagged, signals = [], [], []
    for i, s in enumerate(flow_seq, start=1):
        mean_div = masked_mean(np.abs(s["div"]), s["mask"])
        mean_mag = masked_mean(s["mag"], s["mask"])
        norm = mean_div / (mean_mag + 1e-6)
        series.append(norm)
        if norm > threshold and mean_mag > 0.3:
            flagged.append(i)
            signals.append({"frame": int(i), "signal_type": "fluid_divergence",
                            "score": round(float(norm / max(threshold, 1e-6)), 3)})

    n = max(len(flow_seq), 1)
    severity = min(int(len(flagged) / n * 250), 100)  # *250 saturates: ~40% flagged frames -> severity 100
    peak = float(max(series, default=0.0))
    mean = float(np.mean(series) if series else 0.0)
    time = [i / fps for i in range(1, len(flow_seq) + 1)]
    return {
        "time": time,
        "series": {"normalized |∇·v|": series},
        "flagged": flagged,
        "severity": severity,
        "color": severity_color(severity),
        "signals": signals,
        "metrics": [
            {"label": "Flagged frames", "value": str(len(flagged)), "sub": "∇·v above threshold"},
            {"label": "Peak |∇·v|/|v|", "value": f"{peak:.3f}", "sub": "normalized divergence"},
            {"label": "Mean |∇·v|/|v|", "value": f"{mean:.3f}", "sub": "0 = incompressible"},
        ],
        "summary": (f"{len(flagged)} source/sink frame(s) detected."
                    if flagged else "Flow is incompressible; no sources/sinks."),
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps — computing divergence…"}
    await asyncio.sleep(0)

    r = analyze(frames, fps, cfg)

    yield {"type": "plotly",
           "data": timeseries_figure(
               r["time"], [("normalized |∇·v|", r["series"]["normalized |∇·v|"], "#1a54c4")],
               "Incompressibility — normalized divergence",
               threshold=float(cfg.get("divergence_threshold", 0.25)),
               ythresh_label="divergence threshold"),
           "caption": "Normalized |∇·v| inside the water region. Spikes = sources/sinks."}
    yield {"type": "signal", "source": "s3_water_incompressibility",
           "source_name": "Water Incompressibility", "fps": float(fps),
           "n_frames": int(len(frames)), "severity": r["severity"], "signals": r["signals"]}
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "Incompressibility violation score",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "warn" if r["flagged"] else "success", "text": r["summary"]}
    yield {"type": "done"}

"""
Stage 3 · Water Specialist — Vorticity / Turbulence
---------------------------------------------------
Curl ∇×v inside the water region measures swirl/eddy structure. Real moving
water carries a characteristic band of vorticity; generated water is often
implausibly smooth (laminar sheets with motion but no swirl) or implausibly
chaotic. Flags frames whose normalized vorticity falls outside a plausible
band while the water is actually moving.
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
    vmin = float(cfg.get("min_vorticity", 0.02))   # normalized lower band
    vmax = float(cfg.get("max_vorticity", 0.80))   # normalized upper band
    require_motion = bool(cfg.get("require_motion", True))
    motion_floor = float(cfg.get("motion_floor", 0.3))  # px/frame in mask
    if flow_seq is None:
        flow_seq = compute_flow_sequence(frames, backend=backend, mask_method=mask_method)

    series, flagged, signals = [], [], []
    for i, s in enumerate(flow_seq, start=1):
        mean_curl = masked_mean(np.abs(s["curl"]), s["mask"])
        mean_mag = masked_mean(s["mag"], s["mask"])
        norm = mean_curl / (mean_mag + 1e-6)
        series.append(norm)
        moving = mean_mag > motion_floor
        if (not require_motion or moving) and (norm < vmin or norm > vmax):
            flagged.append(i)
            dev = (vmin - norm) if norm < vmin else (norm - vmax)
            signals.append({"frame": int(i), "signal_type": "vorticity_anomaly",
                            "score": round(float(abs(dev) / max(vmax, 1e-6)), 3)})

    n = max(len(flow_seq), 1)
    severity = min(int(len(flagged) / n * 250), 100)  # *250 saturates: ~40% flagged frames -> severity 100
    time = [i / fps for i in range(1, len(flow_seq) + 1)]
    return {
        "time": time,
        "series": {"normalized |∇×v|": series},
        "flagged": flagged,
        "severity": severity,
        "color": severity_color(severity),
        "signals": signals,
        "metrics": [
            {"label": "Out-of-band frames", "value": str(len(flagged)), "sub": f"outside [{vmin}, {vmax}]"},
            {"label": "Mean |∇×v|/|v|", "value": f"{np.mean(series) if series else 0:.3f}", "sub": "normalized vorticity"},
            {"label": "Peak |∇×v|/|v|", "value": f"{max(series, default=0.0):.3f}", "sub": "swirl strength"},
        ],
        "summary": (f"{len(flagged)} implausible-vorticity frame(s) detected."
                    if flagged else "Vorticity within a plausible band."),
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps — computing vorticity…"}
    await asyncio.sleep(0)

    r = analyze(frames, fps, cfg)

    yield {"type": "plotly",
           "data": timeseries_figure(
               r["time"], [("normalized |∇×v|", r["series"]["normalized |∇×v|"], "#7c3aed")],
               "Vorticity / turbulence",
               threshold=float(cfg.get("max_vorticity", 0.80)),
               ythresh_label="upper band"),
           "caption": "Normalized |∇×v| inside the water region; flagged when outside the plausible band."}
    yield {"type": "signal", "source": "s3_water_vorticity",
           "source_name": "Water Vorticity", "fps": float(fps),
           "n_frames": int(len(frames)), "severity": r["severity"], "signals": r["signals"]}
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "Vorticity anomaly score",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "warn" if r["flagged"] else "success", "text": r["summary"]}
    yield {"type": "done"}

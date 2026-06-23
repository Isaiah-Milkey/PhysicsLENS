"""
Stage 3 · Water Specialist — Surface & Splash Coherence
-------------------------------------------------------
Warp the previous frame by its optical flow and compare the prediction to the
actual current frame inside the water region (normalized cross-correlation).
Real foam/spray/surface texture *advects with the flow*, so the warp predicts
the next frame well. Generated water that flickers in place (texture re-drawn
each frame instead of carried by motion) yields low correlation.
"""
import asyncio, json
from typing import AsyncGenerator, List, Optional

import cv2
import numpy as np

from tools.video import load_frames
from tools.fluid import compute_flow_sequence, severity_color, timeseries_figure


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() < 16:
        return 1.0  # too little water to judge → treat as coherent
    av, bv = a[m].astype(np.float64), b[m].astype(np.float64)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    if denom < 1e-6:
        return 1.0
    return float((av * bv).sum() / denom)


def analyze(frames: List[np.ndarray], fps: float, cfg: dict,
            flow_seq: Optional[list] = None) -> dict:
    backend = cfg.get("backend", "auto")
    mask_method = cfg.get("mask_method", "auto")
    corr_floor = float(cfg.get("coherence_floor", 0.35))  # min plausible NCC
    if flow_seq is None:
        flow_seq = compute_flow_sequence(frames, backend=backend, mask_method=mask_method)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    h, w = grays[0].shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

    series, flagged, signals = [], [], []
    for i, s in enumerate(flow_seq, start=1):
        map_x = (gx + s["u"]).astype(np.float32)
        map_y = (gy + s["v"]).astype(np.float32)
        warped = cv2.remap(grays[i - 1], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        corr = _ncc(warped, grays[i], s["mask"])
        series.append(corr)
        if corr < corr_floor:
            flagged.append(i)
            signals.append({"frame": int(i), "signal_type": "surface_incoherence",
                            "score": round(float(corr_floor - corr), 3)})

    n = max(len(flow_seq), 1)
    severity = min(int(len(flagged) / n * 250), 100)
    time = [i / fps for i in range(1, len(flow_seq) + 1)]
    return {
        "time": time,
        "series": {"advection NCC": series},
        "flagged": flagged,
        "severity": severity,
        "color": severity_color(severity),
        "signals": signals,
        "metrics": [
            {"label": "Incoherent frames", "value": str(len(flagged)), "sub": f"NCC < {corr_floor}"},
            {"label": "Mean advection NCC", "value": f"{np.mean(series) if series else 0:.2f}", "sub": "1 = texture moves with flow"},
            {"label": "Min advection NCC", "value": f"{min(series, default=1.0):.2f}", "sub": "worst frame"},
        ],
        "summary": (f"{len(flagged)} flicker/incoherent surface frame(s) detected."
                    if flagged else "Surface texture advects coherently with the flow."),
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps — checking surface advection…"}
    await asyncio.sleep(0)

    r = analyze(frames, fps, cfg)

    yield {"type": "plotly",
           "data": timeseries_figure(
               r["time"], [("advection NCC", r["series"]["advection NCC"], "#1a7a3c")],
               "Surface & splash coherence",
               threshold=float(cfg.get("coherence_floor", 0.35)),
               ythresh_label="coherence floor"),
           "caption": "Normalized cross-correlation of flow-warped vs actual frame; low = flicker-in-place."}
    yield {"type": "signal", "source": "s3_water_surface_coherence",
           "source_name": "Water Surface Coherence", "fps": float(fps),
           "n_frames": int(len(frames)), "severity": r["severity"], "signals": r["signals"]}
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "Surface incoherence score",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "warn" if r["flagged"] else "success", "text": r["summary"]}
    yield {"type": "done"}

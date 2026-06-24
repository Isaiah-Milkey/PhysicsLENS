"""
Stage 3 · Water Specialist — Impact / Splash Dynamics
-----------------------------------------------------
A genuine drop/impact into water produces a *sharp, brief* motion impulse — the
splash spikes then decays. Generated water tends to spread the same motion
smoothly across time (no crisp causal impact), so the temporal profile of fluid
motion is "smeared". This measures the impulse sharpness of the masked flow
magnitude (peak / median). When there IS motion but it lacks a sharp impulse,
the event looks temporally implausible.

Calibrated on the Egg-Drop set (one real + two AI of the same scene): impact
impulse was ~46 for the real clip vs ~14–16 for Kling/Veo, so a floor of 25
separates them. This is event-conditional (assumes the clip contains an impact)
and calibrated on n=1 scene — a provisional discriminator, not a validated one.
"""
import asyncio, json
from typing import AsyncGenerator, List, Optional

import numpy as np

from tools.video import load_frames
from tools.fluid import (compute_flow_sequence, resize_frames, masked_mean,
                         severity_color, timeseries_figure)


def analyze(frames: List[np.ndarray], fps: float, cfg: dict,
            flow_seq: Optional[list] = None) -> dict:
    backend = cfg.get("backend", "auto")
    mask_method = cfg.get("mask_method", "auto")
    min_impulse = float(cfg.get("min_impulse", 25.0))     # peak/median floor
    motion_floor = float(cfg.get("impact_motion_floor", 0.5))  # need a real event
    if flow_seq is None:
        flow_seq = compute_flow_sequence(frames, backend=backend, mask_method=mask_method)

    fm = [masked_mean(s["mag"], s["mask"]) for s in flow_seq]
    n = len(fm)
    arr = np.array(fm, dtype=np.float64) if n else np.array([0.0])
    peak = float(arr.max())
    median = float(np.median(arr))
    impulse = peak / (median + 1e-6)
    has_event = peak > motion_floor

    flagged, signals, severity = [], [], 0
    if has_event and impulse < min_impulse:
        deficit = (min_impulse - impulse) / max(min_impulse, 1e-6)
        severity = min(int(deficit * 200), 100)
        peak_frame = int(arr.argmax()) + 1
        flagged = [peak_frame]
        signals.append({"frame": peak_frame, "signal_type": "weak_impact_impulse",
                        "score": round(float(deficit), 3)})

    if severity > 0:
        summary = (f"Fluid motion lacks a sharp impact impulse (peak/median {impulse:.0f} "
                   f"< {min_impulse:.0f}) — motion is temporally smeared (AI-like).")
    elif has_event:
        summary = f"Sharp impact impulse ({impulse:.0f}) consistent with a real splash/drop."
    else:
        summary = "No clear impact event in the water region (impulse test inconclusive)."

    return {
        "time": [i / fps for i in range(1, n + 1)],
        "series": {"fluid motion (px/frame)": fm},
        "flagged": flagged,
        "severity": severity,
        "color": severity_color(severity),
        "signals": signals,
        "metrics": [
            {"label": "Impact impulse", "value": f"{impulse:.1f}", "sub": f"peak/median (≥ {min_impulse:.0f} plausible)"},
            {"label": "Peak fluid motion", "value": f"{peak:.2f}", "sub": "px/frame in water region"},
            {"label": "Median fluid motion", "value": f"{median:.3f}", "sub": "px/frame"},
        ],
        "summary": summary,
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    frames = resize_frames(frames, int(cfg.get("max_height", 480)))
    yield {"type": "log", "level": "info", "text": "Measuring fluid impact dynamics…"}
    await asyncio.sleep(0)

    r = analyze(frames, fps, cfg)
    yield {"type": "plotly",
           "data": timeseries_figure(
               r["time"], [("fluid motion (px/frame)", r["series"]["fluid motion (px/frame)"], "#c05621")],
               "Impact / splash dynamics — fluid motion over time"),
           "caption": r["summary"]}
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "Weak-impact (smeared-motion) score",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "warn" if r["severity"] else "success", "text": r["summary"]}
    yield {"type": "done"}

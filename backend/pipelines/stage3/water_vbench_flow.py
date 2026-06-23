"""
Stage 3 · SOTA Comparator — VBench-style Flow Metric
----------------------------------------------------
Represents the VBench family of temporal metrics: whole-frame motion
smoothness + flicker via dense optical flow. Deliberately NOT water-aware and
produces a single coarse plausibility score — included to contrast with the
grounded, localized water tests.
"""
import asyncio, json
from typing import AsyncGenerator, List, Optional

import cv2
import numpy as np

from tools.video import load_frames
from tools.fluid import compute_flow_sequence, severity_color


def analyze(frames: List[np.ndarray], fps: float, cfg: dict,
            flow_seq: Optional[list] = None) -> dict:
    backend = cfg.get("backend", "auto")
    if flow_seq is None:
        flow_seq = compute_flow_sequence(frames, backend=backend, mask_method="hsv")

    # Motion smoothness: 1 - normalized change between consecutive flow fields.
    smooth_terms = []
    for a, b in zip(flow_seq[:-1], flow_seq[1:]):
        da = np.abs(b["u"] - a["u"]) + np.abs(b["v"] - a["v"])
        ref = np.abs(a["u"]) + np.abs(a["v"]) + 1.0
        smooth_terms.append(float((da / ref).mean()))
    motion_instability = float(np.mean(smooth_terms)) if smooth_terms else 0.0

    # Flicker: mean abs intensity change in low-motion pixels.
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    flick = []
    for i, s in enumerate(flow_seq, start=1):
        low_motion = s["mag"] < 0.5
        if low_motion.any():
            diff = np.abs(grays[i] - grays[i - 1])
            flick.append(float(diff[low_motion].mean()))
    flicker = float(np.mean(flick)) if flick else 0.0

    score = min(int(motion_instability * 120 + flicker * 1.5), 100)
    return {
        "severity": score,
        "color": severity_color(score),
        "metrics": [
            {"label": "Motion instability", "value": f"{motion_instability:.3f}", "sub": "1 - smoothness"},
            {"label": "Flicker", "value": f"{flicker:.2f}", "sub": "static-pixel intensity jitter"},
        ],
        "details": {"motion_instability": motion_instability, "flicker": flicker},
        "summary": f"VBench-style coarse plausibility score: {score}/100.",
    }


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info", "text": "Computing VBench-style flow metrics…"}
    await asyncio.sleep(0)
    r = analyze(frames, fps, cfg)
    for m in r["metrics"]:
        yield {"type": "metric", **m}
    yield {"type": "severity", "label": "VBench-style implausibility (coarse)",
           "value": r["severity"], "color": r["color"]}
    yield {"type": "log", "level": "info", "text": r["summary"]}
    yield {"type": "done"}

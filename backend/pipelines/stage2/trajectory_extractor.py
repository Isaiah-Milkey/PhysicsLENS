"""
Stage 2 · Step 2 — Trajectory Extractor
------------------------------------------
Converts raw bounding-box tracks (from object_tracker.py) into higher-level
kinematic descriptors used by the hypothesis generator:

  • Position (centre-of-mass per frame)
  • Velocity  (first finite difference, smoothed)
  • Acceleration (second finite difference)
  • Contact timing (frames where two tracks' boxes overlap or touch)
  • Angular velocity proxy (box aspect-ratio change rate)

Output schema (per track):
  {
    "track_id": int,
    "positions":     [[x, y], ...],
    "velocities":    [[vx, vy], ...],
    "accelerations": [[ax, ay], ...],
    "contact_events": [{"frame": int, "with_track": int}, ...]
  }
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg           = json.loads(settings) if settings else {}
    smooth_window = int(cfg.get("smoothing_window", 5))   # Savitzky-Golay window
    fps           = float(cfg.get("fps", 30.0))

    yield {"type": "log", "level": "info", "text": "Trajectory Extractor — dummy mode"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "smoothing_window": smooth_window,
        "fps": fps,
        "trajectories": [],   # list of trajectory dicts described above
        "note": "Replace with real kinematics extraction from track bounding boxes.",
    }

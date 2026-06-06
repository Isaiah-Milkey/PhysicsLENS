"""
Stage 2 · Step 3 — Event Localizer
--------------------------------------
Uses Stage 1 suspicious timestamps to crop the video timeline into candidate
failure windows.  For each anomaly flagged by Stage 1, it:

  1. Expands the timestamp into a ±context_frames window.
  2. Identifies which tracks are active in that window.
  3. Labels the window with the Stage 1 signal type that triggered it
     (e.g. "flow_spike", "embedding_jump", "temporal_anomaly").

These windows are the *search space* passed to physics_hypothesis_generator.py —
only suspicious regions are analysed in depth, keeping Stage 2 cost moderate.

Output schema (per window):
  {
    "window_id": int,
    "frame_start": int,
    "frame_end": int,
    "trigger": str,          # Stage 1 signal that created this window
    "active_tracks": [int, ...]
  }
"""
import asyncio, json
from typing import AsyncGenerator


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg            = json.loads(settings) if settings else {}
    context_frames = int(cfg.get("context_frames", 15))   # frames before/after anomaly
    stage1_signals = cfg.get("stage1_signals", [])        # list of {frame, signal_type}

    yield {"type": "log", "level": "info", "text": "Event Localizer — dummy mode"}

    await asyncio.sleep(0.1)

    yield {
        "type": "result",
        "status": "dummy",
        "context_frames": context_frames,
        "stage1_signals_received": len(stage1_signals),
        "candidate_windows": [],   # list of window dicts described above
        "note": "Replace with real Stage 1 signal → failure window extraction.",
    }

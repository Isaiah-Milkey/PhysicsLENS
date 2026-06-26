"""
Stage 1 · Test 4 — VLM Suspicion Score
----------------------------------------
Sample keyframes and ask a VLM (via OpenRouter) whether the MOTION across the
video looks physically suspicious. All frames are sent in a single multi-frame
call and the model returns one holistic verdict — physics violations live in how
things move between frames, not in any single still. This separates AI-generated
from real video far better than independent per-frame scoring
(scripts/vlm_failure_mode_eval.py: AUC 0.90 vs 0.58). Decoding is deterministic
(temperature 0 + fixed seed) so repeated runs on the same video agree.

Without an API key the pipeline runs in demo mode (random placeholder score).
Set the key via the settings panel or the OPENROUTER_API_KEY env var.
"""
import asyncio, json, os
from typing import AsyncGenerator

import numpy as np
import plotly.graph_objects as go

from tools.video import load_frames, sample_frames
from tools.vlm   import score_frames, OPENROUTER_MODELS


def _zone_color(score: float) -> str:
    return "#E24B4A" if score > 0.5 else "#EF9F27" if score > 0.25 else "#4CAF50"


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    model_key  = cfg.get("model", "gpt-4o")
    num_frames = max(2, int(cfg.get("num_frames", 8)))
    api_key    = cfg.get("api_key", "").strip() or os.environ.get("OPENROUTER_API_KEY", "")
    demo_mode  = not api_key

    if demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "No API key — running in demo mode (placeholder score)."}
        yield {"type": "log", "level": "info",
               "text": "Set OPENROUTER_API_KEY or enter it in test settings to use a real model."}
    else:
        yield {"type": "log", "level": "info",
               "text": f"Using model: {OPENROUTER_MODELS.get(model_key, model_key)}"}

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n == 0:
        yield {"type": "error", "text": "Could not read any frames from the video."}
        return

    keyframes = sample_frames(frames, num_frames)
    duration  = n / fps if fps else 0.0

    yield {"type": "log", "level": "info",
           "text": f"Scoring {len(keyframes)} frames from {n} total in one multi-frame pass…"}
    await asyncio.sleep(0)

    if demo_mode:
        await asyncio.sleep(0.4)
        result = {
            "suspicion_score":   float(np.random.beta(2, 5)),
            "suspected_failure": None,
            "explanation":       "Demo mode — no API key provided.",
            "confidence":        0.0,
        }
    else:
        try:
            result = await score_frames(keyframes, model_key=model_key, api_key=api_key)
        except Exception as exc:
            yield {"type": "log", "level": "error", "text": f"VLM call failed: {exc}"}
            result = {"suspicion_score": None, "suspected_failure": None,
                      "explanation": str(exc), "confidence": 0.0}

    raw_score   = result.get("suspicion_score")
    parsed_ok   = isinstance(raw_score, (int, float))
    score       = min(max(float(raw_score), 0.0), 1.0) if parsed_ok else 0.0
    label       = (result.get("suspected_failure") or "").strip()
    if label.lower() in ("", "null", "none"):
        label = "—"
    explanation = result.get("explanation", "")
    confidence  = float(result.get("confidence", 0) or 0)

    if not parsed_ok and not demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "Model did not return a usable score — reporting 0 (see explanation)."}

    level = "warn" if score > 0.5 else "info"
    yield {"type": "log", "level": level,
           "text": f"Verdict: score={score:.2f}  [{label}]  {explanation}"}

    # ── Gauge: one holistic suspicion score ─────────────────────────────────────
    color = _zone_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(valueformat=".2f", font=dict(size=42)),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor="#888",
                      tickvals=[0, 0.25, 0.5, 0.75, 1]),
            bar=dict(color=color, thickness=0.3),
            borderwidth=0,
            steps=[
                dict(range=[0,    0.25], color="#E8F5E9"),
                dict(range=[0.25, 0.5 ], color="#FFF3E0"),
                dict(range=[0.5,  1.0 ], color="#FFEBEE"),
            ],
            threshold=dict(line=dict(color="#E24B4A", width=3), thickness=0.85, value=0.5),
        ),
        title=dict(
            text=f"VLM Suspicion — {model_key}" + ("  [DEMO]" if demo_mode else ""),
            font=dict(size=15),
        ),
    ))
    fig.update_layout(
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=20),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": (f"Holistic suspicion over {len(keyframes)} frames "
                    f"({duration:.1f}s of video). "
                    "Red ≥ 0.5 (suspicious), amber ≥ 0.25 (uncertain), green < 0.25 (plausible)."),
    }

    severity = min(int(score * 100), 100)
    yield {"type": "metric", "label": "Suspicion score",   "value": f"{score:.2f}",     "sub": "0–1 (whole video)"}
    yield {"type": "metric", "label": "Suspected failure",  "value": label,              "sub": "VLM label"}
    yield {"type": "metric", "label": "Model confidence",   "value": f"{confidence:.0%}", "sub": "self-reported"}
    yield {"type": "severity", "label": "VLM suspicion level", "value": severity, "color": color}

    if explanation and not demo_mode:
        yield {"type": "log", "level": "info", "text": f"Explanation: {explanation}"}
    if demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "Result is a demo placeholder. Provide an OpenRouter API key for real scoring."}
    yield {"type": "done"}

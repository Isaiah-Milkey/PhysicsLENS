"""
Stage 1 · Test 4 — VLM Suspicion Scores
-----------------------------------------
Sample keyframes and ask a VLM (via OpenRouter) whether the interaction
looks physically suspicious.

Without an API key the pipeline runs in demo mode (random placeholder scores).
Set the key via the settings panel or the OPENROUTER_API_KEY env var.
"""
import asyncio, json, os
from typing import AsyncGenerator

import numpy as np
import plotly.graph_objects as go

from tools.video import load_frames, sample_frames
from tools.vlm   import score_frame, OPENROUTER_MODELS


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    model_key  = cfg.get("model", "gpt-4o")
    num_frames = max(1, int(cfg.get("num_frames", 5)))
    api_key    = cfg.get("api_key", "").strip() or os.environ.get("OPENROUTER_API_KEY", "")
    demo_mode  = not api_key

    if demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "No API key — running in demo mode (placeholder scores)."}
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
    kf_times  = [i / max(len(keyframes) - 1, 1) * (n / fps) for i in range(len(keyframes))]

    yield {"type": "log", "level": "info",
           "text": f"Scoring {len(keyframes)} keyframes from {n} total…"}
    await asyncio.sleep(0)

    scores, labels, explanations, confidences = [], [], [], []

    for i, (frame, t) in enumerate(zip(keyframes, kf_times)):
        yield {"type": "log", "level": "info",
               "text": f"Scoring keyframe {i+1}/{len(keyframes)}  (t ≈ {t:.1f}s)…"}
        await asyncio.sleep(0)

        if demo_mode:
            await asyncio.sleep(0.25)
            result = {
                "suspicion_score":   float(np.random.beta(2, 5)),
                "suspected_failure": None,
                "time_interval":     f"t ≈ {t:.1f}s",
                "explanation":       "Demo mode — no API key provided.",
                "confidence":        0.0,
            }
        else:
            try:
                result = await score_frame(frame, model_key=model_key, api_key=api_key)
            except Exception as exc:
                yield {"type": "log", "level": "error", "text": f"Frame {i+1} error: {exc}"}
                result = {
                    "suspicion_score": 0.0, "suspected_failure": None,
                    "explanation": str(exc), "confidence": 0.0,
                }

        s = float(result.get("suspicion_score", 0))
        scores.append(s)
        labels.append(result.get("suspected_failure") or "—")
        explanations.append(result.get("explanation", ""))
        confidences.append(float(result.get("confidence", 0)))

        level = "warn" if s > 0.5 else "info"
        yield {"type": "log", "level": level,
               "text": f"  score={s:.2f}  [{labels[-1]}]  {explanations[-1]}"}

    # ── Plotly bar chart ──────────────────────────────────────────────────────
    bar_colors = [
        "#E24B4A" if s > 0.5 else "#EF9F27" if s > 0.25 else "#4CAF50"
        for s in scores
    ]
    x_labels = [f"t={t:.1f}s" for t in kf_times]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels, y=scores,
        marker=dict(color=bar_colors, opacity=0.85, line=dict(color="white", width=1.5)),
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        customdata=[[lb, ex, f"{cf:.0%}"] for lb, ex, cf in zip(labels, explanations, confidences)],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Suspicion: <b>%{y:.2f}</b><br>"
            "Label: %{customdata[0]}<br>"
            "Explanation: %{customdata[1]}<br>"
            "Confidence: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0.5,  line=dict(color="#E24B4A", dash="dash", width=1.5),
                  annotation_text="suspicious ≥ 0.5",
                  annotation_font=dict(color="#E24B4A", size=11),
                  annotation_position="top right")
    fig.add_hline(y=0.25, line=dict(color="#EF9F27", dash="dot",  width=1.2),
                  annotation_text="uncertain ≥ 0.25",
                  annotation_font=dict(color="#EF9F27", size=11),
                  annotation_position="bottom right")

    # Shade the suspicious zone
    fig.add_hrect(y0=0.5, y1=1.05, fillcolor="#E24B4A", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0.25, y1=0.5, fillcolor="#EF9F27", opacity=0.05, line_width=0)

    fig.update_layout(
        title=dict(
            text=f"VLM Suspicion Scores — {model_key}" + ("  [DEMO]" if demo_mode else ""),
            font=dict(size=15),
        ),
        xaxis=dict(title="Keyframe", showgrid=False),
        yaxis=dict(title="Suspicion score (0–1)", range=[0, 1.15],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=80, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        showlegend=False,
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": "Per-keyframe suspicion score. Red ≥ 0.5 (suspicious), amber ≥ 0.25 (uncertain), green < 0.25 (plausible).",
    }

    mean_score = float(np.mean(scores)) if scores else 0.0
    max_score  = float(max(scores))     if scores else 0.0
    n_sus      = sum(1 for s in scores if s > 0.5)
    severity   = min(int(mean_score * 100), 100)
    color      = "#E24B4A" if severity > 50 else "#EF9F27" if severity > 25 else "#4CAF50"

    yield {"type": "metric", "label": "Mean suspicion",    "value": f"{mean_score:.2f}", "sub": "0–1"}
    yield {"type": "metric", "label": "Peak suspicion",    "value": f"{max_score:.2f}",  "sub": "0–1"}
    yield {"type": "metric", "label": "Suspicious frames", "value": str(n_sus),          "sub": f"of {len(keyframes)} sampled (>0.5)"}
    yield {"type": "severity", "label": "VLM suspicion level", "value": severity, "color": color}

    if demo_mode:
        yield {"type": "log", "level": "warn",
               "text": "Results are demo placeholders. Provide an OpenRouter API key for real scoring."}
    yield {"type": "done"}

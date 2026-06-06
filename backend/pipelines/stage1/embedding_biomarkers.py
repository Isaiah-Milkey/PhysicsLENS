"""
Stage 1 · Test 3 — Embedding Biomarkers
-----------------------------------------
Encode each sampled frame with DINOv2 / CLIP / SigLIP (whole-frame embedding).
Compute latent velocity (‖Δz‖) and acceleration (‖Δ²z‖).
Spikes reveal hidden visual or physics discontinuities.
"""
import asyncio, json
from typing import AsyncGenerator

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    model_key    = cfg.get("model", "dinov2")
    sample_every = max(1, int(cfg.get("sample_every", 5)))
    acc_thresh   = float(cfg.get("accel_threshold", 0.5))

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path, step=sample_every)
    n = len(frames)
    if n < 3:
        yield {"type": "error",
               "text": f"Too few sampled frames ({n}). Reduce 'sample_every' or use a longer video."}
        return

    yield {"type": "log", "level": "info",
           "text": f"{n} frames sampled (every {sample_every}) — loading {model_key}…"}
    await asyncio.sleep(0)

    try:
        if model_key == "dinov2":
            from tools.embeddings import load_dinov2, embed_frames_dinov2
            model = load_dinov2()
            yield {"type": "log", "level": "info", "text": "Encoding whole-frame embeddings with DINOv2…"}
            await asyncio.sleep(0)
            embs = embed_frames_dinov2(frames, model)
        elif model_key == "clip":
            from tools.embeddings import load_clip, embed_frames_clip
            model, preprocess = load_clip()
            yield {"type": "log", "level": "info", "text": "Encoding whole-frame embeddings with CLIP…"}
            await asyncio.sleep(0)
            embs = embed_frames_clip(frames, model, preprocess)
        elif model_key == "siglip":
            from tools.embeddings import load_siglip, embed_frames_siglip
            model, processor = load_siglip()
            yield {"type": "log", "level": "info", "text": "Encoding whole-frame embeddings with SigLIP…"}
            await asyncio.sleep(0)
            embs = embed_frames_siglip(frames, model, processor)
        else:
            yield {"type": "error", "text": f"Unknown model '{model_key}'."}
            return
    except Exception as exc:
        yield {"type": "error", "text": f"Could not load {model_key}: {exc}"}
        return

    yield {"type": "log", "level": "info",
           "text": f"Embeddings: {embs.shape} — computing latent dynamics…"}
    await asyncio.sleep(0)

    vel      = np.diff(embs, axis=0)
    acc      = np.diff(vel,  axis=0)
    vel_norm = np.linalg.norm(vel, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)

    time_ax = np.arange(n) * sample_every / fps
    flagged = set(np.where(acc_norm > acc_thresh)[0] + 2)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        subplot_titles=(
            f"Latent Velocity  ‖Δz‖  ({model_key.upper()})",
            "Latent Acceleration  ‖Δ²z‖ — anomalies highlighted",
        ),
    )

    fig.add_trace(go.Scatter(
        x=time_ax[1:].tolist(), y=vel_norm.tolist(),
        mode="lines", name="‖Δz‖ (latent vel)",
        line=dict(color="#1a54c4", width=1.6),
        fill="tozeroy", fillcolor="rgba(26,84,196,0.09)",
        hovertemplate="<b>t = %{x:.2f}s</b><br>‖Δz‖ = %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time_ax[2:].tolist(), y=acc_norm.tolist(),
        mode="lines", name="‖Δ²z‖ (latent acc)",
        line=dict(color="#c05621", width=1.6),
        hovertemplate="<b>t = %{x:.2f}s</b><br>‖Δ²z‖ = %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(
        y=acc_thresh,
        line=dict(color="#E24B4A", dash="dash", width=1.5),
        annotation_text=f"threshold = {acc_thresh:.3f}",
        annotation_font=dict(color="#E24B4A", size=11),
        annotation_position="top right",
        row=2, col=1,
    )

    for fi in sorted(flagged):
        if fi < len(time_ax):
            hw = sample_every / fps * 0.6
            fig.add_vrect(
                x0=float(time_ax[fi] - hw), x1=float(time_ax[fi] + hw),
                fillcolor="#E24B4A", opacity=0.18, line_width=0,
                row=2, col=1,
            )

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="‖Δz‖",    row=1, col=1)
    fig.update_yaxes(title_text="‖Δ²z‖",   row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"Embedding Biomarkers — {model_key.upper()} Whole-Frame Latent Trajectory",
            font=dict(size=15),
        ),
        height=540,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=45, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        hovermode="x unified",
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": "Latent velocity (top) and acceleration (bottom). Red shaded regions = frames with abrupt embedding shifts.",
    }

    n_flagged = len(flagged)
    score     = min(int(n_flagged / max(n, 1) * 300), 100)
    color     = "#E24B4A" if score > 40 else "#EF9F27" if score > 15 else "#4CAF50"

    yield {"type": "metric", "label": "Flagged frames",  "value": str(n_flagged),            "sub": f"of {n} sampled"}
    yield {"type": "metric", "label": "Peak latent acc", "value": f"{acc_norm.max():.4f}",   "sub": "‖Δ²z‖"}
    yield {"type": "metric", "label": "Mean latent vel", "value": f"{vel_norm.mean():.4f}",  "sub": "‖Δz‖"}
    yield {"type": "metric", "label": "Embedding dim",   "value": str(embs.shape[1]),         "sub": model_key}
    yield {"type": "severity", "label": "Embedding anomaly score", "value": score, "color": color}

    msg = ("No significant latent discontinuities detected." if not flagged
           else f"{n_flagged} frame(s) with abrupt embedding shifts.")
    yield {"type": "log", "level": "success" if not flagged else "warn", "text": msg}
    yield {"type": "done"}

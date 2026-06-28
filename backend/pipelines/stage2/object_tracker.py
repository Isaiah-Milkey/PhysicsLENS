"""
Stage 2 · Step 1 — Object Tracker
------------------------------------
Detect salient moving objects with Shi-Tomasi corners + Lucas-Kanade tracking.
Cluster nearby keypoints into persistent object tracks.
Optionally embed each track's bounding-box crops with DINOv2 (from archive
sam_track_compare.py) to measure appearance drift — an identity-stability
biomarker.  A physically consistent object keeps a near-flat drift curve;
flickering / morphing AI objects spike.

Metrics
-------
  • Tracked objects count
  • Keypoint loss rate
  • Mean / peak track duration
  • Per-track mean & peak cosine drift  (DINOv2)
  • Overall tracking instability score  (0–100)

GPU not required — falls back gracefully if DINOv2 / torch are absent.
"""
import asyncio, base64, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video    import encode_video_browser
from tools.tracking import get_tracks, _PALETTE, _hex_to_bgr
from tools.evidence import EVIDENCE, video_id


def _render_annotated(
    frames: list, obj_tracks: list[dict], max_frames: int = 240
) -> tuple[list, int]:
    """Draw each track's box + 'obj N' label (legend colors) on every frame.
    Returns (annotated frames, subsample step)."""
    by_frame: dict[int, list] = {}
    for ct in obj_tracks:
        for f, box in zip(ct["frames"], ct["boxes"]):
            by_frame.setdefault(f, []).append((ct["id"], box))

    n    = len(frames)
    step = max(1, -(-n // max_frames))   # ceil division
    out  = []
    for f in range(0, n, step):
        img = frames[f].copy()
        for tid, (x0, y0, x1, y1) in by_frame.get(f, []):
            color = _hex_to_bgr(_PALETTE[tid % len(_PALETTE)])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            label = f"obj {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = y0 - 6 if y0 - th - 10 >= 0 else min(y1 + th + 6, img.shape[0] - 4)
            cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), color, -1)
            cv2.putText(img, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(img)
    return out, step


# ── DINOv2 appearance drift (ported from sam_track_compare.py) ───────────────

def _embed_crops_dinov2(crops: list[np.ndarray], model=None) -> np.ndarray:
    """
    L2-normalised DINOv2 (facebook/dinov2-base) descriptors for a list of
    BGR uint8 crop arrays.  Ported from sam_track_compare._embed_track —
    same checkpoint, so the 0.35 drift threshold keeps its original meaning.
    Returns (K, 768) float32.  The loader is cached, so per-track calls
    reuse one model instance.
    """
    from tools.embeddings import load_dinov2, embed_frames_dinov2
    if model is None:
        model = load_dinov2()
    embs  = embed_frames_dinov2(crops, model)       # (K, D)
    # Ensure L2 normalisation (safe double-norm)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.where(norms < 1e-8, 1.0, norms)
    return embs


def _drift_curve(embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine-distance drift relative to the first embedding.
    Ported directly from sam_track_compare._drift_curve:
      drift[t] = 1 − <e_t , e_0>  ∈ [0, 2]
    Flat curve = stable identity.  Spike = morphing / appearance jump.
    """
    ref = embeddings[0]
    return 1.0 - embeddings @ ref


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    num_kp       = max(5,  int(cfg.get("num_keypoints", 60)))
    sample_every = max(1,  int(cfg.get("sample_every",   1)))
    use_dinov2   = str(cfg.get("use_dinov2", "true")).lower() not in ("false", "0", "no")
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")

    yield {"type": "log", "level": "info", "text": "Loading video & tracking keypoints…"}
    await asyncio.sleep(0)

    # ── Phase 1+2: shared, cached LK tracking + clustering ────────────────────
    loop = asyncio.get_event_loop()
    tr = await loop.run_in_executor(
        None, lambda: get_tracks(video_path, num_kp=num_kp,
                                 sample_every=sample_every, max_objects=8))
    frames     = tr["frames"]
    eff_fps    = tr["fps"]
    meta       = tr["meta"]
    obj_tracks = tr["tracks"]
    n  = meta["n_frames"]
    H, W = meta["H"], meta["W"]
    nk = meta["nk"]
    kp_loss_pct = meta["kp_loss_pct"]
    kp_lost     = int(round(kp_loss_pct * nk))

    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    if not obj_tracks:
        yield {"type": "error", "text": "No keypoints detected in any frame."}
        return
    if meta["start"] > 0:
        yield {"type": "log", "level": "warn",
               "text": f"First {meta['start']} frame(s) featureless — tracking started later."}

    n_objects = len(obj_tracks)
    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {eff_fps:.1f} fps — {n_objects} object track(s) "
                   f"from {nk} keypoints."}
    await asyncio.sleep(0)

    # Publish canonical tracks to the evidence bus for downstream stages.
    EVIDENCE.put(video_id(video_path), "s2_object_tracker", {
        "fps": float(eff_fps), "n_frames": int(n),
        "tracks": [{"id": ct["id"], "frames": ct["frames"], "boxes": ct["boxes"],
                    "cx": ct["cx"], "cy": ct["cy"], "n_kp": ct["n_kp"]}
                   for ct in obj_tracks],
    })

    # ── Phase 2b: Annotated video — show which region each "obj N" is ─────────
    if render_video and n_objects > 0:
        yield {"type": "log", "level": "info", "text": "Rendering labeled object video…"}
        await asyncio.sleep(0)
        try:
            ann_frames, step = _render_annotated(frames, obj_tracks)
            data, mime = await loop.run_in_executor(
                None, encode_video_browser, ann_frames, eff_fps / step
            )
            yield {
                "type": "video",
                "data": base64.b64encode(data).decode(),
                "mime": mime,
                "caption": ("Tracked objects — box colors and labels match the "
                            "plot legend (obj N below)."),
            }
            yield {"type": "log", "level": "info",
                   "text": f"Labeled video ready ({len(data)/1024:.0f} KB, {mime})."}
        except Exception as exc:
            yield {"type": "log", "level": "warn",
                   "text": f"Labeled video rendering failed ({exc}) — continuing."}
        await asyncio.sleep(0)

    # ── Phase 3: DINOv2 appearance drift (ported from sam_track_compare) ──────
    dino_drifts: dict[int, np.ndarray] = {}
    dino_available = False

    if use_dinov2 and n_objects > 0:
        yield {"type": "log", "level": "info",
               "text": "Loading DINOv2 for appearance-drift analysis (sam_track_compare approach)…"}
        await asyncio.sleep(0)
        try:
            from tools.embeddings import load_dinov2
            dino_model = load_dinov2()
            for ct in obj_tracks:
                crops = []
                for f, box in zip(ct["frames"], ct["boxes"]):
                    x0, y0, x1, y1 = box
                    crop = frames[f][y0:y1, x0:x1]
                    if crop.size > 0:
                        crops.append(crop)

                if len(crops) < 2:
                    continue

                embs  = _embed_crops_dinov2(crops, dino_model)  # (K, D) L2-normed
                drift = _drift_curve(embs)              # (K,)  cosine dist to frame 0
                dino_drifts[ct["id"]] = drift

            dino_available = True
            yield {"type": "log", "level": "info",
                   "text": f"Appearance drift computed for {len(dino_drifts)} track(s)."}
        except Exception as exc:
            yield {"type": "log", "level": "warn",
                   "text": f"DINOv2 unavailable ({exc}) — skipping appearance embedding."}
    await asyncio.sleep(0)

    # ── Phase 4: Plotly visualisation ─────────────────────────────────────────
    yield {"type": "log", "level": "info", "text": "Building plots…"}
    await asyncio.sleep(0)

    n_rows    = 2 if dino_available else 1
    subtitles = ["Object Trajectories (centroid, px)"]
    if dino_available:
        subtitles.append("DINOv2 Appearance Drift per Track — cosine dist to frame 0")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.12, subplot_titles=subtitles,
    )

    for ct in obj_tracks:
        color  = _PALETTE[ct["id"] % len(_PALETTE)]
        t_secs = [f / eff_fps for f in ct["frames"]]

        # Y-centroid trajectory (inverted: 0=bottom, H=top in pixel space → flip)
        fig.add_trace(go.Scatter(
            x=t_secs,
            y=[H - cy for cy in ct["cy"]],
            mode="lines+markers",
            marker=dict(size=4, color=color),
            line=dict(color=color, width=1.6),
            name=f"obj {ct['id']}  ({ct['n_kp']} kp)",
            hovertemplate=(
                "<b>t = %{x:.2f}s</b><br>"
                "y-centroid = %{y:.0f}px<br>"
                f"track {ct['id']}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        if dino_available and ct["id"] in dino_drifts:
            drift   = dino_drifts[ct["id"]]
            t_drift = t_secs[:len(drift)]
            # Flag high-drift anomalies
            thresh = 0.35
            fig.add_trace(go.Scatter(
                x=t_drift, y=drift.tolist(),
                mode="lines+markers",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=1.6),
                name=f"obj {ct['id']} drift",
                showlegend=False,
                hovertemplate=(
                    "<b>t = %{x:.2f}s</b><br>"
                    "cosine dist = %{y:.3f}<br>"
                    f"track {ct['id']}"
                    "<extra></extra>"
                ),
            ), row=2, col=1)

            # Highlight anomalous drift segments
            spikes = np.where(drift > thresh)[0]
            for si in spikes:
                if si < len(t_drift):
                    hw = 0.5 / eff_fps
                    fig.add_vrect(
                        x0=t_drift[si] - hw, x1=t_drift[si] + hw,
                        fillcolor=color, opacity=0.12, line_width=0, row=2, col=1,
                    )

    if dino_available:
        fig.add_hline(
            y=0.35, line=dict(color="#E24B4A", dash="dash", width=1.2),
            annotation_text="appearance jump threshold (0.35)",
            annotation_font=dict(color="#E24B4A", size=11),
            annotation_position="top right", row=2, col=1,
        )

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid, title_text="Time (s)", row=n_rows, col=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    fig.update_yaxes(title_text="Y position (px, bottom=0)", row=1, col=1)
    if dino_available:
        fig.update_yaxes(title_text="Cosine distance to frame 0", range=[0, None], row=2, col=1)

    fig.update_layout(
        title=dict(text="Object Tracker — LK Trajectories & DINOv2 Appearance Drift",
                   font=dict(size=15)),
        height=380 + (260 if dino_available else 0),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=40, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        hovermode="x unified",
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": (
            "Y-centroid trajectories of detected object tracks (top). "
            + ("DINOv2 appearance drift — flat = stable, spikes = identity jump (bottom)."
               if dino_available else "")
        ),
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    durations  = [len(ct["frames"]) for ct in obj_tracks]
    mean_dur   = float(np.mean(durations))  if durations  else 0.0
    persistent = sum(1 for d in durations if d >= n * 0.5)

    if dino_drifts:
        all_drift  = np.concatenate(list(dino_drifts.values()))
        mean_drift = float(all_drift.mean())
        peak_drift = float(all_drift.max())
    else:
        mean_drift = peak_drift = float("nan")

    yield {"type": "metric", "label": "Objects tracked",   "value": str(n_objects),
           "sub": f"{persistent} persistent (≥50% of video)"}
    yield {"type": "metric", "label": "Keypoint loss",     "value": f"{kp_loss_pct:.0%}",
           "sub": f"{kp_lost} of {nk} keypoints lost"}
    yield {"type": "metric", "label": "Mean track length", "value": f"{mean_dur:.0f}",
           "sub": f"frames (video = {n})"}

    if not np.isnan(mean_drift):
        yield {"type": "metric", "label": "Mean appearance drift", "value": f"{mean_drift:.3f}",
               "sub": "cosine dist to frame 0 (DINOv2)"}
        yield {"type": "metric", "label": "Peak drift",            "value": f"{peak_drift:.3f}",
               "sub": "max identity deviation across all tracks"}

    # Instability score: weight keypoint loss + appearance drift
    drift_contrib = (min(mean_drift, 0.6) / 0.6 * 60) if not np.isnan(mean_drift) else 30
    loss_contrib  = kp_loss_pct * 40
    instability   = min(int(drift_contrib + loss_contrib), 100)
    sev_color     = "#E24B4A" if instability > 50 else "#EF9F27" if instability > 25 else "#4CAF50"
    yield {"type": "severity", "label": "Tracking instability score",
           "value": instability, "color": sev_color}

    msg = (
        f"{n_objects} track(s), {persistent} persistent, "
        f"{kp_loss_pct:.0%} keypoint loss"
        + (f", mean drift {mean_drift:.3f}" if not np.isnan(mean_drift) else "") + "."
    )
    yield {"type": "log", "level": "success" if instability < 30 else "warn", "text": msg}
    yield {"type": "done"}

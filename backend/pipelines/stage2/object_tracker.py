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
import asyncio, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames, frame_to_gray
from tools.flow  import detect_keypoints, track_keypoints

_PALETTE = [
    '#1a54c4', '#c05621', '#7c3aed', '#1a7a3c',
    '#e24b4a', '#d97706', '#0891b2', '#be185d',
    '#0f766e', '#7e22ce',
]

# ── Tunables (adapted from sam_track_compare.py) ──────────────────────────────
CROP_PAD    = 0.10   # fractional padding around bounding box before DINOv2
MIN_CROP_PX = 12     # ignore degenerate boxes smaller than this


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _box_from_points(pts: np.ndarray, H: int, W: int) -> Optional[tuple]:
    """Padded XYXY box for a set of (x, y) 2-D points, or None if degenerate."""
    if len(pts) == 0:
        return None
    x0, x1 = int(pts[:, 0].min()), int(pts[:, 0].max())
    y0, y1 = int(pts[:, 1].min()), int(pts[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    # Expand point-like clusters
    if bw < MIN_CROP_PX:
        pad = (MIN_CROP_PX - bw) // 2 + 4
        x0, x1 = x0 - pad, x1 + pad
        bw = x1 - x0
    if bh < MIN_CROP_PX:
        pad = (MIN_CROP_PX - bh) // 2 + 4
        y0, y1 = y0 - pad, y1 + pad
        bh = y1 - y0
    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
        return None
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))


def _cluster_points(pts: np.ndarray, dist_thresh: float) -> list[list[int]]:
    """Greedy single-linkage clustering by pairwise Euclidean distance."""
    assigned = [False] * len(pts)
    clusters: list[list[int]] = []
    for i in range(len(pts)):
        if assigned[i]:
            continue
        grp = [i]; assigned[i] = True
        for j in range(i + 1, len(pts)):
            if not assigned[j] and np.linalg.norm(pts[i] - pts[j]) < dist_thresh:
                grp.append(j); assigned[j] = True
        clusters.append(grp)
    return clusters


# ── DINOv2 appearance drift (ported from sam_track_compare.py) ───────────────

def _embed_crops_dinov2(crops: list[np.ndarray]) -> np.ndarray:
    """
    L2-normalised DINOv2 (ViT-S/14) CLS-token embeddings for a list of
    BGR/RGB uint8 crop arrays.  Ported from sam_track_compare._embed_track.
    Returns (K, D) float32.
    """
    from tools.embeddings import load_dinov2, embed_frames_dinov2
    model = load_dinov2()
    embs  = embed_frames_dinov2(crops, model)       # (K, D), already L2-normed inside
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
    num_kp       = max(5,  int(cfg.get("num_keypoints", 50)))
    sample_every = max(1,  int(cfg.get("sample_every",   1)))
    use_dinov2   = str(cfg.get("use_dinov2", "true")).lower() not in ("false", "0", "no")

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path, step=sample_every)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return

    H, W = frames[0].shape[:2]
    eff_fps = fps / sample_every

    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {eff_fps:.1f} fps — detecting {num_kp} Shi-Tomasi keypoints…"}
    await asyncio.sleep(0)

    # ── Phase 1: LK sparse tracking ───────────────────────────────────────────
    gray0 = frame_to_gray(frames[0])
    pts0  = detect_keypoints(gray0, n=num_kp)
    if pts0 is None or len(pts0) == 0:
        yield {"type": "error", "text": "No keypoints detected in first frame."}
        return

    nk       = len(pts0)
    tracks   = [[pts0[i, 0].tolist()] for i in range(nk)]   # list of [x,y] or None
    active   = [pts0[i : i + 1] for i in range(nk)]         # current LK input

    prev_gray = gray0
    for f in range(1, n):
        curr_gray = frame_to_gray(frames[f])
        for ki in range(nk):
            if active[ki] is None:
                tracks[ki].append(None)
                continue
            _, good = track_keypoints(prev_gray, curr_gray, active[ki])
            if len(good) == 0:
                active[ki] = None
                tracks[ki].append(None)
            else:
                active[ki] = good.reshape(-1, 1, 2)
                tracks[ki].append(good.reshape(-1, 2)[0].tolist())
        prev_gray = curr_gray
        if f % 30 == 0:
            yield {"type": "log", "level": "info", "text": f"Tracking… {f}/{n} frames"}
            await asyncio.sleep(0)

    kp_lost     = sum(1 for ki in range(nk) if active[ki] is None)
    kp_loss_pct = kp_lost / max(nk, 1)

    # ── Phase 2: Cluster keypoints → object tracks ────────────────────────────
    yield {"type": "log", "level": "info", "text": "Clustering keypoints into object tracks…"}
    await asyncio.sleep(0)

    dist_thresh = max(40.0, min(W, H) * 0.08)
    clusters    = _cluster_points(pts0[:, 0, :], dist_thresh)

    obj_tracks: list[dict] = []
    for ci, idx_list in enumerate(clusters):
        frame_idxs, boxes, cx_list, cy_list = [], [], [], []
        for f in range(n):
            active_pts = [tracks[ki][f] for ki in idx_list if tracks[ki][f] is not None]
            if not active_pts:
                continue
            arr = np.array(active_pts)
            box = _box_from_points(arr, H, W)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            frame_idxs.append(f)
            boxes.append(box)
            cx_list.append((x0 + x1) / 2)
            cy_list.append((y0 + y1) / 2)

        if len(frame_idxs) < 3:
            continue
        obj_tracks.append({
            "id": ci, "frames": frame_idxs, "boxes": boxes,
            "cx": cx_list, "cy": cy_list, "n_kp": len(idx_list),
        })

    obj_tracks.sort(key=lambda t: -len(t["frames"]))
    obj_tracks = obj_tracks[:8]   # cap at 8 objects
    n_objects  = len(obj_tracks)

    yield {"type": "log", "level": "info",
           "text": f"{n_objects} distinct object track(s) found from {nk} keypoints."}
    await asyncio.sleep(0)

    # ── Phase 3: DINOv2 appearance drift (ported from sam_track_compare) ──────
    dino_drifts: dict[int, np.ndarray] = {}
    dino_available = False

    if use_dinov2 and n_objects > 0:
        yield {"type": "log", "level": "info",
               "text": "Loading DINOv2 for appearance-drift analysis (sam_track_compare approach)…"}
        await asyncio.sleep(0)
        try:
            for ct in obj_tracks:
                crops = []
                for f, box in zip(ct["frames"], ct["boxes"]):
                    x0, y0, x1, y1 = box
                    crop = frames[f][y0:y1, x0:x1]
                    if crop.size > 0:
                        crops.append(crop)

                if len(crops) < 2:
                    continue

                embs  = _embed_crops_dinov2(crops)      # (K, D) L2-normed
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

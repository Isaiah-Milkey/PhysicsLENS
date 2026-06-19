"""
Stage 1 · Test 1 — Temporal Smoothness Anomalies
-------------------------------------------------
Track N keypoints frame-to-frame with Lucas-Kanade.
Compute per-track velocity and acceleration; flag abrupt jumps.
"""
import asyncio, json
from itertools import cycle
from typing import AsyncGenerator

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames, frame_to_gray
from tools.flow  import detect_keypoints, track_keypoints

_PALETTE = ['#1a54c4','#c05621','#7c3aed','#1a7a3c','#e24b4a','#d97706','#0891b2','#be185d']


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    num_kp     = max(1, int(cfg.get("num_keypoints", 5)))
    acc_thresh = float(cfg.get("accel_threshold", 5.0))

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": "Video too short (need ≥ 3 frames)."}
        return

    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {fps:.1f} fps — detecting {num_kp} keypoints…"}
    await asyncio.sleep(0)

    gray0 = frame_to_gray(frames[0])
    pts0  = detect_keypoints(gray0, n=num_kp)
    if pts0 is None or len(pts0) == 0:
        yield {"type": "error", "text": "No keypoints found in first frame."}
        return

    nk       = len(pts0)
    tracks   = [[pts0[i, 0].tolist()] for i in range(nk)]
    curr_pts = [pts0[i : i + 1] for i in range(nk)]

    prev_gray = gray0
    for f in range(1, n):
        curr_gray = frame_to_gray(frames[f])
        for ki in range(nk):
            if curr_pts[ki] is None:
                continue
            _, good = track_keypoints(prev_gray, curr_gray, curr_pts[ki])
            if len(good) == 0:
                curr_pts[ki] = None
            else:
                curr_pts[ki] = good.reshape(-1, 1, 2)
                tracks[ki].append(good.reshape(-1, 2)[0].tolist())
        prev_gray = curr_gray
        if f % 60 == 0:
            yield {"type": "log", "level": "info", "text": f"Tracking… {f}/{n} frames"}
            await asyncio.sleep(0)

    yield {"type": "log", "level": "info", "text": "Building plot…"}
    await asyncio.sleep(0)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
        subplot_titles=(
            "Keypoint Velocity  (px/s)",
            "Keypoint Acceleration  (px/s²) — anomalies highlighted",
        ),
    )

    all_accs, flagged = [], set()
    color_cycle = cycle(_PALETTE)

    for ki, traj in enumerate(tracks):
        c = next(color_cycle)
        if len(traj) < 3:
            continue
        pos = np.array(traj)
        vel = np.diff(pos, axis=0) * fps
        acc = np.diff(vel,  axis=0) * fps
        vm  = np.linalg.norm(vel, axis=1)
        am  = np.linalg.norm(acc, axis=1)
        t_v = [i / fps for i in range(1, 1 + len(vm))]
        t_a = [i / fps for i in range(2, 2 + len(am))]

        fig.add_trace(go.Scatter(
            x=t_v, y=vm.tolist(), mode="lines", name=f"kp {ki}",
            line=dict(color=c, width=1.6),
            hovertemplate="<b>t = %{x:.2f}s</b><br>velocity = %{y:.1f} px/s<extra>kp %{meta}</extra>",
            meta=ki,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t_a, y=am.tolist(), mode="lines", name=f"kp {ki}",
            line=dict(color=c, width=1.6), showlegend=False,
            hovertemplate="<b>t = %{x:.2f}s</b><br>accel = %{y:.1f} px/s²<extra>kp %{meta}</extra>",
            meta=ki,
        ), row=2, col=1)

        for si in np.where(am > acc_thresh)[0]:
            flagged.add(si + 2)
        all_accs.append(am)

    # Threshold reference line
    fig.add_hline(
        y=acc_thresh,
        line=dict(color="#E24B4A", dash="dash", width=1.5),
        annotation_text=f"threshold = {acc_thresh} px/s²",
        annotation_font=dict(color="#E24B4A", size=11),
        annotation_position="top right",
        row=2, col=1,
    )

    # Highlight flagged frames as red bands
    for fi in sorted(flagged):
        hw = max(0.5 / fps, 0.02)
        fig.add_vrect(
            x0=fi / fps - hw, x1=fi / fps + hw,
            fillcolor="#E24B4A", opacity=0.18, line_width=0,
            row=2, col=1,
        )

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="px / s",   row=1, col=1)
    fig.update_yaxes(title_text="px / s²",  row=2, col=1)
    fig.update_layout(
        title=dict(text="Temporal Smoothness — Keypoint Trajectory Analysis", font=dict(size=15)),
        height=540,
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=45, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        hovermode="x unified",
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": "Velocity per keypoint (top). Acceleration with red-highlighted anomalous frames (bottom).",
    }

    # ── Structured signals for Stage 2 Event Localizer ────────────────────────
    # Per flagged frame, score = peak acceleration there relative to the threshold.
    acc_by_frame: dict[int, float] = {}
    for am in all_accs:
        for j, a in enumerate(am):
            fr = j + 2
            if a > acc_thresh:
                acc_by_frame[fr] = max(acc_by_frame.get(fr, 0.0), float(a))
    signals = [
        {"frame": int(fi), "signal_type": "temporal_anomaly",
         "score": round(acc_by_frame.get(int(fi), acc_thresh) / max(acc_thresh, 1e-6), 3)}
        for fi in sorted(flagged)
    ]
    n_flagged = len(flagged)
    peak_acc  = float(max((a.max() for a in all_accs if len(a)), default=0))
    mean_acc  = float(np.mean(np.concatenate(all_accs)) if all_accs else 0)
    score     = min(int(n_flagged / max(n, 1) * 300), 100)
    color     = "#E24B4A" if score > 40 else "#EF9F27" if score > 15 else "#4CAF50"

    yield {"type": "signal", "source": "s1_temporal", "source_name": "Temporal Smoothness",
           "fps": float(fps), "n_frames": int(n), "severity": score,
           "type_severities": {"temporal_anomaly": score}, "signals": signals}

    yield {"type": "metric", "label": "Flagged frames",    "value": str(n_flagged),    "sub": f"of {n} total"}
    yield {"type": "metric", "label": "Peak acceleration", "value": f"{peak_acc:.1f}", "sub": "px/s²"}
    yield {"type": "metric", "label": "Mean acceleration", "value": f"{mean_acc:.1f}", "sub": "px/s²"}
    yield {"type": "severity", "label": "Temporal anomaly score", "value": score, "color": color}

    msg = ("No significant temporal jumps detected." if n_flagged == 0
           else f"{n_flagged} frame(s) show abrupt acceleration spikes.")
    yield {"type": "log", "level": "success" if n_flagged == 0 else "warn", "text": msg}
    yield {"type": "done"}

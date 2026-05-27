"""
Optical Flow Comparison pipeline
---------------------------------
Compares optical flow statistics between a ground-truth video and an AI-generated
video. Produces a matplotlib report image and streams metrics to the UI.

Requires:  pip install opencv-python-headless numpy matplotlib
"""

import asyncio
import base64
import io
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np


# ─── Core computation (sync, called in a thread) ─────────────────────────────

def _flow_from_video(video_path: str, height: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def read_gray():
        ret, frame = cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if height:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(height * w / h), height))
        return gray

    all_flow = []
    prev = read_gray()
    while prev is not None:
        nxt = read_gray()
        if nxt is None:
            break
        all_flow.append(dis.calc(prev, nxt, None))
        prev = nxt

    cap.release()
    return np.array(all_flow)


def _compute_stats(flows: np.ndarray) -> dict:
    u = flows[..., 0].flatten()
    v = flows[..., 1].flatten()
    speed = np.sqrt(u**2 + v**2)
    angle = np.degrees(np.arctan2(v, u))
    dudx = np.concatenate([np.gradient(f[..., 0], axis=1).flatten() for f in flows])
    return {"u": u, "v": v, "speed": speed, "angle": angle, "dudx": dudx}


def _build_report(gt_stats: dict, ai_stats: dict, label_gt: str, label_ai: str) -> bytes:
    """Render matplotlib figure, return PNG bytes."""

    def log_hist(ax, a, b, title, xlabel):
        for data, color, ls, label in [
            (a, "#1a54c4", "-",  label_gt),
            (b, "#b91c1c", "--", label_ai),
        ]:
            h, e = np.histogram(data, bins=100, density=True)
            h = np.where(h == 0, np.nan, h)
            ax.semilogy(e[:-1], h, color=color, linestyle=ls, label=label, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("log(density)", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle("Optical Flow Comparison — GT vs AI", fontsize=13, fontweight="bold")

    log_hist(axs[0, 0], gt_stats["u"],     ai_stats["u"],     "Horizontal motion (u)", "pixels/frame")
    log_hist(axs[0, 1], gt_stats["v"],     ai_stats["v"],     "Vertical motion (v)",   "pixels/frame")
    log_hist(axs[1, 0], gt_stats["speed"], ai_stats["speed"], "Speed",                 "pixels/frame")
    log_hist(axs[2, 0], gt_stats["dudx"],  ai_stats["dudx"],  "Flow derivative (du/dx)", "1/frame")

    axs[1, 1].hist(gt_stats["angle"], bins=180, range=(-180, 180), density=True,
                   alpha=0.5, color="#1a54c4", label=label_gt)
    axs[1, 1].hist(ai_stats["angle"], bins=180, range=(-180, 180), density=True,
                   histtype="step", color="#b91c1c", linewidth=1.5, label=label_ai)
    axs[1, 1].set_title("Flow direction", fontsize=11)
    axs[1, 1].set_xlabel("degrees", fontsize=9)
    axs[1, 1].legend(fontsize=9)
    axs[1, 1].grid(True, alpha=0.3)

    axs[2, 1].axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _similarity_score(gt_stats: dict, ai_stats: dict) -> float:
    """
    Simple distributional similarity: compare mean speed and direction spread.
    Returns 0–100 (higher = more similar to GT).
    """
    speed_diff = abs(gt_stats["speed"].mean() - ai_stats["speed"].mean())
    speed_ref  = max(gt_stats["speed"].mean(), 1e-6)
    speed_score = max(0.0, 1.0 - speed_diff / speed_ref)

    angle_std_gt = gt_stats["angle"].std()
    angle_std_ai = ai_stats["angle"].std()
    angle_diff   = abs(angle_std_gt - angle_std_ai)
    angle_score  = max(0.0, 1.0 - angle_diff / max(angle_std_gt, 1e-6))

    return round((speed_score * 0.6 + angle_score * 0.4) * 100, 1)


# ─── Async pipeline entry point ───────────────────────────────────────────────

async def run(gt_path: str, ai_path: str) -> AsyncGenerator[dict, None]:
    """
    Accepts two video paths (gt_path, ai_path).
    Registered in main.py under requires_pair=True.
    """
    loop = asyncio.get_event_loop()

    # --- GT flow ---
    yield {"type": "log", "level": "info", "text": f"Loading GT video: {Path(gt_path).name}"}
    try:
        gt_flows = await loop.run_in_executor(None, _flow_from_video, gt_path)
    except Exception as e:
        yield {"type": "error", "text": f"GT video error: {e}"}
        return
    yield {"type": "log", "level": "info", "text": f"GT: computed {len(gt_flows)} flow fields"}

    # --- AI flow ---
    yield {"type": "log", "level": "info", "text": f"Loading AI video: {Path(ai_path).name}"}
    try:
        ai_flows = await loop.run_in_executor(None, _flow_from_video, ai_path)
    except Exception as e:
        yield {"type": "error", "text": f"AI video error: {e}"}
        return
    yield {"type": "log", "level": "info", "text": f"AI: computed {len(ai_flows)} flow fields"}

    # --- Stats ---
    yield {"type": "log", "level": "info", "text": "Computing flow statistics…"}
    gt_stats = await loop.run_in_executor(None, _compute_stats, gt_flows)
    ai_stats = await loop.run_in_executor(None, _compute_stats, ai_flows)

    # --- Metrics ---
    score = _similarity_score(gt_stats, ai_stats)
    gt_mean_spd = round(float(gt_stats["speed"].mean()), 3)
    ai_mean_spd = round(float(ai_stats["speed"].mean()), 3)
    spd_delta   = round(abs(gt_mean_spd - ai_mean_spd), 3)

    yield {"type": "metric", "label": "Flow similarity",   "value": f"{score}%",     "sub": "GT vs AI overall"}
    yield {"type": "metric", "label": "GT mean speed",     "value": str(gt_mean_spd), "sub": "px/frame"}
    yield {"type": "metric", "label": "AI mean speed",     "value": str(ai_mean_spd), "sub": "px/frame"}
    yield {"type": "metric", "label": "Speed delta",       "value": str(spd_delta),   "sub": "px/frame abs diff"}

    sev_color = "#1a7a3c" if score >= 70 else "#9a6200" if score >= 40 else "#b91c1c"
    yield {"type": "severity", "label": "Flow similarity score", "value": int(score), "color": sev_color}

    # --- Plot ---
    yield {"type": "log", "level": "info", "text": "Rendering comparison report…"}
    png_bytes = await loop.run_in_executor(
        None, _build_report, gt_stats, ai_stats,
        Path(gt_path).name, Path(ai_path).name
    )
    b64 = base64.b64encode(png_bytes).decode()
    yield {"type": "image", "data": b64, "mime": "image/png", "caption": "Optical flow comparison"}

    yield {"type": "log", "level": "success", "text": "Optical flow comparison complete."}
    yield {"type": "done"}

"""
Sparse Optical Flow Comparison pipeline
-----------------------------------------
Tracks Shi-Tomasi corners with Lucas-Kanade across GT and AI videos,
renders coloured trajectory trails on every frame, encodes each result
as a playable MP4, then compares the resulting flow distributions.

Produces:
  1. GT video with sparse flow trails overlaid  (video/mp4)
  2. AI video with sparse flow trails overlaid  (video/mp4)
  3. Statistical comparison report              (image/png)

Requires:  pip install opencv-python-headless numpy matplotlib
"""

import asyncio
import base64
import io
from pathlib import Path
from typing import AsyncGenerator

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None
    _PIL_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_CORNERS   = 150     # max tracked points at any moment
QUALITY_LEVEL = 0.01
MIN_DISTANCE  = 10
TRAIL_LEN     = 25      # frames of history shown per trail
RESEED_EVERY  = 8       # re-detect features in motion regions every N frames
MOG2_HISTORY  = 20      # background model history (frames); short = fast warmup
MOG2_THRESH   = 30      # sensitivity; lower = picks up more motion

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

_RNG = np.random.default_rng(42)
_COLOR_POOL = [tuple(int(c) for c in row)
               for row in _RNG.integers(60, 220, size=(10_000, 3))]

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


# ─── Core computation ─────────────────────────────────────────────────────────

def _track_and_render(
    video_path: str, height: int = 360
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """
    Run motion-seeded LK sparse optical flow.

    Feature points are detected only inside the MOG2 foreground mask so that
    static background regions (walls, floor) are ignored and moving objects
    (e.g. balls) are tracked.  New seeds are injected every RESEED_EVERY frames
    to follow objects as they move across the frame.

    Returns
    -------
    rendered_frames : list of BGR ndarrays with trails drawn
    flow_vectors    : (N, 2) float32 array of (dx, dy) per tracked pair
    fps             : source video frame rate (fallback 25.0)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_THRESH, detectShadows=False
    )

    def read_frame():
        ret, frame = cap.read()
        if not ret:
            return None, None
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(height * w / h), height))
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def foreground_mask(bgr_frame: np.ndarray) -> np.ndarray:
        """MOG2 foreground mask, morphologically cleaned."""
        fg = bg_sub.apply(bgr_frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _MORPH_KERNEL)
        fg = cv2.dilate(fg, _MORPH_KERNEL, iterations=1)
        return fg

    def seed_in_mask(gray: np.ndarray, fg: np.ndarray,
                     existing: np.ndarray) -> list[tuple[float, float]]:
        """Detect Shi-Tomasi corners inside fg, skipping near existing pts."""
        if fg.sum() < 300:          # not enough foreground pixels
            return []
        raw = cv2.goodFeaturesToTrack(gray, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, mask=fg)
        if raw is None:
            return []
        result = []
        for p in raw:
            x, y = float(p[0][0]), float(p[0][1])
            if len(existing) > 0:
                dists = np.hypot(existing[:, 0, 0] - x, existing[:, 0, 1] - y)
                if dists.min() < MIN_DISTANCE * 1.5:
                    continue
            result.append((x, y))
        return result

    # ── initialise ────────────────────────────────────────────────────────────
    frame, gray_prev = read_frame()
    if frame is None:
        cap.release()
        return [], np.empty((0, 2), dtype=np.float32), fps

    # Warm up bg model on frame 0 (returns all-foreground, not useful for seeding)
    _ = foreground_mask(frame)

    next_id:    int                              = 0
    all_tracks: dict[int, list[tuple[int, int]]] = {}
    active_ids: list[int]                        = []
    pts = np.empty((0, 1, 2), dtype=np.float32)

    all_flow_vecs: list[np.ndarray] = []
    rendered_frames = [frame.copy()]   # frame 0: no trails yet
    frame_count = 0

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        frame, gray_next = read_frame()
        if frame is None:
            break

        frame_count += 1
        fg = foreground_mask(frame)   # updates background model every frame

        # Re-seed points in foreground (motion) regions
        if frame_count % RESEED_EVERY == 0 and len(active_ids) < MAX_CORNERS * 2:
            for x, y in seed_in_mask(gray_next, fg, pts):
                all_tracks[next_id] = [(int(x), int(y))]
                active_ids.append(next_id)
                next_id += 1
                new_pt = np.array([[[x, y]]], dtype=np.float32)
                pts = np.vstack([pts, new_pt]) if len(pts) else new_pt

        # Track existing points with LK
        if len(pts) == 0:
            rendered_frames.append(frame.copy())
            gray_prev = gray_next
            continue

        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_next, pts, None, **LK_PARAMS
        )

        new_pts:    list[np.ndarray]  = []
        new_ids:    list[int]         = []
        frame_vecs: list[list[float]] = []

        for tid, st, p_new, p_old in zip(active_ids, status.flatten(), pts_next, pts):
            if st:
                x, y = int(p_new[0][0]), int(p_new[0][1])
                all_tracks[tid].append((x, y))
                new_pts.append(p_new)
                new_ids.append(tid)
                frame_vecs.append([
                    float(p_new[0][0] - p_old[0][0]),
                    float(p_new[0][1] - p_old[0][1]),
                ])

        if frame_vecs:
            all_flow_vecs.append(np.array(frame_vecs, dtype=np.float32))

        pts        = np.array(new_pts, dtype=np.float32) if new_pts else np.empty((0, 1, 2), dtype=np.float32)
        active_ids = new_ids
        gray_prev  = gray_next
        rendered_frames.append(_draw_trails(frame.copy(), all_tracks, TRAIL_LEN))

    cap.release()
    flow_vectors = (np.concatenate(all_flow_vecs, axis=0)
                    if all_flow_vecs else np.empty((0, 2), dtype=np.float32))
    return rendered_frames, flow_vectors, fps


def _draw_trails(frame: np.ndarray, tracks: dict[int, list], trail_len: int) -> np.ndarray:
    for tid, pts_list in tracks.items():
        tail  = pts_list[-trail_len:]
        color = _COLOR_POOL[tid % len(_COLOR_POOL)]
        n     = len(tail)
        for j in range(1, n):
            alpha     = j / n
            thickness = max(1, int(alpha * 2))
            cv2.line(frame, tail[j - 1], tail[j], color, thickness, cv2.LINE_AA)
        if tail:
            cv2.circle(frame, tail[-1], 3, color, -1, cv2.LINE_AA)
    return frame


def _encode_animation(frames: list[np.ndarray], fps: float) -> tuple[bytes, str]:
    """
    Encode BGR frames to an animated image using Pillow (no codec needed).
    Tries WebP first (full colour), falls back to GIF (256-colour, fewer frames).
    Returns (bytes, mime_type).
    Raises RuntimeError if Pillow is unavailable.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed. Run: pip install pillow")

    if not frames:
        raise RuntimeError("No frames to encode.")

    duration_ms = max(33, int(1000 / fps))

    # WebP: keep up to 120 frames at full resolution
    webp_frames = frames
    if len(webp_frames) > 120:
        idx = np.linspace(0, len(frames) - 1, 120, dtype=int)
        webp_frames = [frames[i] for i in idx]

    pil_webp = [_PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in webp_frames]

    buf = io.BytesIO()
    try:
        pil_webp[0].save(
            buf, format="WEBP", save_all=True, append_images=pil_webp[1:],
            loop=0, duration=duration_ms, method=3,
        )
        data = buf.getvalue()
        if len(data) > 100:           # sanity-check: not an empty/stub file
            return data, "image/webp"
    except Exception:
        pass

    # GIF fallback: limit to 50 frames at half-width to keep quantisation fast
    gif_max = 50
    idx = np.linspace(0, len(frames) - 1, min(gif_max, len(frames)), dtype=int)
    gif_frames = [frames[i] for i in idx]
    gif_duration = int(len(frames) / len(gif_frames) * duration_ms)   # stretch to preserve speed

    def _half(f):
        h, w = f.shape[:2]
        small = cv2.resize(f, (w // 2, h // 2))
        return _PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    pil_gif = [_half(f) for f in gif_frames]

    buf = io.BytesIO()
    pil_gif[0].save(
        buf, format="GIF", save_all=True, append_images=pil_gif[1:],
        loop=0, duration=gif_duration,
    )
    data = buf.getvalue()
    if len(data) < 100:
        raise RuntimeError("GIF output is empty — quantisation may have failed.")
    return data, "image/gif"


def _compute_stats(flow_vectors: np.ndarray) -> dict:
    if len(flow_vectors) == 0:
        z = np.zeros(1, dtype=np.float32)
        return {"u": z, "v": z, "speed": z, "angle": z}
    u     = flow_vectors[:, 0]
    v     = flow_vectors[:, 1]
    speed = np.sqrt(u**2 + v**2)
    angle = np.degrees(np.arctan2(v, u))
    return {"u": u, "v": v, "speed": speed, "angle": angle}


def _similarity_score(gt: dict, ai: dict) -> float:
    speed_diff  = abs(gt["speed"].mean() - ai["speed"].mean())
    speed_ref   = max(float(gt["speed"].mean()), 1e-6)
    speed_score = max(0.0, 1.0 - speed_diff / speed_ref)

    angle_std_gt = float(gt["angle"].std())
    angle_std_ai = float(ai["angle"].std())
    angle_diff   = abs(angle_std_gt - angle_std_ai)
    angle_score  = max(0.0, 1.0 - angle_diff / max(angle_std_gt, 1e-6))

    return round((speed_score * 0.6 + angle_score * 0.4) * 100, 1)


# ─── Plot helper ──────────────────────────────────────────────────────────────

def _build_stats_report(gt_stats: dict, ai_stats: dict, label_gt: str, label_ai: str) -> bytes:

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

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Sparse Optical Flow Statistics — GT vs AI", fontsize=13, fontweight="bold")

    log_hist(axs[0, 0], gt_stats["u"],     ai_stats["u"],     "Horizontal motion (u)", "pixels/frame")
    log_hist(axs[0, 1], gt_stats["v"],     ai_stats["v"],     "Vertical motion (v)",   "pixels/frame")
    log_hist(axs[1, 0], gt_stats["speed"], ai_stats["speed"], "Speed",                 "pixels/frame")

    axs[1, 1].hist(gt_stats["angle"], bins=180, range=(-180, 180), density=True,
                   alpha=0.5, color="#1a54c4", label=label_gt)
    axs[1, 1].hist(ai_stats["angle"], bins=180, range=(-180, 180), density=True,
                   histtype="step", color="#b91c1c", linewidth=1.5, label=label_ai)
    axs[1, 1].set_title("Flow direction", fontsize=11)
    axs[1, 1].set_xlabel("degrees", fontsize=9)
    axs[1, 1].legend(fontsize=9)
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Async pipeline entry point ───────────────────────────────────────────────

async def run(gt_path: str, ai_path: str) -> AsyncGenerator[dict, None]:
    """
    Accepts two video paths (gt_path, ai_path).
    Registered in main.py under requires_pair=True.
    """
    loop = asyncio.get_event_loop()

    yield {"type": "log", "level": "info", "text": f"Loading GT video: {Path(gt_path).name}"}
    try:
        gt_frames, gt_vecs, gt_fps = await loop.run_in_executor(None, _track_and_render, gt_path)
    except Exception as e:
        yield {"type": "error", "text": f"GT video error: {e}"}
        return
    yield {"type": "log", "level": "info", "text": f"GT: {len(gt_frames)} frames at {gt_fps:.1f} fps"}

    yield {"type": "log", "level": "info", "text": f"Loading AI video: {Path(ai_path).name}"}
    try:
        ai_frames, ai_vecs, ai_fps = await loop.run_in_executor(None, _track_and_render, ai_path)
    except Exception as e:
        yield {"type": "error", "text": f"AI video error: {e}"}
        return
    yield {"type": "log", "level": "info", "text": f"AI: {len(ai_frames)} frames at {ai_fps:.1f} fps"}

    # Stats
    yield {"type": "log", "level": "info", "text": "Computing sparse flow statistics…"}
    gt_stats = _compute_stats(gt_vecs)
    ai_stats = _compute_stats(ai_vecs)

    score       = _similarity_score(gt_stats, ai_stats)
    gt_mean_spd = round(float(gt_stats["speed"].mean()), 3)
    ai_mean_spd = round(float(ai_stats["speed"].mean()), 3)
    spd_delta   = round(abs(gt_mean_spd - ai_mean_spd), 3)

    yield {"type": "metric", "label": "Flow similarity",   "value": f"{score}%",       "sub": "GT vs AI overall"}
    yield {"type": "metric", "label": "GT mean speed",     "value": str(gt_mean_spd),  "sub": "px/frame (sparse)"}
    yield {"type": "metric", "label": "AI mean speed",     "value": str(ai_mean_spd),  "sub": "px/frame (sparse)"}
    yield {"type": "metric", "label": "Speed delta",       "value": str(spd_delta),    "sub": "px/frame abs diff"}
    yield {"type": "metric", "label": "GT LK points",      "value": str(len(gt_vecs)), "sub": "total observations"}
    yield {"type": "metric", "label": "AI LK points",      "value": str(len(ai_vecs)), "sub": "total observations"}

    sev_color = "#1a7a3c" if score >= 70 else "#9a6200" if score >= 40 else "#b91c1c"
    yield {"type": "severity", "label": "Sparse flow similarity", "value": int(score), "color": sev_color}

    # GT animation
    yield {"type": "log", "level": "info",
           "text": f"Encoding GT animation ({len(gt_frames)} frames)…"}
    try:
        gt_anim_bytes, gt_mime = await loop.run_in_executor(
            None, _encode_animation, gt_frames, gt_fps
        )
        yield {"type": "log", "level": "info",
               "text": f"GT animation ready: {len(gt_anim_bytes)/1024:.0f} KB ({gt_mime})"}
        yield {
            "type":    "video",
            "data":    base64.b64encode(gt_anim_bytes).decode(),
            "mime":    gt_mime,
            "caption": f"GT sparse flow — {Path(gt_path).name}",
        }
    except Exception as e:
        yield {"type": "error", "text": f"GT animation encode failed: {e}"}

    # AI animation
    yield {"type": "log", "level": "info",
           "text": f"Encoding AI animation ({len(ai_frames)} frames)…"}
    try:
        ai_anim_bytes, ai_mime = await loop.run_in_executor(
            None, _encode_animation, ai_frames, ai_fps
        )
        yield {"type": "log", "level": "info",
               "text": f"AI animation ready: {len(ai_anim_bytes)/1024:.0f} KB ({ai_mime})"}
        yield {
            "type":    "video",
            "data":    base64.b64encode(ai_anim_bytes).decode(),
            "mime":    ai_mime,
            "caption": f"AI sparse flow — {Path(ai_path).name}",
        }
    except Exception as e:
        yield {"type": "error", "text": f"AI animation encode failed: {e}"}

    # Stats report
    yield {"type": "log", "level": "info", "text": "Rendering statistics report…"}
    stats_bytes = await loop.run_in_executor(
        None, _build_stats_report, gt_stats, ai_stats,
        Path(gt_path).name, Path(ai_path).name,
    )
    yield {
        "type":    "image",
        "data":    base64.b64encode(stats_bytes).decode(),
        "mime":    "image/png",
        "caption": "Sparse optical flow statistics comparison",
    }

    yield {"type": "log", "level": "success", "text": "Sparse optical flow comparison complete."}
    yield {"type": "done"}

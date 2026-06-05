"""
SAM3 + DINOv2 Object Tracking Comparison pipeline
---------------------------------------------------
Segments an object of interest (named by a free-text prompt, e.g. "the ball")
with **SAM 3** promptable-concept video segmentation, tracks that single object
across every frame, and embeds its cropped appearance each frame with **DINOv2**.

The headline output is an **embedding-drift plot**: cosine distance of the
object's DINOv2 descriptor at frame *t* vs. its descriptor at the first frame it
appears.  A physically-consistent object keeps a near-flat drift curve; an AI
object that morphs / flickers identity spikes.  GT and AI curves are overlaid so
you can see where the AI video diverges from the ground truth.

Paired pipeline:  run(gt_path, ai_path, prompt="the ball").
Registered in main.py under requires_pair=True, requires_prompt=True.

Requires the GPU stack (see backend/requirements-gpu.txt):
    torch, transformers>=5.5  (SAM3 lives in transformers >=5.1; 5.9 verified),
    accelerate, safetensors, pillow, opencv-python-headless, numpy, matplotlib

Models (downloaded on first use into $HF_HOME):
    SAM 3   : facebook/sam3          (GATED — requires `hf auth login` + approval)
    DINOv2  : facebook/dinov2-base   (Apache-2.0, non-gated, 768-d CLS descriptor)
"""

import asyncio
import base64
import html
import io
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# ─── Tunables ─────────────────────────────────────────────────────────────────

SAM3_MODEL_ID   = "facebook/sam3"
DINOV2_MODEL_ID = "facebook/dinov2-base"   # 768-d CLS descriptor

MAX_FRAMES   = 80      # cap frames sent through SAM3 (uniformly subsampled if longer)
TARGET_H     = 720     # downscale tall frames to this height before tracking
CROP_PAD     = 0.08    # fractional padding around the object box before DINOv2
MIN_CROP_PX  = 8       # ignore degenerate boxes smaller than this
EMBED_BATCH  = 128     # DINOv2 crops per forward pass

# ─── Lazy, process-wide model singletons ──────────────────────────────────────
# Loading SAM3 (~3.4 GB) + DINOv2 is expensive, so load once and reuse across
# requests.  A lock serialises GPU inference (the FastAPI threadpool may invoke
# the pipeline concurrently).

_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()
_GPU_LOCK  = threading.Lock()


def _load_models():
    """Load SAM3 + DINOv2 once. Raises a clear error if SAM3 access is missing."""
    if _MODELS:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS:                       # double-checked: another thread won the race
            return _MODELS

        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            Sam3VideoModel,
            Sam3VideoProcessor,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — the SAM3+DINOv2 pipeline needs a GPU. "
                "Run the backend on the H100 host inside the GPU env."
            )

        device = "cuda"
        dtype  = torch.bfloat16

        try:
            sam_model = (
                Sam3VideoModel.from_pretrained(SAM3_MODEL_ID, dtype=dtype)
                .to(device).eval()
            )
            sam_proc = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
        except Exception as exc:                                   # noqa: BLE001
            raise RuntimeError(
                f"Could not load SAM3 ({SAM3_MODEL_ID}). It is a GATED model — "
                "request access on its HuggingFace page and authenticate with "
                "`hf auth login` (or set HF_TOKEN) using an approved account. "
                f"Underlying error: {exc}"
            ) from exc

        dino_proc  = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        dino_model = (
            AutoModel.from_pretrained(
                DINOV2_MODEL_ID, dtype=dtype, attn_implementation="sdpa"
            )
            .to(device).eval()
        )

        _MODELS.update(
            torch=torch, device=device, dtype=dtype,
            sam_model=sam_model, sam_proc=sam_proc,
            dino_model=dino_model, dino_proc=dino_proc,
        )
        return _MODELS


# ─── Frame loading ────────────────────────────────────────────────────────────

def _load_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    """Read a video into a list of RGB uint8 frames (subsampled + downscaled)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        h, w = bgr.shape[:2]
        if h > TARGET_H:
            bgr = cv2.resize(bgr, (int(TARGET_H * w / h), TARGET_H))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    if len(frames) > MAX_FRAMES:                     # uniform subsample
        idx = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
        frames = [frames[i] for i in idx]
    return frames, fps


# ─── SAM3 segmentation + single-object selection ──────────────────────────────

def _segment_track(frames: list[np.ndarray], prompt: str) -> tuple[dict[int, np.ndarray], int]:
    """
    Run SAM3 text-concept video tracking, then pick ONE object of interest and
    return its per-frame boolean masks.

    Returns
    -------
    masks_by_frame : {frame_idx: bool mask (H, W)} for the chosen object only
    target_id      : the SAM3 object id that was tracked
    """
    m = _load_models()
    torch, device, dtype = m["torch"], m["device"], m["dtype"]
    sam_model, sam_proc  = m["sam_model"], m["sam_proc"]

    with _GPU_LOCK:
        session = sam_proc.init_video_session(
            video=frames,
            inference_device=device,
            video_storage_device="cpu",   # keep raw frames off-GPU; H100 has room either way
            dtype=dtype,
        )
        # add_text_prompt may mutate the session in place or return a new one.
        session = sam_proc.add_text_prompt(inference_session=session, text=prompt) or session

        # frame_idx -> {obj_id -> (mask bool HxW, score float)}
        per_frame: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
        with torch.inference_mode():
            for out in sam_model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(frames)
            ):
                res = sam_proc.postprocess_outputs(session, out)
                fidx = int(getattr(out, "frame_idx", len(per_frame)))
                obj_ids = res["object_ids"].tolist()
                masks   = res["masks"].cpu().numpy().astype(bool)   # (num_obj, H, W)
                scores  = res["scores"].float().cpu().numpy()       # (num_obj,)
                per_frame[fidx] = {
                    int(oid): (masks[i], float(scores[i]))
                    for i, oid in enumerate(obj_ids)
                }

    if not per_frame or all(not d for d in per_frame.values()):
        raise RuntimeError(
            f'SAM3 found no object matching "{prompt}". Try a simpler/visual '
            'phrase (e.g. "the ball" instead of "the bouncing tennis ball").'
        )

    # Pick the most persistent, confident object: rank by (frames present, avg score).
    presence: dict[int, list[float]] = {}
    for d in per_frame.values():
        for oid, (_mask, score) in d.items():
            presence.setdefault(oid, []).append(score)
    target_id = max(presence, key=lambda oid: (len(presence[oid]), float(np.mean(presence[oid]))))

    masks_by_frame = {
        fidx: d[target_id][0] for fidx, d in per_frame.items() if target_id in d
    }
    return masks_by_frame, target_id


# ─── DINOv2 embedding of the tracked object's crop ────────────────────────────

def _crop_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Padded XYXY bbox of a boolean mask, or None if degenerate."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    if bw < MIN_CROP_PX or bh < MIN_CROP_PX:
        return None
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    H, W = mask.shape
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px + 1), min(H, y1 + py + 1))


def _embed_track(
    frames: list[np.ndarray], masks_by_frame: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    DINOv2-embed the object crop for each frame it is present in.

    Returns
    -------
    present_frames : (K,) int frame indices that had a usable crop, sorted
    embeddings     : (K, 768) L2-normalised float32 descriptors aligned to them
    """
    from PIL import Image

    m = _load_models()
    torch, device, dtype = m["torch"], m["device"], m["dtype"]
    dino_model, dino_proc = m["dino_model"], m["dino_proc"]

    present_frames: list[int] = []
    crops: list[Image.Image] = []
    for fidx in sorted(masks_by_frame):
        box = _crop_box(masks_by_frame[fidx])
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crop = frames[fidx][y0:y1, x0:x1]
        if crop.size == 0:
            continue
        present_frames.append(fidx)
        crops.append(Image.fromarray(crop))

    if len(crops) < 2:
        raise RuntimeError(
            "Tracked object yielded fewer than 2 usable crops — cannot measure "
            "embedding drift. The object may be too small or briefly visible."
        )

    feats: list[np.ndarray] = []
    with _GPU_LOCK, torch.inference_mode():
        for i in range(0, len(crops), EMBED_BATCH):
            batch = crops[i:i + EMBED_BATCH]
            inputs = dino_proc(images=batch, return_tensors="pt").to(device, dtype)
            out = dino_model(**inputs)
            cls = out.last_hidden_state[:, 0]                       # (B, 768) CLS token
            cls = torch.nn.functional.normalize(cls, dim=-1)
            feats.append(cls.float().cpu().numpy())

    return np.asarray(present_frames), np.concatenate(feats, axis=0)


def _drift_curve(
    frames_idx: np.ndarray, embeddings: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine-distance drift to the first embedding, on a normalised [0,1] timeline.

    The x-axis is the object's position in the *whole* video (frame / last frame),
    so GT and AI share a true timeline even when the object is only tracked for
    part of each clip.  Embeddings are L2-normalised, so drift = 1 - <e_t, e_0>.
    Returns (t in [0,1], drift in [0,2]).
    """
    ref = embeddings[0]
    drift = 1.0 - embeddings @ ref
    span = max(n_frames - 1, 1)
    t = frames_idx.astype(float) / span
    return t, drift


# ─── Plot + preview rendering ─────────────────────────────────────────────────

def _build_drift_plot(
    gt_t: np.ndarray, gt_drift: np.ndarray,
    ai_t: np.ndarray, ai_drift: np.ndarray,
    label_gt: str, label_ai: str, prompt: str,
) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(gt_t, gt_drift, color="#1a54c4", lw=1.8, marker="o", ms=3, label=f"GT — {label_gt}")
    ax.plot(ai_t, ai_drift, color="#b91c1c", lw=1.8, ls="--", marker="s", ms=3, label=f"AI — {label_ai}")
    ax.set_title(f'DINOv2 appearance-embedding drift — "{prompt}"', fontsize=13, fontweight="bold")
    ax.set_xlabel("normalised time (frame / last frame)", fontsize=10)
    ax.set_ylabel("cosine distance to first frame", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    ax.text(0.99, 0.02, "flat = stable identity · spikes = morphing / flicker",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#7a7873")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _overlay_preview(frame_rgb: np.ndarray, mask: np.ndarray, caption_color=(26, 84, 196)) -> bytes:
    """Tint + outline the tracked object on its first-detection frame → PNG bytes."""
    overlay = frame_rgb.copy()
    color = np.array(caption_color, dtype=np.uint8)
    overlay[mask] = (0.45 * color + 0.55 * overlay[mask]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in caption_color), 2)

    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode preview PNG.")
    return png.tobytes()


# ─── Async pipeline entry point ───────────────────────────────────────────────

async def run(gt_path: str, ai_path: str, prompt: Optional[str] = None) -> AsyncGenerator[dict, None]:
    """
    Paired SAM3+DINOv2 tracking. Accepts two video paths and a text prompt
    naming the object of interest. Registered with requires_pair + requires_prompt.
    """
    loop = asyncio.get_event_loop()
    prompt = (prompt or "").strip()
    safe_prompt = html.escape(prompt)          # log text is rendered as HTML by the UI

    if not prompt:
        yield {"type": "error", "text": "Please enter an object to segment (e.g. \"the ball\")."}
        return

    yield {"type": "log", "level": "info", "text": f'Object of interest: "{safe_prompt}"'}
    yield {"type": "log", "level": "info", "text": "Loading models (SAM3 + DINOv2)… first run downloads weights."}
    try:
        await loop.run_in_executor(None, _load_models)
    except Exception as exc:                                        # noqa: BLE001
        yield {"type": "error", "text": html.escape(str(exc))}
        return

    results = {}
    for tag, path, color in (("GT", gt_path, (26, 84, 196)), ("AI", ai_path, (185, 28, 28))):
        name = Path(path).name
        yield {"type": "log", "level": "info", "text": f"[{tag}] Loading video: {name}"}
        try:
            frames, fps = await loop.run_in_executor(None, _load_frames, path)
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "error", "text": f"[{tag}] {html.escape(str(exc))}"}
            return
        yield {"type": "log", "level": "info", "text": f"[{tag}] {len(frames)} frames @ {fps:.1f} fps"}

        yield {"type": "log", "level": "info", "text": f"[{tag}] SAM3 segmenting + tracking \"{safe_prompt}\"…"}
        try:
            masks_by_frame, target_id = await loop.run_in_executor(None, _segment_track, frames, prompt)
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "error", "text": f"[{tag}] {html.escape(str(exc))}"}
            return
        yield {"type": "log", "level": "info",
               "text": f"[{tag}] tracked object #{target_id} across {len(masks_by_frame)} frames"}

        # "What got segmented" preview — first frame the object appears in.
        first_fidx = min(masks_by_frame)
        try:
            png = await loop.run_in_executor(
                None, _overlay_preview, frames[first_fidx], masks_by_frame[first_fidx], color
            )
            yield {"type": "image", "data": base64.b64encode(png).decode(), "mime": "image/png",
                   "caption": f"{tag}: SAM3 segmentation of \"{safe_prompt}\" (frame {first_fidx})"}
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn", "text": f"[{tag}] preview skipped: {html.escape(str(exc))}"}

        yield {"type": "log", "level": "info", "text": f"[{tag}] DINOv2 embedding object crops…"}
        try:
            fidx_arr, embs = await loop.run_in_executor(None, _embed_track, frames, masks_by_frame)
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "error", "text": f"[{tag}] {html.escape(str(exc))}"}
            return

        t, drift = _drift_curve(fidx_arr, embs, len(frames))
        results[tag] = {"t": t, "drift": drift, "name": name}
        yield {"type": "metric", "label": f"{tag} mean drift", "value": f"{float(drift.mean()):.3f}",
               "sub": "avg cosine dist to frame 0"}
        yield {"type": "metric", "label": f"{tag} max drift", "value": f"{float(drift.max()):.3f}",
               "sub": "peak identity deviation"}

    # ── Comparison: how close is the AI drift profile to GT's? ────────────────
    # Score only over the time window where BOTH videos actually tracked the
    # object, so we don't compare against extrapolated values.
    gt, ai = results["GT"], results["AI"]
    lo = max(float(gt["t"].min()), float(ai["t"].min()))
    hi = min(float(gt["t"].max()), float(ai["t"].max()))
    if hi > lo:
        grid = np.linspace(lo, hi, 50)
        gt_i = np.interp(grid, gt["t"], gt["drift"])
        ai_i = np.interp(grid, ai["t"], ai["drift"])
        mad  = float(np.mean(np.abs(gt_i - ai_i)))        # mean abs drift-profile difference
        # Map MAD (0 = identical profiles) to a 0–100 similarity; 0.3 gap → ~0.
        score = int(round(max(0.0, 1.0 - mad / 0.3) * 100))
        overlap_pct = int(round((hi - lo) * 100))
    else:
        mad, score, overlap_pct = float("nan"), 0, 0      # no shared tracked window

    yield {"type": "metric", "label": "Drift-profile gap", "value": f"{mad:.3f}",
           "sub": f"mean |GT−AI| over {overlap_pct}% shared window"}
    sev_color = "#1a7a3c" if score >= 70 else "#9a6200" if score >= 40 else "#b91c1c"
    yield {"type": "severity", "label": "Appearance-stability match (GT vs AI)", "value": score, "color": sev_color}

    yield {"type": "log", "level": "info", "text": "Rendering embedding-drift plot…"}
    png = await loop.run_in_executor(
        None, _build_drift_plot,
        gt["t"], gt["drift"], ai["t"], ai["drift"], gt["name"], ai["name"], prompt,
    )
    yield {"type": "image", "data": base64.b64encode(png).decode(), "mime": "image/png",
           "caption": "DINOv2 embedding-drift: GT vs AI"}

    yield {"type": "log", "level": "success", "text": "SAM3 + DINOv2 tracking comparison complete."}
    yield {"type": "done"}

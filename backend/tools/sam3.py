"""
Shared SAM3 video segmentation for Stage 3 specialists.
-------------------------------------------------------
Ported from archive_files/sam_track_compare.py (the verified SAM3 path on the
GPU host). Loads facebook/sam3 once per process and returns per-frame boolean
masks for a single tracked object.

facebook/sam3 is GATED — authenticate with `hf auth login` (or HF_TOKEN) using
an approved account before first use. Requires CUDA; callers should catch
RuntimeError and degrade gracefully on CPU-only hosts.
"""
import threading
from typing import Optional

import numpy as np
import cv2

SAM3_MODEL_ID = "facebook/sam3"
MAX_FRAMES    = 80      # cap frames sent through SAM3 (uniformly subsampled)
TARGET_H      = 720     # downscale tall frames before tracking

_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()
_GPU_LOCK  = threading.Lock()


def _load_sam3():
    """Load SAM3 once. Raises RuntimeError with a clear message on failure."""
    if _MODELS:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS:
            return _MODELS

        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — SAM3 segmentation needs a GPU. "
                "Run the backend on the GPU host inside the GPU env."
            )
        device, dtype = "cuda", torch.bfloat16
        try:
            model = (Sam3VideoModel.from_pretrained(SAM3_MODEL_ID, dtype=dtype)
                     .to(device).eval())
            proc = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
        except Exception as exc:                                   # noqa: BLE001
            raise RuntimeError(
                f"Could not load SAM3 ({SAM3_MODEL_ID}). It is a GATED model — "
                "request access on its HuggingFace page and authenticate with "
                "`hf auth login` (or set HF_TOKEN). "
                f"Underlying error: {exc}"
            ) from exc

        _MODELS.update(torch=torch, device=device, dtype=dtype,
                       model=model, proc=proc)
        return _MODELS


def _prep_frames(frames_bgr: list) -> tuple[list, list[int]]:
    """BGR → RGB, downscale tall frames, uniform subsample to MAX_FRAMES.

    Returns (rgb_frames, orig_indices) where orig_indices[i] is the index of
    rgb_frames[i] in the caller's frame list — callers map masks back with it.
    """
    n = len(frames_bgr)
    idx = (list(range(n)) if n <= MAX_FRAMES
           else [int(i) for i in np.linspace(0, n - 1, MAX_FRAMES)])
    rgb = []
    for i in idx:
        bgr = frames_bgr[i]
        h, w = bgr.shape[:2]
        if h > TARGET_H:
            bgr = cv2.resize(bgr, (int(TARGET_H * w / h), TARGET_H))
        rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return rgb, idx


def _run_text_session(rgb: list, orig_idx: list[int], text: str) -> dict:
    """One SAM3 text-prompt session over prepped RGB frames.

    Returns {orig_frame_idx: bool mask} for the most persistent + confident
    object matching the concept (empty dict if nothing matched).
    """
    m = _load_sam3()
    torch, device, dtype = m["torch"], m["device"], m["dtype"]
    model, proc = m["model"], m["proc"]

    with _GPU_LOCK:
        session = proc.init_video_session(
            video=rgb, inference_device=device,
            video_storage_device="cpu", dtype=dtype,
        )
        session = proc.add_text_prompt(inference_session=session, text=text) or session

        per_frame: dict[int, dict[int, tuple]] = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(rgb)
            ):
                res = proc.postprocess_outputs(session, out)
                fidx = int(getattr(out, "frame_idx", len(per_frame)))
                obj_ids = res["object_ids"].tolist()
                masks = res["masks"].cpu().numpy().astype(bool)
                scores = res["scores"].float().cpu().numpy()
                per_frame[fidx] = {int(o): (masks[i], float(scores[i]))
                                   for i, o in enumerate(obj_ids)}

    presence: dict[int, list[float]] = {}
    for d in per_frame.values():
        for oid, (_mk, sc) in d.items():
            presence.setdefault(oid, []).append(sc)
    if not presence:
        return {}
    target = max(presence, key=lambda o: (len(presence[o]), float(np.mean(presence[o]))))
    return {orig_idx[f]: d[target][0] for f, d in per_frame.items() if target in d}


def segment_concepts(frames_bgr: list, concepts: list[str]) -> dict:
    """Mask-track each named subject through the clip (one session per name).

    Returns {"subjects": {name: {orig_frame_idx: bool mask}}, "scale": float,
             "sampled_frames": [orig_frame_idx...]}.
    Subjects whose concept matched nothing are omitted. Masks are at the
    (possibly downscaled) tracking resolution — resize by 1/scale to overlay
    on original frames. SAM3 only sees `sampled_frames` (uniform subsample,
    ≤ MAX_FRAMES) — compute presence/coverage against that grid, not against
    the full frame count.
    """
    rgb, orig_idx = _prep_frames(frames_bgr)
    scale = rgb[0].shape[0] / frames_bgr[0].shape[0]
    subjects = {}
    for name in concepts:
        masks = _run_text_session(rgb, orig_idx, name)
        if masks:
            subjects[name] = masks
    return {"subjects": subjects, "scale": scale, "sampled_frames": list(orig_idx)}


def segment_video(frames_bgr: list, *, box: Optional[tuple] = None,
                  text: str = "the main moving object") -> dict:
    """Track ONE object through the clip; return its per-frame masks.

    Prompting: tries the box (XYXY, coordinates in the ORIGINAL frame size) on
    the first frame if the installed transformers exposes a box-prompt API;
    otherwise falls back to the verified text-prompt path. When several objects
    match, keeps the most persistent + confident one (as sam_track_compare did).

    Returns {"masks": {orig_frame_idx: bool (h, w) mask}, "prompt_mode": str,
             "scale": float (mask size / original size), "target_id": int}.
    Masks are at the (possibly downscaled) tracking resolution; callers scale
    coordinates by 1/scale to map boxes back onto original frames.
    """
    m = _load_sam3()
    torch, device, dtype = m["torch"], m["device"], m["dtype"]
    model, proc = m["model"], m["proc"]

    rgb, orig_idx = _prep_frames(frames_bgr)
    H0 = frames_bgr[0].shape[0]
    scale = rgb[0].shape[0] / H0

    with _GPU_LOCK:
        session = proc.init_video_session(
            video=rgb, inference_device=device,
            video_storage_device="cpu", dtype=dtype,
        )

        prompt_mode = "text"
        if box is not None:
            sb = [int(c * scale) for c in box]           # box in tracking-res coords
            for name in ("add_box_prompt", "add_new_points_or_boxes", "add_boxes"):
                fn = getattr(proc, name, None)
                if fn is None:
                    continue
                try:
                    session = fn(inference_session=session, frame_idx=0,
                                 obj_ids=[1], boxes=[[sb]]) or session
                    prompt_mode = "box"
                    break
                except Exception:                        # noqa: BLE001
                    continue
        if prompt_mode == "text":
            session = proc.add_text_prompt(inference_session=session, text=text) or session

        per_frame: dict[int, dict[int, tuple]] = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(rgb)
            ):
                res = proc.postprocess_outputs(session, out)
                fidx = int(getattr(out, "frame_idx", len(per_frame)))
                obj_ids = res["object_ids"].tolist()
                masks = res["masks"].cpu().numpy().astype(bool)
                scores = res["scores"].float().cpu().numpy()
                per_frame[fidx] = {int(o): (masks[i], float(scores[i]))
                                   for i, o in enumerate(obj_ids)}

    if not per_frame or all(not d for d in per_frame.values()):
        raise RuntimeError(
            f'SAM3 found no object (prompt mode: {prompt_mode}, text "{text}"). '
            "Try a simpler visual phrase in the object prompt setting."
        )

    presence: dict[int, list[float]] = {}
    for d in per_frame.values():
        for oid, (_mk, sc) in d.items():
            presence.setdefault(oid, []).append(sc)
    target = max(presence, key=lambda o: (len(presence[o]), float(np.mean(presence[o]))))

    masks = {orig_idx[f]: d[target][0] for f, d in per_frame.items() if target in d}
    return {"masks": masks, "prompt_mode": prompt_mode,
            "scale": scale, "target_id": int(target)}


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Binary mask → PNG bytes (~KBs) for cheap storage on the evidence bus."""
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def decode_mask_png(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr > 127


def border_ratio(mask: np.ndarray) -> float:
    """How much of a mask hugs the frame edge (≥~0.5 = partially out of frame)."""
    edge = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return edge / max(np.sqrt(max(mask.sum(), 1)), 1.0)


def usable_frames(masks: dict) -> list[int]:
    """Frame indices whose masks are usable for appearance comparison.

    Filters ONLY frames where the object is partially out of frame (heavy
    border contact) or the mask is a degenerate sliver. Deliberately does NOT
    filter area outliers elsewhere: a mid-video area anomaly (fragmentation,
    morphing) is the physics defect being hunted, not noise.
    """
    present = sorted(masks)
    good = [fi for fi in present
            if int(masks[fi].sum()) >= 64 and border_ratio(masks[fi]) < 0.5]
    return good if len(good) >= 2 else present


def representative_frames(masks: dict, k: int) -> list[int]:
    """Pick ≤k usable frame indices, evenly spaced."""
    pool = usable_frames(masks)
    if len(pool) <= k:
        return pool
    return [pool[i] for i in np.linspace(0, len(pool) - 1, k, dtype=int)]


def mask_box(mask: np.ndarray, pad_frac: float = 0.25,
             min_px: int = 16) -> Optional[tuple[int, int, int, int]]:
    """Padded XYXY bbox of a boolean mask (mask-resolution coords), or None."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return None
    px = max(min_px, int(bw * pad_frac))
    py = max(min_px, int(bh * pad_frac))
    H, W = mask.shape
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px + 1), min(H, y1 + py + 1))

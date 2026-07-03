"""
SAM 3 promptable-concept video segmentation + tracking.

Process-wide singletons (model load is ~3.4 GB / ~10 s) guarded by a load lock;
a GPU lock serialises inference since FastAPI may call concurrently.
Ported/generalised from archive_files/sam_track_compare.py — that file tracked a
single object from one text prompt; here we return EVERY instance of a concept so
the object tracker can label each segment individually.
"""
import threading
from typing import Optional

import numpy as np

SAM3_MODEL_ID = "facebook/sam3"

_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def load_sam3(device: str = "cuda:0"):
    """Load SAM3 video model + processor once. Raises a clear error if gated/absent."""
    if _MODELS:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS:
            return _MODELS
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable — SAM3 needs a GPU.")
        dtype = torch.bfloat16
        try:
            model = Sam3VideoModel.from_pretrained(SAM3_MODEL_ID, dtype=dtype).to(device).eval()
            proc = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load SAM3 ({SAM3_MODEL_ID}); it is GATED — authenticate "
                f"with an approved HF account (`hf auth login`). Error: {exc}"
            ) from exc
        _MODELS.update(torch=torch, device=device, dtype=dtype, model=model, proc=proc)
        return _MODELS


def segment_concept(
    frames: list[np.ndarray],
    concept: str,
    min_presence: float = 0.15,
    min_score: float = 0.45,
    max_instances: int = 6,
) -> list[dict]:
    """
    Segment & track every instance of `concept` (a short text phrase, e.g. "ball")
    across `frames` (list of RGB uint8 HxWx3).

    Returns a list of instance dicts, sorted by persistence then confidence:
        {"concept": str, "sam_id": int, "masks": {frame_idx: bool mask},
         "mean_score": float, "n_frames": int}
    Instances present in fewer than `min_presence` of frames or below `min_score`
    mean confidence are dropped.
    """
    m = load_sam3()
    torch, device, dtype = m["torch"], m["device"], m["dtype"]
    model, proc = m["model"], m["proc"]
    n = len(frames)

    with _GPU_LOCK:
        session = proc.init_video_session(
            video=frames, inference_device=device,
            video_storage_device="cpu", dtype=dtype,
        )
        session = proc.add_text_prompt(inference_session=session, text=concept) or session
        per_frame: dict[int, dict[int, tuple]] = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=n
            ):
                res = proc.postprocess_outputs(session, out)
                fidx = int(getattr(out, "frame_idx", len(per_frame)))
                oids = res["object_ids"].tolist()
                masks = res["masks"].cpu().numpy().astype(bool)   # (k, H, W)
                scores = res["scores"].float().cpu().numpy()      # (k,)
                per_frame[fidx] = {
                    int(o): (masks[i], float(scores[i])) for i, o in enumerate(oids)
                }

    # Group masks per SAM object id
    by_id: dict[int, dict[int, np.ndarray]] = {}
    score_by_id: dict[int, list[float]] = {}
    for fidx, d in per_frame.items():
        for oid, (mask, sc) in d.items():
            if mask.any():
                by_id.setdefault(oid, {})[fidx] = mask
                score_by_id.setdefault(oid, []).append(sc)

    instances = []
    for oid, masks in by_id.items():
        n_frames = len(masks)
        mean_score = float(np.mean(score_by_id[oid]))
        if n_frames < max(2, int(min_presence * n)) or mean_score < min_score:
            continue
        instances.append({
            "concept": concept, "sam_id": oid, "masks": masks,
            "mean_score": mean_score, "n_frames": n_frames,
        })
    instances.sort(key=lambda d: (-d["n_frames"], -d["mean_score"]))
    return instances[:max_instances]


def mask_bbox(mask: np.ndarray, pad: float = 0.08, min_px: int = 8) -> Optional[tuple]:
    """Padded XYXY bbox of a boolean mask, or None if degenerate."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0, y1 - y0
    if bw < min_px or bh < min_px:
        return None
    px, py = int(bw * pad), int(bh * pad)
    H, W = mask.shape
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px + 1), min(H, y1 + py + 1))


def mask_iou(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    """Mean per-frame IoU over frames where both instances are present."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    ious = []
    for f in shared:
        inter = np.logical_and(a[f], b[f]).sum()
        union = np.logical_or(a[f], b[f]).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

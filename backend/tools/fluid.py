"""Shared fluid-analysis toolkit: dense flow, water masking, Helmholtz
decomposition, and small plotting helpers.

Hybrid auto-detect: prefer the GPU/high-fidelity backend, fall back to the
CPU/light one. Backend choice is cached per process.
"""
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np


def helmholtz(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Divergence (∂u/∂x + ∂v/∂y) and curl (∂v/∂x − ∂u/∂y) via central diff.
    axis=1 is x (columns), axis=0 is y (rows)."""
    du_dx = np.gradient(u, axis=1)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    dv_dy = np.gradient(v, axis=0)
    divergence = du_dx + dv_dy
    curl       = dv_dx - du_dy
    return divergence.astype(np.float32), curl.astype(np.float32)


def flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v).astype(np.float32)


def masked_mean(field: np.ndarray, mask: np.ndarray) -> float:
    """Mean of `field` over True pixels of `mask`; 0.0 if the mask is empty."""
    m = mask.astype(bool)
    if not m.any():
        return 0.0
    return float(field[m].mean())


def severity_color(score: int) -> str:
    return "#E24B4A" if score > 40 else "#EF9F27" if score > 15 else "#4CAF50"


_FLOW_BACKEND: Optional[str] = None
_RAFT_MODEL = None


def _detect_flow_backend() -> str:
    global _FLOW_BACKEND
    if _FLOW_BACKEND is not None:
        return _FLOW_BACKEND
    backend = "farneback"
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            from torchvision.models.optical_flow import raft_small  # noqa: F401
            backend = "raft"
    except Exception:
        backend = "farneback"
    _FLOW_BACKEND = backend
    return backend


def _raft_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    global _RAFT_MODEL
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    if _RAFT_MODEL is None:
        _RAFT_MODEL = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).eval().cuda()

    def prep(g: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t.unsqueeze(0).cuda()

    h, w = prev_gray.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8  # RAFT needs /8 dims
    a, b = prep(prev_gray), prep(curr_gray)
    a = F.pad(a, (0, pw, 0, ph)); b = F.pad(b, (0, pw, 0, ph))
    with torch.no_grad():
        flow = _RAFT_MODEL(a, b)[-1][0].cpu().numpy()  # (2, H, W)
    u, v = flow[0, :h, :w], flow[1, :h, :w]
    return u.astype(np.float32), v.astype(np.float32)


def dense_flow(prev_gray: np.ndarray, curr_gray: np.ndarray,
               backend: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    use = _detect_flow_backend() if backend == "auto" else backend
    if use in ("raft", "gpu"):
        try:
            return _raft_flow(prev_gray, curr_gray)
        except Exception:
            use = "farneback"
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def _detect_mask_method() -> str:
    # SAM3 is a future drop-in; HSV is the reliable default that runs anywhere.
    return "hsv"


def _hsv_water_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 90) & (h <= 140) & (s >= 40)      # blue/teal water
    foam = (s <= 45) & (v >= 180)                  # bright white foam/spray
    raw = (blue | foam).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)
    return cleaned > 0


def water_mask(frame_bgr: np.ndarray, method: str = "auto") -> Tuple[np.ndarray, str]:
    use = _detect_mask_method() if method == "auto" else method
    # `use == "hsv"` is the only implemented path; any other value falls through to HSV.
    return _hsv_water_mask(frame_bgr), "hsv"


def resize_frames(frames: List[np.ndarray], max_height: int = 480) -> List[np.ndarray]:
    """Downscale frames to `max_height` (aspect-preserving, even width) for a
    consistent working resolution — keeps flow-based thresholds comparable
    across input resolutions and speeds up dense flow. No-op if already short
    enough or max_height <= 0."""
    if not frames or max_height <= 0:
        return frames
    h, w = frames[0].shape[:2]
    if h <= max_height:
        return frames
    nw = int(round(w * max_height / h))
    nw -= nw % 2
    return [cv2.resize(f, (nw, max_height)) for f in frames]


def motion_mask(frames: List[np.ndarray], min_floor: float = 6.0) -> np.ndarray:
    """Sequence-level water-activity mask: pixels whose grayscale intensity
    varies over time above an *absolute* floor (moving surface, ripples,
    splashes) — the static background falls below it. Works where colour
    heuristics fail (clear/dark water). An absolute (not percentile) floor is
    deliberate: it makes coverage reflect *how much* of the frame actually
    moves, so a globally-moving scene (camera motion) yields near-full coverage
    and is detectable as such by `resolve_water_region`."""
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    tstd = g.std(axis=0)
    raw = (tstd > float(min_floor)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return raw > 0


def resolve_water_region(frames: List[np.ndarray],
                         mask_method: str = "auto") -> Tuple[Optional[np.ndarray], str]:
    """Decide the water-region masking strategy for a whole clip.

    Returns (static_mask | None, label). A static mask is applied to every
    frame pair; `None` means 'compute the per-frame HSV mask in the loop'.
    `auto` uses the motion mask when its coverage is plausible (static camera),
    otherwise falls back to per-frame HSV.
    """
    h, w = frames[0].shape[:2]
    if mask_method == "none":
        return np.ones((h, w), dtype=bool), "none"
    if mask_method == "motion":
        return motion_mask(frames), "motion"
    if mask_method == "hsv":
        return None, "hsv"
    mm = motion_mask(frames)            # auto
    cov = float(mm.mean())
    # Use the motion region when activity is localized enough to be meaningful.
    # Upper bound 0.80 tolerates busy water scenes (and AI shimmer that inflates
    # coverage); ~full-frame activity (a moving camera) falls back to HSV.
    if 0.01 <= cov <= 0.80:
        return mm, "motion"
    return None, "hsv"


# ---------------------------------------------------------------------------
# SAM3 learned water segmentation (optional; GPU + HF-gated facebook/sam3).
# Mirrors the loader/segmenter in archive_files/sam_track_compare.py. Falls
# back gracefully (raises) when torch/CUDA/SAM3 are unavailable, so callers can
# drop to the motion/HSV mask.
# ---------------------------------------------------------------------------
_SAM3: dict = {}
_SAM3_LOAD_LOCK = threading.Lock()
_SAM3_GPU_LOCK = threading.Lock()
SAM3_MODEL_ID = "facebook/sam3"


def _load_sam3() -> dict:
    if _SAM3:
        return _SAM3
    with _SAM3_LOAD_LOCK:
        if _SAM3:
            return _SAM3
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 needs a CUDA GPU — run the backend on the GPU host.")
        dtype = torch.bfloat16
        model = Sam3VideoModel.from_pretrained(SAM3_MODEL_ID, dtype=dtype).to("cuda").eval()
        proc = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
        _SAM3.update(torch=torch, model=model, proc=proc, device="cuda", dtype=dtype)
        return _SAM3


def sam3_water_masks(frames: List[np.ndarray], prompt: str = "water",
                     max_frames: int = 80) -> List[np.ndarray]:
    """Per-frame boolean water masks via SAM3 promptable video segmentation.

    Returns one mask per input frame (the union of all `prompt` instances in
    that frame). SAM3 runs on a uniform subsample (≤ max_frames) for memory;
    each original frame is mapped to its nearest subsampled mask. Requires
    torch+CUDA and gated `facebook/sam3` access; raises RuntimeError otherwise.
    """
    m = _load_sam3()
    torch, model, proc = m["torch"], m["model"], m["proc"]
    device, dtype = m["device"], m["dtype"]
    n = len(frames)
    h, w = frames[0].shape[:2]
    idx = np.linspace(0, n - 1, min(n, max_frames)).astype(int)
    rgb = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx]

    with _SAM3_GPU_LOCK:
        session = proc.init_video_session(
            video=rgb, inference_device=device, video_storage_device="cpu", dtype=dtype)
        session = proc.add_text_prompt(inference_session=session, text=prompt) or session
        sub: dict = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(
                    inference_session=session, max_frame_num_to_track=len(rgb)):
                res = proc.postprocess_outputs(session, out)
                fidx = int(getattr(out, "frame_idx", len(sub)))
                masks = res["masks"].cpu().numpy().astype(bool)          # (num_obj, H, W)
                sub[fidx] = np.any(masks, axis=0) if len(masks) else np.zeros((h, w), bool)

    if not sub or not any(mm.any() for mm in sub.values()):
        raise RuntimeError(f'SAM3 found no "{prompt}" region (try a different water_prompt).')

    sub_list = []
    for k in range(len(rgb)):
        mm = sub.get(k, np.zeros((h, w), bool))
        if mm.shape != (h, w):
            mm = cv2.resize(mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        sub_list.append(mm)
    return [sub_list[int(np.argmin(np.abs(idx - i)))] for i in range(n)]


def compute_flow_sequence(frames: List[np.ndarray], backend: str = "auto",
                          mask_method: str = "auto",
                          static_mask: Optional[np.ndarray] = None,
                          frame_masks: Optional[List[np.ndarray]] = None) -> List[dict]:
    """Per-pair dense flow + Helmholtz + water mask. Mask precedence per pair i
    (the second frame of the pair): `frame_masks[i]` if given (e.g. SAM3),
    else a clip-level `static_mask` (motion/none), else a per-frame HSV mask."""
    if len(frames) < 2:
        return []
    if frame_masks is None and static_mask is None and mask_method not in ("hsv", "sam3"):
        static_mask, _ = resolve_water_region(frames, mask_method)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    seq: List[dict] = []
    for i in range(1, len(frames)):
        u, v = dense_flow(grays[i - 1], grays[i], backend=backend)
        div, curl = helmholtz(u, v)
        mag = flow_magnitude(u, v)
        if frame_masks is not None:
            mask = frame_masks[i]
        elif static_mask is not None:
            mask = static_mask
        else:
            mask = water_mask(frames[i], method="hsv")[0]
        seq.append({"u": u, "v": v, "mask": mask, "div": div, "curl": curl, "mag": mag})
    return seq


def timeseries_figure(time, traces, title: str,
                      threshold: Optional[float] = None,
                      ythresh_label: str = "") -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, yvals, color in traces:
        fig.add_trace(go.Scatter(x=list(time), y=list(yvals), mode="lines",
                                 name=name, line=dict(color=color, width=1.6)))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color="red", dash="dash", width=1.2),
                      annotation_text=ythresh_label or f"threshold = {threshold}",
                      annotation_position="top right")
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a1917")),
        height=420, legend=dict(orientation="h", y=1.08),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
    )
    return fig.to_json()

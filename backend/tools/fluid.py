"""Shared fluid-analysis toolkit: dense flow, water masking, Helmholtz
decomposition, and small plotting helpers.

Hybrid auto-detect: prefer the GPU/high-fidelity backend, fall back to the
CPU/light one. Backend choice is cached per process.
"""
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

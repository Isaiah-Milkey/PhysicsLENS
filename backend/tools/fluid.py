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

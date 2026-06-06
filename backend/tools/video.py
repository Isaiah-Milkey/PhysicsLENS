"""Shared video I/O and frame utilities."""
import io, base64
from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    step: int = 1,
) -> Tuple[List[np.ndarray], float]:
    """Return (frames, fps). `step` keeps every Nth frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, fps


def sample_frames(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """Uniformly sample exactly n frames (or all if len ≤ n)."""
    if len(frames) <= n:
        return frames
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in idxs]


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

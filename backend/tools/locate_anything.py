"""
Shared NVIDIA LocateAnything-3B client — open-set object detection for
semantic labeling of Stage 2 object tracks.

Single-image only (no video/tracking of its own): we call it once per video
and match its grounded boxes onto the existing LK object tracks by IOU.

License: NVIDIA non-commercial research license (academic/non-profit use
only — see https://huggingface.co/nvidia/LocateAnything-3B).

Output format (documented on the model card):
  <ref>label</ref><box><x1><y1><x2><y2></box>  — coords are ints in [0, 1000]
  <box>none</box>                              — no match
"""
import re
import threading
from typing import Optional

import cv2
import numpy as np

MODEL_ID = "nvidia/LocateAnything-3B"
DEFAULT_CATEGORY = "physical object"

_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()

_BOX_RE = re.compile(
    r"(?:<ref>(?P<label>.*?)</ref>)?<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)


def _load_model():
    """Load model + processor directly (trust_remote_code) — NOT via pipeline().

    pipeline("image-text-to-text", ...) resolves the model class through
    AutoModelForImageTextToText's registry, which this custom architecture
    isn't registered in on current transformers — it raises "Unrecognized
    configuration class". AutoModel + AutoProcessor + chat template + generate()
    is the standard remote-code path and doesn't hit that registry.
    """
    if _MODELS:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS:
            return _MODELS
        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — LocateAnything-3B needs a GPU. "
                "Run the backend on the GPU host."
            )
        try:
            model = AutoModel.from_pretrained(
                MODEL_ID, trust_remote_code=True, dtype="auto"
            ).to("cuda").eval()
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        except Exception as exc:                                    # noqa: BLE001
            raise RuntimeError(
                f"Could not load {MODEL_ID}: {exc}. If this is an access/auth "
                "error, run `hf auth login` with an approved account."
            ) from exc
        _MODELS.update(model=model, processor=processor, torch=torch)
        return _MODELS


def detect(frame_bgr: np.ndarray, category: str = DEFAULT_CATEGORY) -> list[dict]:
    """Detect + label distinct objects in one frame.

    Returns [{"label": str, "box": (x0, y0, x1, y1)}] in pixel coordinates
    (XYXY, origin top-left, matching tools/tracking.py's box convention).
    Raises RuntimeError if the model/GPU is unavailable.
    """
    from PIL import Image

    m = _load_model()
    model, processor, torch = m["model"], m["processor"], m["torch"]
    H, W = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",
             "text": f"Locate all the instances that matches the following description: {category}."},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=True)[0]

    out = []
    for m in _BOX_RE.finditer(raw):
        x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
        label = (m.group("label") or category).strip()
        out.append({
            "label": label,
            "box": (int(x1 / 1000 * W), int(y1 / 1000 * H),
                    int(x2 / 1000 * W), int(y2 / 1000 * H)),
        })
    return out


def iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_label(track_box: tuple, detections: list[dict],
                min_iou: float = 0.15) -> Optional[str]:
    """Best-IOU label for a track's box, or None if nothing clears min_iou."""
    best_label, best_iou = None, min_iou
    for d in detections:
        score = iou(track_box, d["box"])
        if score > best_iou:
            best_iou, best_label = score, d["label"]
    return best_label

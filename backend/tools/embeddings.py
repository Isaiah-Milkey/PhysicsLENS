"""Frame-level visual embeddings: DINOv2, CLIP, SigLIP.

All loaders return a cached (model, processor) bundle — repeated calls within
one server process reuse the same instance. All embedders accept a list of
BGR uint8 frames, run batched on CUDA when available, and return
L2-normalised float32 arrays of shape (T, D).

DINOv2 is facebook/dinov2-base via HuggingFace: the same checkpoint and
extraction path (pooler output == CLS token, L2-normalised) used by the
world_models_evaluation notebooks, the Physics-IQ baseline statistics,
scripts/check_models.py, and the archived SAM3 pipeline. Keep them aligned —
latent-kinematics thresholds and baselines are only meaningful in this space.
"""
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np

BATCH_SIZE = 16


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_pil_rgb(frames: List[np.ndarray]) -> list:
    from PIL import Image
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def _l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms < 1e-8, 1.0, norms)


def _embed_batched(frames: List[np.ndarray], encode) -> np.ndarray:
    """Run `encode(pil_batch) -> tensor` over BGR frames in batches."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(frames), BATCH_SIZE):
            batch = _to_pil_rgb(frames[i : i + BATCH_SIZE])
            out.append(encode(batch).float().cpu().numpy())
    return _l2_normalize(np.concatenate(out, axis=0).astype(np.float32))


# ── DINOv2 ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_dinov2(variant: str = "facebook/dinov2-base") -> Tuple:
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = AutoModel.from_pretrained(variant).to(device()).eval()
    return model, processor


def embed_frames_dinov2(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 768) L2-normalised DINOv2 descriptors (pooler output == CLS token)."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        out = model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    return _embed_batched(frames, encode)


# ── CLIP ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_clip(variant: str = "openai/clip-vit-base-patch32") -> Tuple:
    from transformers import AutoImageProcessor, CLIPModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = CLIPModel.from_pretrained(variant).to(device()).eval()
    return model, processor


def embed_frames_clip(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 512) L2-normalised CLIP image features."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        return model.get_image_features(**inputs)

    return _embed_batched(frames, encode)


# ── SigLIP ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_siglip(variant: str = "google/siglip-base-patch16-224") -> Tuple:
    from transformers import AutoImageProcessor, SiglipModel
    processor = AutoImageProcessor.from_pretrained(variant)
    model = SiglipModel.from_pretrained(variant).to(device()).eval()
    return model, processor


def embed_frames_siglip(frames: List[np.ndarray], bundle) -> np.ndarray:
    """(T, 768) L2-normalised SigLIP image features."""
    model, processor = bundle
    dev = next(model.parameters()).device

    def encode(batch):
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        return model.get_image_features(**inputs)

    return _embed_batched(frames, encode)

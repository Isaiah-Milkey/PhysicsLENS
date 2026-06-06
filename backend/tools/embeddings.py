"""Frame-level visual embeddings: DINOv2, CLIP, SigLIP."""
from typing import List, Tuple

import cv2
import numpy as np


# ── DINOv2 ────────────────────────────────────────────────────────────────────

def load_dinov2(variant: str = "dinov2_vits14"):
    import torch
    model = torch.hub.load("facebookresearch/dinov2", variant, pretrained=True)
    model.eval()
    return model


def embed_frames_dinov2(frames: List[np.ndarray], model) -> np.ndarray:
    import torch
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t   = tf(Image.fromarray(rgb)).unsqueeze(0)
            embeddings.append(model(t).squeeze(0).cpu().numpy())
    return np.stack(embeddings)   # (T, D)


# ── CLIP ──────────────────────────────────────────────────────────────────────

def load_clip(variant: str = "ViT-B/32") -> Tuple:
    import clip  # pip install git+https://github.com/openai/CLIP.git
    model, preprocess = clip.load(variant)
    model.eval()
    return model, preprocess


def embed_frames_clip(frames: List[np.ndarray], model, preprocess) -> np.ndarray:
    import torch
    from PIL import Image

    embeddings = []
    with torch.no_grad():
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t   = preprocess(Image.fromarray(rgb)).unsqueeze(0)
            embeddings.append(model.encode_image(t).squeeze(0).float().cpu().numpy())
    return np.stack(embeddings)


# ── SigLIP ────────────────────────────────────────────────────────────────────

def load_siglip(variant: str = "google/siglip-base-patch16-224") -> Tuple:
    from transformers import SiglipProcessor, SiglipModel
    model     = SiglipModel.from_pretrained(variant).eval()
    processor = SiglipProcessor.from_pretrained(variant)
    return model, processor


def embed_frames_siglip(frames: List[np.ndarray], model, processor) -> np.ndarray:
    import torch
    from PIL import Image

    embeddings = []
    with torch.no_grad():
        for frame in frames:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
            out    = model.get_image_features(**inputs)
            embeddings.append(out.squeeze(0).cpu().numpy())
    return np.stack(embeddings)

"""
Local open-source VLM helpers (no API key needed).

A registry of open-weight vision-language models, each loadable as a
process-wide singleton:
  • name_objects()     — list the distinct physical objects in a scene (labels
    that the SAM3 segmenter then turns into masks).
  • score_video()      — multi-frame physics-plausibility judgement (scalar).
  • analyze_physics()  — multi-frame physics-anomaly extraction: which physical
    laws are broken, what is observed, and how it violates real-world physics.

Only one local VLM is kept on the GPU at a time — selecting a different model
in the UI evicts the previous one (each is 5–17 GB of VRAM).

Measured separation (scripts/vlm_multimodel_eval.py, 5 AI vs 5 real clips,
multi-frame protocol): Qwen2.5-VL-7B AUC 0.92 · InternVL3-8B AUC 0.70 ·
SmolVLM2-2.2B AUC 0.50.
"""
import gc
import json
import re
import threading
from typing import Optional

import cv2
import numpy as np

# key → registry entry. `auc` is the measured AI-vs-real rank AUC from
# scripts/vlm_multimodel_eval.py; `vram_gb` is approximate bf16 footprint.
LOCAL_VLMS = {
    "qwen2.5-vl-7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "label": "Qwen2.5-VL 7B", "vram_gb": 17, "auc": 0.92,
    },
    "internvl3-8b": {
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "label": "InternVL3 8B", "vram_gb": 18, "auc": 0.70,
    },
    "smolvlm2-2.2b": {
        "hf_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "label": "SmolVLM2 2.2B", "vram_gb": 5, "auc": 0.50,
    },
}
DEFAULT_VLM = "qwen2.5-vl-7b"

_VLM: dict = {}          # currently-loaded bundle: {key, torch, model, proc, device}
_LOAD_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def load_local_vlm(model_key: str = DEFAULT_VLM, device: str = "cuda:0"):
    """Load (or switch to) a registered local VLM. Evicts any other loaded model."""
    if model_key not in LOCAL_VLMS:
        raise ValueError(f"Unknown local VLM '{model_key}'. Options: {list(LOCAL_VLMS)}")
    if _VLM.get("key") == model_key:
        return _VLM
    with _LOAD_LOCK:
        if _VLM.get("key") == model_key:
            return _VLM
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if _VLM:  # evict the previous model before loading a new one
            _VLM.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        hf_id = LOCAL_VLMS[model_key]["hf_id"]
        proc = AutoProcessor.from_pretrained(hf_id)
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=torch.bfloat16, device_map=device
        ).eval()
        _VLM.update(key=model_key, torch=torch, model=model, proc=proc, device=device)
        return _VLM


def load_qwen_vl(device: str = "cuda:0"):
    """Back-compat alias used by the SAM3 object tracker's scene naming."""
    return load_local_vlm(DEFAULT_VLM, device)


def _to_pil(frame_rgb: np.ndarray, max_side: int = 640):
    from PIL import Image
    h, w = frame_rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame_rgb = cv2.resize(frame_rgb, (int(w * s), int(h * s)))
    return Image.fromarray(frame_rgb)


def _generate(images: list, prompt: str, max_new_tokens: int = 256,
              model_key: str = DEFAULT_VLM) -> str:
    m = load_local_vlm(model_key)
    torch, model, proc = m["torch"], m["model"], m["proc"]
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    with _GPU_LOCK:
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]


def _parse_json(raw: str) -> Optional[dict | list]:
    raw = raw.replace("{{", "{").replace("}}", "}")  # some models echo literal {{ }}
    m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:  # tolerate trailing junk
            s = m.group(0)
            return json.loads(s[:s.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            return None


_NAME_PROMPT = (
    "List the distinct, physically-interacting objects visible in this scene "
    "(things that move, fall, collide, pour, or deform). Use short, concrete "
    "noun phrases a segmentation model could find (e.g. \"ball\", \"wooden block\", "
    "\"glass\", \"water\"). Ignore the static background, walls, floor, and sky.\n"
    'Reply with VALID JSON ONLY: {"objects": ["...", "..."]}'
)


def name_objects(frames_rgb: list[np.ndarray], max_objects: int = 5) -> list[str]:
    """Return human-readable object names for a scene (uses 2 representative frames)."""
    idx = np.linspace(0, len(frames_rgb) - 1, min(2, len(frames_rgb))).astype(int)
    imgs = [_to_pil(frames_rgb[i]) for i in idx]
    raw = _generate(imgs, _NAME_PROMPT, max_new_tokens=128)
    parsed = _parse_json(raw)
    objs = []
    if isinstance(parsed, dict):
        objs = parsed.get("objects", [])
    elif isinstance(parsed, list):
        objs = parsed
    # Normalise: short noun phrases, dedupe, cap
    seen, out = set(), []
    for o in objs:
        if not isinstance(o, str):
            continue
        o = o.strip().lower().strip(".")
        if 0 < len(o) <= 30 and o not in seen and len(o.split()) <= 3:
            seen.add(o)
            out.append(o)
    return out[:max_objects]


_SCORE_PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. Judge whether the MOTION and INTERACTIONS across frames obey "
    "real-world physics (gravity, momentum, object permanence, rigid-body and fluid "
    "behavior). AI-generated videos often show objects that morph, appear/disappear, "
    "move without forces, or deform implausibly.\n"
    "Reply with VALID JSON ONLY: "
    '{{"suspicion_score": <float 0.0-1.0>, "suspected_failure": "<brief label or null>", '
    '"explanation": "<one or two sentences>", "confidence": <float 0.0-1.0>}}'
)


def score_video(frames_rgb: list[np.ndarray], n_sample: int = 8,
                model_key: str = DEFAULT_VLM) -> dict:
    """Multi-frame physics-plausibility judgement of a whole clip (scalar verdict)."""
    idx = np.linspace(0, len(frames_rgb) - 1, min(n_sample, len(frames_rgb))).astype(int)
    imgs = [_to_pil(frames_rgb[i]) for i in idx]
    raw = _generate(imgs, _SCORE_PROMPT.format(n=len(imgs)), max_new_tokens=256,
                    model_key=model_key)
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return {"suspicion_score": None, "suspected_failure": None,
                "explanation": f"unparseable: {raw[:120]}", "confidence": 0.0}
    return parsed


_ANALYZE_PROMPT = (
    "You are a physics expert reviewing a video for physical plausibility. These {n} "
    "frames are sampled in temporal order from one video (frame 1 = start, frame {n} = "
    "end). Examine the MOTION and INTERACTIONS across frames: gravity and free fall, "
    "momentum and collisions, object permanence, rigid-body and fluid behavior, "
    "contact and support, energy conservation. AI-generated videos often show objects "
    "that morph, appear/disappear, float, pass through each other, or move without "
    "forces.\n"
    "For EACH physics violation you can actually see, report: which physical law or "
    "principle is broken, what you observe in the frames, and how that observation "
    "contradicts real-world physics. If the video is physically plausible, return an "
    "empty violations list.\n"
    "Reply with VALID JSON ONLY - no markdown fences:\n"
    '{{"suspicion_score": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, '
    '"overall_assessment": "<one or two sentences>", '
    '"violations": [{{"law": "<physical law/principle broken>", '
    '"observation": "<what is seen, incl. which frames>", '
    '"why_impossible": "<how this breaks real-world physics>", '
    '"severity": <float 0.0-1.0>}}]}}'
)


def analyze_physics(frames_rgb: list[np.ndarray], n_sample: int = 8,
                    model_key: str = DEFAULT_VLM) -> dict:
    """
    Multi-frame physics-anomaly extraction: a holistic suspicion score plus a list
    of concrete violations, each naming the broken law, the observation, and why
    it is physically impossible.
    """
    idx = np.linspace(0, len(frames_rgb) - 1, min(n_sample, len(frames_rgb))).astype(int)
    imgs = [_to_pil(frames_rgb[i]) for i in idx]
    raw = _generate(imgs, _ANALYZE_PROMPT.format(n=len(imgs)), max_new_tokens=768,
                    model_key=model_key)
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return {"suspicion_score": None, "confidence": 0.0,
                "overall_assessment": f"unparseable: {raw[:150]}", "violations": []}

    # Normalise violations so the pipeline can render them without re-validating.
    violations = []
    for v in parsed.get("violations") or []:
        if not isinstance(v, dict):
            continue
        sev = v.get("severity")
        violations.append({
            "law":            str(v.get("law", "") or "unspecified").strip(),
            "observation":    str(v.get("observation", "") or "").strip(),
            "why_impossible": str(v.get("why_impossible", "") or "").strip(),
            "severity":       min(max(float(sev), 0.0), 1.0)
                              if isinstance(sev, (int, float)) else None,
        })
    parsed["violations"] = violations
    return parsed

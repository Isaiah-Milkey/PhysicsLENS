"""
Async client for ASU CreateAI (api-*.aiml.asu.edu) — used by Stage 3
specialists for VLM adjudication via the Gemini models it proxies
(e.g. "geminiflash2_5", "geminiflash2_5-lite").

Credentials come from .env at the PhysicsLENS root:
  CREATEAI_TOKEN, CREATEAI_BASE_URL
(scripts/createai_vision.py is the sync/manual-test counterpart.)
"""
import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass                                    # fall back to plain os.environ

DEFAULT_MODEL = "geminiflash2_5"


def frame_to_data_url(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def credentials() -> tuple[Optional[str], Optional[str]]:
    return os.environ.get("CREATEAI_TOKEN"), os.environ.get("CREATEAI_BASE_URL")


async def query_vision(query: str, frame_bgr: np.ndarray, *,
                       model: str = DEFAULT_MODEL,
                       system_prompt: Optional[str] = None,
                       timeout_s: float = 60.0) -> Dict[str, Any]:
    """One image + prompt → CreateAI /query response JSON.

    Sends both the flat "model" key (newer CreateAI examples) and the vision
    endpoint fields, so either payload style the deployment expects is present.
    Raises RuntimeError on missing credentials or HTTP failure.
    """
    import aiohttp

    token, base_url = credentials()
    if not token or not base_url:
        raise RuntimeError(
            "CreateAI credentials missing — set CREATEAI_TOKEN and "
            "CREATEAI_BASE_URL in the PhysicsLENS .env file."
        )

    payload: Dict[str, Any] = {
        "endpoint": "vision",
        "request_source": "override_params",
        "query": query,
        "input": query,
        "image_file": frame_to_data_url(frame_bgr),
        "model": model,
        "model_name": model,
    }
    if system_prompt:
        payload["model_params"] = {"system_prompt": system_prompt}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            base_url.rstrip("/") + "/query",
            json=payload,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"CreateAI HTTP {resp.status}: {str(data)[:300]}")
    return data


_SUBJECTS_PROMPT = (
    "The image shows two sampled frames (side by side) from one video. List "
    "the primary distinct physical objects — the moving/acting subjects whose "
    "physical behavior matters, then key interacting surfaces. Skip pure "
    "background. Use short visual noun phrases a segmentation model can "
    'ground (e.g. "basketball", "wooden crate"). At most {k}. Reply with ONLY '
    'strict JSON: {{"subjects": ["...", "..."]}}'
)


async def name_subjects(frames_bgr: list, *, max_subjects: int = 3,
                        model: str = DEFAULT_MODEL) -> list[str]:
    """Ask the VLM to name the primary subjects across sampled frames.

    `frames_bgr`: 1–2 frames (e.g. first + middle) tiled side by side so
    subjects that only appear mid-action are still named. Returns a list of
    short noun phrases (may be empty). Raises RuntimeError on missing
    credentials / HTTP failure — callers degrade to their fallback.
    """
    from tools.vlm import parse_vlm_json

    tiles = frames_bgr if isinstance(frames_bgr, list) else [frames_bgr]
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (max(2, int(t.shape[1] * h / t.shape[0])), h))
             for t in tiles]
    gap = np.full((h, 12, 3), 255, np.uint8)
    composite = tiles[0]
    for t in tiles[1:]:
        composite = np.concatenate([composite, gap, t], axis=1)

    data = await query_vision(_SUBJECTS_PROMPT.format(k=max_subjects),
                              composite, model=model)
    parsed = parse_vlm_json(response_text(data) or "")
    subjects = parsed.get("subjects") if isinstance(parsed, dict) else None
    if not isinstance(subjects, list):
        return []
    out = [str(s).strip() for s in subjects if str(s).strip()]
    return out[:max_subjects]


def response_text(data: Dict[str, Any]) -> str:
    """Best-effort extraction of the model's text from a CreateAI response."""
    for key in ("response", "answer", "output", "text", "result", "content"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = response_text(v)
            if inner:
                return inner
    return ""

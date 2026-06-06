"""VLM suspicion scoring via OpenRouter."""
import base64, json
from typing import Any, Dict, List

import cv2
import numpy as np

# Map of friendly key → OpenRouter model ID
OPENROUTER_MODELS: Dict[str, str] = {
    "gpt-4o":          "openai/gpt-4o",
    "gemini-2-flash":  "google/gemini-2.0-flash-001",
    "claude-3.5":      "anthropic/claude-3.5-sonnet",
    "llava-video":     "mistralai/mistral-nemo",   # placeholder; swap for video-capable model
    "videochatgpt":    "openai/gpt-4o",            # placeholder
}

SUSPICION_PROMPT = (
    "You are a physics expert reviewing a video frame.\n\n"
    "Does the interaction shown violate physical common sense?\n\n"
    "Reply with VALID JSON ONLY — no markdown fences, no extra text:\n"
    '{"suspicion_score": <float 0.0-1.0>, '
    '"suspected_failure": "<brief label or null>", '
    '"time_interval": "<rough description or null>", '
    '"explanation": "<one sentence>", '
    '"confidence": <float 0.0-1.0>}'
)


def frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


async def score_frame(
    frame: np.ndarray,
    model_key: str = "gpt-4o",
    api_key: str = "",
) -> Dict[str, Any]:
    """Send one frame to OpenRouter and parse the JSON response."""
    try:
        import aiohttp
    except ImportError:
        raise RuntimeError("aiohttp not installed — pip install aiohttp")

    model_id = OPENROUTER_MODELS.get(model_key, OPENROUTER_MODELS["gpt-4o"])
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": SUSPICION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(frame)}"},
                },
            ],
        }],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://physicslens.local",
            },
            json=payload,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as resp:
            data = await resp.json()

    raw = data["choices"][0]["message"]["content"]
    # Strip accidental markdown fences
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)

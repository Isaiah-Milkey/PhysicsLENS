"""
Stage 3 · SOTA Comparator — VLM-as-Judge (water physics)
--------------------------------------------------------
Represents VideoPhy / PhyGenEval: sample keyframes, ask a vision-language
model to rate fluid-physics plausibility against a rubric, average the verdict
into a single score. Self-contained OpenRouter call (no edits to tools/vlm.py);
reuses frame_to_b64 / OPENROUTER_MODELS read-only.
"""
import json
from typing import Any, AsyncGenerator, Dict, List

import numpy as np

from tools.video import load_frames, sample_frames
from tools.vlm import frame_to_b64, OPENROUTER_MODELS
from tools.fluid import severity_color

WATER_RUBRIC = (
    "You are a fluid-dynamics expert reviewing a frame from a video of water.\n\n"
    "Judge ONLY water/fluid physical plausibility: mass continuity (water must not "
    "appear or vanish), splash and spray plausibility, foam advection, surface waves, "
    "and reflections.\n\n"
    "Reply with VALID JSON ONLY — no markdown fences, no extra text:\n"
    '{"plausibility": <float 0.0-1.0, 1=fully physical>, '
    '"violations": [<short strings>], '
    '"explanation": "<one sentence>"}'
)


def parse_verdict(raw: str) -> Dict[str, Any]:
    """Tolerant parser: strip fences, default to nulls on failure."""
    try:
        import re
        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
        d = json.loads(s)
        return {
            "plausibility": float(d["plausibility"]) if d.get("plausibility") is not None else None,
            "violations": list(d.get("violations") or []),
            "explanation": str(d.get("explanation") or ""),
        }
    except Exception:
        return {"plausibility": None, "violations": [], "explanation": ""}


def severity_from_verdicts(verdicts: List[Dict[str, Any]]) -> int:
    vals = [v["plausibility"] for v in verdicts if v.get("plausibility") is not None]
    if not vals:
        return 0
    mean_plaus = float(np.mean(vals))
    return int(round((1.0 - mean_plaus) * 100))


async def score_keyframes(frames: List[np.ndarray], model_key: str, api_key: str) -> List[Dict[str, Any]]:
    import aiohttp
    model_id = OPENROUTER_MODELS.get(model_key, OPENROUTER_MODELS["gpt-4o"])
    verdicts: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for fr in frames:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": WATER_RUBRIC},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(fr)}"}},
                ]}],
            }
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://physicslens.local"},
                json=payload, timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"OpenRouter {resp.status}: {body[:200]}")
                data = await resp.json()
            verdicts.append(parse_verdict(data["choices"][0]["message"]["content"]))
    return verdicts


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    model_key = cfg.get("model", "gpt-4o")
    num_frames = int(cfg.get("num_frames", 5))
    api_key = cfg.get("api_key", "")

    if not api_key:
        yield {"type": "error", "text": "VLM judge requires an OpenRouter API key (set it in settings)."}
        return

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 1:
        yield {"type": "error", "text": "Empty video."}
        return
    keyframes = sample_frames(frames, num_frames)
    yield {"type": "log", "level": "info",
           "text": f"Scoring {len(keyframes)} keyframes with {model_key}…"}
    try:
        verdicts = await score_keyframes(keyframes, model_key, api_key)
    except Exception as exc:
        yield {"type": "error", "text": f"VLM request failed: {exc}"}
        return

    severity = severity_from_verdicts(verdicts)
    all_violations = sorted({v for vd in verdicts for v in vd["violations"]})
    yield {"type": "metric", "label": "Keyframes judged", "value": str(len(verdicts)), "sub": model_key}
    yield {"type": "metric", "label": "Distinct violations", "value": str(len(all_violations)),
           "sub": (", ".join(all_violations[:3]) or "none")}
    yield {"type": "severity", "label": "VLM-judged implausibility (coarse)",
           "value": severity, "color": severity_color(severity)}
    for vd in verdicts:
        if vd["explanation"]:
            yield {"type": "log", "level": "info", "text": f"VLM: {vd['explanation']}"}
    yield {"type": "log", "level": "warn" if severity > 15 else "success",
           "text": f"VLM-as-judge plausibility score: {100 - severity}/100."}
    yield {"type": "done"}

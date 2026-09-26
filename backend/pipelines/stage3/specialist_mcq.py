"""
Stage 3 · Specialist Evaluation — single forced-choice VLM question.
---------------------------------------------------------------------
Replaces the seven separate per-category specialists (collision, gravity,
momentum, friction, deformation, fluid, causality — see
backend/archive_files or git history for the retired implementations) with
one multiple-choice question over all seven families plus "permanence" and
"none". This is the design validated in the paper's automated-diagnosis
results (Section 5.4) and produced by backend/scripts/mcq_probe.py; this
module is that same question, ported to the pipeline contract so the live
tool's Stage 3 matches what was actually evaluated.

WHY ONE QUESTION INSTEAD OF SEVEN. Scored independently, each of the old
specialists got its own scale and nothing forced them to disagree — collision
would answer "yes" on nearly every clip (grippers do touch things) while
deformation answered "no" on nearly every clip, regardless of what actually
went wrong. A single forced-choice question makes the options COMPETE: the
letter probabilities are normalised by construction, so alleging every
failure is collision costs probability mass that has to come from somewhere
else. It is also one VLM call instead of seven.

POSITION BIAS IS CONTROLLED the same way as the evaluation script: option
order is shuffled per video with a seed derived from the video's content
hash, so a model's tendency to favour "whatever is listed first" shows up as
noise rather than a systematic win for one family.

Two backends, matching backend/scripts/mcq_probe.py's --api and local paths:
  • local  (tools.vlm_local.mcq_probs) — no API key, runs on the GPU.
  • openai (tools.llm_api.query_vision_token_probs) — needs OPENAI_API_KEY;
    same OpenAI-compatible Chat Completions logprobs read as mcq_probe.py
    --api. OpenRouter is not wired here: logprobs support is inconsistent
    across the models OpenRouter proxies, and it was never the path this
    design was evaluated against.
"""
import asyncio
import json
from typing import AsyncGenerator

import cv2
import numpy as np
import plotly.graph_objects as go

from tools.evidence import EVIDENCE, video_id
from tools.video import load_frames, sample_frames

# Keep in sync with backend/scripts/mcq_probe.py's OPTIONS — same wording, so
# any future re-evaluation of the live tool is directly comparable to the
# paper's numbers. (name, description-shown-to-the-model)
OPTIONS = [
    ("collision", "Two things overlap or pass through each other instead of "
                  "meeting at their surfaces, or the gripper grasps without "
                  "closing on the object"),
    ("deformation", "A rigid object changes its shape, length or thickness"),
    ("causality", "An object moves or changes on its own, with nothing visibly "
                  "touching it"),
    ("gravity", "Something hangs in the air, floats, or fails to fall when "
                "nothing is holding it"),
    ("momentum", "Something speeds up after being released or struck, or stops "
                 "dead for no reason"),
    ("permanence", "An object appears from nowhere or vanishes"),
    ("fluid", "Liquid appears, vanishes, or holds an impossible rigid shape"),
    ("friction", "Something slides when it should grip, or keeps sliding with "
                 "nothing pushing it"),
    ("none", "No physics problem — everything behaves as it should"),
]
NAMES = [n for n, _ in OPTIONS]
LETTERS = "ABCDEFGHI"  # one per option, in NAMES order before shuffling

PROMPT = ('Look at these {n} frames, sampled in order from a video{task_clause}.\n\n'
          'Which ONE of these best describes the main physics problem in this '
          'video?\n\n{opts}\n\n'
          'Answer with exactly one letter.')


def _build(video_key: str, nframes: int, task: str):
    """Prompt text + letter->family mapping, options shuffled per video."""
    rng = np.random.default_rng(abs(hash(video_key)) % (2 ** 31))
    order = list(rng.permutation(len(OPTIONS)))
    lines, mapping = [], {}
    for pos, idx in enumerate(order):
        name, desc = OPTIONS[idx]
        letter = LETTERS[pos]
        lines.append(f"{letter}) {desc}")
        mapping[letter] = name
    task_clause = f' of: "{task}"' if task else ""
    prompt = PROMPT.format(n=nframes, task_clause=task_clause, opts="\n".join(lines))
    return prompt, mapping


def _normalise(raw_probs: dict, mapping: dict) -> dict | None:
    """Raw {token: probability} -> {family: probability}, renormalised.
    Sums surface-form variants the same way mcq_probe.py's normalise() does."""
    agg: dict[str, float] = {}
    for k, v in raw_probs.items():
        key = str(k).strip().upper()
        if key in mapping:
            agg[mapping[key]] = agg.get(mapping[key], 0.0) + v
    tot = sum(agg.values())
    if tot <= 1e-6:
        return None
    return {n: agg.get(n, 0.0) / tot for n in NAMES}


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg        = json.loads(settings) if settings else {}
    model_key  = str(cfg.get("model") or "qwen2.5-vl-7b")
    api_key    = str(cfg.get("api_key", "")).strip()
    num_frames = max(4, min(16, int(cfg.get("num_frames", 8))))
    task       = str(cfg.get("task_description", "")).strip()

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames_bgr, _fps = await loop.run_in_executor(None, load_frames, video_path)
    if len(frames_bgr) < 2:
        yield {"type": "error", "text": f"Video too short ({len(frames_bgr)} frames)."}
        return
    sampled = sample_frames(frames_bgr, num_frames)
    n = len(sampled)

    prompt, mapping = _build(vid, n, task)

    from tools.vlm_local import LOCAL_VLMS
    is_local = model_key in LOCAL_VLMS
    yield {"type": "log", "level": "info",
           "text": f"Single forced-choice question over {n} frame(s), "
                   f"{len(OPTIONS)} options, via {model_key}"
                   f"{' (local)' if is_local else ''}…"}

    raw_probs: dict = {}
    try:
        if is_local:
            from tools.vlm_local import mcq_probs
            sampled_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]
            raw_probs = await loop.run_in_executor(
                None, mcq_probs, sampled_rgb, prompt, "".join(mapping), n, model_key)
        else:
            from tools.vlm_router import resolve, key_status
            provider, provider_model = resolve(model_key)
            if provider != "openai":
                yield {"type": "error",
                       "text": f"Provider '{provider}' isn't wired for the Stage 3 "
                                "MCQ (only local models or an openai:… model key are "
                                "supported — logprob scoring needs a Chat Completions "
                                "logprobs response)."}
                return
            have_key, key_desc = key_status(model_key, api_key)
            if not have_key:
                yield {"type": "log", "level": "warn",
                       "text": f"No API key ({key_desc}) — running in demo mode "
                                "(placeholder score)."}
                yield {"type": "metric", "label": "Top family", "value": "demo",
                       "sub": "Demo mode — no API key provided."}
                yield {"type": "severity", "label": "Specialist violation score",
                       "value": 0, "color": "#4CAF50"}
                yield {"type": "done"}
                return
            from tools.llm_api import query_vision_token_probs
            raw_probs = await query_vision_token_probs(
                prompt, sampled, model=provider_model,
                token=(api_key or None))
    except Exception as exc:                                        # noqa: BLE001
        yield {"type": "error", "text": f"VLM call failed: {str(exc)[:300]}"}
        return

    probs = _normalise(raw_probs, mapping)
    if not probs:
        yield {"type": "error",
               "text": "Model did not answer with a recognizable option letter — "
                       "no usable probability distribution."}
        return

    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top_family, top_p = ranked[0]
    p_violation = 1.0 - probs.get("none", 0.0)
    desc_by_name = dict(OPTIONS)

    yield {"type": "log", "level": "success" if top_family == "none" else "warn",
           "text": f"Top answer: {top_family} ({top_p:.0%})"
                   + (f" — {desc_by_name[top_family]}" if top_family != "none" else "")}

    for name, p in ranked[:5]:
        yield {"type": "metric", "label": name.capitalize(), "value": f"{p:.0%}",
               "sub": desc_by_name[name]}

    fig = go.Figure(go.Bar(
        x=[p for _, p in ranked][::-1], y=[n.capitalize() for n, _ in ranked][::-1],
        orientation="h", marker_color=[_sev_color(p * 100) for _, p in ranked][::-1],
        text=[f"{p:.0%}" for _, p in ranked][::-1], textposition="outside"))
    fig.update_xaxes(range=[0, 1.05], title_text="Probability",
                     showgrid=True, gridcolor="#ebebeb")
    fig.update_layout(
        title=dict(text="Physics-failure family (forced choice)", font=dict(size=15)),
        height=160 + 26 * len(ranked), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=110, r=60, t=60, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13))
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "One question, nine competing options (seven physics "
                      "families + object permanence + none) — probabilities "
                      "sum to 1, so naming one family costs probability mass "
                      "that must come from another."}

    yield {"type": "severity", "label": "Specialist violation score",
           "value": round(p_violation * 100, 1), "color": _sev_color(p_violation * 100)}

    finding = {
        "family_probs": {k: round(v, 4) for k, v in probs.items()},
        "top_family": top_family,
        "p_violation": round(p_violation, 4),
        "explanation": (f"Forced-choice VLM judged \"{top_family}\" the most likely "
                        f"physics failure ({top_p:.0%}): {desc_by_name[top_family]}."
                        if top_family != "none" else
                        "Forced-choice VLM found no likely physics failure "
                        f"({probs['none']:.0%} confidence)."),
    }
    yield {"type": "result", "status": "ok", "mcq": finding, "mapping": mapping}
    EVIDENCE.put(vid, "s3_specialist", finding)
    yield {"type": "done"}

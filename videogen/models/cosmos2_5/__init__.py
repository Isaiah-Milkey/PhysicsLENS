"""NVIDIA Cosmos-Predict2.5 family — importable API via diffusers.

    from models import cosmos2_5
    cosmos2_5.generate("a marble rolls off a wooden table")            # text2world
    cosmos2_5.generate("the marble bounces", image="start.png")        # image2world
    cosmos2_5.generate("...", video="clip.mp4")                        # video2world

(From generate.py / gen.sh the family is spelled "cosmos2.5" — dots are
normalized to underscores when resolving the package.)

ONE checkpoint serves all three conditionings (the 2.5 models are unified) —
no separate t2v/i2v downloads like Wan. The pipeline loads on first call and
stays resident.

Guardrail note: diffusers' Cosmos pipelines want the `cosmos_guardrail`
package (NVIDIA's content-safety filter: prompt screening + face blur). It is
not installed in the shared env, so a pass-through stub is used instead and a
warning is printed per load. This matches the existing cosmos-predict2.5
workflow in ~/world_models, which runs no guardrail either. To run WITH the
guardrail, install `cosmos-guardrail` into the env and delete the stub.
"""
import json
import time
from pathlib import Path

from .. import DEFAULT_SECONDS, frames_for, free_gb, is_cached, slug

MODELS = {
    "2b": {
        "repo": "nvidia/Cosmos-Predict2.5-2B", "params": "2B",
        "revision": "diffusers/base/post-trained",
        "size_gb": 35.0, "height": 704, "width": 1280,
        "num_frames": 93, "fps": 16,
        "tier": "efficient",
        "note": "unified text/image/video2world. Post-trained base.",
    },
    "14b": {
        "repo": "nvidia/Cosmos-Predict2.5-14B", "params": "14B",
        "revision": "diffusers/base/post-trained",
        "size_gb": 60.0, "height": 704, "width": 1280,
        "num_frames": 93, "fps": 16,
        "tier": "best",
        "note": "same conditioning as 2b, higher fidelity.",
    },
}
DEFAULT = "2b"

# NVIDIA's default negative prompt (from scripts/diffusers_inference.py).
NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no "
    "motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, pixelated images, poorly lit areas, underexposed and "
    "overexposed scenes, poor color balance, washed out colors, choppy "
    "sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


class _NoGuardrail:
    """Pass-through stand-in for CosmosSafetyChecker (see module docstring)."""

    def to(self, *a, **k):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, frames):
        return frames


# (repo, revision, dtype) -> loaded pipeline
_PIPES: dict = {}


def load(model: str | None = None, dtype: str = "bfloat16", offload: bool = False):
    """Load the unified pipeline for `model` and keep it resident for reuse.

    Returns (checkpoint_name, model_cfg, pipeline).
    """
    name = model or DEFAULT
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}. Options: {', '.join(MODELS)}")
    m = MODELS[name]
    key = (m["repo"], m["revision"], dtype)
    if key in _PIPES:
        return name, m, _PIPES[key]

    import torch
    from diffusers import Cosmos2_5_PredictBasePipeline

    if not is_cached(m["repo"]):
        if free_gb() < m["size_gb"] + 5:
            raise RuntimeError(f"not enough disk: {free_gb():.0f} GB free, "
                               f"need ~{m['size_gb']:.0f} GB + 5 GB headroom")
        print(f"[cosmos2.5] downloading ~{m['size_gb']:.0f} GB on first use "
              f"({free_gb():.0f} GB free)…")

    t0 = time.time()
    print("[cosmos2.5] WARNING: running WITHOUT NVIDIA's content guardrail "
          "(cosmos_guardrail not installed) — local benchmarking only.")
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        m["repo"], revision=m["revision"], torch_dtype=dt,
        safety_checker=_NoGuardrail())
    if offload:
        pipe.enable_model_cpu_offload()
        print("[cosmos2.5] CPU offload enabled (slower, lower VRAM)")
    else:
        pipe.to("cuda")

    print(f"[cosmos2.5] {name} ({m['params']} params, {m['repo']}) "
          f"ready in {time.time() - t0:.0f}s")
    _PIPES[key] = pipe
    return name, m, pipe


def generate(prompt: str, image=None, images=None, video=None,
             mode: str | None = None, model: str | None = None, out=None,
             seed: int = 0, steps: int = 36, seconds: float | None = None,
             negative: str = NEGATIVE, dtype: str = "bfloat16",
             offload: bool = False, **overrides) -> Path:
    """One call, one clip. Returns the mp4 Path; a .json sidecar sits beside it.

    Conditioning is inferred from what you pass: nothing -> "text"
    (text2world), an image -> "text+image" (image2world), a video path ->
    "text+video" (video2world). Text is ALWAYS used — the visual input pins
    the starting state, the prompt drives what happens. `mode` is accepted
    for interface parity but never needed. Clip length is `seconds` (default
    5); pass num_frames to override exactly. `overrides` may also set height,
    width, fps; None values are ignored.
    """
    import torch
    from diffusers.utils import export_to_video, load_image, load_video

    if images:
        if len(images) > 1:
            raise ValueError("cosmos takes at most one conditioning image")
        image = images[0]
    if image is not None and video is not None:
        raise ValueError("pass either image= or video=, not both")
    if mode not in (None, "text", "text+image", "text+video"):
        raise ValueError(f"cosmos2.5 has no mode {mode!r} — conditioning is "
                         "inferred: none -> text, image= -> text+image, "
                         "video= -> text+video")

    name, m, pipe = load(model, dtype, offload)

    cfg = {k: m.get(k) for k in ("height", "width", "num_frames", "fps")}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if not overrides.get("num_frames"):
        cfg["num_frames"] = frames_for(
            DEFAULT_SECONDS if seconds is None else seconds, cfg["fps"])

    pil_image = load_image(str(image)) if image is not None else None
    vid_frames = load_video(str(video)) if video is not None else None
    cond = ("text+video" if video is not None
            else "text+image" if image is not None else "text")

    t0 = time.time()
    frames = pipe(
        image=pil_image, video=vid_frames, prompt=prompt,
        negative_prompt=negative,
        height=cfg["height"], width=cfg["width"],
        num_frames=cfg["num_frames"], num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
    ).frames[0]
    gen_s = time.time() - t0

    out_path = Path(out) if out else Path("outputs") / f"{slug(prompt)}_s{seed}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=cfg["fps"])
    out_path.with_suffix(".json").write_text(json.dumps(
        {"family": "cosmos2.5", "conditioning": cond, "checkpoint": name,
         "params": m["params"], "model": m["repo"], "revision": m["revision"],
         "seconds": round(cfg["num_frames"] / cfg["fps"], 2),
         "prompt": prompt, "negative_prompt": negative, "seed": seed,
         "steps": steps, "guardrail": False, "gen_seconds": round(gen_s, 1),
         **{k: cfg[k] for k in ("height", "width", "num_frames", "fps")}},
        indent=1))
    print(f"[cosmos2.5 {name} {m['params']}] -> {out_path}  ({gen_s:.0f}s)")
    return out_path

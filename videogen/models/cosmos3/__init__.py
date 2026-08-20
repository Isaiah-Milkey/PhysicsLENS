"""NVIDIA Cosmos 3 family — omnimodal world models via diffusers.

    from models import cosmos3
    cosmos3.generate("a marble rolls off a wooden table")            # text
    cosmos3.generate("the marble bounces", image="start.png")        # text+image
    cosmos3.generate("...", video="clip.mp4")                        # text+video

Cosmos 3 checkpoints are omnimodal: ONE model covers text/image/video
conditioning (the pipeline infers the mode from the inputs), can also emit
audio (not wired here), and ships without a mandatory guardrail component.
`nano` (15.8B) is the default; the 64B `super-i2v*` checkpoints are
image-conditioned only, ~120 GB each, and need CPU offload or 2-GPU sharding
on 95 GB cards. (Cosmos3-Edge, 3.9B, was tried and dropped: its newer
nemotron-dense backbone needs diffusers>=0.40, and even then its output on
our physics prompts was unusable noise.)

Caveat: only `super-i2v-4step` declares our exact pipeline class
(Cosmos3OmniPipeline @ diffusers 0.39). `nano`/`super-i2v` were published
against a neighbouring version — they are loaded with the same class and may
need a diffusers upgrade if components mismatch; the smoke test decides.
"""
import json
import time
from pathlib import Path

from .. import DEFAULT_SECONDS, frames_for, free_gb, is_cached, slug

MODELS = {
    "nano": {
        "repo": "nvidia/Cosmos3-Nano", "params": "15.8B",
        "size_gb": 32.6, "fps": 24, "steps": 35, "guidance_scale": 6.0,
        "tier": "efficient",
        "note": "omnimodal: text / text+image / text+video from one checkpoint.",
    },
    "super-i2v": {
        "repo": "nvidia/Cosmos3-Super-Image2Video", "params": "64.6B",
        "size_gb": 121.7, "fps": 24, "steps": 35, "guidance_scale": 6.0,
        "needs_image": True, "tier": "best",
        "note": "image-conditioned only, highest fidelity.",
    },
    "super-i2v-4step": {
        "repo": "nvidia/Cosmos3-Super-Image2Video-4Step", "params": "64B",
        # DMD2-distilled: 4 denoising steps, no classifier-free guidance.
        "size_gb": 120.6, "fps": 24, "steps": 4, "guidance_scale": 1.0,
        "needs_image": True, "tier": "fast",
        "note": "DMD2 distillation of super-i2v — near-same quality in 4 steps.",
    },
}
DEFAULT = "nano"

# (repo, dtype) -> loaded pipeline
_PIPES: dict = {}


def load(model: str | None = None, dtype: str = "bfloat16", offload: bool = False):
    """Load the omni pipeline for `model` and keep it resident for reuse.

    Returns (checkpoint_name, model_cfg, pipeline).
    """
    name = model or DEFAULT
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}. Options: {', '.join(MODELS)}")
    m = MODELS[name]
    key = (m["repo"], dtype)
    if key in _PIPES:
        return name, m, _PIPES[key]

    import torch
    from diffusers import Cosmos3OmniPipeline

    if not is_cached(m["repo"]):
        if free_gb() < m["size_gb"] + 5:
            raise RuntimeError(f"not enough disk: {free_gb():.0f} GB free, "
                               f"need ~{m['size_gb']:.0f} GB + 5 GB headroom")
        print(f"[cosmos3] downloading ~{m['size_gb']:.0f} GB on first use "
              f"({free_gb():.0f} GB free)…")

    t0 = time.time()
    print("[cosmos3] note: NVIDIA content guardrail disabled "
          "(enable_safety_checker=False) — local benchmarking only.")
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = Cosmos3OmniPipeline.from_pretrained(
        m["repo"], torch_dtype=dt, enable_safety_checker=False)
    if offload:
        pipe.enable_model_cpu_offload()
        print("[cosmos3] CPU offload enabled (slower, lower VRAM)")
    else:
        pipe.to("cuda")

    print(f"[cosmos3] {name} ({m['params']}, {m['repo']}) "
          f"ready in {time.time() - t0:.0f}s")
    _PIPES[key] = pipe
    return name, m, pipe


def generate(prompt: str, image=None, images=None, video=None,
             mode: str | None = None, model: str | None = None, out=None,
             seed: int = 0, steps: int | None = None,
             seconds: float | None = None, negative: str | None = None,
             dtype: str = "bfloat16", offload: bool = False,
             **overrides) -> Path:
    """One call, one clip. Returns the mp4 Path; a .json sidecar sits beside it.

    Conditioning is inferred: nothing -> "text", an image -> "text+image", a
    video path -> "text+video". Text is always used. Clip length is `seconds`
    (default 5); pass num_frames to override exactly. `steps` defaults to the
    checkpoint's native count (35, or 4 for the distilled model). `overrides`
    may also set height, width, fps, guidance_scale; None values are ignored.
    """
    import torch
    from diffusers.utils import export_to_video, load_image, load_video

    if images:
        if len(images) > 1:
            raise ValueError("cosmos3 takes at most one conditioning image")
        image = images[0]
    if image is not None and video is not None:
        raise ValueError("pass either image= or video=, not both")
    if mode not in (None, "text", "text+image", "text+video"):
        raise ValueError(f"cosmos3 has no mode {mode!r} — conditioning is "
                         "inferred: none -> text, image= -> text+image, "
                         "video= -> text+video")

    # Validate BEFORE load(): an argument error must never trigger a
    # 100+ GB download.
    name = model or DEFAULT
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}. Options: {', '.join(MODELS)}")
    if MODELS[name].get("needs_image") and image is None:
        raise ValueError(f"{name} is image-conditioned only — pass image= "
                         "(or use the 'nano' omnimodel for text-only)")

    name, m, pipe = load(name, dtype, offload)

    cfg = {"fps": m["fps"], "guidance_scale": m["guidance_scale"],
           "height": None, "width": None, "num_frames": None}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if not cfg["num_frames"]:
        cfg["num_frames"] = frames_for(
            DEFAULT_SECONDS if seconds is None else seconds, cfg["fps"])
    n_steps = steps or m["steps"]

    pil_image = load_image(str(image)) if image is not None else None
    vid_frames = load_video(str(video)) if video is not None else None
    cond = ("text+video" if video is not None
            else "text+image" if image is not None else "text")

    kw = dict(prompt=prompt, negative_prompt=negative, image=pil_image,
              video=vid_frames, num_frames=cfg["num_frames"], fps=cfg["fps"],
              guidance_scale=cfg["guidance_scale"],
              num_inference_steps=n_steps,
              generator=torch.Generator().manual_seed(seed))
    # height/width default to the checkpoint's native size inside the pipeline.
    if cfg["height"]:
        kw["height"] = cfg["height"]
    if cfg["width"]:
        kw["width"] = cfg["width"]

    t0 = time.time()
    # Cosmos3's output is .video (one sample, list of PIL frames) — not the
    # .frames batch list the other diffusers video pipelines use.
    frames = pipe(**kw).video
    gen_s = time.time() - t0

    out_path = Path(out) if out else Path("outputs") / f"{slug(prompt)}_s{seed}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=cfg["fps"])
    out_path.with_suffix(".json").write_text(json.dumps(
        {"family": "cosmos3", "conditioning": cond, "checkpoint": name,
         "params": m["params"], "model": m["repo"],
         "seconds": round(cfg["num_frames"] / cfg["fps"], 2),
         "prompt": prompt, "negative_prompt": negative, "seed": seed,
         "steps": n_steps, "gen_seconds": round(gen_s, 1),
         "num_frames": cfg["num_frames"], "fps": cfg["fps"],
         "guidance_scale": cfg["guidance_scale"],
         **({"height": cfg["height"], "width": cfg["width"]}
            if cfg["height"] else {})}, indent=1))
    print(f"[cosmos3 {name} {m['params']}] -> {out_path}  ({gen_s:.0f}s)")
    return out_path

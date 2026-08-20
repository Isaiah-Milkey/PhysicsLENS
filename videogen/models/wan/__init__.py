"""Wan family — importable API, pipeline kept resident between calls.

    from models import wan
    wan.generate("a marble rolls off a wooden table")              # t2v
    wan.generate("the marble bounces", image="start.png")          # i2v
    wan.generate("...", images=["a.png", "b.png"], mode="flf")     # first -> last
    wan.generate("...", images=["cat.png"], mode="ref")            # reference subjects

The first call loads the checkpoint (minutes cold, seconds from page cache);
every later call with the same mode reuses it. Each mode's pipeline class is
loaded independently — see the note in load() for why weight-sharing via
from_pipe() is deliberately NOT used.
"""
import json
import time
from pathlib import Path

from .. import (DEFAULT_SECONDS, HF_ROOT, frames_for, free_gb,  # noqa: F401
                is_cached, slug)

# ── Model registry ────────────────────────────────────────────────────────────
# `size_gb` is the full repo download: transformer(s) + the ~10.6 GB UMT5-XXL
# text encoder + VAE. Every Wan repo ships its own copy of that text encoder, so
# a second model costs ~10 GB more than parameter count suggests.
#
# `moe`: Wan2.2 A14B models are mixture-of-experts — two transformers, a
# high-noise and a low-noise expert, switched at `boundary_ratio` (read from the
# checkpoint config). They take two guidance scales, one per expert.
MODELS = {
    "ti2v-5b": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "modes": ["t2v", "i2v"], "size_gb": 31.9, "moe": False,
        "height": 704, "width": 1280, "num_frames": 121, "fps": 24,
        "guidance_scale": 5.0, "flow_shift": 5.0,
        "tier": "efficient",
        "note": "one checkpoint serves BOTH t2v and i2v. Best size/quality trade.",
    },
    "t2v-a14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "modes": ["t2v"], "size_gb": 117.5, "moe": True,
        "height": 720, "width": 1280, "num_frames": 81, "fps": 16,
        "guidance_scale": 4.0, "guidance_scale_2": 3.0, "flow_shift": 5.0,
        "tier": "best",
        "note": "Wan2.2 MoE. Highest-quality text-to-video.",
    },
    "i2v-a14b": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "modes": ["i2v"], "size_gb": 117.5, "moe": True,
        "height": 720, "width": 1280, "num_frames": 81, "fps": 16,
        "guidance_scale": 4.0, "guidance_scale_2": 3.0, "flow_shift": 5.0,
        "tier": "best",
        "note": "Wan2.2 MoE. Highest-quality image-to-video.",
    },
    "i2v-14b-720p": {
        "repo": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "modes": ["i2v"], "size_gb": 83.9, "moe": False,
        "height": 720, "width": 1280, "num_frames": 81, "fps": 16,
        "guidance_scale": 5.0, "flow_shift": 5.0,
        "tier": "good",
        "note": "Wan2.1 dense 14B. Cheaper than the A14B MoE, still strong.",
    },
    "flf2v-14b": {
        "repo": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        "modes": ["flf"], "size_gb": 83.9, "moe": False,
        "height": 720, "width": 1280, "num_frames": 81, "fps": 16,
        "guidance_scale": 5.5, "flow_shift": 5.0,
        "tier": "best",
        "note": "the only checkpoint trained for first-frame->last-frame interpolation.",
    },
    "vace-1.3b": {
        "repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "modes": ["ref"], "size_gb": 17.7, "moe": False,
        "height": 480, "width": 832, "num_frames": 81, "fps": 16,
        "guidance_scale": 5.0, "flow_shift": 3.0,
        "tier": "efficient",
        "note": "cheapest true multi-reference conditioning.",
    },
    "vace-14b": {
        "repo": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "modes": ["ref"], "size_gb": 70.0, "moe": False,
        "height": 720, "width": 1280, "num_frames": 81, "fps": 16,
        "guidance_scale": 5.0, "flow_shift": 5.0,
        "tier": "best",
        "note": "same conditioning as vace-1.3b, much better fidelity.",
    },
}

# mode -> (pipeline class, images required (-1 = one or more), default model, description)
MODES = {
    "t2v": ("WanPipeline",             0, "ti2v-5b",   "text only"),
    "i2v": ("WanImageToVideoPipeline", 1, "ti2v-5b",   "text + 1 image (first frame)"),
    "flf": ("WanImageToVideoPipeline", 2, "flf2v-14b", "text + first & last frame"),
    "ref": ("WanVACEPipeline",        -1, "vace-1.3b", "text + N reference images"),
}

# Wan's own recommended negative prompt (from the model cards). Suppresses the
# static / oversaturated / low-quality modes the base models drift toward.
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "artwork, painting, picture, still, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, motionless image, cluttered background, three legs, many "
    "people in the background, walking backwards"
)


def resolve(mode: str, model: str | None):
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}. Choices: {', '.join(MODES)}")
    pipe_cls, n_img, default, _ = MODES[mode]
    name = model or default
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}. Options: {', '.join(MODELS)}")
    m = MODELS[name]
    if mode not in m["modes"]:
        ok = [k for k, v in MODELS.items() if mode in v["modes"]]
        raise ValueError(f"model {name!r} does not support mode {mode!r}. "
                         f"Models for {mode}: {', '.join(ok)}")
    return name, m, pipe_cls, n_img


# (repo, pipe_cls, dtype, offload, flow_shift) -> loaded pipeline
_PIPES: dict = {}


def load(mode: str = "t2v", model: str | None = None, dtype: str = "bfloat16",
         offload: bool = False, flow_shift: float | None = None):
    """Load the pipeline serving `mode` and keep it resident for reuse.

    Returns (checkpoint_name, model_cfg, pipeline).
    """
    name, m, pipe_cls, _ = resolve(mode, model)
    shift = flow_shift if flow_shift is not None else m.get("flow_shift")
    key = (m["repo"], pipe_cls, dtype, offload, shift)
    if key in _PIPES:
        return name, m, _PIPES[key]

    import torch
    import diffusers
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

    PipeCls = getattr(diffusers, pipe_cls)
    t0 = time.time()

    # Each (checkpoint, pipeline class) is loaded independently, even when the
    # weights overlap (ti2v-5b serves t2v AND i2v). Deriving the second
    # pipeline via from_pipe() to share weights looked free, but measurably
    # poisoned inference: every call after the derive ran ~7x slower
    # (162s -> 1126s per clip). Two resident 5B copies cost ~20 GB VRAM and
    # load in seconds from page cache — cheap next to that.
    if not is_cached(m["repo"]):
        if free_gb() < m["size_gb"] + 5:
            raise RuntimeError(
                f"not enough disk: {free_gb():.0f} GB free, need "
                f"~{m['size_gb']:.0f} GB + 5 GB headroom in {HF_ROOT}")
        print(f"[wan] downloading ~{m['size_gb']:.0f} GB on first use "
              f"({free_gb():.0f} GB free)…")
    # The Wan VAE stays fp32: in bf16 its decoder shows colour banding and
    # can emit NaNs on longer clips. It is small next to the transformer.
    vae = AutoencoderKLWan.from_pretrained(m["repo"], subfolder="vae",
                                           torch_dtype=torch.float32)
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    pipe = PipeCls.from_pretrained(m["repo"], vae=vae, torch_dtype=dt)
    if offload:
        # Streams submodules to GPU on demand; needed for 14B on smaller
        # cards. Do NOT also call .to("cuda") — they conflict.
        pipe.enable_model_cpu_offload()
        print("[wan] CPU offload enabled (slower, lower VRAM)")
    else:
        pipe.to("cuda")

    # flow_shift warps the flow-matching schedule. Wan's cards use ~5.0 at 720p
    # and ~3.0 at 480p; it visibly affects motion, so it is set per model rather
    # than left at the scheduler default.
    if shift is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=shift)

    print(f"[wan] {name} ({pipe_cls}) ready in {time.time() - t0:.0f}s")
    _PIPES[key] = pipe
    return name, m, pipe


# The cross-family conditioning vocabulary (see models/README.md) and its
# mapping onto Wan's per-mode pipeline classes.
_COND = {"t2v": "text", "i2v": "text+image", "flf": "text+first+last",
         "ref": "text+references"}
_MODE_ALIASES = {v: k for k, v in _COND.items()}


def generate(prompt: str, image=None, images=None, mode: str | None = None,
             model: str | None = None, out=None, seed: int = 0, steps: int = 40,
             seconds: float | None = None, negative: str = NEGATIVE,
             dtype: str = "bfloat16", offload: bool = False,
             flow_shift: float | None = None, **overrides) -> Path:
    """One call, one clip. Returns the mp4 Path; a .json sidecar sits beside it.

    Mode is inferred: no image -> t2v, one image -> i2v. flf/ref must be named
    explicitly (two images are ambiguous); the common conditioning terms
    ("text", "text+image", "text+first+last", "text+references") are accepted
    as mode names too. Clip length is `seconds` (default 5); pass num_frames
    to override exactly. `overrides` may also set height, width, fps,
    guidance_scale, guidance_scale_2; None values are ignored.
    """
    import torch
    from diffusers.utils import export_to_video, load_image
    from PIL import Image

    imgs = [image] if image is not None else list(images or [])
    mode = _MODE_ALIASES.get(mode, mode)
    if mode is None:
        mode = "t2v" if not imgs else "i2v"
    name, m, _cls, n_img = resolve(mode, model)
    if n_img >= 0 and len(imgs) != n_img:
        raise ValueError(f"mode {mode!r} needs exactly {n_img} image(s), got {len(imgs)}")
    if n_img == -1 and not imgs:
        raise ValueError("ref needs at least one reference image")

    _, _, pipe = load(mode, model, dtype, offload, flow_shift)

    cfg = {k: m.get(k) for k in ("height", "width", "num_frames", "fps",
                                 "guidance_scale", "guidance_scale_2")}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if not overrides.get("num_frames"):
        cfg["num_frames"] = frames_for(
            DEFAULT_SECONDS if seconds is None else seconds, cfg["fps"])

    pil = [load_image(str(p)).convert("RGB")
           .resize((cfg["width"], cfg["height"]), Image.LANCZOS) for p in imgs]

    kw = dict(prompt=prompt, negative_prompt=negative,
              height=cfg["height"], width=cfg["width"],
              num_frames=cfg["num_frames"],
              guidance_scale=cfg["guidance_scale"],
              num_inference_steps=steps,
              generator=torch.Generator(device="cuda").manual_seed(seed))
    if m["moe"]:
        # Only MoE checkpoints have a second expert to steer.
        kw["guidance_scale_2"] = cfg.get("guidance_scale_2") or cfg["guidance_scale"]
    if mode == "i2v":
        kw["image"] = pil[0]
    elif mode == "flf":
        kw["image"], kw["last_image"] = pil[0], pil[1]
    elif mode == "ref":
        kw["reference_images"] = pil

    t0 = time.time()
    frames = pipe(**kw).frames[0]
    gen_s = time.time() - t0

    out_path = Path(out) if out else Path("outputs") / f"{slug(prompt)}_s{seed}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=cfg["fps"])
    out_path.with_suffix(".json").write_text(json.dumps(
        {"family": "wan", "conditioning": _COND[mode], "mode": mode,
         "checkpoint": name, "model": m["repo"],
         "seconds": round(cfg["num_frames"] / cfg["fps"], 2),
         "prompt": prompt, "negative_prompt": negative, "seed": seed,
         "steps": steps, "n_input_images": len(pil),
         "gen_seconds": round(gen_s, 1),
         **{k: cfg[k] for k in ("height", "width", "num_frames", "fps",
                                "guidance_scale")}}, indent=1))
    print(f"[wan] -> {out_path}  ({gen_s:.0f}s)")
    return out_path

#!/usr/bin/env python
"""
Wan video generation — one CLI, four conditioning modes, seven models.

  MODES (what you condition on)
    t2v   text only
    i2v   text + 1 image          (image becomes the first frame)
    flf   text + first & last     (model interpolates between them)
    ref   text + N reference imgs (subjects to keep consistent)

  MODELS are picked with --model, or left to a per-mode default.
  Run `--list` to see every mode/model pairing with its download size.

Nothing downloads until you run a real generation. `--dry-run` validates the
whole invocation (args, images, disk, cache status) and touches no weights.

For the drop-in-a-folder workflow, use ../run.sh instead of calling this
directly. This script is what run.sh invokes.

Examples
--------
  python wan_generate.py --list
  python wan_generate.py --mode t2v --prompt "a marble rolls off a table" --out a.mp4
  python wan_generate.py --mode t2v --model t2v-a14b --prompt "..." --out a.mp4
  python wan_generate.py --mode i2v --image start.png --prompt "..." --out b.mp4
  python wan_generate.py --mode flf --image start.png --image end.png --prompt "..." --out c.mp4
  python wan_generate.py --mode ref --image cat.png --image sofa.png --prompt "..." --out d.mp4

  # batch: model loads ONCE for the whole folder
  python wan_generate.py --mode t2v --prompt-file prompts.txt --out-dir clips/
  python wan_generate.py --mode i2v --prompt-file prompts.txt --image-dir imgs/ --out-dir clips/
"""
import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

HF_ROOT = "/data/ssagar6/hf_cache"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── Model registry ────────────────────────────────────────────────────────────
# `size_gb` is the full repo download: transformer(s) + the ~10.6 GB UMT5-XXL
# text encoder + VAE. Every Wan repo ships its own copy of that text encoder, so
# a second model costs ~10 GB more than parameter count suggests.
#
# `moe`: Wan2.2 A14B models are mixture-of-experts — two transformers, a
# high-noise and a low-noise expert, switched at `boundary_ratio` (read from the
# checkpoint config). They take two guidance scales, one per expert.
#
# Defaults below come from each model card and are NOT yet verified here.
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


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def is_cached(repo: str) -> bool:
    """True if the repo already has a materialised snapshot in HF_HOME."""
    d = Path(HF_ROOT) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def free_gb(path=HF_ROOT) -> float:
    return shutil.disk_usage(path).free / 2**30


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-") or "clip")


def print_table():
    print(f"\nHF cache: {HF_ROOT}   free disk: {free_gb():.0f} GB\n")
    print(f"  {'mode':6}{'model':16}{'size':>8}  {'cached':8}{'tier':11}{'output':22}note")
    print("  " + "-" * 108)
    for mode, (_, _, default, _d) in MODES.items():
        for name, m in MODELS.items():
            if mode not in m["modes"]:
                continue
            star = " *" if name == default else "  "
            res = f"{m['width']}x{m['height']} {m['num_frames']}f@{m['fps']}"
            print(f"  {mode:6}{name + star:16}{m['size_gb']:7.1f}G  "
                  f"{'YES' if is_cached(m['repo']) else 'no':8}{m['tier']:11}{res:22}{m['note']}")
    print("  " + "-" * 108)
    print("  * = default for that mode.  Pick another with --model.")
    print("  Models marked cached=no download on first use.\n")


def resolve(mode: str, model: str | None):
    pipe_cls, n_img, default, _ = MODES[mode]
    name = model or default
    if name not in MODELS:
        die(f"unknown --model {name!r}. Options: {', '.join(MODELS)}  (see --list)")
    m = MODELS[name]
    if mode not in m["modes"]:
        ok = [k for k, v in MODELS.items() if mode in v["modes"]]
        die(f"model {name!r} does not support --mode {mode}. "
            f"Models for {mode}: {', '.join(ok)}")
    return name, m, pipe_cls, n_img


def build_jobs(mode, prompts, image_paths, out_dir, out_single):
    """Expand (prompts x images) into a flat job list BEFORE the model loads, so
    a whole folder costs one model load and any mismatch fails immediately.

    Pairing rules:
      t2v  N prompts                      -> N videos
      i2v  N images, N prompts            -> pair by sorted order
           N images, 1 prompt             -> that prompt for every image
      flf  exactly 2 images, N prompts    -> N videos (first->last)
      ref  all images as references       -> N videos, one per prompt
    """
    jobs = []
    if mode == "t2v":
        for i, p in enumerate(prompts):
            jobs.append((p, [], f"{i:03d}_{slug(p)}"))
    elif mode == "i2v":
        if len(prompts) == 1:
            prompts = prompts * len(image_paths)
        if len(prompts) != len(image_paths):
            die(f"i2v needs one prompt per image, or exactly one prompt for all.\n"
                f"       got {len(prompts)} prompt(s) and {len(image_paths)} image(s).")
        for i, (p, img) in enumerate(zip(prompts, image_paths)):
            jobs.append((p, [img], f"{i:03d}_{Path(img).stem}_{slug(p, 24)}"))
    elif mode == "flf":
        if len(image_paths) != 2:
            die(f"flf needs exactly 2 images (first, last); got {len(image_paths)}")
        for i, p in enumerate(prompts):
            jobs.append((p, list(image_paths), f"{i:03d}_flf_{slug(p)}"))
    elif mode == "ref":
        if not image_paths:
            die("ref needs at least one reference image")
        for i, p in enumerate(prompts):
            jobs.append((p, list(image_paths), f"{i:03d}_ref_{slug(p)}"))

    if out_single and len(jobs) == 1:
        return [(jobs[0][0], jobs[0][1], Path(out_single))]
    return [(p, im, Path(out_dir) / f"{nm}.mp4") for p, im, nm in jobs]


def load_pipeline(m, pipe_cls, dtype_str, offload, flow_shift):
    import torch
    import diffusers
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    PipeCls = getattr(diffusers, pipe_cls)

    print(f"[wan] model  {m['repo']}")
    if not is_cached(m["repo"]):
        print(f"[wan] downloading ~{m['size_gb']:.0f} GB on first use "
              f"({free_gb():.0f} GB free)…")
    t0 = time.time()

    # The Wan VAE stays fp32: in bf16 its decoder shows colour banding and can
    # emit NaNs on longer clips. It is small next to the transformer.
    vae = AutoencoderKLWan.from_pretrained(m["repo"], subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = PipeCls.from_pretrained(m["repo"], vae=vae, torch_dtype=dtype)

    # flow_shift warps the flow-matching schedule. Wan's cards use ~5.0 at 720p
    # and ~3.0 at 480p; it visibly affects motion, so it is set per model rather
    # than left at the scheduler default.
    shift = flow_shift if flow_shift is not None else m.get("flow_shift")
    if shift is not None:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=shift)
        print(f"[wan] flow_shift {shift}")

    if offload:
        # Streams submodules to GPU on demand; needed for 14B on smaller cards.
        # Do NOT also call .to("cuda") — they conflict.
        pipe.enable_model_cpu_offload()
        print("[wan] CPU offload enabled (slower, lower VRAM)")
    else:
        pipe.to("cuda")

    print(f"[wan] ready in {time.time() - t0:.0f}s")
    return pipe


def load_images(paths, height, width):
    from diffusers.utils import load_image
    from PIL import Image
    out = []
    for p in paths:
        if not Path(p).exists():
            die(f"image not found: {p}")
        out.append(load_image(str(p)).convert("RGB").resize((width, height), Image.LANCZOS))
    return out


def generate(pipe, m, mode, prompt, images, seed, steps, out_path, negative, cfg):
    import torch
    from diffusers.utils import export_to_video

    kw = dict(prompt=prompt, negative_prompt=negative,
              height=cfg["height"], width=cfg["width"],
              num_frames=cfg["num_frames"],
              guidance_scale=cfg["guidance_scale"],
              num_inference_steps=steps,
              generator=torch.Generator(device="cuda").manual_seed(seed))

    # Only MoE checkpoints have a second expert to steer.
    if m["moe"]:
        kw["guidance_scale_2"] = cfg.get("guidance_scale_2") or cfg["guidance_scale"]

    if mode == "i2v":
        kw["image"] = images[0]
    elif mode == "flf":
        kw["image"], kw["last_image"] = images[0], images[1]
    elif mode == "ref":
        kw["reference_images"] = images

    t0 = time.time()
    frames = pipe(**kw).frames[0]
    dt = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=cfg["fps"])
    out_path.with_suffix(".json").write_text(json.dumps(
        {"mode": mode, "model": m["repo"], "prompt": prompt,
         "negative_prompt": negative, "seed": seed, "steps": steps,
         "n_input_images": len(images), "gen_seconds": round(dt, 1),
         **{k: cfg[k] for k in ("height", "width", "num_frames", "fps",
                                "guidance_scale")}}, indent=1))
    print(f"[wan] -> {out_path}  ({dt:.0f}s)")


def main():
    ap = argparse.ArgumentParser(
        description="Wan video generation: text / +image / +first&last / +references.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--list", action="store_true", help="show all mode/model pairings and exit")
    ap.add_argument("--mode", choices=list(MODES))
    ap.add_argument("--model", help="override the default model for the mode")
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file", help="one prompt per line")
    ap.add_argument("--image", action="append", default=[], help="repeat as needed")
    ap.add_argument("--image-dir", help="use every image in this directory (sorted)")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--offload", action="store_true", help="lower VRAM, slower")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--flow-shift", type=float)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate everything and print the plan; download nothing")
    a = ap.parse_args()

    if a.list:
        print_table()
        return
    if not a.mode:
        die("--mode is required (or use --list). Choices: " + ", ".join(MODES))

    prompts = []
    if a.prompt_file:
        pf = Path(a.prompt_file)
        if not pf.exists():
            die(f"prompt file not found: {pf}")
        prompts = [l.strip() for l in pf.read_text().splitlines()
                   if l.strip() and not l.lstrip().startswith("#")]
    elif a.prompt:
        prompts = [a.prompt]
    if not prompts:
        die("no prompts — give --prompt, or a --prompt-file with at least one "
            "non-comment line")

    images = list(a.image)
    if a.image_dir:
        d = Path(a.image_dir)
        if not d.is_dir():
            die(f"--image-dir not a directory: {d}")
        images += [str(p) for p in sorted(d.iterdir())
                   if p.suffix.lower() in IMAGE_EXTS]
    if a.mode == "t2v" and images:
        die("--mode t2v takes no images; use i2v, flf or ref")

    name, m, pipe_cls, n_img = resolve(a.mode, a.model)
    cfg = {k: m.get(k) for k in ("height", "width", "num_frames", "fps",
                                 "guidance_scale", "guidance_scale_2")}
    for src, key in ((a.height, "height"), (a.width, "width"),
                     (a.frames, "num_frames"), (a.guidance, "guidance_scale")):
        if src:
            cfg[key] = src

    single = a.out if (len(prompts) == 1 and not a.prompt_file) else None
    jobs = build_jobs(a.mode, prompts, images, a.out_dir, single)

    cached = is_cached(m["repo"])
    need = 0.0 if cached else m["size_gb"]
    print(f"\n[plan] mode={a.mode} ({MODES[a.mode][3]})")
    print(f"[plan] model={name}  {m['repo']}")
    print(f"[plan] pipeline={pipe_cls}  moe={m['moe']}  tier={m['tier']}")
    print(f"[plan] {len(prompts)} prompt(s), {len(images)} image(s) -> {len(jobs)} video(s)")
    print(f"[plan] output={cfg['width']}x{cfg['height']} "
          f"{cfg['num_frames']}f @ {cfg['fps']}fps, {a.steps} steps")
    print(f"[plan] cached={cached}  download={need:.1f} GB  free={free_gb():.0f} GB")

    if need and free_gb() < need + 5:
        die(f"not enough disk: {free_gb():.0f} GB free, need ~{need:.0f} GB + 5 GB "
            f"headroom.\n       Free space or choose a smaller model (--list).")
    for p in images:
        if not Path(p).exists():
            die(f"image not found: {p}")

    if a.dry_run:
        for _p, _im, out in jobs[:8]:
            print(f"[plan]   -> {out}")
        if len(jobs) > 8:
            print(f"[plan]   … and {len(jobs) - 8} more")
        print("[plan] dry run — nothing downloaded, nothing generated.\n")
        return

    pipe = load_pipeline(m, pipe_cls, a.dtype, a.offload, a.flow_shift)

    ok = 0
    for i, (prompt, img_paths, out_path) in enumerate(jobs):
        print(f"\n[wan] ({i + 1}/{len(jobs)}) {out_path.name}")
        try:
            imgs = load_images(img_paths, cfg["height"], cfg["width"])
            generate(pipe, m, a.mode, prompt, imgs, a.seed + i, a.steps,
                     out_path, a.negative, cfg)
            ok += 1
        except Exception as exc:                                   # noqa: BLE001
            # One bad prompt/image should not lose the rest of a long batch.
            print(f"[wan] FAILED {out_path.name}: {exc}", file=sys.stderr)
        gc.collect()

    print(f"\n[wan] done: {ok}/{len(jobs)} generated -> {Path(a.out_dir).resolve()}\n")


if __name__ == "__main__":
    main()

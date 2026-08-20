#!/usr/bin/env python
"""
Wan batch CLI — thin wrapper over models/wan (which owns the registry,
loader and generation; the pipeline loads once per process and is reused
for every job).

  MODES (what you condition on)
    t2v   text only
    i2v   text + 1 image          (image becomes the first frame)
    flf   text + first & last     (model interpolates between them)
    ref   text + N reference imgs (subjects to keep consistent)

  MODELS are picked with --model, or left to a per-mode default.
  Run `--list` to see every mode/model pairing with its download size.

Nothing downloads until you run a real generation. `--dry-run` validates the
whole invocation (args, images, disk, cache status) and touches no weights.

For the drop-in-a-folder workflow, use ./run.sh instead of calling this
directly. For single programmatic/one-off calls, use generate.py / gen.sh.

Examples
--------
  python wan_generate.py --list
  python wan_generate.py --mode t2v --prompt "a marble rolls off a table" --out a.mp4
  python wan_generate.py --mode i2v --image start.png --prompt "..." --out b.mp4
  python wan_generate.py --mode flf --image start.png --image end.png --prompt "..." --out c.mp4
  python wan_generate.py --mode ref --image cat.png --image sofa.png --prompt "..." --out d.mp4

  # batch: model loads ONCE for the whole folder
  python wan_generate.py --mode t2v --prompt-file prompts.txt --out-dir clips/
  python wan_generate.py --mode i2v --prompt-file prompts.txt --image-dir imgs/ --out-dir clips/
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import wan
from models.wan import (HF_ROOT, MODELS, MODES, NEGATIVE, free_gb, is_cached,
                        slug)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


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


def build_jobs(mode, prompts, image_paths, out_dir, out_single):
    """Expand (prompts x images) into a flat job list so any mismatch fails
    before the model loads.

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

    try:
        name, m, pipe_cls, _ = wan.resolve(a.mode, a.model)
    except ValueError as e:
        die(str(e))
    cfg = {k: m.get(k) for k in ("height", "width", "num_frames", "fps")}
    for src, key in ((a.height, "height"), (a.width, "width"),
                     (a.frames, "num_frames")):
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

    ok = 0
    for i, (prompt, img_paths, out_path) in enumerate(jobs):
        print(f"\n[wan] ({i + 1}/{len(jobs)}) {out_path.name}")
        try:
            wan.generate(prompt, images=img_paths, mode=a.mode, model=a.model,
                         out=out_path, seed=a.seed + i, steps=a.steps,
                         negative=a.negative, dtype=a.dtype, offload=a.offload,
                         flow_shift=a.flow_shift, height=a.height,
                         width=a.width, num_frames=a.frames,
                         guidance_scale=a.guidance)
            ok += 1
        except Exception as exc:                                   # noqa: BLE001
            # One bad prompt/image should not lose the rest of a long batch.
            print(f"[wan] FAILED {out_path.name}: {exc}", file=sys.stderr)
        gc.collect()

    print(f"\n[wan] done: {ok}/{len(jobs)} generated -> {Path(a.out_dir).resolve()}\n")


if __name__ == "__main__":
    main()

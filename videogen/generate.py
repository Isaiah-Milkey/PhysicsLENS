#!/usr/bin/env python
"""One entry point for every model family under models/.

From Python (model stays loaded between calls):

    from generate import generate
    generate("a marble rolls off a wooden table")               # default family
    generate("the marble bounces", image="start.png")           # text + image
    generate("...", model="wan:t2v-a14b", steps=50)             # pick checkpoint

From the shell (one-off; pays the model load each invocation — use run.sh
for batches):

    ./gen.sh "a marble rolls off a wooden table"
    ./gen.sh "the marble bounces" --image start.png
    ./gen.sh "..." --model wan:t2v-a14b

`model` is "family" or "family:checkpoint"; family = a folder under models/.
"""
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_FAMILY = "wan"


def generate(prompt: str, image=None, model: str | None = None, **kw) -> Path:
    family, _, ckpt = (model or DEFAULT_FAMILY).partition(":")
    try:
        # Version dots become underscores: family "cosmos2.5" -> models/cosmos2_5
        mod = import_module(f"models.{family.replace('.', '_')}")
    except ModuleNotFoundError:
        have = sorted(p.name.replace("_", ".") if p.name[-1].isdigit() else p.name
                      for p in (Path(__file__).parent / "models").iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
        raise ValueError(f"unknown model family {family!r}. Available: {', '.join(have)}")
    if ckpt:
        kw["model"] = ckpt
    return mod.generate(prompt, image=image, **kw)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate one clip with any registered model family.")
    ap.add_argument("prompt")
    ap.add_argument("--model", help='family[:checkpoint], e.g. "wan" or "wan:t2v-a14b"')
    ap.add_argument("--image", action="append", default=[], help="repeat as needed")
    ap.add_argument("--mode", help="only needed for flf/ref; t2v/i2v are inferred")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seconds", type=float, help="clip length (default 5)")
    ap.add_argument("--frames", type=int, help="exact frame count (overrides --seconds)")
    ap.add_argument("--offload", action="store_true", help="lower VRAM, slower")
    a = ap.parse_args()

    generate(a.prompt, model=a.model, mode=a.mode, out=a.out, seed=a.seed,
             seconds=a.seconds, num_frames=a.frames, offload=a.offload,
             **({"steps": a.steps} if a.steps else {}),
             **({"images": a.image} if len(a.image) > 1 else
                {"image": a.image[0]} if a.image else {}))


if __name__ == "__main__":
    main()

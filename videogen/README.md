# videogen — Wan video generation

Drop prompts in, drop images in, run one script, get videos out. Used to generate
clips for PhysicsLENS benchmarking. Metrics and evaluation are decided
separately — this is only the input → model → output path.

> **Status: implemented, not yet executed.** No model has been downloaded and no
> video generated. Argument handling, mode resolution, batch pairing and disk
> guards are all verified with `--dry-run`; the generation defaults come from the
> Wan model cards and have not been validated on this machine. Expect to tune
> them on the first real run.

---

## Use it

```bash
cd videogen

# 1. write prompts, one per line
$EDITOR inputs/prompts.txt

# 2. optionally drop images in inputs/images/

# 3. go
./run.sh
```

Videos land in `outputs/` as `000_<name>.mp4`, each with a matching `.json`
recording the prompt, seed, model and settings — so any clip can be regenerated
exactly.

**Always dry-run first.** It validates everything and downloads nothing:

```bash
./run.sh --dry-run
```

---

## Modes

Mode is auto-detected — no images means `t2v`, images present means `i2v`.
Override by naming it:

```bash
./run.sh t2v     # text only
./run.sh i2v     # text + 1 image each
./run.sh flf     # text + first & last frame (exactly 2 images)
./run.sh ref     # text + all images as reference subjects
```

| mode | inputs | produces |
|---|---|---|
| `t2v` | N prompts | N videos |
| `i2v` | N images + N prompts | N videos, **paired in sorted filename order** |
| `i2v` | N images + 1 prompt | N videos, that prompt applied to each |
| `flf` | exactly 2 images + N prompts | N videos interpolating first → last |
| `ref` | N images + M prompts | M videos, all images as reference subjects |

> **Pairing gotcha for `i2v`:** images are matched to prompts by *sorted
> filename*, not by content. Name them to match prompt order —
> `001_marble.png`, `002_water.png`, `003_crate.png` — or you will silently get
> the wrong prompt on the wrong image.

`flf` gives the model a start and end frame and asks it to fill the middle.
`ref` treats the images as subjects to keep visually consistent while the prompt
drives the action — they are not frames.

---

## Options

Anything after the mode is passed through to `wan_generate.py`:

```bash
./run.sh t2v --model t2v-a14b --steps 50
./run.sh i2v --offload                  # if you hit CUDA OOM
./run.sh --dry-run
```

Env overrides: `MODEL=` `STEPS=` `SEED=` `OUTDIR=` `PY=`

```bash
MODEL=t2v-a14b STEPS=50 ./run.sh t2v
```

Common flags: `--frames` `--height` `--width` `--guidance` `--flow-shift`
`--dtype float16` `--negative`.

---

## Models

`./run.sh --dry-run` prints which model it will use;
`python wan_generate.py --list` shows all of them and what is already downloaded.

| `--model` | HuggingFace repo | size | modes | tier |
|---|---|---|---|---|
| `ti2v-5b` * | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 31.9 G | `t2v`, `i2v` | efficient |
| `t2v-a14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 117.5 G | `t2v` | best |
| `i2v-a14b` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 117.5 G | `i2v` | best |
| `i2v-14b-720p` | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 83.9 G | `i2v` | good |
| `flf2v-14b` * | `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` | 83.9 G | `flf` | best |
| `vace-1.3b` * | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | 17.7 G | `ref` | efficient |
| `vace-14b` | `Wan-AI/Wan2.1-VACE-14B-diffusers` | 70.0 G | `ref` | best |

`*` = default for its mode. Models download on first use into
`$HF_HOME` (`/data/ssagar6/hf_cache`) and are reused after that.

**Sizes are larger than parameter count suggests** — every Wan repo ships its own
copy of the ~10.6 GB UMT5-XXL text encoder, so each additional model costs about
10 GB more than you would guess.

**Start with `ti2v-5b`.** It is the only checkpoint covering both text-only and
image-conditioned generation, at 720p, from a single 31.9 GB download. The A14B
models are better but cost 117.5 GB each — worth it only for a benchmark set you
intend to publish.

---

## Prompting

Wan responds to concrete physical description:

- **Name the motion and its cause** — "the cue ball strikes the red ball, which
  rolls into the corner pocket" beats "billiards".
- **Give materials** — "a *glass* marble on *polished wood*" changes bounce and
  reflection.
- **One clear action per clip.** These are ~5 seconds; two chained events usually
  produces neither cleanly.
- **Pin the camera** — "static camera, side view" stops the model inventing a pan.
  This matters if the clips feed PhysicsLENS, where camera motion is itself one
  of the detectors.

A long negative prompt (Wan's own recommended one) is applied by default to
suppress static/oversaturated output. Override with `--negative`.

---

## Requirements

- `physicslens` conda env — torch 2.12.0+cu130, transformers 5.10.2,
  diffusers 0.39.0, ftfy, imageio-ffmpeg
- one H100; the 5B fits easily, the 14B models fit but use `--offload` if the
  card is shared
- `HF_HOME=/data/ssagar6/hf_cache`

`run.sh` uses `/data/ssagar6/miniforge3/envs/physicslens/bin/python` explicitly —
the default `python` on PATH is the miniforge base env and lacks these packages.
Point elsewhere with `PY=`.

---

## Troubleshooting

**`ERROR: not enough disk`** — deliberate; the script refuses a download that
would leave under 5 GB headroom. Check `df -h /data`, free space, or pick a
smaller model.

**CUDA out of memory** — add `--offload`, lower `--frames`, or check `nvidia-smi`
for other users on the card.

**First run is slow** — it is downloading tens of GB. Later runs load from cache
in about a minute. The model loads **once** per `run.sh` invocation regardless of
how many prompts, so batch rather than looping the script.

**One clip failed, the rest are fine** — intended. A failing prompt or image is
logged and skipped so a long batch is not lost; the final line reports `N/M
generated`.

**Output looks static or washed out** — raise `--steps`, be more specific about
motion, keep the default negative prompt, try `--flow-shift` between 3 and 7.

"""
Does a second CRITIC pass rescue the low-accuracy specialists?

The specialists that fail (gravity, collision, deformation, momentum) fail in a
specific way: they say "yes, somewhat" to nearly everything. The probe asks
"does this defect happen?" and the model answers from overall impression, which
is exactly what an appearance detector produces. A critic pass attacks that by
forcing a second, harder judgement about the SAME clip.

Four critic styles, because they fail differently and we do not know in advance
which the model can do:

  verify      "a checker flagged this clip for X — is the flag correct?"
              Cheapest. Mostly tests whether restating the claim changes
              anything, and is the baseline the others must beat.
  refute      "find the strongest reason the claim is WRONG." Scored INVERTED.
              Asking for the counter-case is a different computation from asking
              for the case, so its errors need not correlate with the probe's —
              which is the only way a second pass adds information rather than
              repeating the first.
  evidence    "name the object and the two frames; if you cannot point to a
              specific object and frame pair, answer 1." Directly targets the
              observed failure mode: the model flags a defect it cannot localise.
  alternative "could this be explained by camera motion, occlusion, or the
              robot's own action?" Scored INVERTED. Physics violations and
              benign explanations are confusable in exactly these four
              specialists, so naming the benign options is the discriminating
              question.

Scoring is the same token-probability estimator as the probes, so critic and
probe scores are directly combinable. Three combinations are reported: critic
alone, rank-mean with the probe, and probe alone as the baseline.

Usage:
  python backend/scripts/critic_pass.py --data data/robotbench \
      --probes domain_probes_robot__f4.json --style evidence --cats gravity,collision
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from domain_probes import BATTERIES, frames_ordered, caption_for  # noqa: E402
from videophy_eval import _token_probs, client  # noqa: E402
from gepa_optimize import load  # noqa: E402

MODEL = "gemma4-31b-it"

# plain-language name of the defect, substituted into every critic prompt
DEFECT = {
    "gravity": "an object hanging in the air instead of falling",
    "collision": "the gripper or hand overlapping an object instead of meeting "
                 "its surface",
    "deformation": "a rigid object changing shape or size",
    "momentum": "an object moving faster after being released or struck",
    "causality": "an object moving with nothing touching it",
    "permanence": "an object appearing or disappearing",
    "fluid": "liquid behaving impossibly",
    "friction": "an object sliding when it should grip",
}

HEAD = ('Look at these {n} frames, sampled in order from a video of: '
        '"{caption}".\n')

SCALE = ("\nAnswer on this scale:\n"
         "1 = not at all, 2 = barely, 3 = somewhat, 4 = clearly, "
         "5 = blatantly.\nAnswer with exactly one digit, 1 to 5.")

STYLES = {
    "verify": ("An automated physics checker flagged this clip for {defect}.\n"
               "Look at the frames yourself. How clearly is that flag CORRECT?",
               False),
    "refute": ("Someone claims this clip shows {defect}.\n"
               "Find the strongest reason that claim is WRONG. How strong is "
               "that reason?", True),
    "evidence": ("Someone claims this clip shows {defect}.\n"
                 "To agree you must be able to point to ONE specific object and "
                 "TWO specific frames where it is visible. If you cannot point "
                 "to a specific object and a specific pair of frames, answer 1.\n"
                 "How clearly can you point to specific evidence?", False),
    "alternative": ("This clip may appear to show {defect}.\n"
                    "Could what you see instead be explained by ordinary camera "
                    "movement, something passing in front, or the robot's own "
                    "deliberate action? How strongly does an ordinary "
                    "explanation fit?", True),
}


def score_one(c, data, clip, cats, style, model, nframes, capmode):
    imgs = frames_ordered(data, clip, nframes, "temporal")
    cap = caption_for(clip, capmode)
    head = HEAD.format(n=len(imgs), caption=cap) if cap else \
        f"Look at these {len(imgs)} frames, sampled in order from a video.\n"
    tmpl, invert = STYLES[style]
    out = {}
    for cat in cats:
        q = tmpl.format(defect=DEFECT[cat])
        try:
            p = _token_probs(c, model, imgs, head + q + SCALE)
            mass = {int(k): v for k, v in p.items()
                    if len(k) == 1 and k in "12345"}
            if sum(mass.values()) <= 1e-4:
                continue
            v = float(min(max(1.0 - p.get("1", 0.0), 0.0), 1.0))
            # inverted styles score "how well does the BENIGN story fit", so a
            # high value means LESS likely a violation
            out[cat] = 1.0 - v if invert else v
        except Exception as e:  # noqa: BLE001
            print(f"    {cat} fail {clip['clip_id'][:24]}: {str(e)[:40]}",
                  file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--style", default="evidence", choices=list(STYLES))
    ap.add_argument("--cats", default="gravity,collision,deformation,momentum")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--capmode", default="full")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--all-clips", action="store_true")
    a = ap.parse_args()

    data, clips = load(a.data)
    if not a.all_clips:
        clips = [c for c in clips if len(c.get("violated_rules") or "") > 4]
    cats = [c.strip() for c in a.cats.split(",") if c.strip() in DEFECT]
    print(f"critic[{a.style}] {a.model}: {len(cats)} specialists x "
          f"{len(clips)} clips = {len(cats)*len(clips)} calls", flush=True)

    c = client()
    res = [None] * len(clips)
    t0, done = time.time(), [0]

    def work(i):
        res[i] = score_one(c, data, clips[i], cats, a.style, a.model,
                           a.frames, a.capmode)
        done[0] += 1
        if done[0] % 50 == 0:
            el = time.time() - t0
            print(f"    {done[0]}/{len(clips)} ({el:.0f}s, eta "
                  f"{el/done[0]*(len(clips)-done[0])/60:.0f}m)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, range(len(clips))))

    out = {cl["clip_id"]: r for cl, r in zip(clips, res) if r}
    cells = sum(len(v) for v in out.values())
    want = len(clips) * len(cats)
    tag = f"{a.style}__{a.model}" + (f"__f{a.frames}" if a.frames != 8 else "")
    p = data / f"critic_{tag}.json"
    p.write_text(json.dumps({"style": a.style, "model": a.model, "cats": cats,
                             "frames": a.frames, "n": len(out),
                             "critic_scores": out}, indent=1))
    print(f"\n  {len(out)}/{len(clips)} clips, {cells}/{want} cells "
          f"({time.time()-t0:.0f}s)\n  -> {p}")
    if want and cells < 0.5 * want:
        sys.exit(f"ERROR: only {100*cells/want:.0f}% scored — not a usable run")


if __name__ == "__main__":
    main()

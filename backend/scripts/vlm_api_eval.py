"""
Score hosted VLMs on the Rapidata human-agreement benchmark via an
OpenAI-compatible gateway (ASU Research Computing).

Same protocol as scripts/vlm_multimodel_eval.py / vlm_rapidata_eval.py — same
8-frame multi-frame sampling, same two prompts, same Spearman / median-split-AUC
metrics, and the SAME stratified 60-clip sample (read from the staged
data/rapidata60/labels.json). So numbers land directly comparable to the local
model baselines, rather than being a fresh incomparable eval.

Two scores per clip:
  json_score    the model's self-written suspicion_score, 0-1
  logprob_score P(Yes)/(P(Yes)+P(No)) from the next-token distribution — the
                calibrated one. Needs a gateway that returns logprobs; CreateAI
                strips them, this gateway does not.

Usage:
  python backend/scripts/vlm_api_eval.py                       # default models
  python backend/scripts/vlm_api_eval.py gemma4-31b-it
  python backend/scripts/vlm_api_eval.py gemma4-31b-it qwen3-vl-32b-instruct --workers 8
  python backend/scripts/vlm_api_eval.py --limit 10            # quick smoke
"""
import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman, median_split_auc  # noqa: E402

LABELS = ROOT / "data" / "rapidata60" / "labels.json"
VIDEO_DIR = ROOT / "data" / "rapidata60" / "ai"
OUT = Path(__file__).parent / "vlm_api_results.json"   # --order writes a suffixed file

DEFAULT_MODELS = ["gemma4-31b-it", "qwen3-vl-32b-instruct"]

# Verbatim from vlm_multimodel_eval.py so the comparison is apples-to-apples.
PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. Judge whether the MOTION and INTERACTIONS across frames obey "
    "real-world physics (gravity, momentum, object permanence, rigid-body and fluid "
    "behavior). AI-generated videos often show objects that morph, appear/disappear, "
    "move without forces, or deform implausibly.\n"
    "Reply with VALID JSON ONLY - no markdown fences:\n"
    '{{"suspicion_score": <float 0.0-1.0>, "suspected_failure": "<brief label or null>", '
    '"explanation": "<one or two sentences>", "confidence": <float 0.0-1.0>}}'
)
YESNO_PROMPT = (
    "You are a physics expert. These {n} frames are sampled in temporal order from "
    "one video. AI-generated videos often show objects that morph, appear/disappear, "
    "float, pass through each other, or move without forces. Judging the MOTION and "
    "INTERACTIONS across frames: does this video violate real-world physics?\n"
    "Answer with exactly one word: Yes or No."
)


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    base = os.environ.get("ASU_OPENAI_BASE_URL")
    key = os.environ.get("ASU_OPENAI_API_KEY")
    if not (base and key):
        sys.exit("ERROR: set ASU_OPENAI_BASE_URL and ASU_OPENAI_API_KEY in .env")
    return OpenAI(base_url=base, api_key=key, timeout=180, max_retries=3)


# ── frames ────────────────────────────────────────────────────────────────────

def frames_b64(path: Path, n: int = 8, max_side: int = 512,
               order: str = "temporal", seed: int = 0) -> list[str]:
    """n evenly-spaced frames as JPEG data URLs. max_side=512 matches the local
    eval's `to_pil`, so the models see the same pixels.

    `order` is the control knob for the temporal-sensitivity ablation:
      temporal  frames in true time order (the normal protocol)
      shuffled  same frames, randomised order — a model that scores this the
                same as `temporal` is not reading motion at all, only static
                appearance. A real physics judge should find shuffled frames
                MORE implausible, since the arrow of time is visibly broken.
      reversed  same frames, played backwards.
    Identical pixels in every condition, so any score difference is purely the
    model's response to temporal structure.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"unreadable: {path}")
    out = []
    for i in np.linspace(0, total - 1, min(n, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            out.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())
    cap.release()
    if not out:
        raise RuntimeError(f"no frames decoded: {path}")
    if order == "shuffled":
        # Seeded per clip so the shuffle is reproducible across models/runs —
        # otherwise two models get different permutations and aren't comparable.
        rnd = random.Random(f"{path.name}:{seed}")
        rnd.shuffle(out)
    elif order == "reversed":
        out.reverse()
    return out


def _msg(imgs, text):
    return [{"role": "user", "content":
             [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
             + [{"type": "text", "text": text}]}]


# ── scoring ───────────────────────────────────────────────────────────────────

def parse_json(raw: str):
    import re
    if not raw:
        return None
    for cand in (raw, *re.findall(r"\{.*\}", raw, re.S)):
        try:
            return json.loads(cand.strip().strip("`").removeprefix("json").strip())
        except Exception:  # noqa: BLE001
            continue
    return None


def ask_json(c, model, imgs):
    r = c.chat.completions.create(model=model, max_tokens=300, temperature=0,
                                  messages=_msg(imgs, PROMPT.format(n=len(imgs))))
    d = parse_json(r.choices[0].message.content or "")
    if not isinstance(d, dict):
        return None
    try:
        return float(np.clip(float(d["suspicion_score"]), 0.0, 1.0))
    except Exception:  # noqa: BLE001
        return None


def ask_logprob(c, model, imgs):
    """P(Yes)/(P(Yes)+P(No)) from the first sampled token's distribution.

    Sums over casing/leading-space variants because tokenizers split " Yes",
    "Yes" and "yes" into different ids — the same normalisation the local eval
    does, so the two are comparable.
    """
    r = c.chat.completions.create(model=model, max_tokens=1, temperature=0,
                                  logprobs=True, top_logprobs=20,
                                  messages=_msg(imgs, YESNO_PROMPT.format(n=len(imgs))))
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return None
    p_yes = p_no = 0.0
    for t in lp.content[0].top_logprobs:
        tok = t.token.strip().lower()
        p = math.exp(t.logprob)
        if tok == "yes":
            p_yes += p
        elif tok == "no":
            p_no += p
    return p_yes / (p_yes + p_no) if (p_yes + p_no) > 1e-4 else None


def score_clip(c, model, path, order='temporal'):
    imgs = frames_b64(path, order=order)
    js = lp = None
    try:
        js = ask_json(c, model, imgs)
    except Exception as e:  # noqa: BLE001
        print(f"    json  fail {path.name}: {str(e)[:70]}", file=sys.stderr)
    try:
        lp = ask_logprob(c, model, imgs)
    except Exception as e:  # noqa: BLE001
        print(f"    lprob fail {path.name}: {str(e)[:70]}", file=sys.stderr)
    return js, lp


def eval_model(c, model, clips, workers=6, order='temporal'):
    print(f"\n=== {model} [{order}] — {len(clips)} clips, {workers} workers", flush=True)
    t0 = time.time()
    rows = [None] * len(clips)

    def work(i):
        key, human, path = clips[i]
        js, lp = score_clip(c, model, path, order)
        rows[i] = {"clip": key, "human": human, "json_score": js, "logprob_score": lp}
        done = sum(1 for r in rows if r is not None)
        if done % 10 == 0:
            print(f"    {done}/{len(clips)}  ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, range(len(clips))))

    human = [r["human"] for r in rows]
    summary = {"model": model, "order": order, "n": len(rows),
               "seconds": round(time.time() - t0, 1)}
    for field, tag in (("json_score", ""), ("logprob_score", "logprob_")):
        pairs = [(r[field], r["human"]) for r in rows if r[field] is not None]
        summary[f"{tag}n_parsed"] = len(pairs)
        if len(pairs) >= 8 and len({p for p, _ in pairs}) > 1:
            xs = [p for p, _ in pairs]
            ys = [h for _, h in pairs]
            summary[f"{tag}spearman"] = round(spearman(xs, ys), 3)
            summary[f"{tag}median_split_auc"] = round(
                median_split_auc([r[field] for r in rows], human), 3)
            summary[f"{tag}distinct_values"] = len({p for p, _ in pairs})
        else:
            summary[f"{tag}spearman"] = None
            summary[f"{tag}median_split_auc"] = None
    print(f"  => json rho={summary.get('spearman')} AUC={summary.get('median_split_auc')}"
          f" ({summary['n_parsed']}/{summary['n']} parsed)  |  "
          f"logprob rho={summary.get('logprob_spearman')} "
          f"AUC={summary.get('logprob_median_split_auc')} "
          f"({summary['logprob_n_parsed']}/{summary['n']})", flush=True)
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, help="first N clips only (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--order", default="temporal",
                    choices=["temporal", "shuffled", "reversed"],
                    help="frame order ablation (same pixels, different sequence)")
    a = ap.parse_args()
    models = a.models or DEFAULT_MODELS

    if not LABELS.exists():
        sys.exit(f"ERROR: {LABELS} missing — run rapidata_prepare.py first")
    meta = json.loads(LABELS.read_text())
    clips = []
    for key, v in sorted(meta["clips"].items()):
        p = VIDEO_DIR / v["file"]
        if p.exists():
            clips.append((key, float(v["human_score"]), p))
    if a.limit:
        clips = clips[:a.limit]
    print(f"{len(clips)} clips (same stratified sample as the local baselines)")

    c = client()
    out = {"dataset": meta.get("dataset"), "n_clips": len(clips),
           "protocol": "8 frames, max_side 512, identical prompts to "
                       "vlm_multimodel_eval.py", "models": {}}
    for m in models:
        try:
            out["models"][f"{m}|{a.order}"] = eval_model(c, m, clips, a.workers, a.order)
        except Exception as e:  # noqa: BLE001
            print(f"  {m} FAILED: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        outp = OUT if a.order == "temporal" else OUT.with_name(f"vlm_api_results_{a.order}.json")
        outp.write_text(json.dumps(out, indent=1))

    print(f"\n-> {outp}")
    print("\nBaselines on this exact 60-clip sample (local models):")
    print("  Qwen2.5-VL-7B      rho=0.297  AUC=0.644   logprob rho=0.343  AUC=0.700")
    print("  InternVL3-8B       rho=0.157  AUC=0.533   logprob rho=0.513  AUC=0.816")


if __name__ == "__main__":
    main()

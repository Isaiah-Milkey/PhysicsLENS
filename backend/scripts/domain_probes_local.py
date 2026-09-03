"""
Run domain-knowledge batteries on a LOCAL VLM.

Why this exists: fluid reached 0.944 by fusing the SAME specific question across
three models (gemma4, qwen3-vl-32b, qwen2.5-vl-7b), while fusing seven models on
the vague holistic question gained nothing — a specific question makes each
model's residual noise independent, a vague one leaves shared bias. Reproducing
that for the other categories needs a third model, and the gateway does not
offer one: gemma3-27b is text-only (400s on any image) and llama4-scout caps at
~5 images per request, which would break the fixed 8-frame protocol. qwen2.5-vl-7b
runs locally on an idle GPU, costs nothing, and was already one of the three in
the fluid ensemble.

Scoring is identical to the API path — the same degree-scale prompt and the same
`1 - P("1")` estimator — so the scores are poolable with the hosted models rather
than being a separate incomparable measurement.

Usage:
  python backend/scripts/domain_probes_local.py --cats winners --model qwen2.5-vl-7b
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from videophy_eval_local import MODELS, load_model          # noqa: E402
from probes_local import frames_pil                         # noqa: E402
from domain_probes import BATTERIES, _P, _A                 # noqa: E402
from gepa_optimize import load                              # noqa: E402


def digit_score(torch, model, proc, imgs, question):
    """1 - P("1") over the first generated token — the same estimator the API
    path uses. Not the expected value over digits: when the model is confident
    the answer is "1" the other digits fall out of the truncated tail, the EV
    pins to exactly 1.0, and every clip ties at 0.0 with no gradation."""
    content = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": question})
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    tok = getattr(proc, "tokenizer", proc)
    p1, seen = 0.0, set()
    for v in ("1", " 1"):
        enc = tok.encode(v, add_special_tokens=False)
        if enc and enc[0] not in seen:
            seen.add(enc[0])
            p1 += float(probs[enc[0]].item())
    # confirm the model is actually answering with digits at all
    dig = 0.0
    for d in "12345":
        for v in (d, " " + d):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                dig += float(probs[e[0]].item())
    if dig < 1e-4:
        return None
    return float(min(max(1.0 - p1, 0.0), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--cats", default="winners")
    ap.add_argument("--model", default="qwen2.5-vl-7b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    data, clips = load(a.data)
    clips = [c for c in clips if len(c.get("violated_rules") or "") > 4]
    if a.limit:
        clips = clips[:a.limit]
    cats = [x.strip() for x in a.cats.split(",") if x.strip() in BATTERIES]
    probes = [(nm, q) for cat in cats for nm, q in BATTERIES[cat]]
    torch, model, proc = load_model(a.model, a.device)
    print(f"domain [{a.model}]: {len(probes)} probes x {len(clips)} clips",
          flush=True)

    t0 = time.time()
    out = {}
    for i, cl in enumerate(clips):
        imgs = frames_pil(data, cl, 8)
        head = _P.format(n=len(imgs), caption=cl.get("caption", "")[:200])
        r = {}
        for nm, q in probes:
            try:
                v = digit_score(torch, model, proc, imgs, head + q + _A)
                if v is not None:
                    r[nm] = v
            except Exception as e:  # noqa: BLE001
                print(f"    {nm} fail {cl['clip_id'][:26]}: {str(e)[:45]}",
                      file=sys.stderr)
        if r:
            out[cl["clip_id"]] = r
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{len(clips)} ({el:.0f}s, eta "
                  f"{el/(i+1)*(len(clips)-i-1)/60:.0f}m)", flush=True)

    outp = data / f"domain_probes_{'_'.join(cats)}__{a.model}.json"
    outp.write_text(json.dumps({"model": a.model, "local": True, "cats": cats,
                                "n": len(out), "probe_scores": out}, indent=1))
    vals = [x for r in out.values() for x in r.values()]
    print(f"\n  {len(out)}/{len(clips)} clips ({time.time()-t0:.0f}s)")
    if vals:
        print(f"  exact-zero {100*np.mean([v == 0 for v in vals]):.0f}%  "
              f"distinct {len(set(vals))}/{len(vals)}")
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()

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
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from videophy_eval_local import MODELS, load_model          # noqa: E402
from probes_local import frames_pil                         # noqa: E402
from domain_probes import percentile_table, build_injection  # noqa: E402
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


def caption_for(clip, mode="full"):
    """Caption variants, cut from the UNTRUNCATED caption.

    Every run before this one effectively used `scene`: staging capped the
    caption at 400 chars and the prompt cut it again at 200, so the "Action:"
    line — the only part saying what was supposed to HAPPEN — reached the model
    on almost no clip. The specialists were told what the scene looked like and
    never what it was meant to do.

    scene  = appearance only. A specialist that scores well on this alone is
             probably reading render quality, not physics.
    action = the intended event, from the generation prompt.
    task   = the robot's goal, one line, from the source dataset.
    """
    full = (clip.get("caption_full") or clip.get("caption") or "")
    if mode == "none":
        return ""
    if mode == "task":
        return (clip.get("task") or "")[:200]
    m_s = re.search(r"Scene:\s*(.*?)(?:\n\s*Action:|$)", full, re.S)
    m_a = re.search(r"Action:\s*(.*)$", full, re.S)
    scene = (m_s.group(1).strip() if m_s else full).strip()
    action = (m_a.group(1).strip() if m_a else "").strip()
    if mode == "scene":
        return scene[:280]
    if mode == "action":
        return action[:200] or (clip.get("task") or "")[:200]
    # full: keep the action even when the scene is long — the action is the
    # short, high-value half and truncation used to delete exactly it
    return (scene[:260] + (". " + action[:200] if action else ""))[:480]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--cats", default="winners")
    ap.add_argument("--model", default="qwen2.5-vl-7b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=8,
                    help="frames per prompt; use 4 to match the API path, whose "
                         "gateway caps gemma4-31b-it at 4 images")
    ap.add_argument("--capmode", default="full",
                    choices=["full", "scene", "action", "task", "none"],
                    help="which part of the caption to show")
    ap.add_argument("--no-caption", action="store_true",
                    help="caption-free control")
    ap.add_argument("--order", default="temporal",
                    choices=["temporal", "shuffled"],
                    help="frame order; shuffled is the temporal-reasoning control")
    ap.add_argument("--inject", default=None,
                    help="stage_signals.json -> prepend Stage-1/2 evidence")
    ap.add_argument("--all-clips", action="store_true",
                    help="keep clips with no rule text (clean negatives)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    data, clips = load(a.data)
    if not a.all_clips:
        clips = [c for c in clips if len(c.get("violated_rules") or "") > 4]
    if a.limit:
        clips = clips[:a.limit]
    cats = [x.strip() for x in a.cats.split(",") if x.strip() in BATTERIES]
    probes = [(nm, q) for cat in cats for nm, q in BATTERIES[cat]]
    SIG = percentile_table(data / a.inject) if a.inject else None
    torch, model, proc = load_model(a.model, a.device)
    print(f"domain [{a.model}]: {len(probes)} probes x {len(clips)} clips",
          flush=True)

    t0 = time.time()
    out = {}
    for i, cl in enumerate(clips):
        imgs = frames_pil(data, cl, a.frames)
        if a.order == "shuffled":
            # reuse the ordering frozen at staging so the control is identical
            # to the API path's and reproducible across runs
            idx = cl["order_shuffled"][:len(imgs)]
            imgs = [imgs[i] for i in idx if i < len(imgs)]
        cap = "" if a.no_caption else caption_for(cl, a.capmode)
        if cap:
            head = _P.format(n=len(imgs), caption=cap)
        else:
            head = (f"Look at these {len(imgs)} frames, sampled in order from a "
                    "video.\n")
        if SIG is not None:
            head += build_injection(cl["clip_id"], SIG)
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

    tag = '_'.join(cats) + f"__{a.model}"
    if a.no_caption:
        tag += "__nocap"
    if a.order != "temporal":
        tag += f"__{a.order}"
    if a.inject:
        tag += "__inject"
    if a.frames != 8:
        tag += f"__f{a.frames}"
    if a.capmode != "full":
        tag += f"__cap{a.capmode}"
    outp = data / f"domain_probes_{tag}.json"
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

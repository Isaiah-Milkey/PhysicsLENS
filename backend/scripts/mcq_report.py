"""
Score the forced-choice (MCQ) probe against the per-specialist labels.

Three things are measured, because a single "accuracy" would hide which one the
MCQ actually buys:

  ATTRIBUTION   per specialist, AUC of P(that option) — directly comparable to
                the independent-probe tables, same clips, same labels.
  ARGMAX        does the single chosen option match a true label for the clip?
                This is the number a person means by "does it say the right
                thing", and it is only available in the forced-choice framing.
  NONE          does P(none) separate clean clips from broken ones? The
                independent probes could not answer this at all.

POSITION BIAS IS CHECKED, NOT ASSUMED. Options are shuffled per clip, so if the
model is really answering "whatever is listed first", P(option) will correlate
with the option's printed position rather than with the labels. That correlation
is reported; without it a strong-looking result could be pure list-order.

Chance levels are stated for every number, since with 9 options and a skewed
label distribution "37% correct" can be either excellent or below chance.

Usage:
  python backend/scripts/mcq_report.py --data data/robotbench
"""
import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, boot_ci, tie_frac, spread, cats_of  # noqa: E402
from ablation_grid import category_scores  # noqa: E402
from benchmark_all import rule_cats  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mcq", default=None)
    ap.add_argument("--compare", default="domain_probes_robot__internvl3-8b.json")
    ap.add_argument("--min-pos", type=int, default=8)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    f = Path(a.mcq) if a.mcq else next(iter(sorted(data.glob("mcq_*.json"))), None)
    if not f:
        sys.exit("no mcq_*.json — run mcq_probe.py first")
    raw = json.loads(f.read_text())
    M = raw["mcq"]
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}
    # VideoPhy-2 has no human `categories` in its manifest; labels come from the
    # keyword taxonomy file. rule_cats() prefers human labels and falls back.
    CAT = rule_cats(data, clips)
    cmp_ps = None
    if (data / a.compare).exists():
        cmp_ps = json.loads((data / a.compare).read_text())["probe_scores"]

    print(f"{f.name} | model {raw['model']} | {len(M)} clips scored\n")

    ai = [c for c in M if c in clips and clips[c].get("generator") != "real"]
    viol = [c for c in ai if len(str(clips[c].get("violated_rules") or "")) > 4]
    clean_ai = [c for c in ai if clips[c].get("has_violation") is False]
    real = [c for c in M if c in clips and clips[c].get("generator") == "real"]

    # ── 1. attribution, per specialist ──────────────────────────────────────
    print("=" * 92)
    print("1. ATTRIBUTION — P(option) vs this-violation-or-a-different-one")
    print("=" * 92)
    print(f"{'specialist':13s} {'pos':>4s} {'neg':>4s} {'MCQ AUC [95% CI]':>24s} "
          f"{'8-probe':>9s} {'delta':>7s}")
    print("-" * 92)
    rows = {}
    for cat in ORDER:
        pos = [c for c in viol if cat in CAT.get(c, set())]
        neg = [c for c in viol if cat not in CAT.get(c, set())]
        if len(pos) < a.min_pos or len(neg) < 10:
            continue
        ids = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        v = np.array([M[c]["probs"][cat] for c in ids], float)
        m_auc = auc(y, v)
        lo, hi = boot_ci(y, v, n_boot=3000)
        base = float("nan")
        if cmp_ps:
            sc = category_scores(cmp_ps, [c for c in ids if c in cmp_ps]).get(cat)
            if sc and all(c in sc for c in ids):
                base = auc(y, [sc[c] for c in ids])
        mark = "!" if tie_frac(v) >= 0.75 else "~" if spread(v) < 0.10 else " "
        print(f"{cat:13s} {len(pos):4d} {len(neg):4d} "
              f"{f'{m_auc:.3f}{mark}[{lo:.2f},{hi:.2f}]':>24s} "
              f"{base:9.3f} {m_auc-base:+7.3f}")
        rows[cat] = dict(auc=m_auc, lo=lo, hi=hi, base=base, n_pos=len(pos))
    if rows:
        mm = np.mean([r["auc"] for r in rows.values()])
        bb = np.nanmean([r["base"] for r in rows.values()])
        print("-" * 92)
        print(f"{'MEAN':13s} {'':4s} {'':4s} {mm:24.3f} {bb:9.3f} {mm-bb:+7.3f}")

    # ── 2. argmax accuracy ──────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("2. ARGMAX — is the single chosen option one of the clip's true labels?")
    print("=" * 92)
    hit = tot = 0
    conf = Counter()
    for c in viol:
        pick = max(M[c]["probs"], key=M[c]["probs"].get)
        true = CAT.get(c, set())
        conf[pick] += 1
        tot += 1
        hit += pick in true
    # chance = pick the most common label every time
    base_lbl = Counter(x for c in viol for x in CAT.get(c, set()))
    top_lbl, top_n = base_lbl.most_common(1)[0]
    chance = sum(1 for c in viol if top_lbl in CAT.get(c, set())) / max(tot, 1)
    print(f"  correct on {hit}/{tot} = {hit/max(tot,1):.1%}")
    print(f"  chance (always answer '{top_lbl}', the commonest label) = {chance:.1%}")
    print(f"  uniform-random over 9 options            = {1/9:.1%}")
    print(f"\n  what it actually picks: "
          f"{dict(conf.most_common())}")
    print(f"  true label frequency  : {dict(base_lbl.most_common())}")

    # ── 3. the NONE option ──────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("3. NONE option — can P(none) separate clean from broken?")
    print("=" * 92)
    for label, negs in (("vs clean AI clips", clean_ai), ("vs real demos", real)):
        if len(negs) < 5:
            continue
        ids = viol + negs
        y = np.array([1] * len(viol) + [0] * len(negs))
        v = np.array([1.0 - M[c]["probs"]["none"] for c in ids], float)
        lo, hi = boot_ci(y, v, n_boot=3000)
        print(f"  1 - P(none) {label:22s} AUC {auc(y, v):.3f} "
              f"[{lo:.2f},{hi:.2f}]   ({len(viol)} broken vs {len(negs)})")
    pn = np.array([M[c]["probs"]["none"] for c in viol])
    pc = np.array([M[c]["probs"]["none"] for c in clean_ai]) if clean_ai else None
    print(f"\n  mean P(none) on broken clips : {pn.mean():.3f}")
    if pc is not None and len(pc):
        print(f"  mean P(none) on clean clips  : {pc.mean():.3f}")
    if real:
        pr = np.array([M[c]["probs"]["none"] for c in real])
        print(f"  mean P(none) on REAL demos   : {pr.mean():.3f}")

    # ── 4. position-bias control ────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("4. POSITION BIAS CONTROL — is it answering by list order?")
    print("=" * 92)
    letters = string.ascii_uppercase
    pos_p, lab_p = [], []
    for c in M:
        mp = M[c]["map"]
        for L, name in mp.items():
            pos_p.append(letters.index(L))
            lab_p.append(M[c]["probs"][name])
    r = float(np.corrcoef(pos_p, lab_p)[0, 1])
    print(f"  correlation(printed position, P(option)) = {r:+.3f}")
    print("  0 = no position preference; strongly negative = favours early letters")
    bypos = {}
    for p, v in zip(pos_p, lab_p):
        bypos.setdefault(p, []).append(v)
    print("  mean P by printed position: " + "  ".join(
        f"{letters[k]}:{np.mean(bypos[k]):.3f}" for k in sorted(bypos)))

    (data / "mcq_report.json").write_text(json.dumps(
        {"model": raw["model"], "attribution": rows,
         "argmax_acc": hit / max(tot, 1), "argmax_chance": chance,
         "position_corr": r}, indent=1))
    print(f"\n-> {data}/mcq_report.json")


if __name__ == "__main__":
    main()

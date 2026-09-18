"""
Does WRITING Stage-1/2 evidence into the prompt improve the specialist?

Two different ways to pass upstream evidence to Stage 3, and they do not give
the same answer:

  numeric fusion   combine the specialist's score with the signal OUTSIDE the
                   model, by rank-mean. Measured in stage_fusion_study.py.
  prompt injection state the evidence in words inside the prompt, e.g.
                   "Automated motion analysis of this clip reports that many
                   tracked points disappear in the middle of the frame."
                   Measured here.

Injection is the mechanism people usually mean by "pass Stage 2 to Stage 3", and
it is the one that composes with a normal VLM API. So it matters whether the
gain, if any, survives.

The comparison is PAIRED: identical clips, identical probes, identical judge,
identical frames — the prompt is the only difference. And it carries its own
manipulation check. Only ~75% of clips have any signal extreme enough to be worth
mentioning, so the rest get a byte-identical prompt. Those clips are a built-in
placebo group: if injected clips move and un-injected clips do not, the model
demonstrably read the text, and a null result is then about the evidence being
useless rather than unread.

Usage:
  python backend/scripts/injection_effect.py --data data/robotbench
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, cats_of  # noqa: E402
from ablation_grid import label, category_scores  # noqa: E402
from domain_probes import percentile_table, build_injection  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation", "collision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}
    SIG = percentile_table(data / "stage_signals.json")

    pairs = []
    for f in sorted(data.glob("domain_probes_*__inject.json")):
        base = data / f.name.replace("__inject", "")
        if base.exists():
            pairs.append((label(base.name), label(f.name)))
    if not pairs:
        sys.exit("no base/inject pairs found")

    def scores(name):
        p = data / ("domain_probes_" + name.replace("/", "__") + ".json")
        return json.loads(p.read_text())["probe_scores"]

    print(f"{len(pairs)} paired conditions (base vs +injected evidence)\n")
    print("=" * 96)
    print("MANIPULATION CHECK — did the injected text reach the model?")
    print("=" * 96)
    print(f"{'condition':30s} {'|d| with block':>15s} {'|d| no block':>14s} "
          f"{'%moved w/':>10s} {'%moved w/o':>11s}")
    print("-" * 96)
    for base, inj in pairs:
        b, i = scores(base), scores(inj)
        ids = [c for c in b if c in i]
        w = [c for c in ids if build_injection(c, SIG)]
        n = [c for c in ids if not build_injection(c, SIG)]

        def chg(sub):
            d = [abs(i[c][k] - b[c][k]) for c in sub for k in b[c] if k in i[c]]
            return (np.mean(d), np.mean([x > 1e-6 for x in d])) if d else (0, 0)
        mw, fw = chg(w)
        mn, fn = chg(n)
        print(f"{base[:30]:30s} {mw:15.4f} {mn:14.4f} {100*fw:9.0f}% {100*fn:10.0f}%")
    print("\nClips with no extreme signal get a byte-identical prompt, so their "
          "delta is the noise floor.")

    print("\n" + "=" * 96)
    print("EFFECT ON ACCURACY — paired delta in attribution AUC (inject - base)")
    print("=" * 96)
    print(f"{'condition':30s}" + "".join(f"{c[:9]:>11s}" for c in ORDER)
          + f"{'mean':>9s}")
    print("-" * 96)
    alld = {c: [] for c in ORDER}
    for base, inj in pairs:
        b, i = scores(base), scores(inj)
        row, ds = f"{base[:30]:30s}", []
        for cat in ORDER:
            ai = [c for c in clips if clips[c].get("generator") != "real"]
            pos = [c for c in ai if cat in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]
            neg = [c for c in ai if cat not in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]
            ids = pos + neg
            sb = category_scores(b, [c for c in ids if c in b]).get(cat)
            si = category_scores(i, [c for c in ids if c in i]).get(cat)
            if not sb or not si or not all(c in sb and c in si for c in ids):
                row += f"{'—':>11s}"
                continue
            y = np.array([1] * len(pos) + [0] * len(neg))
            d = auc(y, [si[c] for c in ids]) - auc(y, [sb[c] for c in ids])
            ds.append(d)
            alld[cat].append(d)
            row += f"{d:+11.3f}"
        row += f"{np.mean(ds):+9.3f}" if ds else f"{'—':>9s}"
        print(row)
    print("-" * 96)
    flat = [x for v in alld.values() for x in v]
    print(f"{'MEAN':30s}" + "".join(
        f"{np.mean(alld[c]):+11.3f}" if alld[c] else f"{'—':>11s}"
        for c in ORDER) + f"{np.mean(flat):+9.3f}")

    rng = np.random.default_rng(0)
    bs = [np.mean(rng.choice(flat, len(flat), replace=True))
          for _ in range(a.boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\noverall mean delta {np.mean(flat):+.4f}  95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]  (n={len(flat)} specialist-condition cells)")
    print("SIGNIFICANT" if lo > 0 or hi < 0 else
          "NOT SIGNIFICANT — injected evidence changes the answer without "
          "improving it.")
    (data / "injection_effect.json").write_text(json.dumps(
        {"mean_delta": float(np.mean(flat)), "ci": [float(lo), float(hi)],
         "n_cells": len(flat),
         "per_cat": {c: (float(np.mean(v)) if v else None)
                     for c, v in alld.items()}}, indent=1))
    print(f"\n-> {data}/injection_effect.json")


if __name__ == "__main__":
    main()

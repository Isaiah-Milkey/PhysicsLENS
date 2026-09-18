"""
Pick one configuration per specialist and report what it is honestly worth.

THE PROBLEM THIS SOLVES. The ablation grid scores ~20 conditions per specialist
and the best one gets reported. With 12-76 positives per category, the best of 20
is substantially the luckiest of 20. On VideoPhy-2 exactly this move inflated
specialist AUCs by +0.140, and that was over fewer conditions.

THE FIX. Repeated stratified split-half:
    select the winning condition on half A  ->  score it on half B
    select on half B                        ->  score it on half A
Averaged over many splits, the B-scores are what a newly chosen config would
actually deliver on clips nobody selected it against. The gap between the two is
the selection bias, printed rather than assumed.

The frozen config is still chosen on the full data — that is the best point
estimate of which condition is right. What split-half supplies is the honest
expectation for its AUC, which is what belongs in a report.

Usage:
  python backend/scripts/freeze_config.py --data data/robotbench --splits 200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import (PROBE_CAT, auc, boot_ci, tie_frac,  # noqa: E402
                                 spread, cats_of)
from ablation_grid import label  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", default="attribution",
                    choices=["attribution", "detect_clean", "detect_real"])
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}

    conds = {}
    for f in sorted(data.glob("domain_probes_*.json")):
        try:
            ps = json.loads(f.read_text()).get("probe_scores")
        except Exception:  # noqa: BLE001
            continue
        if ps:
            conds[label(f.name)] = ps
    print(f"{len(conds)} conditions | test = {a.test}\n")

    rng = np.random.default_rng(a.seed)
    out = {}
    for probe, cat in PROBE_CAT.items():
        ai = [c for c in clips if clips[c].get("generator") != "real"]
        pos = [c for c in ai if cat in cats_of(clips[c])
               and clips[c].get("has_violation") is not False]
        if a.test == "attribution":
            neg = [c for c in ai if cat not in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]
        elif a.test == "detect_clean":
            neg = [c for c in ai if clips[c].get("has_violation") is False]
        else:
            neg = [c for c in clips if clips[c].get("generator") == "real"]
        if len(pos) < a.min_pos or len(neg) < 5:
            continue

        # usable conditions: every clip scored, and the probe is not degenerate
        usable = {}
        for nm, ps in conds.items():
            if not all(c in ps and probe in ps[c] for c in pos + neg):
                continue
            v = np.array([ps[c][probe] for c in pos + neg], float)
            if tie_frac(v) >= 0.75 or spread(v) < 0.10:
                continue
            usable[nm] = v
        if not usable:
            continue
        y = np.array([1] * len(pos) + [0] * len(neg))

        # full-data pick = the frozen choice
        full = {nm: auc(y, v) for nm, v in usable.items()}
        pick = max(full, key=full.get)

        # repeated split-half: select on one half, score on the other
        sel_s, hel_s = [], []
        ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
        for _ in range(a.splits):
            pa = rng.permutation(ip)
            na = rng.permutation(ineg)
            A = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
            B = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
            for tr, te in ((A, B), (B, A)):
                if y[tr].sum() < 2 or y[te].sum() < 2:
                    continue
                sc = {nm: auc(y[tr], v[tr]) for nm, v in usable.items()}
                w = max(sc, key=sc.get)
                sel_s.append(sc[w])
                hel_s.append(auc(y[te], usable[w][te]))
        lo, hi = boot_ci(y, usable[pick], n_boot=3000)
        out[cat] = dict(
            n_pos=len(pos), n_neg=len(neg), n_conditions=len(usable),
            frozen=pick, frozen_auc=full[pick],
            frozen_lo=lo, frozen_hi=hi,
            selected_mean=float(np.mean(sel_s)) if sel_s else float("nan"),
            heldout_mean=float(np.mean(hel_s)) if hel_s else float("nan"),
            bias=float(np.mean(sel_s) - np.mean(hel_s)) if sel_s else float("nan"),
            runner_up=sorted(full.values(), reverse=True)[1] if len(full) > 1
            else float("nan"))

    print("=" * 100)
    print(f"FROZEN CONFIGURATION PER SPECIALIST  ({a.test})")
    print("=" * 100)
    print(f"{'specialist':12s} {'pos':>4s} {'cond':>5s} {'frozen condition':26s} "
          f"{'AUC [95% CI]':>20s} {'held-out':>9s} {'bias':>7s}")
    print("-" * 100)
    for cat in ORDER:
        r = out.get(cat)
        if not r:
            continue
        ci = f"{r['frozen_auc']:.3f} [{r['frozen_lo']:.2f},{r['frozen_hi']:.2f}]"
        print(f"{cat:12s} {r['n_pos']:4d} {r['n_conditions']:5d} "
              f"{r['frozen'][:26]:26s} {ci:>20s} {r['heldout_mean']:9.3f} "
              f"{r['bias']:+7.3f}")
    hv = [r["heldout_mean"] for r in out.values() if np.isfinite(r["heldout_mean"])]
    bv = [r["bias"] for r in out.values() if np.isfinite(r["bias"])]
    if hv:
        print("-" * 100)
        print(f"{'MEAN':12s} {'':4s} {'':5s} {'':26s} "
              f"{np.mean([r['frozen_auc'] for r in out.values()]):>20.3f} "
              f"{np.mean(hv):9.3f} {np.mean(bv):+7.3f}")
    print("\n'AUC' is the frozen condition scored on all clips — the number a "
          "best-of-N search reports.")
    print("'held-out' is what that search delivers on clips it did not choose "
          "against. Quote this one.")
    print("'bias' is the difference: how much the search flattered itself.")

    p = data / f"frozen_config_{a.test}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()

"""
Is per-specialist tuning worth it, or does ONE config generalise better?

Tuning a config per specialist looks obviously better — every specialist gets its
best condition. But with 12-77 positives per category and ~40 conditions to
choose from, "best" is substantially "luckiest", and the winner does not repeat.
Selection bias here grew from +0.066 at 20 conditions to +0.100 at 40 while the
held-out mean did not improve, which is the signature of a search that has run
out of signal and is now fitting noise.

This compares three strategies under the SAME repeated split-half protocol, so
the comparison is like-for-like:

  per-specialist   choose each specialist's condition on the training half
  global           choose ONE condition (best mean across specialists) on the
                   training half, apply it to every specialist
  fixed            a named condition, chosen by nobody, applied always — the
                   floor that any adaptive strategy has to beat to justify itself

Everything is scored on the held-out half. A strategy only earns its complexity
if it wins here, not in the table it selected itself on.

Usage:
  python backend/scripts/global_vs_tuned.py --data data/robotbench --splits 300
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, tie_frac, spread, cats_of  # noqa: E402
from ablation_grid import label, category_scores  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation", "collision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", default="attribution",
                    choices=["attribution", "detect_clean", "detect_real"])
    ap.add_argument("--splits", type=int, default=300)
    ap.add_argument("--fixed", default="robot/f4",
                    help="named always-on condition to use as the floor")
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

    # per specialist: labels + a usable score vector from every condition
    S, Y = {}, {}
    for cat in ORDER:
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
        ids = pos + neg
        usable = {}
        for nm, ps in conds.items():
            sc = category_scores(ps, [c for c in ids if c in ps]).get(cat)
            if not sc or not all(c in sc for c in ids):
                continue
            v = np.array([sc[c] for c in ids], float)
            if tie_frac(v) >= 0.75 or spread(v) < 0.10:
                continue
            usable[nm] = v
        if usable:
            S[cat] = usable
            Y[cat] = np.array([1] * len(pos) + [0] * len(neg))

    shared = set.intersection(*[set(v) for v in S.values()]) if S else set()
    print(f"{len(conds)} conditions | {len(S)} specialists | "
          f"{len(shared)} conditions usable for ALL specialists\n")

    rng = np.random.default_rng(a.seed)
    res = {k: {c: [] for c in S} for k in ("per_spec", "global", "fixed")}
    for _ in range(a.splits):
        idx = {}
        for cat in S:
            y = Y[cat]
            ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
            pa, na = rng.permutation(ip), rng.permutation(ineg)
            A = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
            B = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
            idx[cat] = (A, B)
        for tr_i, te_i in (0, 1), (1, 0):
            # global: one condition, chosen by mean training AUC over specialists
            gs = {}
            for nm in shared:
                v = [auc(Y[c][idx[c][tr_i]], S[c][nm][idx[c][tr_i]]) for c in S]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    gs[nm] = float(np.mean(v))
            gpick = max(gs, key=gs.get) if gs else None
            for cat in S:
                y, tr, te = Y[cat], idx[cat][tr_i], idx[cat][te_i]
                if y[te].sum() < 2 or (1 - y[te]).sum() < 2:
                    continue
                sc = {nm: auc(y[tr], v[tr]) for nm, v in S[cat].items()}
                sc = {k: v for k, v in sc.items() if np.isfinite(v)}
                if sc:
                    w = max(sc, key=sc.get)
                    res["per_spec"][cat].append(auc(y[te], S[cat][w][te]))
                if gpick and gpick in S[cat]:
                    res["global"][cat].append(auc(y[te], S[cat][gpick][te]))
                if a.fixed in S[cat]:
                    res["fixed"][cat].append(auc(y[te], S[cat][a.fixed][te]))

    print("=" * 84)
    print(f"HELD-OUT AUC BY STRATEGY  ({a.test}, {a.splits} split-halves)")
    print("=" * 84)
    print(f"{'specialist':13s} {'per-specialist':>16s} {'global (1 cfg)':>16s} "
          f"{'fixed':>16s}")
    print("-" * 84)
    for cat in ORDER:
        if cat not in S:
            continue
        row = [np.mean(res[k][cat]) if res[k][cat] else float("nan")
               for k in ("per_spec", "global", "fixed")]
        print(f"{cat:13s} {row[0]:16.3f} {row[1]:16.3f} {row[2]:16.3f}")
    print("-" * 84)
    means = []
    for k in ("per_spec", "global", "fixed"):
        v = [np.mean(res[k][c]) for c in S if res[k][c]]
        means.append(float(np.mean(v)) if v else float("nan"))
    print(f"{'MEAN':13s} {means[0]:16.3f} {means[1]:16.3f} {means[2]:16.3f}")
    print(f"\nfixed condition = {a.fixed}")
    best = ["per-specialist", "global", f"fixed ({a.fixed})"][int(np.nanargmax(means))]
    print(f"WINNER: {best}")
    if means[0] <= max(means[1], means[2]) + 1e-9:
        print("Per-specialist tuning does NOT beat a single config out of sample: "
              "the per-specialist winners are selection noise, not real "
              "specialisation.")
    (data / f"global_vs_tuned_{a.test}.json").write_text(json.dumps(
        {"test": a.test, "means": dict(zip(
            ["per_spec", "global", "fixed"], means)), "fixed": a.fixed,
         "per_cat": {k: {c: (float(np.mean(v)) if v else None)
                         for c, v in res[k].items()} for k in res}}, indent=1))
    print(f"\n-> {data}/global_vs_tuned_{a.test}.json")


if __name__ == "__main__":
    main()

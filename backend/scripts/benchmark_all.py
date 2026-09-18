"""
Benchmark every specialist across every dataset we have.

WHY CROSS-DATASET IS THE STRONGEST TEST WE CAN RUN. Within one dataset, choosing
the best of N conditions inflates the result, and the only defence is split-half
machinery that estimates the inflation. Across datasets there is nothing to
estimate: pick the configuration on VideoPhy-2, evaluate it on RobotBench, and
the number is clean by construction. It also asks a harder and more useful
question — not "which config wins here" but "does a config chosen anywhere
survive being moved". A specialist that needs a different setup per dataset has
not been tuned, it has been fitted.

The two datasets are genuinely different, which is the point:
  VideoPhy-2   704 annotated clips, general physics (balls, ramps, sports),
               7 text-to-video generators, dramatic failures
  RobotBench   145 annotated clips, robot manipulation, 2 generators, quiet
               failures (a gripper closing wrong, a cloth sliding untouched),
               plus 82 real demonstrations as true clean negatives

Per-category positives are 4-5x larger on VideoPhy-2, so it is also where the
under-powered specialists (friction 7, fluid 12 on RobotBench) finally get enough
positives for an AUC to mean anything.

Usage:
  python backend/scripts/benchmark_all.py --datasets data/videophy1200,data/robotbench
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, boot_ci, tie_frac, spread, cats_of  # noqa: E402
from ablation_grid import label, category_scores  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def rule_cats(data: Path, clips):
    """Category labels. RobotBench has human `categories`; VideoPhy-2 does not,
    so fall back to the keyword taxonomy's per-clip output."""
    rc = data / "rule_categories.json"
    per = json.loads(rc.read_text())["per_clip"] if rc.exists() else {}
    out = {}
    for cid, c in clips.items():
        v = cats_of(c) or set(per.get(cid, []))
        out[cid] = v
    return out


def collect(dpath: Path, variants):
    """All usable (condition -> per-specialist score vector) for one dataset."""
    clips = {c["clip_id"]: c
             for c in json.loads((dpath / "manifest.json").read_text())["clips"]}
    cat_of = rule_cats(dpath, clips)
    conds = {}
    for tag, root in variants:
        root = Path(root)
        if not (root / "manifest.json").exists():
            continue
        for f in sorted(root.glob("domain_probes_*.json")):
            try:
                ps = json.loads(f.read_text()).get("probe_scores")
            except Exception:  # noqa: BLE001
                continue
            if ps:
                conds[f"{tag}:{label(f.name)}"] = ps
    return clips, cat_of, conds


def spec_vectors(clips, cat_of, conds, cat, min_pos=8):
    ai = [c for c in clips if clips[c].get("generator") != "real"]
    pos = [c for c in ai if cat in cat_of.get(c, set())
           and clips[c].get("has_violation") is not False
           and len(str(clips[c].get("violated_rules") or "")) > 4]
    neg = [c for c in ai if cat not in cat_of.get(c, set())
           and clips[c].get("has_violation") is not False
           and len(str(clips[c].get("violated_rules") or "")) > 4]
    if len(pos) < min_pos or len(neg) < 10:
        return None
    ids = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    out = {}
    for nm, ps in conds.items():
        sc = category_scores(ps, [c for c in ids if c in ps]).get(cat)
        if not sc or not all(c in sc for c in ids):
            continue
        v = np.array([sc[c] for c in ids], float)
        if tie_frac(v) >= 0.75 or spread(v) < 0.10:
            continue
        out[nm] = v
    return (y, out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="data/videophy1200,data/robotbench")
    ap.add_argument("--min-pos", type=int, default=8)
    a = ap.parse_args()

    DS = {}
    for d in a.datasets.split(","):
        d = d.strip()
        if not d:
            continue
        base = Path(d) if Path(d).is_absolute() else ROOT / d
        variants = [("plain", base), ("trails", Path(str(base) + "_trails")),
                    ("diff", Path(str(base) + "_diff"))]
        clips, cat_of, conds = collect(base, variants)
        DS[base.name] = dict(clips=clips, cat_of=cat_of, conds=conds)
        print(f"{base.name}: {len(clips)} clips, {len(conds)} conditions "
              f"(incl. motion variants)")

    data = {}
    for nm, D in DS.items():
        data[nm] = {}
        for cat in ORDER:
            r = spec_vectors(D["clips"], D["cat_of"], D["conds"], cat, a.min_pos)
            if r:
                data[nm][cat] = r

    # ── per dataset, best condition (optimistic — selected in-sample) ────────
    print(f"\n{'='*100}")
    print("PER DATASET — best condition per specialist (in-sample, optimistic)")
    print("=" * 100)
    names = list(data)
    print(f"{'specialist':13s}" + "".join(
        f"{n[:22]:>26s}" for n in names))
    print("-" * 100)
    for cat in ORDER:
        row = f"{cat:13s}"
        for n in names:
            if cat not in data[n]:
                row += f"{'—':>26s}"
                continue
            y, cols = data[n][cat]
            nm, v = max(((k, v) for k, v in cols.items()),
                        key=lambda kv: auc(y, kv[1]))
            row += f"{auc(y, v):8.3f} (n={int(y.sum()):3d}) {nm.split(':')[0][:6]:>6s}"
        print(row)

    # ── cross-dataset transfer: choose on A, score on B ─────────────────────
    if len(names) >= 2:
        A, B = names[0], names[1]
        print(f"\n{'='*100}")
        print(f"CROSS-DATASET TRANSFER — configuration chosen on one dataset, "
              f"scored on the other")
        print("=" * 100)
        print("No selection bias is possible here: the evaluation clips played "
              "no part in the choice.\n")
        print(f"{'specialist':13s} {'pick on':>12s} {'-> score on':>12s} "
              f"{'AUC [95% CI]':>22s} {'in-sample best':>15s} {'drop':>7s}")
        print("-" * 100)
        summary = {}
        for cat in ORDER:
            if cat not in data[A] or cat not in data[B]:
                continue
            yA, cA = data[A][cat]
            yB, cB = data[B][cat]
            shared = sorted(set(cA) & set(cB))
            if not shared:
                continue
            for src, dst, ys, cs, yd, cd in ((A, B, yA, cA, yB, cB),
                                             (B, A, yB, cB, yA, cA)):
                pick = max(shared, key=lambda k: auc(ys, cs[k]))
                a_dst = auc(yd, cd[pick])
                lo, hi = boot_ci(yd, cd[pick], n_boot=3000)
                best = max(auc(yd, cd[k]) for k in cd)
                print(f"{cat:13s} {src[:12]:>12s} {dst[:12]:>12s} "
                      f"{f'{a_dst:.3f} [{lo:.2f},{hi:.2f}]':>22s} "
                      f"{best:15.3f} {a_dst-best:+7.3f}")
                summary.setdefault(cat, {})[f"{src}->{dst}"] = dict(
                    config=pick, auc=a_dst, lo=lo, hi=hi, in_sample_best=best,
                    n_pos=int(yd.sum()), n=len(yd))
        vals = [v["auc"] for c in summary.values() for v in c.values()]
        if vals:
            print("-" * 100)
            print(f"{'MEAN transferred AUC':13s} {np.mean(vals):>52.3f}")
        out = ROOT / "data" / "benchmark_all.json"
        out.write_text(json.dumps(summary, indent=1))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()

"""
Evaluate the domain-knowledge probe batteries, per specialist.

Answers three things, in increasing order of usefulness:

  1. Does the battery beat the single generic probe for its category?
  2. Does it beat the all-signal ceiling measured earlier (55 engineered
     signals + 768 embedding dims)? That ceiling is the bar to clear — if a
     5-question battery beats 823 dimensions of everything else, the missing
     ingredient really was domain specificity.
  3. WHICH sub-check carries the signal. This is the part that transfers: if
     "free fall must accelerate" discriminates gravity and the other four
     gravity questions do not, that is a concrete finding about what to ask,
     reusable outside this benchmark.

Scored on the discriminative set throughout — positives are clips whose human
rule is in category S, negatives are clips carrying a violation of a DIFFERENT
category. Against clean clips instead, any "something is wrong" detector looks
like a specialist.

Usage:
  python backend/scripts/domain_report.py --data data/videophy1200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load                                   # noqa: E402
from specialist_report import auc, auc_ci                        # noqa: E402
from specialist_fusion import fit_logistic, apply_logistic       # noqa: E402
from retrieval_memory import folds                               # noqa: E402
from rule_taxonomy import PROBE_FOR                              # noqa: E402
from domain_probes import BATTERIES                              # noqa: E402

# Ceilings measured before domain probes existed: all 55 engineered signals plus
# 768 DINOv2 dims, 5-fold CV, same discriminative sets. The bar to beat.
PRIOR_CEILING = {"collision": 0.734, "gravity": 0.517, "deformation": 0.599,
                 "momentum": 0.613, "friction": 0.771, "fluid": 0.892,
                 "permanence": 0.569}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    a = ap.parse_args()
    data, clips = load(a.data)
    n = len(clips)
    cats_of = json.loads((data / "rule_categories.json").read_text())["per_clip"]
    co = [set(cats_of.get(c["clip_id"], [])) for c in clips]
    has = [len(c.get("violated_rules") or "") > 4 for c in clips]

    # every domain battery file written so far
    dom = {}
    for f in sorted(data.glob("domain_probes_*.json")):
        d = json.loads(f.read_text())
        if d.get("n", 0) < 100:
            continue
        for cid, r in d["probe_scores"].items():
            dom.setdefault(cid, {}).update(r)
    if not dom:
        sys.exit("no full domain-probe files yet")
    print(f"domain sub-scores for {len(dom)} clips")

    # generic single probe, for the head-to-head
    gen = {}
    p = data / "method_probes.json"
    if p.exists():
        gen = json.loads(p.read_text())["probe_scores"]

    F = folds(n, 5)
    print(f"\n{'specialist':13s} {'n+':>4s} {'generic':>8s} {'mean':>7s} "
          f"{'bestsub':>8s} {'BATTERY(fit)':>20s} {'ceiling':>8s} {'best-vs-ceil':>13s}")
    print("-" * 96)
    out = {}
    for cat, probes in BATTERIES.items():
        # "momentum_v2" is a second question-set for the SAME label as "momentum";
        # strip the suffix for ground truth so v1 and v2 are directly comparable.
        lab = cat.replace("_v2", "")
        names = [nm for nm, _ in probes]
        if not any(nm in v for v in dom.values() for nm in names):
            continue
        y = np.array([1.0 if lab in s else 0.0 for s in co])
        ds = [i for i in range(n)
              if ((lab in co[i]) or (has[i] and co[i]))
              and dom.get(clips[i]["clip_id"]) is not None
              and all(nm in dom[clips[i]["clip_id"]] for nm in names)]
        npos = int(sum(1 for i in ds if lab in co[i]))
        if npos < 12:
            continue
        X = np.zeros((n, len(names)))
        for i in range(n):
            r = dom.get(clips[i]["clip_id"]) or {}
            X[i] = [r.get(nm, 0.0) for nm in names]
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)

        # Two fixes over the first version, both of which suppressed the fitted
        # score below its own best single feature:
        #  1. TRAIN ON THE DISCRIMINATIVE SET ONLY. Fitting on all clips teaches
        #     "broken vs clean" and then we evaluate "which kind of broken" —
        #     a distribution mismatch that wastes the model's capacity.
        #  2. BALANCE THE CLASSES. friction is 35 positives in 704; unweighted
        #     descent with L2 collapses toward the intercept and predicts
        #     everything negative. Oversampling the positives fixes it without
        #     touching the evaluation.
        dss = set(ds)
        pred, mean_pred, best_pred = {}, {}, {}
        for f in range(5):
            te = [i for i in F[f] if i in dss]
            tr = [i for g in range(5) if g != f for i in F[g] if i in dss]
            if not te or not tr:
                continue
            pos = [i for i in tr if y[i] > 0.5]
            neg = [i for i in tr if y[i] <= 0.5]
            if not pos or not neg:
                continue
            reps = max(1, int(round(len(neg) / len(pos))))
            bal = neg + pos * reps
            w = fit_logistic(X[bal], y[bal], l2=5.0, iters=800)
            for i, v in zip(te, apply_logistic(X[te], w)):
                pred[i] = float(v)
            # fit-free baseline: unweighted mean of the sub-checks, with each
            # sub-check's sign set from the TRAINING fold so it cannot peek
            sign = np.array([1.0 if auc([X[i, j] for i in pos],
                                        [X[i, j] for i in neg]) >= 0.5 else -1.0
                             for j in range(len(names))])
            for i in te:
                mean_pred[i] = float(np.mean(X[i] * sign))
            # single best sub-check, CHOSEN ON TRAIN, applied to test — the
            # honest version of "best sub-check", free of selection bias
            aucs = [auc([X[i, j] for i in pos], [X[i, j] for i in neg]) or 0.5
                    for j in range(len(names))]
            bj = int(np.argmax([max(v, 1 - v) for v in aucs]))
            s = 1.0 if aucs[bj] >= 0.5 else -1.0
            for i in te:
                best_pred[i] = float(X[i, bj] * s)
        def ev(d):
            p_ = [d[i] for i in ds if i in d and lab in co[i]]
            q_ = [d[i] for i in ds if i in d and lab not in co[i]]
            return (auc(p_, q_), auc_ci(p_, q_)) if p_ and q_ else (None, (None, None))
        ab, (lo, hi) = ev(pred)
        am, _ = ev(mean_pred)
        abest, _ = ev(best_pred)

        gname = PROBE_FOR.get(lab)
        ag = None
        if gen and gname:
            gv = [(gen.get(clips[i]["clip_id"]) or {}).get(gname) for i in range(n)]
            gi = [i for i in ds if gv[i] is not None]
            if gi:
                ag = auc([gv[i] for i in gi if lab in co[i]],
                         [gv[i] for i in gi if lab not in co[i]])
        cl = PRIOR_CEILING.get(lab)
        # headline = the best domain-only variant, so a weak optimiser cannot
        # be mistaken for a weak idea
        cands = [v for v in (ab, am, abest) if v is not None]
        top = max(cands) if cands else 0.5
        d = (top - cl) if cl else 0.0
        mark = "  *" if d > 0 else ""
        nan = float('nan')
        print(f"{cat:13s} {npos:4d} {ag or nan:8.3f} {am or nan:7.3f} "
              f"{abest or nan:8.3f} {ab or nan:.3f} [{lo:.2f},{hi:.2f}] "
              f"{cl or nan:8.3f} {d:+13.3f}{mark}")
        out[cat] = {"n_pos": npos, "generic": round(ag, 3) if ag else None,
                    "mean": round(am, 3) if am else None,
                    "best_subcheck_cv": round(abest, 3) if abest else None,
                    "battery_fit": round(ab, 3) if ab else None,
                    "ci": [round(lo, 3), round(hi, 3)] if lo else None,
                    "prior_ceiling": cl, "best_minus_ceiling": round(d, 3)}

        # which individual check carries it
        sub = []
        for j, nm in enumerate(names):
            v = auc([X[i, j] for i in ds if lab in co[i]],
                    [X[i, j] for i in ds if lab not in co[i]])
            if v is not None:
                sub.append((max(v, 1 - v), nm, v))
        sub.sort(reverse=True)
        print("      sub-checks: " + "  ".join(f"{nm}={v:.3f}" for _, nm, v in sub))
        out[cat]["sub_checks"] = {nm: round(v, 3) for _, nm, v in sub}

    print("\n  *  best domain-only variant beats the prior all-signal ceiling")
    print("  note: `ceiling` was fit on ALL clips; domain columns are fit on the")
    print("        discriminative set only, so treat the gap as indicative.")
    outp = data / "domain_report.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()

"""
Per-specialist: VLM probe vs physical tools vs retrieval vs all three fused.

Every number is 5-fold cross-validated and scored on the DISCRIMINATIVE set —
positives are clips whose human rule is in category S, negatives are clips that
have a violation of a DIFFERENT category. Scoring against clean clips instead
would reward a detector that only knows "something is wrong here", which is
exactly what we are trying to move past.

Channels:
  probe      the VLM's category-matched defect probe (no fitting)
  tools      logistic regression over that specialist's physical features from
             specialist_tools.py, fit on the training folds only
  retrieval  k-NN over DINOv2 clip embeddings predicting P(category violated)
             from neighbours' labels, with same-caption neighbours EXCLUDED
             (VideoPhy-2 renders one caption with several generators, so without
             that exclusion the neighbour is often the same prompt and the model
             recalls prompt difficulty instead of judging the clip)
  fused      rank mean of whichever channels are available

Usage:
  python backend/scripts/specialist_fusion.py --data data/videophy1200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gepa_optimize import load                                  # noqa: E402
from method_report import rank01                                # noqa: E402
from rule_taxonomy import CATEGORIES, PROBE_FOR                 # noqa: E402
from retrieval_memory import folds                              # noqa: E402
from specialist_report import auc, auc_ci, knn_category         # noqa: E402

# Which physical features belong to which specialist. `coll` borrows the
# momentum impulse features because "action at a distance" is defined by a
# velocity kick, and the kick magnitude is what makes the distance meaningful.
GROUPS = {
    "gravity":     ["grav_"],
    "momentum":    ["mom_"],
    "collision":   ["coll_", "mom_impulse", "mom_gain"],
    "deformation": ["deform_"],
    "friction":    ["fric_"],
    "permanence":  ["perm_"],
    "fluid":       [],
}


def fit_logistic(X, y, l2=1.0, iters=300, lr=0.5):
    """Tiny batch logistic regression. Written out rather than pulled from
    sklearn to keep this script dependency-free and the regularisation explicit —
    some categories have only ~35 positives and will overfit without it."""
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        gr = X.T @ (p - y) / len(X) + l2 * np.r_[w[:-1], 0.0] / len(X)
        w -= lr * gr
    return w


def apply_logistic(X, w):
    X = np.hstack([X, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/videophy1200")
    ap.add_argument("--model", default=None, help="probe model; default = best")
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()

    data, clips = load(a.data)
    n = len(clips)
    cats = json.loads((data / "rule_categories.json").read_text())["per_clip"]
    cat_of = [set(cats.get(c["clip_id"], [])) for c in clips]
    caps = [c["caption"] for c in clips]
    has_rule = [len(c.get("violated_rules") or "") > 4 for c in clips]

    tools = json.loads((data / "specialist_tools.json").read_text())["features"]
    tnames = sorted({k for v in tools.values() for k in v})
    X_all = np.array([[float((tools.get(c["clip_id"]) or {}).get(t, np.nan))
                       for t in tnames] for c in clips])
    col_med = np.nanmedian(X_all, axis=0)
    X_all = np.where(np.isfinite(X_all), X_all, col_med)
    X_all = (X_all - X_all.mean(0)) / (X_all.std(0) + 1e-9)

    models = {}
    for f in sorted(data.glob("method_probes*.json")):
        d = json.loads(f.read_text())
        if d.get("n", 0) >= 100 and "probe_scores" in d:
            models[d.get("model", f.stem)] = d["probe_scores"]
    mname = a.model or max(models, key=lambda m: len(models[m]))
    ps = models[mname]
    print(f"{n} clips | probe model: {mname} | tool features: {len(tnames)}")

    emb = data / "emb_dinov2.npy"
    X_emb = np.load(emb) if emb.exists() else None
    F = folds(n, 5)

    order = [c for c in CATEGORIES if PROBE_FOR.get(c)]
    print(f"\n{'specialist':13s} {'n+':>4s} {'probe':>8s} {'tools':>8s} "
          f"{'retriev':>8s} {'FUSED':>8s} {'95% CI':>16s} {'best gain':>10s}")
    print("-" * 92)
    out = {}
    for cat in order:
        y = np.array([1.0 if cat in s else 0.0 for s in cat_of])
        # discriminative set: this category vs OTHER violators
        dset = [i for i in range(n)
                if (cat in cat_of[i]) or (has_rule[i] and cat_of[i])]
        if sum(y[i] for i in dset) < 15:
            print(f"{cat:13s} {int(y.sum()):4d}   (too few positives)")
            continue

        pname = PROBE_FOR[cat]
        probe = [(ps.get(c["clip_id"]) or {}).get(pname) for c in clips]

        cols = [j for j, t in enumerate(tnames)
                if any(t.startswith(p) for p in GROUPS.get(cat, []))]
        tool_pred, ret_pred = {}, {}
        for f in range(5):
            te = [i for i in F[f] if i in set(dset)]
            tr = [i for g in range(5) if g != f for i in F[g]]
            if cols and te:
                w = fit_logistic(X_all[np.ix_(tr, cols)], y[tr], l2=5.0)
                for i, v in zip(te, apply_logistic(X_all[np.ix_(te, cols)], w)):
                    tool_pred[i] = float(v)
            if X_emb is not None and te:
                ret_pred.update(knn_category(X_emb, y, tr, te, a.k, exclude=caps))

        idx = [i for i in dset if probe[i] is not None]
        chans, names = [], []
        s = set(idx)
        chans.append(rank01([probe[i] if i in s else None for i in range(n)]))
        names.append("probe")
        if tool_pred:
            chans.append(rank01([tool_pred.get(i) if i in s else None
                                 for i in range(n)]))
            names.append("tools")
        if ret_pred:
            chans.append(rank01([ret_pred.get(i) if i in s else None
                                 for i in range(n)]))
            names.append("retriev")
        idx = [i for i in idx if all(c[i] is not None for c in chans)]

        def sp(d):
            return ([d[i] for i in idx if cat in cat_of[i]],
                    [d[i] for i in idx if cat not in cat_of[i]])
        singles = {}
        for nm, ch in zip(names, chans):
            p_, q_ = sp({i: ch[i] for i in idx})
            singles[nm] = auc(p_, q_) or 0.5
        fu = {i: float(np.mean([c[i] for c in chans])) for i in idx}
        fp, fn = sp(fu)
        af = auc(fp, fn) or 0.5
        lo, hi = auc_ci(fp, fn)
        gain = af - singles["probe"]
        print(f"{cat:13s} {int(y.sum()):4d} {singles['probe']:8.3f} "
              f"{singles.get('tools', float('nan')):8.3f} "
              f"{singles.get('retriev', float('nan')):8.3f} {af:8.3f} "
              f"[{lo:.2f},{hi:.2f}]  {gain:+10.3f}")
        out[cat] = {"n_pos": int(y.sum()), **{k: round(v, 3) for k, v in singles.items()},
                    "fused": round(af, 3),
                    "ci": [round(lo, 3), round(hi, 3)] if lo else None,
                    "gain_over_probe": round(gain, 3)}

    outp = data / "specialist_fusion.json"
    outp.write_text(json.dumps({"model": mname, "results": out}, indent=1))
    if out:
        print(f"\nmean fused AUC across specialists: "
              f"{np.mean([v['fused'] for v in out.values()]):.3f}   "
              f"(probe alone {np.mean([v['probe'] for v in out.values()]):.3f})")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()

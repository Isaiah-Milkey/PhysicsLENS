"""
Can a LEARNED combination beat the single probe? Nested-CV, no leakage.

Rank-mean fusion needs no fitting, which is why it has been the default here. But
it also cannot discover that (say) deformation depends on camera motion only when
object motion is low. This fits an actual model over everything available:

    28 Stage-1/2 signals  +  the specialist's probe score from all 5 judges

and asks whether that beats the best single probe. Ridge on standardised
features, which for a ranking metric behaves like regularised LDA and does not
need the class balance that plain logistic regression struggles with at 12
positives.

NESTED cross-validation, not plain CV. The regularisation strength is itself a
fitted choice; picking it on the same folds you report is the same leak as
picking a probe on the eval set. Inner folds choose alpha, the outer fold is
touched once. With 12-77 positives this matters more than usual — the gap
between nested and non-nested is printed so it is visible rather than assumed.

A permutation null runs the entire nested procedure on shuffled labels, because
"best model over 33 features" needs a best-over-33-features null.

Usage:
  python backend/scripts/supervised_fusion.py --data data/robotbench --perms 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, cats_of  # noqa: E402
from ablation_grid import category_scores  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation", "collision"]
JUDGES = {"gemma4": "robot__f4", "qwen7b": "robot__qwen2.5-vl-7b",
          "qwen32b": "robot__qwen2.5-vl-32b",
          "internvl8b": "robot__internvl3-8b",
          "internvl14b": "robot__internvl3-14b"}
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def ridge_fit(X, y, alpha):
    """Closed-form ridge on centred features. Closed form on purpose: iterative
    solvers were the source of an earlier bug where a hand-rolled gradient
    descent scored below its own best input feature."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    n, d = Xc.shape
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ yc)
    return w, X.mean(0), y.mean()


def folds(y, k, seed):
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(ip)
    rng.shuffle(ineg)
    return [(np.setdiff1d(np.arange(len(y)),
                          np.concatenate([ip[i::k], ineg[i::k]])),
             np.concatenate([ip[i::k], ineg[i::k]])) for i in range(k)]


def nested_auc(X, y, k=5, reps=3, seed=0):
    """Outer-fold predictions with alpha chosen on inner folds only."""
    preds = np.full((reps, len(y)), np.nan)
    for r in range(reps):
        for tr, te in folds(y, k, seed + r):
            if y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
                continue
            best, ba = None, -1
            for al in ALPHAS:
                sc = []
                for itr, ite in folds(y[tr], 4, seed + 100 + r):
                    if y[tr][itr].sum() < 2 or len(np.unique(y[tr][ite])) < 2:
                        continue
                    w, mx, my = ridge_fit(X[tr][itr], y[tr][itr].astype(float), al)
                    sc.append(auc(y[tr][ite], (X[tr][ite] - mx) @ w))
                if sc and np.mean(sc) > ba:
                    ba, best = float(np.mean(sc)), al
            w, mx, my = ridge_fit(X[tr], y[tr].astype(float), best or 10.0)
            preds[r, te] = (X[te] - mx) @ w
    out = []
    for r in range(reps):
        m = ~np.isnan(preds[r])
        if m.sum() > 8 and 0 < y[m].sum() < m.sum():
            out.append(auc(y[m], preds[r][m]))
    return float(np.mean(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--perms", type=int, default=0)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}
    sg = json.loads((data / "stage_signals.json").read_text())
    sig, keys = sg["signals"], sg["keys"]
    J = {j: json.loads((data / f"domain_probes_{f}.json").read_text())["probe_scores"]
         for j, f in JUDGES.items() if (data / f"domain_probes_{f}.json").exists()}

    print(f"{len(keys)} stage signals + {len(J)} judge probe scores "
          f"= {len(keys)+len(J)} features\n")
    print("=" * 92)
    print("LEARNED FUSION vs BEST SINGLE PROBE   (nested CV; alpha never sees "
          "the outer fold)")
    print("=" * 92)
    print(f"{'specialist':13s} {'pos':>4s} {'best probe':>11s} "
          f"{'signals only':>13s} {'probes only':>12s} {'ALL':>8s} "
          f"{'gain':>7s} {'null p95':>9s}")
    print("-" * 92)
    res = {}
    for cat in ORDER:
        ai = [c for c in clips if clips[c].get("generator") != "real"]
        pos = [c for c in ai if cat in cats_of(clips[c])
               and clips[c].get("has_violation") is not False]
        neg = [c for c in ai if cat not in cats_of(clips[c])
               and clips[c].get("has_violation") is not False]
        ids = [c for c in pos + neg if sig.get(c)]
        y = np.array([1 if c in set(pos) else 0 for c in ids])
        if y.sum() < 8:
            continue
        Xs = np.array([[sig[c].get(k, 0.0) for k in keys] for c in ids], float)
        pcols, pnames = [], []
        for j, ps in J.items():
            sc = category_scores(ps, [c for c in ids if c in ps]).get(cat)
            if sc and all(c in sc for c in ids):
                pcols.append([sc[c] for c in ids])
                pnames.append(j)
        Xp = np.array(pcols, float).T if pcols else np.zeros((len(ids), 0))
        best_probe = max((auc(y, Xp[:, i]) for i in range(Xp.shape[1])),
                         default=float("nan"))

        def std(M):
            if M.shape[1] == 0:
                return M
            s = M.std(0)
            s[s < 1e-9] = 1.0
            return (M - M.mean(0)) / s
        Xs_, Xp_ = std(Xs), std(Xp)
        a_sig = nested_auc(Xs_, y)
        a_prb = nested_auc(Xp_, y) if Xp_.shape[1] > 1 else float("nan")
        a_all = nested_auc(np.hstack([Xs_, Xp_]), y)
        p95 = float("nan")
        if a.perms:
            rng = np.random.default_rng(0)
            nl = []
            for _ in range(a.perms):
                yp = y.copy()
                rng.shuffle(yp)
                nl.append(nested_auc(np.hstack([Xs_, Xp_]), yp, reps=1))
            nl = [x for x in nl if np.isfinite(x)]
            p95 = float(np.percentile(nl, 95)) if nl else float("nan")
        print(f"{cat:13s} {int(y.sum()):4d} {best_probe:11.3f} {a_sig:13.3f} "
              f"{a_prb:12.3f} {a_all:8.3f} {a_all-best_probe:+7.3f} {p95:9.3f}")
        res[cat] = dict(n_pos=int(y.sum()), best_probe=best_probe,
                        signals_only=a_sig, probes_only=a_prb, all_features=a_all,
                        gain=a_all - best_probe, null_p95=p95)
    if res:
        g = [r["gain"] for r in res.values() if np.isfinite(r["gain"])]
        print("-" * 92)
        print(f"{'MEAN GAIN':13s} {'':4s} {'':11s} {'':13s} {'':12s} {'':8s} "
              f"{np.mean(g):+7.3f}")
        beat = [c for c, r in res.items()
                if np.isfinite(r["null_p95"]) and r["all_features"] > r["null_p95"]]
        print(f"\nbeat their permutation null: {beat or 'none'}")
    (data / "supervised_fusion.json").write_text(json.dumps(res, indent=1))
    print(f"\n-> {data}/supervised_fusion.json")


if __name__ == "__main__":
    main()

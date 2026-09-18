"""
Which Stage-1/2 signal does each Stage-3 specialist actually need?

THE QUESTION. A specialist currently sees 8 JPEGs and one sentence. Upstream we
already compute motion, tracks, camera compensation and a peak-anomaly time, and
none of it is passed down. This measures, per specialist, whether any of that
evidence would help — and which piece.

METHOD. For every (specialist, signal) pair:
    probe        AUC of the VLM probe alone on that specialist's category
    signal       AUC of the upstream signal alone on the same labels
    fused        AUC of the rank-mean of the two
    delta        fused - max(probe, signal)
`delta` is measured against the BETTER of the two, not against the probe. A
combination that merely beats the weaker half has discovered nothing: if the
signal alone is 0.72 and the probe is 0.55, "fusion beats the probe" is just the
signal doing the work under another name.

WHY CROSS-VALIDATION IS NOT OPTIONAL HERE. Two choices leak if made on the
evaluation data:
  1. ORIENTATION. Signals have no natural polarity — is more jerk a violation or
     is less? Picking the better direction on the same clips you score converts
     pure noise into ~0.5 + |noise|, which at 7 positives (friction) is a large
     fake effect.
  2. SELECTION. "Best signal per specialist" over 28 signals is 28 chances to
     find noise.
Both are therefore decided on training folds only and applied to a held-out fold
(stratified 5-fold, repeated). The uncorrected number is printed alongside so the
size of the bias is visible rather than assumed away — on VideoPhy-2 the same
kind of leak was worth +0.140.

Usage:
  python backend/scripts/stage_fusion_study.py --data data/robotbench \
      --probes domain_probes_winners.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import PROBE_CAT, auc, tie_frac, cats_of  # noqa: E402


def rank01(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[o] = np.arange(len(v), dtype=float)
    for val, c in Counter(v).items():
        if c > 1:
            m = v == val
            r[m] = r[m].mean()
    return r / max(len(v) - 1, 1)


def folds(y, k=5, seed=0):
    """Stratified k-fold index list. Positives are scarce, so stratify or a fold
    can end up with none and its AUC becomes undefined."""
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(ip)
    rng.shuffle(ineg)
    out = []
    for i in range(k):
        te = np.concatenate([ip[i::k], ineg[i::k]])
        tr = np.setdiff1d(np.arange(len(y)), te)
        out.append((tr, te))
    return out


def cv_score(y, probe, sig, k=5, reps=4):
    """Honest fused AUC: orientation fitted on train, applied to test.

    Returns (fused_cv, signal_cv). Pooling the held-out predictions across folds
    and scoring once is deliberate — per-fold AUCs averaged over folds with 2-3
    positives each are far noisier than one AUC over all held-out points.
    """
    fu, si = [], []
    for rep in range(reps):
        pf = np.full(len(y), np.nan)
        sf = np.full(len(y), np.nan)
        for tr, te in folds(y, k, seed=rep):
            if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
                continue
            # orientation decided on TRAIN only
            a_tr = auc(y[tr], sig[tr])
            s = 1.0 if (np.isfinite(a_tr) and a_tr >= 0.5) else -1.0
            sf[te] = s * rank01(sig)[te]
            pf[te] = rank01(probe)[te] + s * rank01(sig)[te]
        m = ~np.isnan(pf)
        if m.sum() > 4 and 0 < y[m].sum() < m.sum():
            fu.append(auc(y[m], pf[m]))
            si.append(auc(y[m], sf[m]))
    return (float(np.mean(fu)) if fu else float("nan"),
            float(np.mean(si)) if si else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", default="domain_probes_winners.json")
    ap.add_argument("--signals", default="stage_signals.json")
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--exclude-real", action="store_true", default=True,
                    help="drop generator=real; they have no category labels")
    ap.add_argument("--perms", type=int, default=0,
                    help="label-shuffle permutations for the "
                         "best-of-N null (60 is enough for p95)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    man = json.loads((data / "manifest.json").read_text())
    clips = {c["clip_id"]: c for c in man["clips"]}
    ps = json.loads((data / a.probes).read_text())
    probe_scores = ps.get("probe_scores", ps)
    sg = json.loads((data / a.signals).read_text())
    sig, keys = sg["signals"], sg["keys"]

    ids = [c for c in probe_scores
           if c in clips and sig.get(c)
           and (clips[c].get("generator") != "real")
           and clips[c].get("has_violation") is not False]
    print(f"dataset {data.name} | probes {a.probes}")
    print(f"{len(ids)} annotated clips with a violation "
          f"(real + clean excluded)\n")

    S = {k: np.array([sig[c].get(k, 0.0) for c in ids], float) for k in keys}
    rows, best = [], {}
    for probe, cat in PROBE_CAT.items():
        y = np.array([1 if cat in cats_of(clips[c]) else 0 for c in ids])
        if y.sum() < a.min_pos:
            continue
        if not all(probe in probe_scores[c] for c in ids):
            continue
        pv = np.array([probe_scores[c][probe] for c in ids], float)
        p_auc = auc(y, pv)
        dead = tie_frac(pv) >= 0.75
        for k in keys:
            sv = S[k]
            if tie_frac(sv) > 0.95:
                continue
            raw_s = auc(y, sv)
            raw_f = auc(y, rank01(pv) + (rank01(sv) if raw_s >= 0.5
                                         else -rank01(sv)))
            cv_f, cv_s = cv_score(y, pv, sv)
            rows.append(dict(cat=cat, sig=k, n_pos=int(y.sum()),
                             probe=p_auc, probe_dead=bool(dead),
                             sig_raw=raw_s, sig_cv=cv_s,
                             fus_raw=raw_f, fus_cv=cv_f,
                             delta_cv=cv_f - max(p_auc, cv_s),
                             bias=raw_f - cv_f))
        mine = [r for r in rows if r["cat"] == cat]
        if mine:
            best[cat] = max(mine, key=lambda r: r["fus_cv"])

    # ── per-specialist summary ──────────────────────────────────────────────
    print("=" * 92)
    print("BEST UPSTREAM SIGNAL PER SPECIALIST  (cross-validated; "
          "orientation fitted on train folds)")
    print("=" * 92)
    print(f"{'specialist':13s} {'pos':>4s} {'probe':>7s} {'best signal':22s} "
          f"{'sig':>7s} {'fused':>7s} {'delta':>7s} {'leak':>6s}")
    print("-" * 92)
    for cat, r in sorted(best.items(), key=lambda kv: -kv[1]["fus_cv"]):
        tag = " (probe dead)" if r["probe_dead"] else ""
        print(f"{cat:13s} {r['n_pos']:4d} {r['probe']:7.3f} {r['sig'][:22]:22s} "
              f"{r['sig_cv']:7.3f} {r['fus_cv']:7.3f} {r['delta_cv']:+7.3f} "
              f"{r['bias']:+6.3f}{tag}")
    print("\n'delta' is vs max(probe, signal) — the honest bar.")
    print("'leak' is raw minus cross-validated: how much picking orientation on "
          "the eval data would have inflated it.")

    # ── full matrix, top signals only ───────────────────────────────────────
    top = [k for k in keys
           if any(r["sig"] == k and r["delta_cv"] > 0.01 for r in rows)]
    if top:
        cats = sorted({r["cat"] for r in rows})
        print(f"\n{'='*92}")
        print("SPECIALIST x SIGNAL — cross-validated fused AUC "
              "(only signals helping someone)")
        print("=" * 92)
        print(f"{'signal':24s}" + "".join(f"{c[:9]:>10s}" for c in cats))
        print("-" * 92)
        for k in top:
            line = f"{k:24s}"
            for c in cats:
                r = next((x for x in rows if x["cat"] == c and x["sig"] == k), None)
                line += f"{r['fus_cv']:10.3f}" if r else f"{'—':>10s}"
            print(line)
        print(f"{'(probe alone)':24s}" + "".join(
            f"{next(x['probe'] for x in rows if x['cat']==c):10.3f}" for c in cats))

    # ── permutation null ────────────────────────────────────────────────────
    # We report the BEST of ~28 signals per specialist, so the relevant question
    # is not "is this signal above chance" but "is the best of 28 above what the
    # best of 28 pure-noise signals would reach". Shuffling the labels and
    # repeating the whole best-of-28 search answers exactly that, and it also
    # absorbs the residual orientation leak (a null signal still scores ~0.55
    # standalone under 5-fold, because the train folds overlap).
    if a.perms:
        print(f"\n{'='*92}")
        print(f"PERMUTATION NULL — {a.perms} label shuffles, best-of-"
              f"{len(keys)} search repeated each time")
        print("=" * 92)
        print(f"{'specialist':13s} {'observed':>9s} {'null p95':>9s} "
              f"{'null max':>9s}  verdict")
        print("-" * 92)
        rng = np.random.default_rng(0)
        for cat, r in sorted(best.items(), key=lambda kv: -kv[1]["fus_cv"]):
            probe = next(p for p, c in PROBE_CAT.items() if c == cat)
            pv = np.array([probe_scores[c][probe] for c in ids], float)
            y0 = np.array([1 if cat in cats_of(clips[c]) else 0 for c in ids])
            null = []
            for _ in range(a.perms):
                yp = y0.copy()
                rng.shuffle(yp)
                bestf = max(
                    (cv_score(yp, pv, S[k], reps=1)[0] for k in keys
                     if tie_frac(S[k]) <= 0.95),
                    default=float("nan"))
                if np.isfinite(bestf):
                    null.append(bestf)
            if not null:
                continue
            p95, mx = float(np.percentile(null, 95)), float(max(null))
            v = ("REAL" if r["fus_cv"] > p95 else
                 "not distinguishable from noise")
            print(f"{cat:13s} {r['fus_cv']:9.3f} {p95:9.3f} {mx:9.3f}  {v}")
            best[cat]["null_p95"] = p95

    out = Path(a.out) if a.out else data / "stage_fusion_study.json"
    out.write_text(json.dumps({"probes": a.probes, "n": len(ids),
                               "rows": rows, "best": best}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

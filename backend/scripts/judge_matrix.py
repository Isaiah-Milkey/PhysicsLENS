"""
Is the best base model per specialist REAL, or is it selection noise?

The tempting move is to read a specialist x judge table, take each row's max,
and ship "gravity uses InternVL, fluid uses Qwen-32B". With 12-77 positives per
row and 5-8 judges, the row max is substantially the luckiest judge, and it does
not repeat. This tests the claim three ways, weakest to strongest:

  1. SPLIT-HALF        choose the judge on one half, score on the other. If
                       per-specialist choice beats always using one fixed judge,
                       the specialisation survives its own selection.
  2. STABILITY         how often does the same judge win across resamples? A
                       genuine preference wins most of the time; noise gives a
                       near-uniform spread over judges.
  3. CROSS-DATASET     does the judge chosen on VideoPhy-2 also win on
                       RobotBench? Nothing can leak across datasets, so agreement
                       here is the only evidence that cannot be explained by
                       overfitting one benchmark.

Only a claim that survives all three belongs in a config card.

Usage:
  python backend/scripts/judge_matrix.py --data data/robotbench
  python backend/scripts/judge_matrix.py --data data/robotbench --cross data/videophy1200
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, tie_frac, spread, cats_of  # noqa: E402
from ablation_grid import category_scores  # noqa: E402
from benchmark_all import rule_cats  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def judge_files(root: Path, battery="robot"):
    """condition-file -> judge name, for one probe battery and plain frames."""
    out = {}
    for f in sorted(root.glob(f"domain_probes_{battery}*.json")):
        stem = f.stem.replace(f"domain_probes_{battery}", "").strip("_")
        # skip ablations; we only want the judge axis
        # any ablation axis other than the judge itself. f4 was missing here,
        # so robot__qwen2.5-vl-7b__f4 became a phantom judge "qwen2.5-vl-7bf4"
        # and appeared as a second column for the same model.
        if any(t in stem for t in ("nocap", "shuffled", "inject", "cap",
                                   "f2", "f4", "f16")):
            continue
        judge = stem.replace("__", "") or "gemma4-31b-it"
        if judge in ("f4",):
            judge = "gemma4-31b-it"
        out[judge] = f
    return out


def vectors(root: Path, battery, min_pos=8):
    clips = {c["clip_id"]: c
             for c in json.loads((root / "manifest.json").read_text())["clips"]}
    cat_of = rule_cats(root, clips)
    files = judge_files(root, battery)
    data = {}
    for cat in ORDER:
        ai = [c for c in clips if clips[c].get("generator") != "real"]
        pos = [c for c in ai if cat in cat_of.get(c, set())
               and clips[c].get("has_violation") is not False
               and len(str(clips[c].get("violated_rules") or "")) > 4]
        neg = [c for c in ai if cat not in cat_of.get(c, set())
               and clips[c].get("has_violation") is not False
               and len(str(clips[c].get("violated_rules") or "")) > 4]
        if len(pos) < min_pos or len(neg) < 10:
            continue
        ids = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        cols = {}
        for j, f in files.items():
            ps = json.loads(f.read_text()).get("probe_scores") or {}
            sc = category_scores(ps, [c for c in ids if c in ps]).get(cat)
            if not sc or not all(c in sc for c in ids):
                continue
            v = np.array([sc[c] for c in ids], float)
            if tie_frac(v) >= 0.75 or spread(v) < 0.10:
                continue
            cols[j] = v
        if len(cols) >= 2:
            data[cat] = (y, cols)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--battery", default="robot")
    ap.add_argument("--cross", default=None)
    ap.add_argument("--splits", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    D = vectors(root, a.battery)
    if not D:
        sys.exit("no usable judge columns")
    judges = sorted({j for _, c in D.values() for j in c})

    print(f"{root.name} | battery={a.battery} | {len(judges)} judges\n")
    print("=" * 104)
    print("1. SPECIALIST x JUDGE  (in-sample AUC — the tempting table)")
    print("=" * 104)
    print(f"{'specialist':13s}{'pos':>5s}" + "".join(f"{j[:13]:>15s}" for j in judges))
    print("-" * 104)
    for cat in ORDER:
        if cat not in D:
            continue
        y, cols = D[cat]
        row = f"{cat:13s}{int(y.sum()):5d}"
        for j in judges:
            row += f"{auc(y, cols[j]):15.3f}" if j in cols else f"{'—':>15s}"
        print(row)

    # ── 2. split-half: per-specialist judge vs one fixed judge ──────────────
    rng = np.random.default_rng(0)
    per, fixed = {c: [] for c in D}, {j: {c: [] for c in D} for j in judges}
    winners = {c: Counter() for c in D}
    for _ in range(a.splits):
        for cat, (y, cols) in D.items():
            ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
            pa, na = rng.permutation(ip), rng.permutation(ineg)
            A = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
            B = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
            for tr, te in ((A, B), (B, A)):
                if y[te].sum() < 2 or (1 - y[te]).sum() < 2:
                    continue
                sc = {j: auc(y[tr], v[tr]) for j, v in cols.items()}
                sc = {k: v for k, v in sc.items() if np.isfinite(v)}
                if not sc:
                    continue
                w = max(sc, key=sc.get)
                winners[cat][w] += 1
                per[cat].append(auc(y[te], cols[w][te]))
                for j, v in cols.items():
                    fixed[j][cat].append(auc(y[te], v[te]))

    print(f"\n{'='*104}")
    print(f"2. HELD-OUT: per-specialist judge choice vs ONE fixed judge "
          f"({a.splits} split-halves)")
    print("=" * 104)
    print(f"{'specialist':13s}{'per-spec':>11s}" + "".join(f"{j[:13]:>15s}" for j in judges))
    print("-" * 104)
    for cat in ORDER:
        if cat not in D:
            continue
        row = f"{cat:13s}{np.mean(per[cat]):11.3f}"
        for j in judges:
            v = fixed[j][cat]
            row += f"{np.mean(v):15.3f}" if v else f"{'—':>15s}"
        print(row)
    # Judges cover different specialists (a dead probe drops that cell), so a
    # mean over each judge's own subset compares different tasks. Restrict every
    # mean to the specialists ALL judges cover; otherwise a judge that survives
    # only on the two easy specialists "wins" by never being tested on the hard
    # ones.
    common = [c for c in D if all(fixed[j][c] for j in judges)]
    if common:
        pm = np.mean([np.mean(per[c]) for c in common])
        fm = {j: np.mean([np.mean(fixed[j][c]) for c in common]) for j in judges}
        scope = f"over {len(common)} specialists covered by all judges: {common}"
    else:
        pm = np.mean([np.mean(per[c]) for c in D if per[c]])
        fm = {j: (np.mean([np.mean(fixed[j][c]) for c in D if fixed[j][c]])
                  if any(fixed[j][c] for c in D) else float("nan"))
              for j in judges}
        scope = "NO specialist is covered by all judges — means are NOT comparable"
    bestj = max(fm, key=lambda k: (fm[k] if np.isfinite(fm[k]) else -1))
    print("-" * 104)
    print(f"{'MEAN':13s}{pm:11.3f}" + "".join(f"{fm[j]:15.3f}" for j in judges))
    print(f"  ({scope})")
    print(f"{'coverage':13s}{'':11s}" + "".join(
        f"{sum(1 for c in D if fixed[j][c]):14d}/{len(D)}" for j in judges))
    print(f"\nper-specialist choice {pm:.3f}  vs  best single judge "
          f"({bestj}) {fm[bestj]:.3f}   ->  "
          + ("per-specialist WINS" if pm > fm[bestj] + 0.005 else
             "NO BENEFIT from per-specialist judges"))

    # ── 3. stability ────────────────────────────────────────────────────────
    print(f"\n{'='*104}")
    print("3. STABILITY — how often each judge wins its specialist across "
          "resamples")
    print("=" * 104)
    print(f"{'specialist':13s} {'modal judge':22s} {'win rate':>9s}  "
          f"{'uniform baseline':>17s}  verdict")
    print("-" * 104)
    for cat in ORDER:
        if cat not in winners or not winners[cat]:
            continue
        j, n = winners[cat].most_common(1)[0]
        tot = sum(winners[cat].values())
        k = len(D[cat][1])
        base = 1.0 / k
        rate = n / tot
        v = ("stable" if rate > 3 * base else
             "weak" if rate > 1.8 * base else "coin flip")
        print(f"{cat:13s} {j[:22]:22s} {rate:9.1%}  {base:17.1%}  {v}")

    # ── 4. cross-dataset agreement ──────────────────────────────────────────
    if a.cross:
        cr = Path(a.cross) if Path(a.cross).is_absolute() else ROOT / a.cross
        C = vectors(cr, a.battery)
        shared = [c for c in D if c in C]
        if shared:
            print(f"\n{'='*104}")
            print(f"4. CROSS-DATASET — best judge on {cr.name} vs on {root.name}")
            print("=" * 104)
            print(f"{'specialist':13s} {'best on '+cr.name[:12]:22s} "
                  f"{'best on '+root.name[:12]:22s} {'agree':>7s} "
                  f"{'transferred AUC':>16s}")
            print("-" * 104)
            agree = 0
            for cat in shared:
                yC, cC = C[cat]
                yD, cD = D[cat]
                sh = sorted(set(cC) & set(cD))
                if not sh:
                    continue
                bc = max(sh, key=lambda j: auc(yC, cC[j]))
                bd = max(sh, key=lambda j: auc(yD, cD[j]))
                ok = bc == bd
                agree += ok
                print(f"{cat:13s} {bc[:22]:22s} {bd[:22]:22s} "
                      f"{'YES' if ok else 'no':>7s} {auc(yD, cD[bc]):16.3f}")
            print(f"\nagreement: {agree}/{len(shared)} specialists pick the same "
                  f"judge on both datasets")

    (root / f"judge_matrix_{a.battery}.json").write_text(json.dumps(
        {"judges": judges,
         "per_specialist_heldout": {c: float(np.mean(per[c])) for c in D if per[c]},
         "fixed_judge_heldout": {j: float(v) for j, v in fm.items()},
         "modal_winner": {c: winners[c].most_common(1)[0][0] for c in winners
                          if winners[c]}}, indent=1))
    print(f"\n-> {root}/judge_matrix_{a.battery}.json")


if __name__ == "__main__":
    main()

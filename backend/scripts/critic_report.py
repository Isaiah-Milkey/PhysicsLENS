"""
Did the critic pass help? Per specialist, per critic style.

Reports four numbers per cell so the mechanism is visible, not just the outcome:

  probe        the specialist's own score (baseline)
  critic       the critic's score alone
  fused        rank-mean of the two
  corr         Spearman correlation between probe and critic

`corr` is the diagnostic that matters. A second pass can only add information if
it makes DIFFERENT mistakes. If probe and critic correlate at 0.9 the critic has
merely restated the probe in other words, and any AUC change is noise — which is
precisely what happened when we ensembled five judges (they agreed with each
other at 0.43 and with humans at 0.18, and the ensemble gained nothing).

Gains are checked against a paired bootstrap over clips rather than eyeballed,
because at 12-77 positives a +0.03 swing is well inside the noise.

Usage:
  python backend/scripts/critic_report.py --data data/robotbench \
      --probes domain_probes_robot__f4.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, cats_of, _avg_rank  # noqa: E402
from ablation_grid import category_scores  # noqa: E402
from benchmark_all import rule_cats  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def rank01(v):
    r = _avg_rank(np.asarray(v, float))
    return (r - 1) / max(len(r) - 1, 1)


def paired_boot(y, a_s, b_s, n=3000, seed=0):
    """CI on AUC(b) - AUC(a) resampling CLIPS, so the two scores stay paired."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        i = np.concatenate([rng.choice(ip, len(ip), True),
                            rng.choice(ineg, len(ineg), True)])
        d.append(auc(y[i], np.asarray(b_s)[i]) - auc(y[i], np.asarray(a_s)[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", default="domain_probes_robot__f4.json")
    ap.add_argument("--min-pos", type=int, default=8)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}
    cat_of = rule_cats(data, clips)
    ps = json.loads((data / a.probes).read_text())["probe_scores"]

    crits = {}
    for f in sorted(data.glob("critic_*.json")):
        d = json.loads(f.read_text())
        crits[d["style"]] = d["critic_scores"]
    if not crits:
        sys.exit("no critic_*.json found")

    print(f"{data.name} | probes {a.probes} | critic styles: "
          f"{', '.join(crits)}\n")
    print("=" * 108)
    print("CRITIC EFFECT PER SPECIALIST   (fused = rank-mean of probe + critic)")
    print("=" * 108)
    print(f"{'specialist':13s} {'pos':>4s} {'style':12s} {'probe':>7s} "
          f"{'critic':>7s} {'fused':>7s} {'gain':>7s} {'95% CI':>18s} "
          f"{'corr':>6s}")
    print("-" * 108)
    best = {}
    for cat in ORDER:
        ai = [c for c in clips if clips[c].get("generator") != "real"]
        pos = [c for c in ai if cat in cat_of.get(c, set())
               and clips[c].get("has_violation") is not False
               and len(str(clips[c].get("violated_rules") or "")) > 4]
        neg = [c for c in ai if cat not in cat_of.get(c, set())
               and clips[c].get("has_violation") is not False
               and len(str(clips[c].get("violated_rules") or "")) > 4]
        if len(pos) < a.min_pos or len(neg) < 10:
            continue
        ids = pos + neg
        sc = category_scores(ps, [c for c in ids if c in ps]).get(cat)
        if not sc or not all(c in sc for c in ids):
            continue
        y = np.array([1] * len(pos) + [0] * len(neg))
        pv = np.array([sc[c] for c in ids], float)
        p_auc = auc(y, pv)
        first = True
        for style, cs in crits.items():
            if not all(c in cs and cat in cs[c] for c in ids):
                continue
            cv = np.array([cs[c][cat] for c in ids], float)
            c_auc = auc(y, cv)
            fv = rank01(pv) + rank01(cv)
            f_auc = auc(y, fv)
            lo, hi = paired_boot(y, pv, fv)
            corr = float(np.corrcoef(_avg_rank(pv), _avg_rank(cv))[0, 1])
            sig = "" if lo <= 0 <= hi else ("  SIG+" if lo > 0 else "  SIG-")
            print(f"{cat if first else '':13s} {int(y.sum()) if first else '':>4} "
                  f"{style:12s} {p_auc:7.3f} {c_auc:7.3f} {f_auc:7.3f} "
                  f"{f_auc-p_auc:+7.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s} "
                  f"{corr:6.2f}{sig}")
            first = False
            k = (cat, style)
            best[f"{cat}/{style}"] = dict(probe=p_auc, critic=c_auc, fused=f_auc,
                                          gain=f_auc - p_auc, lo=lo, hi=hi,
                                          corr=corr, n_pos=int(y.sum()))
        print("-" * 108)

    rows = list(best.values())
    if rows:
        g = [r["gain"] for r in rows]
        sig = [k for k, r in best.items() if r["lo"] > 0]
        print(f"\nmean gain over {len(rows)} specialist-style cells: "
              f"{np.mean(g):+.3f}")
        print(f"significantly positive: {sig or 'none'}")
        print(f"mean probe-critic rank correlation: "
              f"{np.mean([r['corr'] for r in rows]):.2f}  "
              f"(high = the critic is restating the probe, not adding to it)")
    (data / "critic_report.json").write_text(json.dumps(best, indent=1))
    print(f"\n-> {data}/critic_report.json")


if __name__ == "__main__":
    main()

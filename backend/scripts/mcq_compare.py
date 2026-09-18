"""
Compare every MCQ run — all models, all datasets — in one table.

Reports the three numbers that behave differently, because collapsing them into
one "accuracy" hides the finding that motivated this experiment: forced choice
improves RANKING while its single ANSWER is worse than a constant guess.

  attribution AUC   per specialist, from P(option). Continuous, threshold-free.
  argmax accuracy   does the chosen option match a true label? Always printed
                    against the majority-class baseline, never against 1/9 —
                    with a skewed label distribution "always say collision"
                    scores 61%, so 1/9 = 11% is a meaningless bar.
  position corr     correlation(printed option position, P(option)). Options are
                    permuted per clip, so anything far from 0 means the model is
                    answering by list order and the AUCs are an artefact.

Usage:
  python backend/scripts/mcq_compare.py --datasets data/robotbench,data/videophy1200
"""
import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import auc, boot_ci, tie_frac, spread  # noqa: E402
from benchmark_all import rule_cats  # noqa: E402

ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def analyse(data: Path, f: Path, min_pos=8):
    raw = json.loads(f.read_text())
    M = raw["mcq"]
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}
    CAT = rule_cats(data, clips)
    ai = [c for c in M if c in clips and clips[c].get("generator") != "real"]
    viol = [c for c in ai if len(str(clips[c].get("violated_rules") or "")) > 4]
    clean = [c for c in ai if clips[c].get("has_violation") is False]
    real = [c for c in M if c in clips and clips[c].get("generator") == "real"]
    if len(viol) < 20:
        return None

    per = {}
    for cat in ORDER:
        pos = [c for c in viol if cat in CAT.get(c, set())]
        neg = [c for c in viol if cat not in CAT.get(c, set())]
        if len(pos) < min_pos or len(neg) < 10:
            continue
        ids = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        v = np.array([M[c]["probs"][cat] for c in ids], float)
        per[cat] = dict(auc=auc(y, v), n_pos=len(pos),
                        tie=tie_frac(v), idr=spread(v))

    hit = sum(1 for c in viol
              if max(M[c]["probs"], key=M[c]["probs"].get) in CAT.get(c, set()))
    lbl = Counter(x for c in viol for x in CAT.get(c, set()))
    top = lbl.most_common(1)[0][0] if lbl else None
    chance = (sum(1 for c in viol if top in CAT.get(c, set())) / len(viol)
              if top else float("nan"))
    picks = Counter(max(M[c]["probs"], key=M[c]["probs"].get) for c in viol)

    none_auc = {}
    for tag, negs in (("clean", clean), ("real", real)):
        if len(negs) >= 5:
            ids = viol + negs
            y = np.array([1] * len(viol) + [0] * len(negs))
            v = np.array([1.0 - M[c]["probs"]["none"] for c in ids], float)
            none_auc[tag] = auc(y, v)

    letters = string.ascii_uppercase
    pp, lp = [], []
    for c in M:
        for L, name in M[c]["map"].items():
            pp.append(letters.index(L))
            lp.append(M[c]["probs"][name])
    pos_r = float(np.corrcoef(pp, lp)[0, 1]) if len(pp) > 10 else float("nan")

    return dict(model=raw["model"], frames=raw.get("frames"), n=len(M),
                n_viol=len(viol), per=per, argmax=hit / len(viol),
                chance=chance, top_label=top, picks=dict(picks.most_common(4)),
                none=none_auc, pos_corr=pos_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="data/robotbench,data/videophy1200")
    a = ap.parse_args()
    out = {}
    for d in a.datasets.split(","):
        d = d.strip()
        if not d:
            continue
        data = Path(d) if Path(d).is_absolute() else ROOT / d
        if not (data / "manifest.json").exists():
            continue
        runs = []
        for f in sorted(data.glob("mcq_*.json")):
            if f.name == "mcq_report.json":
                continue
            r = analyse(data, f)
            if r:
                runs.append(r)
        if runs:
            out[data.name] = runs

    for ds, runs in out.items():
        cats = [c for c in ORDER if any(c in r["per"] for r in runs)]
        print("=" * 112)
        print(f"{ds.upper()}   ({runs[0]['n_viol']} annotated clips with a violation)")
        print("=" * 112)
        print(f"{'model':22s}{'f':>3s}" + "".join(f"{c[:9]:>11s}" for c in cats)
              + f"{'mean':>9s}")
        print("-" * 112)
        for r in sorted(runs, key=lambda r: -np.mean(
                [v["auc"] for v in r["per"].values()] or [0])):
            row = f"{r['model'][:22]:22s}{r['frames']:>3d}"
            v = []
            for c in cats:
                e = r["per"].get(c)
                if not e:
                    row += f"{'—':>11s}"
                    continue
                mk = "!" if e["tie"] >= 0.75 else "~" if e["idr"] < 0.10 else ""
                row += f"{e['auc']:>10.3f}{mk}"
                v.append(e["auc"])
            row += f"{np.mean(v):9.3f}" if v else f"{'—':>9s}"
            print(row)
        print(f"\n{'model':22s} {'argmax':>8s} {'majority':>9s} {'vs base':>8s} "
              f"{'none|clean':>11s} {'none|real':>10s} {'pos corr':>9s}  top picks")
        print("-" * 112)
        for r in sorted(runs, key=lambda r: -r["argmax"]):
            nc = r["none"].get("clean", float("nan"))
            nr = r["none"].get("real", float("nan"))
            print(f"{r['model'][:22]:22s} {r['argmax']:8.1%} {r['chance']:9.1%} "
                  f"{r['argmax']-r['chance']:+8.1%} {nc:11.3f} {nr:10.3f} "
                  f"{r['pos_corr']:+9.3f}  {r['picks']}")
        print()

    p = ROOT / "data" / "mcq_compare.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {p}")


if __name__ == "__main__":
    main()

"""
Compare every probe condition side by side, per specialist.

Reads all domain_probes_*.json in a staged dataset and scores each one the same
way, so wording, caption, frame order, judge model and injected evidence become
directly comparable columns rather than separate runs to remember.

THREE TESTS, NOT ONE. This dataset supports a distinction VideoPhy-2 never could,
because it ships the real robot demonstrations the clips were generated from:

  attribution   pos = this violation      neg = a DIFFERENT violation (AI only)
                Hardest, and the only one VideoPhy-2 could run. Immune to
                "spot the AI" shortcuts: every clip on both sides is generated.

  detect_real   pos = this violation      neg = the REAL demonstrations
                Easiest, and the most confounded: real and generated video
                differ in resolution, codec, motion smoothness and lighting, so
                a high score here can be an AI detector rather than a physics
                detector. Reported because it is what "accuracy" usually means
                to a reader, and hiding it invites someone else to quote it
                without the caveat.

  detect_clean  pos = this violation      neg = AI clips annotated as clean
                The honest detection test — both sides generated, only the
                physics differs. Small (~20 negatives) but not confounded.

A specialist that scores well on detect_real and poorly on detect_clean has
learned to recognise generated video, not broken physics. Printing them adjacent
makes that failure visible instead of flattering.

Usage:
  python backend/scripts/ablation_grid.py --data data/robotbench
  python backend/scripts/ablation_grid.py --data data/robotbench --test attribution
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from specialist_accuracy import (PROBE_CAT, auc, boot_ci, tie_frac,  # noqa: E402
                                 spread, cats_of)

ORDER = ["gravity", "collision", "deformation", "momentum",
         "causality", "permanence", "fluid", "friction"]


def label(fn: str) -> str:
    """domain_probes_winners__nocap.json -> winners/nocap"""
    s = re.sub(r"^domain_probes_", "", fn).replace(".json", "")
    return s.replace("__", "/")


def _subprobe_map():
    """sub-probe key -> specialist, read from the battery definitions themselves.

    Full-battery runs emit g_no_accel, fl_shape, cz_uncaused_motion … rather than
    the eight w_* keys, so PROBE_CAT alone cannot score them. Deriving the map
    from BATTERIES means a renamed or added probe is picked up automatically
    instead of silently scoring nothing. v2 batteries fold into their base
    category (deformation_v2 -> deformation).
    """
    from domain_probes import BATTERIES
    m = {}
    for cat, probes in BATTERIES.items():
        if cat in ("winners", "robot"):
            continue
        base = cat.replace("_v2", "")
        for k, _ in probes:
            m[k] = base
    return m


def category_scores(ps, clip_ids):
    """clip -> {specialist: score}, fusing multi-probe batteries by rank-mean.

    A battery gives a category several weak sub-checks. Rank-mean is the right
    combiner here for the same reason it was for the channels: it needs no
    weights fitted to the labels, so nothing leaks. Sub-probes that collapsed to
    one value are dropped first — a constant column shifts every clip equally
    and only dilutes the informative ones.
    """
    sub = _subprobe_map()
    keys = sorted({k for c in clip_ids for k in ps.get(c, {})})
    direct = [k for k in keys if k in PROBE_CAT]
    if direct:                       # winners / robot style: one probe per cat
        return {PROBE_CAT[k]: {c: ps[c][k] for c in clip_ids if k in ps.get(c, {})}
                for k in direct}
    bycat = {}
    for k in keys:
        cat = sub.get(k)
        if not cat:
            continue
        vals = [ps[c][k] for c in clip_ids if k in ps.get(c, {})]
        if len(vals) < 20 or tie_frac(vals) >= 0.95:
            continue
        bycat.setdefault(cat, []).append(k)
    out = {}
    for cat, ks in bycat.items():
        acc, n = {}, {}
        for k in ks:
            ids = [c for c in clip_ids if k in ps.get(c, {})]
            v = np.array([ps[c][k] for c in ids], float)
            o = v.argsort().argsort().astype(float) / max(len(v) - 1, 1)
            for c, r in zip(ids, o):
                acc[c] = acc.get(c, 0.0) + r
                n[c] = n.get(c, 0) + 1
        out[cat] = {c: acc[c] / n[c] for c in acc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", default="attribution",
                    choices=["attribution", "detect_real", "detect_clean"])
    ap.add_argument("--min-pos", type=int, default=8)
    ap.add_argument("--ci", action="store_true", help="bootstrap CIs (slower)")
    a = ap.parse_args()

    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    clips = {c["clip_id"]: c
             for c in json.loads((data / "manifest.json").read_text())["clips"]}

    files = sorted(data.glob("domain_probes_*.json"))
    if not files:
        sys.exit(f"no domain_probes_*.json in {data}")

    grid, meta = {}, {}
    for f in files:
        try:
            raw = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        ps = raw.get("probe_scores")
        if not ps:
            continue
        name = label(f.name)
        meta[name] = dict(model=raw.get("model", "?"), n=len(ps))
        grid[name] = {}
        allids = [c for c in ps if c in clips]
        catsc = category_scores(ps, allids)
        for cat, scoremap in catsc.items():
            ids = [c for c in allids if c in scoremap]
            if not ids:
                continue
            real = [c for c in ids if clips[c].get("generator") == "real"]
            ai = [c for c in ids if clips[c].get("generator") != "real"]
            pos = [c for c in ai if cat in cats_of(clips[c])
                   and clips[c].get("has_violation") is not False]
            if a.test == "attribution":
                neg = [c for c in ai if cat not in cats_of(clips[c])
                       and clips[c].get("has_violation") is not False]
            elif a.test == "detect_real":
                neg = real
            else:
                neg = [c for c in ai if clips[c].get("has_violation") is False]
            if len(pos) < a.min_pos or len(neg) < 5:
                continue
            y = np.array([1] * len(pos) + [0] * len(neg))
            s = np.array([scoremap[c] for c in pos] + [scoremap[c] for c in neg])
            e = dict(auc=auc(y, s), n_pos=len(pos), n_neg=len(neg),
                     tie=tie_frac(s), idr=spread(s))
            if a.ci:
                e["lo"], e["hi"] = boot_ci(y, s, n_boot=2000)
            grid[name][cat] = e

    cats = [c for c in ORDER if any(c in v for v in grid.values())]
    names = sorted(grid, key=lambda n: -np.mean(
        [grid[n][c]["auc"] for c in cats if c in grid[n]] or [0]))

    print(f"dataset {data.name} | test = {a.test.upper()}")
    for c in cats:
        e = next((grid[n][c] for n in names if c in grid[n]), None)
        if e:
            print(f"   {c:12s} {e['n_pos']:3d} pos vs {e['n_neg']:3d} neg")
    print()
    w = 11
    print(f"{'condition':22s}" + "".join(f"{c[:9]:>{w}s}" for c in cats)
          + f"{'mean':>{w}s}")
    print("-" * (22 + w * (len(cats) + 1)))
    for n in names:
        line = f"{n[:22]:22s}"
        vals = []
        for c in cats:
            e = grid[n].get(c)
            if not e:
                line += f"{'—':>{w}s}"
                continue
            # ! = collapsed onto one value; ~ = distinct but almost no spread
            mark = ("!" if e["tie"] >= 0.75 else
                    "~" if e["idr"] < 0.10 else " ")
            line += f"{e['auc']:>{w-1}.3f}{mark}"
            vals.append(e["auc"])
        line += f"{np.mean(vals):>{w}.3f}" if vals else f"{'—':>{w}s}"
        print(line)
    print("\n! = dead: >=75% of clips share one score; the AUC is tie-breaking "
          "noise.\n~ = flat: values distinct but interdecile range <0.10, so the "
          "probe barely commits.")

    # best condition per specialist, ignoring dead probes
    print(f"\n{'='*76}\nBEST CONDITION PER SPECIALIST ({a.test})\n{'='*76}")
    print(f"{'specialist':13s} {'best condition':26s} {'AUC':>7s}  "
          f"{'vs winners/base':>16s}")
    print("-" * 76)
    best = {}
    for c in cats:
        cand = [(n, grid[n][c]) for n in names
                if c in grid[n] and grid[n][c]["tie"] < 0.75
                and grid[n][c]["idr"] >= 0.10]
        if not cand:
            cand = [(n, grid[n][c]) for n in names if c in grid[n]]
        if not cand:
            continue
        n, e = max(cand, key=lambda kv: kv[1]["auc"])
        base = grid.get("winners", {}).get(c, {}).get("auc")
        d = f"{e['auc']-base:+.3f}" if base is not None else "—"
        print(f"{c:13s} {n[:26]:26s} {e['auc']:7.3f}  {d:>16s}")
        best[c] = dict(condition=n, **{k: v for k, v in e.items()})

    out = data / f"ablation_grid_{a.test}.json"
    out.write_text(json.dumps({"test": a.test, "meta": meta,
                               "grid": grid, "best": best}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

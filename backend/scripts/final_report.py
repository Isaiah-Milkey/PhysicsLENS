"""
Assemble every result on a staged dataset into one markdown report.

Reads whatever the runs produced — ablation grids, frozen configs, the Stage-1/2
fusion studies — and writes the tables. Nothing is computed here that is not
already computed and checked by the scripts that own it; this only formats.

The report deliberately leads with held-out numbers rather than best-of-N
numbers, and prints the gap between them, because the whole point of the
split-half machinery is that a best-of-40 search flatters itself and the
difference belongs in front of the reader, not in a footnote.

Usage:
  python backend/scripts/final_report.py --data data/robotbench \
      --out eval_reports/ROBOTBENCH_EVAL.md
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["fluid", "causality", "gravity", "momentum", "deformation",
         "collision", "permanence", "friction"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    out = Path(a.out) if a.out else ROOT / "eval_reports" / f"{data.name.upper()}_EVAL.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    man = json.loads((data / "manifest.json").read_text())
    clips = man["clips"]
    gens = Counter(c["generator"] for c in clips)
    cats = Counter(x for c in clips for x in set(c.get("categories") or []))
    ai = [c for c in clips if c["generator"] != "real"]

    L = []
    A = L.append
    A(f"# Specialist evaluation — {data.name}\n")
    A(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC._\n")

    # ── dataset ─────────────────────────────────────────────────────────────
    A("## 1. Dataset\n")
    A(f"{len(clips)} clips, {len(ai)} AI-generated and annotated, "
      f"{gens.get('real', 0)} real robot demonstrations.\n")
    A("| generator | clips | mean rating (1-4) |")
    A("|---|---|---|")
    for g, n in gens.most_common():
        r = [c["pc"] for c in clips if c["generator"] == g]
        A(f"| {g} | {n} | {np.mean(r):.2f} |")
    A("")
    A("Every AI clip was generated from a start frame of its real counterpart, "
      "so a real/AI pair shares scene, objects, camera and task. That is what "
      "makes the clean-negative test below possible at all — on VideoPhy-2 "
      "every clip was AI-generated and 'clean' did not exist.\n")
    A("| specialist | positives |")
    A("|---|---|")
    for c in ORDER:
        if cats.get(c):
            A(f"| {c} | {cats[c]} |")
    A("")

    # ── headline table ──────────────────────────────────────────────────────
    A("## 2. Final per-specialist table\n")
    A("`held-out` is the number to quote: the frozen config scored on clips it "
      "was **not** selected against, via repeated stratified split-half. "
      "`AUC` is what a best-of-N search reports about itself; `bias` is the "
      "difference.\n")
    for test, title, note in [
        ("attribution", "Attribution — this violation vs a DIFFERENT violation",
         "Hardest test, and immune to 'spot the AI' shortcuts: every clip on "
         "both sides is generated. 0.5 = cannot tell which kind of broken."),
        ("detect_clean", "Detection — this violation vs CLEAN AI clips",
         "Both sides generated, only the physics differs. Few negatives, but "
         "not confounded."),
        ("detect_real", "Detection — this violation vs REAL demonstrations",
         "Easiest and most confounded: real and generated video differ in "
         "codec, resolution and motion smoothness, so a high score here can be "
         "an AI detector rather than a physics detector.")]:
        fz = load(data / f"frozen_config_{test}.json")
        if not fz:
            continue
        A(f"### {title}\n")
        A(f"_{note}_\n")
        A("| specialist | pos | conds | frozen config | AUC [95% CI] | "
          "**held-out** | bias |")
        A("|---|---|---|---|---|---|---|")
        hv, bv = [], []
        for c in ORDER:
            r = fz.get(c)
            if not r:
                continue
            ci = f"{r['frozen_auc']:.3f} [{r['frozen_lo']:.2f}, {r['frozen_hi']:.2f}]"
            A(f"| {c} | {r['n_pos']} | {r['n_conditions']} | `{r['frozen']}` | "
              f"{ci} | **{f3(r['heldout_mean'])}** | {r['bias']:+.3f} |")
            hv.append(r["heldout_mean"])
            bv.append(r["bias"])
        if hv:
            A(f"| **mean** | | | | | **{np.mean(hv):.3f}** | {np.mean(bv):+.3f} |")
        A("")

    # ── what each specialist needs upstream ─────────────────────────────────
    fus = sorted(data.glob("fusion__*.json")) + [data / "stage_fusion_study.json"]
    per = {}
    for p in fus:
        d = load(p)
        if not d:
            continue
        for cat, r in (d.get("best") or {}).items():
            per.setdefault(cat, []).append(dict(r, source=Path(p).stem))
    if per:
        A("## 3. What each specialist needs from Stage 1 / Stage 2\n")
        A("Cross-validated: signal orientation is fitted on training folds only. "
          "`null p95` repeats the whole best-of-28-signals search on shuffled "
          "labels — the bar a selected signal must clear, since picking the best "
          "of 28 reaches ~0.65 on noise alone.\n")
        A("**Reported as a pass rate, not a best case.** The study was run "
          "against every candidate probe configuration, so quoting the best "
          "fused AUC would add a second selection layer on top of the one the "
          "null already corrects for — and that layer is large: taking the max "
          "makes all six specialists look significant, while the median makes "
          "only two. A signal that helps under one probe configuration and not "
          "the other nine is a property of that configuration, not a "
          "requirement of the specialist.\n")
        A("| specialist | best signal (modal) | median fused | null p95 | "
          "configs passing | verdict |")
        A("|---|---|---|---|---|---|")
        for c in ORDER:
            v = per.get(c)
            if not v:
                continue
            fv = [r["fus_cv"] for r in v]
            nl = [r["null_p95"] for r in v if r.get("null_p95") is not None]
            npass = sum(1 for r in v
                        if r.get("null_p95") is not None
                        and r["fus_cv"] > r["null_p95"])
            sig = Counter(r["sig"] for r in v).most_common(1)[0][0]
            rate = npass / len(v)
            verdict = ("**REAL**" if rate >= 0.5 else
                       "weak / config-dependent" if rate >= 0.25 else
                       "not distinguishable from noise")
            A(f"| {c} | `{sig}` | {np.median(fv):.3f} | "
              f"{(np.mean(nl) if nl else float('nan')):.3f} | "
              f"{npass}/{len(v)} | {verdict} |")
        A("")

    # ── full ablation grid ──────────────────────────────────────────────────
    A("## 4. Complete ablation grid\n")
    A("Every condition on every specialist. `!` = dead (>=75% of clips share "
      "one score, so the AUC is tie-breaking noise). `~` = flat (values "
      "distinct but interdecile range <0.10, so the probe barely commits). "
      "Marked cells are excluded from config selection.\n")
    for test in ("attribution", "detect_clean", "detect_real"):
        g = load(data / f"ablation_grid_{test}.json")
        if not g:
            continue
        grid = g["grid"]
        cs = [c for c in ORDER if any(c in v for v in grid.values())]
        names = sorted(grid, key=lambda n: -np.mean(
            [grid[n][c]["auc"] for c in cs if c in grid[n]] or [0]))
        A(f"### {test}\n")
        A("| condition | " + " | ".join(cs) + " | mean |")
        A("|" + "---|" * (len(cs) + 2))
        for n in names:
            cells, vals = [], []
            for c in cs:
                e = grid[n].get(c)
                if not e:
                    cells.append("—")
                    continue
                m = ("!" if e["tie"] >= 0.75 else "~" if e["idr"] < 0.10 else "")
                cells.append(f"{e['auc']:.3f}{m}")
                vals.append(e["auc"])
            mean = f"{np.mean(vals):.3f}" if vals else "—"
            A(f"| `{n}` | " + " | ".join(cells) + f" | {mean} |")
        A("")

    out.write_text("\n".join(L))
    print(f"-> {out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()

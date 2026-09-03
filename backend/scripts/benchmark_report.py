"""
Regenerate every table in the PhysicsLENS benchmark report from saved eval output.

All analysis here is a pure re-scoring of `summary.json` files already on disk —
no GPU, no API, no pipeline re-runs. That is deliberate: the expensive part is
running the pipeline, so every table below can be recomputed (or a new reduction
added) for free, and next week's re-run only needs the pipeline, not the analysis.

Usage:
  python backend/scripts/benchmark_report.py                       # all sections
  python backend/scripts/benchmark_report.py --section human       # one section

Sections:
  human        human-agreement + candidate reductions + ablations   (needs --labels)
  extremes     AUC restricted to the human-score tails
  confound     is severity tracking resolution/duration, not physics?
  separation   AI-vs-real per pipeline (two run dirs, identical config)
  cost         runtime share vs separation earned
  determinism  repeat-run variance -> the noise floor for every number above
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vlm_rapidata_eval import spearman, median_split_auc  # noqa: E402

# Pipelines whose severity fires on ≥80% of clips in both the AI and real sets —
# measured, not assumed (see the `human` section's saturation table).
SATURATED = {"s1_temporal", "s1_vlm", "s3_fluid", "s3_deformation", "s4_report"}
SILENT = {"s2_event_localizer"}


# ── loading ───────────────────────────────────────────────────────────────────

def max_severity(pipeline_entry: dict):
    vals = [s["value"] for s in pipeline_entry.get("severities", [])
            if isinstance(s.get("value"), (int, float))]
    return max(vals) if vals else None


def load_run(run_dir: Path, prefix: str = None) -> dict:
    """{clip_key: {pipeline_id: max_severity}} for every complete clip."""
    out = {}
    for spath in sorted(run_dir.glob("*/summary.json")):
        key = spath.parent.name
        if prefix and not key.startswith(prefix):
            continue
        out[key] = {pid: max_severity(p)
                    for pid, p in json.loads(spath.read_text()).get("pipelines", {}).items()}
    return out


def load_wall(run_dir: Path) -> dict:
    """{pipeline_id: [wall_s per clip]}."""
    out = {}
    for spath in sorted(run_dir.glob("*/summary.json")):
        for pid, p in json.loads(spath.read_text()).get("pipelines", {}).items():
            out.setdefault(pid, []).append(p.get("wall_s", 0) or 0)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def boot_ci(xs, ys, n=2000, seed=0):
    """Percentile bootstrap CI for Spearman. Deterministic given `seed` — the
    analysis must not itself be a source of run-to-run drift."""
    rnd = random.Random(seed)
    n_obs, out = len(xs), []
    for _ in range(n):
        idx = [rnd.randrange(n_obs) for _ in range(n_obs)]
        try:
            r = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        except Exception:  # noqa: BLE001  degenerate resample
            continue
        # A resample drawn from a near-constant score has zero variance, so
        # Spearman is undefined and comes back NaN rather than raising. Sorting
        # a list containing NaN silently yields a garbage interval (observed:
        # lo > hi), so drop them here and report how many survived.
        if r == r:
            out.append(r)
    if len(out) < 0.5 * n:      # majority degenerate -> the score cannot rank
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def reductions(run: dict, keys: list) -> dict:
    """Every candidate way to collapse 17 severities into one number."""
    def vals(k, pool):
        return [run[k][p] for p in pool if run[k].get(p) is not None]

    allp = sorted({p for r in run.values() for p in r})
    healthy = [p for p in allp if p not in SATURATED and p not in SILENT]

    # Rank fusion: average each pipeline's rank across clips. Robust to a
    # saturated member (a constant contributes a constant rank), unlike max.
    ranks = {}
    for p in healthy:
        pairs = [(k, run[k].get(p)) for k in keys if run[k].get(p) is not None]
        order = sorted(pairs, key=lambda kv: kv[1])
        for i, (k, _) in enumerate(order):
            ranks.setdefault(k, []).append(i / max(1, len(order) - 1))

    return {
        "max_all (headline)":       [max(vals(k, allp)) for k in keys],
        "max_healthy":              [max(vals(k, healthy)) for k in keys],
        "mean_healthy":             [mean(vals(k, healthy)) for k in keys],
        "mean_all":                 [mean(vals(k, allp)) for k in keys],
        "rank_fusion_healthy":      [mean(ranks.get(k, [])) for k in keys],
        "n_flagged_healthy":        [float(sum(1 for v in vals(k, healthy) if v >= 50)) for k in keys],
        "s3_collision alone":       [run[k].get("s3_collision") for k in keys],
        "s3_momentum alone":        [run[k].get("s3_momentum") for k in keys],
        "s1_optical_flow alone":    [run[k].get("s1_optical_flow") for k in keys],
    }


def score_table(cands: dict, human: list, title: str):
    print(f"\n  {title}")
    print(f"    {'reduction':30s}{'rho':>8}{'95% CI':>18}{'AUC':>8}")
    print("    " + "-" * 62)
    rows = {}
    for nm, xs in cands.items():
        pairs = [(x, h) for x, h in zip(xs, human) if x is not None]
        if len(pairs) < 8:
            print(f"    {nm:30s}   too few clips")
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        if len(set(a)) < 2:
            print(f"    {nm:30s}   CONSTANT — cannot rank (all {a[0]:.0f})")
            rows[nm] = {"constant": True, "value": a[0]}
            continue
        rho, auc = spearman(a, b), median_split_auc(a, b)
        lo, hi = boot_ci(a, b)
        star = " *" if lo is not None and (lo > 0 or hi < 0) else ""
        ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else "-"
        print(f"    {nm:30s}{rho:+8.3f}{ci:>18}{auc:8.3f}{star}")
        rows[nm] = {"n": len(pairs), "spearman": round(rho, 3),
                    "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                    "auc": round(auc, 3)}
    print("    " + "-" * 62)
    print("    * = 95% CI excludes zero.  Multiple reductions tested — post-hoc, treat with suspicion.")
    return rows


# ── sections ──────────────────────────────────────────────────────────────────

def sec_human(run_dir: Path, labels_path: Path):
    lab = json.loads(labels_path.read_text())["clips"]
    run = {k: v for k, v in load_run(run_dir).items() if k in lab}
    keys = sorted(run)
    human = [float(lab[k]["human_score"]) for k in keys]
    print(f"\n=== HUMAN AGREEMENT — n={len(keys)} clips ===")
    print("  label: higher = humans found it MORE implausible; severity is also")
    print("  higher = worse, so a POSITIVE correlation is expected.")

    print("\n  Saturation (why a max-based aggregate cannot rank):")
    print(f"    {'pipeline':26s}{'mean':>7}{'sd':>7}{'>=50':>7}{'distinct':>10}")
    sat = {}
    for pid in sorted({p for r in run.values() for p in r}):
        v = [r[pid] for r in run.values() if r.get(pid) is not None]
        if not v:
            print(f"    {pid:26s}{'— emits no severity —':>31}")
            continue
        sat[pid] = {"mean": round(st.mean(v), 1), "sd": round(st.pstdev(v), 1),
                    "frac_ge50": round(sum(1 for x in v if x >= 50) / len(v), 3),
                    "distinct": len(set(v))}
        print(f"    {pid:26s}{st.mean(v):7.1f}{st.pstdev(v):7.1f}"
              f"{sum(1 for x in v if x >= 50) / len(v):6.0%}{len(set(v)):10d}")

    rows = score_table(reductions(run, keys), human, "Candidate reductions vs human score:")

    print("\n  Per-pipeline severity vs human score (with CI):")
    print(f"    {'pipeline':26s}{'rho':>8}{'95% CI':>18}")
    per = {}
    for pid in sorted({p for r in run.values() for p in r}):
        pairs = [(run[k][pid], h) for k, h in zip(keys, human) if run[k].get(pid) is not None]
        if len(pairs) < 8 or len({p[0] for p in pairs}) < 2:
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        rho = spearman(a, b)
        lo, hi = boot_ci(a, b)
        per[pid] = {"spearman": round(rho, 3),
                    "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None}
        print(f"    {pid:26s}{rho:+8.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}")
    return {"saturation": sat, "reductions": rows, "per_pipeline": per,
            "n": len(keys)}


def sec_extremes(run_dir: Path, labels_path: Path, frac=0.25):
    """The staged sample is middle-heavy; a median split therefore asks the tool
    to separate clips humans themselves barely distinguished. Restricting to the
    tails is the fairer test of whether ANY signal exists."""
    lab = json.loads(labels_path.read_text())["clips"]
    run = {k: v for k, v in load_run(run_dir).items() if k in lab}
    keys = sorted(run, key=lambda k: float(lab[k]["human_score"]))
    n = max(3, int(len(keys) * frac))
    low, high = keys[:n], keys[-n:]
    print(f"\n=== TAILS ONLY — bottom {n} vs top {n} by human score ===")
    print(f"  bottom mean human={mean([float(lab[k]['human_score']) for k in low]):.3f}  "
          f"top mean human={mean([float(lab[k]['human_score']) for k in high]):.3f}")
    sub = low + high
    human = [float(lab[k]["human_score"]) for k in sub]
    return score_table(reductions(run, sub), human,
                       "Candidate reductions, tails only:")


def sec_confound(run_dir: Path, labels_path: Path, videos_dir: Path):
    """If severity tracks resolution or duration rather than physics, that is a
    mechanism finding: s1_temporal's threshold is in absolute px/s²."""
    import cv2
    lab = json.loads(labels_path.read_text())["clips"]
    run = {k: v for k, v in load_run(run_dir).items() if k in lab}
    keys = sorted(run)
    props = {}
    for k in keys:
        f = videos_dir / "ai" / lab[k]["file"]
        if not f.exists():
            continue
        c = cv2.VideoCapture(str(f))
        n = c.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = c.get(cv2.CAP_PROP_FPS) or 30.0
        props[k] = {"pixels": c.get(cv2.CAP_PROP_FRAME_WIDTH) * c.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "duration": n / fps if fps else 0.0}
        c.release()
    print(f"\n=== CONFOUNDS — severity vs video properties (n={len(props)}) ===")
    ks = [k for k in keys if k in props]
    for field in ("pixels", "duration"):
        uniq = {props[k][field] for k in ks}
        if len(uniq) < 2:
            print(f"  {field}: CONSTANT across the set ({uniq.pop():.0f}) — this predictor")
            print(f"    has zero variance, so the test is inapplicable, not failed. It does")
            print(f"    mean no severity variance here can be blamed on {field}.")
    if all(len({props[k][f] for k in ks}) < 2 for f in ("pixels", "duration")):
        return {"inapplicable": "all clips share one resolution/duration/fps"}
    print("  A detector tracking resolution/duration instead of physics shows up here.")
    print(f"    {'pipeline':26s}{'rho vs pixels':>15}{'rho vs duration':>17}")
    out = {}
    for pid in sorted({p for r in run.values() for p in r}):
        pairs = [(run[k][pid], props[k]) for k in ks if run[k].get(pid) is not None]
        if len(pairs) < 8 or len({p[0] for p in pairs}) < 2:
            continue
        sev = [p[0] for p in pairs]
        rp = spearman(sev, [p[1]["pixels"] for p in pairs])
        rd = spearman(sev, [p[1]["duration"] for p in pairs])
        out[pid] = {"rho_pixels": round(rp, 3), "rho_duration": round(rd, 3)}
        flag = "  <-- resolution-driven" if abs(rp) >= 0.4 else ""
        print(f"    {pid:26s}{rp:+15.3f}{rd:+17.3f}{flag}")
    return out


def sec_separation(ai_dir: Path, real_dir: Path):
    ai = load_run(ai_dir)
    real = load_run(real_dir, prefix="real__")
    print(f"\n=== AI vs REAL — {len(ai)} AI clips vs {len(real)} real clips ===")
    print("  NOTE: the sets differ in content as well as origin, so part of any")
    print("  gap is domain (generative film vs robot manipulation), not authenticity.")
    print(f"    {'pipeline':26s}{'real':>8}{'AI':>8}{'delta':>9}   verdict")
    out = {}
    pids = sorted({p for r in list(ai.values()) + list(real.values()) for p in r})
    rows = []
    for pid in pids:
        r, a = mean([v.get(pid) for v in real.values()]), mean([v.get(pid) for v in ai.values()])
        if r != r and a != a:
            continue
        rows.append((pid, r, a, a - r))
    for pid, r, a, d in sorted(rows, key=lambda x: -(x[3] if x[3] == x[3] else -99)):
        if d >= 15:
            v = "SEPARATES"
        elif d <= -10:
            v = "INVERTED"
        elif min(r, a) >= 75:
            v = "saturated — no signal"
        else:
            v = "weak"
        out[pid] = {"real": round(r, 1), "ai": round(a, 1), "delta": round(d, 1), "verdict": v}
        print(f"    {pid:26s}{r:8.1f}{a:8.1f}{d:+9.1f}   {v}")
    return out


def sec_cost(run_dir: Path, sep: dict):
    W = load_wall(run_dir)
    tot = sum(st.mean(v) for v in W.values())
    print(f"\n=== COST vs SIGNAL — mean {tot / 60:.2f} min/clip over {len(W)} pipelines ===")
    print(f"    {'pipeline':26s}{'s/clip':>8}{'% run':>8}{'delta sep':>11}")
    out = {}
    for pid, v in sorted(W.items(), key=lambda x: -st.mean(x[1])):
        mw = st.mean(v)
        d = sep.get(pid, {}).get("delta")
        tag = ""
        if d is not None:
            if d >= 15 and mw < 60:
                tag = "  <-- earns its slot"
            elif mw >= 30 and d < 5:
                tag = "  <-- expensive, no signal"
        out[pid] = {"s_per_clip": round(mw, 1), "pct_run": round(mw / tot, 4), "delta_sep": d}
        print(f"    {pid:26s}{mw:8.1f}{mw / tot:8.1%}"
              f"{(f'{d:+.1f}' if d is not None else '—'):>11}{tag}")
    return out


def sec_determinism(root: Path):
    """Same clips, same config, repeated. This is the noise floor: any
    week-over-week delta smaller than this is unmeasurable."""
    reps = sorted(p for p in root.glob("rep*") if p.is_dir())
    runs = [load_run(p) for p in reps]
    runs = [r for r in runs if r]
    print(f"\n=== DETERMINISM — {len(runs)} repeats, identical clips + config ===")
    if len(runs) < 2:
        print("  fewer than 2 completed repeats — skipped")
        return {}
    clips = sorted(set.intersection(*[set(r) for r in runs]))
    pids = sorted({p for r in runs for v in r.values() for p in v})
    print(f"  {len(clips)} clip(s) x {len(runs)} repeats")
    print(f"    {'pipeline':26s}{'sd across repeats':>19}{'max spread':>13}")
    out = {}
    for pid in pids:
        sds, spreads = [], []
        for c in clips:
            vals = [r[c].get(pid) for r in runs if c in r and r[c].get(pid) is not None]
            if len(vals) >= 2:
                sds.append(st.pstdev(vals))
                spreads.append(max(vals) - min(vals))
        if not sds:
            continue
        out[pid] = {"mean_sd": round(st.mean(sds), 1), "max_spread": round(max(spreads), 1)}
        flag = "  <-- unstable" if max(spreads) >= 20 else ""
        print(f"    {pid:26s}{st.mean(sds):19.1f}{max(spreads):13.1f}{flag}")
    allsp = [v["max_spread"] for v in out.values()]
    print(f"\n  Worst single-pipeline spread across repeats: {max(allsp):.0f} severity points.")
    print("  Any before/after change smaller than this cannot be distinguished from noise.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    ap.add_argument("--rapidata", default="eval_reports/rapidata60")
    ap.add_argument("--labels", default="data/rapidata60/labels.json")
    ap.add_argument("--videos", default="data/rapidata60")
    ap.add_argument("--ai", default="eval_reports/ai_misc")
    ap.add_argument("--real", default="eval_reports/2026-07-23")
    ap.add_argument("--determinism", default="eval_reports/determinism")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    want = lambda s: a.section in ("all", s)  # noqa: E731

    res = {}
    if want("human") and Path(a.labels).exists():
        res["human"] = sec_human(Path(a.rapidata), Path(a.labels))
    if want("extremes") and Path(a.labels).exists():
        res["extremes"] = sec_extremes(Path(a.rapidata), Path(a.labels))
    if want("confound") and Path(a.labels).exists():
        res["confound"] = sec_confound(Path(a.rapidata), Path(a.labels), Path(a.videos))
    sep = {}
    if want("separation") and Path(a.real).exists():
        sep = sec_separation(Path(a.ai), Path(a.real))
        res["separation"] = sep
    if want("cost"):
        res["cost"] = sec_cost(Path(a.ai), sep)
    if want("determinism") and Path(a.determinism).exists():
        res["determinism"] = sec_determinism(Path(a.determinism))

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=1))
        print(f"\n-> {a.json_out}")


if __name__ == "__main__":
    main()

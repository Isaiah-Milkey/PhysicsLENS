"""
Full-framework evaluation: run every implemented pipeline, in stage order, on
every video under test_videos/, saving each pipeline's event stream (heavy
payloads stripped) plus a per-video summary. Aggregate afterwards with
--aggregate to get real-vs-AI separation per pipeline.

All pipelines run in ONE process, in stage order, so the in-process evidence
bus (stage 2 → 3 → 4) works and the local Qwen VLM loads once (shared by
tracker naming, s1_vlm and s3_causality).

Usage (from repo root, physics-lens env, CUDA_VISIBLE_DEVICES pinned):
  python backend/scripts/eval_framework.py --out eval_reports/2026-07-23
  python backend/scripts/eval_framework.py --aggregate --out eval_reports/2026-07-23
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif"}

ORDER = [
    "s1_temporal", "s1_optical_flow", "s1_embeddings", "s1_camera_motion",
    "s1_vlm",
    "s2_object_tracker", "s2_trajectory_extractor", "s2_event_localizer",
    "s2_hypothesis_generator",
    "s3_collision", "s3_gravity", "s3_momentum", "s3_friction",
    "s3_deformation", "s3_fluid", "s3_causality",
    "s4_report",
]

# Eval-mode overrides: skip rendered overlay videos (time + useless in saved
# streams); gravity deps run explicitly in stage order, so no planner call.
# All VLM calls route to CreateAI Gemini 3.1 Pro (user choice 2026-07-23:
# "bigger VLMs via API wherever possible") — keeps the local 17 GB Qwen off
# the GPU entirely. Exception: s3_causality NEEDS local Yes/No token logits,
# so it runs as a separate pass (--only s3_causality) when VRAM allows.
_PRO = "createai:geminipro3_1"
OVERRIDES = {
    "s1_vlm":                  {"model": _PRO},
    "s2_object_tracker":       {"render_video": "false", "naming_model": _PRO},
    "s2_trajectory_extractor": {"render_video": "false"},
    "s2_hypothesis_generator": {"model": _PRO},
    "s3_collision":            {"model": _PRO},
    "s3_gravity":              {"auto_deps": "off"},          # model already pro
    "s3_momentum":             {"model": _PRO},
    "s3_friction":             {"model": _PRO},
    "s3_deformation":          {"model": _PRO},
    "s3_fluid":                {"model": _PRO},
    "s4_report":               {"summary_model": "geminipro3_1"},
}

PIPELINE_TIMEOUT_S = 1800

_STAGES = {"s1": (1, "Stage 1 — Screening"),
           "s2": (2, "Stage 2 — Differential Diagnosis"),
           "s3": (3, "Stage 3 — Specialist Evaluation"),
           "s4": (4, "Stage 4 — Final Diagnosis")}


def prev_results(summary: dict) -> list[dict]:
    """Per-test results in the shape s4_report's `previous_results` expects
    (mirrors what the frontend passes after running earlier stages)."""
    out = []
    for pid, p in summary["pipelines"].items():
        if pid == "s4_report":
            continue
        num, name = _STAGES[pid[:2]]
        out.append({"id": pid, "pipelineId": pid, "stageId": pid[:2],
                    "stageNum": num, "stageName": name,
                    "status": "done" if p["status"] == "ok" else p["status"],
                    "severities": p["severities"], "metrics": p["metrics"],
                    "logs": p.get("logs", [])})
    return out


def find_videos():
    vids = [p for p in sorted((ROOT / "test_videos").rglob("*"))
            if p.suffix.lower() in VIDEO_EXTS]
    return [(p, "real" if "real" in p.relative_to(ROOT / "test_videos").parts[:1]
             else "ai") for p in vids]


def slug(video: Path) -> str:
    return "__".join(video.relative_to(ROOT / "test_videos").with_suffix("").parts)


def slim(obj, limit=2048):
    """Recursively truncate long strings (base64 images/videos/plotly blobs)."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else f"<stripped {len(obj)} chars>"
    if isinstance(obj, dict):
        return {k: slim(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [slim(v, limit) for v in obj]
    return obj


async def run_pipeline(run_fn, video: Path, settings: dict | None, out_path: Path):
    events, status, err = [], "ok", ""
    t0 = time.time()

    async def consume():
        async for ev in run_fn(str(video), json.dumps(settings) if settings else None):
            events.append(ev)

    try:
        await asyncio.wait_for(consume(), PIPELINE_TIMEOUT_S)
    except asyncio.TimeoutError:
        status, err = "timeout", f"exceeded {PIPELINE_TIMEOUT_S}s"
    except Exception as e:                                   # noqa: BLE001
        status, err = "error", f"{type(e).__name__}: {e}"
    wall = time.time() - t0

    with out_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(slim(ev)) + "\n")

    summ = {
        "status": status, "error": err, "wall_s": round(wall, 1),
        "n_events": len(events),
        "severities": [{"label": e.get("label"), "value": e.get("value")}
                       for e in events if e.get("type") == "severity"],
        "metrics": [{"label": e.get("label"), "value": slim(e.get("value")),
                     "sub": e.get("sub")}
                    for e in events if e.get("type") == "metric"],
        "stream_errors": [e.get("text") or e.get("message") or str(slim(e))
                          for e in events if e.get("type") == "error"],
        "logs": [{"level": e.get("level"), "text": e.get("text")}
                 for e in events if e.get("type") == "log"],
    }
    if any(e.get("type") == "error" for e in events) and status == "ok":
        summ["status"] = "stream_error"
    return summ


async def run_phase(phase_ids, out_dir: Path, only, video_filter, limit):
    from main import PIPELINES                       # heavy import (torch etc.)

    ids = [i for i in phase_ids if not only or i in only]
    videos = find_videos()
    if video_filter:
        videos = [(v, l) for v, l in videos if video_filter in str(v)]
    if limit:
        videos = videos[:limit]
    print(f"[eval] {len(videos)} videos x {len(ids)} pipelines -> {out_dir}",
          flush=True)

    for n, (video, label) in enumerate(videos, 1):
        vdir = out_dir / slug(video)
        vdir.mkdir(parents=True, exist_ok=True)
        spath = vdir / "summary.json"
        summary = json.loads(spath.read_text()) if spath.exists() else {
            "video": str(video.relative_to(ROOT)), "label": label, "pipelines": {}}
        for pid in ids:
            if summary["pipelines"].get(pid, {}).get("status") == "ok":
                continue                              # resumable: retry non-ok
            print(f"[eval] ({n}/{len(videos)}) {slug(video)} :: {pid}", flush=True)
            settings = dict(OVERRIDES.get(pid, {}))
            if pid == "s4_report":
                settings["previous_results"] = prev_results(summary)
            summ = await run_pipeline(PIPELINES[pid]["run"], video,
                                      settings or None, vdir / f"{pid}.jsonl")
            summary["pipelines"][pid] = summ
            spath.write_text(json.dumps(summary, indent=1))
            print(f"[eval]    {summ['status']} {summ['wall_s']}s "
                  f"sev={[s['value'] for s in summ['severities']]} "
                  f"{summ['error'][:120]}", flush=True)
    print("[eval] phase complete", flush=True)


def aggregate(out_dir: Path):
    rows = []
    for spath in sorted(out_dir.glob("*/summary.json")):
        s = json.loads(spath.read_text())
        for pid, p in s["pipelines"].items():
            sev = max((x["value"] for x in p["severities"]
                       if isinstance(x["value"], (int, float))), default=None)
            rows.append({"video": s["video"], "label": s["label"], "pipeline": pid,
                         "status": p["status"], "wall_s": p["wall_s"],
                         "max_severity": sev, "error": p["error"][:200]})
    import csv
    with (out_dir / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-pipeline real-vs-AI separation (max severity as the score)
    pids = sorted({r["pipeline"] for r in rows})
    lines = ["pipeline, n_ok, n_err/timeout, real_mean, ai_mean, auc"]
    for pid in pids:
        pr = [r for r in rows if r["pipeline"] == pid]
        ok = [r for r in pr if r["max_severity"] is not None]
        bad = len(pr) - len(ok)
        real = [r["max_severity"] for r in ok if r["label"] == "real"]
        ai = [r["max_severity"] for r in ok if r["label"] == "ai"]
        auc = ""
        if real and ai:
            wins = sum((a > r) + 0.5 * (a == r) for a in ai for r in real)
            auc = f"{wins / (len(ai) * len(real)):.2f}"
        fm = lambda xs: f"{sum(xs)/len(xs):.1f}" if xs else "-"
        lines.append(f"{pid}, {len(ok)}, {bad}, {fm(real)}, {fm(ai)}, {auc}")
    (out_dir / "separation.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_reports/run")
    ap.add_argument("--only", help="comma-separated pipeline ids")
    ap.add_argument("--skip", help="comma-separated pipeline ids to exclude")
    ap.add_argument("--videos", help="substring filter on video path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    out_dir = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate(out_dir)
        return
    only = set(a.only.split(",")) if a.only else None
    order = [i for i in ORDER if i not in set((a.skip or "").split(","))]
    asyncio.run(run_phase(order, out_dir, only, a.videos, a.limit))


if __name__ == "__main__":
    main()

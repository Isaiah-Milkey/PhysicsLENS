"""
Agentic evidence planner — Stage-3 pre-step.
--------------------------------------------
Before a specialist analyzes, it can ask this module to look at what Stage-2
evidence already exists on the bus for this video and fetch what's missing:

  * mode="agent"  — ONE VLM call sees a mid-frame + a compact summary of the
                    bus and picks which producers (if any) to run first.
  * mode="rules"  — deterministic fallback: run exactly the specialist's
                    declared `needs` that are absent, in dependency order.
                    (Also the automatic fallback when the agent has no
                    credentials, returns garbage, or the call fails.)

Fetched producers run inline via their normal `run()` generators with default
settings (credentials come from .env); only their `log` events are re-yielded,
prefixed `[auto <id>]`, so the specialist's own report stays uncluttered.
Evidence lands on the bus exactly as if the user had run the stage manually.

Spec: docs/superpowers/specs/2026-07-12-agentic-evidence-planner.md
"""
import asyncio
import json
from typing import AsyncGenerator, Awaitable, Callable, Optional

from tools.evidence import EVIDENCE, video_id

# Fixed dependency order: the tracker feeds the trajectory extractor; the
# localizer and hypothesis generator are independent enrichers.
DEP_ORDER = ["s2_object_tracker", "s2_trajectory_extractor",
             "s2_event_localizer", "s2_hypothesis_generator"]

STAGE2_TOOLS: dict[str, dict] = {
    "s2_object_tracker": {
        "label": "Object Tracker",
        "desc": "SAM3 subject masks + VLM-named tracks (labels/masks every "
                "later stage reuses)",
        "cost": "expensive — GPU segmentation"},
    "s2_trajectory_extractor": {
        "label": "Trajectory Extractor",
        "desc": "per-object kinematics: positions, velocities, contact and "
                "impact events",
        "cost": "medium"},
    "s2_event_localizer": {
        "label": "Event Localizer",
        "desc": "aggregates Stage-1 signals into suspicious time windows",
        "cost": "cheap"},
    "s2_hypothesis_generator": {
        "label": "Hypothesis Generator",
        "desc": "one VLM call ranking likely physics-failure types with time "
                "windows",
        "cost": "cheap — 1 VLM call"},
}

PLANNER_PROMPT = (
    "You are the evidence planner for a physics-failure specialist about to "
    "analyze this video (one frame attached).\n"
    "SPECIALIST NEEDS: {need_desc}\n"
    "EVIDENCE ALREADY ON THE BUS (null = that producer has not run):\n"
    "{summary}\n"
    "TOOLS YOU MAY RUN FIRST (id — what it produces — cost):\n"
    "{menu}\n"
    "Decide whether the existing evidence is sufficient for the specialist to "
    "make a sound decision. If not, choose the FEWEST tools (max 3) worth "
    "running first — weigh cost against what each adds for THIS video's "
    "content. Never pick a tool whose output is already present unless the "
    "summary shows it is clearly unusable (e.g. zero tracks).\n"
    'Reply with ONLY strict JSON: {{"sufficient": true/false, '
    '"run": ["<tool id>", ...], "reason": "<one short sentence>"}}'
)


def summarize_evidence(vid: str) -> dict:
    """Compact per-producer summary of the evidence bus for one video."""
    out: dict = {}
    tr = EVIDENCE.get(vid, "s2_object_tracker")
    out["s2_object_tracker"] = None if not tr else {
        "mode": tr.get("mode"),
        "n_tracks": len(tr.get("tracks") or []),
        "labels": [t.get("label") for t in (tr.get("tracks") or [])][:6]}
    tj = EVIDENCE.get(vid, "s2_trajectory_extractor")
    out["s2_trajectory_extractor"] = None if not tj else {
        "n_trajectories": len(tj.get("trajectories") or []),
        "n_contacts": len(tj.get("contacts") or []),
        "severity": tj.get("severity")}
    loc = EVIDENCE.get(vid, "s2_event_localizer")
    out["s2_event_localizer"] = None if not loc else {
        "n_markers": len(loc.get("markers") or [])}
    hy = EVIDENCE.get(vid, "s2_hypothesis_generator")
    out["s2_hypothesis_generator"] = None if not hy else {
        "top": [{"specialist": h.get("specialist"),
                 "confidence": h.get("confidence")}
                for h in (hy.get("hypotheses") or [])[:4]]}
    return out


def rules_plan(summary: dict, needs: list[str]) -> list[str]:
    """Deterministic plan: the missing hard needs, in dependency order."""
    return [s for s in DEP_ORDER if s in needs and not summary.get(s)]


def sanitize_plan(parsed) -> Optional[list[str]]:
    """VLM reply → validated plan. [] = nothing to run; None = unusable."""
    if not isinstance(parsed, dict):
        return None
    if parsed.get("sufficient") is True:
        return []
    run = parsed.get("run")
    if not isinstance(run, list):
        return None
    ids = {s for s in run if isinstance(s, str) and s in STAGE2_TOOLS}
    return [s for s in DEP_ORDER if s in ids][:3]


def _mid_frame(video_path: str):
    from tools.video import load_frames
    frames, _ = load_frames(video_path, max_frames=48, step=4)
    return frames[len(frames) // 2] if frames else None


async def _default_ask(prompt: str, video_path: str,
                       model_key: str, api_key: str) -> dict:
    from tools.vlm_router import ask_vision_json
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, lambda: _mid_frame(video_path))
    if frame is None:
        raise RuntimeError("could not decode a frame for the planner")
    return await ask_vision_json(prompt, frame, model_key, api_key, timeout=45)


def _default_runners() -> dict:
    from pipelines.stage2.event_localizer import run as r_loc
    from pipelines.stage2.object_tracker import run as r_tracker
    from pipelines.stage2.physics_hypothesis_generator import run as r_hyp
    from pipelines.stage2.trajectory_extractor import run as r_traj
    return {"s2_object_tracker": r_tracker, "s2_trajectory_extractor": r_traj,
            "s2_event_localizer": r_loc, "s2_hypothesis_generator": r_hyp}


async def ensure_evidence(video_path: str, needs: list[str], *,
                          mode: str = "agent", model_key: str = "",
                          api_key: str = "", need_desc: str = "",
                          _runners: Optional[dict] = None,
                          _ask: Optional[Callable[[str], Awaitable[dict]]] = None,
                          ) -> AsyncGenerator[dict, None]:
    """Plan + fetch missing Stage-2 evidence; yields only `log` events.

    `_runners` / `_ask` are test seams — production uses the real pipeline
    run() generators and a real VLM call.
    """
    vid = video_id(video_path)
    summary = summarize_evidence(vid)
    plan: list[str] = []

    if mode == "agent":
        from tools.vlm_router import key_status
        have, _desc = key_status(model_key, api_key)
        if not have:
            yield {"type": "log", "level": "warn",
                   "text": "Planner: no VLM credentials — using the rules "
                           "fallback (fetch missing dependencies)."}
            plan = rules_plan(summary, needs)
        else:
            menu = "\n".join(f"- {sid} — {m['desc']} — {m['cost']}"
                             for sid, m in STAGE2_TOOLS.items())
            prompt = PLANNER_PROMPT.format(
                need_desc=need_desc or "general physics evidence",
                summary=json.dumps(summary, default=str),
                menu=menu)
            try:
                parsed = await (_ask(prompt) if _ask else
                                _default_ask(prompt, video_path, model_key,
                                             api_key))
                plan = sanitize_plan(parsed)
                reason = str((parsed or {}).get("reason") or "").strip() \
                    if isinstance(parsed, dict) else ""
                if plan is None:
                    yield {"type": "log", "level": "warn",
                           "text": "Planner reply unusable — rules fallback."}
                    plan = rules_plan(summary, needs)
                elif not plan:
                    yield {"type": "log", "level": "info",
                           "text": f"Planner: existing evidence is sufficient"
                                   f"{' — ' + reason if reason else ''}."}
                else:
                    yield {"type": "log", "level": "info",
                           "text": f"Planner: fetching {', '.join(plan)} first"
                                   f"{' — ' + reason if reason else ''}."}
            except Exception as e:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"Planner call failed ({type(e).__name__}: {e})"
                               " — rules fallback."}
                plan = rules_plan(summary, needs)
    else:                                                          # rules mode
        plan = rules_plan(summary, needs)
        if plan:
            yield {"type": "log", "level": "info",
                   "text": f"Missing Stage-2 evidence — auto-running "
                           f"{', '.join(plan)}."}

    if not plan:
        return

    runners = _runners or _default_runners()
    for sid in plan:
        meta = STAGE2_TOOLS[sid]
        yield {"type": "log", "level": "info",
               "text": f"[auto {sid}] {meta['label']} ({meta['cost']})…"}
        try:
            async for ev in runners[sid](video_path, None):
                et = ev.get("type")
                if et == "log":
                    yield {"type": "log", "level": ev.get("level", "info"),
                           "text": f"[auto {sid}] {ev.get('text', '')}"}
                elif et == "error":
                    yield {"type": "log", "level": "warn",
                           "text": f"[auto {sid}] error: {ev.get('text', '')}"}
                # metrics/images/plots stay in the producer's own report
        except Exception as e:                                     # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"[auto {sid}] crashed: {type(e).__name__}: {e} — "
                           "continuing with what's available."}

    after = summarize_evidence(vid)
    got = [s for s in plan if after.get(s)]
    yield {"type": "log", "level": "success" if got else "warn",
           "text": f"Evidence fetch complete — bus gained: "
                   f"{', '.join(got) if got else 'nothing (see warnings)'}."}

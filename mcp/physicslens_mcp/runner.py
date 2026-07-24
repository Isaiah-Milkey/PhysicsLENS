"""Run orchestration: resolve a video (path / file_id / durable handle) to a
live server file_id, stream a pipeline through the aggregator with self-heal on
a forgotten file_id, and persist runs to the store. Also: temporal segmenting,
budget/stage/pipeline-scoped evaluation with Stage 2->3 hypothesis routing,
multi-video batches, and Stage 4 report generation from our own run history.

Kept separate from server.py so the MCP tools stay thin, and separate from
client.py so HTTP stays pure.
"""
import json
from pathlib import Path
from typing import Awaitable, Callable, Optional

from . import aggregate, client, report, selection
from . import store as store_mod
from .client import FileIdUnknown, PhysicsLensError

# on_log(level, text) — async sink for pipeline log lines (wired to ctx.info etc.)
OnLog = Optional[Callable[[str, str], Awaitable[None]]]

_VIDEO_EXTS = {".gif", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv", ".mpg",
               ".mpeg", ".ogv", ".wmv", ".flv", ".3gp", ".ts", ".mts", ".m2ts"}


async def _emit(on_log: OnLog, level: str, text: str) -> None:
    if on_log is not None:
        await on_log(level, text)


# ── Video resolution ─────────────────────────────────────────────────────────

async def resolve_video(st: dict, *, video: Optional[str] = None,
                        file_id: Optional[str] = None,
                        handle: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Return (server_file_id, handle) for a video given as a durable `handle`,
    a raw server `file_id`, or a local `video` path (uploaded if not already
    known). Exactly one identifying kwarg should be meaningful; `handle` wins
    if given since it's the most specific."""
    if handle:
        v = store_mod.get_video(st, handle)
        if v is None:
            raise PhysicsLensError(
                f"Unknown video handle '{handle}'. Use list_videos() to see known handles.")
        return v.get("server_file_id"), handle

    if not video and not file_id:
        raise PhysicsLensError(
            "Provide one of `video` (a local file path), `file_id`, or `handle`.")

    if video:
        p = Path(video)
        if not p.exists():
            raise PhysicsLensError(f"File not found: {video}")
        chash = store_mod.content_hash(video)
        h = store_mod.handle_for(p.name, chash)
        known = store_mod.get_video(st, h)
        if known and known.get("server_file_id"):
            return known["server_file_id"], h      # reuse; self-heal covers staleness
        rec = await client.upload_file(video)
        store_mod.upsert_video(st, h, name=p.name,
                               source_path=str(p.resolve()), content_hash=chash,
                               server_file_id=rec["id"], is_segment=False)
        store_mod.save(st)
        return rec["id"], h

    # Raw file_id: map to a known handle if we have one on record.
    for h, v in st["videos"].items():
        if v.get("server_file_id") == file_id:
            return file_id, h
    return file_id, None


def _ensure_handle(st: dict, fid: str, handle: Optional[str]) -> str:
    """A run needs a handle to persist under; mint a minimal one for a raw
    file_id we don't otherwise recognize."""
    if handle is not None:
        return handle
    h = f"fileid-{fid}"
    store_mod.upsert_video(st, h, name=h, server_file_id=fid, is_segment=False)
    store_mod.save(st)
    return h


async def _reheal(st: dict, handle: Optional[str]) -> str:
    """Self-heal a forgotten file_id (backend restarted, in-memory registry
    lost). A plain video re-uploads from its recorded local source path; a
    segment recursively re-heals its parent (a restart forgets everything, so
    the parent's id is presumed stale too) then re-derives the clip via
    /dataset/segment — no local bytes of a segment are ever cached to re-upload."""
    if handle is None:
        raise PhysicsLensError(
            "The backend has forgotten this file_id and no local handle is on "
            "record to re-derive it from. Re-run with the original `video` path.")
    v = store_mod.get_video(st, handle)
    if v is None:
        raise PhysicsLensError(f"Unknown video handle '{handle}' — cannot self-heal.")

    if v.get("is_segment"):
        parent_handle = v.get("parent_handle")
        if not parent_handle:
            raise PhysicsLensError(
                f"Segment '{handle}' has no recorded parent — cannot regenerate.")
        parent_fid = await _reheal(st, parent_handle)
        rec = await client.segment_file(parent_fid, v["t0"], v["t1"])
        store_mod.upsert_video(st, handle, server_file_id=rec["id"])
        store_mod.save(st)
        return rec["id"]

    src = v.get("source_path")
    if not src or not Path(src).exists():
        raise PhysicsLensError(
            "The backend has forgotten this video's file_id (it likely restarted), "
            + (f"and its source file '{src}' is no longer available to re-upload. "
               if src else "and no local source path is on record to re-upload. ")
            + "Re-run with the original `video` path.")
    rec = await client.upload_file(src)
    store_mod.upsert_video(st, handle, server_file_id=rec["id"])
    store_mod.save(st)
    return rec["id"]


# ── Single pipeline run (core; shared by run_pipeline/run_evaluation) ───────

async def _run_one(st: dict, *, handle: str, fid: str, pipeline_id: str,
                   settings_json: Optional[str], on_log: OnLog) -> tuple[dict, str]:
    """Stream one pipeline (with self-heal on a forgotten file_id), fold to an
    aggregated result, persist it under `handle`, and return (result, the
    file_id actually used — may differ from `fid` if a self-heal occurred)."""
    agg = aggregate.new_result(pipeline_id)

    async def _stream(use_fid: str) -> None:
        async for ev in client.run_stream(use_fid, pipeline_id, settings_json):
            aggregate.fold(agg, ev)
            t = ev.get("type")
            if t == "log":
                await _emit(on_log, ev.get("level", "info"), ev.get("text", ""))
            elif t == "error":
                await _emit(on_log, "error", ev.get("text", ""))

    try:
        await _stream(fid)
    except FileIdUnknown:
        await _emit(on_log, "warn",
                    "Backend forgot the file_id — re-uploading/regenerating and retrying…")
        fid = await _reheal(st, handle)
        agg = aggregate.new_result(pipeline_id)   # discard the failed attempt
        await _stream(fid)

    result = aggregate.finalize(agg)
    store_mod.add_run(st, handle, {
        "pipeline_id": pipeline_id,
        "status": result["status"],
        "settings": json.loads(settings_json) if settings_json else {},
        "result": result,
        "created_at": store_mod.now(),
    })
    store_mod.save(st)
    return result, fid


async def run_pipeline(*, pipeline_id: str, video: Optional[str] = None,
                       file_id: Optional[str] = None, handle: Optional[str] = None,
                       settings: Optional[dict] = None,
                       on_log: OnLog = None) -> dict:
    """Run one pipeline end-to-end; return the aggregated result (with
    `handle` and `file_id`)."""
    st = store_mod.load()
    fid, h = await resolve_video(st, video=video, file_id=file_id, handle=handle)
    h = _ensure_handle(st, fid, h)
    settings_json = json.dumps(settings) if settings else None
    result, fid = await _run_one(st, handle=h, fid=fid, pipeline_id=pipeline_id,
                                 settings_json=settings_json, on_log=on_log)
    result["handle"] = h
    result["file_id"] = fid
    return result


# ── Temporal segmentation ────────────────────────────────────────────────────

async def create_segment(*, video: Optional[str] = None, file_id: Optional[str] = None,
                         handle: Optional[str] = None, start_s: float,
                         end_s: float) -> dict:
    """Trim [start_s, end_s)s off a video and register the clip as its own
    durable handle, so any pipeline can be re-run against just that segment
    (the object tracker included — its cache is keyed by the segment's own
    content hash, so it recomputes cleanly)."""
    st = store_mod.load()
    fid, h = await resolve_video(st, video=video, file_id=file_id, handle=handle)
    h = _ensure_handle(st, fid, h)

    try:
        rec = await client.segment_file(fid, start_s, end_s)
    except FileIdUnknown:
        fid = await _reheal(st, h)
        rec = await client.segment_file(fid, start_s, end_s)

    seg_handle = f"{h}-seg{rec['t0']:.2f}-{rec['t1']:.2f}"
    store_mod.upsert_video(st, seg_handle, name=rec["name"], server_file_id=rec["id"],
                           is_segment=True, parent_handle=h,
                           t0=rec["t0"], t1=rec["t1"])
    store_mod.save(st)
    return {"handle": seg_handle, "file_id": rec["id"], "parent_handle": h,
           "t0": rec["t0"], "t1": rec["t1"], "n_frames": rec["n_frames"], "fps": rec["fps"]}


# ── Scoped evaluation (budget / stages / explicit pipelines + routing) ─────

async def run_evaluation(*, video: Optional[str] = None, file_id: Optional[str] = None,
                         handle: Optional[str] = None,
                         budget: Optional[str] = None,
                         stages: Optional[list[int]] = None,
                         pipelines: Optional[list[str]] = None,
                         top_n_specialists: int = 3,
                         settings_by_pipeline: Optional[dict[str, dict]] = None,
                         on_log: OnLog = None) -> dict:
    """Run a scoped set of pipelines on one video, in stage order. If the
    Hypothesis Generator (s2_hypothesis_generator) is in scope, its top-N
    recommended Stage 3 specialists are automatically inserted right after it
    (mirrors the app's batch routing) — even if not explicitly selected.
    Returns a compact per-pipeline summary list plus the video handle (fetch
    full results per pipeline via get_run_history).
    """
    st = store_mod.load()
    fid, h = await resolve_video(st, video=video, file_id=file_id, handle=handle)
    h = _ensure_handle(st, fid, h)

    catalog = await client.list_pipelines()
    catalog_by_id = {p["id"]: p for p in catalog}
    try:
        chosen = selection.select_pipelines(catalog, pipelines=pipelines,
                                            stages=stages, budget=budget)
    except ValueError as exc:
        raise PhysicsLensError(str(exc)) from exc

    if not chosen:
        raise PhysicsLensError(
            "No pipelines matched that scope (check `budget`/`stages`/`pipelines`).")

    queue = list(chosen)
    queued_ids = {p["id"] for p in queue}
    routed_ids: set[str] = set()
    summaries: list[dict] = []

    qi = 0
    while qi < len(queue):
        p = queue[qi]
        pid = p["id"]
        settings = (settings_by_pipeline or {}).get(pid)
        settings_json = json.dumps(settings) if settings else None
        await _emit(on_log, "info", f"Running {pid} ({qi + 1}/{len(queue)})…")

        result, fid = await _run_one(st, handle=h, fid=fid, pipeline_id=pid,
                                     settings_json=settings_json, on_log=on_log)
        summaries.append({
            "pipeline_id": pid, "status": result["status"],
            "max_severity": result["max_severity"], "timing_ms": result["timing_ms"],
            "routed": pid in routed_ids,
            "errors": result["errors"] or None,
        })

        if pid == "s2_hypothesis_generator" and result["status"] != "error":
            hyp_payload = next((r for r in result["results"] if "hypotheses" in r), None)
            if hyp_payload:
                new_specialists = selection.routed_specialists(
                    hyp_payload, catalog_by_id, top_n_specialists)
                # Insert right after the current position — NOT at the queue's
                # end — so a routed specialist always runs before anything
                # already queued later (e.g. s4_report), matching the fix made
                # to the frontend's own batch driver for the identical bug.
                inserted = 0
                for rp in new_specialists:
                    if rp["id"] in queued_ids:
                        continue
                    queue.insert(qi + 1 + inserted, rp)
                    queued_ids.add(rp["id"])
                    routed_ids.add(rp["id"])
                    inserted += 1
                if inserted:
                    await _emit(on_log, "info",
                               f"Hypothesis Generator routed in: "
                               + ", ".join(rp["id"] for rp in new_specialists))
        qi += 1

    return {"handle": h, "file_id": fid, "pipelines_run": summaries}


# ── Batch (many videos, same scope) ─────────────────────────────────────────

async def run_batch(*, videos: Optional[list[str]] = None, folder: Optional[str] = None,
                    budget: Optional[str] = None, stages: Optional[list[int]] = None,
                    pipelines: Optional[list[str]] = None,
                    top_n_specialists: int = 3, on_log: OnLog = None) -> dict:
    """Run the same scope across many local videos (a list of paths, and/or
    every video file directly under `folder`). Returns compact per-video
    summaries only — drill into a video's results with get_run_history/handle
    to avoid flooding the caller's context with every pipeline's full output.
    """
    paths = list(videos or [])
    if folder:
        fp = Path(folder)
        if not fp.is_dir():
            raise PhysicsLensError(f"Not a directory: {folder}")
        paths += sorted(str(p) for p in fp.iterdir()
                       if p.suffix.lower() in _VIDEO_EXTS)
    if not paths:
        raise PhysicsLensError(
            "Provide `videos` (a list of local paths) and/or `folder` "
            "(a directory of videos).")

    out: list[dict] = []
    for i, vp in enumerate(paths):
        async def per_video_log(level: str, text: str, _i: int = i, _vp: str = vp) -> None:
            await _emit(on_log, level, f"[{_i + 1}/{len(paths)}] {Path(_vp).name}: {text}")

        try:
            r = await run_evaluation(video=vp, budget=budget, stages=stages,
                                     pipelines=pipelines,
                                     top_n_specialists=top_n_specialists,
                                     on_log=per_video_log)
            sevs = [p["max_severity"] for p in r["pipelines_run"]
                   if p["max_severity"] is not None]
            out.append({
                "video": Path(vp).name, "handle": r["handle"],
                "status": ("error" if any(p["status"] == "error"
                                          for p in r["pipelines_run"]) else "done"),
                "pipelines_run": len(r["pipelines_run"]),
                "max_severity": max(sevs) if sevs else None,
            })
        except PhysicsLensError as exc:
            out.append({"video": Path(vp).name, "handle": None,
                       "status": "error", "error": str(exc)})

    return {"videos": out}


# ── Stage 4 report (built from OUR OWN persisted run history) ──────────────

async def generate_report(*, video: Optional[str] = None, file_id: Optional[str] = None,
                          handle: Optional[str] = None, use_llm_summary: bool = True,
                          model: Optional[str] = None, api_key: Optional[str] = None,
                          on_log: OnLog = None) -> dict:
    """Run s4_report using this video's own prior runs (persisted here) as its
    evidence — reconstructing the same `previous_results` shape the app's
    frontend would forward. Run some diagnostics on this video first."""
    st = store_mod.load()
    fid, h = await resolve_video(st, video=video, file_id=file_id, handle=handle)
    h = _ensure_handle(st, fid, h)

    v = store_mod.get_video(st, h) or {}
    prior_runs = v.get("runs", [])
    if not prior_runs:
        await _emit(on_log, "warn",
                    "No prior runs recorded for this video — the report will "
                    "be numeric-only/empty. Run some diagnostics first.")

    catalog = await client.list_pipelines()
    catalog_by_id = {p["id"]: p for p in catalog}
    previous_results = report.build_previous_results(prior_runs, catalog_by_id)

    settings: dict = {
        "previous_results": previous_results,
        "video_name": v.get("name", h),
        "use_llm_summary": str(bool(use_llm_summary)).lower(),
    }
    if model:
        settings["summary_model"] = model
    if api_key:
        settings["api_key"] = api_key

    result, fid = await _run_one(st, handle=h, fid=fid, pipeline_id="s4_report",
                                 settings_json=json.dumps(settings), on_log=on_log)
    result["handle"] = h
    result["file_id"] = fid
    return result

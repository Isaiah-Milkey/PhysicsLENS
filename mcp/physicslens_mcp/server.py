"""PhysicsLENS MCP server.

Exposes the running backend's diagnostic pipelines to an MCP client: catalog
browsing, running single pipelines or a budget/stage-scoped evaluation (with
automatic Stage 2->3 hypothesis routing), batches across many videos, temporal
segmenting, Stage 4 report generation, and read-back of everything persisted.

Run (stdio transport):
    python -m physicslens_mcp.server
"""
from fastmcp import Context, FastMCP

from . import client, runner
from . import store as store_mod
from .taxonomy import cost_rank, stage_of

mcp = FastMCP("PhysicsLENS")


def _summarize(p: dict) -> dict:
    """Compact one registry entry for listing (drops the verbose settings blob;
    `get_pipeline_details` returns that)."""
    pid = p.get("id", "")
    return {
        "id": pid,
        "name": p.get("name", pid),
        "stage": stage_of(pid),          # 1=screening … 4=report
        "cost": p.get("badge", "—"),     # cheap | medium | expensive | output
        "stub": bool(p.get("dummy", False)),
        "description": p.get("desc", ""),
        "num_settings": len(p.get("settings", []) or []),
    }


def _make_on_log(ctx: Context, prefix: str = ""):
    """Best-effort log forwarder: pipeline log lines -> MCP progress
    notifications. Never lets a failed notification (e.g. no active MCP
    session, as in a bare test harness) abort the actual run."""
    async def on_log(level: str, text: str) -> None:
        msg = f"{prefix}{text}" if prefix else text
        try:
            if level == "warn":
                await ctx.warning(msg)
            elif level == "error":
                await ctx.error(msg)
            else:
                await ctx.info(msg)
        except Exception:       # noqa: BLE001
            pass
    return on_log


# ── Catalog + reachability ───────────────────────────────────────────────────

@mcp.tool
async def health_check() -> dict:
    """Check that the PhysicsLENS backend is reachable. Returns its API URL and
    how many pipelines it exposes. Call this first if other tools fail — a clear
    'not reachable' here means the `uvicorn` backend isn't running (or
    PHYSICSLENS_API_URL points to the wrong host/port)."""
    return await client.health()


@mcp.tool
async def list_pipelines(include_stubs: bool = False) -> list[dict]:
    """List the available physics-diagnostic pipelines.

    Each entry has its stage (1=screening, 2=differential, 3=specialist,
    4=report) and cost tier (cheap/medium/expensive/output) so you can pick how
    much to spend — e.g. run only 'cheap' pipelines for a fast estimate. Results
    are sorted by stage, then ascending cost. Stub (placeholder) pipelines are
    hidden unless include_stubs=True.
    """
    pipelines = await client.list_pipelines()
    out = [_summarize(p) for p in pipelines if include_stubs or not p.get("dummy")]
    out.sort(key=lambda p: (p["stage"] if p["stage"] is not None else 99,
                            cost_rank(p["cost"])))
    return out


@mcp.tool
async def get_pipeline_details(pipeline_id: str) -> dict:
    """Full details for one pipeline, including its editable settings schema
    (each setting's id, type, default, and options). Use this to learn exactly
    what knobs a diagnostic accepts — e.g. which VLM model to select — before
    passing them as `settings` to a run.
    """
    pipelines = await client.list_pipelines()
    match = next((p for p in pipelines if p.get("id") == pipeline_id), None)
    if match is None:
        available = ", ".join(sorted(p.get("id", "") for p in pipelines))
        raise client.PhysicsLensError(
            f"Unknown pipeline '{pipeline_id}'. Available: {available}"
        )
    return {
        **_summarize(match),
        "requires_pair": bool(match.get("requires_pair", False)),
        "settings": match.get("settings", []) or [],
    }


# ── Running diagnostics ──────────────────────────────────────────────────────

@mcp.tool
async def run_diagnostic(pipeline_id: str, ctx: Context,
                         video: str | None = None,
                         file_id: str | None = None,
                         handle: str | None = None,
                         settings: dict | None = None) -> dict:
    """Run a single diagnostic pipeline on a video and return its aggregated
    result: metrics, severities, structured findings (hypotheses/violations/
    report), any LLM summary, and timing/GPU. Heavy image/video/plot data is
    omitted (only captions are kept) to stay compact.

    Identify the video with ONE of: `video` (a local file path — uploaded
    automatically), `file_id` (a video already registered on the server), or
    `handle` (a durable id returned by a previous call to this or any other
    run/segment/evaluation tool — the most reliable choice for a video you've
    already worked with). `settings` overrides pipeline defaults — see
    get_pipeline_details for the schema (e.g. {"model": "createai:geminipro3_1"}).

    The run is saved to the persistent store and tagged with a durable `handle`
    (returned in the result) that survives backend restarts.
    """
    return await runner.run_pipeline(
        pipeline_id=pipeline_id, video=video, file_id=file_id, handle=handle,
        settings=settings, on_log=_make_on_log(ctx, f"[{pipeline_id}] "))


@mcp.tool
async def run_evaluation(ctx: Context,
                         video: str | None = None, file_id: str | None = None,
                         handle: str | None = None,
                         budget: str | None = None,
                         stages: list[int] | None = None,
                         pipelines: list[str] | None = None,
                         top_n_specialists: int = 3) -> dict:
    """Run a scoped set of pipelines on one video, in stage order, and return a
    compact per-pipeline summary (status/severity/timing) — use
    get_run_history(handle) for full per-pipeline detail.

    Scope precedence (pick ONE approach): explicit `pipelines` (exact ids) >
    `stages` (e.g. [1, 2]) > `budget`. `budget` is one of:
      - "cheap"    — only cheap-tier pipelines (fast estimate, ~free)
      - "standard" — cheap + medium tiers (default if nothing else is given)
      - "thorough" — everything, including expensive Stage 3 specialists AND
                     the Stage 4 report
    If the Hypothesis Generator (s2_hypothesis_generator) ends up in scope, its
    top `top_n_specialists` recommended Stage 3 specialists are automatically
    run right after it — even if you didn't list them explicitly.

    Video identified the same way as run_diagnostic: `video` path, `file_id`,
    or a durable `handle` from an earlier call.
    """
    return await runner.run_evaluation(
        video=video, file_id=file_id, handle=handle, budget=budget,
        stages=stages, pipelines=pipelines, top_n_specialists=top_n_specialists,
        on_log=_make_on_log(ctx))


@mcp.tool
async def run_batch(ctx: Context,
                    videos: list[str] | None = None, folder: str | None = None,
                    budget: str | None = None, stages: list[int] | None = None,
                    pipelines: list[str] | None = None,
                    top_n_specialists: int = 3) -> dict:
    """Run the same scope (see run_evaluation for `budget`/`stages`/`pipelines`)
    across many LOCAL videos — pass `videos` (a list of file paths) and/or
    `folder` (every video file directly under that directory).

    Returns only a compact per-video summary (status, severity, pipeline count)
    to avoid flooding your context — use get_run_history(handle) or
    get_report(handle=...) to drill into any one video afterward.
    """
    return await runner.run_batch(
        videos=videos, folder=folder, budget=budget, stages=stages,
        pipelines=pipelines, top_n_specialists=top_n_specialists,
        on_log=_make_on_log(ctx))


# ── Temporal segmentation ────────────────────────────────────────────────────

@mcp.tool
async def segment_video(start_s: float, end_s: float,
                        video: str | None = None, file_id: str | None = None,
                        handle: str | None = None) -> dict:
    """Trim a video to the time window [start_s, end_s) and register the clip
    as its own durable handle. Use this to re-run tools (e.g. a specialist, or
    the object tracker) on just the part of a video around a flagged event —
    pass the returned `handle` into run_diagnostic/run_evaluation.

    Video identified the same way as run_diagnostic: `video` path, `file_id`,
    or a durable `handle` (segments can themselves be segmented further).
    """
    return await runner.create_segment(
        video=video, file_id=file_id, handle=handle, start_s=start_s, end_s=end_s)


# ── Stage 4 report ───────────────────────────────────────────────────────────

@mcp.tool
async def get_report(ctx: Context,
                     video: str | None = None, file_id: str | None = None,
                     handle: str | None = None, use_llm_summary: bool = True,
                     model: str | None = None, api_key: str | None = None) -> dict:
    """Generate the Stage 4 diagnostic report for a video, using ITS OWN prior
    runs (persisted by this MCP) as evidence — run some diagnostics on it
    first via run_diagnostic/run_evaluation. Returns the aggregated result,
    including the report JSON and (if use_llm_summary) an LLM-written summary.

    Video identified the same way as run_diagnostic: `video` path, `file_id`,
    or a durable `handle`. `model`/`api_key` override the summary LLM's
    defaults (see s4_report's settings via get_pipeline_details).
    """
    return await runner.generate_report(
        video=video, file_id=file_id, handle=handle,
        use_llm_summary=use_llm_summary, model=model, api_key=api_key,
        on_log=_make_on_log(ctx, "[s4_report] "))


# ── Read-back (pure store reads, no backend round-trip) ─────────────────────

@mcp.tool
def list_videos() -> list[dict]:
    """List every video this MCP has a durable handle for (uploads and
    segments alike), newest first, with its run count — the starting point for
    'what have I already tested?'."""
    return store_mod.list_videos(store_mod.load())


@mcp.tool
def get_run_history(handle: str) -> list[dict]:
    """Every run recorded for one video handle (each already aggregated and
    blob-free) — use this to inspect past results without re-running anything."""
    return store_mod.get_run_history(store_mod.load(), handle)


@mcp.tool
def list_reports() -> list[dict]:
    """Every Stage 4 report generated so far, across all videos, newest first
    — a quick index before drilling into one with get_report(handle=...)."""
    return store_mod.list_reports(store_mod.load())


if __name__ == "__main__":
    mcp.run()

"""Async HTTP client wrapping the PhysicsLENS FastAPI backend.

This is the ONLY module that knows the backend is reached over HTTP — every
tool goes through here. If the project ever switches from an HTTP client to an
in-process design, this is the single file to swap.
"""
import json
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastmcp.exceptions import ToolError

from . import config


class PhysicsLensError(ToolError):
    """A friendly, agent-readable error (server unreachable, unknown id, HTTP
    failure). Subclasses FastMCP's ToolError so its message is surfaced to the
    caller as an expected tool error — without a server-side traceback — instead
    of being masked/logged as an unexpected crash."""


class FileIdUnknown(PhysicsLensError):
    """The backend doesn't recognize a file_id (HTTP 404) — usually because it
    restarted and lost its in-memory file registry. Callers can self-heal by
    re-uploading the source file."""

    def __init__(self, file_id: str):
        super().__init__(f"Backend does not recognize file_id '{file_id}'.")
        self.file_id = file_id


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(config.HTTP_READ_TIMEOUT, connect=config.HTTP_CONNECT_TIMEOUT)


def _unreachable(exc: Exception) -> str:
    return (f"Can't reach the PhysicsLENS backend at {config.API_URL} ({exc}). "
            "Is `uvicorn main:app` running? If it's on another host/port, set "
            "the PHYSICSLENS_API_URL environment variable.")


# ── Read-only catalog + reachability ─────────────────────────────────────────

async def _get(path: str) -> httpx.Response:
    url = config.API_URL + path
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as http:
            return await http.get(url)
    except httpx.ConnectError as exc:
        raise PhysicsLensError(_unreachable(exc)) from exc
    except httpx.HTTPError as exc:
        raise PhysicsLensError(f"HTTP error talking to {url}: {exc}") from exc


async def health() -> dict[str, Any]:
    """Best-effort reachability probe. There is no dedicated health route, so
    `/pipelines` doubles as the liveness signal."""
    resp = await _get("/pipelines")
    if resp.status_code != 200:
        raise PhysicsLensError(
            f"Backend reachable but /pipelines returned HTTP {resp.status_code}."
        )
    return {
        "reachable": True,
        "api_url": config.API_URL,
        "pipeline_count": len(resp.json()),
    }


async def list_pipelines() -> list[dict]:
    """The registry (each pipeline minus its `run` callable) exactly as the
    backend exposes it: id, name, desc, badge, dummy, requires_pair, settings."""
    resp = await _get("/pipelines")
    if resp.status_code != 200:
        raise PhysicsLensError(
            f"/pipelines returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    return resp.json()


# ── Upload + run ─────────────────────────────────────────────────────────────

async def upload_file(path: str) -> dict:
    """Upload a local video to /dataset/upload; return its {'id','name'} record."""
    p = Path(path)
    if not p.exists():
        raise PhysicsLensError(f"File not found: {path}")
    url = config.API_URL + "/dataset/upload"
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as http:
            with open(p, "rb") as fh:
                resp = await http.post(
                    url, files={"files": (p.name, fh, "application/octet-stream")})
    except httpx.ConnectError as exc:
        raise PhysicsLensError(_unreachable(exc)) from exc
    if resp.status_code != 200:
        raise PhysicsLensError(
            f"Upload failed (HTTP {resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    recs = data.get("files") or []
    if not recs:
        raise PhysicsLensError(
            f"Upload of '{p.name}' registered no file (skipped={data.get('skipped')}). "
            "Is the extension a supported video type?")
    return recs[0]


async def segment_file(file_id: str, t0: float, t1: float) -> dict:
    """POST /dataset/segment — trim a registered video to [t0, t1)s and
    register the clip as a new file. Returns {'id','name','parent_id','t0',
    't1','fps','n_frames'}. Raises FileIdUnknown if the parent id has been
    forgotten (e.g. backend restart) so callers can self-heal it first."""
    url = config.API_URL + "/dataset/segment"
    form = {"file_id": file_id, "t0": str(t0), "t1": str(t1)}
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as http:
            resp = await http.post(url, data=form)
    except httpx.ConnectError as exc:
        raise PhysicsLensError(_unreachable(exc)) from exc
    if resp.status_code == 404:
        raise FileIdUnknown(file_id)
    if resp.status_code != 200:
        raise PhysicsLensError(
            f"/dataset/segment returned HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def run_stream(file_id: str, pipeline_id: str,
                     settings: Optional[str] = None) -> AsyncIterator[dict]:
    """POST /dataset/run and yield parsed NDJSON event dicts.

    `settings` is the raw JSON string the backend expects (or None). Raises
    FileIdUnknown on HTTP 404 so callers can self-heal by re-uploading; other
    non-200s raise PhysicsLensError.
    """
    url = config.API_URL + "/dataset/run"
    form = {"pipeline_id": pipeline_id, "file_id": file_id}
    if settings:
        form["settings"] = settings
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as http:
            async with http.stream("POST", url, data=form) as resp:
                if resp.status_code == 404:
                    await resp.aread()
                    raise FileIdUnknown(file_id)
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise PhysicsLensError(
                        f"/dataset/run returned HTTP {resp.status_code}: "
                        f"{body[:200]!r}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue          # skip any non-JSON keepalive/noise
    except httpx.ConnectError as exc:
        raise PhysicsLensError(_unreachable(exc)) from exc

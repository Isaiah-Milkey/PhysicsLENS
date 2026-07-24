"""Persistent JSON store: durable video handles + their run history.

A single consolidated file (``config.STORE_PATH``), written atomically. The
agent references videos by a durable ``handle`` — a stable slug from name +
content hash — that survives backend restarts; the ephemeral server ``file_id``
is an internal detail we refresh via self-heal. Runs are nested under each video
so results are never lost. (Phase 3 nests reports here too.)

Kept to one file on purpose: no per-run file sprawl.
"""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

_SCHEMA = "physicslens_mcp_store"
_VERSION = 1


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def content_hash(path: str) -> str:
    """Fast content id: sha1 of size + head/tail 1 MiB. Cheap on large files and
    stable, so the same file always yields the same handle."""
    st = os.stat(path)
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    chunk = 1 << 20
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def _slug(name: str) -> str:
    base = re.sub(r"[^\w.-]+", "-", Path(name).stem).strip("-").lower()
    return base or "video"


def handle_for(name: str, chash: str) -> str:
    """Deterministic durable handle from a video's name + content hash."""
    return f"{_slug(name)}-{chash[:8]}"


# ── Load / save (atomic) ─────────────────────────────────────────────────────

def load() -> dict:
    """Load the store, returning a fresh empty one if absent or unreadable."""
    try:
        with open(config.STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("videos"), dict):
            return data
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError):
        pass          # corrupt/unreadable → start fresh (atomic writes make this rare)
    return {"schema": _SCHEMA, "version": _VERSION, "videos": {}}


def save(store: dict) -> None:
    """Write the store atomically (temp file in the same dir, then os.replace)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(config.DATA_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        os.replace(tmp, config.STORE_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── Video + run helpers ──────────────────────────────────────────────────────

def get_video(store: dict, handle: str) -> Optional[dict]:
    return store["videos"].get(handle)


def upsert_video(store: dict, handle: str, **fields) -> dict:
    """Create or update a video record. Only non-None fields overwrite, so a
    partial update (e.g. refreshing just server_file_id) is safe."""
    v = store["videos"].get(handle)
    if v is None:
        v = {"handle": handle, "created_at": now(), "runs": []}
        store["videos"][handle] = v
    for k, val in fields.items():
        if val is not None:
            v[k] = val
    return v


def add_run(store: dict, handle: str, run: dict) -> None:
    v = store["videos"].setdefault(
        handle, {"handle": handle, "created_at": now(), "runs": []})
    v.setdefault("runs", []).append(run)


# ── Read-only queries (pure, no network) ─────────────────────────────────────

def list_videos(store: dict) -> list[dict]:
    """Compact summary of every known video handle, newest first."""
    out = []
    for handle, v in store["videos"].items():
        runs = v.get("runs", [])
        out.append({
            "handle": handle, "name": v.get("name", handle),
            "is_segment": bool(v.get("is_segment")),
            "parent_handle": v.get("parent_handle"),
            "num_runs": len(runs),
            "created_at": v.get("created_at"),
            "last_run_at": runs[-1]["created_at"] if runs else None,
        })
    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out


def get_run_history(store: dict, handle: str) -> list[dict]:
    """Every run recorded for one video handle (already aggregated/blob-free)."""
    v = store["videos"].get(handle)
    return list(v.get("runs", [])) if v else []


def list_reports(store: dict) -> list[dict]:
    """Every s4_report run across every video, newest first."""
    out = []
    for handle, v in store["videos"].items():
        for run in v.get("runs", []):
            if run.get("pipeline_id") != "s4_report":
                continue
            res = run.get("result") or {}
            out.append({
                "handle": handle, "video_name": v.get("name", handle),
                "created_at": run.get("created_at"),
                "max_severity": res.get("max_severity"),
                "has_llm_summary": bool(res.get("llm_summary")),
            })
    out.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return out

# PhysicsLENS MCP

An [MCP](https://modelcontextprotocol.io) server that exposes the PhysicsLENS
physics-diagnostic pipelines to an agent (Claude Desktop, Claude Code, …).

It's a **thin HTTP client** over the existing FastAPI backend — it does not
import torch / SAM3 / any GPU code. The pipelines run inside your normal
`uvicorn` backend; this server just lets an agent drive them, then stores the
runs and reports so results aren't lost.

```
agent ──MCP(stdio)──> physicslens-mcp ──HTTP──> uvicorn backend (GPU) 
```

## Status

Phases 1–3 done — all 11 tools are live:

| Tool | Purpose |
|---|---|
| `health_check` | Backend reachability + pipeline count |
| `list_pipelines` / `get_pipeline_details` | Catalog with stage + cost tier; full settings schema |
| `run_diagnostic` | Run one pipeline on a video |
| `run_evaluation` | Run a scoped set (`budget` cheap/standard/thorough, `stages`, or explicit `pipelines`) in stage order, with the Hypothesis Generator auto-routing its top-N recommended Stage 3 specialists |
| `run_batch` | The same scope across many local videos / a folder — compact per-video summaries |
| `segment_video` | Trim a video to a time window, registered as its own durable handle (re-run any tool on just that clip) |
| `get_report` | Generate the Stage 4 report from a video's own persisted run history |
| `list_videos` / `get_run_history` / `list_reports` | Read back everything persisted, no backend round-trip |

A video is always identified by ONE of: a local `video` path (uploaded
automatically), a raw backend `file_id`, or a durable `handle` (returned by any
run/segment/evaluation call — the most reliable choice for a video you've
already worked with; it survives backend restarts via self-heal).

Every run is persisted to `data/store.json` (single consolidated file, atomic
writes) under a **durable handle** derived from the video's name + content
hash — stable even if the backend restarts and forgets its in-memory file
registry. If a pipeline run hits a forgotten `file_id`, the MCP transparently
re-uploads (or, for a segment, re-derives the clip from its parent) and
retries once.

Backend addition this required: `POST /dataset/segment` (`file_id`, `t0`, `t1`
→ new `file_id`), added to `dataset_api.py`, plus a small `save_frames_as_video`
helper in `tools/video.py` (OpenCV mp4v — no system ffmpeg dependency).

## Setup

Requires Python 3.10+. Kept in its own environment (no GPU deps):

```bash
cd mcp
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e .        # installs fastmcp + httpx
```

The backend must be running (locally or on a remote GPU box):

```bash
cd ../backend && uvicorn main:app --port 8000
```

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `PHYSICSLENS_API_URL` | `http://localhost:8000` | Backend base URL. Point at a forwarded port for a remote GPU box. |
| `PHYSICSLENS_MCP_DATA` | `mcp/data` | Directory for the persistent JSON store (gitignored). |
| `PHYSICSLENS_READ_TIMEOUT` | `1800` | Seconds to wait on a run (GPU runs are slow). |
| `PHYSICSLENS_CONNECT_TIMEOUT` | `10` | Seconds to wait for a connection. |

## Run it

```bash
python -m physicslens_mcp.server      # stdio transport
```

### Register with Claude Code

`.mcp.json` (project root) or via `claude mcp add`:

```json
{
  "mcpServers": {
    "physicslens": {
      "command": "python",
      "args": ["-m", "physicslens_mcp.server"],
      "cwd": "C:/Users/isami/Documents/Code/Spring_2026/WorldModels/PhysicsLENS/mcp",
      "env": { "PHYSICSLENS_API_URL": "http://localhost:8000" }
    }
  }
}
```

(Use the venv's Python in `command` if you installed into one.)

## Smoke test

With the backend running:

```bash
python scripts/smoke_test.py
```

Prints reachability + every live pipeline with its stage and cost tier.

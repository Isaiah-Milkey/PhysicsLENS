"""Runtime configuration for the PhysicsLENS MCP, all overridable via env vars.

  PHYSICSLENS_API_URL      base URL of the running FastAPI backend
                           (default http://localhost:8000). Point this at a
                           forwarded port when the backend is on a remote GPU box.
  PHYSICSLENS_MCP_DATA     directory for the persistent JSON store
                           (default: <this repo>/mcp/data).
  PHYSICSLENS_CONNECT_TIMEOUT   seconds to wait for a connection (default 10).
  PHYSICSLENS_READ_TIMEOUT      seconds to wait for a response (default 1800 =
                           30 min, because GPU specialist runs are slow).
"""
import os
from pathlib import Path

API_URL: str = os.environ.get("PHYSICSLENS_API_URL", "http://localhost:8000").rstrip("/")

# Default the store to mcp/data (sibling of this package), kept out of git.
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"
DATA_DIR: Path = Path(os.environ.get("PHYSICSLENS_MCP_DATA", str(_DEFAULT_DATA)))
STORE_PATH: Path = DATA_DIR / "store.json"

# Generous read timeout — a single expensive specialist can take minutes on GPU.
HTTP_CONNECT_TIMEOUT: float = float(os.environ.get("PHYSICSLENS_CONNECT_TIMEOUT", "10"))
HTTP_READ_TIMEOUT: float = float(os.environ.get("PHYSICSLENS_READ_TIMEOUT", "1800"))

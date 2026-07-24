"""PhysicsLENS MCP — an MCP server that exposes the PhysicsLENS physics
diagnostic pipelines (running in the FastAPI backend) to an agent.

Thin HTTP client over the backend: no GPU/model dependencies live here, so the
same server drives pipelines whether the backend runs locally or on a remote
GPU box (set PHYSICSLENS_API_URL). See README.md.
"""

__version__ = "0.1.0"

"""Standalone smoke test for the PhysicsLENS MCP (Phase 1).

Talks to a LIVE backend — start it first:  cd backend && uvicorn main:app --port 8000
Then, from the mcp/ directory:              python scripts/smoke_test.py

Matches the repo convention of standalone `python` scripts (no pytest). Prints a
PASS/FAIL summary; exits non-zero on failure.
"""
import asyncio
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physicslens_mcp import client              # noqa: E402
from physicslens_mcp.taxonomy import cost_rank, stage_of  # noqa: E402


async def main() -> int:
    failures = 0

    try:
        h = await client.health()
        print(f"[health] reachable at {h['api_url']} — {h['pipeline_count']} pipeline(s)")
    except client.PhysicsLensError as exc:
        print(f"[health] FAIL: {exc}")
        return 1  # nothing else will work if the server is down

    pipes = await client.list_pipelines()
    live = [p for p in pipes if not p.get("dummy")]
    stubs = [p for p in pipes if p.get("dummy")]
    print(f"[pipelines] {len(pipes)} total — {len(live)} live, {len(stubs)} stub")

    for p in sorted(live, key=lambda p: (stage_of(p.get("id", "")) or 99,
                                         cost_rank(p.get("badge", "")))):
        st = stage_of(p.get("id", ""))
        if st is None:
            print(f"  ! {p.get('id')} has no sN_ stage prefix")
            failures += 1
        print(f"  s{st} · {p.get('badge','—'):9} · {p.get('id')}")

    print("PASS" if failures == 0 else f"FAIL ({failures} issue(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

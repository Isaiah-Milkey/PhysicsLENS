"""Phase 2 live smoke test — runs one REAL pipeline against your running backend.

Unlike smoke_test.py (read-only catalog check), this exercises the actual
upload -> run -> aggregate -> persist path via `runner.run_pipeline`, using a
cheap Stage 1 pipeline by default so it's fast and free (no VLM/API cost).

Usage (from mcp/, with the backend running and this venv active):
    python scripts/live_run_test.py
    python scripts/live_run_test.py --pipeline s1_temporal
    python scripts/live_run_test.py --video "../test_videos/ai_generated/bowling-ball-drop.mp4"
    python scripts/live_run_test.py --pipeline s3_collision   # a real (paid/slow) specialist
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physicslens_mcp import client, runner          # noqa: E402
from physicslens_mcp import store as store_mod      # noqa: E402
from physicslens_mcp import config                  # noqa: E402

DEFAULT_VIDEO = (Path(__file__).resolve().parent.parent.parent
                 / "test_videos" / "ai_generated" / "Ai Bouncing Ball.mp4")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(DEFAULT_VIDEO),
                    help="local video path to run against (default: a small sample clip)")
    ap.add_argument("--pipeline", default="s1_optical_flow",
                    help="pipeline id to run (default: a cheap Stage 1 check)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"FAIL: video not found: {video}")
        return 1

    print(f"[setup] backend      = {config.API_URL}")
    print(f"[setup] store        = {config.STORE_PATH}")
    print(f"[setup] video        = {video.name} ({video.stat().st_size / 1024:.0f} KB)")
    print(f"[setup] pipeline     = {args.pipeline}")

    async def on_log(level: str, text: str) -> None:
        tag = {"warn": "WARN", "error": "ERR "}.get(level, "log ")
        print(f"  [{tag}] {text}")

    try:
        result = await runner.run_pipeline(
            pipeline_id=args.pipeline, video=str(video), on_log=on_log)
    except client.PhysicsLensError as exc:
        print(f"FAIL: {exc}")
        return 1

    print()
    print(f"[result] status       = {result['status']}")
    print(f"[result] handle       = {result['handle']}")
    print(f"[result] file_id      = {result['file_id']}")
    print(f"[result] max_severity = {result['max_severity']}")
    print(f"[result] timing_ms    = {result['timing_ms']}")
    print(f"[result] gpu_mb       = {result['gpu_mb']}")
    print(f"[result] metrics      = {len(result['metrics'])}")
    print(f"[result] media        = { {k: len(v) for k, v in result['media'].items()} }")
    if result.get("errors"):
        print(f"[result] errors       = {result['errors']}")

    # Verify it actually landed in the persistent store.
    st = store_mod.load()
    v = store_mod.get_video(st, result["handle"])
    ok = bool(v and v.get("runs") and v["runs"][-1]["pipeline_id"] == args.pipeline)
    print()
    print(f"[store]  video record = {'found' if v else 'MISSING'}")
    print(f"[store]  runs stored  = {len(v['runs']) if v else 0}")
    print(f"[store]  source_path  = {v.get('source_path') if v else None}")

    failed = result["status"] == "error" or not ok
    print()
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

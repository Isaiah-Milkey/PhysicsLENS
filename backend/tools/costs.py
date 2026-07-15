"""
Uniform cost instrumentation for every diagnostic test.

Every test streams the same trailing cost metrics so runs are directly
comparable across tests and settings (the cost/accuracy tradeoff panel):

  • Runtime        — wall-clock seconds for the whole test
  • Peak GPU memory— torch.cuda peak allocated during the run (if CUDA present)
  • Cost tier      — the registry badge (cheap / medium / expensive / output)

Wrap any pipeline event generator with `instrument(gen, badge=...)`; the cost
metrics are emitted just before the final "done" event, so they always appear
at the bottom of the test's metric list.
"""
import time
from typing import AsyncGenerator

TIER_DESC = {
    "cheap":     "CPU-light, seconds",
    "medium":    "GPU model inference",
    "expensive": "heavy multi-model GPU",
    "output":    "aggregation only",
}


def _cuda():
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:  # noqa: BLE001
        pass
    return None


async def instrument(gen, badge: str = "—") -> AsyncGenerator[dict, None]:
    """Re-yield every event from `gen`, then append standard cost metrics
    (before the terminal "done" event if the test emitted one)."""
    torch = _cuda()
    if torch:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            torch = None

    t0 = time.perf_counter()
    done_ev = None
    async for ev in gen:
        if isinstance(ev, dict) and ev.get("type") == "done":
            done_ev = ev          # hold it — cost metrics go before "done"
            continue
        yield ev
    elapsed = time.perf_counter() - t0

    # Machine-readable duration for the frontend to attach to the run entry
    # (per-tool timing → per-stage aggregation, benchmarking, export). The
    # human-readable "Runtime" metric below stays for the cost panel.
    yield {"type": "timing", "duration_ms": round(elapsed * 1000),
           "badge": badge}
    yield {"type": "metric", "label": "Runtime", "value": f"{elapsed:.1f}s",
           "sub": "wall clock, whole test"}
    if torch:
        try:
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            yield {"type": "metric", "label": "Peak GPU memory",
                   "value": f"{peak_gb:.1f} GB",
                   "sub": "CUDA peak allocated this run"}
        except Exception:  # noqa: BLE001
            pass
    yield {"type": "metric", "label": "Cost tier", "value": badge,
           "sub": TIER_DESC.get(badge, "")}

    if done_ev is not None:
        yield done_ev

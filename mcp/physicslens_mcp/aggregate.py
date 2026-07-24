"""Fold a pipeline's NDJSON event stream into one compact, agent-friendly result.

Pipelines emit many events (log/metric/severity/image/plotly/video/marker_video/
signal/result/llm_summary/timing/done/error). An agent wants the numbers and the
structured findings, not megabytes of base64 imagery — so heavy blobs
(image/video/plot `data`) are dropped here, keeping only that they exist and
their caption.

Pure functions (no I/O), so they're trivially testable: `new_result` →
`fold` per event → `finalize`.
"""
from typing import Any

# event type → media bucket name (their `data` blob is dropped, caption kept)
_MEDIA_BUCKET = {"image": "images", "plotly": "plots",
                 "video": "videos", "marker_video": "marker_videos"}
_LOG_TAIL = 12          # keep at most this many info-level log lines


def new_result(pipeline_id: str) -> dict:
    """A fresh accumulator for one pipeline run."""
    return {
        "pipeline_id": pipeline_id,
        "status": "running",
        "metrics": [],
        "severities": [],
        "results": [],          # payloads of `result` events (hypotheses/violations/report)
        "llm_summary": None,
        "timing_ms": None,
        "gpu_mb": None,
        "warnings": [],
        "errors": [],
        "log_tail": [],
        "media": {"images": [], "plots": [], "videos": [], "marker_videos": []},
        "signals": [],
    }


def fold(agg: dict, ev: dict) -> None:
    """Accumulate one event into `agg` (mutates in place)."""
    t = ev.get("type")
    if t == "log":
        level, text = ev.get("level", "info"), ev.get("text", "")
        if level == "warn":
            agg["warnings"].append(text)
        elif level == "error":
            agg["errors"].append(text)
        else:
            agg["log_tail"].append(text)
            if len(agg["log_tail"]) > _LOG_TAIL:
                agg["log_tail"].pop(0)
    elif t == "metric":
        agg["metrics"].append({"label": ev.get("label"), "value": ev.get("value"),
                               "sub": ev.get("sub", "")})
    elif t == "severity":
        agg["severities"].append({"label": ev.get("label"), "value": ev.get("value"),
                                  "color": ev.get("color")})
    elif t == "result":
        agg["results"].append({k: v for k, v in ev.items() if k != "type"})
    elif t == "llm_summary":
        agg["llm_summary"] = ev.get("markdown") or ev.get("summary")
    elif t == "timing":
        agg["timing_ms"] = ev.get("duration_ms")
        agg["gpu_mb"] = ev.get("gpu_mb")
    elif t == "signal":
        agg["signals"].append(ev.get("source"))
    elif t in _MEDIA_BUCKET:
        agg["media"][_MEDIA_BUCKET[t]].append(ev.get("caption") or "")
    elif t == "error":
        agg["errors"].append(ev.get("text", ""))
        agg["status"] = "error"
    # "done" is handled in finalize (an earlier error may already have won)


def finalize(agg: dict) -> dict:
    """Resolve final status, add a top-line severity, and prune empty media."""
    if agg["status"] != "error":
        agg["status"] = "done"
    agg["media"] = {k: v for k, v in agg["media"].items() if v}
    sev_vals = [s["value"] for s in agg["severities"]
                if isinstance(s.get("value"), (int, float))]
    agg["max_severity"] = max(sev_vals) if sev_vals else None
    return agg

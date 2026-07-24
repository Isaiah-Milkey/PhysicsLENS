"""Reconstruct s4_report's expected `previous_results` shape from the MCP's own
persisted run history for a video.

The app's frontend accumulates each run's frontend-shaped entry client-side and
forwards the list into s4_report's settings; the MCP has no frontend, so this
rebuilds the same shape from `store.py`'s persisted (already-aggregated,
blob-free) runs.
"""
from .taxonomy import STAGE_NAMES, stage_of


def build_previous_results(runs: list[dict], catalog_by_id: dict[str, dict]) -> list[dict]:
    """One entry per prior run (excluding earlier s4_report runs — a report
    should never summarize a previous report)."""
    out = []
    for i, run in enumerate(runs):
        pid = run.get("pipeline_id", "")
        if pid == "s4_report":
            continue
        res = run.get("result") or {}
        logs = ([{"level": "warn", "text": w} for w in res.get("warnings", [])]
                + [{"level": "error", "text": e} for e in res.get("errors", [])])
        st_num = stage_of(pid)
        p = catalog_by_id.get(pid, {})
        out.append({
            "id": f"{pid}-{i}",
            "stageId": f"stage{st_num}" if st_num else "unknown",
            "stageName": STAGE_NAMES.get(st_num, "Unknown Stage"),
            "stageNum": st_num,
            "pipelineId": pid,
            "pipelineName": p.get("name", pid),
            "timestamp": run.get("created_at"),
            "status": res.get("status", run.get("status")),
            "metrics": res.get("metrics", []),
            "severities": res.get("severities", []),
            "logs": logs,
            "comment": "",
            "excluded": False,
        })
    return out

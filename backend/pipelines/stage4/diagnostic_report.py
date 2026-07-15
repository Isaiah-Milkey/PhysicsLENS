"""
Stage 4 · Diagnostic Report Generator

First real version:
- Aggregates previous per-test frontend reports.
- Emits a unified JSON report.
- Emits a simple visual global report using Plotly.

Future version:
- Add stronger per-test schemas.
- Add failure-type aggregation.
- Add visual evidence selection.
- Add LLM summary.
- Add PDF/HTML rendering.
"""

import asyncio
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import AsyncGenerator, Any


import plotly.graph_objects as go


def _to_float(x, default=None):
    try:
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return float(x)
    except Exception:
        return default


def _extract_numeric_metrics(test: dict) -> list[dict]:
    out = []
    for m in test.get("metrics", []) or []:
        value = m.get("value")
        num = _to_float(value)
        if num is not None:
            out.append({
                "label": m.get("label", ""),
                "value": num,
                "raw_value": value,
                "sub": m.get("sub", ""),
            })
    return out


def _extract_max_severity(test: dict) -> float | None:
    vals = []
    for s in test.get("severities", []) or []:
        v = _to_float(s.get("value"))
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def _status_to_score(status: str) -> float:
    """
    Very simple placeholder scoring:
    done/success/pass -> good
    error/fail -> bad
    running/unknown -> uncertain
    """
    s = (status or "").lower()
    if s in {"done", "success", "pass"}:
        return 1.0
    if s in {"error", "fail", "failed"}:
        return 0.0
    return 0.5


def _build_unified_report(previous_results: list[dict], video_name: str) -> dict:
    included = [
        r for r in previous_results
        if not r.get("excluded", False) and r.get("pipelineId") != "s4_report"
    ]

    by_stage = defaultdict(list)
    for r in included:
        by_stage[str(r.get("stageName", "Unknown Stage"))].append(r)

    status_counts = Counter((r.get("status") or "unknown") for r in included)

    severity_values = []
    local_reports = []

    for r in included:
        max_sev = _extract_max_severity(r)
        if max_sev is not None:
            severity_values.append(max_sev)

        numeric_metrics = _extract_numeric_metrics(r)

        warnings = [
            l.get("text", "")
            for l in r.get("logs", []) or []
            if l.get("level") == "warn"
        ]

        errors = [
            l.get("text", "")
            for l in r.get("logs", []) or []
            if l.get("level") == "error"
        ]

        local_reports.append({
            "test_id": r.get("id"),
            "stage": {
                "id": r.get("stageId"),
                "name": r.get("stageName"),
                "num": r.get("stageNum"),
            },
            "pipeline": {
                "id": r.get("pipelineId"),
                "name": r.get("pipelineName"),
            },
            "status": r.get("status"),
            "timestamp": r.get("timestamp"),
            "max_severity": max_sev,
            "numeric_metrics": numeric_metrics,
            "num_metrics": len(r.get("metrics", []) or []),
            "num_severity_items": len(r.get("severities", []) or []),
            "num_warnings": len(warnings),
            "num_errors": len(errors),
            "warnings": warnings[:5],
            "errors": errors[:5],
            "user_comment": r.get("comment", ""),
        })

    if severity_values:
        avg_severity = sum(severity_values) / len(severity_values)
        max_severity = max(severity_values)
    else:
        avg_severity = None
        max_severity = None

    status_score = (
        sum(_status_to_score(r.get("status")) for r in included) / len(included)
        if included else None
    )

    # Placeholder global score.
    # Later, replace this with a real physics scoring model.
    if status_score is None:
        physics_consistency_score = None
    elif avg_severity is None:
        physics_consistency_score = round(status_score * 100, 2)
    else:
        physics_consistency_score = round(
            max(0.0, min(100.0, status_score * 100.0 - avg_severity * 0.35)),
            2,
        )

    if max_severity is None:
        global_severity_label = "unknown"
    elif max_severity >= 75:
        global_severity_label = "critical"
    elif max_severity >= 50:
        global_severity_label = "high"
    elif max_severity >= 25:
        global_severity_label = "medium"
    else:
        global_severity_label = "low"

    recommended_followup = []
    if not included:
        recommended_followup.append("Run at least one Stage 1 or Stage 2 diagnostic test before generating the final report.")
    if any(r["num_errors"] > 0 for r in local_reports):
        recommended_followup.append("Inspect tests with backend errors before trusting the final diagnosis.")
    if max_severity is not None and max_severity >= 50:
        recommended_followup.append("Run localized and specialist diagnostics around the high-severity event windows.")
    if len(included) < 2:
        recommended_followup.append("Run multiple complementary tests to improve confidence.")

    report = {
        "schema_version": "0.1",
        "report_type": "physicslens_global_diagnosis",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": video_name,

        "global_summary": {
            "num_tests_total": len(previous_results),
            "num_tests_included": len(included),
            "status_counts": dict(status_counts),
            "physics_consistency_score": physics_consistency_score,
            "global_severity": {
                "label": global_severity_label,
                "max_value": max_severity,
                "average_value": round(avg_severity, 2) if avg_severity is not None else None,
            },
            "confidence": {
                "level": "low" if len(included) < 2 else "medium",
                "reason": "This is a first-pass aggregation based only on available frontend test outputs.",
            },
        },

        "stage_summary": {
            stage: {
                "num_tests": len(items),
                "pipelines": [x.get("pipelineName") for x in items],
                "statuses": dict(Counter(x.get("status") or "unknown" for x in items)),
            }
            for stage, items in by_stage.items()
        },

        "local_reports": local_reports,

        "recommended_followup": recommended_followup,

        "llm_ready_context": {
            "instruction": (
                "Summarize the global physics diagnosis using the local reports, "
                "highlight likely failure modes, severity, confidence, and recommended follow-up tests."
            ),
            "inputs": {
                "global_summary": "See global_summary.",
                "stage_summary": "See stage_summary.",
                "local_reports": "See local_reports.",
            },
        },
    }

    return report


def _build_visual_summary(report: dict) -> str:
    local_reports = report.get("local_reports", [])

    names = []
    severities = []

    for r in local_reports:
        pipeline_name = r.get("pipeline", {}).get("name", "Unknown test")
        stage_name = r.get("stage", {}).get("name", "Unknown stage")
        max_sev = r.get("max_severity")

        if max_sev is None:
            max_sev = 0

        names.append(f"{pipeline_name}<br><sup>{stage_name}</sup>")
        severities.append(max_sev)

    if not names:
        names = ["No previous tests"]
        severities = [0]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=severities,
        text=[f"{v:.1f}%" for v in severities],
        textposition="auto",
        name="Max severity",
    ))

    fig.update_layout(
        title="Global Diagnostic Report: Severity by Test",
        xaxis_title="Diagnostic Test",
        yaxis_title="Max Severity (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=20, t=60, b=120),
        height=420,
    )

    return fig.to_json()

def _build_llm_prompt(report: dict) -> str:
    """
    Convert the unified PhysicsLENS JSON report into a prompt for the LLM.
    """
    report_text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))

    return f"""
You are a physics diagnostic assistant for AI-generated videos.

You will receive a structured JSON report from PhysicsLENS.
Your job is to write a concise final diagnosis for a human evaluator.

Important rules:
- Do not invent evidence that is not present in the JSON.
- If a score is described as prototype, first-pass, or placeholder, say that clearly.
- Distinguish between "no issue detected" and "no severity data reported".
- Focus on physics consistency, detected anomalies, severity, confidence, and recommended follow-up tests.
- Do not claim that the video has a specific physical failure unless the JSON supports it.
- Use careful language such as "the available diagnostics suggest" rather than overconfident claims.

Please return the answer in this format:

# PhysicsLENS Final Diagnosis

## Overall Assessment
...

## Main Evidence
...

## Severity and Confidence
...

## Recommended Follow-up
...

Here is the PhysicsLENS JSON report:

{report_text}
""".strip()


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}

    output_format = cfg.get("output_format", "json_visual")
    previous_results = cfg.get("previous_results", [])
    video_name = cfg.get("video_name", "uploaded video")
    use_llm_summary = str(cfg.get("use_llm_summary", "false")).lower() == "true"
    api_key = str(cfg.get("api_key", "")).strip()
    summary_model = str(cfg.get("summary_model") or "geminiflash2_5").strip()

    yield {
        "type": "log",
        "level": "info",
        "text": f"Diagnostic Report Generator received {len(previous_results)} previous test result(s).",
    }

    await asyncio.sleep(0)

    report = _build_unified_report(previous_results, video_name)

    score = report["global_summary"]["physics_consistency_score"]
    sev = report["global_summary"]["global_severity"]

    yield {
        "type": "metric",
        "label": "Included tests",
        "value": report["global_summary"]["num_tests_included"],
        "sub": "Previous non-excluded test results aggregated into this report.",
    }

    yield {
        "type": "metric",
        "label": "Physics consistency score",
        "value": "N/A" if score is None else score,
        "sub": "Placeholder score for first-pass testing.",
    }

    if sev["max_value"] is not None:
        color = "#dc2626" if sev["max_value"] >= 75 else "#f97316" if sev["max_value"] >= 50 else "#facc15" if sev["max_value"] >= 25 else "#22c55e"
        yield {
            "type": "severity",
            "label": f"Global severity: {sev['label']}",
            "value": round(sev["max_value"], 2),
            "color": color,
        }

    if output_format in {"json_visual", "json"}:
        yield {
            "type": "result",
            "title": "Unified Diagnostic JSON",
            "report": report,
        }

    if output_format == "json_visual":
        yield {
            "type": "plotly",
            "caption": "Global visual summary",
            "data": _build_visual_summary(report),
        }

    if use_llm_summary:
        yield {
            "type": "log",
            "level": "info",
            "text": f"Requesting LLM summary from CreateAI ({summary_model}).",
        }

        try:
            from tools.createai import query_text, response_text
            data = await query_text(_build_llm_prompt(report),
                                    model=summary_model,
                                    token=api_key or None)
            llm_summary = response_text(data) or json.dumps(data, indent=2)

            yield {
                "type": "log",
                "level": "info",
                "text": f"CreateAI responded ({summary_model}). Summary length: {len(llm_summary)} characters.",
            }

            report["llm_summary"] = {
                "provider": "CreateAI",
                "model": summary_model,
                "summary": llm_summary,
            }

            yield {
                "type": "llm_summary",
                "title": "LLM Diagnostic Summary",
                "markdown": llm_summary,
            }

        except Exception as e:
            yield {
                "type": "log",
                "level": "error",
                "text": f"LLM summary failed: {e}",
            }

    yield {"type": "done"}
"""
Stage 4 · Water Physics Benchmark — Homemade vs. SOTA
-----------------------------------------------------
Runs the four grounded water tests and the SOTA-style comparators on one clip
and renders them side by side. Computes the shared dense-flow + mask sequence
once and feeds it to the grounded analyses (no recompute). Highlights where the
grounded signals LOCALIZE a failure that the coarse comparators only summarize.
"""
import asyncio, json
from typing import AsyncGenerator, List

from tools.video import load_frames, sample_frames
from tools.fluid import compute_flow_sequence, severity_color

from pipelines.stage3.water_incompressibility import analyze as a_incompress
from pipelines.stage3.water_mass_conservation import analyze as a_mass
from pipelines.stage3.water_vorticity import analyze as a_vort
from pipelines.stage3.water_surface_coherence import analyze as a_surface
from pipelines.stage3.water_impact_dynamics import analyze as a_impact
from pipelines.stage3.water_vbench_flow import analyze as a_vbench

GROUNDED = [
    ("incompressibility", "Incompressibility", a_incompress, True),
    ("mass_conservation", "Mass conservation", a_mass, False),
    ("vorticity", "Vorticity", a_vort, True),
    ("surface_coherence", "Surface coherence", a_surface, True),
    ("impact_dynamics", "Impact dynamics", a_impact, True),
]


def compare(frames: List, fps: float, cfg: dict) -> dict:
    flow_seq = compute_flow_sequence(frames, backend=cfg.get("backend", "auto"),
                                     mask_method=cfg.get("mask_method", "auto"))
    methods, localized = {}, []
    for key, label, fn, uses_flow in GROUNDED:
        res = fn(frames, fps, cfg, flow_seq=flow_seq) if uses_flow else fn(frames, fps, cfg)
        methods[key] = {"label": label, "severity": res["severity"], "kind": "grounded"}
        for sig in res.get("signals", []):
            localized.append({"method": label, "time": round(sig["frame"] / fps, 2),
                              "type": sig["signal_type"], "score": sig["score"]})

    vb = a_vbench(frames, fps, cfg, flow_seq=flow_seq)
    methods["vbench_flow"] = {"label": "VBench-style flow", "severity": vb["severity"], "kind": "sota"}

    # Agreement notes: each localized grounded event vs the coarse SOTA verdicts.
    sota_flagged = methods["vbench_flow"]["severity"] > 15
    agreement = []
    for ev in sorted(localized, key=lambda e: e["time"])[:12]:
        agreement.append(
            f"t={ev['time']}s — {ev['method']} localized '{ev['type']}' "
            f"(score {ev['score']}); VBench-style only gives a clip-level "
            f"{'elevated' if sota_flagged else 'low'} score.")
    return {"methods": methods, "localized": localized, "agreement": agreement,
            "sota_flagged": sota_flagged}


def _bar_figure(methods: dict) -> str:
    import plotly.graph_objects as go
    keys = list(methods)
    labels = [methods[k]["label"] for k in keys]
    sev = [methods[k]["severity"] for k in keys]
    colors = ["#7c3aed" if methods[k]["kind"] == "grounded" else "#c05621" for k in keys]
    fig = go.Figure(go.Bar(x=labels, y=sev, marker_color=colors,
                           text=sev, textposition="outside"))
    fig.update_layout(
        title=dict(text="Water physics: homemade (purple) vs SOTA (orange)",
                   font=dict(size=15, color="#1a1917")),
        height=440, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=80),
        yaxis=dict(title="Severity (0–100)", range=[0, 105], showgrid=True, gridcolor="#ebebeb"),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    return fig.to_json()


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}
    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return
    yield {"type": "log", "level": "info",
           "text": "Running 4 grounded tests + VBench-style comparator…"}
    await asyncio.sleep(0)

    out = compare(frames, fps, cfg)

    # Optional VLM-judge (only if a key is supplied) — appended to the chart.
    api_key = cfg.get("api_key", "")
    if api_key:
        try:
            from pipelines.stage3.water_vlm_judge import score_keyframes, severity_from_verdicts
            kf = sample_frames(frames, int(cfg.get("num_frames", 5)))
            verdicts = await score_keyframes(kf, cfg.get("model", "gpt-4o"), api_key)
            out["methods"]["vlm_judge"] = {"label": "VLM-as-judge",
                "severity": severity_from_verdicts(verdicts), "kind": "sota"}
        except Exception as exc:
            yield {"type": "log", "level": "warn", "text": f"VLM-judge skipped: {exc}"}

    yield {"type": "plotly", "data": _bar_figure(out["methods"]),
           "caption": "Severity per method. Grounded tests localize; SOTA comparators summarize."}
    for key, m in out["methods"].items():
        yield {"type": "metric", "label": m["label"], "value": str(m["severity"]),
               "sub": "grounded" if m["kind"] == "grounded" else "SOTA comparator"}

    grounded_sev = [m["severity"] for m in out["methods"].values() if m["kind"] == "grounded"]
    headline = max(grounded_sev) if grounded_sev else 0
    yield {"type": "severity", "label": "Water physics violation (grounded headline)",
           "value": headline, "color": severity_color(headline)}

    if out["agreement"]:
        yield {"type": "log", "level": "info", "text": "— Agreement analysis —"}
        for line in out["agreement"]:
            yield {"type": "log", "level": "warn", "text": line}
    else:
        yield {"type": "log", "level": "success",
               "text": "No grounded water-physics violations localized."}
    yield {"type": "done"}

"""
Stage 3 · Specialist — Fluid Specialist
-----------------------------------------
Consolidated grounded water-physics specialist. Computes the shared dense-flow +
water-mask sequence once, then runs the four grounded analyses on it:

  • Incompressibility   — divergence ∇·v (sources/sinks: water created/destroyed)
  • Mass conservation   — water-region area continuity (water popping in/out)
  • Vorticity           — curl ∇×v plausibility band (implausibly smooth/chaotic)
  • Surface coherence   — foam/texture advection vs the flow (flicker-in-place)

Emits one combined report: a chart + metrics + severity per sub-signal, an
aggregate Stage-2 `signal`, and an overall fluid-physics severity (the max of the
four sub-severities). The individual `water_*.py` modules remain importable and
are reused here and by the Stage-4 benchmark.
"""
import asyncio, json
from typing import AsyncGenerator, List, Optional

from tools.video import load_frames
from tools.fluid import compute_flow_sequence, severity_color, timeseries_figure

from pipelines.stage3.water_incompressibility import analyze as a_incompress
from pipelines.stage3.water_mass_conservation import analyze as a_mass
from pipelines.stage3.water_vorticity import analyze as a_vort
from pipelines.stage3.water_surface_coherence import analyze as a_surface

# (key, label, analyze_fn, uses_flow_seq, plot traces [(series-label, colour)], threshold (cfg-key, default, label))
_SUB = [
    {"key": "incompressibility", "label": "Incompressibility (∇·v)",
     "fn": a_incompress, "uses_flow": True,
     "traces": [("normalized |∇·v|", "#1a54c4")],
     "thr": ("divergence_threshold", 0.08, "divergence threshold")},
    {"key": "mass_conservation", "label": "Mass / area conservation",
     "fn": a_mass, "uses_flow": False,
     "traces": [("water area fraction", "#1a54c4"), ("|Δarea| rate", "#E24B4A")],
     "thr": ("jump_threshold", 0.20, "jump threshold")},
    {"key": "vorticity", "label": "Vorticity / turbulence (∇×v)",
     "fn": a_vort, "uses_flow": True,
     "traces": [("normalized |∇×v|", "#7c3aed")],
     "thr": ("max_vorticity", 0.80, "upper band")},
    {"key": "surface_coherence", "label": "Surface & splash coherence",
     "fn": a_surface, "uses_flow": True,
     "traces": [("advection NCC", "#1a7a3c")],
     "thr": ("coherence_floor", 0.35, "coherence floor")},
]


def analyze_all(frames: List, fps: float, cfg: dict, flow_seq: Optional[list] = None) -> dict:
    """Run the four grounded analyses on one shared flow sequence.

    Returns {"subs": {key: <analyze dict>}, "overall_severity": int, "signals": [...]}.
    `overall_severity` is the max of the four sub-severities (worst signal wins).
    """
    if flow_seq is None:
        flow_seq = compute_flow_sequence(
            frames, backend=cfg.get("backend", "auto"),
            mask_method=cfg.get("mask_method", "auto"))

    subs, signals = {}, []
    for spec in _SUB:
        fn = spec["fn"]
        r = fn(frames, fps, cfg, flow_seq=flow_seq) if spec["uses_flow"] else fn(frames, fps, cfg)
        subs[spec["key"]] = r
        signals.extend(r.get("signals", []))

    overall = max((r["severity"] for r in subs.values()), default=0)
    return {"subs": subs, "overall_severity": overall, "signals": signals}


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg = json.loads(settings) if settings else {}

    yield {"type": "log", "level": "info", "text": "Loading video…"}
    frames, fps = load_frames(video_path)
    if len(frames) < 2:
        yield {"type": "error", "text": "Video too short (need ≥ 2 frames)."}
        return

    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps — computing shared flow + running 4 fluid checks…"}
    await asyncio.sleep(0)

    flow_seq = compute_flow_sequence(
        frames, backend=cfg.get("backend", "auto"), mask_method=cfg.get("mask_method", "auto"))
    result = analyze_all(frames, fps, cfg, flow_seq=flow_seq)

    worst = []
    for spec in _SUB:
        r = result["subs"][spec["key"]]
        yield {"type": "log", "level": "info", "text": f"— {spec['label']} —"}
        traces = [(lbl, r["series"][lbl], col) for (lbl, col) in spec["traces"] if lbl in r["series"]]
        thr_key, thr_def, thr_label = spec["thr"]
        yield {"type": "plotly",
               "data": timeseries_figure(r["time"], traces, spec["label"],
                                         threshold=float(cfg.get(thr_key, thr_def)),
                                         ythresh_label=thr_label),
               "caption": r["summary"]}
        for m in r["metrics"]:
            yield {"type": "metric", **m}
        yield {"type": "severity", "label": spec["label"], "value": r["severity"], "color": r["color"]}
        if r["severity"] > 15:
            worst.append(f"{spec['label']} ({r['severity']})")

    # Aggregate signal for the Stage-2 Event Localizer.
    yield {"type": "signal", "source": "s3_fluid", "source_name": "Fluid Specialist",
           "fps": float(fps), "n_frames": int(len(frames)),
           "severity": result["overall_severity"], "signals": result["signals"]}

    overall = result["overall_severity"]
    yield {"type": "severity", "label": "Fluid physics violation (overall)",
           "value": overall, "color": severity_color(overall)}

    msg = ("Fluid violations: " + ", ".join(worst) + ".") if worst else \
          "No grounded fluid-physics violations detected."
    yield {"type": "log", "level": "warn" if worst else "success", "text": msg}
    yield {"type": "done"}

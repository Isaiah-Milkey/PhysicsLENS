"""
Stage 3 · Specialist — Fluid Specialist
-----------------------------------------
Consolidated grounded water-physics specialist. Computes the shared dense-flow +
water-mask sequence once, then runs the five grounded analyses on it:

  • Incompressibility   — divergence ∇·v (sources/sinks: water created/destroyed)
  • Mass conservation   — water-region area continuity (water popping in/out)
  • Vorticity           — curl ∇×v plausibility band (implausibly smooth/chaotic)
  • Surface coherence   — foam/texture advection vs the flow (flicker-in-place)
  • Impact dynamics     — splash-impulse sharpness (real impact vs smeared AI motion)

Emits one combined report: a masked-region video (the water the specialist
analysed), a chart + metrics + severity per sub-signal, an aggregate Stage-2
`signal`, and an overall fluid-physics severity (the max of the five
sub-severities). The individual `water_*.py` modules remain importable and are
reused here and by the Stage-4 benchmark.
"""
import asyncio, base64, json
from typing import AsyncGenerator, List, Optional

import cv2

from tools.video import load_frames, encode_video_browser
from tools.fluid import (compute_flow_sequence, resize_frames, resolve_water_region,
                         sam3_water_masks, severity_color, timeseries_figure)

from pipelines.stage3.water_incompressibility import analyze as a_incompress
from pipelines.stage3.water_mass_conservation import analyze as a_mass
from pipelines.stage3.water_vorticity import analyze as a_vort
from pipelines.stage3.water_surface_coherence import analyze as a_surface
from pipelines.stage3.water_impact_dynamics import analyze as a_impact

# (key, label, analyze_fn, uses_flow_seq, plot traces [(series-label, colour)], threshold (cfg-key, default, label))
_SUB = [
    {"key": "incompressibility", "label": "Incompressibility (∇·v)",
     "fn": a_incompress, "uses_flow": True,
     "traces": [("normalized |∇·v|", "#1a54c4")],
     "thr": ("divergence_threshold", 0.25, "divergence threshold")},
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
    {"key": "impact_dynamics", "label": "Impact / splash dynamics",
     "fn": a_impact, "uses_flow": True,
     "traces": [("fluid motion (px/frame)", "#c05621")],
     "thr": None},   # impulse is a scalar; the flow-magnitude chart needs no hline
]


def _mask_overlay_frames(frames: List, masks: List, caption: str) -> List:
    """Tint each frame red where its water mask is True + stamp a caption."""
    out = []
    for f, m in zip(frames, masks):
        red = f.copy()
        red[m] = (0, 0, 255)
        o = cv2.addWeighted(f, 0.6, red, 0.4, 0)
        cv2.putText(o, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out.append(o)
    return out


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

    frames = resize_frames(frames, int(cfg.get("max_height", 480)))
    h, w = frames[0].shape[:2]
    yield {"type": "log", "level": "info",
           "text": f"{len(frames)} frames @ {fps:.1f} fps · working res {w}x{h} — locating water region…"}
    await asyncio.sleep(0)

    mask_method = cfg.get("mask_method", "auto")
    static_mask, frame_masks, mask_label = None, None, mask_method
    if mask_method == "sam3":
        yield {"type": "log", "level": "info", "text": "Segmenting water with SAM3 (learned)…"}
        await asyncio.sleep(0)
        try:
            frame_masks = sam3_water_masks(
                frames, prompt=cfg.get("water_prompt", "water"),
                max_frames=int(cfg.get("sam3_max_frames", 80)))
            mask_label = "sam3"
        except Exception as exc:
            yield {"type": "log", "level": "warn",
                   "text": f"SAM3 unavailable ({exc}); falling back to the motion mask."}
            mask_method = "auto"
    if frame_masks is None:
        static_mask, mask_label = resolve_water_region(frames, mask_method)
    flow_seq = compute_flow_sequence(
        frames, backend=cfg.get("backend", "auto"), mask_method=mask_method,
        static_mask=static_mask, frame_masks=frame_masks)

    coverage = float(flow_seq[0]["mask"].mean()) if flow_seq else 0.0
    # A learned SAM3 mask is tight by construction, so high coverage isn't a red flag there.
    low_conf = (coverage < 0.01) or (coverage > 0.60 and mask_label != "sam3")
    yield {"type": "metric", "label": "Water region", "value": f"{coverage * 100:.0f}%",
           "sub": f"mask: {mask_label}" + (" — low confidence" if low_conf else "")}
    if low_conf:
        yield {"type": "log", "level": "warn",
               "text": (f"Water-region mask covers {coverage * 100:.0f}% of the frame ({mask_label}); "
                        "results may be unreliable. Try mask_method=sam3 (learned), motion, or hsv.")}

    yield {"type": "log", "level": "info", "text": "Running 5 grounded fluid checks on the shared flow…"}
    await asyncio.sleep(0)
    result = analyze_all(frames, fps, cfg, flow_seq=flow_seq)
    overall = result["overall_severity"]

    # Masked-region video output — shows exactly which pixels were analysed.
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")
    if render_video and flow_seq:
        yield {"type": "log", "level": "info", "text": "Rendering masked-region video…"}
        await asyncio.sleep(0)
        try:
            masks = (frame_masks if frame_masks is not None
                     else [static_mask] * len(frames) if static_mask is not None
                     else [flow_seq[0]["mask"]] + [s["mask"] for s in flow_seq])
            caption = f"mask {mask_label} {coverage * 100:.0f}%   |   overall {overall}"
            ov = _mask_overlay_frames(frames, masks, caption)
            loop = asyncio.get_event_loop()
            data, mime = await loop.run_in_executor(None, encode_video_browser, ov, fps)
            yield {"type": "video", "data": base64.b64encode(data).decode(), "mime": mime,
                   "caption": "Red = the water region the Fluid Specialist analysed; the four "
                              "flow checks and impact dynamics are measured inside it."}
            yield {"type": "log", "level": "info", "text": f"Masked-region video ready ({len(data) / 1024:.0f} KB)."}
        except Exception as exc:
            yield {"type": "log", "level": "warn", "text": f"Mask video skipped: {exc}"}

    worst = []
    for spec in _SUB:
        r = result["subs"][spec["key"]]
        yield {"type": "log", "level": "info", "text": f"— {spec['label']} —"}
        traces = [(lbl, r["series"][lbl], col) for (lbl, col) in spec["traces"] if lbl in r["series"]]
        threshold, thr_label = None, ""
        if spec["thr"] is not None:
            thr_key, thr_def, thr_label = spec["thr"]
            threshold = float(cfg.get(thr_key, thr_def))
        yield {"type": "plotly",
               "data": timeseries_figure(r["time"], traces, spec["label"],
                                         threshold=threshold, ythresh_label=thr_label),
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

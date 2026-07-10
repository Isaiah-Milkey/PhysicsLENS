"""
Stage 3 · Specialist — Collision & Contact
--------------------------------------------
Verifies collision physics between the masked subjects (absorbs the former
Contact Specialist — same events, same evidence, one report):

1. CONTACT EPISODES  — per subject pair, mask-intersection over time. Overlap
   beyond a threshold fraction of the smaller object marks contact; deep or
   sustained overlap is interpenetration (solid objects cannot share volume).
2. RESTITUTION       — vertical centroid velocity into vs out of each episode:
   e = |v_out| / |v_in|. e > restitution_max means energy from nowhere.
3. PHANTOM BOUNCE    — a velocity reversal while moving, with no contact, no
   nearby subject, and not at the frame bottom (implicit ground), means the
   object bounced off nothing.
4. VLM CONFIRMATION  — the worst interpenetration moments are rendered with
   both objects at full brightness (rest dimmed) and verified by the VLM, so
   see-through containers (ball inside an open crate) don't false-positive.

Evidence comes from the Object Tracker's masks on the evidence bus (inline
segmentation fallback, same as the Consistency Specialist).
"""
import asyncio
import base64
import json
from itertools import combinations
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames
from tools.evidence import EVIDENCE, video_id
from tools.vlm import parse_vlm_json

PENETRATION_PROMPT = (
    "The image shows two tracked objects from one video at full brightness "
    "(labels and timestamp printed on the image); everything else is DIMMED. "
    "An automated overlap metric flagged them as possibly INTERPENETRATING — "
    "occupying the same volume like ghosts, which solid objects cannot do.\n"
    "Careful: one object being INSIDE an open container, resting on, hidden "
    "behind, or seen through holes of the other is normal contact, NOT "
    "interpenetration. Only true merging/phasing of solid material counts.\n"
    "Reply with ONLY strict JSON: "
    '{"verdict": "interpenetration"|"normal_contact", "confidence": <0..1>, '
    '"explanation": "<one sentence>"}'
)


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _jpeg_event(img: np.ndarray, caption: str, quality: int = 90) -> dict:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return {"type": "image", "mime": "image/jpeg",
            "data": base64.b64encode(buf).decode(), "caption": caption}


def _centroids(masks: dict) -> dict[int, tuple[float, float]]:
    out = {}
    for fi, m in masks.items():
        ys, xs = np.where(m)
        if xs.size:
            out[fi] = (float(xs.mean()), float(ys.mean()))
    return out


def _velocity(times: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Smoothed d/dt on an uneven grid."""
    if len(vals) < 3:
        return np.zeros_like(vals)
    v = np.gradient(vals, times)
    k = np.ones(3) / 3.0
    return np.convolve(v, k, mode="same")


def _overlap_series(masks_a: dict, masks_b: dict) -> dict[int, float]:
    """Per common frame: RAW intersection area / smaller object's area.

    Nonzero raw intersection between two instance masks means shared pixels —
    interpenetration evidence (a clean segmenter keeps touching objects
    mutually exclusive)."""
    out = {}
    for fi in sorted(set(masks_a) & set(masks_b)):
        a, b = masks_a[fi], masks_b[fi]
        inter = int(np.logical_and(a, b).sum())
        denom = max(1, min(int(a.sum()), int(b.sum())))
        out[fi] = inter / denom
    return out


def _contact_series(masks_a: dict, masks_b: dict) -> dict[int, float]:
    """Per common frame: DILATED intersection / smaller object's raw area.

    Real contact shows as masks being adjacent, not overlapping — dilating
    both by ~1% of the mask height turns adjacency into a thin measurable
    intersection band."""
    out = {}
    kern = None
    for fi in sorted(set(masks_a) & set(masks_b)):
        a, b = masks_a[fi], masks_b[fi]
        if kern is None:
            r = max(3, int(a.shape[0] * 0.012))
            kern = np.ones((r, r), np.uint8)
        ad = cv2.dilate(a.astype(np.uint8), kern) > 0
        bd = cv2.dilate(b.astype(np.uint8), kern) > 0
        inter = int(np.logical_and(ad, bd).sum())
        denom = max(1, min(int(a.sum()), int(b.sum())))
        out[fi] = inter / denom
    return out


def _episodes(overlap: dict[int, float], thresh: float,
              grid_step: int) -> list[dict]:
    """Group contiguous over-threshold frames into contact episodes."""
    hot = [fi for fi, ov in sorted(overlap.items()) if ov >= thresh]
    if not hot:
        return []
    eps, start, prev = [], hot[0], hot[0]
    for fi in hot[1:]:
        if fi - prev > 2 * grid_step:
            eps.append((start, prev))
            start = fi
        prev = fi
    eps.append((start, prev))
    return [{"f_start": s, "f_end": e,
             "peak_overlap": max(ov for fi, ov in overlap.items() if s <= fi <= e),
             "peak_frame": max((fi for fi in overlap if s <= fi <= e),
                               key=lambda fi: overlap[fi])}
            for s, e in eps]


def _pair_render(frame: np.ndarray, mask_a_s: np.ndarray, mask_b_s: np.ndarray,
                 label: str) -> Optional[np.ndarray]:
    """Both objects bright, rest dimmed, cropped to their padded union box."""
    H, W = frame.shape[:2]
    ma = cv2.resize(mask_a_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    mb = cv2.resize(mask_b_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    keep = ma | mb
    ys, xs = np.where(keep)
    if xs.size < 32:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = max(12, (x1 - x0) // 8), max(12, (y1 - y0) // 8)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(W, x1 + px + 1), min(H, y1 + py + 1)
    crop = frame[y0:y1, x0:x1].copy()
    crop[~keep[y0:y1, x0:x1]] //= 4
    band = np.full((30, crop.shape[1], 3), 20, np.uint8)
    cv2.putText(band, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([band, crop], axis=0)


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg             = json.loads(settings) if settings else {}
    model           = str(cfg.get("model") or "createai:geminiflash2_5")
    api_key         = str(cfg.get("api_key", "")).strip()
    overlap_thresh  = float(cfg.get("overlap_threshold", 0.02))
    deep_overlap    = float(cfg.get("deep_overlap", 0.35))
    restitution_max = float(cfg.get("restitution_max", 1.1))
    max_checks      = max(1, int(cfg.get("max_checks", 3)))
    max_subjects    = max(1, int(cfg.get("max_subjects", 3)))

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    # ── 1. Subject masks (evidence bus, inline fallback) ─────────────────────
    subjects_masks: dict[str, dict[int, np.ndarray]] = {}
    sampled_frames: list[int] = []
    ev_tr = EVIDENCE.get(vid, "s2_object_tracker")
    use_bus_masks = bool(ev_tr and ev_tr.get("masks_png"))

    # Decode frames so their indices match how the masks are keyed. The Object
    # Tracker's SAM3 masks are keyed to load_frames_rgb(num_frames) at
    # target_h=480 — decode identically (→ BGR) when reusing them so mask index
    # i ↔ frames[i]; otherwise decode the full video for inline segmentation.
    if use_bus_masks and ev_tr.get("num_frames"):
        from tools.video import load_frames_rgb
        yield {"type": "log", "level": "info",
               "text": "Loading video (aligned to Object Tracker frames)…"}
        rgb, raw_fps = await loop.run_in_executor(
            None, load_frames_rgb, video_path, int(ev_tr["num_frames"]))
        frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
    else:
        yield {"type": "log", "level": "info", "text": "Loading video…"}
        frames, raw_fps = await loop.run_in_executor(None, load_frames, video_path)
    n = len(frames)
    if n < 3:
        yield {"type": "error", "text": f"Video too short ({n} frames)."}
        return
    eff_fps = raw_fps or 30.0
    H = frames[0].shape[0]

    if use_bus_masks:
        from tools.sam3 import decode_mask_png
        for label, per_frame in ev_tr["masks_png"].items():
            subjects_masks[label] = {int(fi): decode_mask_png(png)
                                     for fi, png in per_frame.items()}
        sampled_frames = ev_tr.get("sampled_frames") or sorted(
            {fi for m in subjects_masks.values() for fi in m})
        yield {"type": "log", "level": "info",
               "text": f"Reusing Object Tracker masks for {len(subjects_masks)} "
                       "subject(s) from the evidence bus."}
    else:
        yield {"type": "log", "level": "warn",
               "text": "No Object Tracker masks on the evidence bus — segmenting "
                       "inline (run the Object Tracker first to share its masks)."}
        try:
            from tools.vlm_router import name_subjects
            from tools.sam3 import segment_concepts
            subjects = await name_subjects([frames[0], frames[n // 2]],
                                           max_subjects=max_subjects,
                                           model_key=model, api_key=api_key)
            if not subjects:
                raise RuntimeError("VLM returned no subject names")
            yield {"type": "log", "level": "info",
                   "text": "Primary subjects: " + ", ".join(f"“{s}”" for s in subjects)}
            seg = await loop.run_in_executor(
                None, lambda: segment_concepts(frames, subjects))
            subjects_masks = seg["subjects"]
            sampled_frames = seg["sampled_frames"]
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "error",
                   "text": f"Could not obtain subject masks: {str(exc)[:300]}"}
            return
    if len(subjects_masks) < 1:
        yield {"type": "error", "text": "No subject masks available."}
        return
    grid_step = (int(np.median(np.diff(sampled_frames)))
                 if len(sampled_frames) > 2 else 1)

    # Per-subject vertical kinematics from mask centroids.
    kin: dict[str, dict] = {}
    for label, masks in subjects_masks.items():
        cents = _centroids(masks)
        fis = sorted(cents)
        if len(fis) < 3:
            continue
        t = np.array([fi / eff_fps for fi in fis])
        y = np.array([cents[fi][1] for fi in fis])
        kin[label] = {"fis": fis, "t": t, "y": y, "vy": _velocity(t, y),
                      "cents": cents}

    # ── 2. Contact episodes per subject pair (mask intersection) ─────────────
    violations: list[dict] = []
    pair_overlaps: dict[tuple, dict[int, float]] = {}
    all_episodes: list[dict] = []

    raw_overlaps: dict[tuple, dict[int, float]] = {}
    for la, lb in combinations(subjects_masks.keys(), 2):
        raw  = _overlap_series(subjects_masks[la], subjects_masks[lb])
        cont = _contact_series(subjects_masks[la], subjects_masks[lb])
        if not cont:
            continue
        pair_overlaps[(la, lb)] = cont
        raw_overlaps[(la, lb)] = raw
        for ep in _episodes(cont, overlap_thresh, grid_step):
            # Interpenetration evidence = peak RAW overlap inside the episode.
            in_ep = {fi: ov for fi, ov in raw.items()
                     if ep["f_start"] <= fi <= ep["f_end"]}
            ep["peak_raw"] = max(in_ep.values(), default=0.0)
            ep["peak_raw_frame"] = (max(in_ep, key=in_ep.get)
                                    if in_ep else ep["peak_frame"])
            ep.update(pair=(la, lb),
                      t_start=round(ep["f_start"] / eff_fps, 3),
                      t_end=round(ep["f_end"] / eff_fps, 3))
            all_episodes.append(ep)

    yield {"type": "log", "level": "info",
           "text": f"{len(all_episodes)} contact episode(s) across "
                   f"{len(pair_overlaps)} subject pair(s) "
                   f"(dilated-mask adjacency ≥ {overlap_thresh:g})."}
    for ep in all_episodes:
        yield {"type": "log", "level": "info",
               "text": f"Contact: “{ep['pair'][0]}” × “{ep['pair'][1]}” "
                       f"t={ep['t_start']:.2f}–{ep['t_end']:.2f}s "
                       f"(peak raw overlap {ep['peak_raw']:.0%})."}
    await asyncio.sleep(0)

    # 2a. Interpenetration candidates: deep RAW overlap within an episode.
    for ep in all_episodes:
        if ep["peak_raw"] >= deep_overlap:
            depth_score = min(1.0, (ep["peak_raw"] - deep_overlap)
                              / max(1e-6, 1.0 - deep_overlap) + 0.4)
            violations.append({
                "type": "interpenetration", "pair": list(ep["pair"]),
                "t": round(ep["peak_raw_frame"] / eff_fps, 3),
                "frame": int(ep["peak_raw_frame"]),
                "peak_overlap": round(ep["peak_raw"], 3),
                "duration_s": round(ep["t_end"] - ep["t_start"], 3),
                "score": round(depth_score, 3), "confirmed": None,
            })
            yield {"type": "log", "level": "warn",
                   "text": f"Possible interpenetration: “{ep['pair'][0]}” × "
                           f"“{ep['pair'][1]}” peaks at "
                           f"{ep['peak_raw']:.0%} raw overlap @ "
                           f"t={ep['peak_raw_frame'] / eff_fps:.2f}s."}

    # 2b. Restitution across each episode (vertical, for each participant).
    v_floor = 0.02 * H                       # px/s — ignore near-static objects
    for ep in all_episodes:
        for label in ep["pair"]:
            k = kin.get(label)
            if k is None:
                continue
            pre  = [v for fi, v in zip(k["fis"], k["vy"]) if fi < ep["f_start"]][-3:]
            post = [v for fi, v in zip(k["fis"], k["vy"]) if fi > ep["f_end"]][:3]
            if not pre or not post:
                continue
            v_in, v_out = float(np.median(pre)), float(np.median(post))
            if v_in < v_floor or v_out > -0.2 * v_floor:   # need: down in, up out
                continue
            e = abs(v_out) / max(abs(v_in), 1e-6)
            msg = (f"“{label}” bounce off “{[p for p in ep['pair'] if p != label][0]}” "
                   f"@ t={ep['t_start']:.2f}s: e = {e:.2f} "
                   f"(in {v_in:.0f}, out {v_out:.0f} px/s)")
            if e > restitution_max:
                violations.append({
                    "type": "energy_gain", "pair": list(ep["pair"]),
                    "label": label, "t": ep["t_start"],
                    "frame": int(ep["f_start"]), "restitution": round(e, 3),
                    "score": round(min(1.0, (e - 1.0) / 1.0), 3), "confirmed": True,
                })
                yield {"type": "log", "level": "warn",
                       "text": f"ENERGY GAIN — {msg} > max {restitution_max:g}."}
            else:
                yield {"type": "log", "level": "info", "text": f"Restitution OK — {msg}."}

    # 2c. Phantom bounces: reversal with no contact, no neighbor, not at floor.
    for label, k in kin.items():
        vy, fis = k["vy"], k["fis"]
        for i in range(1, len(vy)):
            if not (vy[i - 1] > v_floor and vy[i] < -v_floor):
                continue
            fi = fis[i]
            t = fi / eff_fps
            near_contact = any(ep["f_start"] - 3 * grid_step <= fi <= ep["f_end"] + 3 * grid_step
                               and label in ep["pair"] for ep in all_episodes)
            mask = subjects_masks[label].get(fi)
            near_floor = bool(mask is not None and
                              np.where(mask)[0].max() >= mask.shape[0] * 0.93)
            if near_contact or near_floor:
                continue
            violations.append({
                "type": "phantom_bounce", "label": label, "pair": [label],
                "t": round(t, 3), "frame": int(fi),
                "score": 0.8, "confirmed": True,
            })
            yield {"type": "log", "level": "warn",
                   "text": f"PHANTOM BOUNCE — “{label}” reverses direction @ "
                           f"t={t:.2f}s with no contact, no nearby subject, "
                           "and not at the frame bottom."}

    # ── 3. VLM confirms the worst interpenetration candidates ────────────────
    from tools.vlm_router import ask_vision_json, key_status
    have_api, key_desc = key_status(model, api_key)
    inter = sorted([v for v in violations if v["type"] == "interpenetration"],
                   key=lambda v: -v["score"])[:max_checks]
    if inter and have_api:
        for v in inter:
            la, lb = v["pair"]
            img = _pair_render(frames[v["frame"]],
                               subjects_masks[la][v["frame"]],
                               subjects_masks[lb][v["frame"]],
                               f"{la} x {lb}  t={v['t']:.2f}s")
            if img is None:
                continue
            try:
                parsed = await ask_vision_json(PENETRATION_PROMPT, img, model, api_key)
                verdict = str(parsed.get("verdict", "")).lower()
                conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0, 1))
                expl = str(parsed.get("explanation", ""))[:300]
                v["confirmed"] = (verdict == "interpenetration")
                v["vlm_confidence"] = round(conf, 3)
                v["explanation"] = expl
                if v["confirmed"]:
                    v["score"] = round(max(v["score"], conf), 3)
                    yield {"type": "log", "level": "warn",
                           "text": f"VLM CONFIRMS interpenetration “{la}” × “{lb}” "
                                   f"@ t={v['t']:.2f}s ({conf:.0%}): {expl}"}
                else:
                    v["score"] = round(v["score"] * 0.25, 3)
                    yield {"type": "log", "level": "info",
                           "text": f"VLM: normal contact “{la}” × “{lb}” "
                                   f"@ t={v['t']:.2f}s ({conf:.0%}): {expl}"}
                yield _jpeg_event(img, f"“{la}” × “{lb}” @ t={v['t']:.2f}s — "
                                       f"{verdict} ({conf:.0%}) per {model}")
            except Exception as exc:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"VLM check failed @ t={v['t']:.2f}s: {str(exc)[:140]} "
                               "— keeping unverified overlap score."}
            await asyncio.sleep(0)
    elif inter:
        yield {"type": "log", "level": "warn",
               "text": f"No VLM credentials ({key_desc}) — interpenetration "
                       "candidates reported unverified."}

    # ── 4. Aggregate: signals, severity, plot, metrics ────────────────────────
    signals = [{"frame": v["frame"], "signal_type": f"collision_{v['type']}",
                "score": v["score"]} for v in violations]
    severity = int(round(100 * max([v["score"] for v in violations], default=0.0)))

    yield {"type": "signal", "source": "s3_collision",
           "source_name": "Collision & Contact",
           "fps": float(eff_fps), "n_frames": int(n), "severity": severity,
           "type_severities": {
               t: int(round(100 * max([v["score"] for v in violations
                                       if v["type"] == t], default=0.0)))
               for t in ("interpenetration", "energy_gain", "phantom_bounce")},
           "signals": signals}

    if pair_overlaps or kin:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                            subplot_titles=("Mask overlap per subject pair "
                                            "(fraction of smaller object)",
                                            "Vertical velocity per subject (px/s, +down)"))
        palette = ["#1a54c4", "#c05621", "#1a7a3c", "#7c3aed", "#be185d", "#0891b2"]
        for i, ((la, lb), ov) in enumerate(pair_overlaps.items()):
            color = palette[i % len(palette)]
            fis = sorted(ov)
            fig.add_trace(go.Scatter(
                x=[fi / eff_fps for fi in fis], y=[ov[fi] for fi in fis],
                mode="lines", name=f"{la} × {lb} (contact)",
                line=dict(color=color, width=1.6)), row=1, col=1)
            raw = raw_overlaps.get((la, lb), {})
            if raw:
                rfis = sorted(raw)
                fig.add_trace(go.Scatter(
                    x=[fi / eff_fps for fi in rfis], y=[raw[fi] for fi in rfis],
                    mode="lines", name=f"{la} × {lb} (raw)",
                    line=dict(color=color, width=1.4, dash="dot")), row=1, col=1)
        fig.add_hline(y=overlap_thresh, row=1, col=1,
                      line=dict(color="#EF9F27", dash="dash", width=1.2))
        fig.add_hline(y=deep_overlap, row=1, col=1,
                      line=dict(color="#E24B4A", dash="dash", width=1.2))
        for i, (label, k) in enumerate(kin.items()):
            fig.add_trace(go.Scatter(
                x=k["t"].tolist(), y=k["vy"].tolist(), mode="lines", name=label,
                line=dict(color=palette[i % len(palette)], width=1.6)), row=2, col=1)
        for v in violations:
            fig.add_vline(x=v["t"], line=dict(color="#E24B4A", dash="dot", width=1.2))
        fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                         showgrid=True, gridcolor="#ebebeb")
        fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
        fig.update_layout(
            title=dict(text="Collision & Contact — Overlap and Kinematics",
                       font=dict(size=15)),
            height=560, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
            margin=dict(l=65, r=40, t=100, b=50),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": fig.to_json(),
               "caption": "Top: pair overlap (orange = contact, red = interpenetration "
                          "threshold). Bottom: vertical velocity; dotted red lines "
                          "mark violations."}

    n_inter = sum(1 for v in violations if v["type"] == "interpenetration"
                  and v["confirmed"] is not False)
    yield {"type": "metric", "label": "Contact episodes", "value": str(len(all_episodes)),
           "sub": f"{len(pair_overlaps)} subject pair(s) analyzed"}
    yield {"type": "metric", "label": "Interpenetrations", "value": str(n_inter),
           "sub": f"of {sum(1 for v in violations if v['type'] == 'interpenetration')} "
                  "candidate(s) after VLM review"}
    yield {"type": "metric", "label": "Energy-gain bounces",
           "value": str(sum(1 for v in violations if v["type"] == "energy_gain")),
           "sub": f"restitution > {restitution_max:g}"}
    yield {"type": "metric", "label": "Phantom bounces",
           "value": str(sum(1 for v in violations if v["type"] == "phantom_bounce")),
           "sub": "direction reversal with no contact"}

    yield {"type": "severity", "label": "Collision violation",
           "value": severity, "color": _sev_color(severity)}

    yield {"type": "result", "status": "ok",
           "subjects": list(subjects_masks.keys()),
           "episodes": [dict(ep, pair=list(ep["pair"])) for ep in all_episodes],
           "violations": violations, "severity": severity}
    EVIDENCE.put(vid, "s3_collision", {
        "severity": severity, "violations": violations,
        "episodes": [{"pair": list(ep["pair"]), "t_start": ep["t_start"],
                      "t_end": ep["t_end"], "peak_overlap": ep["peak_overlap"]}
                     for ep in all_episodes],
        "signals": signals,
    })

    if severity > 30:
        yield {"type": "log", "level": "warn",
               "text": f"Collision physics VIOLATION — severity {severity}%."}
    else:
        yield {"type": "log", "level": "success",
               "text": "Collision physics look plausible."}
    yield {"type": "done"}

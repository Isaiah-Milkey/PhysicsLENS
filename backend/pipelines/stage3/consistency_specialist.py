"""
Stage 3 · Specialist — Object Consistency
-------------------------------------------
Adjudicates object-permanence violations per subject: does each masked object
morph in appearance, or vanish/reappear, over the clip?

Detection vs explanation
------------------------
DINOv2 embeddings DETECT; the VLM EXPLAINS. Per subject (masks come from the
Stage 2 Object Tracker's evidence, or inline segmentation as fallback):

1. Every usable masked crop is DINOv2-embedded. Two drift signals:
     * consecutive-frame cosine distance  → sharp events (fragmentation, swap)
     * distance to the first appearance   → gradual morphs that never spike
   Quantitative and reproducible — no language-model part-counting.
2. Each detected change-point gets ONE VLM check: a before/after crop pair
   (background dimmed) with a structure-comparison prompt.
3. Vanish candidates come free from gaps in the mask presence timeline.

Framing: every crop uses a FIXED per-subject viewport (median mask box size,
anchored on the mask centroid) so bbox blow-ups from fragmentation or mask
noise cannot change the object's apparent scale between tiles.

Embedding crops are hard-masked (background variation must not pollute
drift); VLM crops are dimmed (occlusion context preserved).
"""
import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video import load_frames
from tools.evidence import EVIDENCE, video_id
from tools.vlm import parse_vlm_json

VERDICT_WEIGHTS = {"vanished": 1.0, "changed": 0.85, "consistent": 0.0}

PAIR_PROMPT = (
    "The image shows the SAME tracked object at two moments of one video: "
    "LEFT is BEFORE, RIGHT is AFTER (timestamps printed on the image). The "
    "object is at full brightness; everything else is DIMMED (~4x darker). "
    "Both crops use the same fixed viewport, so scale is comparable.\n"
    "An automated appearance metric flagged this moment as a possible change "
    "— your job is to verify it and say what differs.\n"
    "IMPORTANT: the camera may move, so position, sharpness, or blur "
    "differences are NOT evidence of change. Dimmed regions are other "
    "objects/background — a part hidden behind a dimmed occluder or seen "
    "through openings is HIDDEN, not missing. Other objects entering, "
    "touching, or passing in front/behind are normal scene dynamics.\n"
    "- 'consistent': same object structure; pose/angle/scale changes are fine.\n"
    "- 'changed': its OWN shape, identity, texture, or structure differs in a "
    "physically impossible way (AI-generation artifact).\n"
    "- 'vanished': missing or unrecognizable in the AFTER crop.\n"
    "Reply with ONLY strict JSON: "
    '{"verdict": "consistent"|"changed"|"vanished", "confidence": <0..1>, '
    '"explanation": "<one sentence naming the specific difference>"}'
)


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _jpeg_event(img: np.ndarray, caption: str, quality: int = 90) -> dict:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return {"type": "image", "mime": "image/jpeg",
            "data": base64.b64encode(buf).decode(), "caption": caption}


def _full_mask(frame_shape: tuple, mask_small: np.ndarray) -> np.ndarray:
    H, W = frame_shape[:2]
    return cv2.resize(mask_small.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _viewport_size(frames_shape: tuple, masks: dict, good: list[int],
                   pad_frac: float = 0.30) -> tuple[int, int]:
    """One fixed window size per subject: median mask box dims + padding."""
    H, W = frames_shape[:2]
    ws, hs = [], []
    sh, sw = next(iter(masks.values())).shape
    sy, sx = H / sh, W / sw
    for fi in good:
        ys, xs = np.where(masks[fi])
        if xs.size:
            ws.append((xs.max() - xs.min()) * sx)
            hs.append((ys.max() - ys.min()) * sy)
    if not ws:
        return W, H
    vw = int(np.median(ws) * (1 + 2 * pad_frac))
    vh = int(np.median(hs) * (1 + 2 * pad_frac))
    return max(48, min(W, vw)), max(48, min(H, vh))


def _viewport_crop(frame: np.ndarray, mask_small: np.ndarray,
                   view: tuple[int, int], dim: bool) -> Optional[np.ndarray]:
    """Fixed-size window anchored on the mask centroid; background dimmed
    (dim=True, for the VLM) or blacked out (dim=False, for embeddings)."""
    H, W = frame.shape[:2]
    mask = _full_mask(frame.shape, mask_small)
    ys, xs = np.where(mask)
    if xs.size < 16:
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    vw, vh = view
    x0 = max(0, min(W - vw, cx - vw // 2))
    y0 = max(0, min(H - vh, cy - vh // 2))
    crop = frame[y0:y0 + vh, x0:x0 + vw].copy()
    m = mask[y0:y0 + vh, x0:x0 + vw]
    if dim:
        crop[~m] //= 4
    else:
        crop[~m] = 0
    return crop


def _labeled_tile(crop: np.ndarray, label: str, tile_h: int = 380) -> np.ndarray:
    s = tile_h / crop.shape[0]
    crop = cv2.resize(crop, (max(2, int(crop.shape[1] * s)), tile_h))
    band = np.full((30, crop.shape[1], 3), 20, np.uint8)
    cv2.putText(band, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([band, crop], axis=0)


def _tile_row(tiles: list[np.ndarray]) -> Optional[np.ndarray]:
    if len(tiles) < 2:
        return None
    gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, gap, t], axis=1)
    return row


def _embed_crops(crops: list[np.ndarray]) -> np.ndarray:
    """L2-normalised DINOv2 descriptors, (K, D)."""
    from tools.embeddings import load_dinov2, embed_frames_dinov2
    embs = embed_frames_dinov2(crops, load_dinov2())
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms < 1e-8, 1.0, norms)


async def _judge_pair(pair_img: np.ndarray, model: str) -> dict:
    from tools.createai import query_vision, response_text
    resp = await query_vision(PAIR_PROMPT, pair_img, model=model)
    parsed = parse_vlm_json(response_text(resp) or json.dumps(resp))
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in VERDICT_WEIGHTS:
        raise ValueError(f"unparseable verdict: {str(parsed)[:120]}")
    conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0.0, 1.0))
    return {"verdict": verdict, "confidence": conf,
            "explanation": str(parsed.get("explanation", ""))[:300]}


async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    model        = str(cfg.get("model", "geminiflash2_5"))
    max_subjects = max(1, int(cfg.get("max_subjects", 3)))
    max_checks   = max(1, int(cfg.get("max_checks", 4)))
    min_gap_s    = max(0.05, float(cfg.get("min_vanish_gap_s", 0.3)))
    strip_tiles  = max(3, min(10, int(cfg.get("strip_tiles", 6))))
    drift_thresh = max(0.05, float(cfg.get("drift_threshold", 0.30)))

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    # ── 1. Subject masks: evidence bus first, inline segmentation fallback ────
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
            from tools.createai import name_subjects
            from tools.sam3 import segment_concepts
            subjects = await name_subjects([frames[0], frames[n // 2]],
                                           max_subjects=max_subjects, model=model)
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
    if not subjects_masks:
        yield {"type": "error", "text": "No subject masks available."}
        return
    grid_step = (int(np.median(np.diff(sampled_frames)))
                 if len(sampled_frames) > 2 else 1)

    from tools.sam3 import usable_frames

    # ── 2. Per subject: drift detection → targeted VLM explanation ────────────
    verdicts: list[dict] = []
    vanish_events: list[dict] = []
    drift_curves: dict[str, dict] = {}          # label → {t, dcons, dref}
    from tools.createai import credentials
    have_api = all(credentials())
    if not have_api:
        yield {"type": "log", "level": "warn",
               "text": "CreateAI credentials missing — drift change-points will be "
                       "reported unverified (no VLM explanation)."}

    for label, masks in subjects_masks.items():
        present = sorted(masks)
        if len(present) < 2:
            continue

        # 2a. Vanish gaps in the presence timeline (grid-aware)
        gap_min = max(int(round(min_gap_s * eff_fps)), 2 * grid_step)
        prev = present[0]
        for fi in present[1:]:
            if fi - prev > gap_min:
                gap_s = (fi - prev) / eff_fps
                vanish_events.append({
                    "label": label,
                    "t_start": round((prev + 1) / eff_fps, 3),
                    "t_end": round((fi - 1) / eff_fps, 3),
                    "gap_s": round(gap_s, 3),
                    "score": round(min(1.0, gap_s / 0.5), 3),
                })
            prev = fi

        # 2b. Fixed viewport + DINOv2 drift curves
        good = usable_frames(masks)
        if len(good) < 3:
            yield {"type": "log", "level": "warn",
                   "text": f"“{label}”: too few usable masks for drift analysis."}
            continue
        view = _viewport_size(frames[0].shape, masks, good)

        embed_crops, embed_fis = [], []
        for fi in good:
            c = _viewport_crop(frames[fi], masks[fi], view, dim=False)
            if c is not None:
                embed_crops.append(c)
                embed_fis.append(fi)
        if len(embed_crops) < 3:
            continue
        try:
            embs = await loop.run_in_executor(None, _embed_crops, embed_crops)
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"DINOv2 unavailable ({str(exc)[:140]}) — skipping "
                           f"drift analysis for “{label}”."}
            continue
        dcons = np.concatenate([[0.0], 1.0 - np.sum(embs[1:] * embs[:-1], axis=1)])
        dref  = 1.0 - embs @ embs[0]
        drift_curves[label] = {"t": [fi / eff_fps for fi in embed_fis],
                               "dcons": dcons.tolist(), "dref": dref.tolist()}

        # Change-points: consecutive spikes; plus worst ref-drift if gradual.
        spike_is = [int(i) for i in np.where(dcons > drift_thresh)[0]]
        events = [("spike", i) for i in spike_is]
        if not spike_is and float(dref.max()) > drift_thresh:
            events.append(("gradual", int(dref.argmax())))
        if not events:
            yield {"type": "log", "level": "info",
                   "text": f"“{label}”: appearance stable (peak consecutive drift "
                           f"{dcons.max():.3f}, vs first {dref.max():.3f}, "
                           f"threshold {drift_thresh:g})."}
        else:
            yield {"type": "log", "level": "warn",
                   "text": f"“{label}”: {len(events)} appearance change-point(s) "
                           f"detected (peak {max(dcons.max(), dref.max()):.3f} > "
                           f"{drift_thresh:g})."}
        if len(events) > max_checks:
            keep = np.linspace(0, len(events) - 1, max_checks, dtype=int)
            events = [events[i] for i in keep]

        # 2c. VLM explains each change-point (before/after pair)
        for kind, i in events:
            if kind == "spike":
                i0, i1 = max(0, i - 1), i
            else:                                   # gradual: first vs worst
                i0, i1 = 0, i
            f0, f1 = embed_fis[i0], embed_fis[i1]
            t0, t1 = f0 / eff_fps, f1 / eff_fps
            score = float(dcons[i] if kind == "spike" else dref[i])
            base = {"label": label, "kind": kind, "t": round(t1, 3),
                    "drift": round(score, 3)}
            c0 = _viewport_crop(frames[f0], masks[f0], view, dim=True)
            c1 = _viewport_crop(frames[f1], masks[f1], view, dim=True)
            pair = _tile_row([_labeled_tile(c0, f"BEFORE t={t0:.2f}s"),
                              _labeled_tile(c1, f"AFTER t={t1:.2f}s")]) \
                if c0 is not None and c1 is not None else None
            if pair is None:
                continue
            if not have_api:
                verdicts.append({**base, "verdict": "changed",
                                 "confidence": round(min(1.0, score / (2 * drift_thresh)), 3),
                                 "explanation": "unverified drift change-point (no VLM)"})
                yield _jpeg_event(pair, f"“{label}” drift {score:.3f} @ t={t1:.2f}s "
                                        "(unverified — no VLM)")
                continue
            try:
                v = {**base, **await _judge_pair(pair, model)}
                verdicts.append(v)
                lvl = "info" if v["verdict"] == "consistent" else "warn"
                yield {"type": "log", "level": lvl,
                       "text": f"“{label}” @ t={t1:.2f}s (drift {score:.3f}) → "
                               f"{v['verdict']} ({v['confidence']:.0%}): {v['explanation']}"}
                yield _jpeg_event(pair, f"“{label}” @ t={t1:.2f}s — {v['verdict']} "
                                        f"({v['confidence']:.0%}) per {model}, "
                                        f"drift {score:.3f}")
            except Exception as exc:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"VLM check “{label}” @ t={t1:.2f}s failed: {str(exc)[:140]}"}
            await asyncio.sleep(0)

        # 2d. Overview strip (display only — verdicts come from the pairs above)
        from tools.sam3 import representative_frames
        tiles = []
        for fi in representative_frames(masks, strip_tiles):
            c = _viewport_crop(frames[fi], masks[fi], view, dim=True)
            if c is not None:
                tiles.append(_labeled_tile(c, f"t={fi / eff_fps:.2f}s"))
        strip = _tile_row(tiles)
        if strip is not None:
            yield _jpeg_event(strip, f"“{label}” over time (fixed viewport, "
                                     "background dimmed) — evidence overview")
        await asyncio.sleep(0)

    if vanish_events:
        yield {"type": "log", "level": "warn",
               "text": f"{len(vanish_events)} presence gap(s) in mask timelines "
                       "(possible vanish/reappear)."}

    # ── 3. Aggregate: signals, severity, metrics, plot ────────────────────────
    flagged = [v for v in verdicts if v["verdict"] != "consistent"]
    signals = (
        [{"frame": int(round(v["t"] * eff_fps)), "signal_type": "object_morph",
          "score": round(v["confidence"], 3)}
         for v in flagged if v["verdict"] == "changed"] +
        [{"frame": int(round(g["t_start"] * eff_fps)),
          "signal_type": "object_vanish", "score": g["score"]}
         for g in vanish_events] +
        [{"frame": int(round(v["t"] * eff_fps)), "signal_type": "object_vanish",
          "score": round(v["confidence"], 3)}
         for v in flagged if v["verdict"] == "vanished"]
    )
    morph_sev  = max((VERDICT_WEIGHTS["changed"] * v["confidence"]
                      for v in flagged if v["verdict"] == "changed"), default=0.0)
    vanish_sev = max([VERDICT_WEIGHTS["vanished"] * v["confidence"]
                      for v in flagged if v["verdict"] == "vanished"] +
                     [g["score"] for g in vanish_events] or [0.0])
    severity = int(round(100 * max(morph_sev, vanish_sev)))

    yield {"type": "signal", "source": "s3_consistency",
           "source_name": "Object Consistency",
           "fps": float(eff_fps), "n_frames": int(n), "severity": severity,
           "type_severities": {"object_morph": int(round(100 * morph_sev)),
                               "object_vanish": int(round(100 * vanish_sev))},
           "signals": signals}

    # Drift curves per subject + presence gaps.
    if drift_curves:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                            subplot_titles=("Consecutive-frame appearance drift",
                                            "Drift vs first appearance"))
        palette = ["#1a54c4", "#c05621", "#1a7a3c", "#7c3aed", "#be185d", "#0891b2"]
        for i, (label, d) in enumerate(drift_curves.items()):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(x=d["t"], y=d["dcons"], mode="lines",
                                     name=label, legendgroup=label,
                                     line=dict(color=color, width=1.6)), row=1, col=1)
            fig.add_trace(go.Scatter(x=d["t"], y=d["dref"], mode="lines",
                                     name=label, legendgroup=label, showlegend=False,
                                     line=dict(color=color, width=1.6)), row=2, col=1)
        for r in (1, 2):
            fig.add_hline(y=drift_thresh, row=r, col=1,
                          line=dict(color="#E24B4A", dash="dash", width=1.2))
        fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                         showgrid=True, gridcolor="#ebebeb")
        fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
        fig.update_layout(
            title=dict(text="Object Consistency — DINOv2 Appearance Drift", font=dict(size=15)),
            height=520, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
            margin=dict(l=65, r=40, t=100, b=50),
            font=dict(family="IBM Plex Sans, sans-serif", size=13),
        )
        yield {"type": "plotly", "data": fig.to_json(),
               "caption": "Spikes above the red threshold are detected change-points "
                          "(each verified by the VLM); the bottom curve catches "
                          "gradual morphs."}

    n_changed = sum(1 for v in flagged if v["verdict"] == "changed")
    for label in subjects_masks:
        subj_flags = [v for v in flagged if v["label"] == label]
        worst = max(subj_flags, key=lambda v: v["confidence"], default=None)
        yield {"type": "metric", "label": f"“{label}”",
               "value": worst["verdict"].upper() if worst else "CONSISTENT",
               "sub": (f"{worst['confidence']:.0%} @ t={worst['t']}s ({model})"
                       if worst else "no change-points confirmed")}
    yield {"type": "metric", "label": "Change-points checked", "value": str(len(verdicts)),
           "sub": f"{n_changed} confirmed changed · drift threshold {drift_thresh:g}"}
    yield {"type": "metric", "label": "Vanish events", "value": str(len(vanish_events)),
           "sub": f"mask gaps ≥ {min_gap_s:g}s (grid-aware)"}

    yield {"type": "severity", "label": "Object consistency violation",
           "value": severity, "color": _sev_color(severity)}

    yield {"type": "result", "status": "ok",
           "subjects": list(subjects_masks.keys()),
           "verdicts": verdicts, "vanish_events": vanish_events,
           "severity": severity}
    EVIDENCE.put(vid, "s3_consistency", {
        "severity": severity, "verdicts": verdicts,
        "vanish_events": vanish_events, "signals": signals,
    })

    if severity > 30:
        yield {"type": "log", "level": "warn",
               "text": f"Object consistency VIOLATION — severity {severity}%."}
    else:
        yield {"type": "log", "level": "success",
               "text": "Objects appear consistent across the clip."}
    yield {"type": "done"}

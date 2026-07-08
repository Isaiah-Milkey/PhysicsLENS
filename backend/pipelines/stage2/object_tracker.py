"""
Stage 2 · Step 1 — Object Tracker
------------------------------------
Detect salient moving objects with Shi-Tomasi corners + Lucas-Kanade tracking.
Cluster nearby keypoints into persistent object tracks.
Optionally embed each track's bounding-box crops with DINOv2 (from archive
sam_track_compare.py) to measure appearance drift — an identity-stability
biomarker.  A physically consistent object keeps a near-flat drift curve;
flickering / morphing AI objects spike.

Metrics
-------
  • Tracked objects count
  • Keypoint loss rate
  • Mean / peak track duration
  • Per-track mean & peak cosine drift  (DINOv2)
  • Overall tracking instability score  (0–100)

GPU not required — falls back gracefully if DINOv2 / torch are absent.
"""
import asyncio, base64, json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.video    import encode_video_browser
from tools.tracking import get_tracks, _PALETTE, _hex_to_bgr
from tools.evidence import EVIDENCE, video_id


def _render_annotated(
    frames: list, obj_tracks: list[dict], max_frames: int = 240
) -> tuple[list, int]:
    """Draw each track's box + label (legend colors) on every frame.
    Uses the track's semantic label (LocateAnything) if present, else 'obj N'.
    Returns (annotated frames, subsample step)."""
    by_frame: dict[int, list] = {}
    for ct in obj_tracks:
        for f, box in zip(ct["frames"], ct["boxes"]):
            by_frame.setdefault(f, []).append((ct["id"], ct.get("label"), box))

    n    = len(frames)
    step = max(1, -(-n // max_frames))   # ceil division
    out  = []
    for f in range(0, n, step):
        img = frames[f].copy()
        for tid, tlabel, (x0, y0, x1, y1) in by_frame.get(f, []):
            color = _hex_to_bgr(_PALETTE[tid % len(_PALETTE)])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            label = f"obj {tid} ({tlabel})" if tlabel else f"obj {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = y0 - 6 if y0 - th - 10 >= 0 else min(y1 + th + 6, img.shape[0] - 4)
            cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), color, -1)
            cv2.putText(img, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(img)
    return out, step


# ── Motion hotspot (frame differencing — near-free) ──────────────────────────

def _motion_hotspot(frames: list, max_pairs: int = 12) -> Optional[dict]:
    """Locate the region with the most motion across the clip.

    Accumulated |gray frame diff| over ≤max_pairs evenly spaced frame pairs,
    blurred, thresholded, largest contour. Returns {"box": XYXY (original px),
    "peak_frame": idx of the pair with the strongest motion} or None.
    Pure numpy/cv2 — no models, no API.
    """
    n = len(frames)
    if n < 3:
        return None
    idxs = np.linspace(0, n - 2, min(max_pairs, n - 1), dtype=int)
    H, W = frames[0].shape[:2]
    scale = min(1.0, 360.0 / H)
    size = (max(2, int(W * scale)), max(2, int(H * scale)))

    accum = np.zeros(size[::-1], np.float32)
    peak_frame, peak_energy = int(idxs[0]), -1.0
    for i in idxs:
        a = cv2.cvtColor(cv2.resize(frames[i], size), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(frames[i + 1], size), cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(a, b).astype(np.float32)
        accum += d
        e = float(d.sum())
        if e > peak_energy:
            peak_energy, peak_frame = e, int(i)
    if accum.max() < 1e-3:
        return None

    norm = (accum / accum.max() * 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (9, 9), 0)
    _, mask = cv2.threshold(norm, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w * h < mask.size * 0.001:
        return None
    inv = 1.0 / scale
    return {"box": (int(x * inv), int(y * inv),
                    int((x + w) * inv), int((y + h) * inv)),
            "peak_frame": peak_frame}


# ── DINOv2 appearance drift (ported from sam_track_compare.py) ───────────────

def _embed_crops_dinov2(crops: list[np.ndarray], model=None) -> np.ndarray:
    """
    L2-normalised DINOv2 (facebook/dinov2-base) descriptors for a list of
    BGR uint8 crop arrays.  Ported from sam_track_compare._embed_track —
    same checkpoint, so the 0.35 drift threshold keeps its original meaning.
    Returns (K, 768) float32.  The loader is cached, so per-track calls
    reuse one model instance.
    """
    from tools.embeddings import load_dinov2, embed_frames_dinov2
    if model is None:
        model = load_dinov2()
    embs  = embed_frames_dinov2(crops, model)       # (K, D)
    # Ensure L2 normalisation (safe double-norm)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.where(norms < 1e-8, 1.0, norms)
    return embs


def _drift_curve(embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine-distance drift relative to the first embedding.
    Ported directly from sam_track_compare._drift_curve:
      drift[t] = 1 − <e_t , e_0>  ∈ [0, 2]
    Flat curve = stable identity.  Spike = morphing / appearance jump.
    """
    ref = embeddings[0]
    return 1.0 - embeddings @ ref


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg          = json.loads(settings) if settings else {}
    num_kp       = max(5,  int(cfg.get("num_keypoints", 60)))
    sample_every = max(1,  int(cfg.get("sample_every",   1)))
    use_dinov2   = str(cfg.get("use_dinov2", "true")).lower() not in ("false", "0", "no")
    render_video = str(cfg.get("render_video", "true")).lower() not in ("false", "0", "no")
    use_locate_anything = str(cfg.get("use_locate_anything", "true")).lower() not in ("false", "0", "no")
    tracker_mode = str(cfg.get("tracker_mode", "auto")).lower()          # auto | sam3 | lk
    max_subjects = max(1, int(cfg.get("max_subjects", 3)))
    naming_model = str(cfg.get("naming_model", "geminiflash2_5"))

    loop = asyncio.get_event_loop()

    # ── Phase 1 (primary): segmentation-first subject tracking ────────────────
    # Gemini names the primary subjects from the first frame; SAM3 mask-tracks
    # each named subject. Tracks come out in the same schema as the LK path so
    # every downstream consumer (drift, plots, evidence bus) works unchanged.
    obj_tracks: list[dict] = []
    masks_by_track: dict[int, dict[int, np.ndarray]] = {}   # track id → {frame: mask}
    mask_scale = 1.0
    seg_mode = False

    if tracker_mode in ("auto", "sam3"):
        try:
            from tools.video import load_frames
            from tools.createai import name_subjects, credentials
            from tools.sam3 import segment_concepts, mask_box

            if not all(credentials()):
                raise RuntimeError("CreateAI credentials missing (needed to name subjects)")

            yield {"type": "log", "level": "info", "text": "Loading video…"}
            await asyncio.sleep(0)
            frames, raw_fps = await loop.run_in_executor(
                None, lambda: load_frames(video_path, step=sample_every))
            eff_fps = (raw_fps or 30.0) / sample_every
            n = len(frames)
            if n < 3:
                yield {"type": "error", "text": f"Video too short ({n} frames)."}
                return
            H, W = frames[0].shape[:2]

            # Motion hotspot (frame differencing, near-free): guarantees the
            # fastest-moving object gets named and, below, gets tracked even
            # if SAM3 misses it by name.
            hotspot = await loop.run_in_executor(None, _motion_hotspot, frames)
            motion_crop = None
            if hotspot:
                hx0, hy0, hx1, hy1 = hotspot["box"]
                pf = hotspot["peak_frame"]
                px, py = max(16, (hx1 - hx0) // 6), max(16, (hy1 - hy0) // 6)
                motion_crop = frames[pf][max(0, hy0 - py):hy1 + py,
                                         max(0, hx0 - px):hx1 + px].copy()
                yield {"type": "log", "level": "info",
                       "text": f"Motion hotspot: box {hotspot['box']} "
                               f"(peak around frame {pf})."}

            yield {"type": "log", "level": "info",
                   "text": "Naming primary subjects (Gemini via CreateAI, "
                           "first + middle frame + motion crop)…"}
            subjects = await name_subjects([frames[0], frames[n // 2]],
                                           max_subjects=max_subjects,
                                           model=naming_model,
                                           motion_crop=motion_crop)
            if not subjects:
                raise RuntimeError("VLM returned no subject names")
            yield {"type": "log", "level": "info",
                   "text": "Primary subjects: " + ", ".join(f"“{s}”" for s in subjects)}
            await asyncio.sleep(0)

            yield {"type": "log", "level": "info",
                   "text": f"SAM3 mask-tracking {len(subjects)} subject(s) through the clip…"}
            await asyncio.sleep(0)
            seg = await loop.run_in_executor(
                None, lambda: segment_concepts(frames, subjects))
            mask_scale = seg["scale"]
            sampled_frames = seg["sampled_frames"]      # SAM3's uniform subsample grid
            n_sampled = len(sampled_frames)
            if not seg["subjects"]:
                raise RuntimeError("SAM3 matched none of the named subjects")

            inv = 1.0 / mask_scale
            tid = 0
            for name, masks in seg["subjects"].items():
                f_idxs, boxes, cx, cy = [], [], [], []
                for fi in sorted(masks):
                    b = mask_box(masks[fi], pad_frac=0.0)
                    if b is None:
                        continue
                    x0, y0, x1, y1 = (int(v * inv) for v in b)
                    f_idxs.append(fi)
                    boxes.append((x0, y0, x1, y1))
                    cx.append((x0 + x1) / 2.0)
                    cy.append((y0 + y1) / 2.0)
                if len(f_idxs) < 3:
                    continue
                obj_tracks.append({"id": tid, "frames": f_idxs, "boxes": boxes,
                                   "cx": cx, "cy": cy, "n_kp": 0, "label": name})
                masks_by_track[tid] = masks
                tid += 1
            if not obj_tracks:
                raise RuntimeError("no subject produced a usable mask track")

            # Motion rescue: if no subject's mask covers the motion hotspot,
            # the fastest-moving object slipped through naming — box-prompt
            # SAM3 on the hotspot itself so it gets tracked regardless.
            if hotspot:
                bx0, by0, bx1, by1 = [int(v * mask_scale) for v in hotspot["box"]]
                pf = hotspot["peak_frame"]
                covered = False
                for masks in masks_by_track.values():
                    for fi in sorted(masks, key=lambda f: abs(f - pf))[:3]:
                        region = masks[fi][max(0, by0):max(1, by1),
                                           max(0, bx0):max(1, bx1)]
                        if region.size and float(region.mean()) > 0.05:
                            covered = True
                            break
                    if covered:
                        break
                if not covered:
                    yield {"type": "log", "level": "warn",
                           "text": "No subject mask covers the motion hotspot — "
                                   "box-prompting SAM3 on it directly (motion rescue)."}
                    await asyncio.sleep(0)
                    try:
                        from tools.sam3 import segment_video
                        res = await loop.run_in_executor(
                            None, lambda: segment_video(
                                frames, box=hotspot["box"],
                                text="the main moving object",
                                box_frame=hotspot["peak_frame"]))
                        inv_r = 1.0 / res["scale"]
                        f_idxs, boxes, cx, cy = [], [], [], []
                        for fi in sorted(res["masks"]):
                            b = mask_box(res["masks"][fi], pad_frac=0.0)
                            if b is None:
                                continue
                            x0, y0, x1, y1 = (int(v * inv_r) for v in b)
                            f_idxs.append(fi)
                            boxes.append((x0, y0, x1, y1))
                            cx.append((x0 + x1) / 2.0)
                            cy.append((y0 + y1) / 2.0)
                        if len(f_idxs) >= 3:
                            rid = len(obj_tracks)
                            obj_tracks.append({"id": rid, "frames": f_idxs,
                                               "boxes": boxes, "cx": cx, "cy": cy,
                                               "n_kp": 0, "label": "moving object"})
                            masks_by_track[rid] = res["masks"]
                            yield {"type": "log", "level": "info",
                                   "text": f"Motion rescue tracked “moving object” on "
                                           f"{len(f_idxs)} frame(s) "
                                           f"(prompt mode: {res['prompt_mode']})."}
                        else:
                            yield {"type": "log", "level": "warn",
                                   "text": "Motion rescue found no stable object."}
                    except Exception as exc:                        # noqa: BLE001
                        yield {"type": "log", "level": "warn",
                               "text": f"Motion rescue failed: {str(exc)[:160]}"}

            seg_mode = True
            meta = {"n_frames": n, "H": H, "W": W, "start": 0,
                    "nk": 0, "kp_loss_pct": 0.0}
            nk, kp_loss_pct, kp_lost = 0, 0.0, 0
            for ct in obj_tracks:
                yield {"type": "log", "level": "info",
                       "text": f"“{ct['label']}” masked on {len(ct['frames'])}/{n_sampled} "
                               "sampled frame(s)."}
        except Exception as exc:                                    # noqa: BLE001
            if tracker_mode == "sam3":
                yield {"type": "error",
                       "text": f"Segmentation mode failed: {str(exc)[:300]}"}
                return
            yield {"type": "log", "level": "warn",
                   "text": f"Segmentation path unavailable ({str(exc)[:160]}) — "
                           "falling back to LK keypoint tracking."}
            obj_tracks, masks_by_track, seg_mode = [], {}, False

    # ── Phase 1 (fallback): shared, cached LK tracking + clustering ───────────
    if not seg_mode:
        yield {"type": "log", "level": "info", "text": "Loading video & tracking keypoints…"}
        await asyncio.sleep(0)
        tr = await loop.run_in_executor(
            None, lambda: get_tracks(video_path, num_kp=num_kp,
                                     sample_every=sample_every, max_objects=8))
        frames     = tr["frames"]
        eff_fps    = tr["fps"]
        meta       = tr["meta"]
        obj_tracks = tr["tracks"]
        n  = meta["n_frames"]
        H, W = meta["H"], meta["W"]
        nk = meta["nk"]
        kp_loss_pct = meta["kp_loss_pct"]
        kp_lost     = int(round(kp_loss_pct * nk))

        if n < 3:
            yield {"type": "error", "text": f"Video too short ({n} frames)."}
            return
        if not obj_tracks:
            yield {"type": "error", "text": "No keypoints detected in any frame."}
            return
        if meta["start"] > 0:
            yield {"type": "log", "level": "warn",
                   "text": f"First {meta['start']} frame(s) featureless — tracking started later."}

    n_objects = len(obj_tracks)
    yield {"type": "log", "level": "info",
           "text": f"{n} frames @ {eff_fps:.1f} fps — {n_objects} object track(s) "
                   + ("via SAM3 subject masks." if seg_mode else f"from {nk} keypoints.")}
    await asyncio.sleep(0)

    # ── Phase 1b: Semantic labeling (NVIDIA LocateAnything-3B, one-shot) ──────
    # LK-fallback mode only — in segmentation mode subjects are already named.
    # Single-image open-set detector — run once on the first tracked frame and
    # match its grounded boxes onto the LK tracks by IOU. Doesn't change what
    # gets tracked, only what it's called. Degrades gracefully (no GPU/model).
    if use_locate_anything and not seg_mode:
        yield {"type": "log", "level": "info",
               "text": "Running LocateAnything-3B for semantic object labels…"}
        await asyncio.sleep(0)
        try:
            from tools.locate_anything import detect, match_label
            det_frame_idx = meta["start"]
            detections = await loop.run_in_executor(
                None, lambda: detect(frames[det_frame_idx]))
            labeled = 0
            for ct in obj_tracks:
                if det_frame_idx in ct["frames"]:
                    fi = ct["frames"].index(det_frame_idx)
                    box = ct["boxes"][fi]
                else:
                    box = ct["boxes"][0]                # nearest available box
                label = match_label(box, detections)
                if label:
                    ct["label"] = label
                    labeled += 1
            yield {"type": "log", "level": "info",
                   "text": f"{len(detections)} object(s) detected; "
                           f"{labeled}/{n_objects} track(s) labeled."}
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"LocateAnything-3B unavailable ({str(exc)[:180]}) — "
                           "tracks stay unlabeled (obj N)."}
        await asyncio.sleep(0)

    # Publish canonical tracks — plus, in segmentation mode, the per-subject
    # masks (PNG-compressed, a few KB each) — to the evidence bus. Stage 3's
    # Consistency Specialist consumes the masks for its VLM judgment; the
    # tracker itself only segments, tracks, and labels.
    masks_png: dict = {}
    if seg_mode:
        from tools.sam3 import encode_mask_png
        for ct in obj_tracks:
            masks_png[ct["label"]] = {
                int(fi): encode_mask_png(mk)
                for fi, mk in masks_by_track[ct["id"]].items()}
    EVIDENCE.put(video_id(video_path), "s2_object_tracker", {
        "fps": float(eff_fps), "n_frames": int(n),
        "mode": "sam3" if seg_mode else "lk",
        "mask_scale": mask_scale,
        "sampled_frames": (sampled_frames if seg_mode else None),
        "masks_png": masks_png,
        "tracks": [{"id": ct["id"], "frames": ct["frames"], "boxes": ct["boxes"],
                    "cx": ct["cx"], "cy": ct["cy"], "n_kp": ct["n_kp"],
                    "label": ct.get("label")}
                   for ct in obj_tracks],
    })

    # ── Phase 2b: Annotated video — show which region each "obj N" is ─────────
    if render_video and n_objects > 0:
        yield {"type": "log", "level": "info", "text": "Rendering labeled object video…"}
        await asyncio.sleep(0)
        try:
            ann_frames, step = _render_annotated(frames, obj_tracks)
            data, mime = await loop.run_in_executor(
                None, encode_video_browser, ann_frames, eff_fps / step
            )
            yield {
                "type": "video",
                "data": base64.b64encode(data).decode(),
                "mime": mime,
                "caption": ("Tracked objects — box colors and labels match the "
                            "plot legend (obj N below)."),
            }
            yield {"type": "log", "level": "info",
                   "text": f"Labeled video ready ({len(data)/1024:.0f} KB, {mime})."}
        except Exception as exc:
            yield {"type": "log", "level": "warn",
                   "text": f"Labeled video rendering failed ({exc}) — continuing."}
        await asyncio.sleep(0)

    # ── Phase 3: DINOv2 appearance drift (ported from sam_track_compare) ──────
    dino_drifts: dict[int, np.ndarray] = {}
    dino_available = False

    if use_dinov2 and n_objects > 0:
        yield {"type": "log", "level": "info",
               "text": "Loading DINOv2 for appearance-drift analysis (sam_track_compare approach)…"}
        await asyncio.sleep(0)
        try:
            from tools.embeddings import load_dinov2
            dino_model = load_dinov2()
            for ct in obj_tracks:
                crops = []
                for f, box in zip(ct["frames"], ct["boxes"]):
                    x0, y0, x1, y1 = box
                    crop = frames[f][y0:y1, x0:x1]
                    if crop.size > 0:
                        crops.append(crop)

                if len(crops) < 2:
                    continue

                embs  = _embed_crops_dinov2(crops, dino_model)  # (K, D) L2-normed
                drift = _drift_curve(embs)              # (K,)  cosine dist to frame 0
                dino_drifts[ct["id"]] = drift

            dino_available = True
            yield {"type": "log", "level": "info",
                   "text": f"Appearance drift computed for {len(dino_drifts)} track(s)."}
        except Exception as exc:
            yield {"type": "log", "level": "warn",
                   "text": f"DINOv2 unavailable ({exc}) — skipping appearance embedding."}
    await asyncio.sleep(0)

    # ── Phase 4: Plotly visualisation ─────────────────────────────────────────
    yield {"type": "log", "level": "info", "text": "Building plots…"}
    await asyncio.sleep(0)

    n_rows    = 2 if dino_available else 1
    subtitles = ["Object Trajectories (centroid, px)"]
    if dino_available:
        subtitles.append("DINOv2 Appearance Drift per Track — cosine dist to frame 0")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.12, subplot_titles=subtitles,
    )

    for ct in obj_tracks:
        color  = _PALETTE[ct["id"] % len(_PALETTE)]
        t_secs = [f / eff_fps for f in ct["frames"]]

        # Y-centroid trajectory (inverted: 0=bottom, H=top in pixel space → flip)
        fig.add_trace(go.Scatter(
            x=t_secs,
            y=[H - cy for cy in ct["cy"]],
            mode="lines+markers",
            marker=dict(size=4, color=color),
            line=dict(color=color, width=1.6),
            name=(f"obj {ct['id']} — {ct['label']}" if ct.get("label")
                  else f"obj {ct['id']}  ({ct['n_kp']} kp)"),
            hovertemplate=(
                "<b>t = %{x:.2f}s</b><br>"
                "y-centroid = %{y:.0f}px<br>"
                f"track {ct['id']}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        if dino_available and ct["id"] in dino_drifts:
            drift   = dino_drifts[ct["id"]]
            t_drift = t_secs[:len(drift)]
            # Flag high-drift anomalies
            thresh = 0.35
            fig.add_trace(go.Scatter(
                x=t_drift, y=drift.tolist(),
                mode="lines+markers",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=1.6),
                name=f"obj {ct['id']} drift",
                showlegend=False,
                hovertemplate=(
                    "<b>t = %{x:.2f}s</b><br>"
                    "cosine dist = %{y:.3f}<br>"
                    f"track {ct['id']}"
                    "<extra></extra>"
                ),
            ), row=2, col=1)

            # Highlight anomalous drift segments
            spikes = np.where(drift > thresh)[0]
            for si in spikes:
                if si < len(t_drift):
                    hw = 0.5 / eff_fps
                    fig.add_vrect(
                        x0=t_drift[si] - hw, x1=t_drift[si] + hw,
                        fillcolor=color, opacity=0.12, line_width=0, row=2, col=1,
                    )

    if dino_available:
        fig.add_hline(
            y=0.35, line=dict(color="#E24B4A", dash="dash", width=1.2),
            annotation_text="appearance jump threshold (0.35)",
            annotation_font=dict(color="#E24B4A", size=11),
            annotation_position="top right", row=2, col=1,
        )

    _grid = dict(showgrid=True, gridcolor="#ebebeb", gridwidth=1)
    fig.update_xaxes(**_grid, title_text="Time (s)", row=n_rows, col=1)
    fig.update_xaxes(**_grid)
    fig.update_yaxes(**_grid, zeroline=False)
    fig.update_yaxes(title_text="Y position (px, bottom=0)", row=1, col=1)
    if dino_available:
        fig.update_yaxes(title_text="Cosine distance to frame 0", range=[0, None], row=2, col=1)

    fig.update_layout(
        title=dict(text="Object Tracker — LK Trajectories & DINOv2 Appearance Drift",
                   font=dict(size=15)),
        height=380 + (260 if dino_available else 0),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=65, r=40, t=110, b=55),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        hovermode="x unified",
    )

    yield {
        "type": "plotly", "data": fig.to_json(),
        "caption": (
            "Y-centroid trajectories of detected object tracks (top). "
            + ("DINOv2 appearance drift — flat = stable, spikes = identity jump (bottom)."
               if dino_available else "")
        ),
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    durations  = [len(ct["frames"]) for ct in obj_tracks]
    mean_dur   = float(np.mean(durations))  if durations  else 0.0
    # Seg-mode tracks live on SAM3's sampling grid, not the full frame count.
    persist_base = n_sampled if seg_mode else n
    persistent = sum(1 for d in durations if d >= persist_base * 0.5)

    if dino_drifts:
        all_drift  = np.concatenate(list(dino_drifts.values()))
        mean_drift = float(all_drift.mean())
        peak_drift = float(all_drift.max())
    else:
        mean_drift = peak_drift = float("nan")

    labeled_names = [ct["label"] for ct in obj_tracks if ct.get("label")]
    yield {"type": "metric", "label": "Objects tracked",   "value": str(n_objects),
           "sub": (f"{persistent} persistent · " + ", ".join(labeled_names)) if labeled_names
                  else f"{persistent} persistent (≥50% of video)"}
    if seg_mode:
        coverage = float(np.mean([len(ct["frames"]) / max(n_sampled, 1)
                                  for ct in obj_tracks]))
        yield {"type": "metric", "label": "Mask coverage", "value": f"{coverage:.0%}",
               "sub": f"mean fraction of {n_sampled} sampled frames each subject is masked on"}
    else:
        yield {"type": "metric", "label": "Keypoint loss",     "value": f"{kp_loss_pct:.0%}",
               "sub": f"{kp_lost} of {nk} keypoints lost"}
    yield {"type": "metric", "label": "Mean track length", "value": f"{mean_dur:.0f}",
           "sub": f"frames (video = {n})"}

    if not np.isnan(mean_drift):
        yield {"type": "metric", "label": "Mean appearance drift", "value": f"{mean_drift:.3f}",
               "sub": "cosine dist to frame 0 (DINOv2)"}
        yield {"type": "metric", "label": "Peak drift",            "value": f"{peak_drift:.3f}",
               "sub": "max identity deviation across all tracks"}

    # Instability score.
    # Seg mode: mask coverage gaps + appearance drift (consistency judgment
    #           itself is Stage 3's job — see the Consistency Specialist).
    # LK mode:  keypoint loss + appearance drift (as before).
    drift_contrib = (min(mean_drift, 0.6) / 0.6 * 60) if not np.isnan(mean_drift) else 30
    if seg_mode:
        loss_contrib = (1.0 - coverage) * 40
    else:
        loss_contrib = kp_loss_pct * 40
    instability   = min(int(drift_contrib + loss_contrib), 100)
    sev_color     = "#E24B4A" if instability > 50 else "#EF9F27" if instability > 25 else "#4CAF50"
    yield {"type": "severity", "label": "Tracking instability score",
           "value": instability, "color": sev_color}

    msg = (
        f"{n_objects} track(s), {persistent} persistent, "
        + (f"{1 - coverage:.0%} mask gaps" if seg_mode else f"{kp_loss_pct:.0%} keypoint loss")
        + (f", mean drift {mean_drift:.3f}" if not np.isnan(mean_drift) else "") + "."
    )
    yield {"type": "log", "level": "success" if instability < 30 else "warn", "text": msg}
    yield {"type": "done"}

"""
Stage 3 · Specialist — Fluid Specialist
-----------------------------------------
Real fluid motion is dense, continuous, and causally tied to what pushes it:
mass is conserved (no water appearing/vanishing), free-flight droplets follow
gravity-consistent parabolic arcs, splashes start when (and only when) an
impactor hits, and the bulk flow field changes gradually rather than
teleporting between states. This specialist tracks many points through the
fluid with dense optical flow and hunts fluid-shaped defects in the result.

1. REGION — the fluid area to seed points in. If the Object Tracker named a
   fluid-like subject ("water", "wave", "splash", "liquid", ...), its SAM3
   masks are reused directly. Otherwise the region falls back to a
   motion-energy mask (accumulated frame-differencing, the same idea as the
   Object Tracker's own motion-hotspot rescue) — one fixed region for the
   whole clip; this is a deliberate simplification, noted where it matters.
2. DENSE FLOW + ADVECTION — Farneback dense flow is computed between each
   pair of analysis frames (downscaled for speed). A grid of points is seeded
   inside the fluid region and *advected* frame-to-frame by the local flow
   vector, re-seeding points that leave the region or the frame so there is
   always a live population — this builds real multi-frame arcs/streaklines,
   not just per-frame flow samples.
3. VIOLATIONS —
     * ballistic_arc: an airborne point (above the estimated resting surface
       line) whose path doesn't fit a downward-curving parabola — real
       droplets in free flight are gravity-ballistic;
     * incompressibility: frame-local flow divergence (relative to the
       region's own flow speed) spikes — a source or sink, i.e. fluid mass
       appearing or vanishing;
     * splash_timing: a burst in the region's mean flow energy (a splash)
       with no impactor contact nearby, or one that precedes its impactor
       (effect before cause);
     * flow_discontinuity: the region's bulk flow vector jumps abruptly with
       no splash/impactor event to explain it.
   Each flagged event gets ONE VLM check (annotated crop) to confirm/reject —
   same detect→explain pattern as the other specialists.
4. VLM REALISM — in addition to the targeted checks above, ONE holistic call
   shows keyframes spanning the clip and asks whether the fluid's overall
   look and motion are physically realistic (catches wrong color/transparency
   /foam/refraction that pure motion math can't).

`viscosity_mode` (water vs. syrup) scales the divergence/discontinuity
tolerances: a viscous fluid is expected to move slower and smoother, so its
thresholds are tighter.
"""
import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.evidence import EVIDENCE, video_id

_FLUID_KEYWORDS = ("water", "wave", "splash", "liquid", "ocean", "sea", "river",
                  "pool", "lake", "rain", "waterfall", "foam", "fluid",
                  "stream", "puddle", "fountain", "wet")

_FLOW_DIM = 480          # working resolution for Farneback (long edge, px)
_MAX_ANALYSIS_FRAMES = 90  # bound compute regardless of source video length

HOLISTIC_PROMPT = (
    "The image shows {n} frames sampled across one video, evenly spaced in "
    "time (timestamps printed on each tile). Judge the LIQUID/FLUID visible "
    "in these frames (water, waves, splashes, pouring, etc.) for overall "
    "visual and physical realism: color, transparency, reflections, foam/"
    "spray texture, how it interacts with objects and surfaces, and whether "
    "its motion (from tile to tile) looks like real fluid dynamics rather "
    "than a rigid or gelatinous approximation.\n"
    "Reply with ONLY strict JSON: "
    '{{"verdict": "realistic"|"unrealistic", "confidence": <0..1>, '
    '"explanation": "<one sentence>"}}'
)

ANOMALY_PROMPT = (
    "One moment of a video showing fluid motion (a region is outlined/"
    "annotated; timestamp printed on the image).\n"
    "An automated fluid-motion metric flagged this: {desc}\n"
    "Verify it. A real physical cause (a visible splash from an impact, "
    "spray following a plausible arc, mixing at a boundary, the camera or "
    "an object leaving/entering frame) makes it 'plausible'. Fluid "
    "appearing/vanishing with no source, defying gravity, or changing "
    "abruptly with nothing to cause it is a 'violation' (AI-generation "
    "artifact).\n"
    "Reply with ONLY strict JSON: "
    '{{"verdict": "plausible"|"violation", "confidence": <0..1>, '
    '"explanation": "<one sentence naming the cause or its absence>"}}'
)


def _sev_color(sev: float) -> str:
    return "#E24B4A" if sev > 60 else "#EF9F27" if sev > 30 else "#4CAF50"


def _jpeg_event(img: np.ndarray, caption: str, quality: int = 88) -> dict:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return {"type": "image", "mime": "image/jpeg",
            "data": base64.b64encode(buf).decode(), "caption": caption}


def _labeled_tile(crop: np.ndarray, label: str, tile_h: int = 340) -> np.ndarray:
    s = tile_h / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (max(2, int(crop.shape[1] * s)), tile_h))
    band = np.full((30, crop.shape[1], 3), 20, np.uint8)
    cv2.putText(band, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([band, crop], axis=0)


def _tile_row(tiles: list) -> Optional[np.ndarray]:
    tiles = [t for t in tiles if t is not None]
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [t if t.shape[0] == h else
             cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    gap = np.full((h, 10, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, gap, t], axis=1)
    return row


def _region_crop(frame: np.ndarray, mask: Optional[np.ndarray] = None,
                 box: Optional[tuple] = None, pad: float = 0.25) -> Optional[np.ndarray]:
    """Crop around a fluid mask/box (outlined), padded for scene context."""
    H, W = frame.shape[:2]
    if box is None and mask is not None:
        ys, xs = np.where(mask)
        if xs.size < 4:
            return None
        box = (xs.min(), ys.min(), xs.max(), ys.max())
    if box is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    img = frame.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 3)
    px, py = int((x1 - x0) * pad) + 16, int((y1 - y0) * pad) + 16
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
    if cx1 - cx0 < 24 or cy1 - cy0 < 24:
        return None
    return img[cy0:cy1, cx0:cx1]


# ── Frame budget ──────────────────────────────────────────────────────────────

def _cap_frames(frames: list, max_n: int) -> tuple[list, list[int]]:
    """Uniformly subsample to ≤max_n frames. Returns (frames, orig_indices)."""
    n = len(frames)
    if n <= max_n:
        return frames, list(range(n))
    idx = [int(i) for i in np.linspace(0, n - 1, max_n)]
    return [frames[i] for i in idx], idx


# ── Region: named fluid subject (Object Tracker masks) or motion-energy ──────

def _find_fluid_label(tracks: list[dict]) -> Optional[str]:
    for tr in tracks:
        label = (tr.get("label") or "").lower()
        if any(kw in label for kw in _FLUID_KEYWORDS):
            return tr.get("label")
    return None


def _motion_energy_mask(frames_small: list) -> np.ndarray:
    """One fixed region mask for the whole clip: accumulated |frame diff|,
    thresholded + closed. A deliberate simplification (like the Object
    Tracker's motion-hotspot) — good enough to seed points in the active
    fluid area even with no named subject."""
    n = len(frames_small)
    idx = np.linspace(0, n - 2, min(24, max(1, n - 1)), dtype=int)
    accum = np.zeros(frames_small[0].shape[:2], np.float32)
    for i in idx:
        a = cv2.cvtColor(frames_small[i], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames_small[i + 1], cv2.COLOR_BGR2GRAY)
        accum += cv2.absdiff(a, b).astype(np.float32)
    if accum.max() < 1e-3:
        return np.zeros(frames_small[0].shape[:2], bool)
    norm = (accum / accum.max() * 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (9, 9), 0)
    _, mask = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask > 0


def _surface_line(mask: np.ndarray) -> float:
    """Median top-of-mask y per column — the resting fluid surface height."""
    tops = []
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x])[0]
        if col.size:
            tops.append(int(col.min()))
    return float(np.median(tops)) if tops else float(mask.shape[0])


# ── Dense flow + particle advection ───────────────────────────────────────────

def _flow_field(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _sample_flow(flow: np.ndarray, x: float, y: float) -> tuple[float, float]:
    H, W = flow.shape[:2]
    xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
    return float(flow[yi, xi, 0]), float(flow[yi, xi, 1])


def _seed_grid(mask: np.ndarray, spacing: int, budget: int) -> list[tuple[float, float]]:
    ys, xs = np.where(mask[::spacing, ::spacing])
    pts = list(zip((xs * spacing).astype(float), (ys * spacing).astype(float)))
    if len(pts) > budget:
        idx = np.linspace(0, len(pts) - 1, budget, dtype=int)
        pts = [pts[i] for i in idx]
    return pts


def _advect_particles(frames_small: list, masks_by_i: dict, flows: list,
                      grid_spacing: int, max_points: int) -> list[dict]:
    """Seeds a grid inside the fluid region and follows every point through
    the flow field, re-seeding to keep the population near `max_points`.
    Returns finished paths: [{"id", "is", "xs", "ys"}], is = local frame idx."""
    n = len(frames_small)
    mask0 = masks_by_i.get(0, np.zeros(frames_small[0].shape[:2], bool))
    active: list[dict] = [{"id": i, "is": [0], "xs": [x], "ys": [y]}
                          for i, (x, y) in enumerate(_seed_grid(mask0, grid_spacing, max_points))]
    next_id = len(active)
    finished: list[dict] = []
    H, W = frames_small[0].shape[:2]

    for i in range(n - 1):
        flow = flows[i]
        mask_next = masks_by_i.get(i + 1, masks_by_i.get(i))
        still_active = []
        for p in active:
            x, y = p["xs"][-1], p["ys"][-1]
            dx, dy = _sample_flow(flow, x, y)
            nx, ny = x + dx, y + dy
            in_frame = 0 <= nx < W and 0 <= ny < H
            in_region = in_frame and mask_next is not None and \
                mask_next[int(np.clip(ny, 0, H - 1)), int(np.clip(nx, 0, W - 1))]
            if in_region:
                p["is"].append(i + 1)
                p["xs"].append(nx)
                p["ys"].append(ny)
                still_active.append(p)
            else:
                if len(p["is"]) >= 3:
                    finished.append(p)
        active = still_active

        # Re-seed to keep the tracked population near max_points.
        deficit = max_points - len(active)
        if deficit > 0 and mask_next is not None and mask_next.any():
            for x, y in _seed_grid(mask_next, grid_spacing, deficit):
                active.append({"id": next_id, "is": [i + 1], "xs": [x], "ys": [y]})
                next_id += 1

    finished.extend(p for p in active if len(p["is"]) >= 3)
    return finished


# ── Ballistic-arc fit ──────────────────────────────────────────────────────────

def _fit_parabola(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y = a·t² + b·t + c. Returns (a, r²)."""
    if len(t) < 4 or np.ptp(t) < 1e-6:
        return 0.0, 0.0
    a, b, c = np.polyfit(t, y, 2)
    pred = a * t ** 2 + b * t + c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-9
    return float(a), float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg              = json.loads(settings) if settings else {}
    model            = str(cfg.get("model") or "createai:geminiflash2_5")
    api_key          = str(cfg.get("api_key", "")).strip()
    viscosity_mode   = str(cfg.get("viscosity_mode", "low"))
    visc_scale       = 0.6 if viscosity_mode == "high" else 1.0
    grid_spacing     = max(4, int(cfg.get("grid_spacing", 14)))
    max_points       = max(20, int(cfg.get("max_points", 250)))
    arc_r2_min       = float(np.clip(cfg.get("arc_r2_min", 0.6), 0.0, 1.0))
    div_tol          = max(0.2, float(cfg.get("divergence_tolerance", 1.5))) * visc_scale
    disc_tol         = max(0.2, float(cfg.get("discontinuity_tolerance", 1.8))) * visc_scale
    causality_window = max(0.1, float(cfg.get("causality_window_s", 1.0)))
    max_checks       = max(1, int(cfg.get("max_checks", 3)))
    use_holistic     = str(cfg.get("use_holistic_vlm", "true")).lower() not in ("false", "0", "no")

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    # ── 1. Frames + tracker evidence (fluid label, other subjects for
    #        splash-causality, fps) ────────────────────────────────────────────
    ev_tr = EVIDENCE.get(vid, "s2_object_tracker")
    fluid_label = None
    if ev_tr and ev_tr.get("tracks"):
        fluid_label = _find_fluid_label(ev_tr["tracks"])

    if ev_tr and ev_tr.get("num_frames"):
        from tools.video import load_frames_rgb
        yield {"type": "log", "level": "info",
               "text": "Loading video (aligned to Object Tracker frames)…"}
        rgb, raw_fps = await loop.run_in_executor(
            None, load_frames_rgb, video_path, int(ev_tr["num_frames"]))
        frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
        eff_fps = float(ev_tr.get("fps") or raw_fps or 30.0)
    else:
        from tools.video import load_frames
        yield {"type": "log", "level": "info", "text": "Loading video…"}
        frames, raw_fps = await loop.run_in_executor(None, load_frames, video_path)
        eff_fps = raw_fps or 30.0

    frames, orig_idx = _cap_frames(frames, _MAX_ANALYSIS_FRAMES)
    n = len(frames)
    if n < 6:
        yield {"type": "error", "text": f"Not enough frames for flow analysis ({n})."}
        return
    H, W = frames[0].shape[:2]
    scale = min(1.0, _FLOW_DIM / max(H, W))
    small = [cv2.resize(f, (max(2, int(W * scale)), max(2, int(H * scale))))
            for f in frames]
    Hs, Ws = small[0].shape[:2]

    # ── 2. Fluid region: named mask (per analysis frame) or motion-energy ────
    masks_by_i: dict[int, np.ndarray] = {}
    region_src = "motion-energy region (static for the clip)"
    if fluid_label and ev_tr.get("masks_png"):
        from tools.sam3 import decode_mask_png
        per_frame = ev_tr["masks_png"].get(fluid_label, {})
        keys = list(per_frame.keys())
        for i, oi in enumerate(orig_idx):
            if not keys:
                break
            k = oi if oi in per_frame else min(keys, key=lambda kk: abs(kk - oi))
            m = decode_mask_png(per_frame[k])
            masks_by_i[i] = cv2.resize(m.astype(np.uint8), (Ws, Hs),
                                       interpolation=cv2.INTER_NEAREST) > 0
        if masks_by_i:
            region_src = f"named subject “{fluid_label}”"
    if not masks_by_i:
        yield {"type": "log", "level": "info" if fluid_label is None else "warn",
               "text": ("No fluid-like subject named by the Object Tracker — "
                        if fluid_label is None else
                        f"“{fluid_label}” had no usable masks — ")
                       + "falling back to a motion-energy region."}
        static_mask = await loop.run_in_executor(None, _motion_energy_mask, small)
        if not static_mask.any():
            yield {"type": "error", "text": "No moving fluid region detected."}
            return
        masks_by_i = {i: static_mask for i in range(n)}
    yield {"type": "log", "level": "info", "text": f"Fluid region: {region_src}."}
    await asyncio.sleep(0)

    surface_y = _surface_line(masks_by_i[0])

    # Other tracked subjects' boxes — impactor proxy for splash-causality,
    # since a dedicated Collision run isn't required.
    impactor_boxes: dict[int, list[tuple]] = {}
    if ev_tr and ev_tr.get("tracks"):
        for tr in ev_tr["tracks"]:
            if tr.get("label") == fluid_label:
                continue
            for fi, box in zip(tr.get("frames", []), tr.get("boxes", [])):
                impactor_boxes.setdefault(fi, []).append(box)

    # ── 3. Dense flow between consecutive analysis frames ─────────────────────
    yield {"type": "log", "level": "info",
           "text": f"Computing dense optical flow across {n} frame(s)…"}
    await asyncio.sleep(0)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]

    def _all_flows():
        return [_flow_field(grays[i], grays[i + 1]) for i in range(n - 1)]
    flows = await loop.run_in_executor(None, _all_flows)

    # ── 4. Particle advection — "many points, see their arcs" ────────────────
    paths = await loop.run_in_executor(
        None, _advect_particles, small, masks_by_i, flows, grid_spacing, max_points)
    yield {"type": "log", "level": "info",
           "text": f"Advected {len(paths)} point path(s) through the fluid region."}
    await asyncio.sleep(0)

    # ── 5. Region-level flow signal (for divergence/discontinuity/splash) ────
    t_axis, mean_speed, div_norm, jump_norm = [], [], [], []
    prev_vec = None
    for i in range(n - 1):
        flow = flows[i]
        m = masks_by_i.get(i, masks_by_i.get(0))
        m_small = (cv2.resize(m.astype(np.uint8), (Ws, Hs),
                              interpolation=cv2.INTER_NEAREST) > 0
                  if m.shape[:2] != (Hs, Ws) else m)
        if not m_small.any():
            t_axis.append(orig_idx[i] / eff_fps); mean_speed.append(0.0)
            div_norm.append(0.0); jump_norm.append(0.0); prev_vec = None
            continue
        fx, fy = flow[..., 0][m_small], flow[..., 1][m_small]
        speed = float(np.hypot(fx, fy).mean())
        vec = np.array([float(fx.mean()), float(fy.mean())])

        du_dx = np.gradient(flow[..., 0], axis=1)
        dv_dy = np.gradient(flow[..., 1], axis=0)
        div = (du_dx + dv_dy)[m_small]
        dnorm = float(np.abs(div).mean()) / max(speed, 0.05)

        jnorm = (float(np.hypot(*(vec - prev_vec))) / max(speed, 0.05)
                if prev_vec is not None else 0.0)

        t_axis.append(orig_idx[i] / eff_fps)
        mean_speed.append(speed)
        div_norm.append(dnorm)
        jump_norm.append(jnorm)
        prev_vec = vec

    t_axis, mean_speed = np.array(t_axis), np.array(mean_speed)
    div_norm, jump_norm = np.array(div_norm), np.array(jump_norm)

    # Splash onsets: bursts in mean flow energy relative to a rolling baseline.
    baseline = np.median(mean_speed) if len(mean_speed) else 0.0
    splash_is = [i for i, s in enumerate(mean_speed)
                if s > 3.0 * max(baseline, 0.15) and s > 0.3]

    def _nearest_impactor_t(i: int) -> Optional[float]:
        oi = orig_idx[i]
        window = int(round(causality_window * eff_fps))
        best = None
        for fi in range(max(0, oi - window), oi + window + 1):
            if impactor_boxes.get(fi):
                d = abs(fi - oi)
                if best is None or d < best[0]:
                    best = (d, fi)
        return (best[1] / eff_fps) if best else None

    # ── 6. Violations ─────────────────────────────────────────────────────────
    violations: list[dict] = []

    # 6a. ballistic_arc: airborne path segments that don't fit a downward parabola.
    for p in paths:
        ts = np.array([orig_idx[i] / eff_fps for i in p["is"]])
        ys = np.array(p["ys"])
        airborne = ys < surface_y
        if airborne.sum() < 4:
            continue
        idxs = np.where(airborne)[0]
        # Longest contiguous airborne run.
        runs, start = [], idxs[0]
        for a, b in zip(idxs, idxs[1:]):
            if b - a > 1:
                runs.append((start, a)); start = b
        runs.append((start, idxs[-1]))
        i0, i1 = max(runs, key=lambda r: r[1] - r[0])
        if i1 - i0 < 3:
            continue
        a_coef, r2 = _fit_parabola(ts[i0:i1 + 1], ys[i0:i1 + 1])
        bad_fit = r2 < arc_r2_min
        anti_gravity = a_coef < -0.02 * Hs   # curves upward instead of falling back
        if not (bad_fit or anti_gravity):
            continue
        score = float(np.clip((arc_r2_min - r2) / max(arc_r2_min, 1e-6), 0, 1)) * 0.5 + \
                (0.5 if anti_gravity else 0.0)
        violations.append({
            "type": "ballistic_arc", "path_id": p["id"],
            "frame": int(orig_idx[p["is"][i0]]), "t": round(float(ts[i0]), 3),
            "t_end": round(float(ts[i1]), 3), "r2": round(r2, 3),
            "confirmed": None, "score": round(min(score, 1.0), 3),
            "path_xy": [(float(x), float(y)) for x, y in
                       zip(np.array(p["xs"])[i0:i1 + 1], ys[i0:i1 + 1])],
            "desc": (f"an airborne droplet's path (t={ts[i0]:.2f}–{ts[i1]:.2f}s) "
                     f"{'defies gravity (curves upward)' if anti_gravity else 'does not fit a ballistic parabola'} "
                     f"(fit R²={r2:.2f})"),
        })

    # 6b. incompressibility: normalized divergence spikes (source/sink).
    hot = [i for i, d in enumerate(div_norm) if d > div_tol]
    for i in hot:
        violations.append({
            "type": "incompressibility", "frame": int(orig_idx[i]),
            "t": round(float(t_axis[i]), 3), "t_end": round(float(t_axis[i]), 3),
            "confirmed": None,
            "score": round(float(np.clip(div_norm[i] / (2 * div_tol), 0, 1)) * 0.85, 3),
            "desc": (f"the fluid region's flow divergence spikes to {div_norm[i]:.2f}× "
                     f"its own flow speed at t={t_axis[i]:.2f}s — fluid appearing "
                     "or vanishing rather than flowing continuously"),
        })

    # 6c. splash_timing: energy burst with no nearby impactor, or backwards causality.
    for i in splash_is:
        near_t = _nearest_impactor_t(i)
        splash_t = float(t_axis[i])
        if near_t is None:
            violations.append({
                "type": "splash_timing", "frame": int(orig_idx[i]),
                "t": round(splash_t, 3), "t_end": round(splash_t, 3), "confirmed": None,
                "score": 0.6,
                "desc": (f"a splash/energy burst at t={splash_t:.2f}s has no "
                         "tracked object contacting the fluid nearby"),
            })
        elif near_t - splash_t > 0.1:
            violations.append({
                "type": "splash_timing", "frame": int(orig_idx[i]),
                "t": round(splash_t, 3), "t_end": round(near_t, 3), "confirmed": None,
                "score": 0.7,
                "desc": (f"a splash at t={splash_t:.2f}s precedes its apparent "
                         f"impactor's contact at t={near_t:.2f}s — effect before cause"),
            })

    # 6d. flow_discontinuity: bulk-flow-vector jump with nothing (splash/impactor) to explain it.
    splash_frames = {orig_idx[i] for i in splash_is}
    for i, jn in enumerate(jump_norm):
        if jn <= disc_tol:
            continue
        oi = orig_idx[i]
        near_event = any(abs(oi - sf) <= int(round(causality_window * eff_fps))
                        for sf in splash_frames) or bool(impactor_boxes.get(oi))
        if near_event:
            continue
        violations.append({
            "type": "flow_discontinuity", "frame": int(oi),
            "t": round(float(t_axis[i]), 3), "t_end": round(float(t_axis[i]), 3),
            "confirmed": None,
            "score": round(float(np.clip(jn / (2 * disc_tol), 0, 1)) * 0.75, 3),
            "desc": (f"the fluid's overall flow direction/speed jumps abruptly at "
                     f"t={t_axis[i]:.2f}s with no splash or contact to explain it"),
        })

    violations = sorted(violations, key=lambda v: -v["score"])
    for v in violations:
        yield {"type": "log", "level": "warn", "text": f"FLAGGED {v['type']}: {v['desc']}."}

    # ── 7. VLM: targeted verification of the worst flagged events ────────────
    from tools.vlm_router import key_status
    have_api, key_desc = key_status(model, api_key)
    to_check = violations[:max_checks]
    if to_check and have_api:
        from tools.vlm_router import ask_vision_json
        for v in to_check:
            i = next((j for j, oi in enumerate(orig_idx) if oi == v["frame"]), None)
            if i is None:
                continue
            frame = frames[i]
            if v["type"] == "ballistic_arc" and v.get("path_xy"):
                img = frame.copy()
                pts = np.array([(x / scale, y / scale) for x, y in v["path_xy"]],
                               dtype=np.int32)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(img, tuple(pts[0]), 6, (0, 255, 0), -1)
                cv2.circle(img, tuple(pts[-1]), 6, (0, 0, 255), -1)
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                crop = _region_crop(img, box=(x0, y0, x1, y1), pad=0.6)
                anno_img = _tile_row([_labeled_tile(crop, f"t={v['t']:.2f}-{v['t_end']:.2f}s")]) \
                    if crop is not None else None
            else:
                mask = masks_by_i.get(i)
                mask_full = (cv2.resize(mask.astype(np.uint8), (W, H),
                                        interpolation=cv2.INTER_NEAREST) > 0
                            if mask is not None else None)
                crop = _region_crop(frame, mask=mask_full)
                anno_img = _tile_row([_labeled_tile(crop, f"t={v['t']:.2f}s")]) \
                    if crop is not None else None
            if anno_img is None:
                continue
            try:
                parsed = await ask_vision_json(
                    ANOMALY_PROMPT.format(desc=v["desc"]), anno_img, model, api_key)
                verdict = str(parsed.get("verdict", "")).lower()
                conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0, 1))
                expl = str(parsed.get("explanation", ""))[:300]
                v["confirmed"] = (verdict == "violation")
                v["vlm_confidence"] = round(conf, 3)
                v["explanation"] = expl
                if v["confirmed"]:
                    v["score"] = round(max(v["score"], conf), 3)
                    yield {"type": "log", "level": "warn",
                           "text": f"VLM CONFIRMS {v['type']} @ t={v['t']:.2f}s "
                                   f"({conf:.0%}): {expl}"}
                else:
                    v["score"] = round(v["score"] * 0.25, 3)
                    yield {"type": "log", "level": "info",
                           "text": f"VLM: plausible cause @ t={v['t']:.2f}s "
                                   f"({conf:.0%}): {expl}"}
                yield _jpeg_event(anno_img,
                                  f"{v['type']} @ t={v['t']:.2f}s — "
                                  f"{verdict} ({conf:.0%}) per {model}")
            except Exception as exc:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"VLM check @ t={v['t']:.2f}s failed: {str(exc)[:140]} "
                               "— keeping unverified score."}
            await asyncio.sleep(0)
        violations = sorted(violations, key=lambda v: -v["score"])
    elif to_check:
        yield {"type": "log", "level": "warn",
               "text": f"No VLM credentials ({key_desc}) — flagged events reported unverified."}

    # ── 8. Holistic realism pass ───────────────────────────────────────────────
    holistic = None
    if use_holistic and have_api:
        from tools.vlm_router import ask_vision_json
        n_tiles = min(5, n)
        idxs = [int(round(x)) for x in np.linspace(0, n - 1, n_tiles)]
        tiles = []
        for i in idxs:
            mask_full = cv2.resize(masks_by_i.get(i, masks_by_i[0]).astype(np.uint8),
                                   (W, H), interpolation=cv2.INTER_NEAREST) > 0
            crop = _region_crop(frames[i], mask=mask_full)
            if crop is not None:
                tiles.append(_labeled_tile(crop, f"t={orig_idx[i] / eff_fps:.2f}s"))
        composite = _tile_row(tiles)
        if composite is not None:
            try:
                parsed = await ask_vision_json(
                    HOLISTIC_PROMPT.format(n=len(tiles)), composite, model, api_key)
                verdict = str(parsed.get("verdict", "")).lower()
                conf = float(np.clip(float(parsed.get("confidence", 0.5)), 0, 1))
                expl = str(parsed.get("explanation", ""))[:300]
                holistic = {"verdict": verdict, "confidence": conf, "explanation": expl}
                lvl = "warn" if verdict == "unrealistic" else "success"
                yield {"type": "log", "level": lvl,
                       "text": f"Holistic realism: {verdict} ({conf:.0%}): {expl}"}
                yield _jpeg_event(composite,
                                  f"Fluid realism check — {verdict} ({conf:.0%}) per {model}")
            except Exception as exc:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"Holistic realism check failed: {str(exc)[:160]}"}
    elif use_holistic:
        yield {"type": "log", "level": "warn",
               "text": f"No VLM credentials ({key_desc}) — skipping holistic realism check."}

    # ── 9. Aggregate: signals, severity, plots, metrics ───────────────────────
    signals = [{"frame": v["frame"], "signal_type": v["type"], "score": v["score"]}
               for v in violations]
    holistic_score = (holistic["confidence"] if holistic and holistic["verdict"] == "unrealistic" else 0.0)
    severity = int(round(100 * max([v["score"] for v in violations] + [holistic_score], default=0.0)))

    yield {"type": "signal", "source": "s3_fluid", "source_name": "Fluid",
           "fps": float(eff_fps), "n_frames": int(n), "severity": severity,
           "type_severities": {
               t: int(round(100 * max([v["score"] for v in violations
                                       if v["type"] == t], default=0.0)))
               for t in ("ballistic_arc", "incompressibility", "splash_timing",
                        "flow_discontinuity")},
           "signals": signals}

    # Arc overlay: every advected path drawn on a representative frame.
    overlay = frames[n // 2].copy()
    flagged_ids = {v["path_id"] for v in violations if v.get("path_id") is not None}
    for p in paths:
        pts = np.array([(x / scale, y / scale) for x, y in zip(p["xs"], p["ys"])],
                       dtype=np.int32)
        if len(pts) < 2:
            continue
        color = (0, 0, 220) if p["id"] in flagged_ids else (210, 160, 40)
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    cv2.line(overlay, (0, int(surface_y / scale)), (W, int(surface_y / scale)),
             (255, 255, 0), 1, cv2.LINE_AA)
    yield _jpeg_event(overlay,
                      f"Advected point arcs ({len(paths)} paths) — cyan/gold = normal, "
                      "red = flagged; yellow line = estimated fluid surface.")

    # Divergence + discontinuity over time.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("Flow divergence (normalized by region flow speed)",
                                        "Bulk-flow discontinuity (frame-to-frame jump, normalized)"))
    fig.add_trace(go.Scatter(x=t_axis.tolist(), y=div_norm.tolist(), mode="lines",
                             name="divergence", line=dict(color="#1a54c4", width=1.6)),
                  row=1, col=1)
    fig.add_hline(y=div_tol, row=1, col=1, line=dict(color="#E24B4A", dash="dash", width=1.2))
    fig.add_trace(go.Scatter(x=t_axis.tolist(), y=jump_norm.tolist(), mode="lines",
                             name="discontinuity", line=dict(color="#c05621", width=1.6)),
                  row=2, col=1)
    fig.add_hline(y=disc_tol, row=2, col=1, line=dict(color="#E24B4A", dash="dash", width=1.2))
    for v in violations:
        r = 1 if v["type"] == "incompressibility" else 2 if v["type"] == "flow_discontinuity" else None
        if r:
            fig.add_vline(x=v["t"], row=r, col=1,
                         line=dict(color="#E24B4A", dash="dot", width=1.2))
    fig.update_xaxes(title_text="Time (s)", row=2, col=1, showgrid=True, gridcolor="#ebebeb")
    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
    fig.update_layout(
        title=dict(text="Fluid — Flow Continuity & Discontinuity", font=dict(size=15)),
        height=520, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
        margin=dict(l=65, r=40, t=100, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "Dashed red = violation threshold; dotted red markers = flagged "
                      "moments (top: incompressibility, bottom: discontinuity)."}

    yield {"type": "metric", "label": "Points tracked", "value": str(len(paths)),
           "sub": (f"avg path length {np.mean([len(p['is']) for p in paths]):.1f} frame(s)"
                  if paths else "no paths")}
    n_conf = sum(1 for v in violations if v["confirmed"])
    yield {"type": "metric", "label": "Fluid anomalies", "value": str(len(violations)),
           "sub": f"{n_conf} VLM-confirmed"}
    if holistic:
        yield {"type": "metric", "label": "Holistic realism",
               "value": holistic["verdict"].upper(),
               "sub": f"{holistic['confidence']:.0%} confidence ({model})"}

    yield {"type": "severity", "label": "Fluid violation",
           "value": severity, "color": _sev_color(severity)}

    yield {"type": "result", "status": "ok",
           "region_source": region_src, "n_paths": len(paths),
           "violations": violations, "holistic": holistic, "severity": severity}
    EVIDENCE.put(vid, "s3_fluid", {
        "severity": severity, "violations": violations,
        "holistic": holistic, "signals": signals,
    })

    if severity > 30:
        yield {"type": "log", "level": "warn", "text": f"Fluid VIOLATION — severity {severity}%."}
    else:
        yield {"type": "log", "level": "success", "text": "Fluid motion looks physically plausible."}
    yield {"type": "done"}

"""
Stage 3 · Specialist — Momentum Specialist
--------------------------------------------
True physical momentum is unrecoverable from monocular RGB (mass, depth, and
camera calibration are unknown), so this specialist builds a *motion
signature* per tracked subject and hunts momentum-shaped defects in it.

Per subject (tracks/masks reused from the Object Tracker's evidence; shared
cached LK tracks as fallback — no re-tracking):

1. FINGERPRINT — image-space kinematics on the subject's own frame grid:
   max/avg speed, peak acceleration, jerk-based smoothness, path length,
   straightness, arc height/range/symmetry, mean curvature — combined into a
   0–100 momentum score (weights per the motion-signature spec).
2. MASS PROXY — median mask/box area, optionally refined by ONE VLM call that
   ranks the subjects' apparent relative mass (a bowling ball outweighs a
   balloon of the same size). Momentum proxy p(t) = m̂ · v(t).
3. VIOLATIONS —
     * momentum_jump: |Δp| spikes with no contact, no other subject nearby,
       and not at the frame border (motion changed with no visible cause);
     * transfer_anomaly: at a contact episode, the pair's momentum exchange
       is one-sided or appears from nowhere (Δp_A + Δp_B far from balanced
       given how hard the mover hit).
   Each flagged event gets ONE VLM check (labeled before/after frames) to
   confirm or reject — same detect→explain pattern as the other specialists.
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

from tools.evidence import EVIDENCE, video_id

MASS_PROMPT = (
    "The image shows one crop per tracked object from a single video (labels "
    "printed above each tile). Estimate each object's RELATIVE physical mass "
    "from its apparent material and build (e.g. a bowling ball is far heavier "
    "than a balloon of the same size). Use a relative scale where 1.0 is the "
    "typical object in this scene; stay within 0.1–10.\n"
    "Reply with ONLY strict JSON: {\"masses\": {\"<label>\": <number>, ...}}"
)

ANOMALY_PROMPT = (
    "Two moments of one video: LEFT is BEFORE, RIGHT is AFTER (timestamps "
    "printed on the image; the object in question is outlined).\n"
    "An automated motion metric flagged this: {desc}\n"
    "Verify it. A real physical cause (visible push, collision, bounce off "
    "something in frame, gravity on a falling object, exiting the frame) "
    "makes it 'plausible'. Motion that starts, stops, or redirects with NO "
    "visible cause is a 'violation' (AI-generation artifact).\n"
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


def _smooth(v: np.ndarray, k: int = 3) -> np.ndarray:
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="same")


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


def _event_crop(frame: np.ndarray, box: tuple, pad: float = 0.6) -> Optional[np.ndarray]:
    """Crop around a subject's box (outlined), padded for scene context."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    img = frame.copy()
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
    px, py = int((x1 - x0) * pad) + 24, int((y1 - y0) * pad) + 24
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
    if cx1 - cx0 < 24 or cy1 - cy0 < 24:
        return None
    return img[cy0:cy1, cx0:cx1]


# ── Kinematics + fingerprint ──────────────────────────────────────────────────

def _kinematics(track: dict, eff_fps: float) -> Optional[dict]:
    """Per-subject motion curves on its own (possibly gapped) frame grid."""
    fis = track["frames"]
    if len(fis) < 4:
        return None
    t = np.asarray([fi / eff_fps for fi in fis], dtype=float)
    x = _smooth(np.asarray(track["cx"], dtype=float))
    y = _smooth(np.asarray(track["cy"], dtype=float))
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(vx, vy)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    accel = np.hypot(ax, ay)
    jerk = np.hypot(np.gradient(ax, t), np.gradient(ay, t))
    areas = np.asarray([max(1.0, (bx1 - bx0) * (by1 - by0))
                        for bx0, by0, bx1, by1 in track["boxes"]], dtype=float)
    return {"fis": fis, "t": t, "x": x, "y": y, "vx": vx, "vy": vy,
            "speed": speed, "accel": accel, "jerk": jerk, "areas": areas}


def _fingerprint(k: dict, H: int) -> dict:
    """Motion signature + 0–100 momentum score (image-space, frame-relative)."""
    t, x, y, speed = k["t"], k["x"], k["y"], k["speed"]
    dur = max(t[-1] - t[0], 1e-6)

    steps = np.hypot(np.diff(x), np.diff(y))
    path_len = float(steps.sum())
    net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = net_disp / max(path_len, 1e-6)

    # Arc: vertical excursion + rise/fall symmetry around the apex (min y = top).
    arc_height = float(y.max() - y.min())
    arc_range = float(x.max() - x.min())
    apex = int(np.argmin(y))
    rise_t = t[apex] - t[0]
    fall_t = t[-1] - t[apex]
    arc_symmetry = (min(rise_t, fall_t) / max(rise_t, fall_t)
                    if min(rise_t, fall_t) > 0 else 0.0)

    # Curvature: mean absolute turning angle per step (deg), speed-gated.
    headings = np.arctan2(np.diff(y), np.diff(x))
    turns = np.abs(np.degrees(np.diff(np.unwrap(headings))))
    moving = steps[1:] > 1.0
    curvature = float(turns[moving].mean()) if moving.any() else 0.0

    # Smoothness: 1 / (1 + normalized jerk RMS).
    jerk_rms = float(np.sqrt(np.mean(k["jerk"] ** 2)))
    smoothness = 1.0 / (1.0 + jerk_rms / max(8.0 * H, 1e-6))

    vmax, vavg = float(speed.max()), float(speed.mean())
    amax = float(k["accel"].max())

    return {
        "max_speed_px_s": round(vmax, 1), "avg_speed_px_s": round(vavg, 1),
        "max_accel_px_s2": round(amax, 1),
        "path_length_px": round(path_len, 1),
        "straightness": round(straightness, 3),
        "arc_height_px": round(arc_height, 1), "arc_range_px": round(arc_range, 1),
        "arc_symmetry": round(arc_symmetry, 3),
        "curvature_deg": round(curvature, 1),
        "smoothness": round(smoothness, 3),
        "duration_s": round(dur, 2),
    }


def _momentum_score(fp: dict, mass_w: float, H: int) -> int:
    """Weighted 0–100 combination (weights per the motion-signature spec:
    velocity high, acceleration/path medium, curvature/smoothness low)."""
    n = lambda v, scale: float(np.clip(v / scale, 0.0, 1.0))
    score = (
        0.30 * n(fp["max_speed_px_s"], 1.5 * H) +
        0.20 * n(fp["avg_speed_px_s"], 0.5 * H) +
        0.15 * n(mass_w, 3.0) +
        0.15 * n(fp["max_accel_px_s2"], 8.0 * H) +
        0.10 * n(fp["path_length_px"], 3.0 * H) +
        0.05 * n(fp["curvature_deg"], 45.0) +
        0.05 * fp["smoothness"]
    )
    return int(round(100 * score))


# ── Contacts (box adjacency on the shared grid — cheap, grid-safe) ────────────

def _contact_episodes(tracks: dict, eff_fps: float, grid_step: int,
                      H: int) -> list[dict]:
    """Per pair: contiguous frames where padded boxes intersect."""
    pad = 0.04 * H
    box_by = {lb: dict(zip(tr["frames"], tr["boxes"])) for lb, tr in tracks.items()}
    episodes = []
    for la, lb in combinations(tracks.keys(), 2):
        hot = []
        for fi in sorted(set(box_by[la]) & set(box_by[lb])):
            ax0, ay0, ax1, ay1 = box_by[la][fi]
            bx0, by0, bx1, by1 = box_by[lb][fi]
            if (ax0 - pad < bx1 and bx0 - pad < ax1 and
                    ay0 - pad < by1 and by0 - pad < ay1):
                hot.append(fi)
        if not hot:
            continue
        start = prev = hot[0]
        for fi in hot[1:] + [None]:
            if fi is None or fi - prev > 2 * grid_step:
                episodes.append({"pair": (la, lb), "f_start": start, "f_end": prev,
                                 "t_start": round(start / eff_fps, 3),
                                 "t_end": round(prev / eff_fps, 3)})
                start = fi
            prev = fi if fi is not None else prev
    return episodes


def _near_contact(fi: int, label: str, episodes: list[dict], slack: int) -> bool:
    return any(label in ep["pair"] and
               ep["f_start"] - slack <= fi <= ep["f_end"] + slack
               for ep in episodes)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg           = json.loads(settings) if settings else {}
    model         = str(cfg.get("model") or "createai:geminiflash2_5")
    api_key       = str(cfg.get("api_key", "")).strip()
    momentum_tol  = max(0.05, float(cfg.get("momentum_tolerance", 0.5)))
    transfer_tol  = max(0.05, float(cfg.get("transfer_tolerance", 0.35)))
    max_checks    = max(1, int(cfg.get("max_checks", 3)))
    use_mass_vlm  = str(cfg.get("use_mass_ranking", "true")).lower() not in ("false", "0", "no")

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    # ── 1. Tracks + frames: tracker evidence first, cached LK fallback ───────
    ev_tr = EVIDENCE.get(vid, "s2_object_tracker")
    tracks: dict[str, dict] = {}
    if ev_tr and ev_tr.get("tracks"):
        if ev_tr.get("num_frames"):        # SAM3 grid → decode identically
            from tools.video import load_frames_rgb
            yield {"type": "log", "level": "info",
                   "text": "Loading video (aligned to Object Tracker frames)…"}
            rgb, raw_fps = await loop.run_in_executor(
                None, load_frames_rgb, video_path, int(ev_tr["num_frames"]))
            frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
        else:
            from tools.video import load_frames
            yield {"type": "log", "level": "info", "text": "Loading video…"}
            frames, raw_fps = await loop.run_in_executor(None, load_frames, video_path)
        eff_fps = float(ev_tr.get("fps") or raw_fps or 30.0)
        for i, tr in enumerate(ev_tr["tracks"]):
            if len(tr.get("frames", [])) >= 4:
                tracks[tr.get("label") or f"obj {i}"] = tr
        src = f"Object Tracker evidence ({ev_tr.get('mode', '?')} mode)"
    else:
        from tools.tracking import get_tracks
        yield {"type": "log", "level": "info",
               "text": "No Object Tracker evidence — using shared cached LK tracks."}
        data = await loop.run_in_executor(None, get_tracks, video_path)
        frames, eff_fps = data["frames"], data["fps"] or 30.0
        for tr in data["tracks"]:
            if len(tr["frames"]) >= 4:
                tracks[tr.get("label") or f"obj {tr['id']}"] = tr
        src = "cached LK tracks (fallback)"

    n = len(frames)
    if n < 4 or not tracks:
        yield {"type": "error",
               "text": f"Not enough motion data ({n} frames, {len(tracks)} usable track(s))."}
        return
    H, W = frames[0].shape[:2]
    all_fis = sorted({fi for tr in tracks.values() for fi in tr["frames"]})
    grid_step = int(np.median(np.diff(all_fis))) if len(all_fis) > 2 else 1
    yield {"type": "log", "level": "info",
           "text": f"{len(tracks)} subject(s) from {src}: "
                   + ", ".join(f"“{lb}”" for lb in tracks)}
    await asyncio.sleep(0)

    # ── 2. Kinematics + mass proxy ────────────────────────────────────────────
    kin = {lb: k for lb, tr in tracks.items()
           if (k := _kinematics(tr, eff_fps)) is not None}
    if not kin:
        yield {"type": "error", "text": "No track has enough samples for kinematics."}
        return

    # Drop near-static tracks (background corner clusters in LK mode): a
    # subject must actually travel to have a motion signature worth judging.
    min_travel = 0.10 * H
    static = [lb for lb, k in kin.items()
              if float(np.hypot(np.diff(k["x"]), np.diff(k["y"])).sum()) < min_travel]
    for lb in static:
        kin.pop(lb)
    if static:
        yield {"type": "log", "level": "info",
               "text": f"Ignoring {len(static)} near-static track(s): "
                       + ", ".join(f"“{lb}”" for lb in static)}
    if not kin:
        yield {"type": "error", "text": "All tracks are near-static — nothing to score."}
        return

    area_med = {lb: float(np.median(k["areas"])) for lb, k in kin.items()}
    mean_area = float(np.mean(list(area_med.values()))) or 1.0
    mass_w = {lb: a / mean_area for lb, a in area_med.items()}   # size proxy, mean≈1

    from tools.vlm_router import key_status
    have_api, key_desc = key_status(model, api_key)

    if use_mass_vlm and have_api and len(kin) >= 2:
        try:
            from tools.vlm_router import ask_vision_json
            tiles = []
            for lb, tr in tracks.items():
                if lb not in kin:
                    continue
                mid = tr["frames"][len(tr["frames"]) // 2]
                box = tr["boxes"][tr["frames"].index(mid)]
                crop = _event_crop(frames[mid], box, pad=0.15)
                if crop is not None:
                    tiles.append(_labeled_tile(crop, lb))
            comp = _tile_row(tiles)
            if comp is not None:
                parsed = await ask_vision_json(MASS_PROMPT, comp, model, api_key)
                masses = parsed.get("masses") or {}
                applied = []
                for lb in list(mass_w):
                    v = masses.get(lb)
                    if isinstance(v, (int, float)) and 0.05 <= float(v) <= 20:
                        mass_w[lb] *= float(np.clip(v, 0.1, 10.0))
                        applied.append(f"{lb}={float(v):g}")
                if applied:
                    m = float(np.mean(list(mass_w.values()))) or 1.0
                    mass_w = {lb: w / m for lb, w in mass_w.items()}  # renormalize
                    yield {"type": "log", "level": "info",
                           "text": "VLM relative-mass ranking applied: " + ", ".join(applied)}
        except Exception as exc:                                    # noqa: BLE001
            yield {"type": "log", "level": "warn",
                   "text": f"VLM mass ranking failed ({str(exc)[:140]}) — "
                           "using mask/box area alone."}
    elif use_mass_vlm and not have_api:
        yield {"type": "log", "level": "warn",
               "text": f"No VLM credentials ({key_desc}) — mass proxy is area-only, "
                       "anomalies will be reported unverified."}
    await asyncio.sleep(0)

    # ── 3. Fingerprints + momentum proxy curves ──────────────────────────────
    fingerprints: dict[str, dict] = {}
    p_curves: dict[str, dict] = {}
    for lb, k in kin.items():
        fp = _fingerprint(k, H)
        fp["mass_weight"] = round(mass_w[lb], 3)
        fp["momentum_score"] = _momentum_score(fp, mass_w[lb], H)
        fingerprints[lb] = fp
        p = mass_w[lb] * k["speed"]
        p_curves[lb] = {"t": k["t"].tolist(), "p": p.tolist(),
                        "speed": k["speed"].tolist()}
        yield {"type": "log", "level": "info",
               "text": f"“{lb}”: momentum score {fp['momentum_score']} — "
                       f"vmax {fp['max_speed_px_s']:.0f} px/s, "
                       f"path {fp['path_length_px']:.0f} px, "
                       f"arc h {fp['arc_height_px']:.0f} px "
                       f"(sym {fp['arc_symmetry']:.2f}), "
                       f"smooth {fp['smoothness']:.2f}, m̂ {mass_w[lb]:.2f}"}
    await asyncio.sleep(0)

    # ── 4. Violations ─────────────────────────────────────────────────────────
    episodes = _contact_episodes(tracks, eff_fps, grid_step, H)
    if episodes:
        yield {"type": "log", "level": "info",
               "text": f"{len(episodes)} contact episode(s) between subjects."}

    violations: list[dict] = []
    v_floor = 0.03 * H                       # px/s — ignore near-static motion
    slack = 3 * grid_step

    # 4a. momentum_jump: |Δp| spike with no contact / border / other cause slot.
    box_by = {lb: dict(zip(tr["frames"], tr["boxes"])) for lb, tr in tracks.items()}
    for lb, k in kin.items():
        p = mass_w[lb] * k["speed"]
        # Denominator floor: median momentum, but never below 15% of the
        # track's peak — a mostly-idle object otherwise yields absurd ratios.
        p_med = max(float(np.median(p)), 0.15 * float(p.max()),
                    mass_w[lb] * v_floor)
        # Skip the first/last two samples: np.gradient + smoothing are
        # unstable at track boundaries and fake a "jump" at t≈0.
        for i in range(2, len(p) - 1):
            jump = abs(p[i] - p[i - 1]) / p_med
            if jump <= momentum_tol:
                continue
            fi = k["fis"][i]
            if _near_contact(fi, lb, episodes, slack):
                continue
            bx0, by0, bx1, by1 = box_by[lb][fi]
            at_border = (bx0 < 0.02 * W or by0 < 0.02 * H or
                         bx1 > 0.98 * W or by1 > 0.98 * H)
            if at_border:
                continue
            # Gravity slot: vertical speed-up while clearly falling is expected.
            if k["vy"][i] > v_floor and k["vy"][i] >= k["vy"][i - 1] > 0:
                continue
            violations.append({
                "type": "momentum_jump", "label": lb,
                "frame": int(fi), "t": round(fi / eff_fps, 3),
                "jump_ratio": round(float(jump), 3),
                "score": round(float(np.clip(jump / (2 * momentum_tol), 0, 1)) * 0.8, 3),
                "f_before": int(k["fis"][i - 1]), "confirmed": None,
                "desc": (f"“{lb}” changed momentum by {jump:.0%} of its typical "
                         f"level in one step with no contact or visible cause"),
            })

    # 4b. transfer_anomaly at contacts: one-sided / unbalanced exchange.
    for ep in episodes:
        la, lb_ = ep["pair"]
        if la not in kin or lb_ not in kin:
            continue
        dps = {}
        for lb2 in (la, lb_):
            k = kin[lb2]
            pre = [j for j, fi in enumerate(k["fis"]) if fi < ep["f_start"]][-3:]
            post = [j for j, fi in enumerate(k["fis"]) if fi > ep["f_end"]][:3]
            if not pre or not post:
                dps = {}
                break
            v_pre = np.array([np.median(k["vx"][pre]), np.median(k["vy"][pre])])
            v_post = np.array([np.median(k["vx"][post]), np.median(k["vy"][post])])
            dps[lb2] = {"dp": mass_w[lb2] * (v_post - v_pre),
                        "p_in": mass_w[lb2] * float(np.hypot(*v_pre))}
        if not dps:
            continue
        p_scale = max(max(d["p_in"] for d in dps.values()),
                      max(mass_w[l2] for l2 in (la, lb_)) * v_floor)
        if p_scale <= 0:
            continue
        resid = float(np.hypot(*(dps[la]["dp"] + dps[lb_]["dp"]))) / p_scale
        moved = {l2: float(np.hypot(*dps[l2]["dp"])) / p_scale for l2 in (la, lb_)}
        fast_in = max(dps.values(), key=lambda d: d["p_in"])["p_in"] / p_scale
        one_sided = (fast_in > 0.6 and max(moved.values()) > transfer_tol
                     and min(moved.values()) < 0.15 * max(moved.values()))
        if resid > transfer_tol or one_sided:
            fi = ep["f_end"]
            why = ("momentum appears from nowhere / vanishes"
                   if resid > transfer_tol else
                   "one object changes momentum while its partner doesn't respond")
            violations.append({
                "type": "transfer_anomaly", "label": la, "pair": [la, lb_],
                "frame": int(fi), "t": ep["t_end"],
                "residual": round(resid, 3),
                "score": round(float(np.clip(max(resid, max(moved.values()))
                                             / (2 * transfer_tol), 0, 1)) * 0.9, 3),
                "f_before": int(ep["f_start"]), "confirmed": None,
                "desc": (f"contact between “{la}” and “{lb_}” at "
                         f"t={ep['t_end']:.2f}s: {why} "
                         f"(residual {resid:.0%} of incoming momentum)"),
            })

    # Dedupe: one physical event spans several grid steps — keep the strongest
    # flag per (subject, type) within a 3-grid-step window.
    violations.sort(key=lambda v: (v["label"], v["type"], v["frame"]))
    deduped: list[dict] = []
    for v in violations:
        prev = deduped[-1] if deduped else None
        if (prev and prev["label"] == v["label"] and prev["type"] == v["type"]
                and v["frame"] - prev["frame"] <= 3 * grid_step):
            if v["score"] > prev["score"]:
                deduped[-1] = v
        else:
            deduped.append(v)
    violations = sorted(deduped, key=lambda v: -v["score"])

    for v in violations:
        yield {"type": "log", "level": "warn",
               "text": f"FLAGGED {v['type']}: {v['desc']}."}

    # 4c. VLM verifies the worst flagged events.
    to_check = violations[:max_checks]
    if to_check and have_api:
        from tools.vlm_router import ask_vision_json
        for v in to_check:
            lb = v["label"]
            f0, f1 = v["f_before"], v["frame"]
            b0 = box_by[lb].get(f0) or box_by[lb][min(box_by[lb],
                                                      key=lambda f: abs(f - f0))]
            b1 = box_by[lb].get(f1) or box_by[lb][min(box_by[lb],
                                                      key=lambda f: abs(f - f1))]
            pair_img = _tile_row([
                _labeled_tile(_event_crop(frames[f0], b0),
                              f"BEFORE t={f0 / eff_fps:.2f}s"),
                _labeled_tile(_event_crop(frames[f1], b1),
                              f"AFTER t={f1 / eff_fps:.2f}s")])
            if pair_img is None:
                continue
            try:
                parsed = await ask_vision_json(
                    ANOMALY_PROMPT.format(desc=v["desc"]), pair_img, model, api_key)
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
                yield _jpeg_event(pair_img,
                                  f"“{lb}” {v['type']} @ t={v['t']:.2f}s — "
                                  f"{verdict} ({conf:.0%}) per {model}")
            except Exception as exc:                                # noqa: BLE001
                yield {"type": "log", "level": "warn",
                       "text": f"VLM check @ t={v['t']:.2f}s failed: {str(exc)[:140]} "
                               "— keeping unverified score."}
            await asyncio.sleep(0)
    elif to_check:
        yield {"type": "log", "level": "warn",
               "text": "Flagged events reported unverified (no VLM credentials)."}

    # ── 5. Aggregate: signals, severity, plots, metrics ──────────────────────
    signals = [{"frame": v["frame"], "signal_type": v["type"], "score": v["score"]}
               for v in violations]
    severity = int(round(100 * max([v["score"] for v in violations], default=0.0)))

    yield {"type": "signal", "source": "s3_momentum",
           "source_name": "Momentum",
           "fps": float(eff_fps), "n_frames": int(n), "severity": severity,
           "type_severities": {
               t: int(round(100 * max([v["score"] for v in violations
                                       if v["type"] == t], default=0.0)))
               for t in ("momentum_jump", "transfer_anomaly")},
           "signals": signals}

    # Trajectory overlay: each subject's path on a representative frame.
    overlay = frames[all_fis[len(all_fis) // 2]].copy()
    palette = [(196, 84, 26), (33, 86, 192), (60, 122, 26),
               (237, 62, 124), (178, 24, 190), (11, 145, 200)]
    for i, (lb, k) in enumerate(kin.items()):
        color = palette[i % len(palette)]
        pts = np.stack([k["x"], k["y"]], axis=1).astype(np.int32)
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        for j in range(0, len(pts), max(1, len(pts) // 24)):
            cv2.circle(overlay, tuple(pts[j]), 4, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, lb, tuple(np.clip(pts[0] + [6, -8], 4,
                                               [W - 8, H - 8])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    yield _jpeg_event(overlay, "Trajectory overlay — each subject's centroid path.")

    # Momentum-proxy + speed curves.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("Momentum proxy  p(t) = m̂ · v(t)",
                                        "Speed (px/s)"))
    hexp = ["#c05621", "#1a54c4", "#1a7a3c", "#be185d", "#7c3aed", "#0891b2"]
    for i, (lb, d) in enumerate(p_curves.items()):
        color = hexp[i % len(hexp)]
        fig.add_trace(go.Scatter(x=d["t"], y=d["p"], mode="lines", name=lb,
                                 legendgroup=lb,
                                 line=dict(color=color, width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["t"], y=d["speed"], mode="lines", name=lb,
                                 legendgroup=lb, showlegend=False,
                                 line=dict(color=color, width=1.6)), row=2, col=1)
    for ep in episodes:
        for r in (1, 2):
            fig.add_vrect(x0=ep["t_start"], x1=ep["t_end"], fillcolor="#EF9F27",
                          opacity=0.12, line_width=0, row=r, col=1)
    for v in violations:
        for r in (1, 2):
            fig.add_vline(x=v["t"], line=dict(color="#E24B4A", dash="dot",
                                              width=1.2), row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                     showgrid=True, gridcolor="#ebebeb")
    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
    fig.update_layout(
        title=dict(text="Momentum — Motion Signature & Conservation Proxy",
                   font=dict(size=15)),
        height=560, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
        margin=dict(l=65, r=40, t=100, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "Orange bands = contact episodes (momentum may change); "
                      "dotted red = flagged momentum anomalies."}

    for lb, fp in sorted(fingerprints.items(),
                         key=lambda kv: -kv[1]["momentum_score"]):
        yield {"type": "metric", "label": f"“{lb}” momentum score",
               "value": str(fp["momentum_score"]),
               "sub": (f"vmax {fp['max_speed_px_s']:.0f} px/s · "
                       f"m̂ {fp['mass_weight']:.2f} · "
                       f"smooth {fp['smoothness']:.2f}")}
    n_conf = sum(1 for v in violations if v["confirmed"])
    yield {"type": "metric", "label": "Momentum anomalies",
           "value": str(len(violations)),
           "sub": f"{n_conf} VLM-confirmed · {len(episodes)} contact episode(s)"}

    yield {"type": "severity", "label": "Momentum violation",
           "value": severity, "color": _sev_color(severity)}

    yield {"type": "result", "status": "ok",
           "fingerprints": fingerprints, "mass_weights":
               {lb: round(w, 3) for lb, w in mass_w.items()},
           "violations": violations,
           "contact_episodes": [{**ep, "pair": list(ep["pair"])} for ep in episodes],
           "severity": severity}
    EVIDENCE.put(vid, "s3_momentum", {
        "severity": severity, "fingerprints": fingerprints,
        "violations": violations, "signals": signals,
    })

    if severity > 30:
        yield {"type": "log", "level": "warn",
               "text": f"Momentum VIOLATION — severity {severity}%."}
    else:
        yield {"type": "log", "level": "success",
               "text": "Motion signatures look physically plausible."}
    yield {"type": "done"}

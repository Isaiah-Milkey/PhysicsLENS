"""
Stage 3 · Specialist — Friction Specialist
--------------------------------------------
Real sliding contact bleeds energy: an unpowered object coasting across a
surface should measurably decelerate (kinetic friction), and any speed
change should trace to a visible cause (push, collision, gravity on a
falling body). This specialist rebuilds each tracked subject's speed curve
from trajectory + optical-flow evidence and hunts friction-shaped defects
in it — motion that never slows down, motion that speeds up on its own, or
motion that stops instantly with nothing to explain it.

Per subject (tracks/masks reused from the Object Tracker's evidence; shared
cached LK tracks as fallback — no re-tracking):

1. FINGERPRINT — image-space kinematics on the subject's own frame grid
   (speed, accel, position), split into COAST SEGMENTS: contiguous runs
   where the subject is moving, not touching another subject, and not at
   the frame border. Each segment gets a linear speed-vs-time fit
   (slope = deceleration rate, r² = how cleanly it decays).
2. EXPECTED-MOTION PROFILE — ONE VLM call per subject shows a handful of
   labeled frames spanning that subject's own track and asks: what IS this
   object, and what should its speed do here, given its build and how it
   appears to be moving (self-powered, pushed/thrown then released, falling,
   rolling downhill, etc.)? The VLM returns one or more time-phased
   expectations ("accelerate" | "decelerate" | "constant") covering the
   whole clip — e.g. a flicked book: brief "accelerate" while the hand
   pushes it, then "decelerate" once released as friction takes over. Each
   coast segment inherits the expectation for its time slot; a simple
   ground/airborne heuristic is the fallback when no VLM credentials are
   configured.
3. FLOW PROXY — for any segment flagged as a violation candidate, a cheap
   Farneback optical-flow check across the segment's first/mid/last frames
   corroborates (or discounts) the track-derived slope, independent of
   box-center noise.
4. VIOLATIONS —
     * no_friction: a ground-level coast segment holds speed flat for a
       sustained stretch (no measurable deceleration) — perpetual sliding;
     * self_acceleration: speed trends upward mid-coast with no contact,
       no border exit, and no gravity to explain it;
     * abrupt_stop: speed collapses within one or two frames with no
       contact or border cause — a teleport-stop.
   Each flagged event gets ONE VLM check (labeled before/after frames) to
   confirm or reject — same detect→explain pattern as the other specialists.
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

ANOMALY_PROMPT = (
    "Two moments of one video: LEFT is BEFORE, RIGHT is AFTER (timestamps "
    "printed on the image; the object in question is outlined).\n"
    "An automated motion metric flagged this: {desc}\n"
    "Verify it. A real physical cause (visible push or throw, collision, "
    "hitting a surface/wall, gravity on a falling object, exiting the "
    "frame, a slope or surface change) makes it 'plausible'. Speed that "
    "stays constant forever with no friction, speeds up with no visible "
    "cause, or halts instantly with nothing to stop it is a 'violation' "
    "(AI-generation artifact).\n"
    "Reply with ONLY strict JSON: "
    '{{"verdict": "plausible"|"violation", "confidence": <0..1>, '
    '"explanation": "<one sentence naming the cause or its absence>"}}'
)

PROFILE_PROMPT = (
    "The image shows {n} labeled frames sampled across ONE tracked object's "
    "full motion in a video (timestamp and % through its clip printed above "
    "each frame; the object is outlined).\n"
    "1. Identify the object (short name, e.g. \"book\", \"soccer ball\", "
    "\"toy car\").\n"
    "2. Reason about what its speed SHOULD do over this clip, given "
    "real-world physics for an object of this kind — its likely material "
    "and weight, whether it looks self-powered (car, animal, person) versus "
    "passive, whether something appears to push/throw/kick/release it, and "
    "the surface or setting it's moving through.\n"
    "3. Break the clip into one or more time phases as fractions of the "
    "FULL clip duration (0.0 to 1.0, phases must cover the whole clip with "
    "no gaps or overlaps) and label each phase's EXPECTED speed trend:\n"
    "   - \"accelerate\": actively pushed/pulled/powered/kicked here, or "
    "falling/rolling downhill\n"
    "   - \"decelerate\": coasting unpowered against friction/drag/air "
    "resistance after a push, throw, or release\n"
    "   - \"constant\": self-powered at a steady speed, or negligible "
    "friction expected (e.g. gliding briefly right after release)\n"
    "A single object can legitimately have several phases — e.g. a book "
    "flicked across a table: \"accelerate\" for the brief moment the hand "
    "pushes it, then \"decelerate\" for the rest of the clip as friction "
    "takes over once released. A rolling ball starting on a downhill slope "
    "then reaching flat ground: \"accelerate\" then \"decelerate\".\n"
    "Reply with ONLY strict JSON: {{\"object_name\": \"<short name>\", "
    "\"phases\": [{{\"start_frac\": <0..1>, \"end_frac\": <0..1>, "
    "\"expected\": \"accelerate\"|\"decelerate\"|\"constant\", "
    "\"reason\": \"<short phrase>\"}}, ...]}}"
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


# ── Kinematics ─────────────────────────────────────────────────────────────

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
    return {"fis": fis, "t": t, "x": x, "y": y, "vx": vx, "vy": vy,
            "speed": speed, "accel": accel}


# ── Contacts (box adjacency on the shared grid — cheap, grid-safe) ────────────

def _contact_episodes(tracks: dict, eff_fps: float, grid_step: int,
                      H: int) -> list[dict]:
    """Per pair: contiguous frames where padded boxes intersect."""
    from itertools import combinations
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


# ── Coast segments (the friction fingerprint) ─────────────────────────────────

def _fit_slope(t: np.ndarray, speed: np.ndarray) -> tuple[float, float, float]:
    """Linear speed-vs-time fit. Returns (slope px/s², intercept, r²)."""
    if len(t) < 3 or np.ptp(t) < 1e-6:
        return 0.0, float(speed.mean() if len(speed) else 0.0), 0.0
    b, a = np.polyfit(t, speed, 1)
    pred = a + b * t
    ss_res = float(np.sum((speed - pred) ** 2))
    ss_tot = float(np.sum((speed - speed.mean()) ** 2)) or 1e-9
    r2 = float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))
    return float(b), float(a), r2


def _coast_segments(lb: str, k: dict, episodes: list[dict], grid_step: int,
                    H: int, W: int, v_floor: float, min_len: int) -> list[dict]:
    """Contiguous runs of a subject's own grid where it's away from other
    subjects and away from the frame border — the windows where any speed
    change should be attributable to friction/gravity alone."""
    fis, t, speed, x, y, vy = k["fis"], k["t"], k["speed"], k["x"], k["y"], k["vy"]
    slack = 3 * grid_step
    valid = []
    for i, fi in enumerate(fis):
        at_border = (x[i] < 0.03 * W or x[i] > 0.97 * W or
                     y[i] < 0.03 * H or y[i] > 0.97 * H)
        valid.append((not at_border) and (not _near_contact(fi, lb, episodes, slack)))

    segs = []
    start = None
    for i in range(len(fis)):
        gap = i > 0 and fis[i] - fis[i - 1] > 2 * grid_step
        if valid[i] and not gap:
            start = i if start is None else start
        else:
            if start is not None and i - start >= min_len:
                segs.append((start, i - 1))
            start = i if valid[i] else None
    if start is not None and len(fis) - start >= min_len:
        segs.append((start, len(fis) - 1))

    out = []
    for i0, i1 in segs:
        t_run, s_run, y_run, vy_run = t[i0:i1 + 1], speed[i0:i1 + 1], y[i0:i1 + 1], vy[i0:i1 + 1]
        avg_speed = float(s_run.mean())
        if avg_speed < v_floor:
            continue
        slope, intercept, r2 = _fit_slope(t_run, s_run)
        out.append({
            "i0": i0, "i1": i1, "f0": int(fis[i0]), "f1": int(fis[i1]),
            "t0": float(t_run[0]), "t1": float(t_run[-1]),
            "avg_speed": avg_speed, "slope": slope, "r2": r2,
            "slope_norm": slope / max(avg_speed, v_floor),
            # Ground heuristic: lower-half-of-frame segments are treated as
            # surface contact (friction applies); upper-half segments are
            # treated as projectile/airborne motion (friction is exempt).
            "ground": bool(y_run.mean() > 0.5 * H),
            "falling": bool(vy_run.mean() > v_floor and vy_run[-1] >= vy_run[0]),
        })
    return out


# ── Optical-flow corroboration (cheap, only run on violation candidates) ──────

def _flow_speed(g0: np.ndarray, g1: np.ndarray, dt: float) -> float:
    if g0.shape != g1.shape or g0.size == 0 or dt <= 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float(np.median(mag)) / dt


def _flow_slope_check(frames: list, box_by: dict, lb: str, seg: dict,
                      eff_fps: float) -> Optional[bool]:
    """Cross-checks a segment's track-derived slope sign against a Farneback
    optical-flow estimate over the same window. Returns True/False for
    agreement, or None if the check couldn't be computed (inconclusive)."""
    fis = sorted(box_by[lb].keys())
    fis = [f for f in fis if seg["f0"] <= f <= seg["f1"]]
    if len(fis) < 3:
        return None
    f0, fm, f1 = fis[0], fis[len(fis) // 2], fis[-1]
    boxes = [box_by[lb][f] for f in (f0, fm, f1)]
    x0 = int(min(b[0] for b in boxes)); y0 = int(min(b[1] for b in boxes))
    x1 = int(max(b[2] for b in boxes)); y1 = int(max(b[3] for b in boxes))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    try:
        g0 = cv2.cvtColor(frames[f0][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        gm = cv2.cvtColor(frames[fm][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[f1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        s1 = _flow_speed(g0, gm, (fm - f0) / eff_fps)
        s2 = _flow_speed(gm, g1, (f1 - fm) / eff_fps)
    except Exception:                                                # noqa: BLE001
        return None
    flow_slope = s2 - s1
    return (flow_slope <= 0) == (seg["slope"] <= 0)


# ── Expected-motion profile (VLM identifies the object + phase expectations) ──

_EXPECTED_LABELS = ("accelerate", "decelerate", "constant")


def _profile_tiles(frames: list, tr: dict, k: dict, n_tiles: int = 4):
    """Evenly spaced, labeled crops spanning one subject's own track, plus
    the matching time-fractions (0..1) through that subject's clip."""
    fis, t = k["fis"], k["t"]
    n = len(fis)
    if n < 2:
        return None, []
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(n_tiles, n))))
    t0, t1 = float(t[0]), float(t[-1])
    dur = max(t1 - t0, 1e-6)
    tiles, fracs = [], []
    for idx in idxs:
        fi = fis[idx]
        crop = _event_crop(frames[fi], tr["boxes"][idx], pad=0.35)
        if crop is None:
            continue
        frac = (float(t[idx]) - t0) / dur
        fracs.append(frac)
        tiles.append(_labeled_tile(crop, f"t={t[idx]:.2f}s ({frac:.0%})"))
    return _tile_row(tiles), fracs


def _parse_profile(parsed: dict) -> Optional[dict]:
    """Validates/normalizes the VLM's phase list. Returns None (→ heuristic
    fallback) if the response doesn't yield a usable, clip-covering profile."""
    name = str(parsed.get("object_name", "")).strip()[:60] or "object"
    raw = parsed.get("phases")
    if not isinstance(raw, list) or not raw:
        return None
    phases = []
    for ph in raw:
        try:
            s0 = float(np.clip(float(ph["start_frac"]), 0.0, 1.0))
            s1 = float(np.clip(float(ph["end_frac"]), 0.0, 1.0))
            exp = str(ph.get("expected", "")).strip().lower()
            if exp not in _EXPECTED_LABELS or s1 <= s0:
                continue
            phases.append({"start_frac": s0, "end_frac": s1, "expected": exp,
                           "reason": str(ph.get("reason", ""))[:140]})
        except (KeyError, TypeError, ValueError):
            continue
    if not phases:
        return None
    phases.sort(key=lambda p: p["start_frac"])
    phases[0]["start_frac"] = 0.0
    phases[-1]["end_frac"] = 1.0
    return {"object_name": name, "phases": phases}


def _expected_at(profile: Optional[dict], frac: float) -> tuple[Optional[str], str]:
    """Looks up the VLM-predicted expected label + reason for a time
    fraction. Returns (None, "") if there's no profile (caller falls back
    to the ground/airborne heuristic)."""
    if not profile:
        return None, ""
    for ph in profile["phases"]:
        if ph["start_frac"] <= frac <= ph["end_frac"]:
            return ph["expected"], ph["reason"]
    return None, ""


def _heuristic_expected(seg: dict) -> str:
    """Fallback when no VLM profile is available: ground segments are
    expected to decelerate (friction), falling segments are expected to
    accelerate (gravity), anything else is treated as a wash."""
    if seg["falling"]:
        return "accelerate"
    if seg["ground"]:
        return "decelerate"
    return "constant"


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(video_path: str, settings: str = None) -> AsyncGenerator[dict, None]:
    cfg            = json.loads(settings) if settings else {}
    model          = str(cfg.get("model") or "createai:geminiflash2_5")
    api_key        = str(cfg.get("api_key", "")).strip()
    friction_tol   = max(0.02, float(cfg.get("friction_tolerance", 0.06)))   # |slope|/speed, flat coast
    accel_tol      = max(0.02, float(cfg.get("acceleration_tolerance", 0.20)))  # slope/speed, speedup
    stop_tol       = max(0.10, float(cfg.get("stop_tolerance", 0.55)))       # fractional 1-step speed drop
    max_checks     = max(1, int(cfg.get("max_checks", 3)))
    use_flow_check = str(cfg.get("use_flow_check", "true")).lower() not in ("false", "0", "no")
    use_profile    = str(cfg.get("use_profile_vlm", "true")).lower() not in ("false", "0", "no")
    profile_tiles  = max(2, int(cfg.get("profile_tiles", 4)))

    loop = asyncio.get_event_loop()
    vid = video_id(video_path)

    # ── 1. Tracks + frames: tracker evidence first, cached LK/flow fallback ──
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
               "text": "No Object Tracker evidence — using shared cached LK/optical-flow tracks."}
        data = await loop.run_in_executor(None, get_tracks, video_path)
        frames, eff_fps = data["frames"], data["fps"] or 30.0
        for tr in data["tracks"]:
            if len(tr["frames"]) >= 4:
                tracks[tr.get("label") or f"obj {tr['id']}"] = tr
        src = "cached LK/optical-flow tracks (fallback)"

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

    # ── 2. Kinematics ──────────────────────────────────────────────────────
    kin = {lb: k for lb, tr in tracks.items()
           if (k := _kinematics(tr, eff_fps)) is not None}
    if not kin:
        yield {"type": "error", "text": "No track has enough samples for kinematics."}
        return

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

    v_floor = 0.03 * H     # px/s — ignore near-static motion
    min_len = max(6, 2 * max(1, len(all_fis) // max(1, all_fis[-1] // grid_step + 1)))
    min_len = max(6, min_len)

    box_by = {lb: dict(zip(tr["frames"], tr["boxes"])) for lb, tr in tracks.items()}
    episodes = _contact_episodes(tracks, eff_fps, grid_step, H)
    if episodes:
        yield {"type": "log", "level": "info",
               "text": f"{len(episodes)} contact episode(s) between subjects."}

    from tools.vlm_router import key_status
    have_api, key_desc = key_status(model, api_key)

    # ── 3. Expected-motion profiles — identify each object, predict its
    #        phase-by-phase speed trend (VLM), fall back to a ground/
    #        airborne heuristic when no credentials are configured. ────────
    profiles: dict[str, Optional[dict]] = {}
    if use_profile and have_api:
        from tools.vlm_router import ask_vision_json
        for lb, tr in tracks.items():
            if lb not in kin:
                continue
            tiles_img, _ = _profile_tiles(frames, tr, kin[lb], profile_tiles)
            if tiles_img is None:
                profiles[lb] = None
                continue
            try:
                parsed = await ask_vision_json(
                    PROFILE_PROMPT.format(n=profile_tiles), tiles_img, model, api_key)
                profile = _parse_profile(parsed)
                profiles[lb] = profile
                if profile:
                    summary = " → ".join(dict.fromkeys(
                        p["expected"] for p in profile["phases"]))
                    yield {"type": "log", "level": "info",
                           "text": f"“{lb}” identified as {profile['object_name']!r} — "
                                   f"expected motion: {summary}"}
                else:
                    yield {"type": "log", "level": "warn",
                           "text": f"“{lb}”: VLM profile unusable — "
                                   "falling back to ground/airborne heuristic."}
            except Exception as exc:                                # noqa: BLE001
                profiles[lb] = None
                yield {"type": "log", "level": "warn",
                       "text": f"“{lb}” profile check failed ({str(exc)[:140]}) — "
                               "falling back to ground/airborne heuristic."}
            await asyncio.sleep(0)
    else:
        for lb in kin:
            profiles[lb] = None
        if use_profile and not have_api:
            yield {"type": "log", "level": "warn",
                   "text": f"No VLM credentials ({key_desc}) — object identification "
                           "skipped, using ground/airborne heuristic for all subjects."}

    # ── 4. Coast segments + friction fingerprint ──────────────────────────
    fingerprints: dict[str, dict] = {}
    segs_by_label: dict[str, list[dict]] = {}
    for lb, k in kin.items():
        segs = _coast_segments(lb, k, episodes, grid_step, H, W, v_floor, min_len)
        profile = profiles.get(lb)
        t0, dur = float(k["t"][0]), max(float(k["t"][-1] - k["t"][0]), 1e-6)
        for s in segs:
            frac = ((s["t0"] + s["t1"]) / 2 - t0) / dur
            expected, reason = _expected_at(profile, frac)
            if expected is None:
                expected, reason = _heuristic_expected(s), ""
            s["expected"] = expected
            s["expected_reason"] = reason
        segs_by_label[lb] = segs
        obj_name = profile["object_name"] if profile else lb
        decel_expected = [s["r2"] for s in segs if s["expected"] == "decelerate" and s["slope"] <= 0]
        friction_score = int(round(100 * float(np.mean(decel_expected)))) if decel_expected else 50
        fingerprints[lb] = {
            "object_name": obj_name,
            "avg_speed_px_s": round(float(k["speed"].mean()), 1),
            "max_speed_px_s": round(float(k["speed"].max()), 1),
            "n_coast_segments": len(segs),
            "n_decel_expected_segments": sum(1 for s in segs if s["expected"] == "decelerate"),
            "avg_decel_px_s2": round(float(np.mean([-s["slope"] for s in segs])), 2) if segs else 0.0,
            "friction_score": friction_score,
        }
        yield {"type": "log", "level": "info",
               "text": f"“{lb}” ({obj_name}): {len(segs)} coast segment(s), "
                       f"friction score {friction_score} — "
                       f"vavg {fingerprints[lb]['avg_speed_px_s']:.0f} px/s, "
                       f"avg decel {fingerprints[lb]['avg_decel_px_s2']:.1f} px/s²"}
    await asyncio.sleep(0)

    # ── 5. Violations ─────────────────────────────────────────────────────
    violations: list[dict] = []
    slack = 3 * grid_step

    # 5a. no_friction: a segment expected to decelerate holds speed flat instead.
    for lb, segs in segs_by_label.items():
        obj_name = fingerprints[lb]["object_name"]
        for s in segs:
            if s["expected"] != "decelerate":
                continue
            if abs(s["slope_norm"]) >= friction_tol:
                continue
            if s["avg_speed"] < 3 * v_floor:
                continue
            dur = s["t1"] - s["t0"]
            score = float(np.clip(1 - abs(s["slope_norm"]) / friction_tol, 0, 1)) * \
                    float(np.clip(dur / 1.0, 0, 1)) * \
                    float(np.clip(s["avg_speed"] / (3 * v_floor), 0, 1))
            if score < 0.15:
                continue
            why = f" ({s['expected_reason']})" if s["expected_reason"] else ""
            violations.append({
                "type": "no_friction", "label": lb, "object_name": obj_name,
                "frame": s["f1"], "t": round(s["t1"], 3),
                "f_before": s["f0"], "confirmed": None,
                "slope_norm": round(s["slope_norm"], 4),
                "score": round(score * 0.8, 3),
                "desc": (f"“{lb}” (identified as {obj_name}) coasted at "
                         f"~{s['avg_speed']:.0f} px/s for {dur:.2f}s "
                         f"({s['t0']:.2f}–{s['t1']:.2f}s) with no measurable "
                         f"deceleration, though it was expected to slow down"
                         f"{why}"),
            })

    # 5b. self_acceleration: speed trends up in a segment not expected to accelerate.
    for lb, segs in segs_by_label.items():
        obj_name = fingerprints[lb]["object_name"]
        for s in segs:
            if s["slope_norm"] <= accel_tol:
                continue
            if s["expected"] == "accelerate":     # push/throw/gravity slot — expected
                continue
            dur = s["t1"] - s["t0"]
            score = float(np.clip(s["slope_norm"] / (2 * accel_tol), 0, 1)) * \
                    float(np.clip(dur / 0.6, 0, 1))
            if score < 0.15:
                continue
            why = f" ({s['expected_reason']})" if s["expected_reason"] else ""
            violations.append({
                "type": "self_acceleration", "label": lb, "object_name": obj_name,
                "frame": s["f1"], "t": round(s["t1"], 3),
                "f_before": s["f0"], "confirmed": None,
                "slope_norm": round(s["slope_norm"], 4),
                "score": round(score * 0.85, 3),
                "desc": (f"“{lb}” (identified as {obj_name}) sped up from "
                         f"~{s['avg_speed']:.0f} px/s over {dur:.2f}s "
                         f"({s['t0']:.2f}–{s['t1']:.2f}s) though it was expected to "
                         f"{s['expected']}{why}, with no contact or push visible"),
            })

    # 5c. abrupt_stop: speed collapses within one step, no contact/border cause.
    for lb, k in kin.items():
        obj_name = fingerprints[lb]["object_name"]
        speed, fis, x, y = k["speed"], k["fis"], k["x"], k["y"]
        for i in range(2, len(speed) - 1):
            prev = max(speed[i - 1], v_floor)
            drop = (speed[i - 1] - speed[i]) / prev
            if drop <= stop_tol:
                continue
            fi = fis[i]
            if _near_contact(fi, lb, episodes, slack):
                continue
            at_border = (x[i] < 0.03 * W or x[i] > 0.97 * W or
                         y[i] < 0.03 * H or y[i] > 0.97 * H)
            if at_border:
                continue
            score = float(np.clip(drop / (2 * stop_tol), 0, 1))
            violations.append({
                "type": "abrupt_stop", "label": lb, "object_name": obj_name,
                "frame": int(fi), "t": round(fi / eff_fps, 3),
                "f_before": int(fis[i - 1]), "confirmed": None,
                "drop_ratio": round(float(drop), 3),
                "score": round(score * 0.85, 3),
                "desc": (f"“{lb}” (identified as {obj_name}) lost {drop:.0%} of "
                         f"its speed in one step with no contact or visible "
                         f"obstacle to stop it"),
            })

    # Dedupe: keep the strongest flag per (subject, type) within a short window.
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

    # Optical-flow corroboration on candidates before spending VLM calls.
    if use_flow_check:
        for v in violations:
            if v["type"] not in ("no_friction", "self_acceleration"):
                continue
            seg = next((s for s in segs_by_label[v["label"]]
                        if s["f0"] == v["f_before"] and s["f1"] == v["frame"]), None)
            if seg is None:
                continue
            agree = _flow_slope_check(frames, box_by, v["label"], seg, eff_fps)
            if agree is False:
                v["score"] = round(v["score"] * 0.6, 3)
                v["flow_check"] = "disagrees"
            elif agree is True:
                v["flow_check"] = "agrees"
        violations = sorted(violations, key=lambda v: -v["score"])

    for v in violations:
        yield {"type": "log", "level": "warn",
               "text": f"FLAGGED {v['type']}: {v['desc']}."}

    # 5d. VLM verifies the worst flagged events.
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

    # ── 6. Aggregate: signals, severity, plots, metrics ──────────────────────
    signals = [{"frame": v["frame"], "signal_type": v["type"], "score": v["score"]}
               for v in violations]
    severity = int(round(100 * max([v["score"] for v in violations], default=0.0)))

    yield {"type": "signal", "source": "s3_friction",
           "source_name": "Friction",
           "fps": float(eff_fps), "n_frames": int(n), "severity": severity,
           "type_severities": {
               t: int(round(100 * max([v["score"] for v in violations
                                       if v["type"] == t], default=0.0)))
               for t in ("no_friction", "self_acceleration", "abrupt_stop")},
           "signals": signals}

    # Trajectory overlay, coasting segments highlighted by decel direction.
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

    # Speed curves with coasting segments shaded by slope sign.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("Speed (px/s) — coast segments shaded",
                                        "Deceleration rate per coast segment (px/s²)"))
    hexp = ["#c05621", "#1a54c4", "#1a7a3c", "#be185d", "#7c3aed", "#0891b2"]
    for i, (lb, k) in enumerate(kin.items()):
        color = hexp[i % len(hexp)]
        fig.add_trace(go.Scatter(x=k["t"].tolist(), y=k["speed"].tolist(),
                                 mode="lines", name=lb, legendgroup=lb,
                                 line=dict(color=color, width=1.6)), row=1, col=1)
        segs = segs_by_label[lb]
        if segs:
            fig.add_trace(go.Bar(
                x=[(s["t0"] + s["t1"]) / 2 for s in segs],
                y=[-s["slope"] for s in segs],
                width=[max(s["t1"] - s["t0"], 0.05) for s in segs],
                name=lb, legendgroup=lb, showlegend=False,
                marker=dict(color=color, opacity=0.75)), row=2, col=1)
    for ep in episodes:
        fig.add_vrect(x0=ep["t_start"], x1=ep["t_end"], fillcolor="#EF9F27",
                      opacity=0.12, line_width=0, row=1, col=1)
    for v in violations:
        fig.add_vline(x=v["t"], line=dict(color="#E24B4A", dash="dot", width=1.2),
                      row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#999", width=1), row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1,
                     showgrid=True, gridcolor="#ebebeb")
    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb", zeroline=False)
    fig.update_layout(
        title=dict(text="Friction — Speed Decay & Coast-Segment Analysis",
                   font=dict(size=15)),
        height=560, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.09, x=0, font=dict(size=12)),
        margin=dict(l=65, r=40, t=100, b=50),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
    )
    yield {"type": "plotly", "data": fig.to_json(),
           "caption": "Orange bands = contact episodes (speed may change); "
                      "dotted red = flagged friction anomalies; bottom bars = "
                      "deceleration rate per coast segment (positive = slowing down, "
                      "as friction predicts)."}

    for lb, fp in sorted(fingerprints.items(), key=lambda kv: -kv[1]["friction_score"]):
        yield {"type": "metric", "label": f"“{lb}” friction score",
               "value": str(fp["friction_score"]),
               "sub": (f"{fp['object_name']} · {fp['n_decel_expected_segments']} "
                       f"decel-expected segment(s) · "
                       f"avg decel {fp['avg_decel_px_s2']:.1f} px/s²")}
    n_conf = sum(1 for v in violations if v["confirmed"])
    yield {"type": "metric", "label": "Friction anomalies",
           "value": str(len(violations)),
           "sub": f"{n_conf} VLM-confirmed · {len(episodes)} contact episode(s)"}

    yield {"type": "severity", "label": "Friction violation",
           "value": severity, "color": _sev_color(severity)}

    yield {"type": "result", "status": "ok",
           "fingerprints": fingerprints,
           "profiles": {lb: p for lb, p in profiles.items() if p is not None},
           "coast_segments": {lb: [{k2: v2 for k2, v2 in s.items()} for s in segs]
                              for lb, segs in segs_by_label.items()},
           "violations": violations,
           "contact_episodes": [{**ep, "pair": list(ep["pair"])} for ep in episodes],
           "severity": severity}
    EVIDENCE.put(vid, "s3_friction", {
        "severity": severity, "fingerprints": fingerprints,
        "violations": violations, "signals": signals,
    })

    if severity > 30:
        yield {"type": "log", "level": "warn",
               "text": f"Friction VIOLATION — severity {severity}%."}
    else:
        yield {"type": "log", "level": "success",
               "text": "Speed decay looks physically plausible."}
    yield {"type": "done"}